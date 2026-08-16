//! dovata — the JADE OS inference engine.
//!
//! `dovata` is a provider package: an `impl Provider` from `ovata-infer-protocol`,
//! exported across the C ABI and built as a `cdylib`. A Jade program loads the library
//! and calls `infer` in-process. There is no socket and no second process — the chain
//! is `jadelang -> dovata.so -> nvidia.ko`, which is the ordinary shape for GPU work.
//!
//! It used to be a daemon, with a Unix socket and a host that routed to providers. That
//! put IPC and a process boundary in front of work that is almost entirely GPU-bound,
//! and it made the daemon a second thing to install, supervise, and explain. The engine
//! was always reachable in-process — it implements `Provider`, which is all
//! `export_provider!` needs — so the host was removable without touching the engine.
//! Health checking the GPU is a separate concern and now lives in a separate daemon.
//!
//! ## The shape of a request
//!
//! Nothing reaches backwards. Load happens once, per process; the rest is per request.
//!
//! ```text
//!   dlopen
//!     -> load::Config::from_env          the JADE_* environment
//!     -> load::preflight                 GGUF header; picks a model profile, or
//!                                        refuses the load with a message naming the fix
//!     -> model + context built           llama.cpp allocates; MoE driver activated
//!
//!   per request:
//!     dispatch                           picks one of the three paths
//!       -> decode                        it fits: one sequence, streamed
//!       -> compose                       it does not fit: retrieve, splice KV, answer
//!       -> batch                         many sequences in one forward pass
//!                                        (retained; nothing reaches it today)
//!     each built on tokenize + kv + sample, and emits protocol frames
//! ```
//!
//! ## How this directory is organized
//!
//! Every child of `src/` is one *capability* — a slice of the engine that compiles out
//! whole, its Rust and its llama.cpp seam together, behind one cargo feature. They are
//! flat rather than nested because nesting them by kind made simple questions hard, and
//! because the grouping that helps a reader is a narrative, not a directory: see the
//! directory map in `CLAUDE.md`.
//!
//! Within a capability the rule that used to separate `engine/` from the crate root
//! still holds, one level down: the pure half sits at the capability's root and builds
//! everywhere, and the llama-bound half is `exec.rs` behind `cfg(target_os = "linux")`.
//! That is why `cargo test` runs in under a second on a laptop and still covers the
//! tricky parts — constrained decoding is a state machine over token text, batching is
//! bookkeeping, over-context routing is string splitting. None of it is model-bound.
//!
//! The doc comments below say why each capability sits where it does, because "why here
//! and not inside that one" is the question a flat tree invites.

// ── Admission: what happens once, before any request ──────────────────────────

/// Reading the environment, vetting the model file, and refusing the load when
/// something is wrong. Separate from [`dispatch`] because load-time policy and
/// per-request policy fail differently: this is the only place that can decline to come
/// up at all, and a failed load surfaces to the caller as a null handle, which beats a
/// half-configured provider answering questions.
pub mod load;

/// Per-model vocabulary as data: tool-call delimiters, turn markers, chat templates.
/// A peer of `load` rather than part of it because the profile outlives admission —
/// every request renders its prompt through the active one — and because swapping a
/// `.gguf` must not mean editing the engine.
pub mod model;

/// GGUF header reading: what llama.cpp will see in a model file, answered without
/// llama.cpp. [`load::preflight`] is its caller, and `gguf-inspect` (a bin of this
/// package) is its CLI.
///
/// Was its own crate. It is dependency-free by design — that is the whole point, since
/// it has to run on a laptop against a model no GPU has touched — and a crate boundary
/// between it and its one caller bought a second manifest and nothing else.
pub mod gguf;

// ── Primitives over the context, shared by all three request paths ────────────

/// Text to tokens and back, including the incremental UTF-8 decoder that makes a
/// multi-byte character survive being split across two generated tokens.
pub mod tokenize;

/// KV-cache management: cache-type mapping, cross-request prefix reuse, sequence
/// copies, and the per-sequence save/restore the composition engine is built on.
///
/// A peer of `decode` rather than a part of it. All three request paths use it and use
/// it differently — `decode` for prefix reuse, `compose` for save and restore, `batch`
/// for the sequence operations — so owning it from any one of them would be a lie.
pub mod kv;

/// Choosing the next token: the sampler chain, and the constrained-decoding state
/// machine over the text those tokens spell (anchor spans, stop anchors split across
/// tokens, JSON depth tracking to retire the grammar).
///
/// Also a peer rather than part of `decode`, and for a sharper reason: the chain
/// `temp -> min_p -> dist` was copy-pasted with hardcoded constants into four places
/// across two request paths, so changing the temperature in one silently disagreed with
/// the others. One owner is the fix.
pub mod sample;

// ── The three request paths ───────────────────────────────────────────────────

/// The `Provider` impl, the model and context lifetime, and the routing decision.
/// [`load`] constructs the value; this owns the type and picks a path per request.
#[cfg(target_os = "linux")]
pub mod dispatch;

/// The default path: one sequence, prompt rendering, the response cache, constrained
/// decode, tokens streamed as they are produced.
pub mod decode;

/// Long-context answering by KV composition, for a prompt that does not fit the window.
/// A peer of `decode`, not a mode of it — it retrieves and splices precomputed KV rather
/// than prefilling, which is a different engine, not a flag.
///
/// Without this capability an over-context prompt is refused rather than truncated to
/// fit: a shortened prompt returns a confident answer to a question nobody asked.
#[cfg(feature = "compose")]
pub mod compose;

/// Continuous batching: N sequences contributing tokens to one forward pass.
///
/// Nothing calls its llama-bound half today — the selftests that did were deleted with
/// the daemon — so this compiles and is unverified. The pure scheduler under it is well
/// covered. See `src/batch/README.md` before relying on it.
#[cfg(feature = "batch")]
pub mod batch;

// ── Acceleration, underneath all of it ────────────────────────────────────────

/// MoE expert offload: keep the cold experts in host RAM and page them into VRAM as the
/// router asks for them.
///
/// The only capability that reaches into llama.cpp itself, which is why it owns C++ and
/// CUDA (`driver/`) and a patch series against llama.cpp (`bundle/`) next to its Rust.
/// Putting the three in one directory is the point: they were in three places, and the
/// seam was invisible from this repo entirely.
#[cfg(feature = "moe")]
pub mod moe;

// ── The two ABI entry points ──────────────────────────────────────────────────

/// The entry point the Jade compiler looks for. `export_provider!` below gives this
/// crate the C ABI a non-Jade host loads; the compiler uses its own native-package
/// machinery instead and looks for `jade_pkg_init`. Both live in one library over one
/// `Provider` impl. See the module docs for how the two drifted apart.
pub mod jade_pkg;

// Export the engine across the C ABI, so this crate's cdylib *is* a provider package.
//
// This is the whole product. A Jade program dlopens the library and calls `infer`
// in-process: `jadelang -> dovata.so -> nvidia.ko`, with no socket and no second
// process in front of work that is almost entirely GPU-bound.
//
// Linux-gated because `dispatch` is: on macOS there is no llama.cpp and so nothing to
// export, and the cdylib builds as an empty stub rather than failing.
#[cfg(target_os = "linux")]
ovata_infer_protocol::export_provider!(dispatch::Dovata);

#[cfg(test)]
mod capabilities {
    //! That each capability's pure half is reachable under default features.
    //!
    //! A `#[cfg(feature = "…")]` with a mis-typed feature name compiles fine and
    //! silently drops the module — the failure surfaces later as "unresolved import"
    //! in whatever tries to use it, which on the engine side means a Linux CUDA build
    //! rather than a laptop test run. These name each feature once more, so a typo in
    //! Cargo.toml or in a gate fails here in under a second.
    //!
    //! Each test is itself gated: with the feature off the module is *supposed* to be
    //! absent, so the assertion is only meaningful when the feature is on.

    #[test]
    #[cfg(feature = "compose")]
    fn compose_pure_half_is_present() {
        // route is the over-context corpus/question split.
        let _ = crate::compose::route::DEFAULT_CHUNK_CHARS;
    }

    #[test]
    #[cfg(feature = "batch")]
    fn batch_pure_half_is_present() {
        // sched is the multi-sequence continuous-batching scheduler.
        let _: fn(_, _) -> _ = crate::batch::sched::Scheduler::new;
    }

    #[test]
    #[cfg(feature = "moe")]
    fn moe_pure_half_is_present() {
        // A malformed asset must be rejected, which also proves the loader linked.
        assert!(crate::moe::ExpertOrder::from_json(b"{}").is_err());
    }

    // There is deliberately no test that `default` still lists every capability.
    // `cfg!(feature = …)` reports the build in front of it, not the manifest, so such
    // a test cannot tell "someone dropped a capability from default" from "this is a
    // --no-default-features run" — it would fail on the second and catch nothing on
    // the first. The check that a *shipped* artifact carries every capability belongs
    // where a shipped artifact exists: `packaging` records the enabled list in
    // dist/manifest.json and refuses a release build that is missing one.
}

#[cfg(test)]
mod protocol_compat {
    //! What this crate assumes about the protocol it is built against.
    //!
    //! `src/protocol` is a submodule and the dependency is by path, so the floor is
    //! declared two ways for two different failures. Cargo.toml carries
    //! `version = ">=0.5.0"`, which cargo checks against the submodule's own version —
    //! that catches rolling the pointer *back*. These catch rolling it *forward* into
    //! something that no longer means what the engine thinks it means, which a semver
    //! bump alone would let through silently.

    /// The wire version dovata speaks. The crate's semver and this number move
    /// independently — 0.4.0 and 0.5.1 both speak wire 3 — so the semver floor cannot
    /// stand in for it. Bumping this is a deliberate act: the frames on the wire
    /// changed, and every consumer of a `dovata-<arch>.so` has to be looked at.
    #[test]
    fn speaks_wire_version_3() {
        assert_eq!(
            ovata_infer_protocol::PROTOCOL_VERSION,
            3,
            "the protocol's wire version moved. Read its README for what changed on \
             the wire, update the engine to match, then update this test."
        );
    }

    /// The provider ABI the shipped `.so` advertises. `packaging` stamps this into
    /// `dist/manifest.json` for the engine without dlopening it (it links CUDA and the
    /// release runner has no driver), so the value there is only right while the
    /// packaging xtask and the engine link the same protocol crate.
    #[test]
    fn provider_abi_is_stable() {
        assert_eq!(
            ovata_infer_protocol::PROVIDER_ABI_VERSION,
            1,
            "the provider ABI moved. Every published provider .so was built against \
             the old one and must be rebuilt; update this test once they have been."
        );
    }
}
