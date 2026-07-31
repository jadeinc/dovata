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
//! Loading is where policy runs: [`engine::Dovata::from_config`] preflights the model
//! file (see [`preflight`]), picks the model [`profile`], and fails the load rather than
//! coming up half-configured. A failed load surfaces to the loader as a null handle.
//!
//! The pure, platform-independent core (config, prompt parsing, the response cache, the
//! constrained-decoding control machine, the scheduler, MoE glue, the preflight, and the
//! model [`profile`]) has no llama.cpp dependency, so it builds and unit-tests on any
//! platform. The [`engine`] module is `#[cfg(target_os = "linux")]` and links the
//! vendored `infer-engine` (llama.cpp); on macOS the cdylib builds as an empty stub.

// Each module below is pure (no llama.cpp) and individually unit-tested:
pub mod cache; // bounded LRU cache of past responses (skip the model on repeats)
pub mod config; // env-var parsing + validation into a Config struct
pub mod control; // the constrained-decoding state machine (anchors/grammar/stops)
pub mod preflight; // reads the GGUF header at boot: profile selection + fail-fast checks
pub mod prompt; // turns a request prompt string into chat messages for the model
pub mod profile; // per-model token/turn profiles + chat templates (was jade-model-profile)
pub mod route; // over-context corpus/question split + text chunking for the RLM path
pub mod moe; // MoE expert-offload glue: expert_order asset load + validation (pure half)
pub mod sched; // pure multi-sequence continuous-batching scheduler (RLM concurrency core)

// The local llama.cpp inference engine — the in-process provider. Linux-only: it links
// the vendored infer-engine (llama.cpp). Compiled out on other platforms.
#[cfg(target_os = "linux")]
pub mod engine;

// Export the engine across the C ABI, so this crate's cdylib *is* a provider package.
//
// This is the whole product. A Jade program dlopens the library and calls `infer`
// in-process: `jadelang -> dovata.so -> nvidia.ko`, with no socket and no second
// process in front of work that is almost entirely GPU-bound.
//
// Nothing about the engine had to change to get here. It implements `Provider`, which
// is all `export_provider!` needs, so removing the host was a deletion rather than a
// rewrite.
//
// Linux-gated because `engine` is: on macOS there is no llama.cpp and so nothing to
// export, and the cdylib builds as an empty stub rather than failing.
#[cfg(target_os = "linux")]
ovata_infer_protocol::export_provider!(engine::Dovata);

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
