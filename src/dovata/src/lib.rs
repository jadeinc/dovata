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
