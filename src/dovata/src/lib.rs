//! dovata — JADE OS inference daemon.
//!
//! `dovata` is the local inference daemon: a host that owns the Unix socket plus the
//! standard, surface-level optimization of the llama.cpp engine for Linux. The daemon
//! binary (`src/main.rs` + `src/host.rs`) frames requests off the wire and routes each
//! to a provider. The **local** provider — the llama.cpp engine ([`engine`]) — is
//! compiled in and served in-process; **remote** providers (`anthropic`, `openai`) are
//! separate `.so` packages the host `dlopen`s via [`providers`].
//!
//! The pure, platform-independent core (config, prompt parsing, the response cache, the
//! constrained-decoding control machine, the scheduler, MoE glue, and the model
//! [`profile`]) has no llama.cpp dependency, so it — and the whole host — builds and
//! unit-tests on any platform. The [`engine`] module is `#[cfg(target_os = "linux")]`
//! and links the vendored `infer-engine` (llama.cpp); on macOS the daemon builds as a
//! host without the local engine (remote providers still load).

// Each module below is pure (no llama.cpp) and individually unit-tested:
pub mod cache; // bounded LRU cache of past responses (skip the model on repeats)
pub mod config; // env-var parsing + validation into a Config struct
pub mod control; // the constrained-decoding state machine (anchors/grammar/stops)
pub mod preflight; // reads the GGUF header at boot: profile selection + fail-fast checks
pub mod prompt; // turns a request prompt string into chat messages for the model
pub mod profile; // per-model token/turn profiles + chat templates (was jade-model-profile)
pub mod providers; // host-side loader that dlopens remote provider .so packages (C-ABI)
pub mod route; // over-context corpus/question split + text chunking for the RLM path
pub mod moe; // MoE expert-offload glue: expert_order asset load + validation (pure half)
pub mod sched; // pure multi-sequence continuous-batching scheduler (RLM concurrency core)

// The local llama.cpp inference engine — the in-process provider. Linux-only: it links
// the vendored infer-engine (llama.cpp). Compiled out on other platforms.
#[cfg(target_os = "linux")]
pub mod engine;
