//! Continuous batching: N sequences contributing tokens to one forward pass.
//!
//! `sched.rs` is the pure scheduler — which sequence feeds which token at which
//! position, and when KV may be admitted or retired. It is bookkeeping, not model work,
//! so it is thoroughly unit-tested on any machine. `exec.rs` is the thin llama-bound
//! driver that executes its plans.
//!
//! Nothing calls `exec.rs`. Its only caller was the env-gated selftest module, which
//! went away with the daemon startup phase that dispatched it. The driver is kept
//! because the RLM fan-out design depends on it, but treat it as unverified code until
//! something calls it again. See this directory's README.

pub mod sched;

/// The llama-bound continuous-batching driver. No caller today.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub mod exec;
