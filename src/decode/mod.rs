//! The default request path: one sequence, streamed.
//!
//! Renders the prompt through the active model profile, checks the response cache,
//! reuses whatever KV prefix is already resident, and generates under the constrained
//! decoder, emitting each chunk of client-safe text as it is produced.
//!
//! The pure halves — prompt rendering and the response cache — build everywhere.
//! `exec.rs` is the llama-bound generation loop and is Linux-only.

pub mod cache;
pub mod prompt;

/// The llama-bound generation loop.
#[cfg(target_os = "linux")]
pub mod exec;
