//! Long-context answering by KV composition.
//!
//! When a prompt does not fit the native window, this answers it by splitting the input
//! into a retrievable corpus and a question, retrieving the blocks that matter, and
//! splicing their precomputed K/V into one attention window — roughly 50x cheaper per
//! block on an A10G than re-prefilling. See `docs/blended-rag-rlm.md`.
//!
//! A peer of [`crate::decode`] rather than a mode of it: it is a different engine, not
//! a flag on the same one.
//!
//! `route.rs` is the pure corpus/question split and builds everywhere. `exec.rs` (the
//! composition engine) and `rlm.rs` (its request dispatch and the uploaded-context
//! store) are llama-bound and Linux-only.

pub mod route;

/// The KV-composition engine: block store, retrieval, save/splice/answer.
#[cfg(target_os = "linux")]
pub mod exec;

/// RLM request dispatch and the persistent uploaded-context handles.
#[cfg(target_os = "linux")]
pub mod rlm;
