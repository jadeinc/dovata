//! Admission: everything that happens once, before any request.
//!
//! Reads the `JADE_*` environment into a [`Config`], vets the model file against what
//! llama.cpp will require of it, picks the model profile, and sizes the generation cap
//! from the VRAM left after the weights.
//!
//! This is the only place in the engine that can decline to come up at all, which is
//! why it is its own capability rather than part of [`crate::dispatch`]. Load-time
//! policy and per-request policy fail differently: a bad model path, an unusable GGUF,
//! or a model no profile matches costs a second here and a null handle at the caller,
//! instead of a provider that loads happily and mis-parses every tool call.
//!
//! All of it is pure but for one file read, so it is unit-tested on any machine against
//! synthetic GGUF headers.

pub mod config;
pub mod preflight;
pub mod vram;

pub use config::{Config, KvType};
