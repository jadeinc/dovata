//! dovata — JADE OS inference daemon.
//!
//! This library holds the **pure, platform-independent core** of the daemon:
//! configuration, prompt parsing, the response cache, and the constrained-
//! decoding control state machine. None of it depends on llama.cpp, so it
//! builds and unit-tests on any platform (notably macOS dev machines).
//!
//! The llama.cpp + io_uring backend lives in the binary (`src/inference.rs`)
//! behind the `inference` feature and only builds on Linux.

// Each module below is pure (no llama.cpp) and individually unit-tested:
pub mod cache; // bounded LRU cache of past responses (skip the model on repeats)
pub mod config; // env-var parsing + validation into a Config struct
pub mod control; // the constrained-decoding state machine (anchors/grammar/stops)
pub mod prompt; // turns a request prompt string into chat messages for the model
pub mod providers; // host-side loader that dlopens provider .so packages (C-ABI)
pub mod route; // over-context corpus/question split + text chunking for the RLM path
pub mod moe; // MoE expert-offload glue: expert_order asset load + validation (pure half)
pub mod sched; // pure multi-sequence continuous-batching scheduler (RLM concurrency core)
