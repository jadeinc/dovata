//! jaded — JADE OS inference daemon.
//!
//! This library holds the **pure, platform-independent core** of the daemon:
//! configuration, prompt parsing, the response cache, and the constrained-
//! decoding control state machine. None of it depends on llama.cpp, so it
//! builds and unit-tests on any platform (notably macOS dev machines).
//!
//! The llama.cpp + io_uring backend lives in the binary (`src/inference.rs`)
//! behind the `inference` feature and only builds on Linux.

pub mod cache;
pub mod config;
pub mod control;
pub mod prompt;
