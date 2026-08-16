//! KV-cache management.
//!
//! Cache-type mapping, cross-request prefix reuse, sequence copies, and the
//! per-sequence save/restore the composition engine is built on.
//!
//! A capability in its own right rather than a part of [`crate::decode`], because all
//! three request paths use it and use it differently: `decode` for prefix reuse across
//! requests, `compose` for saving and restoring a block's precomputed K/V, `batch` for
//! the per-sequence operations. Owning it from any one of them would be a lie about who
//! depends on what.
//!
//! Entirely llama-bound — a KV cache is a llama.cpp object, and there is no useful pure
//! half to test. That makes this one of the two capabilities a macOS run says nothing
//! about.

/// The llama-bound cache operations.
#[cfg(target_os = "linux")]
pub mod exec;
