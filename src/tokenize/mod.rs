//! Text to tokens and back.
//!
//! Thin wrappers over llama's tokenizer, plus the incremental UTF-8 decoder that lets a
//! multi-byte character survive being split across two generated tokens — decode each
//! token's bytes independently and a three-byte character emitted in two pieces becomes
//! two replacement characters in the user's output.
//!
//! Linux-only in full: there is no tokenizer without a model. The capability exists as
//! its own directory because all three request paths tokenize, and each carried its own
//! copy of the decoder setup.

/// The llama-bound tokenizer wrappers and the streaming UTF-8 decoder.
#[cfg(target_os = "linux")]
pub mod exec;
