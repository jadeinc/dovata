//! Reading a model file: refusing one that cannot be planned, and reading its shape.
//!
//! [`preflight`] reads the GGUF header off disk, checks the keys llama.cpp requires, and
//! checks that the tensor index is complete. [`model`] gathers every part of a split
//! GGUF into one [`Model`] with one tensor list. [`dims`] reads the model's shape out of
//! the header and prices one token of KV cache.
//!
//! It is its own capability, rather than part of [`crate::gguf`], because it is policy:
//! the GGUF reader reports what a file says, even a truncated one, and this module
//! decides whether that is enough to plan from. All of it is pure but for one file read,
//! so it is unit-tested on any machine against synthetic GGUF headers.

pub mod dims;
pub mod model;
pub mod preflight;

pub use dims::{KvType, ModelDims};
pub use model::{Model, TensorRef};
pub use preflight::Preflight;
