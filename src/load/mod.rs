//! Author: Joe Ricks
//! File: src/load/mod.rs
//! Description: Opens a GGUF model for planning and reads its shape.

pub mod dims;
pub mod model;
pub mod preflight;

pub use dims::{KvType, ModelDims};
pub use model::{Model, PartIdentity, TensorRef};
pub use preflight::Preflight;
