//! Layouts: a model architecture's compute graph as data, in ggml's op vocabulary.
//!
//! ## Why this exists
//!
//! The Jade compiler generates the GPU kernels for a model, and dovata decides where
//! every tensor lives. Both need the same description of what one forward step
//! computes: which ops run, in what order, on which tensors, with which parameters. A
//! layout is that description, written once per architecture as a TOML file and
//! translated from llama.cpp's own graph code, so it can be read side by side with the
//! C++ it came from.
//!
//! A layout says nothing about a particular model. [`Layout::bind`] applies it to one
//! model file and one set of [`BindOptions`], and produces a [`BoundGraph`]: every op of
//! every layer in execution order, and every tensor with its concrete shape, type, byte
//! size and kind. The planner places the tensors of a bound graph; the compiler
//! generates a kernel per op.
//!
//! ## Why it is its own capability
//!
//! The layout is the contract with the compiler's kernels, and the planner only decides
//! where a bound graph's tensors go. Keeping them apart means the format can be
//! reviewed, versioned and extended without touching placement, and the planner never
//! needs to know about architectures.
//!
//! ## The pieces
//!
//! - `format`: reading a layout file and checking everything that needs no model.
//! - `expr`: the small expression language for dims, params and conditions.
//! - `symbols`: the closed set of hparam and option names expressions may use.
//! - [`ops`]: the ggml ops a layout may use, with their params and shape rules.
//! - `bind`: applying a layout to a model, and the bound types the planner reads.
//! - `builtin/*.layout.toml`: one layout per supported architecture.
//!
//! The format itself is documented in this directory's README.

mod bind;
mod expr;
mod format;
pub mod ops;
mod symbols;

#[cfg(test)]
pub(crate) mod testutil;
#[cfg(test)]
mod tests;

use std::fmt;

use crate::load::Model;

pub use bind::{BindOptions, BoundGraph, BoundOp, BoundTensor, Param, TensorId, TensorKind};
pub use format::{Layout, SCHEMA_VERSION};
pub use symbols::SYMBOLS;

/// The architectures with a built-in layout, as `general.architecture` spells them.
/// Only what exists: a name here always has a file in `builtin/`.
pub const SUPPORTED: &[&str] = &["llama", "qwen3", "qwen3moe"];

/// The text of a built-in layout, by architecture.
fn builtin_source(arch: &str) -> Option<&'static str> {
    Some(match arch {
        "llama" => include_str!("builtin/llama.layout.toml"),
        "qwen3" => include_str!("builtin/qwen3.layout.toml"),
        "qwen3moe" => include_str!("builtin/qwen3moe.layout.toml"),
        _ => return None,
    })
}

/// The built-in layout for an architecture, or `None` when dovata has none.
///
/// # Panics
///
/// When a built-in layout fails to parse, which `builtin_layouts_parse` rules out.
pub fn builtin(arch: &str) -> Option<Layout> {
    builtin_source(arch).map(|src| Layout::from_toml(src).unwrap_or_else(|e| panic!("built-in layout {arch} is invalid: {e}")))
}

/// The layout to plan a model with: the caller's when given, else the built-in one.
///
/// A caller's layout must be for the model's architecture; a layout for a different one
/// would bind against tensor names and hparams it was not written for. With no caller
/// layout and no built-in one, the error lists what is supported and says how to supply
/// a layout.
pub fn for_model(model: &Model, caller: Option<Layout>) -> Result<Layout, LayoutError> {
    let arch = model.architecture();
    match caller {
        Some(layout) if layout.arch() == arch => Ok(layout),
        Some(layout) => Err(LayoutError::WrongArch { layout: layout.arch().to_owned(), model: arch.to_owned() }),
        None => builtin(arch).ok_or_else(|| LayoutError::NoLayout { architecture: arch.to_owned(), supported: SUPPORTED }),
    }
}

/// Why a layout could not be read, chosen, or bound to a model.
#[derive(Debug, Clone, PartialEq)]
pub enum LayoutError {
    /// The text is not TOML, or lacks a field every layout has.
    Toml(String),
    /// The layout is not valid: an unknown op, param or symbol, a bad expression, a
    /// misplaced placeholder. `at` names the place in the file.
    Invalid { at: String, detail: String },
    /// No built-in layout exists for the model's architecture and the caller gave none.
    NoLayout { architecture: String, supported: &'static [&'static str] },
    /// The caller's layout is for a different architecture than the model.
    WrongArch { layout: String, model: String },
    /// The bind options do not describe a possible step.
    Options(String),
    /// A value could not be worked out for this model: a missing hparam, an expression
    /// that divides by zero.
    Eval { at: String, detail: String },
    /// A weight the layout requires is not in the model file.
    MissingWeight { name: String },
    /// A weight's shape in the file differs from the layout's `dims`.
    WeightShape { name: String, layout: Vec<u64>, file: Vec<u64> },
    /// An op reads an optional weight this model lacks, without a `when` or a `?`.
    AbsentWeight { name: String, at: String },
    /// An op reads a tensor that is neither declared nor produced by any op.
    UnknownTensor { name: String, at: String },
    /// An op reads an activation before the op that produces it has run.
    UseBeforeProduce { name: String, at: String },
    /// An op's inputs break the shape rule ggml asserts, or its stated dims are wrong.
    Shape { at: String, detail: String },
    /// Tensors in the model file that no op reads and `unused` does not list.
    UnreferencedTensors { names: Vec<String> },
    /// A model tensor whose `ggml_type` dovata cannot size.
    UnsupportedType { name: String, ggml_type: u32 },
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutError::Toml(e) => write!(f, "the layout is not valid TOML for a layout: {e}"),
            LayoutError::Invalid { at, detail } => write!(f, "invalid layout, at {at}: {detail}"),
            LayoutError::NoLayout { architecture, supported } => write!(
                f,
                "dovata has no built-in layout for architecture `{architecture}`. Built-in layouts: {}.\n  \
                 To plan this model anyway, write a layout for `{architecture}` (see src/layout/README.md, \
                 \"Adding a layout\") and pass it as the caller's layout.",
                supported.join(", ")
            ),
            LayoutError::WrongArch { layout, model } => write!(
                f,
                "the layout is for `{layout}`, but the model's general.architecture is `{model}`.\n  \
                 Pass a layout written for `{model}`, or pass none to use dovata's built-in one if it has one."
            ),
            LayoutError::Options(detail) => write!(f, "the bind options are not a possible step: {detail}"),
            LayoutError::Eval { at, detail } => write!(f, "at {at}: {detail}"),
            LayoutError::MissingWeight { name } => write!(
                f,
                "the model has no tensor {name}, which the layout requires. llama.cpp would refuse this file too. \
                 Check that the file is complete and converted for this architecture."
            ),
            LayoutError::WeightShape { name, layout, file } => {
                write!(f, "{name} has shape {file:?} in the model file, but the layout says {layout:?}")
            }
            LayoutError::AbsentWeight { name, at } => write!(
                f,
                "at {at}: the op reads {name}, which this model does not have, or which the weights table \
                 leaves out of this layer with its `when`. Guard the op with a matching `when`, such as \
                 `when = \"has({name})\"`, or mark the input optional with a trailing `?`."
            ),
            LayoutError::UnknownTensor { name, at } => write!(
                f,
                "at {at}: `{name}` is not a declared weight, input, KV or state tensor, and no op produces it. \
                 Declare it, or fix the name."
            ),
            LayoutError::UseBeforeProduce { name, at } => write!(
                f,
                "at {at}: `{name}` is read before any op has produced it here. Move the op that produces it \
                 earlier. A name first assigned in the block lives for one layer only."
            ),
            LayoutError::Shape { at, detail } => write!(f, "at {at}: {detail}"),
            LayoutError::UnreferencedTensors { names } => {
                let shown: Vec<&str> = names.iter().take(8).map(String::as_str).collect();
                write!(
                    f,
                    "{} tensor(s) in the model file are read by no op: {}{}. A layout must account for every \
                     weight. Add the ops that read them, or list them under `unused` with the reason llama.cpp \
                     loads them without reading them.",
                    names.len(),
                    shown.join(", "),
                    if names.len() > shown.len() { ", ..." } else { "" }
                )
            }
            LayoutError::UnsupportedType { name, ggml_type } => write!(
                f,
                "{name} has ggml type {ggml_type}, which dovata cannot size. Add it to gguf::ggml_type_layout."
            ),
        }
    }
}

impl std::error::Error for LayoutError {}
