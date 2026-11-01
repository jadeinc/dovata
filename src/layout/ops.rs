//! Author: Joe Ricks
//! File: src/layout/ops.rs
//! Description: Defines the ggml ops a layout may use, their parameters, and the shape
//! each one produces.

use std::collections::BTreeMap;

use crate::gguf::ggml_type_layout;

use super::bind::Param;

/// `ggml_type` tags the shape rules refer to.
pub(crate) const F32: u32 = 0;
pub(crate) const F16: u32 = 1;
pub(crate) const I32: u32 = 26;
pub(crate) const I64: u32 = 27;

/// Purpose: This enum holds the ggml enum value an op name stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ggml {
    /// `GGML_OP_<name>`.
    Op(&'static str),
    /// `GGML_OP_UNARY` with `GGML_UNARY_OP_<name>`.
    Unary(&'static str),
    /// `GGML_OP_GLU` with `GGML_GLU_OP_<name>`.
    Glu(&'static str),
}

/// Purpose: This enum lists the kinds of op parameter and what each kind resolves to.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Kind {
    /// An integer expression.
    Int,
    /// A float expression. TOML's `inf` is allowed, for `clamp`.
    Float,
    /// An array of integer expressions.
    Ints,
    /// A `ggml_type` by name, such as `"f16"`. Resolves to its tag.
    Type,
    /// One of a fixed set of ggml constants by name. Resolves to its value.
    Enum(&'static [(&'static str, i64)]),
}

/// Purpose: This enum holds an op parameter's default: required, absent, or a value.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Init {
    /// The parameter must be given, in the op or in the layout's `[defaults.<op>]`.
    Required,
    /// The parameter may be left out entirely, with no value in the bound op.
    Absent,
    Int(i64),
    Float(f64),
    Name(&'static str),
}

/// Purpose: This struct holds one parameter an op takes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ParamDef {
    pub name: &'static str,
    pub kind: Kind,
    pub default: Init,
}

/// Purpose: This function builds one op parameter definition.
///
/// Inputs:
/// - `name`: the parameter's name.
/// - `kind`: how the parameter is written.
/// - `default`: its value when the layout leaves it out.
///
/// Outputs: the definition.
const fn p(name: &'static str, kind: Kind, default: Init) -> ParamDef {
    ParamDef { name, kind, default }
}

/// Purpose: This struct holds an op input's shape, byte strides and type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TInfo {
    pub ne: [u64; 4],
    pub nb: [u64; 4],
    pub ty: u32,
}

/// Purpose: This struct holds an op result's shape, type and memory sharing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Out {
    pub ne: [u64; 4],
    pub ty: u32,
    /// Set when the result shares memory with an input instead of having its own.
    pub alias: Option<Alias>,
}

/// Purpose: This struct holds where a view result sits inside one of the op's inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Alias {
    pub input: usize,
    pub offset: u64,
    pub nb: [u64; 4],
}

/// Purpose: This type alias names the signature of an op's shape rule.
type ShapeFn = fn(&[TInfo], &mut BTreeMap<String, Param>) -> Result<Out, String>;

/// Purpose: This struct holds one op a layout may use and its shape rule.
#[derive(Debug)]
pub(crate) struct OpDef {
    pub name: &'static str,
    pub ggml: Ggml,
    pub min_inputs: usize,
    pub max_inputs: usize,
    pub params: &'static [ParamDef],
    pub shape: ShapeFn,
}

const ROPE_MODES: &[(&str, i64)] = &[("norm", 0), ("neox", 2), ("mrope", 8), ("vision", 24), ("imrope", 40)];
const SORT_ORDER: &[(&str, i64)] = &[("asc", 0), ("desc", 1)];
const PREC: &[(&str, i64)] = &[("default", 0), ("f32", 10)];
const HINT: &[(&str, i64)] = &[("none", 0), ("src0_is_hadamard", 1)];

use Init::{Absent, Required};
use Kind::{Float, Int, Ints, Type};

/// The vocabulary. Inputs are in the order of the ggml API call's tensor arguments,
/// which for `set_rows` differs from ggml's internal `src[]` order; see its rule.
pub(crate) const OPS: &[OpDef] = &[
    // Element-wise, with b broadcast over a (ggml_can_repeat).
    OpDef { name: "add", ggml: Ggml::Op("ADD"), min_inputs: 2, max_inputs: 2, params: &[], shape: broadcast },
    OpDef { name: "sub", ggml: Ggml::Op("SUB"), min_inputs: 2, max_inputs: 2, params: &[], shape: broadcast },
    OpDef { name: "mul", ggml: Ggml::Op("MUL"), min_inputs: 2, max_inputs: 2, params: &[], shape: broadcast },
    OpDef { name: "div", ggml: Ggml::Op("DIV"), min_inputs: 2, max_inputs: 2, params: &[], shape: broadcast },
    OpDef { name: "sqr", ggml: Ggml::Op("SQR"), min_inputs: 1, max_inputs: 1, params: &[], shape: same },
    OpDef { name: "sqrt", ggml: Ggml::Op("SQRT"), min_inputs: 1, max_inputs: 1, params: &[], shape: same },
    OpDef {
        name: "scale",
        ggml: Ggml::Op("SCALE"),
        min_inputs: 1,
        max_inputs: 1,
        params: &[p("s", Float, Required), p("b", Float, Init::Float(0.0))],
        shape: same,
    },
    OpDef {
        name: "clamp",
        ggml: Ggml::Op("CLAMP"),
        min_inputs: 1,
        max_inputs: 1,
        params: &[p("min", Float, Required), p("max", Float, Required)],
        shape: same,
    },
    OpDef { name: "sum_rows", ggml: Ggml::Op("SUM_ROWS"), min_inputs: 1, max_inputs: 1, params: &[], shape: sum_rows },
    // Norms.
    OpDef { name: "norm", ggml: Ggml::Op("NORM"), min_inputs: 1, max_inputs: 1, params: &[p("eps", Float, Required)], shape: same },
    OpDef { name: "rms_norm", ggml: Ggml::Op("RMS_NORM"), min_inputs: 1, max_inputs: 1, params: &[p("eps", Float, Required)], shape: same },
    // Matrix products.
    OpDef {
        name: "mul_mat",
        ggml: Ggml::Op("MUL_MAT"),
        min_inputs: 2,
        max_inputs: 2,
        params: &[p("prec", Kind::Enum(PREC), Init::Name("default")), p("hint", Kind::Enum(HINT), Init::Name("none"))],
        shape: mul_mat,
    },
    OpDef { name: "mul_mat_id", ggml: Ggml::Op("MUL_MAT_ID"), min_inputs: 3, max_inputs: 3, params: &[], shape: mul_mat_id },
    // Copies and views.
    OpDef { name: "cpy", ggml: Ggml::Op("CPY"), min_inputs: 1, max_inputs: 2, params: &[p("type", Type, Absent)], shape: cpy },
    OpDef { name: "cont", ggml: Ggml::Op("CONT"), min_inputs: 1, max_inputs: 1, params: &[p("ne", Ints, Absent)], shape: cont },
    OpDef { name: "reshape", ggml: Ggml::Op("RESHAPE"), min_inputs: 1, max_inputs: 1, params: &[p("ne", Ints, Required)], shape: reshape },
    OpDef {
        name: "view",
        ggml: Ggml::Op("VIEW"),
        min_inputs: 1,
        max_inputs: 1,
        params: &[p("ne", Ints, Required), p("nb", Ints, Absent), p("offset", Int, Init::Int(0))],
        shape: view,
    },
    OpDef { name: "permute", ggml: Ggml::Op("PERMUTE"), min_inputs: 1, max_inputs: 1, params: &[p("axes", Ints, Required)], shape: permute },
    OpDef { name: "transpose", ggml: Ggml::Op("TRANSPOSE"), min_inputs: 1, max_inputs: 1, params: &[], shape: transpose },
    // Row gather and scatter.
    OpDef { name: "get_rows", ggml: Ggml::Op("GET_ROWS"), min_inputs: 2, max_inputs: 2, params: &[], shape: get_rows },
    OpDef { name: "set_rows", ggml: Ggml::Op("SET_ROWS"), min_inputs: 3, max_inputs: 3, params: &[], shape: set_rows },
    // Attention.
    OpDef {
        name: "soft_max",
        ggml: Ggml::Op("SOFT_MAX"),
        min_inputs: 1,
        max_inputs: 2,
        params: &[p("scale", Float, Init::Float(1.0)), p("max_bias", Float, Init::Float(0.0))],
        shape: soft_max,
    },
    OpDef {
        name: "rope",
        ggml: Ggml::Op("ROPE"),
        min_inputs: 2,
        max_inputs: 3,
        params: &[
            p("n_dims", Int, Required),
            p("mode", Kind::Enum(ROPE_MODES), Required),
            p("n_ctx_orig", Int, Required),
            p("freq_base", Float, Required),
            p("freq_scale", Float, Required),
            p("ext_factor", Float, Required),
            p("attn_factor", Float, Required),
            p("beta_fast", Float, Required),
            p("beta_slow", Float, Required),
        ],
        shape: rope,
    },
    OpDef {
        name: "flash_attn_ext",
        ggml: Ggml::Op("FLASH_ATTN_EXT"),
        min_inputs: 3,
        max_inputs: 4,
        params: &[
            p("scale", Float, Required),
            p("max_bias", Float, Init::Float(0.0)),
            p("logit_softcap", Float, Init::Float(0.0)),
            p("prec", Kind::Enum(PREC), Init::Name("default")),
        ],
        shape: flash_attn_ext,
    },
    // Expert routing.
    OpDef { name: "argsort", ggml: Ggml::Op("ARGSORT"), min_inputs: 1, max_inputs: 1, params: &[p("order", Kind::Enum(SORT_ORDER), Required)], shape: argsort },
    OpDef { name: "top_k", ggml: Ggml::Op("TOP_K"), min_inputs: 1, max_inputs: 1, params: &[p("k", Int, Required)], shape: top_k },
    // Activations.
    OpDef { name: "silu", ggml: Ggml::Unary("SILU"), min_inputs: 1, max_inputs: 1, params: &[], shape: same },
    OpDef { name: "gelu", ggml: Ggml::Unary("GELU"), min_inputs: 1, max_inputs: 1, params: &[], shape: same },
    OpDef { name: "relu", ggml: Ggml::Unary("RELU"), min_inputs: 1, max_inputs: 1, params: &[], shape: same },
    OpDef { name: "sigmoid", ggml: Ggml::Unary("SIGMOID"), min_inputs: 1, max_inputs: 1, params: &[], shape: same },
    OpDef { name: "tanh", ggml: Ggml::Unary("TANH"), min_inputs: 1, max_inputs: 1, params: &[], shape: same },
    OpDef { name: "exp", ggml: Ggml::Unary("EXP"), min_inputs: 1, max_inputs: 1, params: &[], shape: same },
    OpDef { name: "softplus", ggml: Ggml::Unary("SOFTPLUS"), min_inputs: 1, max_inputs: 1, params: &[], shape: same },
    // Gated linear units: with two inputs, ggml_<op>_split(a, b); with one, the halves of a.
    OpDef { name: "swiglu", ggml: Ggml::Glu("SWIGLU"), min_inputs: 1, max_inputs: 2, params: &[], shape: glu },
    OpDef { name: "geglu", ggml: Ggml::Glu("GEGLU"), min_inputs: 1, max_inputs: 2, params: &[], shape: glu },
    OpDef { name: "reglu", ggml: Ggml::Glu("REGLU"), min_inputs: 1, max_inputs: 2, params: &[], shape: glu },
    // Added for deepseek2: MLA joins the rope and non-rope parts of Q and K with concat,
    // and expert groups mask the unchosen groups with fill.
    OpDef { name: "concat", ggml: Ggml::Op("CONCAT"), min_inputs: 2, max_inputs: 2, params: &[p("dim", Int, Required)], shape: concat },
    OpDef { name: "fill", ggml: Ggml::Op("FILL"), min_inputs: 1, max_inputs: 1, params: &[p("c", Float, Required)], shape: fill },
    // Tiling, as ggml_repeat_4d builds it: llama4's build_moe_ffn copies each token's row
    // once per chosen expert, and deepseek2's legacy attention broadcasts K's rope part.
    OpDef { name: "repeat", ggml: Ggml::Op("REPEAT"), min_inputs: 1, max_inputs: 1, params: &[p("ne", Ints, Required)], shape: repeat },
];

/// Purpose: This function looks up an op by its layout name.
///
/// Inputs:
/// - `name`: the op name, such as `mul_mat`.
///
/// Outputs: the op's definition, or `None`.
pub(crate) fn op(name: &str) -> Option<&'static OpDef> {
    OPS.iter().find(|o| o.name == name)
}

/// Purpose: This function lists every op name a layout may use.
///
/// Inputs: none.
///
/// Outputs: the names, in table order.
pub fn names() -> impl Iterator<Item = &'static str> {
    OPS.iter().map(|o| o.name)
}

/// Purpose: This function returns the ggml enum value an op name stands for.
///
/// Inputs:
/// - `name`: the op name, such as `mul_mat`.
///
/// Outputs: the enum value, or `None` for an unknown name.
pub fn ggml_enum(name: &str) -> Option<Ggml> {
    op(name).map(|o| o.ggml)
}

/// Purpose: This function tells whether an op only reinterprets its input's memory.
///
/// Inputs:
/// - `name`: the op name, such as `mul_mat`.
///
/// Outputs: `true` for `view`, `reshape`, `permute` and `transpose`.
pub fn is_view_op(name: &str) -> bool {
    matches!(name, "view" | "reshape" | "permute" | "transpose")
}

// ---- tensor arithmetic, ported from ggml.c ----

/// Purpose: This function returns a ggml type's block size and bytes per block.
///
/// Inputs:
/// - `ty`: the ggml type number.
///
/// Outputs: the two sizes, or an error for an unknown type.
fn layout_of(ty: u32) -> Result<(u64, u64), String> {
    ggml_type_layout(ty).ok_or_else(|| format!("ggml type {ty} has no known size"))
}

/// Purpose: This function returns the bytes `n` elements of a ggml type take.
///
/// Inputs:
/// - `ty`: the ggml type number.
/// - `n`: the element count.
///
/// Outputs: the size in bytes, or an error for a partial block.
pub(crate) fn row_size(ty: u32, n: u64) -> Result<u64, String> {
    let (blck, size) = layout_of(ty)?;
    if !n.is_multiple_of(blck) {
        return Err(format!("{n} elements of {} is not a whole number of {blck}-element blocks", crate::gguf::ggml_type_name(ty)));
    }
    Ok(n / blck * size)
}

/// Purpose: This function returns the byte strides of a packed tensor.
///
/// Inputs:
/// - `ne`: the four dims.
/// - `ty`: the ggml type number.
///
/// Outputs: the four strides, or an error.
pub(crate) fn packed_nb(ne: [u64; 4], ty: u32) -> Result<[u64; 4], String> {
    let (_, size) = layout_of(ty)?;
    let nb1 = row_size(ty, ne[0])?;
    Ok([size, nb1, nb1 * ne[1], nb1 * ne[1] * ne[2]])
}

/// Purpose: This function returns the bytes a tensor's data spans.
///
/// Inputs:
/// - `ne`: the four dims.
/// - `nb`: the four byte strides.
/// - `ty`: the ggml type number.
///
/// Outputs: the size in bytes, or an error.
pub(crate) fn nbytes(ne: [u64; 4], nb: [u64; 4], ty: u32) -> Result<u64, String> {
    if ne.contains(&0) {
        return Ok(0);
    }
    let (blck, size) = layout_of(ty)?;
    let tail: u64 = (1..4).map(|i| (ne[i] - 1) * nb[i]).sum();
    Ok(if blck == 1 { size + (ne[0] - 1) * nb[0] + tail } else { ne[0] * nb[0] / blck + tail })
}

/// Purpose: This function counts a tensor's elements.
///
/// Inputs:
/// - `ne`: the four dims.
///
/// Outputs: the element count.
pub(crate) fn nelements(ne: [u64; 4]) -> u64 {
    ne.iter().product()
}

/// Purpose: This function tells whether a tensor is packed above dim `n`.
///
/// Inputs:
/// - `t`: the tensor's shape, strides and type.
/// - `n`: the highest dim that may have gaps.
///
/// Outputs: `true` when it is.
fn contiguous_n(t: &TInfo, n: usize) -> bool {
    let Ok((blck, size)) = layout_of(t.ty) else { return false };
    let mut next = size;
    if t.ne[0] != blck && t.nb[0] != next {
        return false;
    }
    next *= t.ne[0] / blck;
    for i in 1..4 {
        if t.ne[i] != 1 {
            if i > n {
                if t.nb[i] != next {
                    return false;
                }
                next *= t.ne[i];
            } else {
                next = t.ne[i] * t.nb[i];
            }
        }
    }
    true
}

/// Purpose: This function tells whether a tensor is fully packed.
///
/// Inputs:
/// - `t`: the tensor's shape, strides and type.
///
/// Outputs: `true` when it is.
pub(crate) fn is_contiguous(t: &TInfo) -> bool {
    contiguous_n(t, 0)
}

/// Purpose: This function turns a failed shape check into an error.
///
/// Inputs:
/// - `ok`: whether the check passed.
/// - `what`: the check, for the error.
///
/// Outputs: `Ok(())`, or the error.
fn check(ok: bool, what: &str) -> Result<(), String> {
    if ok { Ok(()) } else { Err(format!("ggml would refuse this: {what}")) }
}

/// Purpose: This function builds an op result that has its own memory.
///
/// Inputs:
/// - `ne`: the four dims.
/// - `ty`: the result's ggml type number.
///
/// Outputs: the result.
fn fresh(ne: [u64; 4], ty: u32) -> Out {
    Out { ne, ty, alias: None }
}

/// Purpose: This function reads a list parameter.
///
/// Inputs:
/// - `params`: the op's parameters.
/// - `name`: the parameter's name.
///
/// Outputs: the list, or `None`.
fn ints(params: &BTreeMap<String, Param>, name: &str) -> Option<Vec<i64>> {
    match params.get(name) {
        Some(Param::Ints(v)) => Some(v.clone()),
        _ => None,
    }
}

/// Purpose: This function reads an integer parameter.
///
/// Inputs:
/// - `params`: the op's parameters.
/// - `name`: the parameter's name.
///
/// Outputs: the integer, or `None`.
fn int(params: &BTreeMap<String, Param>, name: &str) -> Option<i64> {
    match params.get(name) {
        Some(Param::Int(v)) => Some(*v),
        _ => None,
    }
}

/// Purpose: This function pads one to four sizes out to four dims.
///
/// Inputs:
/// - `v`: the sizes.
/// - `what`: the parameter, for the error.
///
/// Outputs: the four dims, or an error.
fn dims4(v: &[i64], what: &str) -> Result<[u64; 4], String> {
    if v.is_empty() || v.len() > 4 {
        return Err(format!("{what} needs one to four sizes, got {}", v.len()));
    }
    let mut out = [1u64; 4];
    for (i, x) in v.iter().enumerate() {
        out[i] = u64::try_from(*x).map_err(|_| format!("{what} has a negative size {x}"))?;
    }
    Ok(out)
}

// ---- shape rules ----

/// Purpose: This function gives the result shape of ops whose result matches their first
/// input.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `_`: the op's parameters, which this rule does not read.
///
/// Outputs: the result's shape and type.
fn same(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    Ok(fresh(i[0].ne, i[0].ty))
}

/// Purpose: This function gives the result shape of `add`, `sub`, `mul` and `div`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `_`: the op's parameters, which this rule does not read.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn broadcast(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let (a, b) = (&i[0], &i[1]);
    check((0..4).all(|d| b.ne[d] != 0 && a.ne[d].is_multiple_of(b.ne[d])), "the second input does not broadcast over the first (ggml_can_repeat)")?;
    Ok(fresh(a.ne, a.ty))
}

/// Purpose: This function gives the result shape of `sum_rows`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `_`: the op's parameters, which this rule does not read.
///
/// Outputs: the result's shape and type.
fn sum_rows(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    Ok(fresh([1, a.ne[1], a.ne[2], a.ne[3]], a.ty))
}

/// Purpose: This function gives the result shape of `mul_mat`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `_`: the op's parameters, which this rule does not read.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn mul_mat(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let (a, b) = (&i[0], &i[1]);
    check(a.ne[0] == b.ne[0], &format!("mul_mat needs a.ne0 == b.ne0, got {} and {}", a.ne[0], b.ne[0]))?;
    check(b.ne[2].is_multiple_of(a.ne[2]) && b.ne[3].is_multiple_of(a.ne[3]), "mul_mat: a does not broadcast over b's dims 2 and 3")?;
    check(a.nb[0] <= a.nb[1], "mul_mat: the first input is transposed")?;
    Ok(fresh([a.ne[1], b.ne[1], b.ne[2], b.ne[3]], F32))
}

/// Purpose: This function gives the result shape of `mul_mat_id`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `_`: the op's parameters, which this rule does not read.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn mul_mat_id(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let (as_, b, ids) = (&i[0], &i[1], &i[2]);
    check(ids.ty == I32, "mul_mat_id: ids must be i32")?;
    check(as_.ne[3] == 1 && b.ne[3] == 1, "mul_mat_id: the experts and b must be 3d")?;
    check(ids.ne[2] == 1 && ids.ne[3] == 1, "mul_mat_id: ids must be 2d")?;
    check(ids.ne[1] == b.ne[2], &format!("mul_mat_id: ids.ne1 ({}) must equal b.ne2 ({})", ids.ne[1], b.ne[2]))?;
    check(as_.ne[0] == b.ne[0], &format!("mul_mat_id: as.ne0 ({}) must equal b.ne0 ({})", as_.ne[0], b.ne[0]))?;
    check(ids.ne[0].is_multiple_of(b.ne[1]), "mul_mat_id: b.ne1 does not broadcast over ids.ne0")?;
    Ok(fresh([as_.ne[1], ids.ne[0], b.ne[2], 1], F32))
}

/// Purpose: This function gives the result shape of `cpy`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `params`: the op's parameters.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn cpy(i: &[TInfo], params: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let ty = int(params, "type");
    if i.len() == 2 {
        check(ty.is_none(), "cpy into a second input takes its type from that input; drop `type`")?;
        check(nelements(i[0].ne) == nelements(i[1].ne), "cpy: the inputs differ in element count")?;
        return Ok(Out { ne: i[1].ne, ty: i[1].ty, alias: Some(Alias { input: 1, offset: 0, nb: i[1].nb }) });
    }
    let ty = ty.ok_or("cpy with one input is ggml_cast and needs a `type`")?;
    Ok(fresh(i[0].ne, u32::try_from(ty).map_err(|_| "bad type")?))
}

/// Purpose: This function gives the result shape of `cont`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `params`: the op's parameters.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn cont(i: &[TInfo], params: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    let ne = match ints(params, "ne") {
        Some(v) => dims4(&v, "cont's ne")?,
        None => a.ne,
    };
    check(nelements(ne) == nelements(a.ne), "cont: ne changes the element count")?;
    Ok(fresh(ne, a.ty))
}

/// Purpose: This function gives the result shape of `reshape`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `params`: the op's parameters.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn reshape(i: &[TInfo], params: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    let ne = dims4(&ints(params, "ne").unwrap_or_default(), "reshape's ne")?;
    check(is_contiguous(a), "reshape needs a contiguous input; put a `cont` before it")?;
    check(nelements(ne) == nelements(a.ne), &format!("reshape from {:?} to {:?} changes the element count", a.ne, ne))?;
    Ok(Out { ne, ty: a.ty, alias: Some(Alias { input: 0, offset: 0, nb: packed_nb(ne, a.ty)? }) })
}

/// Purpose: This function gives the result shape of `view`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `params`: the op's parameters.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn view(i: &[TInfo], params: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    let given = ints(params, "ne").unwrap_or_default();
    let ne = dims4(&given, "view's ne")?;
    let packed = packed_nb(ne, a.ty)?;
    let mut nb = packed;
    if let Some(strides) = ints(params, "nb") {
        check(strides.len() + 1 == given.len(), &format!("a {}d view takes {} strides in `nb`, got {}", given.len(), given.len() - 1, strides.len()))?;
        for (d, s) in strides.iter().enumerate() {
            nb[d + 1] = u64::try_from(*s).map_err(|_| "a negative stride")?;
        }
        // ggml_view_impl fills the unused dims from the last given stride.
        for d in given.len()..4 {
            nb[d] = nb[d - 1] * ne[d - 1];
        }
    }
    let offset = u64::try_from(int(params, "offset").unwrap_or(0)).map_err(|_| "a negative offset")?;
    // Record the strides the view really has, so the bound op says it outright.
    params.insert("nb".into(), Param::Ints(nb[1..given.len()].iter().map(|v| i64::try_from(*v).unwrap_or(i64::MAX)).collect()));
    Ok(Out { ne, ty: a.ty, alias: Some(Alias { input: 0, offset, nb }) })
}

/// Purpose: This function gives the result shape of `permute`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `params`: the op's parameters.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn permute(i: &[TInfo], params: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    let axes = ints(params, "axes").unwrap_or_default();
    check(axes.len() == 4, "permute takes four axes")?;
    let mut seen = [false; 4];
    let (mut ne, mut nb) = ([0u64; 4], [0u64; 4]);
    for (d, ax) in axes.iter().enumerate() {
        let ax = usize::try_from(*ax).ok().filter(|x| *x < 4).ok_or("permute: an axis is outside 0..4")?;
        check(!seen[ax], "permute: an axis repeats")?;
        seen[ax] = true;
        ne[ax] = a.ne[d];
        nb[ax] = a.nb[d];
    }
    Ok(Out { ne, ty: a.ty, alias: Some(Alias { input: 0, offset: 0, nb }) })
}

/// Purpose: This function gives the result shape of `transpose`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `_`: the op's parameters, which this rule does not read.
///
/// Outputs: the result's shape and type.
fn transpose(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    let ne = [a.ne[1], a.ne[0], a.ne[2], a.ne[3]];
    let nb = [a.nb[1], a.nb[0], a.nb[2], a.nb[3]];
    Ok(Out { ne, ty: a.ty, alias: Some(Alias { input: 0, offset: 0, nb }) })
}

/// Purpose: This function gives the result shape of `get_rows`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `_`: the op's parameters, which this rule does not read.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn get_rows(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let (a, b) = (&i[0], &i[1]);
    check(a.ne[2] == b.ne[1] && a.ne[3] == b.ne[2], "get_rows: the rows' batch dims do not match the ids")?;
    check(b.ne[3] == 1, "get_rows: ids must be at most 3d")?;
    check(b.ty == I32, "get_rows: ids must be i32")?;
    Ok(fresh([a.ne[0], b.ne[0], b.ne[1], b.ne[2]], if a.ty == I32 { I32 } else { F32 }))
}

/// Purpose: This function gives the result shape of `set_rows`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `_`: the op's parameters, which this rule does not read.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn set_rows(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let (a, b, c) = (&i[0], &i[1], &i[2]);
    check(a.ne[0] == b.ne[0], &format!("set_rows: row width {} into a destination of width {}", b.ne[0], a.ne[0]))?;
    check(a.ne[2] == b.ne[2] && a.ne[3] == b.ne[3], "set_rows: dims 2 and 3 must match")?;
    check(b.ne[1] == c.ne[0], &format!("set_rows: {} rows but {} ids", b.ne[1], c.ne[0]))?;
    check(b.ne[2].is_multiple_of(c.ne[1]) && b.ne[3].is_multiple_of(c.ne[2]) && c.ne[3] == 1, "set_rows: ids do not broadcast")?;
    check(b.ty == F32, "set_rows: the rows must be f32")?;
    check(c.ty == I64 || c.ty == I32, "set_rows: ids must be i64 or i32")?;
    Ok(Out { ne: a.ne, ty: a.ty, alias: Some(Alias { input: 0, offset: 0, nb: a.nb }) })
}

/// Purpose: This function gives the result shape of `soft_max`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `_`: the op's parameters, which this rule does not read.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn soft_max(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    check(is_contiguous(a), "soft_max needs a contiguous input")?;
    if let Some(m) = i.get(1) {
        check(m.ty == F16 || m.ty == F32, "soft_max: the mask must be f16 or f32")?;
        check(is_contiguous(m), "soft_max: the mask must be contiguous")?;
        check(m.ne[0] == a.ne[0] && m.ne[1] >= a.ne[1], "soft_max: the mask does not cover the input")?;
        check(a.ne[2].is_multiple_of(m.ne[2]) && a.ne[3].is_multiple_of(m.ne[3]), "soft_max: the mask does not broadcast")?;
    }
    Ok(fresh(a.ne, a.ty))
}

/// Purpose: This function gives the result shape of `rope`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `params`: the op's parameters.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn rope(i: &[TInfo], params: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let (a, pos) = (&i[0], &i[1]);
    let mode = int(params, "mode").unwrap_or(0);
    check(pos.ty == I32 && pos.ne[1] == 1 && pos.ne[2] == 1 && pos.ne[3] == 1, "rope: positions must be an i32 vector")?;
    if mode & 8 != 0 {
        check(a.ne[2] * 4 == pos.ne[0], "rope: mrope needs four positions per token")?;
    } else {
        check(a.ne[2] == pos.ne[0], &format!("rope: {} tokens but {} positions", a.ne[2], pos.ne[0]))?;
    }
    if let Some(f) = i.get(2) {
        let n_dims = u64::try_from(int(params, "n_dims").unwrap_or(0)).unwrap_or(0);
        check(f.ty == F32 && f.ne[0] >= n_dims / 2, "rope: the frequency factors must be f32 and cover n_dims / 2")?;
    }
    Ok(fresh(a.ne, a.ty))
}

/// Purpose: This function gives the result shape of `flash_attn_ext`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `_`: the op's parameters, which this rule does not read.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn flash_attn_ext(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let (q, k, v) = (&i[0], &i[1], &i[2]);
    check(k.ne[0] == q.ne[0], &format!("flash_attn_ext: K head width {} and Q head width {}", k.ne[0], q.ne[0]))?;
    check(q.ne[2].is_multiple_of(k.ne[2]) && q.ne[3].is_multiple_of(k.ne[3]), "flash_attn_ext: Q heads are not a multiple of K heads")?;
    check(q.ne[3] == k.ne[3] && q.ne[3] == v.ne[3], "flash_attn_ext: Q, K and V differ in dim 3")?;
    check(k.ne[1] == v.ne[1], "flash_attn_ext: K and V differ in length")?;
    if let Some(m) = i.get(3) {
        check(m.ty == F16, "flash_attn_ext: the mask must be f16")?;
        check(is_contiguous(m), "flash_attn_ext: the mask must be contiguous")?;
        check(m.ne[0] == k.ne[1] && m.ne[1] >= q.ne[1], "flash_attn_ext: the mask does not cover the KV length and the tokens")?;
        check(q.ne[2].is_multiple_of(m.ne[2]) && q.ne[3].is_multiple_of(m.ne[3]), "flash_attn_ext: the mask does not broadcast")?;
    }
    // ggml returns the result permuted (0, 2, 1, 3): [head_dim_v, n_head, n_tokens].
    Ok(fresh([v.ne[0], q.ne[2], q.ne[1], q.ne[3]], F32))
}

/// Purpose: This function gives the result shape of `argsort`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `_`: the op's parameters, which this rule does not read.
///
/// Outputs: the result's shape and type.
fn argsort(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    Ok(fresh(i[0].ne, I32))
}

/// Purpose: This function gives the result shape of `top_k`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `params`: the op's parameters.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn top_k(i: &[TInfo], params: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    let k = u64::try_from(int(params, "k").unwrap_or(0)).map_err(|_| "top_k: k is negative")?;
    check(a.ne[0] >= k, "top_k: k is larger than the row")?;
    Ok(fresh([k, a.ne[1], a.ne[2], a.ne[3]], I32))
}

/// Purpose: This function gives the result shape of `swiglu`, `geglu` and `reglu`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `_`: the op's parameters, which this rule does not read.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn glu(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    check(contiguous_n(a, 1), "a gated unit needs contiguous rows")?;
    if let Some(b) = i.get(1) {
        check(contiguous_n(b, 1), "a gated unit needs contiguous rows")?;
        check(a.ne == b.ne && a.ty == b.ty, "a split gated unit needs two inputs of the same shape and type")?;
        return Ok(fresh(a.ne, a.ty));
    }
    Ok(fresh([a.ne[0] / 2, a.ne[1], a.ne[2], a.ne[3]], a.ty))
}

/// Purpose: This function gives the result shape of `concat`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `params`: the op's parameters.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the inputs.
fn concat(i: &[TInfo], params: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let (a, b) = (&i[0], &i[1]);
    // ggml_concat: the two inputs share a type and every dim but `dim`, which adds up.
    let dim = int(params, "dim").and_then(|d| usize::try_from(d).ok()).filter(|d| *d < 4).ok_or("concat: dim must be 0, 1, 2 or 3")?;
    check(a.ty == b.ty, "concat: the inputs differ in type")?;
    let mut ne = a.ne;
    for (d, n) in ne.iter_mut().enumerate() {
        if d == dim {
            *n += b.ne[d];
        } else {
            check(*n == b.ne[d], &format!("concat along dim {dim}: dim {d} is {n} in the first input and {} in the second", b.ne[d]))?;
        }
    }
    Ok(fresh(ne, a.ty))
}

/// Purpose: This function gives the result shape of `fill`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `_`: the op's parameters, which this rule does not read.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the input.
fn fill(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    // ggml_fill_impl: a new tensor shaped like the input, every element set to `c`.
    check(a.ty == F32 || a.ty == F16, "fill: the input must be f32 or f16")?;
    check(is_contiguous(a), "fill needs a contiguous input")?;
    Ok(fresh(a.ne, a.ty))
}

/// Purpose: This function gives the result shape of `repeat`.
///
/// Inputs:
/// - `i`: the op's inputs.
/// - `params`: the op's parameters.
///
/// Outputs: the result's shape and type, or an error when ggml would refuse the input.
fn repeat(i: &[TInfo], params: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    let ne = dims4(&ints(params, "ne").unwrap_or_default(), "repeat's ne")?;
    // ggml_can_repeat: every dim of the result is a whole number of the input's, and an
    // empty input repeats only into an empty result.
    let ok = if nelements(a.ne) == 0 { nelements(ne) == 0 } else { (0..4).all(|d| ne[d].is_multiple_of(a.ne[d])) };
    check(ok, &format!("repeat: {:?} is not a whole number of copies of {:?} (ggml_can_repeat)", ne, a.ne))?;
    Ok(fresh(ne, a.ty))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Purpose: This function builds a packed test tensor.
    ///
    /// Inputs:
    /// - `ne`: the four dims.
    /// - `ty`: the ggml type number.
    ///
    /// Outputs: the tensor's shape, strides and type.
    fn t(ne: [u64; 4], ty: u32) -> TInfo {
        TInfo { ne, nb: packed_nb(ne, ty).unwrap(), ty }
    }

    #[test]
    fn mul_mat_shape_and_refusal() {
        let w = t([2048, 4096, 1, 1], 12);
        let x = t([2048, 7, 1, 1], F32);
        let out = mul_mat(&[w, x], &mut BTreeMap::new()).unwrap();
        assert_eq!(out.ne, [4096, 7, 1, 1]);
        assert_eq!(out.ty, F32);
        let wrong = t([1024, 7, 1, 1], F32);
        assert!(mul_mat(&[w, wrong], &mut BTreeMap::new()).is_err());
    }

    #[test]
    fn permute_moves_dims_and_strides() {
        let q = t([128, 32, 7, 1], F32);
        let mut p = BTreeMap::from([("axes".to_string(), Param::Ints(vec![0, 2, 1, 3]))]);
        let out = permute(&[q], &mut p).unwrap();
        assert_eq!(out.ne, [128, 7, 32, 1]);
        assert_eq!(out.alias.unwrap().nb, [4, 128 * 32 * 4, 128 * 4, 128 * 32 * 7 * 4]);
        assert!(!is_contiguous(&TInfo { ne: out.ne, nb: out.alias.unwrap().nb, ty: F32 }));
    }

    #[test]
    fn a_quantized_view_strides_in_whole_blocks() {
        // A q4_0 K cache: 512 wide (4 heads of 128), 4096 cells.
        assert_eq!(row_size(2, 128).unwrap(), 128 / 32 * 18);
        assert!(row_size(2, 100).is_err(), "half a block is not addressable");
        let ne = [512, 4096, 1, 1];
        assert_eq!(nbytes(ne, packed_nb(ne, 2).unwrap(), 2).unwrap(), 512 / 32 * 18 * 4096);
    }

    #[test]
    fn reshape_refuses_a_permuted_input() {
        let q = t([128, 32, 7, 1], F32);
        let mut p = BTreeMap::from([("axes".to_string(), Param::Ints(vec![0, 2, 1, 3]))]);
        let o = permute(&[q], &mut p).unwrap();
        let permuted = TInfo { ne: o.ne, nb: o.alias.unwrap().nb, ty: F32 };
        let mut p = BTreeMap::from([("ne".to_string(), Param::Ints(vec![128 * 32, 7]))]);
        assert!(reshape(&[permuted], &mut p).unwrap_err().contains("cont"));
    }

    #[test]
    fn concat_fill_and_repeat_follow_ggml() {
        // MLA's Q: the absorbed part, 512 per head, then the rope part, 64 per head.
        let absorbed = t([512, 16, 7, 1], F32);
        let rope = t([64, 16, 7, 1], F32);
        let mut dim0 = BTreeMap::from([("dim".to_string(), Param::Int(0))]);
        assert_eq!(concat(&[absorbed, rope], &mut dim0).unwrap().ne, [576, 16, 7, 1]);
        let mut dim1 = BTreeMap::from([("dim".to_string(), Param::Int(1))]);
        assert!(concat(&[absorbed, rope], &mut dim1).is_err(), "dim 0 differs, so they join only along it");
        assert!(concat(&[absorbed, t([64, 16, 7, 1], F16)], &mut dim0).is_err(), "the types differ");
        // fill keeps the shape and type, and needs contiguous f32 or f16.
        assert_eq!(fill(&[rope], &mut BTreeMap::new()).unwrap(), fresh(rope.ne, F32));
        assert!(fill(&[t([64, 16, 1, 1], I32)], &mut BTreeMap::new()).is_err());
        // repeat: K's one rope head copied to 16 heads, but not to a width it does not divide.
        let k_pe = t([64, 1, 7, 1], F32);
        let mut heads = BTreeMap::from([("ne".to_string(), Param::Ints(vec![64, 16, 7]))]);
        assert_eq!(repeat(&[k_pe], &mut heads).unwrap().ne, [64, 16, 7, 1]);
        let mut odd = BTreeMap::from([("ne".to_string(), Param::Ints(vec![96, 16, 7]))]);
        assert!(repeat(&[k_pe], &mut odd).is_err());
    }
}
