//! The op vocabulary: which ggml ops a layout may use, their parameters, and the shape
//! each one produces.
//!
//! ## Why op names are ggml's graph node types
//!
//! A layout op is one node of a ggml graph, named by its `ggml_op` enum value in lower
//! case (`GGML_OP_MUL_MAT` is `mul_mat`). Two enum values are families rather than
//! kernels, `GGML_OP_UNARY` and `GGML_OP_GLU`, so those nodes are named by their sub-op
//! instead (`GGML_UNARY_OP_SILU` is `silu`, `GGML_GLU_OP_SWIGLU` is `swiglu`). Each name
//! is then exactly one kind of node the compiler must generate a kernel for, and each is
//! checked against the vendored `ggml.h` by `every_op_name_exists_in_ggml_h`.
//!
//! The API's function names would not do that. One node type has many constructors
//! (`ggml_rope`, `ggml_rope_ext`, `ggml_rope_ext_inplace`; `ggml_view_1d` to `_4d`), and
//! some functions build several nodes: `ggml_argsort_top_k` is an `argsort` and a
//! `view`, and `ggml_cast` is a `cpy`. A layout written in function names would hide
//! nodes the kernels must still implement.
//!
//! ## Why dovata works out shapes itself
//!
//! Each op below carries the shape rule from its constructor in `ggml/src/ggml.c`,
//! asserts included. Bind runs them, so a layout never states an activation's shape: it
//! falls out of the weights' real shapes and the bind options, the way it does in
//! llama.cpp. A mistake in the layout, such as a weight fed to the wrong matmul, fails
//! at bind time with the ggml assert it would have tripped, instead of planning a buffer
//! of the wrong size. An op may still state `dims`, which bind checks.
//!
//! An op not listed here is refused with a pointer to this file. Adding one means
//! porting its constructor's shape rule, which is the point: dovata cannot plan memory
//! for a node whose output size it does not know.

use std::collections::BTreeMap;

use crate::gguf::ggml_type_layout;

use super::bind::Param;

/// `ggml_type` tags the shape rules refer to.
pub(crate) const F32: u32 = 0;
pub(crate) const F16: u32 = 1;
pub(crate) const I32: u32 = 26;
pub(crate) const I64: u32 = 27;

/// Which ggml enum value an op name stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ggml {
    /// `GGML_OP_<name>`.
    Op(&'static str),
    /// `GGML_OP_UNARY` with `GGML_UNARY_OP_<name>`.
    Unary(&'static str),
    /// `GGML_OP_GLU` with `GGML_GLU_OP_<name>`.
    Glu(&'static str),
}

/// How a parameter is written in the layout and what it resolves to.
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct ParamDef {
    pub name: &'static str,
    pub kind: Kind,
    pub default: Init,
}

const fn p(name: &'static str, kind: Kind, default: Init) -> ParamDef {
    ParamDef { name, kind, default }
}

/// What a shape rule sees of each input: its shape, byte strides and type, the same
/// three things ggml's constructors read from `ne`, `nb` and `type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TInfo {
    pub ne: [u64; 4],
    pub nb: [u64; 4],
    pub ty: u32,
}

/// What a shape rule produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Out {
    pub ne: [u64; 4],
    pub ty: u32,
    /// Set when the result shares memory with an input instead of having its own.
    pub alias: Option<Alias>,
}

/// The result is a view of input `input`, starting `offset` bytes into it, with byte
/// strides `nb`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Alias {
    pub input: usize,
    pub offset: u64,
    pub nb: [u64; 4],
}

type ShapeFn = fn(&[TInfo], &mut BTreeMap<String, Param>) -> Result<Out, String>;

/// One op of the vocabulary.
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
];

/// Look up an op by its layout name.
pub(crate) fn op(name: &str) -> Option<&'static OpDef> {
    OPS.iter().find(|o| o.name == name)
}

/// Every op name a layout may use, in table order.
pub fn names() -> impl Iterator<Item = &'static str> {
    OPS.iter().map(|o| o.name)
}

/// The ggml enum value a layout op name stands for, such as `Ggml::Unary("SILU")` for
/// `silu`. The compiler dispatches its kernels on this.
pub fn ggml_enum(name: &str) -> Option<Ggml> {
    op(name).map(|o| o.ggml)
}

/// Whether the op only reinterprets its input's memory and computes nothing. These are
/// the nodes ggml's backends skip. `set_rows` and a two-input `cpy` also return a view,
/// but they write into it, so they are compute nodes.
pub fn is_view_op(name: &str) -> bool {
    matches!(name, "view" | "reshape" | "permute" | "transpose")
}

// ---- tensor arithmetic, ported from ggml.c ----

fn layout_of(ty: u32) -> Result<(u64, u64), String> {
    ggml_type_layout(ty).ok_or_else(|| format!("ggml type {ty} has no known size"))
}

/// `ggml_row_size`: bytes for `n` elements of `ty`. Refuses a partial block.
pub(crate) fn row_size(ty: u32, n: u64) -> Result<u64, String> {
    let (blck, size) = layout_of(ty)?;
    if !n.is_multiple_of(blck) {
        return Err(format!("{n} elements of {} is not a whole number of {blck}-element blocks", crate::gguf::ggml_type_name(ty)));
    }
    Ok(n / blck * size)
}

/// Byte strides of a contiguous tensor, as `ggml_new_tensor_impl` sets them.
pub(crate) fn packed_nb(ne: [u64; 4], ty: u32) -> Result<[u64; 4], String> {
    let (_, size) = layout_of(ty)?;
    let nb1 = row_size(ty, ne[0])?;
    Ok([size, nb1, nb1 * ne[1], nb1 * ne[1] * ne[2]])
}

/// `ggml_nbytes`: the span a tensor's data covers, contiguous or not.
pub(crate) fn nbytes(ne: [u64; 4], nb: [u64; 4], ty: u32) -> Result<u64, String> {
    if ne.contains(&0) {
        return Ok(0);
    }
    let (blck, size) = layout_of(ty)?;
    let tail: u64 = (1..4).map(|i| (ne[i] - 1) * nb[i]).sum();
    Ok(if blck == 1 { size + (ne[0] - 1) * nb[0] + tail } else { ne[0] * nb[0] / blck + tail })
}

pub(crate) fn nelements(ne: [u64; 4]) -> u64 {
    ne.iter().product()
}

/// `ggml_is_contiguous_n`: dims above `n` must be packed; dims up to `n` may stride.
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

pub(crate) fn is_contiguous(t: &TInfo) -> bool {
    contiguous_n(t, 0)
}

fn check(ok: bool, what: &str) -> Result<(), String> {
    if ok { Ok(()) } else { Err(format!("ggml would refuse this: {what}")) }
}

fn fresh(ne: [u64; 4], ty: u32) -> Out {
    Out { ne, ty, alias: None }
}

fn ints(params: &BTreeMap<String, Param>, name: &str) -> Option<Vec<i64>> {
    match params.get(name) {
        Some(Param::Ints(v)) => Some(v.clone()),
        _ => None,
    }
}

fn int(params: &BTreeMap<String, Param>, name: &str) -> Option<i64> {
    match params.get(name) {
        Some(Param::Int(v)) => Some(*v),
        _ => None,
    }
}

/// Up to four sizes, padded with ones, as ggml's `_1d` to `_4d` constructors do.
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

fn same(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    Ok(fresh(i[0].ne, i[0].ty))
}

/// ggml_add_impl and friends: b must repeat into a.
fn broadcast(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let (a, b) = (&i[0], &i[1]);
    check((0..4).all(|d| b.ne[d] != 0 && a.ne[d].is_multiple_of(b.ne[d])), "the second input does not broadcast over the first (ggml_can_repeat)")?;
    Ok(fresh(a.ne, a.ty))
}

fn sum_rows(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    Ok(fresh([1, a.ne[1], a.ne[2], a.ne[3]], a.ty))
}

fn mul_mat(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let (a, b) = (&i[0], &i[1]);
    check(a.ne[0] == b.ne[0], &format!("mul_mat needs a.ne0 == b.ne0, got {} and {}", a.ne[0], b.ne[0]))?;
    check(b.ne[2].is_multiple_of(a.ne[2]) && b.ne[3].is_multiple_of(a.ne[3]), "mul_mat: a does not broadcast over b's dims 2 and 3")?;
    check(a.nb[0] <= a.nb[1], "mul_mat: the first input is transposed")?;
    Ok(fresh([a.ne[1], b.ne[1], b.ne[2], b.ne[3]], F32))
}

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

/// ggml_cast(a, type) is a one-input `cpy` into a new tensor; ggml_cpy(a, b) writes into
/// b and returns a view of it.
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

fn cont(i: &[TInfo], params: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    let ne = match ints(params, "ne") {
        Some(v) => dims4(&v, "cont's ne")?,
        None => a.ne,
    };
    check(nelements(ne) == nelements(a.ne), "cont: ne changes the element count")?;
    Ok(fresh(ne, a.ty))
}

fn reshape(i: &[TInfo], params: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    let ne = dims4(&ints(params, "ne").unwrap_or_default(), "reshape's ne")?;
    check(is_contiguous(a), "reshape needs a contiguous input; put a `cont` before it")?;
    check(nelements(ne) == nelements(a.ne), &format!("reshape from {:?} to {:?} changes the element count", a.ne, ne))?;
    Ok(Out { ne, ty: a.ty, alias: Some(Alias { input: 0, offset: 0, nb: packed_nb(ne, a.ty)? }) })
}

/// ggml_view_1d to _4d. `nb` lists nb1 and up, in bytes, the way the C call does; left
/// out, the view is packed. `offset` is in bytes.
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

/// ggml_permute: input dim d goes to position axes[d].
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

fn transpose(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    let ne = [a.ne[1], a.ne[0], a.ne[2], a.ne[3]];
    let nb = [a.nb[1], a.nb[0], a.nb[2], a.nb[3]];
    Ok(Out { ne, ty: a.ty, alias: Some(Alias { input: 0, offset: 0, nb }) })
}

fn get_rows(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let (a, b) = (&i[0], &i[1]);
    check(a.ne[2] == b.ne[1] && a.ne[3] == b.ne[2], "get_rows: the rows' batch dims do not match the ids")?;
    check(b.ne[3] == 1, "get_rows: ids must be at most 3d")?;
    check(b.ty == I32, "get_rows: ids must be i32")?;
    Ok(fresh([a.ne[0], b.ne[0], b.ne[1], b.ne[2]], if a.ty == I32 { I32 } else { F32 }))
}

/// ggml_set_rows(a, b, c) writes b's rows into a at the ids in c, and returns a view of
/// a. Inputs here are in that call order; ggml itself stores them as src = [b, c, a].
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

fn argsort(i: &[TInfo], _: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    Ok(fresh(i[0].ne, I32))
}

fn top_k(i: &[TInfo], params: &mut BTreeMap<String, Param>) -> Result<Out, String> {
    let a = &i[0];
    let k = u64::try_from(int(params, "k").unwrap_or(0)).map_err(|_| "top_k: k is negative")?;
    check(a.ne[0] >= k, "top_k: k is larger than the row")?;
    Ok(fresh([k, a.ne[1], a.ne[2], a.ne[3]], I32))
}

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
