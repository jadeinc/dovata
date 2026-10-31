//! Reading a layout file: TOML in, a checked [`Layout`] out.
//!
//! Everything that can be checked without a model is checked here, so a typo in a
//! layout fails when the layout is read, with the place it is in, instead of on the
//! first model someone binds it to. That covers op names, parameter names and kinds,
//! input counts, the syntax of every expression, the names expressions read, the
//! placeholders in tensor names, and whether every tensor an op reads is declared or
//! produced somewhere in the layout. What needs a model (presence, shapes, order) is
//! [`Layout::bind`](super::Layout::bind)'s job.
//!
//! The README in this directory is the reference for the format; this file is its
//! implementation.

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use serde::Deserialize;

use super::expr::Expr;
use super::ops::{self, Init, Kind, OpDef};
use super::symbols;
use super::LayoutError;

/// The format version this dovata reads.
pub const SCHEMA_VERSION: u32 = 1;

/// An architecture's compute graph, read from a layout file and checked.
///
/// Parse one with [`Layout::from_toml`], or get a built-in one with
/// [`super::builtin`]. A `Layout` does not know any model; [`Layout::bind`] applies it to
/// one and produces the [`super::BoundGraph`] the planner reads.
///
/// [`Layout::bind`]: super::Layout::bind
#[derive(Debug, Clone)]
pub struct Layout {
    pub(crate) arch: String,
    pub(crate) description: Option<String>,
    pub(crate) weights: Vec<WeightDecl>,
    pub(crate) unused: Vec<UnusedDecl>,
    pub(crate) tensors: Vec<TensorDecl>,
    pub(crate) prologue: Vec<OpSpec>,
    pub(crate) block: Vec<OpSpec>,
    pub(crate) epilogue: Vec<OpSpec>,
    pub(crate) outputs: Vec<String>,
    /// Every `out` template, to tell "produced later" from "never produced".
    pub(crate) out_names: HashSet<String>,
}

/// A weight the layout expects in the model file.
#[derive(Debug, Clone)]
pub(crate) struct WeightDecl {
    pub name: String,
    pub dims: Option<Vec<Expr>>,
    pub optional: bool,
    pub fallback: Option<String>,
    pub when: Option<Expr>,
}

/// A model tensor llama.cpp loads but one step's graph never reads.
#[derive(Debug, Clone)]
pub(crate) struct UnusedDecl {
    pub name: String,
}

/// What a declared non-weight tensor is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeclKind {
    Input,
    Kv,
    State,
}

/// A declared tensor's type: fixed, or the K or V cache type from the bind options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DType {
    Fixed(u32),
    KvK,
    KvV,
}

#[derive(Debug, Clone)]
pub(crate) struct TensorDecl {
    pub name: String,
    pub kind: DeclKind,
    pub dims: Vec<Expr>,
    pub dtype: DType,
    pub when: Option<Expr>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Section {
    Prologue,
    Block,
    Epilogue,
}

impl fmt::Display for Section {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Section::Prologue => "prologue",
            Section::Block => "block",
            Section::Epilogue => "epilogue",
        })
    }
}

/// One input of an op. `optional` inputs end in `?` and are dropped when the model
/// lacks the weight.
#[derive(Debug, Clone)]
pub(crate) struct InputRef {
    pub name: String,
    pub optional: bool,
}

/// `for = "e in 1..n_expert_used"`: run the op once per value, end exclusive.
#[derive(Debug, Clone)]
pub(crate) struct Loop {
    pub var: String,
    pub start: Expr,
    pub end: Expr,
}

/// A parameter as written, before a model gives it a value.
#[derive(Debug, Clone)]
pub(crate) enum RawParam {
    Expr(Expr),
    List(Vec<Expr>),
    /// An enum or type name, already turned into its ggml value.
    Const(i64),
}

#[derive(Debug, Clone)]
pub(crate) struct OpSpec {
    pub section: Section,
    pub index: usize,
    pub def: &'static OpDef,
    pub inputs: Vec<InputRef>,
    pub out: String,
    pub when: Option<Expr>,
    pub repeat: Option<Loop>,
    pub dims: Option<Vec<Expr>>,
    pub dtype: Option<u32>,
    /// The op's own params over the layout's `[defaults.<op>]`.
    pub params: BTreeMap<String, RawParam>,
}

impl OpSpec {
    /// Where this op is, for error messages: `block op 12 (rope)`.
    pub(crate) fn site(&self) -> String {
        format!("{} op {} ({})", self.section, self.index, self.def.name)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLayout {
    schema_version: u32,
    arch: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    weights: Vec<RawWeight>,
    #[serde(default)]
    unused: Vec<RawUnused>,
    #[serde(default)]
    tensors: Vec<RawTensor>,
    #[serde(default)]
    defaults: BTreeMap<String, toml::Table>,
    #[serde(default)]
    prologue: Vec<toml::Table>,
    #[serde(default)]
    block: Vec<toml::Table>,
    #[serde(default)]
    epilogue: Vec<toml::Table>,
    #[serde(default)]
    outputs: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWeight {
    name: String,
    dims: Option<Vec<toml::Value>>,
    #[serde(default)]
    optional: bool,
    fallback: Option<String>,
    when: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawUnused {
    name: String,
    /// Why llama.cpp loads it and the graph does not read it. Required, so the reason
    /// lives next to the exception.
    #[allow(dead_code)]
    why: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensor {
    name: String,
    kind: String,
    dims: Vec<toml::Value>,
    dtype: String,
    when: Option<String>,
}

/// The keys an op table may carry besides its params.
const OP_KEYS: [&str; 7] = ["op", "in", "out", "when", "for", "dims", "dtype"];

impl Layout {
    /// Read and check a layout.
    pub fn from_toml(src: &str) -> Result<Layout, LayoutError> {
        let raw: RawLayout = toml::from_str(src).map_err(|e| LayoutError::Toml(e.to_string()))?;
        let bad = |at: &str, detail: String| LayoutError::Invalid { at: at.to_owned(), detail };

        if raw.schema_version != SCHEMA_VERSION {
            return Err(bad(
                "schema_version",
                format!("the layout is schema version {}, and this dovata reads version {SCHEMA_VERSION}", raw.schema_version),
            ));
        }
        if raw.arch.trim().is_empty() {
            return Err(bad("arch", "arch is empty; it must match the model's general.architecture".into()));
        }

        // Declarations first: ops are checked against them.
        let mut declared: HashSet<String> = HashSet::new();
        let mut weights = Vec::new();
        for (i, w) in raw.weights.into_iter().enumerate() {
            let at = format!("weights[{i}] ({})", w.name);
            let layered = check_template(&w.name, &[], &at)?;
            let ctx = Ctx { layer: layered, var: None, src: false };
            let dims = w.dims.map(|d| d.iter().map(|v| dim_expr(v, &ctx, &at)).collect::<Result<Vec<_>, _>>()).transpose()?;
            let when = w.when.map(|s| cond_expr(&s, &ctx, &at)).transpose()?;
            if let Some(f) = &w.fallback {
                check_template(f, &[], &at)?;
            }
            if !declared.insert(w.name.clone()) {
                return Err(bad(&at, "declared twice".into()));
            }
            weights.push(WeightDecl { optional: w.optional || w.fallback.is_some(), name: w.name, dims, fallback: w.fallback, when });
        }
        for w in &weights {
            let at = format!("weights ({})", w.name);
            if let Some(f) = &w.fallback {
                if !weights.iter().any(|o| &o.name == f) {
                    return Err(bad(&at, format!("its fallback {f} is not a declared weight")));
                }
            }
            // A misspelled has() would be false everywhere and quietly drop the weight.
            let mut has = Vec::new();
            if let Some(cond) = &w.when {
                cond.has_names(&mut has);
            }
            for h in has {
                if !weights.iter().any(|o| o.name == h) {
                    return Err(bad(&at, format!("has({h}) names a tensor the weights table does not declare")));
                }
            }
        }

        let mut unused = Vec::new();
        for (i, u) in raw.unused.into_iter().enumerate() {
            let at = format!("unused[{i}] ({})", u.name);
            check_template(&u.name, &[], &at)?;
            if declared.contains(&u.name) {
                return Err(bad(&at, "also declared as a weight; a tensor is one or the other".into()));
            }
            unused.push(UnusedDecl { name: u.name });
        }

        let mut tensors = Vec::new();
        for (i, t) in raw.tensors.into_iter().enumerate() {
            let at = format!("tensors[{i}] ({})", t.name);
            let layered = check_template(&t.name, &[], &at)?;
            let ctx = Ctx { layer: layered, var: None, src: false };
            let kind = match t.kind.as_str() {
                "input" => DeclKind::Input,
                "kv" => DeclKind::Kv,
                "state" => DeclKind::State,
                "weight" => return Err(bad(&at, "weights go in the weights table, which knows about optional tensors and fallbacks".into())),
                "activation" => {
                    return Err(bad(&at, "activations are not declared; the op that produces one names it with `out`, and its shape follows from ggml's rules".into()))
                }
                other => return Err(bad(&at, format!("unknown kind `{other}`; the kinds are input, kv and state"))),
            };
            let dtype = match t.dtype.as_str() {
                "kv_k" => DType::KvK,
                "kv_v" => DType::KvV,
                name => DType::Fixed(type_by_name(name).ok_or_else(|| bad(&at, format!("unknown dtype `{name}`")))?),
            };
            if kind != DeclKind::Kv && matches!(dtype, DType::KvK | DType::KvV) {
                return Err(bad(&at, "only kv tensors take their type from the bind options".into()));
            }
            let dims = t.dims.iter().map(|v| dim_expr(v, &ctx, &at)).collect::<Result<Vec<_>, _>>()?;
            if dims.is_empty() || dims.len() > 4 {
                return Err(bad(&at, "dims needs one to four sizes".into()));
            }
            let when = t.when.map(|s| cond_expr(&s, &ctx, &at)).transpose()?;
            let mut has = Vec::new();
            if let Some(cond) = &when {
                cond.has_names(&mut has);
            }
            for h in has {
                if !weights.iter().any(|o| o.name == h) {
                    return Err(bad(&at, format!("has({h}) names a tensor the weights table does not declare")));
                }
            }
            if !declared.insert(t.name.clone()) {
                return Err(bad(&at, "declared twice".into()));
            }
            tensors.push(TensorDecl { name: t.name, kind, dims, dtype, when });
        }

        // Per-op defaults, checked like an op's own params.
        let mut defaults: BTreeMap<&'static str, BTreeMap<String, RawParam>> = BTreeMap::new();
        for (name, table) in &raw.defaults {
            let at = format!("defaults.{name}");
            let def = ops::op(name).ok_or_else(|| bad(&at, unknown_op(name)))?;
            let ctx = Ctx { layer: true, var: None, src: def.min_inputs > 0 };
            let mut params = BTreeMap::new();
            for (k, v) in table {
                params.insert(k.clone(), param(def, k, v, &ctx, &at)?);
            }
            defaults.insert(def.name, params);
        }

        let has_names: HashSet<&str> = weights.iter().map(|w| w.name.as_str()).collect();
        let mut out_names = HashSet::new();
        let mut sections = Vec::new();
        for (section, tables) in [(Section::Prologue, raw.prologue), (Section::Block, raw.block), (Section::Epilogue, raw.epilogue)] {
            let mut specs = Vec::new();
            for (index, table) in tables.into_iter().enumerate() {
                let spec = op_spec(section, index, &table, &defaults, &has_names)?;
                if declared.contains(&spec.out) {
                    return Err(bad(&spec.site(), format!("`{}` is a declared tensor; an op cannot assign to it", spec.out)));
                }
                out_names.insert(spec.out.clone());
                specs.push(spec);
            }
            sections.push(specs);
        }

        // Every input must be declared or produced somewhere; order is bind's question.
        for spec in sections.iter().flatten() {
            for input in &spec.inputs {
                if !declared.contains(&input.name) && !out_names.contains(&input.name) {
                    return Err(LayoutError::UnknownTensor { name: input.name.clone(), at: spec.site() });
                }
                if input.optional && !has_names.contains(input.name.as_str()) {
                    return Err(bad(&spec.site(), format!("only a weight can be an optional input, and `{}` is not one", input.name)));
                }
            }
        }
        for o in &raw.outputs {
            if !out_names.contains(o) {
                return Err(bad("outputs", format!("`{o}` is not produced by any op")));
            }
        }

        let epilogue = sections.pop().unwrap_or_default();
        let block = sections.pop().unwrap_or_default();
        let prologue = sections.pop().unwrap_or_default();
        Ok(Layout {
            arch: raw.arch,
            description: raw.description,
            weights,
            unused,
            tensors,
            prologue,
            block,
            epilogue,
            outputs: raw.outputs,
            out_names,
        })
    }

    /// The `general.architecture` this layout is for.
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// The layout's one-line description of the architecture, if it has one.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

/// Which names an expression may read where it is written.
struct Ctx<'a> {
    /// `L` is in scope: a block op, or a declaration whose name has `{L}`.
    layer: bool,
    /// The op's loop variable, if it has one.
    var: Option<&'a str>,
    /// `src.*` and `row_size` are in scope: an op param with at least one input.
    src: bool,
}

const SRC_NAMES: [&str; 8] = ["src.ne0", "src.ne1", "src.ne2", "src.ne3", "src.nb0", "src.nb1", "src.nb2", "src.nb3"];

fn check_names(e: &Expr, ctx: &Ctx<'_>, at: &str) -> Result<(), LayoutError> {
    let mut names = Vec::new();
    e.names(&mut names);
    for n in names {
        let ok = symbols::is_symbol(n) || (ctx.layer && n == "L") || ctx.var == Some(n) || (ctx.src && SRC_NAMES.contains(&n));
        if !ok {
            let hint = if n == "L" {
                "L is the layer index and exists only in the block".to_string()
            } else if n.starts_with("src.") {
                "src.ne0..3 and src.nb0..3 exist only in the params of an op with an input".to_string()
            } else {
                format!("the symbols are: {}", symbols::SYMBOLS.iter().map(|(s, _)| *s).collect::<Vec<_>>().join(", "))
            };
            return Err(LayoutError::Invalid { at: at.to_owned(), detail: format!("unknown name `{n}`; {hint}") });
        }
    }
    if !ctx.src && e.uses_row_size() {
        return Err(LayoutError::Invalid { at: at.to_owned(), detail: "row_size() exists only in the params of an op with an input".into() });
    }
    Ok(())
}

fn parse_expr(src: &str, ctx: &Ctx<'_>, at: &str) -> Result<Expr, LayoutError> {
    let e = Expr::parse(src).map_err(|detail| LayoutError::Invalid { at: at.to_owned(), detail })?;
    check_names(&e, ctx, at)?;
    let mut has = Vec::new();
    e.has_names(&mut has);
    if !has.is_empty() {
        return Err(LayoutError::Invalid { at: at.to_owned(), detail: "has() belongs in a `when`".into() });
    }
    Ok(e)
}

fn cond_expr(src: &str, ctx: &Ctx<'_>, at: &str) -> Result<Expr, LayoutError> {
    let e = Expr::parse(src).map_err(|detail| LayoutError::Invalid { at: at.to_owned(), detail })?;
    check_names(&e, ctx, at)?;
    Ok(e)
}

fn value_expr(v: &toml::Value, ctx: &Ctx<'_>, at: &str) -> Result<Expr, LayoutError> {
    match v {
        toml::Value::Integer(i) => Ok(Expr::Int(*i)),
        toml::Value::Float(f) => Ok(Expr::Float(*f)),
        toml::Value::String(s) => parse_expr(s, ctx, at),
        other => Err(LayoutError::Invalid { at: at.to_owned(), detail: format!("expected a number or an expression, got {other}") }),
    }
}

fn dim_expr(v: &toml::Value, ctx: &Ctx<'_>, at: &str) -> Result<Expr, LayoutError> {
    value_expr(v, ctx, at)
}

fn param(def: &OpDef, name: &str, v: &toml::Value, ctx: &Ctx<'_>, at: &str) -> Result<RawParam, LayoutError> {
    let bad = |detail: String| LayoutError::Invalid { at: at.to_owned(), detail };
    let pd = def.params.iter().find(|p| p.name == name).ok_or_else(|| {
        let known: Vec<&str> = def.params.iter().map(|p| p.name).collect();
        bad(format!(
            "`{}` takes no param `{name}`; its params are: {}",
            def.name,
            if known.is_empty() { "none".to_string() } else { known.join(", ") }
        ))
    })?;
    match pd.kind {
        Kind::Int | Kind::Float => Ok(RawParam::Expr(value_expr(v, ctx, at)?)),
        Kind::Ints => match v {
            toml::Value::Array(items) => Ok(RawParam::List(items.iter().map(|x| value_expr(x, ctx, at)).collect::<Result<_, _>>()?)),
            _ => Err(bad(format!("`{name}` is a list, such as [\"n_embd\", \"n_tokens\"]"))),
        },
        Kind::Type => match v.as_str().and_then(type_by_name) {
            Some(t) => Ok(RawParam::Const(i64::from(t))),
            None => Err(bad(format!("`{name}` must be a ggml type name, such as \"f16\""))),
        },
        Kind::Enum(choices) => {
            let s = v.as_str().unwrap_or_default();
            choices.iter().find(|(n, _)| *n == s).map(|(_, val)| RawParam::Const(*val)).ok_or_else(|| {
                bad(format!("`{name}` must be one of: {}", choices.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")))
            })
        }
    }
}

fn op_spec(
    section: Section,
    index: usize,
    table: &toml::Table,
    defaults: &BTreeMap<&'static str, BTreeMap<String, RawParam>>,
    has_names: &HashSet<&str>,
) -> Result<OpSpec, LayoutError> {
    let at0 = format!("{section} op {index}");
    let bad = |at: &str, detail: String| LayoutError::Invalid { at: at.to_owned(), detail };
    let op_name = table.get("op").and_then(toml::Value::as_str).ok_or_else(|| bad(&at0, "missing `op`".into()))?;
    let def = ops::op(op_name).ok_or_else(|| bad(&at0, unknown_op(op_name)))?;
    let at = format!("{at0} ({op_name})");
    let layer = section == Section::Block;

    let repeat = match table.get("for") {
        None => None,
        Some(v) => {
            let s = v.as_str().ok_or_else(|| bad(&at, "`for` is a string such as \"e in 1..n_expert_used\"".into()))?;
            Some(parse_loop(s, layer, &at)?)
        }
    };
    let var = repeat.as_ref().map(|r| r.var.as_str());
    let placeholders: Vec<&str> = var.into_iter().collect();

    let inputs = match table.get("in") {
        None => Vec::new(),
        Some(toml::Value::Array(items)) => items
            .iter()
            .map(|v| {
                let s = v.as_str().ok_or_else(|| bad(&at, "`in` lists tensor names".into()))?;
                let (name, optional) = match s.strip_suffix('?') {
                    Some(n) => (n, true),
                    None => (s, false),
                };
                let layered = check_template(name, &placeholders, &at)?;
                if layered && !layer {
                    return Err(bad(&at, format!("`{name}` uses {{L}} outside the block")));
                }
                Ok(InputRef { name: name.to_owned(), optional })
            })
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => return Err(bad(&at, "`in` is a list of tensor names".into())),
    };
    let required = inputs.iter().filter(|i| !i.optional).count();
    if inputs.len() > def.max_inputs || required < def.min_inputs {
        let range = if def.min_inputs == def.max_inputs { def.min_inputs.to_string() } else { format!("{} to {}", def.min_inputs, def.max_inputs) };
        return Err(bad(&at, format!("`{op_name}` takes {range} inputs, got {}", inputs.len())));
    }
    if let Some(first_opt) = inputs.iter().position(|i| i.optional) {
        if inputs[first_opt..].iter().any(|i| !i.optional) {
            return Err(bad(&at, "optional inputs must come last, so the rest keep their ggml argument positions".into()));
        }
    }

    let out = table.get("out").and_then(toml::Value::as_str).ok_or_else(|| bad(&at, "missing `out`".into()))?.to_owned();
    let layered = check_template(&out, &placeholders, &at)?;
    if layered && !layer {
        return Err(bad(&at, format!("`{out}` uses {{L}} outside the block")));
    }
    if layered {
        return Err(bad(&at, format!("`{out}` names an activation with {{L}}; bind adds the layer to every block activation's name already")));
    }

    let when_ctx = Ctx { layer, var, src: false };
    let when = match table.get("when") {
        None => None,
        Some(v) => {
            let s = v.as_str().ok_or_else(|| bad(&at, "`when` is a string condition".into()))?;
            let e = cond_expr(s, &when_ctx, &at)?;
            let mut has = Vec::new();
            e.has_names(&mut has);
            for h in has {
                check_template(h, &placeholders, &at)?;
                if !has_names.contains(h) {
                    return Err(bad(&at, format!("has({h}) names a tensor the weights table does not declare")));
                }
            }
            Some(e)
        }
    };

    let param_ctx = Ctx { layer, var, src: !inputs.is_empty() };
    let dims = match table.get("dims") {
        None => None,
        Some(toml::Value::Array(items)) => Some(items.iter().map(|v| dim_expr(v, &param_ctx, &at)).collect::<Result<Vec<_>, _>>()?),
        Some(_) => return Err(bad(&at, "`dims` is a list".into())),
    };
    let dtype = match table.get("dtype") {
        None => None,
        Some(v) => Some(v.as_str().and_then(type_by_name).ok_or_else(|| bad(&at, "`dtype` must be a ggml type name".into()))?),
    };

    let mut params = defaults.get(def.name).cloned().unwrap_or_default();
    for (k, v) in table {
        if OP_KEYS.contains(&k.as_str()) {
            continue;
        }
        params.insert(k.clone(), param(def, k, v, &param_ctx, &at)?);
    }
    for pd in def.params {
        if matches!(pd.default, Init::Required) && !params.contains_key(pd.name) {
            return Err(bad(&at, format!("`{op_name}` needs `{}`, in the op or in [defaults.{op_name}]", pd.name)));
        }
    }

    Ok(OpSpec { section, index, def, inputs, out, when, repeat, dims, dtype, params })
}

fn parse_loop(s: &str, layer: bool, at: &str) -> Result<Loop, LayoutError> {
    let bad = |detail: String| LayoutError::Invalid { at: at.to_owned(), detail };
    let (var, range) = s.split_once(" in ").ok_or_else(|| bad(format!("`for = \"{s}\"` should read like \"e in 1..n_expert_used\"")))?;
    let var = var.trim();
    if var.is_empty() || !var.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') || var == "L" || symbols::is_symbol(var) {
        return Err(bad(format!("`{var}` cannot be a loop variable")));
    }
    let (start, end) = range.split_once("..").ok_or_else(|| bad(format!("`{range}` should be a range such as 1..n_expert_used")))?;
    let ctx = Ctx { layer, var: None, src: false };
    Ok(Loop { var: var.to_owned(), start: parse_expr(start, &ctx, at)?, end: parse_expr(end, &ctx, at)? })
}

/// Check a tensor name's `{...}` placeholders: `{L}` or one of `vars`. Returns whether
/// it uses `{L}`.
fn check_template(name: &str, vars: &[&str], at: &str) -> Result<bool, LayoutError> {
    let bad = |detail: String| LayoutError::Invalid { at: at.to_owned(), detail };
    if name.is_empty() {
        return Err(bad("an empty tensor name".into()));
    }
    let mut layered = false;
    let mut rest = name;
    while let Some(open) = rest.find('{') {
        let close = rest[open..].find('}').ok_or_else(|| bad(format!("unclosed {{ in `{name}`")))? + open;
        let inner = &rest[open + 1..close];
        if inner == "L" {
            layered = true;
        } else if !vars.contains(&inner) {
            return Err(bad(format!("`{{{inner}}}` in `{name}` is neither {{L}} nor the op's loop variable")));
        }
        rest = &rest[close + 1..];
    }
    if rest.contains('}') {
        return Err(bad(format!("stray }} in `{name}`")));
    }
    Ok(layered)
}

fn unknown_op(name: &str) -> String {
    format!(
        "`{name}` is not an op dovata knows. Ops are ggml graph node types in lower case (GGML_OP_MUL_MAT is mul_mat; \
         unary and glu ops by their sub-op, such as silu and swiglu). Known: {}. If `{name}` is a ggml op, add it with \
         its shape rule to src/layout/ops.rs",
        ops::names().collect::<Vec<_>>().join(", ")
    )
}

/// A `ggml_type` by name, in any case: `f32`, `F16`, `q8_0`, `i32`.
pub(crate) fn type_by_name(name: &str) -> Option<u32> {
    (0..64u32).find(|t| crate::gguf::ggml_type_layout(*t).is_some() && crate::gguf::ggml_type_name(*t).eq_ignore_ascii_case(name))
}
