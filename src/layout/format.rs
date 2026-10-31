//! Author: Joe Ricks
//! File: src/layout/format.rs
//! Description: Reads a layout file's TOML text into a checked `Layout`.

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use serde::Deserialize;

use super::expr::Expr;
use super::ops::{self, Init, Kind, OpDef};
use super::symbols;
use super::LayoutError;

/// The format version this dovata reads.
pub const SCHEMA_VERSION: u32 = 1;

/// Purpose: This struct holds a checked layout, read from a layout file.
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

/// Purpose: This struct holds one weight a layout expects in the model file.
#[derive(Debug, Clone)]
pub(crate) struct WeightDecl {
    pub name: String,
    pub dims: Option<Vec<Expr>>,
    pub optional: bool,
    pub fallback: Option<String>,
    pub when: Option<Expr>,
}

/// Purpose: This struct holds one model tensor a layout lists as unused.
#[derive(Debug, Clone)]
pub(crate) struct UnusedDecl {
    pub name: String,
}

/// Purpose: This enum lists the kinds of declared tensor: input, KV cache and state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeclKind {
    Input,
    Kv,
    State,
}

/// Purpose: This enum holds a declared tensor's ggml type or its KV cache type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DType {
    Fixed(u32),
    KvK,
    KvV,
}

/// Purpose: This struct holds one declared input, KV cache or state tensor.
#[derive(Debug, Clone)]
pub(crate) struct TensorDecl {
    pub name: String,
    pub kind: DeclKind,
    pub dims: Vec<Expr>,
    pub dtype: DType,
    pub when: Option<Expr>,
}

/// Purpose: This enum lists the three sections of a layout: prologue, block and epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Section {
    Prologue,
    Block,
    Epilogue,
}

impl fmt::Display for Section {
    /// Purpose: This function formats a layout section's name: prologue, block or
    /// epilogue.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Section::Prologue => "prologue",
            Section::Block => "block",
            Section::Epilogue => "epilogue",
        })
    }
}

/// Purpose: This struct holds one input name of an op, and whether the input is optional.
#[derive(Debug, Clone)]
pub(crate) struct InputRef {
    pub name: String,
    pub optional: bool,
}

/// Purpose: This struct holds an op's `for` loop: its variable and its range.
#[derive(Debug, Clone)]
pub(crate) struct Loop {
    pub var: String,
    pub start: Expr,
    pub end: Expr,
}

/// Purpose: This enum holds an op parameter as the layout file writes it.
#[derive(Debug, Clone)]
pub(crate) enum RawParam {
    Expr(Expr),
    List(Vec<Expr>),
    /// An enum or type name, already turned into its ggml value.
    Const(i64),
}

/// Purpose: This struct holds one checked op of a layout.
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
    /// Purpose: This function names where an op sits in its layout file, such as
    /// `block op 12 (rope)`.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the section, the op's index in it, and the op's name.
    pub(crate) fn site(&self) -> String {
        format!("{} op {} ({})", self.section, self.index, self.def.name)
    }
}

/// Purpose: This struct holds a layout file's TOML as parsed, before it is checked.
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

/// Purpose: This struct holds one entry of a layout file's `weights` list, as parsed.
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

/// Purpose: This struct holds one entry of a layout file's `unused` list, as parsed.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawUnused {
    name: String,
    /// Why llama.cpp loads it and the graph does not read it. Required, so the reason
    /// lives next to the exception.
    #[allow(dead_code)]
    why: String,
}

/// Purpose: This struct holds one entry of a layout file's `tensors` list, as parsed.
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
    /// Purpose: This function reads a layout file's TOML text and checks it.
    ///
    /// Inputs:
    /// - `src`: the text of a `.layout.toml` file.
    ///
    /// Outputs: the checked layout, or an error naming the first mistake and its place in
    /// the file.
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

    /// Purpose: This function returns the model architecture the layout is for.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the architecture name, such as `qwen3moe`.
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// Purpose: This function returns the layout's one-line description.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the description, or `None` when the layout has none.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

/// Purpose: This struct holds which names an expression may read where it is written.
struct Ctx<'a> {
    /// `L` is in scope: a block op, or a declaration whose name has `{L}`.
    layer: bool,
    /// The op's loop variable, if it has one.
    var: Option<&'a str>,
    /// `src.*` and `row_size` are in scope: an op param with at least one input.
    src: bool,
}

const SRC_NAMES: [&str; 8] = ["src.ne0", "src.ne1", "src.ne2", "src.ne3", "src.nb0", "src.nb1", "src.nb2", "src.nb3"];

/// Purpose: This function checks that an expression reads only names allowed where it is
/// written.
///
/// Inputs:
/// - `e`: the parsed expression.
/// - `ctx`: which names are allowed at this place.
/// - `at`: the place in the layout file, for the error.
///
/// Outputs: `Ok(())`, or an error naming the first name that is not allowed.
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

/// Purpose: This function parses a dim or op parameter expression and checks its names.
///
/// Inputs:
/// - `src`: the expression's text.
/// - `ctx`: which names are allowed at this place.
/// - `at`: the place in the layout file, for the error.
///
/// Outputs: the parsed expression, or an error.
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

/// Purpose: This function parses a `when` condition and checks its names.
///
/// Inputs:
/// - `src`: the condition's text.
/// - `ctx`: which names are allowed at this place.
/// - `at`: the place in the layout file, for the error.
///
/// Outputs: the parsed expression, or an error.
fn cond_expr(src: &str, ctx: &Ctx<'_>, at: &str) -> Result<Expr, LayoutError> {
    let e = Expr::parse(src).map_err(|detail| LayoutError::Invalid { at: at.to_owned(), detail })?;
    check_names(&e, ctx, at)?;
    Ok(e)
}

/// Purpose: This function reads a TOML number or expression string as an expression.
///
/// Inputs:
/// - `v`: the TOML value.
/// - `ctx`: which names are allowed at this place.
/// - `at`: the place in the layout file, for the error.
///
/// Outputs: the expression, or an error for any other kind of value.
fn value_expr(v: &toml::Value, ctx: &Ctx<'_>, at: &str) -> Result<Expr, LayoutError> {
    match v {
        toml::Value::Integer(i) => Ok(Expr::Int(*i)),
        toml::Value::Float(f) => Ok(Expr::Float(*f)),
        toml::Value::String(s) => parse_expr(s, ctx, at),
        other => Err(LayoutError::Invalid { at: at.to_owned(), detail: format!("expected a number or an expression, got {other}") }),
    }
}

/// Purpose: This function reads one entry of a `dims` list as an expression.
///
/// Inputs:
/// - `v`: the TOML value.
/// - `ctx`: which names are allowed at this place.
/// - `at`: the place in the layout file, for the error.
///
/// Outputs: the expression, or an error.
fn dim_expr(v: &toml::Value, ctx: &Ctx<'_>, at: &str) -> Result<Expr, LayoutError> {
    value_expr(v, ctx, at)
}

/// Purpose: This function reads one op parameter by the kind the op's definition gives
/// it.
///
/// Inputs:
/// - `def`: the op's definition.
/// - `name`: the parameter's name.
/// - `v`: the parameter's TOML value.
/// - `ctx`: which names are allowed at this place.
/// - `at`: the place in the layout file, for the error.
///
/// Outputs: the parameter, or an error when the op does not take it or the value is the
/// wrong kind.
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

/// Purpose: This function reads one op's TOML table and checks it against the op's
/// definition.
///
/// Inputs:
/// - `section`: the layout section the op is in.
/// - `index`: the op's position in that section.
/// - `table`: the op's TOML table.
/// - `defaults`: the layout's parameter defaults for each op.
/// - `has_names`: the weight names the layout declares.
///
/// Outputs: the checked op, or an error naming the first mistake.
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

/// Purpose: This function reads an op's `for` value, such as `e in 1..n_expert_used`.
///
/// Inputs:
/// - `s`: the text of the `for` value.
/// - `layer`: whether the layer index `L` is allowed.
/// - `at`: the place in the layout file, for the error.
///
/// Outputs: the loop, or an error when the text is not a valid loop.
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

/// Purpose: This function checks the `{...}` placeholders in a tensor name.
///
/// Inputs:
/// - `name`: the tensor name.
/// - `vars`: the loop variables allowed in the name.
/// - `at`: the place in the layout file, for the error.
///
/// Outputs: `true` when the name uses `{L}`, or an error for a bad placeholder.
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

/// Purpose: This function writes the error text for an unknown op name.
///
/// Inputs:
/// - `name`: the op name the layout used.
///
/// Outputs: the error text, which lists the known ops.
fn unknown_op(name: &str) -> String {
    format!(
        "`{name}` is not an op dovata knows. Ops are ggml graph node types in lower case (GGML_OP_MUL_MAT is mul_mat; \
         unary and glu ops by their sub-op, such as silu and swiglu). Known: {}. If `{name}` is a ggml op, add it with \
         its shape rule to src/layout/ops.rs",
        ops::names().collect::<Vec<_>>().join(", ")
    )
}

/// Purpose: This function looks up a ggml tensor type by name, in any case.
///
/// Inputs:
/// - `name`: the type's name, such as `f16`.
///
/// Outputs: the type's number, or `None` when dovata cannot size a type of that name.
pub(crate) fn type_by_name(name: &str) -> Option<u32> {
    (0..64u32).find(|t| crate::gguf::ggml_type_layout(*t).is_some() && crate::gguf::ggml_type_name(*t).eq_ignore_ascii_case(name))
}
