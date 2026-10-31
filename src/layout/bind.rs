//! Binding: applying a layout to one model and one set of options, producing the
//! [`BoundGraph`] the planner reads.
//!
//! ## What a bound graph is
//!
//! It is one forward step over `n_tokens` tokens, written out in full: every op of every
//! layer in execution order, and every tensor those ops touch, each with a concrete
//! shape, `ggml_type` and byte size. Nothing in it refers back to the layout or the
//! model's metadata. The planner places its tensors; the compiler generates a kernel per
//! op.
//!
//! ## What binding checks
//!
//! - The layout is for this model's architecture.
//! - Every weight the layout requires is in the file, with the shape the layout states.
//! - Every tensor an op reads resolves: a weight, a declared input, KV or state tensor,
//!   or an activation an earlier op produced. Using one before it is produced is its own
//!   error, so a misordered layout is told apart from a misspelled one.
//! - Every op's inputs satisfy the shape rule ggml's own constructor asserts, and any
//!   `dims` or `dtype` the op states matches what the rule produces.
//! - Every tensor in the file is read by at least one op, or is listed in `unused` with
//!   a reason. A layout cannot drop a weight without saying so.
//!
//! Weight types are never part of the layout. A weight's `ggml_type` and size come from
//! the file, so a Q4_K_M file and a BF16 file of one model bind to the same layout and
//! differ only in those two fields.
//!
//! ## Names
//!
//! Variables in a layout are names an op assigns with `out`, as in llama.cpp's
//! `cur = ggml_rms_norm(ctx0, cur, eps)`; assigning a name again gives a new tensor and
//! leaves the old one for the ops that already read it. A name first assigned in the
//! prologue lives for the whole step, which is how `inpL` carries from one layer to the
//! next. A name first assigned in the block lives for one layer only, so a block cannot
//! read the previous layer's value by accident.
//!
//! A block activation's bound name is the variable and the layer, `ffn_inp-3`, which is
//! how llama.cpp's `cb()` names graph tensors, so a bound graph can be compared with a
//! llama.cpp graph dump name for name. A second assignment of a name in the same layer
//! gets a `#1` suffix, then `#2`.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

use crate::load::{KvType, Model};

use super::expr::{Scope, Value};
use super::format::{DType, DeclKind, Layout, OpSpec, RawParam, Section};
use super::ops::{self, Init, Kind, TInfo};
use super::symbols::Hparams;
use super::LayoutError;

/// What binding needs besides the model: the sizes of one step and the KV cache types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindOptions {
    /// KV cache length in tokens. The KV tensors are sized for this many cells, and the
    /// attention reads are planned at this length too, since it is their maximum.
    pub n_ctx: u32,
    /// Tokens in one forward step: the batch the activations are sized for.
    pub n_tokens: u32,
    /// Tokens whose logits the step returns: 1 to sample the next token, `n_tokens` for
    /// logits at every position. llama.cpp's `n_outputs`; the last layer gathers these
    /// rows before the FFN, so it sizes everything from there to the logits.
    pub n_outputs: u32,
    /// The K cache's element type.
    pub kv_k: KvType,
    /// The V cache's element type, chosen separately from K.
    pub kv_v: KvType,
}

impl Serialize for BindOptions {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut st = s.serialize_struct("BindOptions", 5)?;
        st.serialize_field("n_ctx", &self.n_ctx)?;
        st.serialize_field("n_tokens", &self.n_tokens)?;
        st.serialize_field("n_outputs", &self.n_outputs)?;
        st.serialize_field("kv_k", self.kv_k.name())?;
        st.serialize_field("kv_v", self.kv_v.name())?;
        st.end()
    }
}

/// A tensor's index in [`BoundGraph::tensors`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct TensorId(pub u32);

impl TensorId {
    /// The index into [`BoundGraph::tensors`].
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// What a tensor is, which decides where the planner puts it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TensorKind {
    /// A tensor of the model file, addressed by where its bytes are in which part. The
    /// planner maps or copies it from there.
    Weight {
        /// Which part file holds it, from 0.
        part: u16,
        /// Where its data starts in that part's file.
        file_offset: u64,
    },
    /// Filled by the program before each step: token ids, positions, the KQ mask, the
    /// KV write indices, the output row ids. No op produces it; it counts as produced
    /// when the step starts.
    Input,
    /// Produced by one op and read by later ones within the step. The planner gives
    /// activations a shared arena and places each by its lifetime in op order.
    Activation,
    /// The KV cache. Persists across steps and gets its own region.
    Kv,
    /// Recurrent state, such as a delta-net's. Persists across steps like the KV cache.
    /// Reserved for the hybrid architectures; no built-in layout declares one yet.
    State,
    /// Another tensor's memory, seen through a different shape or strides. It has no
    /// memory of its own. `of` is always a tensor that owns memory, never another view,
    /// so the planner finds the address as `of`'s address plus `byte_offset`. A use of
    /// a view is a use of `of` for liveness.
    View {
        /// The tensor whose memory this is.
        of: TensorId,
        /// Where this view starts in `of`, in bytes.
        byte_offset: u64,
        /// Byte strides of all four dims, ggml's `nb`. A view may be non-contiguous.
        nb: [u64; 4],
    },
}

/// One tensor of the step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BoundTensor {
    /// The model's name for a weight; the layout's name for an input, KV or state
    /// tensor; and `<variable>-<layer>` for an activation produced in the block.
    pub name: String,
    pub kind: TensorKind,
    /// ggml's `ne`, fastest dimension first, with trailing 1s dropped (at least one).
    pub dims: Vec<u64>,
    /// The `ggml_type` tag. For a weight, the file's; for KV, the bind options'; for
    /// everything else, what the producing op's ggml rule gives, f32 unless it says
    /// otherwise.
    pub ggml_type: u32,
    /// Bytes of data, from [`crate::gguf::ggml_type_layout`]. For a view this is the
    /// span it covers in `of`, and it is not separate memory.
    pub bytes: u64,
    /// The layer an activation or KV tensor belongs to.
    pub layer: Option<u32>,
    /// Read by the program after the step, such as the logits. The planner keeps it
    /// alive to the end; the compiler copies it out. For a view, keep its `of` alive.
    pub output: bool,
}

/// A resolved op parameter. Floats are kept as f64 and ggml stores them as f32.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Param {
    Int(i64),
    Float(f64),
    Ints(Vec<i64>),
}

/// One ggml node of the step.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BoundOp {
    /// The layout op name: a ggml op in lower case. [`super::ops::ggml_enum`] gives the
    /// enum value it stands for.
    pub op: &'static str,
    /// In ggml API argument order. For `set_rows` that is (destination, rows, ids).
    pub inputs: Vec<TensorId>,
    /// The tensors the op produces. Every ggml op produces exactly one today.
    pub outputs: Vec<TensorId>,
    /// The layer, for a block op.
    pub layer: Option<u32>,
    /// Every parameter, with defaults filled in, as concrete values. Enum params, such
    /// as rope's `mode`, are ggml's integer constants. A view's `nb` and `offset` are
    /// in bytes, as ggml's view calls take them.
    pub params: BTreeMap<String, Param>,
}

/// One forward step of one model, ready to plan.
#[derive(Debug, Clone, Serialize)]
pub struct BoundGraph {
    /// `general.architecture`.
    pub arch: String,
    pub options: BindOptions,
    /// How many times the block ran.
    pub n_layer: u32,
    /// Every tensor. The model's tensors come first, in file order, so a weight's
    /// [`TensorId`] is its index in [`Model::tensors`]; then inputs, KV and state; then
    /// activations and views in the order their ops run.
    pub tensors: Vec<BoundTensor>,
    /// Every op, in execution order.
    pub ops: Vec<BoundOp>,
    /// The tensors the program reads after the step.
    pub outputs: Vec<TensorId>,
}

impl BoundGraph {
    /// One tensor.
    pub fn tensor(&self, id: TensorId) -> &BoundTensor {
        &self.tensors[id.index()]
    }

    /// A tensor by its bound name.
    pub fn find(&self, name: &str) -> Option<TensorId> {
        self.tensors.iter().position(|t| t.name == name).map(|i| TensorId(u32::try_from(i).unwrap_or(u32::MAX)))
    }

    /// The tensor that owns `id`'s memory: `id` itself, or the `of` of a view.
    pub fn storage(&self, id: TensorId) -> TensorId {
        match self.tensor(id).kind {
            TensorKind::View { of, .. } => of,
            _ => id,
        }
    }

    /// Ops whose layer is `layer`.
    pub fn layer_ops(&self, layer: u32) -> impl Iterator<Item = &BoundOp> {
        self.ops.iter().filter(move |o| o.layer == Some(layer))
    }
}

/// Whether a weight the layout declares is in the file.
#[derive(Debug, Clone, Copy)]
enum WeightState {
    /// In the file, or its fallback is.
    Present(TensorId),
    /// Optional, and the file lacks it.
    Absent,
}

struct Binder<'a> {
    layout: &'a Layout,
    model: &'a Model,
    hp: Hparams<'a>,
    tensors: Vec<BoundTensor>,
    /// ne and nb of each tensor, all four dims, for the shape rules.
    shapes: Vec<TInfo>,
    ops: Vec<BoundOp>,
    by_file: HashMap<String, TensorId>,
    weights: HashMap<String, WeightState>,
    /// Weight templates, to tell an undeclared-for-this-layer weight from a typo.
    weight_templates: HashSet<&'a str>,
    declared: HashMap<String, TensorId>,
    env: HashMap<String, TensorId>,
    used: Vec<bool>,
    names: HashMap<String, u32>,
}

/// The scope an expression is evaluated in.
struct Eval<'b, 'm> {
    hp: &'b Hparams<'m>,
    model: &'b Model,
    layer: Option<u32>,
    var: Option<(&'b str, i64)>,
    src: Option<TInfo>,
}

impl Scope for Eval<'_, '_> {
    fn name(&self, name: &str) -> Result<Value, String> {
        if name == "L" {
            return self.layer.map(|l| Value::Int(i64::from(l))).ok_or_else(|| "L outside the block".into());
        }
        if let Some((v, val)) = self.var {
            if v == name {
                return Ok(Value::Int(val));
            }
        }
        if let Some(field) = name.strip_prefix("src.") {
            let src = self.src.ok_or("src.* without an input")?;
            let (arr, i) = match field.split_at(2) {
                ("ne", i) => (src.ne, i),
                ("nb", i) => (src.nb, i),
                _ => return Err(format!("unknown `{name}`")),
            };
            let i: usize = i.parse().map_err(|_| format!("unknown `{name}`"))?;
            return Ok(Value::Int(i64::try_from(arr[i]).map_err(|_| "a size too large")?));
        }
        self.hp.value(name, self.layer)
    }

    fn has(&self, template: &str) -> Result<bool, String> {
        Ok(self.model.tensor(&expand(template, self.layer, self.var)).is_some())
    }

    fn row_size(&self, n: i64) -> Result<i64, String> {
        let src = self.src.ok_or("row_size without an input")?;
        let n = u64::try_from(n).map_err(|_| "row_size of a negative count")?;
        i64::try_from(ops::row_size(src.ty, n)?).map_err(|_| "a size too large".into())
    }
}

/// Substitute `{L}` and the loop variable into a tensor name. The parser has already
/// refused any other placeholder.
fn expand(template: &str, layer: Option<u32>, var: Option<(&str, i64)>) -> String {
    let mut out = template.to_owned();
    if let Some(l) = layer {
        out = out.replace("{L}", &l.to_string());
    }
    if let Some((name, val)) = var {
        out = out.replace(&format!("{{{name}}}"), &val.to_string());
    }
    out
}

/// ggml's `ne` with trailing 1s dropped, keeping at least one dim.
fn trim(ne: [u64; 4]) -> Vec<u64> {
    let mut n = 4;
    while n > 1 && ne[n - 1] == 1 {
        n -= 1;
    }
    ne[..n].to_vec()
}

fn pad(dims: &[u64]) -> [u64; 4] {
    let mut ne = [1u64; 4];
    for (i, d) in dims.iter().take(4).enumerate() {
        ne[i] = *d;
    }
    ne
}

impl Layout {
    /// Apply this layout to a model. See the module docs for what is checked.
    pub fn bind(&self, model: &Model, opts: &BindOptions) -> Result<BoundGraph, LayoutError> {
        if model.architecture() != self.arch {
            return Err(LayoutError::WrongArch { layout: self.arch.clone(), model: model.architecture().to_owned() });
        }
        if opts.n_tokens == 0 || opts.n_outputs == 0 || opts.n_outputs > opts.n_tokens || opts.n_ctx < opts.n_tokens {
            return Err(LayoutError::Options(format!(
                "n_ctx {}, n_tokens {}, n_outputs {}: a step needs 1 <= n_outputs <= n_tokens <= n_ctx",
                opts.n_ctx, opts.n_tokens, opts.n_outputs
            )));
        }
        let hp = Hparams::new(model, opts).map_err(|detail| LayoutError::Eval { at: "the model's hparams".into(), detail })?;
        let mut b = Binder {
            layout: self,
            model,
            hp,
            tensors: Vec::new(),
            shapes: Vec::new(),
            ops: Vec::new(),
            by_file: HashMap::new(),
            weights: HashMap::new(),
            weight_templates: self.weights.iter().map(|w| w.name.as_str()).collect(),
            declared: HashMap::new(),
            env: HashMap::new(),
            used: Vec::new(),
            names: HashMap::new(),
        };
        b.file_tensors()?;
        b.weight_decls()?;
        b.tensor_decls(opts)?;

        b.section(&self.prologue, None)?;
        let outer: HashSet<String> = b.env.keys().cloned().collect();
        for layer in 0..b.hp.n_layer() {
            b.env.retain(|k, _| outer.contains(k));
            b.section(&self.block, Some(layer))?;
        }
        b.env.retain(|k, _| outer.contains(k));
        b.section(&self.epilogue, None)?;

        let mut outputs = Vec::new();
        for name in &self.outputs {
            let id = *b.env.get(name).ok_or_else(|| LayoutError::UseBeforeProduce {
                name: name.clone(),
                at: "outputs".into(),
            })?;
            b.tensors[id.index()].output = true;
            outputs.push(id);
        }
        b.unreferenced()?;

        Ok(BoundGraph { arch: self.arch.clone(), options: *opts, n_layer: b.hp.n_layer(), tensors: b.tensors, ops: b.ops, outputs })
    }
}

impl<'a> Binder<'a> {
    fn eval(&self, layer: Option<u32>, var: Option<(&'a str, i64)>, src: Option<TInfo>) -> Eval<'_, 'a> {
        Eval { hp: &self.hp, model: self.model, layer, var, src }
    }

    fn push(&mut self, t: BoundTensor, shape: TInfo) -> TensorId {
        let id = TensorId(u32::try_from(self.tensors.len()).unwrap_or(u32::MAX));
        *self.names.entry(t.name.clone()).or_insert(0) += 1;
        self.tensors.push(t);
        self.shapes.push(shape);
        id
    }

    /// A name no tensor has yet: the base, then `base#1`, `base#2`.
    fn unique(&mut self, base: String) -> String {
        match self.names.get(&base) {
            None => base,
            Some(&n) => {
                let mut k = n;
                loop {
                    let candidate = format!("{base}#{k}");
                    if !self.names.contains_key(&candidate) {
                        *self.names.get_mut(&base).expect("present") += 1;
                        return candidate;
                    }
                    k += 1;
                }
            }
        }
    }

    /// Every tensor of the file, in file order, as a weight.
    fn file_tensors(&mut self) -> Result<(), LayoutError> {
        for t in self.model.tensors() {
            let info = t.info;
            let bytes = info.bytes().ok_or_else(|| LayoutError::UnsupportedType { name: info.name.clone(), ggml_type: info.kind })?;
            let ne = pad(&info.dims);
            let nb = ops::packed_nb(ne, info.kind).map_err(|_| LayoutError::UnsupportedType { name: info.name.clone(), ggml_type: info.kind })?;
            let id = self.push(
                BoundTensor {
                    name: info.name.clone(),
                    kind: TensorKind::Weight { part: t.part, file_offset: t.file_offset },
                    dims: trim(ne),
                    ggml_type: info.kind,
                    bytes,
                    layer: None,
                    output: false,
                },
                TInfo { ne, nb, ty: info.kind },
            );
            self.by_file.insert(info.name.clone(), id);
        }
        self.used = vec![false; self.tensors.len()];
        Ok(())
    }

    /// The layers a declaration applies to: every layer when its name has `{L}`.
    fn instances(&self, name: &str) -> Vec<Option<u32>> {
        if name.contains("{L}") { (0..self.hp.n_layer()).map(Some).collect() } else { vec![None] }
    }

    fn weight_decls(&mut self) -> Result<(), LayoutError> {
        let layout = self.layout;
        for w in &layout.weights {
            for layer in self.instances(&w.name) {
                let at = || format!("weights ({})", w.name);
                let scope = self.eval(layer, None, None);
                if let Some(cond) = &w.when {
                    if !cond.eval_bool(&scope).map_err(|detail| LayoutError::Eval { at: at(), detail })? {
                        continue;
                    }
                }
                let declared = match &w.dims {
                    Some(d) => Some(d.iter().map(|e| e.eval_u64(&scope)).collect::<Result<Vec<_>, _>>().map_err(|detail| LayoutError::Eval { at: at(), detail })?),
                    None => None,
                };
                let name = expand(&w.name, layer, None);
                let found = match self.by_file.get(&name) {
                    Some(id) => Some(*id),
                    None => match &w.fallback {
                        // TENSOR_DUPLICATED: the fallback tensor stands in for this one.
                        Some(f) => Some(*self.by_file.get(&expand(f, layer, None)).ok_or_else(|| LayoutError::MissingWeight { name: name.clone() })?),
                        None if w.optional => None,
                        None => return Err(LayoutError::MissingWeight { name }),
                    },
                };
                if let (Some(id), Some(want)) = (found, &declared) {
                    let have = &self.tensors[id.index()].dims;
                    if pad(want) != pad(have) {
                        return Err(LayoutError::WeightShape { name: self.tensors[id.index()].name.clone(), layout: want.clone(), file: have.clone() });
                    }
                }
                let state = found.map_or(WeightState::Absent, WeightState::Present);
                if self.weights.insert(name.clone(), state).is_some() {
                    return Err(LayoutError::Invalid { at: at(), detail: format!("{name} is declared twice") });
                }
            }
        }
        Ok(())
    }

    fn tensor_decls(&mut self, opts: &BindOptions) -> Result<(), LayoutError> {
        let layout = self.layout;
        for t in &layout.tensors {
            for layer in self.instances(&t.name) {
                let at = || format!("tensors ({})", t.name);
                let scope = self.eval(layer, None, None);
                if let Some(cond) = &t.when {
                    if !cond.eval_bool(&scope).map_err(|detail| LayoutError::Eval { at: at(), detail })? {
                        continue;
                    }
                }
                let dims = t.dims.iter().map(|e| e.eval_u64(&scope)).collect::<Result<Vec<_>, _>>().map_err(|detail| LayoutError::Eval { at: at(), detail })?;
                let ty = match t.dtype {
                    DType::Fixed(ty) => ty,
                    DType::KvK => opts.kv_k.ggml_type(),
                    DType::KvV => opts.kv_v.ggml_type(),
                };
                let ne = pad(&dims);
                let shape_err = |detail: String| LayoutError::Shape { at: at(), detail };
                let nb = ops::packed_nb(ne, ty).map_err(shape_err)?;
                let bytes = ops::nbytes(ne, nb, ty).map_err(shape_err)?;
                let kind = match t.kind {
                    DeclKind::Input => TensorKind::Input,
                    DeclKind::Kv => TensorKind::Kv,
                    DeclKind::State => TensorKind::State,
                };
                let name = expand(&t.name, layer, None);
                let id = self.push(BoundTensor { name: name.clone(), kind, dims: trim(ne), ggml_type: ty, bytes, layer, output: false }, TInfo { ne, nb, ty });
                self.declared.insert(name, id);
            }
        }
        Ok(())
    }

    fn section(&mut self, specs: &'a [OpSpec], layer: Option<u32>) -> Result<(), LayoutError> {
        for spec in specs {
            let site = |var: Option<(&str, i64)>| {
                let mut s = spec.site();
                if let Some(l) = layer {
                    s.push_str(&format!(", layer {l}"));
                }
                if let Some((v, val)) = var {
                    s.push_str(&format!(", {v} = {val}"));
                }
                s
            };
            let iterations: Vec<Option<(&'a str, i64)>> = match &spec.repeat {
                None => vec![None],
                Some(r) => {
                    let scope = self.eval(layer, None, None);
                    let err = |detail| LayoutError::Eval { at: site(None), detail };
                    let start = r.start.eval_int(&scope).map_err(err)?;
                    let end = r.end.eval_int(&scope).map_err(err)?;
                    (start..end).map(|v| Some((r.var.as_str(), v))).collect()
                }
            };
            for var in iterations {
                self.op(spec, layer, var, &site(var))?;
            }
        }
        Ok(())
    }

    /// Find the tensor an input name means here.
    fn resolve(&mut self, name: &str, template: &str, optional: bool, at: &str) -> Result<Option<TensorId>, LayoutError> {
        if let Some(id) = self.env.get(name) {
            return Ok(Some(*id));
        }
        if let Some(state) = self.weights.get(name) {
            return match *state {
                WeightState::Present(id) => {
                    self.used[id.index()] = true;
                    Ok(Some(id))
                }
                WeightState::Absent if optional => Ok(None),
                WeightState::Absent => Err(LayoutError::AbsentWeight { name: name.to_owned(), at: at.to_owned() }),
            };
        }
        if let Some(id) = self.declared.get(name) {
            return Ok(Some(*id));
        }
        if self.weight_templates.contains(template) {
            // Declared, but its `when` left it out of this layer.
            if optional {
                return Ok(None);
            }
            return Err(LayoutError::AbsentWeight { name: name.to_owned(), at: at.to_owned() });
        }
        if self.layout.out_names.contains(template) {
            return Err(LayoutError::UseBeforeProduce { name: name.to_owned(), at: at.to_owned() });
        }
        Err(LayoutError::UnknownTensor { name: name.to_owned(), at: at.to_owned() })
    }

    fn op(&mut self, spec: &'a OpSpec, layer: Option<u32>, var: Option<(&'a str, i64)>, at: &str) -> Result<(), LayoutError> {
        let eval_err = |detail: String| LayoutError::Eval { at: at.to_owned(), detail };
        if let Some(cond) = &spec.when {
            if !cond.eval_bool(&self.eval(layer, var, None)).map_err(eval_err)? {
                return Ok(());
            }
        }

        let mut inputs = Vec::with_capacity(spec.inputs.len());
        for input in &spec.inputs {
            let name = expand(&input.name, layer, var);
            if let Some(id) = self.resolve(&name, &input.name, input.optional, at)? {
                inputs.push(id);
            }
        }
        let infos: Vec<TInfo> = inputs.iter().map(|id| self.shapes[id.index()]).collect();
        let src = infos.first().copied();

        // Parameters: the op's own, then [defaults.<op>], then ggml's defaults.
        let scope = self.eval(layer, var, src);
        let mut params = BTreeMap::new();
        for pd in spec.def.params {
            let value = match spec.params.get(pd.name) {
                Some(RawParam::Const(c)) => Param::Int(*c),
                Some(RawParam::List(items)) => Param::Ints(items.iter().map(|e| e.eval_int(&scope)).collect::<Result<_, _>>().map_err(eval_err)?),
                Some(RawParam::Expr(e)) => match pd.kind {
                    Kind::Float => Param::Float(e.eval_f64(&scope).map_err(eval_err)?),
                    _ => Param::Int(e.eval_int(&scope).map_err(eval_err)?),
                },
                None => match pd.default {
                    Init::Required => return Err(eval_err(format!("missing `{}`", pd.name))),
                    Init::Absent => continue,
                    Init::Int(v) => Param::Int(v),
                    Init::Float(v) => Param::Float(v),
                    Init::Name(n) => match pd.kind {
                        Kind::Enum(choices) => Param::Int(choices.iter().find(|(c, _)| *c == n).map_or(0, |(_, v)| *v)),
                        _ => return Err(eval_err(format!("bad default for `{}`", pd.name))),
                    },
                },
            };
            params.insert(pd.name.to_owned(), value);
        }

        let out = (spec.def.shape)(&infos, &mut params).map_err(|detail| LayoutError::Shape { at: at.to_owned(), detail })?;

        if let Some(dims) = &spec.dims {
            let want = dims.iter().map(|e| e.eval_u64(&scope)).collect::<Result<Vec<_>, _>>().map_err(eval_err)?;
            if pad(&want) != out.ne {
                return Err(LayoutError::Shape { at: at.to_owned(), detail: format!("the layout says dims {want:?}, and ggml's rule gives {:?}", trim(out.ne)) });
            }
        }
        if let Some(ty) = spec.dtype {
            if ty != out.ty {
                return Err(LayoutError::Shape {
                    at: at.to_owned(),
                    detail: format!(
                        "the layout says dtype {}, and ggml's rule gives {}",
                        crate::gguf::ggml_type_name(ty),
                        crate::gguf::ggml_type_name(out.ty)
                    ),
                });
            }
        }

        let shape_err = |detail: String| LayoutError::Shape { at: at.to_owned(), detail };
        let (kind, nb, bytes) = match out.alias {
            Some(alias) => {
                let src_id = inputs[alias.input];
                let (root, base) = match self.tensors[src_id.index()].kind {
                    TensorKind::View { of, byte_offset, .. } => (of, byte_offset),
                    _ => (src_id, 0),
                };
                let byte_offset = base + alias.offset;
                let span = ops::nbytes(out.ne, alias.nb, out.ty).map_err(shape_err)?;
                let root_bytes = self.tensors[root.index()].bytes;
                if byte_offset + span > root_bytes {
                    return Err(shape_err(format!(
                        "the view covers bytes {byte_offset}..{} of {}, which has {root_bytes}",
                        byte_offset + span,
                        self.tensors[root.index()].name
                    )));
                }
                (TensorKind::View { of: root, byte_offset, nb: alias.nb }, alias.nb, span)
            }
            None => {
                let nb = ops::packed_nb(out.ne, out.ty).map_err(shape_err)?;
                (TensorKind::Activation, nb, ops::nbytes(out.ne, nb, out.ty).map_err(shape_err)?)
            }
        };

        let var_name = expand(&spec.out, layer, var);
        let base = match (spec.section, layer) {
            (Section::Block, Some(l)) => format!("{var_name}-{l}"),
            _ => var_name.clone(),
        };
        let name = self.unique(base);
        let id = self.push(BoundTensor { name, kind, dims: trim(out.ne), ggml_type: out.ty, bytes, layer, output: false }, TInfo { ne: out.ne, nb, ty: out.ty });
        self.ops.push(BoundOp { op: spec.def.name, inputs, outputs: vec![id], layer, params });
        self.env.insert(var_name, id);
        Ok(())
    }

    /// Every file tensor must be read by an op or listed as unused.
    fn unreferenced(&self) -> Result<(), LayoutError> {
        let mut unused: HashSet<String> = HashSet::new();
        for u in &self.layout.unused {
            for layer in self.instances(&u.name) {
                unused.insert(expand(&u.name, layer, None));
            }
        }
        let missing: Vec<String> = self
            .by_file
            .iter()
            .filter(|(name, id)| !self.used[id.index()] && !unused.contains(*name))
            .map(|(name, _)| name.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        if missing.is_empty() { Ok(()) } else { Err(LayoutError::UnreferencedTensors { names: missing }) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_expand_layer_and_loop_variable() {
        assert_eq!(expand("blk.{L}.attn_q.weight", Some(7), None), "blk.7.attn_q.weight");
        assert_eq!(expand("cur_experts.{e}", Some(1), Some(("e", 3))), "cur_experts.3");
        assert_eq!(expand("token_embd.weight", None, None), "token_embd.weight");
    }

    #[test]
    fn dims_drop_trailing_ones() {
        assert_eq!(trim([2048, 1, 1, 1]), vec![2048]);
        assert_eq!(trim([1, 1, 1, 1]), vec![1]);
        assert_eq!(trim([1, 8, 7, 1]), vec![1, 8, 7]);
    }
}
