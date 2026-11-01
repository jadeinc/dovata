//! Author: Joe Ricks
//! File: src/layout/testutil.rs
//! Description: Builds small GGUF headers in memory for the layout tests.

use std::path::PathBuf;

use crate::gguf::{ggml_type_layout, TensorInfo};
use crate::load::Model;

/// Purpose: This struct holds a test GGUF header while it is being built.
pub(crate) struct Header {
    arch: String,
    kvs: Vec<(String, u32, Vec<u8>)>,
    tensors: Vec<(String, Vec<u64>, u32)>,
}

impl Header {
    /// Purpose: This function starts a test header with an architecture and a tokenizer.
    ///
    /// Inputs:
    /// - `arch`: the architecture name.
    /// - `n_vocab`: the number of tokens in the tokenizer.
    ///
    /// Outputs: the header.
    pub(crate) fn new(arch: &str, n_vocab: u64) -> Header {
        let mut h = Header { arch: arch.to_owned(), kvs: Vec::new(), tensors: Vec::new() };
        h.str("general.architecture", arch);
        h.str("tokenizer.ggml.model", "gpt2");
        let mut toks = 8u32.to_le_bytes().to_vec();
        toks.extend_from_slice(&n_vocab.to_le_bytes());
        for _ in 0..n_vocab {
            toks.extend_from_slice(&1u64.to_le_bytes());
            toks.push(b't');
        }
        h.kvs.push(("tokenizer.ggml.tokens".into(), 9, toks));
        h
    }

    /// Purpose: This function adds a string key to the header.
    ///
    /// Inputs:
    /// - `key`: the full key.
    /// - `v`: the value.
    ///
    /// Outputs: the header, for chaining.
    pub(crate) fn str(&mut self, key: &str, v: &str) -> &mut Self {
        let mut p = (v.len() as u64).to_le_bytes().to_vec();
        p.extend_from_slice(v.as_bytes());
        self.kvs.push((key.to_owned(), 8, p));
        self
    }

    /// Purpose: This function adds an `<arch>.<suffix>` integer key to the header.
    ///
    /// Inputs:
    /// - `suffix`: the key after the architecture name.
    /// - `v`: the value.
    ///
    /// Outputs: the header, for chaining.
    pub(crate) fn u32(&mut self, suffix: &str, v: u32) -> &mut Self {
        let key = format!("{}.{suffix}", self.arch);
        self.kvs.push((key, 4, v.to_le_bytes().to_vec()));
        self
    }

    /// Purpose: This function adds an `<arch>.<suffix>` float key to the header.
    ///
    /// Inputs:
    /// - `suffix`: the key after the architecture name.
    /// - `v`: the value.
    ///
    /// Outputs: the header, for chaining.
    pub(crate) fn f32(&mut self, suffix: &str, v: f32) -> &mut Self {
        let key = format!("{}.{suffix}", self.arch);
        self.kvs.push((key, 6, v.to_le_bytes().to_vec()));
        self
    }

    /// Purpose: This function adds a tensor to the header's tensor index.
    ///
    /// Inputs:
    /// - `name`: the tensor's name.
    /// - `dims`: its dims.
    /// - `ty`: its ggml type number.
    ///
    /// Outputs: the header, for chaining.
    pub(crate) fn tensor(&mut self, name: &str, dims: &[u64], ty: u32) -> &mut Self {
        self.tensors.push((name.to_owned(), dims.to_vec(), ty));
        self
    }

    /// Purpose: This function writes the header as GGUF bytes.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the bytes.
    pub(crate) fn bytes(&self) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());
        for (k, tag, payload) in &self.kvs {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(payload);
        }
        let mut offset = 0u64;
        for (name, dims, ty) in &self.tensors {
            assert!(ggml_type_layout(*ty).is_some(), "test tensor {name} has an unknown type");
            b.extend_from_slice(&(name.len() as u64).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&offset.to_le_bytes());
            let info = TensorInfo { name: name.clone(), dims: dims.clone(), kind: *ty, offset };
            let size = info.bytes().unwrap_or_else(|| panic!("test tensor {name} does not divide into blocks"));
            offset += size.div_ceil(32) * 32;
        }
        b
    }

    /// Purpose: This function opens the header as a model.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the model.
    pub(crate) fn model(&self) -> Model {
        Model::from_headers(&[(PathBuf::from("synthetic.gguf"), self.bytes())]).expect("the synthetic header preflights")
    }
}

/// Purpose: This struct holds the sizes of a small test model.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Shape {
    pub n_layer: u32,
    pub n_embd: u64,
    pub n_head: u64,
    pub n_head_kv: u64,
    pub head_dim: u64,
    pub n_ff: u64,
    pub n_vocab: u64,
    pub n_expert: u64,
    pub n_expert_used: u64,
    pub n_ff_exp: u64,
}

pub(crate) const SMALL: Shape =
    Shape { n_layer: 2, n_embd: 64, n_head: 2, n_head_kv: 1, head_dim: 64, n_ff: 96, n_vocab: 32, n_expert: 4, n_expert_used: 2, n_ff_exp: 32 };

const F32: u32 = 0;

/// Purpose: This function starts a header with the keys every test architecture reads.
///
/// Inputs:
/// - `arch`: the architecture name.
/// - `s`: the model shape.
///
/// Outputs: the header.
fn common(arch: &str, s: &Shape) -> Header {
    let mut h = Header::new(arch, s.n_vocab);
    h.u32("block_count", s.n_layer)
        .u32("context_length", 4096)
        .u32("embedding_length", u32::try_from(s.n_embd).unwrap())
        .u32("attention.head_count", u32::try_from(s.n_head).unwrap())
        .u32("attention.head_count_kv", u32::try_from(s.n_head_kv).unwrap())
        .u32("attention.key_length", u32::try_from(s.head_dim).unwrap())
        .u32("attention.value_length", u32::try_from(s.head_dim).unwrap())
        .u32("feed_forward_length", u32::try_from(s.n_ff).unwrap())
        .f32("attention.layer_norm_rms_epsilon", 1e-6)
        .f32("rope.freq_base", 1_000_000.0);
    h
}

/// Purpose: This function adds the token embeddings, the output norm and the output
/// weight.
///
/// Inputs:
/// - `h`: the header to add to.
/// - `s`: the model shape.
/// - `ty`: the type of the large weights.
/// - `tied`: whether to leave out the output weight.
///
/// Outputs: none.
fn ends(h: &mut Header, s: &Shape, ty: u32, tied: bool) {
    h.tensor("token_embd.weight", &[s.n_embd, s.n_vocab], ty);
    h.tensor("output_norm.weight", &[s.n_embd], F32);
    if !tied {
        h.tensor("output.weight", &[s.n_embd, s.n_vocab], ty);
    }
}

/// Purpose: This function adds one layer's attention weights.
///
/// Inputs:
/// - `h`: the header to add to.
/// - `s`: the model shape.
/// - `l`: the layer index.
/// - `ty`: the type of the large weights.
///
/// Outputs: none.
fn attention(h: &mut Header, s: &Shape, l: u32, ty: u32) {
    let q = s.head_dim * s.n_head;
    let kv = s.head_dim * s.n_head_kv;
    h.tensor(&format!("blk.{l}.attn_norm.weight"), &[s.n_embd], F32);
    h.tensor(&format!("blk.{l}.attn_q.weight"), &[s.n_embd, q], ty);
    h.tensor(&format!("blk.{l}.attn_k.weight"), &[s.n_embd, kv], ty);
    h.tensor(&format!("blk.{l}.attn_v.weight"), &[s.n_embd, kv], ty);
    h.tensor(&format!("blk.{l}.attn_output.weight"), &[q, s.n_embd], ty);
}

/// Purpose: This function builds a small qwen3moe test header.
///
/// Inputs:
/// - `s`: the model shape.
/// - `ty`: the type of the large weights.
/// - `tied`: whether to leave out the output weight.
///
/// Outputs: the header.
pub(crate) fn qwen3moe(s: &Shape, ty: u32, tied: bool) -> Header {
    let mut h = common("qwen3moe", s);
    h.u32("expert_count", u32::try_from(s.n_expert).unwrap())
        .u32("expert_used_count", u32::try_from(s.n_expert_used).unwrap())
        .u32("expert_feed_forward_length", u32::try_from(s.n_ff_exp).unwrap());
    ends(&mut h, s, ty, tied);
    for l in 0..s.n_layer {
        attention(&mut h, s, l, ty);
        h.tensor(&format!("blk.{l}.attn_k_norm.weight"), &[s.head_dim], F32);
        h.tensor(&format!("blk.{l}.attn_q_norm.weight"), &[s.head_dim], F32);
        h.tensor(&format!("blk.{l}.ffn_norm.weight"), &[s.n_embd], F32);
        h.tensor(&format!("blk.{l}.ffn_gate_inp.weight"), &[s.n_embd, s.n_expert], F32);
        h.tensor(&format!("blk.{l}.ffn_gate_exps.weight"), &[s.n_embd, s.n_ff_exp, s.n_expert], ty);
        h.tensor(&format!("blk.{l}.ffn_down_exps.weight"), &[s.n_ff_exp, s.n_embd, s.n_expert], ty);
        h.tensor(&format!("blk.{l}.ffn_up_exps.weight"), &[s.n_embd, s.n_ff_exp, s.n_expert], ty);
    }
    h
}

/// Purpose: This function builds a small qwen3 test header.
///
/// Inputs:
/// - `s`: the model shape.
/// - `ty`: the type of the large weights.
/// - `tied`: whether to leave out the output weight.
///
/// Outputs: the header.
pub(crate) fn qwen3(s: &Shape, ty: u32, tied: bool) -> Header {
    let mut h = common("qwen3", s);
    ends(&mut h, s, ty, tied);
    for l in 0..s.n_layer {
        attention(&mut h, s, l, ty);
        h.tensor(&format!("blk.{l}.attn_k_norm.weight"), &[s.head_dim], F32);
        h.tensor(&format!("blk.{l}.attn_q_norm.weight"), &[s.head_dim], F32);
        h.tensor(&format!("blk.{l}.ffn_norm.weight"), &[s.n_embd], F32);
        h.tensor(&format!("blk.{l}.ffn_gate.weight"), &[s.n_embd, s.n_ff], ty);
        h.tensor(&format!("blk.{l}.ffn_down.weight"), &[s.n_ff, s.n_embd], ty);
        h.tensor(&format!("blk.{l}.ffn_up.weight"), &[s.n_embd, s.n_ff], ty);
    }
    h
}

/// Purpose: This function builds a small llama test header, dense or with experts.
///
/// Inputs:
/// - `s`: the model shape.
/// - `ty`: the type of the large weights.
/// - `tied`: whether to leave out the output weight.
/// - `moe`: whether to add experts.
///
/// Outputs: the header.
pub(crate) fn llama(s: &Shape, ty: u32, tied: bool, moe: bool) -> Header {
    let mut h = common("llama", s);
    if moe {
        // Preflight's report requires all three MoE keys once expert_count is set.
        h.u32("expert_count", u32::try_from(s.n_expert).unwrap())
            .u32("expert_used_count", u32::try_from(s.n_expert_used).unwrap())
            .u32("expert_feed_forward_length", u32::try_from(s.n_ff).unwrap());
    }
    ends(&mut h, s, ty, tied);
    for l in 0..s.n_layer {
        attention(&mut h, s, l, ty);
        h.tensor(&format!("blk.{l}.ffn_norm.weight"), &[s.n_embd], F32);
        if moe {
            h.tensor(&format!("blk.{l}.ffn_gate_inp.weight"), &[s.n_embd, s.n_expert], F32);
            h.tensor(&format!("blk.{l}.ffn_gate_exps.weight"), &[s.n_embd, s.n_ff, s.n_expert], ty);
            h.tensor(&format!("blk.{l}.ffn_down_exps.weight"), &[s.n_ff, s.n_embd, s.n_expert], ty);
            h.tensor(&format!("blk.{l}.ffn_up_exps.weight"), &[s.n_embd, s.n_ff, s.n_expert], ty);
        } else {
            h.tensor(&format!("blk.{l}.ffn_gate.weight"), &[s.n_embd, s.n_ff], ty);
            h.tensor(&format!("blk.{l}.ffn_down.weight"), &[s.n_ff, s.n_embd], ty);
            h.tensor(&format!("blk.{l}.ffn_up.weight"), &[s.n_embd, s.n_ff], ty);
        }
    }
    h
}

impl Header {
    /// Purpose: This function adds an `<arch>.<suffix>` key holding an int32 array to the
    /// header.
    ///
    /// Inputs:
    /// - `suffix`: the key after the architecture name.
    /// - `v`: the values, one per layer.
    ///
    /// Outputs: the header, for chaining.
    pub(crate) fn i32s(&mut self, suffix: &str, v: &[i32]) -> &mut Self {
        // gguf-py writes a per-layer list of Python ints, such as deci's head counts, as
        // an INT32 array.
        let key = format!("{}.{suffix}", self.arch);
        let mut p = 5u32.to_le_bytes().to_vec();
        p.extend_from_slice(&(v.len() as u64).to_le_bytes());
        for x in v {
            p.extend_from_slice(&x.to_le_bytes());
        }
        self.kvs.push((key, 9, p));
        self
    }
}

/// Purpose: This function builds a small qwen2 test header.
///
/// Inputs:
/// - `s`: the model shape, whose heads must multiply out to `n_embd`.
/// - `ty`: the type of the large weights.
/// - `tied`: whether to leave out the output weight.
///
/// Outputs: the header.
pub(crate) fn qwen2(s: &Shape, ty: u32, tied: bool) -> Header {
    let mut h = common("qwen2", s);
    ends(&mut h, s, ty, tied);
    let kv = s.head_dim * s.n_head_kv;
    for l in 0..s.n_layer {
        // create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0), with
        // the Q, K and V biases Qwen2 files carry.
        h.tensor(&format!("blk.{l}.attn_norm.weight"), &[s.n_embd], F32);
        h.tensor(&format!("blk.{l}.attn_q.weight"), &[s.n_embd, s.n_embd], ty);
        h.tensor(&format!("blk.{l}.attn_k.weight"), &[s.n_embd, kv], ty);
        h.tensor(&format!("blk.{l}.attn_v.weight"), &[s.n_embd, kv], ty);
        h.tensor(&format!("blk.{l}.attn_q.bias"), &[s.n_embd], F32);
        h.tensor(&format!("blk.{l}.attn_k.bias"), &[kv], F32);
        h.tensor(&format!("blk.{l}.attn_v.bias"), &[kv], F32);
        h.tensor(&format!("blk.{l}.attn_output.weight"), &[s.n_embd, s.n_embd], ty);
        h.tensor(&format!("blk.{l}.ffn_norm.weight"), &[s.n_embd], F32);
        h.tensor(&format!("blk.{l}.ffn_gate.weight"), &[s.n_embd, s.n_ff], ty);
        h.tensor(&format!("blk.{l}.ffn_down.weight"), &[s.n_ff, s.n_embd], ty);
        h.tensor(&format!("blk.{l}.ffn_up.weight"), &[s.n_embd, s.n_ff], ty);
    }
    h
}

/// Purpose: This function builds a small nemotron test header.
///
/// Inputs:
/// - `s`: the model shape, whose heads must multiply out to `n_embd`. Rope turns half of
///   each head, as in Nemotron-4.
/// - `ty`: the type of the large weights.
/// - `no_output`: whether to leave out the output weight, which nemotron requires.
///
/// Outputs: the header.
pub(crate) fn nemotron(s: &Shape, ty: u32, no_output: bool) -> Header {
    // Not `common`: a nemotron file has no RMS epsilon, so a layout that read one would
    // fail here as it would on a real file.
    let mut h = Header::new("nemotron", s.n_vocab);
    let n = |v: u64| u32::try_from(v).unwrap();
    h.u32("block_count", s.n_layer)
        .u32("context_length", 4096)
        .u32("embedding_length", n(s.n_embd))
        .u32("attention.head_count", n(s.n_head))
        .u32("attention.head_count_kv", n(s.n_head_kv))
        .u32("feed_forward_length", n(s.n_ff))
        .u32("rope.dimension_count", n(s.head_dim / 2))
        .f32("attention.layer_norm_epsilon", 1e-5)
        .f32("rope.freq_base", 10000.0);
    ends(&mut h, s, ty, no_output);
    h.tensor("output_norm.bias", &[s.n_embd], F32);
    let kv = s.head_dim * s.n_head_kv;
    for l in 0..s.n_layer {
        h.tensor(&format!("blk.{l}.attn_norm.weight"), &[s.n_embd], F32);
        h.tensor(&format!("blk.{l}.attn_norm.bias"), &[s.n_embd], F32);
        h.tensor(&format!("blk.{l}.attn_q.weight"), &[s.n_embd, s.n_embd], ty);
        h.tensor(&format!("blk.{l}.attn_k.weight"), &[s.n_embd, kv], ty);
        h.tensor(&format!("blk.{l}.attn_v.weight"), &[s.n_embd, kv], ty);
        h.tensor(&format!("blk.{l}.attn_output.weight"), &[s.n_embd, s.n_embd], ty);
        h.tensor(&format!("blk.{l}.ffn_norm.weight"), &[s.n_embd], F32);
        h.tensor(&format!("blk.{l}.ffn_norm.bias"), &[s.n_embd], F32);
        h.tensor(&format!("blk.{l}.ffn_down.weight"), &[s.n_ff, s.n_embd], ty);
        h.tensor(&format!("blk.{l}.ffn_up.weight"), &[s.n_embd, s.n_ff], ty);
    }
    h
}

/// Purpose: This function builds a small deci test header with a given kind of layer at
/// each index.
///
/// Inputs:
/// - `s`: the model shape; its `n_embd`, `head_dim` and `n_vocab` are used, and the
///   layer count and per-layer sizes come from `layers`.
/// - `ty`: the type of the large weights.
/// - `tied`: whether to leave out the output weight.
/// - `layers`: `(n_head, n_head_kv, n_ff)` for each layer.
///
/// Outputs: the header.
pub(crate) fn deci(s: &Shape, ty: u32, tied: bool, layers: &[(u64, u64, u64)]) -> Header {
    // conversion/deci.py writes the head counts and FFN widths as per-layer arrays.
    let mut h = Header::new("deci", s.n_vocab);
    let list = |pick: fn(&(u64, u64, u64)) -> u64| layers.iter().map(|l| i32::try_from(pick(l)).unwrap()).collect::<Vec<_>>();
    let n = |v: u64| u32::try_from(v).unwrap();
    h.u32("block_count", u32::try_from(layers.len()).unwrap())
        .u32("context_length", 4096)
        .u32("embedding_length", n(s.n_embd))
        .i32s("attention.head_count", &list(|l| l.0))
        .i32s("attention.head_count_kv", &list(|l| l.1))
        .i32s("feed_forward_length", &list(|l| l.2))
        .u32("attention.key_length", n(s.head_dim))
        .u32("attention.value_length", n(s.head_dim))
        .u32("rope.dimension_count", n(s.head_dim))
        .f32("attention.layer_norm_rms_epsilon", 1e-5)
        .f32("rope.freq_base", 500_000.0);
    ends(&mut h, s, ty, tied);
    for (l, &(n_head, n_head_kv, n_ff)) in layers.iter().enumerate() {
        // llama_model_deci::load_arch_tensors, branch for branch.
        if n_head_kv == 0 && n_head > 0 {
            h.tensor(&format!("blk.{l}.attn_norm.weight"), &[s.n_embd], F32);
            h.tensor(&format!("blk.{l}.attn_output.weight"), &[s.n_embd, s.n_embd], ty);
        } else if n_head_kv > 0 {
            let (q, kv) = (s.head_dim * n_head, s.head_dim * n_head_kv);
            h.tensor(&format!("blk.{l}.attn_norm.weight"), &[s.n_embd], F32);
            h.tensor(&format!("blk.{l}.attn_q.weight"), &[s.n_embd, q], ty);
            h.tensor(&format!("blk.{l}.attn_k.weight"), &[s.n_embd, kv], ty);
            h.tensor(&format!("blk.{l}.attn_v.weight"), &[s.n_embd, kv], ty);
            h.tensor(&format!("blk.{l}.attn_output.weight"), &[q, s.n_embd], ty);
        }
        if n_ff > 0 {
            h.tensor(&format!("blk.{l}.ffn_norm.weight"), &[s.n_embd], F32);
            h.tensor(&format!("blk.{l}.ffn_gate.weight"), &[s.n_embd, n_ff], ty);
            h.tensor(&format!("blk.{l}.ffn_down.weight"), &[n_ff, s.n_embd], ty);
            h.tensor(&format!("blk.{l}.ffn_up.weight"), &[s.n_embd, n_ff], ty);
        }
    }
    h
}

/// Purpose: This function builds a small llama4 test header, with experts on every
/// `moe_step`-th layer and a dense FFN on the others.
///
/// Inputs:
/// - `s`: the model shape.
/// - `ty`: the type of the large weights.
/// - `moe_step`: the `interleave_moe_layer_step` to write: 1 for Scout, 2 for Maverick.
/// - `fused_qkv`: whether to write one fused QKV weight per layer instead of Q, K and V.
///
/// Outputs: the header.
pub(crate) fn llama4(s: &Shape, ty: u32, moe_step: u32, fused_qkv: bool) -> Header {
    let mut h = common("llama4", s);
    h.u32("expert_count", u32::try_from(s.n_expert).unwrap())
        .u32("expert_used_count", u32::try_from(s.n_expert_used).unwrap())
        .u32("expert_feed_forward_length", u32::try_from(s.n_ff_exp).unwrap())
        .u32("interleave_moe_layer_step", moe_step);
    ends(&mut h, s, ty, false);
    for l in 0..s.n_layer {
        if fused_qkv {
            let qkv = s.head_dim * (s.n_head + 2 * s.n_head_kv);
            h.tensor(&format!("blk.{l}.attn_norm.weight"), &[s.n_embd], F32);
            h.tensor(&format!("blk.{l}.attn_qkv.weight"), &[s.n_embd, qkv], ty);
            h.tensor(&format!("blk.{l}.attn_output.weight"), &[s.head_dim * s.n_head, s.n_embd], ty);
        } else {
            attention(&mut h, s, l, ty);
        }
        h.tensor(&format!("blk.{l}.ffn_norm.weight"), &[s.n_embd], F32);
        // load_arch_tensors' is_moe_layer.
        if moe_step > 0 && (l + 1).is_multiple_of(moe_step) {
            h.tensor(&format!("blk.{l}.ffn_gate_inp.weight"), &[s.n_embd, s.n_expert], F32);
            h.tensor(&format!("blk.{l}.ffn_gate_exps.weight"), &[s.n_embd, s.n_ff_exp, s.n_expert], ty);
            h.tensor(&format!("blk.{l}.ffn_down_exps.weight"), &[s.n_ff_exp, s.n_embd, s.n_expert], ty);
            h.tensor(&format!("blk.{l}.ffn_up_exps.weight"), &[s.n_embd, s.n_ff_exp, s.n_expert], ty);
            h.tensor(&format!("blk.{l}.ffn_gate_shexp.weight"), &[s.n_embd, s.n_ff_exp], ty);
            h.tensor(&format!("blk.{l}.ffn_down_shexp.weight"), &[s.n_ff_exp, s.n_embd], ty);
            h.tensor(&format!("blk.{l}.ffn_up_shexp.weight"), &[s.n_embd, s.n_ff_exp], ty);
        } else {
            h.tensor(&format!("blk.{l}.ffn_gate.weight"), &[s.n_embd, s.n_ff], ty);
            h.tensor(&format!("blk.{l}.ffn_down.weight"), &[s.n_ff, s.n_embd], ty);
            h.tensor(&format!("blk.{l}.ffn_up.weight"), &[s.n_embd, s.n_ff], ty);
        }
    }
    h
}

// ---- deepseek2 ----

impl Header {
    /// Purpose: This function adds an `<arch>.<suffix>` bool key to the header.
    ///
    /// Inputs:
    /// - `suffix`: the key after the architecture name.
    /// - `v`: the value.
    ///
    /// Outputs: the header, for chaining.
    pub(crate) fn bool(&mut self, suffix: &str, v: bool) -> &mut Self {
        let key = format!("{}.{suffix}", self.arch);
        self.kvs.push((key, 7, vec![u8::from(v)]));
        self
    }
}

/// Purpose: This struct holds the deepseek2 variant a test header is built as.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Deepseek2 {
    /// Layers. 26 or 27 makes a variant llama.cpp calls lite, which reads no Q rank.
    pub n_layer: u32,
    /// The rank of the low-rank Q pair, or 0 for one full Q projection.
    pub q_lora_rank: u64,
    /// MLA's split `attn_k_b` and `attn_v_b`; false writes a legacy file's `attn_kv_b`.
    pub mla: bool,
    /// `expert_gating_func`, or `None` to leave the key out, as DeepSeek V2 files do.
    pub gating: Option<u32>,
    /// DeepSeek V3's expert selection bias, `exp_probs_b`.
    pub probs_bias: bool,
    /// `expert_weights_norm`.
    pub weights_norm: bool,
    /// `expert_weights_scale`.
    pub weights_scale: f32,
    /// `expert_group_count`; above 1, half the groups are used.
    pub n_expert_groups: u32,
    /// One fused `ffn_gate_up_exps` in place of `ffn_gate_exps` and `ffn_up_exps`.
    pub gate_up: bool,
    /// `attention.temperature_scale`, as Mistral Large 3 sets it; 0 leaves the key out.
    pub temp_scale: f32,
}

/// DeepSeek V3 in miniature: MLA with a low-rank Q, one dense layer then two MoE layers,
/// sigmoid gating with the selection bias, and normalized, scaled expert weights.
pub(crate) const DEEPSEEK_V3: Deepseek2 = Deepseek2 {
    n_layer: 3,
    q_lora_rank: 32,
    mla: true,
    gating: Some(2),
    probs_bias: true,
    weights_norm: true,
    weights_scale: 2.5,
    n_expert_groups: 0,
    gate_up: false,
    temp_scale: 0.0,
};

/// The sizes every deepseek2 test header shares. The MLA K cache row, DS2_KV_LORA +
/// DS2_ROT, is 128, a multiple of 64, so llama.cpp's rule for rotating a quantized cache
/// holds for it, and the K-only path must still build no rotation.
pub(crate) const DS2_EMBD: u64 = 64;
pub(crate) const DS2_HEAD: u64 = 2;
pub(crate) const DS2_KV_LORA: u64 = 96;
pub(crate) const DS2_ROT: u64 = 32;
pub(crate) const DS2_NOPE: u64 = 32;
pub(crate) const DS2_V: u64 = 32;
const DS2_FF: u64 = 96;
const DS2_FF_EXP: u64 = 32;
const DS2_EXPERT: u64 = 8;
const DS2_EXPERT_USED: u64 = 2;
const DS2_SHARED: u64 = 2;
const DS2_DENSE_LEAD: u32 = 1;
const DS2_VOCAB: u64 = 32;

/// Purpose: This function builds a small deepseek2 test header, following
/// llama_model_deepseek2's load_arch_hparams and load_arch_tensors.
///
/// Inputs:
/// - `v`: the variant.
/// - `ty`: the type of the large weights.
///
/// Outputs: the header.
pub(crate) fn deepseek2(v: &Deepseek2, ty: u32) -> Header {
    let n = |x: u64| u32::try_from(x).unwrap();
    let k_mla = DS2_NOPE + DS2_ROT;
    // The convert script writes an MLA file as MQA with one KV head: K is the compressed
    // KV and the rope part, V the compressed KV. A legacy file states the real heads.
    let (key, value, heads_kv) = if v.mla { (DS2_KV_LORA + DS2_ROT, DS2_KV_LORA, 1) } else { (k_mla, DS2_V, DS2_HEAD) };
    let mut h = Header::new("deepseek2", DS2_VOCAB);
    h.u32("block_count", v.n_layer)
        .u32("context_length", 163_840)
        .u32("embedding_length", n(DS2_EMBD))
        .u32("feed_forward_length", n(DS2_FF))
        .u32("attention.head_count", n(DS2_HEAD))
        .u32("attention.head_count_kv", n(heads_kv))
        .u32("attention.key_length", n(key))
        .u32("attention.value_length", n(value))
        .f32("attention.layer_norm_rms_epsilon", 1e-6)
        .u32("rope.dimension_count", n(DS2_ROT))
        .f32("rope.freq_base", 10000.0)
        .u32("leading_dense_block_count", DS2_DENSE_LEAD)
        .u32("attention.kv_lora_rank", n(DS2_KV_LORA))
        .u32("expert_count", n(DS2_EXPERT))
        .u32("expert_used_count", n(DS2_EXPERT_USED))
        .u32("expert_feed_forward_length", n(DS2_FF_EXP))
        .u32("expert_shared_count", n(DS2_SHARED))
        .f32("expert_weights_scale", v.weights_scale);
    // DeepSeek V3's YaRN: factor 40 from 4096 tokens, and the convert script's
    // yarn_log_multiplier of 0.1 times mscale_all_dim.
    h.str("deepseek2.rope.scaling.type", "yarn")
        .f32("rope.scaling.factor", 40.0)
        .u32("rope.scaling.original_context_length", 4096)
        .f32("rope.scaling.yarn_log_multiplier", 0.1);
    if v.q_lora_rank > 0 {
        h.u32("attention.q_lora_rank", n(v.q_lora_rank));
    }
    if v.mla {
        h.u32("attention.key_length_mla", n(k_mla)).u32("attention.value_length_mla", n(DS2_V));
    }
    if let Some(g) = v.gating {
        h.u32("expert_gating_func", g);
    }
    if v.weights_norm {
        h.bool("expert_weights_norm", true);
    }
    if v.n_expert_groups > 0 {
        h.u32("expert_group_count", v.n_expert_groups).u32("expert_group_used_count", v.n_expert_groups / 2);
    }
    if v.temp_scale != 0.0 {
        h.f32("attention.temperature_scale", v.temp_scale).u32("attention.temperature_length", 8192);
    }

    h.tensor("token_embd.weight", &[DS2_EMBD, DS2_VOCAB], ty);
    h.tensor("output_norm.weight", &[DS2_EMBD], F32);
    h.tensor("output.weight", &[DS2_EMBD, DS2_VOCAB], ty);
    for l in 0..v.n_layer {
        let t = |name: &str| format!("blk.{l}.{name}");
        h.tensor(&t("attn_norm.weight"), &[DS2_EMBD], F32);
        if v.q_lora_rank > 0 {
            h.tensor(&t("attn_q_a_norm.weight"), &[v.q_lora_rank], F32);
        }
        h.tensor(&t("attn_kv_a_norm.weight"), &[DS2_KV_LORA], F32);
        if v.q_lora_rank > 0 {
            h.tensor(&t("attn_q_a.weight"), &[DS2_EMBD, v.q_lora_rank], ty);
            h.tensor(&t("attn_q_b.weight"), &[v.q_lora_rank, DS2_HEAD * k_mla], ty);
        } else {
            h.tensor(&t("attn_q.weight"), &[DS2_EMBD, DS2_HEAD * k_mla], ty);
        }
        h.tensor(&t("attn_kv_a_mqa.weight"), &[DS2_EMBD, DS2_KV_LORA + DS2_ROT], ty);
        if v.mla {
            h.tensor(&t("attn_k_b.weight"), &[DS2_NOPE, DS2_KV_LORA, DS2_HEAD], ty);
            h.tensor(&t("attn_v_b.weight"), &[DS2_KV_LORA, DS2_V, DS2_HEAD], ty);
        } else {
            h.tensor(&t("attn_kv_b.weight"), &[DS2_KV_LORA, DS2_HEAD * (DS2_NOPE + DS2_V)], ty);
        }
        h.tensor(&t("attn_output.weight"), &[DS2_HEAD * DS2_V, DS2_EMBD], ty);
        h.tensor(&t("ffn_norm.weight"), &[DS2_EMBD], F32);
        if l < DS2_DENSE_LEAD {
            h.tensor(&t("ffn_gate.weight"), &[DS2_EMBD, DS2_FF], ty);
            h.tensor(&t("ffn_down.weight"), &[DS2_FF, DS2_EMBD], ty);
            h.tensor(&t("ffn_up.weight"), &[DS2_EMBD, DS2_FF], ty);
            continue;
        }
        h.tensor(&t("ffn_gate_inp.weight"), &[DS2_EMBD, DS2_EXPERT], F32);
        if v.probs_bias {
            h.tensor(&t("exp_probs_b.bias"), &[DS2_EXPERT], F32);
        }
        h.tensor(&t("ffn_down_exps.weight"), &[DS2_FF_EXP, DS2_EMBD, DS2_EXPERT], ty);
        if v.gate_up {
            h.tensor(&t("ffn_gate_up_exps.weight"), &[DS2_EMBD, DS2_FF_EXP * 2, DS2_EXPERT], ty);
        } else {
            h.tensor(&t("ffn_gate_exps.weight"), &[DS2_EMBD, DS2_FF_EXP, DS2_EXPERT], ty);
            h.tensor(&t("ffn_up_exps.weight"), &[DS2_EMBD, DS2_FF_EXP, DS2_EXPERT], ty);
        }
        h.tensor(&t("ffn_gate_shexp.weight"), &[DS2_EMBD, DS2_FF_EXP * DS2_SHARED], ty);
        h.tensor(&t("ffn_down_shexp.weight"), &[DS2_FF_EXP * DS2_SHARED, DS2_EMBD], ty);
        h.tensor(&t("ffn_up_shexp.weight"), &[DS2_EMBD, DS2_FF_EXP * DS2_SHARED], ty);
    }
    h
}
