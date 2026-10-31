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

/// Purpose: This function starts a header with the keys all three test architectures
/// read.
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
