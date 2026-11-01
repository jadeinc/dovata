//! Author: Joe Ricks
//! File: src/layout/tests.rs
//! Description: Tests the layout capability: reading layouts, the built-in layouts, and
//! binding them to models.

use std::collections::{BTreeSet, HashSet};

use super::testutil::{self, Header, Shape, SMALL};
use super::*;
use crate::load::KvType;

const F32: u32 = 0;
const F16: u32 = 1;
const Q8_0: u32 = 8;

/// Purpose: This function builds small bind options for the tests.
///
/// Inputs:
/// - `kv`: the KV cache type.
///
/// Outputs: the options.
fn opts(kv: KvType) -> BindOptions {
    BindOptions { n_ctx: 256, n_tokens: 8, n_outputs: 1, kv_k: kv, kv_v: kv }
}

/// Purpose: This function binds a built-in layout to a test header.
///
/// Inputs:
/// - `arch`: the architecture name.
/// - `h`: the test header.
/// - `kv`: the KV cache type.
///
/// Outputs: the bound graph. It panics on an error.
fn bind(arch: &str, h: &Header, kv: KvType) -> BoundGraph {
    let model = h.model();
    builtin(arch).unwrap().bind(&model, &opts(kv)).unwrap_or_else(|e| panic!("{arch} binds: {e}"))
}

/// Purpose: This function lists a bound graph's ops of one kind.
///
/// Inputs:
/// - `g`: the bound graph.
/// - `op`: the op name.
///
/// Outputs: the matching ops.
fn ops_named<'a>(g: &'a BoundGraph, op: &'a str) -> impl Iterator<Item = &'a BoundOp> + 'a {
    g.ops.iter().filter(move |o| o.op == op)
}

/// Purpose: This function checks the facts every bound graph must have, and panics when
/// one fails.
///
/// Inputs:
/// - `g`: the bound graph.
///
/// Outputs: none.
fn check_invariants(g: &BoundGraph) {
    let names: HashSet<&str> = g.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names.len(), g.tensors.len(), "tensor names are unique");
    let mut produced = vec![false; g.tensors.len()];
    for (i, t) in g.tensors.iter().enumerate() {
        if matches!(t.kind, TensorKind::Weight { .. } | TensorKind::Input | TensorKind::Kv | TensorKind::State) {
            produced[i] = true;
        }
        if let TensorKind::View { of, byte_offset, .. } = t.kind {
            let root = g.tensor(of);
            assert!(!matches!(root.kind, TensorKind::View { .. }), "{} views a view", t.name);
            assert!(byte_offset + t.bytes <= root.bytes, "{} runs past {}", t.name, root.name);
        }
    }
    for op in &g.ops {
        for i in &op.inputs {
            assert!(produced[i.index()], "{} reads {} before it exists", op.op, g.tensor(*i).name);
        }
        assert_eq!(op.outputs.len(), 1);
        for o in &op.outputs {
            assert!(!produced[o.index()], "{} is produced twice", g.tensor(*o).name);
            produced[o.index()] = true;
        }
    }
    for t in &g.tensors {
        assert!(t.bytes > 0 || t.dims.contains(&0), "{} has no bytes", t.name);
    }
}

#[test]
fn every_op_name_exists_in_ggml() {
    // The enums, not the API functions: a layout op is one graph node type, and ggml's
    // node types are the `ggml_op` values, with unary and glu nodes named by their
    // sub-op. See the docs in ops.rs for why function names would not do. The names
    // come from ggml-ops.txt, a copy of the three enums in ggml.h.
    let list = include_str!("ggml-ops.txt");
    let enum_values = |name: &str| -> BTreeSet<String> {
        let header = format!("[{name}]");
        list.lines()
            .map(str::trim)
            .skip_while(|l| *l != header)
            .skip(1)
            .take_while(|l| !l.starts_with('['))
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(str::to_owned)
            .collect()
    };
    let op = enum_values("ggml_op");
    let unary = enum_values("ggml_unary_op");
    let glu = enum_values("ggml_glu_op");
    assert!(op.contains("MUL_MAT") && op.contains("UNARY") && op.contains("GLU"), "the parse found the enum");
    assert!(unary.contains("SILU") && glu.contains("SWIGLU"));

    let mut seen = HashSet::new();
    for name in ops::names() {
        assert!(seen.insert(name), "{name} is in the table twice");
        let (set, value) = match ops::ggml_enum(name).unwrap() {
            ops::Ggml::Op(v) => (&op, v),
            ops::Ggml::Unary(v) => (&unary, v),
            ops::Ggml::Glu(v) => (&glu, v),
        };
        assert!(set.contains(value), "layout op {name} names {value}, which ggml does not have");
        assert_eq!(name.to_ascii_uppercase(), value, "a layout op is its ggml enum value in lower case");
    }
    // A unary or glu name must not also be a plain op, or one layout name would mean two nodes.
    for u in unary.iter().chain(&glu) {
        assert!(!op.contains(u), "{u} is both a ggml_op and a sub-op");
    }
}

#[test]
fn builtin_layouts_parse() {
    for arch in SUPPORTED {
        let layout = builtin(arch).unwrap_or_else(|| panic!("{arch} is in SUPPORTED but has no layout"));
        assert_eq!(layout.arch(), *arch);
    }
    assert!(builtin("mamba2").is_none());
    // Every file in builtin/ is listed, and nothing listed lacks a file.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/layout/builtin");
    let files: BTreeSet<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter_map(|f| f.strip_suffix(".layout.toml").map(str::to_owned))
        .collect();
    let listed: BTreeSet<String> = SUPPORTED.iter().map(|s| (*s).to_owned()).collect();
    assert_eq!(files, listed);
}

#[test]
fn builtin_qwen3moe_binds() {
    let g = bind("qwen3moe", &testutil::qwen3moe(&SMALL, F16, false), KvType::F16);
    check_invariants(&g);
    assert_eq!(g.n_layer, 2);
    // Every file tensor is a weight, in file order, and every one is read.
    let model = testutil::qwen3moe(&SMALL, F16, false).model();
    assert_eq!(model.tensors().count(), 3 + 2 * 12);
    for (i, t) in model.tensors().enumerate() {
        assert_eq!(g.tensors[i].name, t.info.name);
        assert_eq!(g.tensors[i].kind, TensorKind::Weight { part: 0, file_offset: t.file_offset });
    }
    // The logits: [n_vocab, n_outputs], f32, marked as the step's output. One output row is a trailing 1.
    let out = g.tensor(g.outputs[0]);
    assert_eq!((out.name.as_str(), out.dims.as_slice(), out.ggml_type, out.output), ("result_output", &[32u64][..], F32, true));
    // Routing: argsort, viewed to the top n_expert_used, then three expert matmuls.
    let topk = g.find("ffn_moe_topk-0").unwrap();
    assert_eq!(g.tensor(topk).dims, vec![2, 8]);
    assert_eq!(g.tensor(topk).ggml_type, 26, "expert ids are i32");
    assert_eq!(ops_named(&g, "mul_mat_id").count(), 3 * 2);
    // One view per expert and an add chain of n_expert_used - 1 adds, per layer.
    let layer0 = |op: &str| g.layer_ops(0).filter(|o| o.op == op).count();
    assert_eq!(layer0("mul_mat_id"), 3);
    assert!(g.find("cur_experts.1-0").is_some());
    assert!(g.find("cur_experts.2-0").is_none(), "only n_expert_used views");
    // The last layer gathers the output rows before the FFN.
    assert_eq!(g.tensor(g.find("ffn_moe_topk-1").unwrap()).dims, vec![2], "[n_expert_used, n_outputs = 1]");
}

#[test]
fn builtin_qwen3_binds() {
    let g = bind("qwen3", &testutil::qwen3(&SMALL, F16, false), KvType::F16);
    check_invariants(&g);
    assert_eq!(ops_named(&g, "mul_mat_id").count(), 0);
    assert_eq!(ops_named(&g, "swiglu").count(), 2);
    // Q and K are normed per head, then roped: two rms_norms beyond the layer's two.
    assert_eq!(g.layer_ops(0).filter(|o| o.op == "rms_norm").count(), 4);
    let rope = ops_named(&g, "rope").next().unwrap();
    assert_eq!(rope.params["mode"], Param::Int(2), "qwen3 ropes NEOX");
    assert_eq!(rope.params["freq_base"], Param::Float(1_000_000.0), "freq_base comes from the file");
    assert_eq!(rope.params["n_dims"], Param::Int(64), "n_rot falls back to the head width");
}

#[test]
fn builtin_llama_binds() {
    let g = bind("llama", &testutil::llama(&SMALL, F16, false, false), KvType::F16);
    check_invariants(&g);
    let rope = ops_named(&g, "rope").next().unwrap();
    assert_eq!(rope.params["mode"], Param::Int(0), "llama ropes NORM");
    assert_eq!(rope.inputs.len(), 2, "no rope_freqs in this file, so no third input");
    assert_eq!(ops_named(&g, "mul_mat_id").count(), 0);
    let fa = ops_named(&g, "flash_attn_ext").next().unwrap();
    assert_eq!(fa.params["scale"], Param::Float(1.0 / 8.0), "1/sqrt(64)");
    assert_eq!(fa.params["prec"], Param::Int(10), "GGML_PREC_F32");
}

#[test]
fn builtin_llama_moe_variant_binds() {
    // Mixtral loads as llama with experts: the same layout takes its expert path.
    let g = bind("llama", &testutil::llama(&SMALL, F16, false, true), KvType::F16);
    check_invariants(&g);
    assert_eq!(ops_named(&g, "mul_mat_id").count(), 3 * 2);
    assert_eq!(ops_named(&g, "swiglu").count(), 2);
    let down = g.find("ffn_moe_down-0").unwrap();
    assert_eq!(g.tensor(down).dims, vec![64, 2, 8], "[n_embd, n_expert_used, n_tokens]");
}

// ---- llama4 ----

/// Purpose: This function returns a small llama4 shape with four layers, so the default
/// period of 4 gives three chunked rope layers and one NoPE layer.
///
/// Inputs: none.
///
/// Outputs: the shape.
fn llama4_shape() -> Shape {
    Shape { n_layer: 4, ..SMALL }
}

/// Purpose: This function lists the layers whose ops read a tensor.
///
/// Inputs:
/// - `g`: the bound graph.
/// - `name`: the tensor's bound name.
///
/// Outputs: the layers, in order, each once; empty when the graph has no such tensor.
fn layers_reading(g: &BoundGraph, name: &str) -> Vec<u32> {
    let Some(id) = g.find(name) else { return Vec::new() };
    let mut layers: Vec<u32> = g.ops.iter().filter(|o| o.inputs.contains(&id)).filter_map(|o| o.layer).collect();
    layers.dedup();
    layers
}

/// Purpose: This function finds the op that produces a tensor.
///
/// Inputs:
/// - `g`: the bound graph.
/// - `name`: the tensor's bound name.
///
/// Outputs: the op. It panics when no op produces the tensor.
fn producer<'a>(g: &'a BoundGraph, name: &str) -> &'a BoundOp {
    let id = g.find(name).unwrap_or_else(|| panic!("no tensor {name}"));
    g.ops.iter().find(|o| o.outputs.contains(&id)).unwrap_or_else(|| panic!("no op produces {name}"))
}

#[test]
fn builtin_llama4_binds_scout_like() {
    // Scout: experts on every layer, the Q/K L2 norm, and rope frequency factors.
    let mut h = testutil::llama4(&llama4_shape(), F16, 1, false);
    h.tensor("rope_freqs.weight", &[32], F32);
    let g = bind("llama4", &h, KvType::F16);
    check_invariants(&g);
    // Layers 0 to 2 rope and attend in chunks; layer 3 has no rope, scales Q by the
    // temperature, and attends over the whole context.
    let ropes: Vec<usize> = (0..4).map(|l| g.layer_ops(l).filter(|o| o.op == "rope").count()).collect();
    assert_eq!(ropes, vec![2, 2, 2, 0]);
    assert_eq!(layers_reading(&g, "inp_attn_scale"), vec![3]);
    assert_eq!(layers_reading(&g, "attn_inp_kq_mask_swa"), vec![0, 1, 2]);
    assert_eq!(layers_reading(&g, "attn_inp_kq_mask"), vec![3]);
    assert_eq!(layers_reading(&g, "inp_k_idxs_swa"), vec![0, 1, 2]);
    assert_eq!(layers_reading(&g, "inp_v_idxs"), vec![3]);
    assert_eq!(layers_reading(&g, "rope_freqs.weight"), vec![0, 1, 2]);
    let rope = ops_named(&g, "rope").next().unwrap();
    assert_eq!(rope.params["mode"], Param::Int(0), "llama4 ropes NORM");
    assert_eq!(rope.params["freq_base"], Param::Float(1_000_000.0), "no freq_base_swa, so the file's freq_base");
    // The L2 norm is a weightless rms_norm on Q and K of each rope layer.
    let norms: Vec<usize> = (0..4).map(|l| g.layer_ops(l).filter(|o| o.op == "rms_norm").count()).collect();
    assert_eq!(norms, vec![4, 4, 4, 2]);
    // Sigmoid routing that chooses on the raw logits.
    assert_eq!(ops_named(&g, "mul_mat_id").count(), 3 * 4);
    assert_eq!((ops_named(&g, "sigmoid").count(), ops_named(&g, "soft_max").count()), (4, 0));
    assert_eq!(producer(&g, "ffn_moe_argsort-0").inputs[0], g.find("ffn_moe_logits-0").unwrap());
    // The router weight scales each token's repeated row before the experts run.
    let weighted = g.find("ffn_moe_weighted-0").unwrap();
    assert_eq!(g.tensor(weighted).dims, vec![64, 2, 8], "[n_embd, n_expert_used, n_tokens]");
    assert_eq!(producer(&g, "ffn_moe_weighted-0").inputs[0], g.find("repeated-0").unwrap());
    assert_eq!(producer(&g, "ffn_moe_up-0").inputs[1], weighted);
    // The experts' sum and the shared expert make the FFN output.
    let merged = producer(&g, "ffn_out-0");
    assert_eq!(merged.inputs[1], g.find("ffn_moe_shexp-0").unwrap());
}

#[test]
fn llama4_interleaves_dense_and_expert_layers() {
    // Maverick: 128 experts on every second layer, a dense FFN between, and no Q/K norm.
    let s = Shape { n_expert: 128, ..llama4_shape() };
    let g = bind("llama4", &testutil::llama4(&s, F16, 2, false), KvType::F16);
    check_invariants(&g);
    let experts: Vec<usize> = (0..4).map(|l| g.layer_ops(l).filter(|o| o.op == "mul_mat_id").count()).collect();
    assert_eq!(experts, vec![0, 3, 0, 3]);
    assert_eq!(layers_reading(&g, "blk.0.ffn_gate.weight"), vec![0]);
    assert!(g.find("blk.1.ffn_gate.weight").is_none());
    for l in 0..4 {
        assert_eq!(g.layer_ops(l).filter(|o| o.op == "rms_norm").count(), 2, "layer {l} has only its two norms");
    }
    assert!(ops_named(&g, "rope").all(|o| o.inputs.len() == 2), "no rope_freqs in this file");
}

#[test]
fn llama4_without_chunked_attention_has_one_cache() {
    // attention.sliding_window = 0, as for MobileLLM: no chunked layers, so no SWA
    // inputs, and n_no_rope_layer_step is n_layer, so only the last layer has no rope.
    let s = Shape { n_layer: 6, ..SMALL };
    let mut h = testutil::llama4(&s, F16, 1, false);
    h.u32("attention.sliding_window", 0);
    let g = bind("llama4", &h, KvType::F16);
    check_invariants(&g);
    assert!(g.find("attn_inp_kq_mask_swa").is_none() && g.find("inp_k_idxs_swa").is_none());
    assert_eq!(layers_reading(&g, "attn_inp_kq_mask"), (0..6).collect::<Vec<_>>());
    let ropes: Vec<usize> = (0..6).map(|l| g.layer_ops(l).filter(|o| o.op == "rope").count()).collect();
    assert_eq!(ropes, vec![2, 2, 2, 2, 2, 0]);
    // With the default pattern, the same six layers would be chunked except 3.
    let g = bind("llama4", &testutil::llama4(&s, F16, 1, false), KvType::F16);
    assert_eq!(layers_reading(&g, "attn_inp_kq_mask"), vec![3]);
}

#[test]
fn llama4_chunking_and_rope_follow_their_own_patterns() {
    // A sliding_window_pattern of 2 makes layers 0 and 2 chunked, while NoPE stays on
    // every fourth layer. Chunked layers rope with freq_base_swa, the others with
    // freq_base.
    let mut h = testutil::llama4(&llama4_shape(), F16, 1, false);
    h.u32("attention.sliding_window_pattern", 2).f32("rope.freq_base_swa", 250_000.0);
    let g = bind("llama4", &h, KvType::F16);
    check_invariants(&g);
    assert_eq!(layers_reading(&g, "attn_inp_kq_mask_swa"), vec![0, 2]);
    let bases: Vec<(u32, Param)> = ops_named(&g, "rope").map(|o| (o.layer.unwrap(), o.params["freq_base"].clone())).collect();
    let want = |l: u32, b: f64| [(l, Param::Float(b)), (l, Param::Float(b))];
    assert_eq!(bases, [want(0, 250_000.0), want(1, 1_000_000.0), want(2, 250_000.0)].concat());
}

#[test]
fn llama4_rotates_each_cache_with_its_own_matrices() {
    // A quantized cache rotates Q, K and V; each of the two caches has its own matrices.
    // With a fused QKV weight, K on the NoPE layer is still a strided view of the QKV
    // rows, so it is copied before its rotation, as V is on every layer.
    let g = bind("llama4", &testutil::llama4(&llama4_shape(), F16, 1, true), KvType::Q8_0);
    check_invariants(&g);
    assert_eq!(layers_reading(&g, "attn_inp_k_rot_swa"), vec![0, 1, 2]);
    assert_eq!(layers_reading(&g, "attn_inp_v_rot_swa"), vec![0, 1, 2]);
    assert_eq!(layers_reading(&g, "attn_inp_k_rot"), vec![3]);
    assert_eq!(layers_reading(&g, "attn_inp_v_rot"), vec![3]);
    let conts: Vec<usize> = (0..4).map(|l| g.layer_ops(l).filter(|o| o.op == "cont").count()).collect();
    assert_eq!(conts, vec![1, 1, 1, 2]);
    let hadamard = g.ops.iter().filter(|o| o.params.get("hint") == Some(&Param::Int(1))).count();
    assert_eq!(hadamard, 4 * 4, "Q, K, V and the attention output, in every layer");
}

#[test]
fn llama4_symbols_are_refused_for_other_architectures() {
    let src = TINY.replace(r#"out = "x"
"#, "out = \"x\"\nwhen = \"is_swa\"\n");
    let err = Layout::from_toml(&src).unwrap().bind(&tiny(), &opts(KvType::F16)).unwrap_err().to_string();
    assert!(err.contains("read only for llama4"), "{err}");
}

#[test]
fn llama_rope_reads_the_frequency_factors_when_present() {
    let mut h = testutil::llama(&SMALL, F16, false, false);
    h.tensor("rope_freqs.weight", &[32], F32);
    let g = bind("llama", &h, KvType::F16);
    let freqs = g.find("rope_freqs.weight").unwrap();
    for rope in ops_named(&g, "rope") {
        assert_eq!(rope.inputs.len(), 3);
        assert_eq!(rope.inputs[2], freqs, "every layer shares the one tensor");
    }
}

/// A qwen2 or nemotron test shape: their Q is n_embd wide, so 2 heads of 64 make 128.
const QWEN2_SHAPE: Shape = Shape { n_embd: 128, ..SMALL };

#[test]
fn builtin_qwen2_binds() {
    let g = bind("qwen2", &testutil::qwen2(&QWEN2_SHAPE, F16, false), KvType::F16);
    check_invariants(&g);
    let rope = ops_named(&g, "rope").next().unwrap();
    assert_eq!(rope.params["mode"], Param::Int(2), "qwen2 ropes NEOX");
    assert_eq!(rope.inputs.len(), 2, "no frequency factors");
    assert_eq!(g.tensor(rope.outputs[0]).dims, vec![64, 2, 8], "[head width, heads, tokens]");
    // The Q, K and V biases add three ops to the layer's two residual adds.
    assert_eq!(g.layer_ops(0).filter(|o| o.op == "add").count(), 3 + 2);
    // No Q and K norms, unlike qwen3: two RMS norms a layer, and the output norm.
    assert_eq!(ops_named(&g, "rms_norm").count(), 2 * 2 + 1);
}

#[test]
fn qwen2_adds_the_output_bias_when_the_file_has_one() {
    let mut h = testutil::qwen2(&QWEN2_SHAPE, F16, true);
    h.tensor("output.bias", &[32], F32);
    let g = bind("qwen2", &h, KvType::F16);
    check_invariants(&g);
    let (head, bias) = (&g.ops[g.ops.len() - 2], g.ops.last().unwrap());
    assert_eq!(head.inputs[0], g.find("token_embd.weight").unwrap(), "tied: the LM head reads the embeddings");
    assert_eq!((bias.op, bias.inputs[1]), ("add", g.find("output.bias").unwrap()));
    assert_eq!(g.outputs, bias.outputs, "the logits are the biased result");
}

#[test]
fn builtin_nemotron_binds() {
    let g = bind("nemotron", &testutil::nemotron(&QWEN2_SHAPE, F16, false), KvType::F16);
    check_invariants(&g);
    // LayerNorm everywhere, with f_norm_eps: two a layer and the output norm.
    assert_eq!(ops_named(&g, "rms_norm").count(), 0);
    assert_eq!(ops_named(&g, "norm").count(), 2 * 2 + 1);
    assert!(ops_named(&g, "norm").all(|o| o.params["eps"] == Param::Float(f64::from(1e-5f32))));
    // Each norm is followed by its weight and its bias.
    let ln: Vec<&str> = g.layer_ops(0).take(3).map(|o| o.op).collect();
    assert_eq!(ln, ["norm", "mul", "add"]);
    // Partial rotary: rope turns 32 of each head's 64 dims, while the attention scale is
    // over the whole head.
    let rope = ops_named(&g, "rope").next().unwrap();
    assert_eq!((&rope.params["mode"], &rope.params["n_dims"]), (&Param::Int(2), &Param::Int(32)));
    let fa = ops_named(&g, "flash_attn_ext").next().unwrap();
    assert_eq!(fa.params["scale"], Param::Float(1.0 / 8.0));
    // The FFN is up, relu, sqr, down, with no gate.
    assert_eq!((ops_named(&g, "relu").count(), ops_named(&g, "sqr").count(), ops_named(&g, "swiglu").count()), (2, 2, 0));
    let sqr = ops_named(&g, "sqr").next().unwrap();
    assert_eq!(g.tensor(sqr.outputs[0]).dims, vec![96, 8]);
}

#[test]
fn nemotron_requires_its_output_weight() {
    // Unlike llama and qwen2, nemotron's LM head does not fall back to the embeddings.
    let err = builtin("nemotron").unwrap().bind(&testutil::nemotron(&QWEN2_SHAPE, F16, true).model(), &opts(KvType::F16)).unwrap_err();
    assert_eq!(err, LayoutError::MissingWeight { name: "output.weight".into() });
}

/// The deci test layers, as (n_head, n_head_kv, n_ff): self-attention, linear attention,
/// attention-free, a dummy layer with neither attention nor FFN, and a last
/// self-attention layer with its own KV head count and FFN width.
const DECI_LAYERS: &[(u64, u64, u64)] = &[(2, 1, 96), (2, 0, 96), (0, 0, 96), (0, 0, 0), (2, 2, 64)];

#[test]
fn builtin_deci_binds_every_kind_of_layer() {
    let g = bind("deci", &testutil::deci(&SMALL, F16, false, DECI_LAYERS), KvType::F16);
    check_invariants(&g);
    assert_eq!(g.n_layer, 5);
    // A KV cache only where the layer has KV heads, each as wide as that layer's heads.
    let kv: Vec<(&str, &[u64])> = g.tensors.iter().filter(|t| t.kind == TensorKind::Kv).map(|t| (t.name.as_str(), t.dims.as_slice())).collect();
    assert_eq!(kv, [("cache_k_l0", &[64, 256][..]), ("cache_k_l4", &[128, 256]), ("cache_v_l0", &[64, 256]), ("cache_v_l4", &[128, 256])]);
    let fa: Vec<Option<u32>> = ops_named(&g, "flash_attn_ext").map(|o| o.layer).collect();
    assert_eq!(fa, [Some(0), Some(4)]);
    // Linear attention: the norm and the output projection, the residual, then the FFN.
    let ops = |l: u32| g.layer_ops(l).map(|o| o.op).collect::<Vec<_>>();
    assert_eq!(ops(1), ["rms_norm", "mul", "mul_mat", "add", "rms_norm", "mul", "mul_mat", "mul_mat", "swiglu", "mul_mat", "add"]);
    // Attention-free: straight to the FFN norm, which reads the layer's input.
    assert_eq!(ops(2), ["rms_norm", "mul", "mul_mat", "mul_mat", "swiglu", "mul_mat", "add"]);
    assert_eq!(g.layer_ops(2).next().unwrap().inputs[0], g.find("inpL-1").unwrap());
    assert!(g.find("ffn_inp-2").is_none());
    // The dummy layer has no ops, so the next layer reads layer 2's output.
    assert!(ops(3).is_empty());
    assert_eq!(g.layer_ops(4).next().unwrap().inputs[0], g.find("inpL-2").unwrap());
    // Each layer's FFN is its own width; the last runs over the one output row.
    assert_eq!(g.tensor(g.find("ffn_up-0").unwrap()).dims, vec![96, 8]);
    assert_eq!(g.tensor(g.find("ffn_up-4").unwrap()).dims, vec![64]);
    let rope = ops_named(&g, "rope").next().unwrap();
    assert_eq!(rope.params["mode"], Param::Int(0), "deci ropes NORM");
}

#[test]
fn deci_ropes_and_rotates_only_where_it_attends() {
    let mut h = testutil::deci(&SMALL, F16, false, DECI_LAYERS);
    h.tensor("rope_freqs.weight", &[32], F32);
    let g = bind("deci", &h, KvType::Q8_0);
    check_invariants(&g);
    let freqs = g.find("rope_freqs.weight").unwrap();
    let roped: BTreeSet<Option<u32>> = ops_named(&g, "rope").filter(|o| o.inputs.get(2) == Some(&freqs)).map(|o| o.layer).collect();
    assert_eq!(roped, BTreeSet::from([Some(0), Some(4)]));
    // Q, K and V before the write and the output after, in the two attending layers only.
    let hadamard: Vec<Option<u32>> = g.ops.iter().filter(|o| o.params.get("hint") == Some(&Param::Int(1))).map(|o| o.layer).collect();
    assert_eq!(hadamard, [Some(0); 4].into_iter().chain([Some(4); 4]).collect::<Vec<_>>());
}

#[test]
fn block_ops_expand_per_layer() {
    let shape = Shape { n_layer: 4, ..SMALL };
    let g = bind("qwen3", &testutil::qwen3(&shape, F16, false), KvType::F16);
    let per_layer: Vec<usize> = (0..4).map(|l| g.layer_ops(l).count()).collect();
    // Every layer runs the same ops, and the last adds the two output-row gathers.
    assert_eq!(per_layer[0], per_layer[1]);
    assert_eq!(per_layer[1], per_layer[2]);
    assert_eq!(per_layer[3], per_layer[0] + 2);
    let prologue = g.ops.iter().take_while(|o| o.layer.is_none()).count();
    let epilogue = g.ops.iter().rev().take_while(|o| o.layer.is_none()).count();
    assert_eq!((prologue, epilogue), (1, 3));
    assert_eq!(g.ops.len(), prologue + per_layer.iter().sum::<usize>() + epilogue);
    // A layer's ops read that layer's weights and cache, and no other layer's.
    for l in 0..4u32 {
        for op in g.layer_ops(l) {
            for i in &op.inputs {
                let t = g.tensor(*i);
                if let Some(rest) = t.name.strip_prefix("blk.") {
                    assert!(rest.starts_with(&format!("{l}.")), "layer {l} reads {}", t.name);
                }
                if t.kind == TensorKind::Kv {
                    assert_eq!(t.layer, Some(l));
                }
            }
        }
    }
    // Ops run in layer order.
    let layers: Vec<u32> = g.ops.iter().filter_map(|o| o.layer).collect();
    assert!(layers.windows(2).all(|w| w[0] <= w[1]));
    // Activations are named the way llama.cpp's cb() names graph tensors.
    assert!(g.find("ffn_inp-3").is_some());
    assert!(g.find("attn_norm-0").is_some());
}

#[test]
fn quantized_file_binds_same_layout() {
    let f16 = bind("qwen3moe", &testutil::qwen3moe(&SMALL, F16, false), KvType::F16);
    let q8 = bind("qwen3moe", &testutil::qwen3moe(&SMALL, Q8_0, false), KvType::F16);
    assert_eq!(f16.ops, q8.ops, "the same ops, inputs, outputs and params");
    assert_eq!(f16.tensors.len(), q8.tensors.len());
    let mut differ = 0;
    for (a, b) in f16.tensors.iter().zip(&q8.tensors) {
        assert_eq!((&a.name, &a.dims), (&b.name, &b.dims));
        match (&a.kind, &b.kind) {
            (TensorKind::Weight { .. }, TensorKind::Weight { .. }) => {
                if a.ggml_type != b.ggml_type {
                    differ += 1;
                    assert!(b.bytes < a.bytes, "{} is smaller in Q8_0", a.name);
                }
            }
            _ => assert_eq!(a, b, "only weights differ"),
        }
    }
    assert!(differ > 0);
}

#[test]
fn tied_output_falls_back_to_token_embd() {
    let g = bind("qwen3", &testutil::qwen3(&SMALL, F16, true), KvType::F16);
    assert!(g.find("output.weight").is_none());
    let embd = g.find("token_embd.weight").unwrap();
    let head = g.ops.last().unwrap();
    assert_eq!(head.op, "mul_mat");
    assert_eq!(head.inputs[0], embd, "the LM head reads the token embeddings");
    // It is read twice: the embedding lookup and the LM head.
    assert_eq!(g.ops.iter().filter(|o| o.inputs.contains(&embd)).count(), 2);
}

#[test]
fn unsupported_arch_is_refused_with_the_list() {
    let model = Header::new("mamba2", 8).u32("block_count", 1).u32("context_length", 16).u32("embedding_length", 8).u32("attention.head_count", 1).model();
    let err = for_model(&model, None).unwrap_err();
    assert_eq!(err, LayoutError::NoLayout { architecture: "mamba2".into(), supported: SUPPORTED });
    let msg = err.to_string();
    for arch in SUPPORTED {
        assert!(msg.contains(arch), "{msg}");
    }
    assert!(msg.contains("write a layout"), "the message says what to do: {msg}");
    // With the caller's layout for it, the same model is accepted.
    let own = Layout::from_toml("schema_version = 1\narch = \"mamba2\"\n").unwrap();
    assert_eq!(for_model(&model, Some(own)).unwrap().arch(), "mamba2");
}

#[test]
fn caller_layout_for_wrong_arch_refused() {
    let model = testutil::qwen3(&SMALL, F16, false).model();
    let err = for_model(&model, builtin("llama")).unwrap_err();
    assert_eq!(err, LayoutError::WrongArch { layout: "llama".into(), model: "qwen3".into() });
    assert!(err.to_string().contains("Pass a layout written for `qwen3`"), "{err}");
    // bind refuses it too, for a caller that skips for_model.
    assert!(matches!(builtin("llama").unwrap().bind(&model, &opts(KvType::F16)), Err(LayoutError::WrongArch { .. })));
    // And the right one is taken as given.
    assert_eq!(for_model(&model, builtin("qwen3")).unwrap().arch(), "qwen3");
}

/// Purpose: This function builds a tiny one-layer test model.
///
/// Inputs: none.
///
/// Outputs: the model.
fn tiny() -> crate::load::Model {
    let mut h = Header::new("tiny", 16);
    h.u32("block_count", 1).u32("context_length", 16).u32("embedding_length", 32).u32("attention.head_count", 1);
    h.tensor("token_embd.weight", &[32, 16], F32);
    h.tensor("blk.0.w.weight", &[32, 32], F32);
    h.model()
}

const TINY: &str = r#"
schema_version = 1
arch = "tiny"
weights = [
  { name = "token_embd.weight", dims = ["n_embd", "n_vocab"] },
  { name = "blk.{L}.w.weight", dims = ["n_embd", "n_embd"] },
]
tensors = [ { name = "inp_tokens", kind = "input", dtype = "i32", dims = ["n_tokens"] } ]
outputs = ["x"]

[[prologue]]
op = "get_rows"
in = ["token_embd.weight", "inp_tokens"]
out = "x"

[[block]]
op = "mul_mat"
in = ["blk.{L}.w.weight", "x"]
out = "x"
"#;

#[test]
fn table_syntax_binds() {
    // The `[[block]]` form and the one-op-per-line form are the same TOML data.
    let g = Layout::from_toml(TINY).unwrap().bind(&tiny(), &opts(KvType::F16)).unwrap();
    check_invariants(&g);
    assert_eq!(g.ops.len(), 2);
    assert_eq!(g.tensor(g.outputs[0]).name, "x-0");
}

#[test]
fn unknown_tensor_rejected() {
    let src = TINY.replace(r#"in = ["blk.{L}.w.weight", "x"]"#, r#"in = ["blk.{L}.w.weight", "y"]"#);
    match Layout::from_toml(&src).unwrap_err() {
        LayoutError::UnknownTensor { name, at } => {
            assert_eq!(name, "y");
            assert_eq!(at, "block op 0 (mul_mat)");
        }
        other => panic!("expected UnknownTensor, got {other}"),
    }
    // An undeclared weight is unknown too, even when the file has it.
    let src = TINY.replace(r#"  { name = "blk.{L}.w.weight", dims = ["n_embd", "n_embd"] },"#, "");
    assert!(matches!(Layout::from_toml(&src), Err(LayoutError::UnknownTensor { .. })));
}

#[test]
fn use_before_produce_rejected() {
    // `z` is produced, but only after the op that reads it.
    let src = format!("{TINY}\n[[block]]\nop = \"mul\"\nin = [\"x\", \"z\"]\nout = \"x\"\n\n[[block]]\nop = \"silu\"\nin = [\"x\"]\nout = \"z\"\n");
    let err = Layout::from_toml(&src).unwrap().bind(&tiny(), &opts(KvType::F16)).unwrap_err();
    assert!(matches!(&err, LayoutError::UseBeforeProduce { name, .. } if name == "z"), "{err}");
    // A name first assigned in the block lives one layer: the epilogue cannot read it.
    let src = format!("{TINY}\n[[block]]\nop = \"silu\"\nin = [\"x\"]\nout = \"local\"\n\n[[epilogue]]\nop = \"silu\"\nin = [\"local\"]\nout = \"x\"\n");
    let err = Layout::from_toml(&src).unwrap().bind(&tiny(), &opts(KvType::F16)).unwrap_err();
    assert!(matches!(&err, LayoutError::UseBeforeProduce { name, .. } if name == "local"), "{err}");
}

#[test]
fn unreferenced_gguf_tensor_rejected() {
    let mut h = testutil::qwen3(&SMALL, F16, false);
    h.tensor("blk.0.mystery.weight", &[64], F32);
    let err = builtin("qwen3").unwrap().bind(&h.model(), &opts(KvType::F16)).unwrap_err();
    assert_eq!(err, LayoutError::UnreferencedTensors { names: vec!["blk.0.mystery.weight".into()] });
    assert!(err.to_string().contains("unused"), "the message names the escape hatch: {err}");
    // A tensor llama.cpp loads but never reads is listed in `unused`, and binds.
    let mut h = testutil::qwen3(&SMALL, F16, false);
    h.tensor("cls.output.weight", &[64, 2], F16);
    let g = bind("qwen3", &h, KvType::F16);
    let cls = g.find("cls.output.weight").unwrap();
    assert!(g.ops.iter().all(|o| !o.inputs.contains(&cls)), "listed, placed, never read");
}

#[test]
fn a_missing_weight_and_a_wrong_shape_are_refused() {
    // The header says two layers and holds tensors for one.
    let err = builtin("qwen3").unwrap().bind(&lie_about_layers(), &opts(KvType::F16)).unwrap_err();
    assert_eq!(err, LayoutError::MissingWeight { name: "blk.1.attn_norm.weight".into() });
    // A weight the wrong shape for the hparams.
    let src = TINY.replace(r#"dims = ["n_embd", "n_embd"]"#, r#"dims = ["n_embd", "2 * n_embd"]"#);
    let err = Layout::from_toml(&src).unwrap().bind(&tiny(), &opts(KvType::F16)).unwrap_err();
    assert_eq!(err, LayoutError::WeightShape { name: "blk.0.w.weight".into(), layout: vec![32, 64], file: vec![32, 32] });
}

/// Purpose: This function builds a qwen3 test model whose header says two layers but
/// holds one.
///
/// Inputs: none.
///
/// Outputs: the model.
fn lie_about_layers() -> crate::load::Model {
    let s = Shape { n_layer: 1, ..SMALL };
    let mut h = Header::new("qwen3", s.n_vocab);
    h.u32("block_count", 2)
        .u32("context_length", 4096)
        .u32("embedding_length", 64)
        .u32("attention.head_count", 2)
        .u32("attention.head_count_kv", 1)
        .u32("attention.key_length", 64)
        .u32("attention.value_length", 64)
        .u32("feed_forward_length", 96)
        .f32("attention.layer_norm_rms_epsilon", 1e-6);
    h.tensor("token_embd.weight", &[64, 32], F16).tensor("output_norm.weight", &[64], F32).tensor("output.weight", &[64, 32], F16);
    for (n, d, t) in [
        ("attn_norm.weight", vec![64], F32),
        ("attn_q.weight", vec![64, 128], F16),
        ("attn_k.weight", vec![64, 64], F16),
        ("attn_v.weight", vec![64, 64], F16),
        ("attn_output.weight", vec![128, 64], F16),
        ("attn_k_norm.weight", vec![64], F32),
        ("attn_q_norm.weight", vec![64], F32),
        ("ffn_norm.weight", vec![64], F32),
        ("ffn_gate.weight", vec![64, 96], F16),
        ("ffn_down.weight", vec![96, 64], F16),
        ("ffn_up.weight", vec![64, 96], F16),
    ] {
        h.tensor(&format!("blk.0.{n}"), &d, t);
    }
    h.model()
}

#[test]
fn optional_biases_add_their_ops() {
    // qwen2-style Q, K and V biases are optional in create_tensor_qkv.
    let plain = bind("qwen3", &testutil::qwen3(&SMALL, F16, false), KvType::F16);
    let mut h = testutil::qwen3(&SMALL, F16, false);
    for l in 0..SMALL.n_layer {
        h.tensor(&format!("blk.{l}.attn_q.bias"), &[128], F32);
        h.tensor(&format!("blk.{l}.attn_k.bias"), &[64], F32);
        h.tensor(&format!("blk.{l}.attn_v.bias"), &[64], F32);
    }
    let biased = bind("qwen3", &h, KvType::F16);
    check_invariants(&biased);
    assert_eq!(ops_named(&biased, "add").count(), ops_named(&plain, "add").count() + 3 * 2);
}

#[test]
fn a_fused_qkv_file_takes_the_view_path() {
    let s = SMALL;
    let mut h = Header::new("llama", s.n_vocab);
    h.u32("block_count", 1)
        .u32("context_length", 4096)
        .u32("embedding_length", 64)
        .u32("attention.head_count", 2)
        .u32("attention.head_count_kv", 1)
        .u32("feed_forward_length", 96)
        .f32("attention.layer_norm_rms_epsilon", 1e-5);
    h.tensor("token_embd.weight", &[64, 32], F16).tensor("output_norm.weight", &[64], F32);
    h.tensor("blk.0.attn_norm.weight", &[64], F32)
        .tensor("blk.0.attn_qkv.weight", &[64, 32 * 2 + 32 + 32], F16)
        .tensor("blk.0.attn_output.weight", &[64, 64], F16)
        .tensor("blk.0.ffn_norm.weight", &[64], F32)
        .tensor("blk.0.ffn_gate.weight", &[64, 96], F16)
        .tensor("blk.0.ffn_down.weight", &[96, 64], F16)
        .tensor("blk.0.ffn_up.weight", &[64, 96], F16);
    // head width 32 here (64 / 2 heads, no key_length), so no KV rotation applies.
    let g = bind("llama", &h, KvType::Q8_0);
    check_invariants(&g);
    let wqkv = g.find("wqkv-0").unwrap();
    let kcur = g.tensor(g.find("Kcur-0").unwrap());
    assert_eq!(kcur.kind, TensorKind::View { of: wqkv, byte_offset: 32 * 2 * 4, nb: [4, 32 * 4, 128 * 4, 128 * 4 * 8] });
    assert!(g.find("attn_inp_k_rot").is_none());
}

#[test]
fn a_quantized_kv_cache_adds_the_rotation() {
    let f16 = bind("qwen3", &testutil::qwen3(&SMALL, F16, false), KvType::F16);
    let q8 = bind("qwen3", &testutil::qwen3(&SMALL, F16, false), KvType::Q8_0);
    check_invariants(&q8);
    assert!(f16.find("attn_inp_k_rot").is_none());
    let rot = q8.tensor(q8.find("attn_inp_k_rot").unwrap());
    assert_eq!((rot.kind.clone(), rot.dims.clone()), (TensorKind::Input, vec![64, 64]));
    let hadamard: Vec<&BoundOp> = q8.ops.iter().filter(|o| o.params.get("hint") == Some(&Param::Int(1))).collect();
    // Q, K and V before the write, and the attention output after, in every layer.
    assert_eq!(hadamard.len(), 4 * 2);
    // The cache itself is q8_0, sized in whole blocks.
    let cache = q8.tensor(q8.find("cache_k_l0").unwrap());
    assert_eq!((cache.ggml_type, cache.bytes), (8, 64 / 32 * 34 * 256));
}

#[test]
fn an_f32_cache_is_cast_for_flash_attention() {
    let g = bind("qwen3", &testutil::qwen3(&SMALL, F16, false), KvType::F32);
    let casts: Vec<&BoundOp> = ops_named(&g, "cpy").collect();
    assert_eq!(casts.len(), 2 * 2);
    assert_eq!(g.tensor(casts[0].outputs[0]).ggml_type, F16);
}

#[test]
fn kv_writes_and_reads_are_views_of_the_cache() {
    let g = bind("qwen3moe", &testutil::qwen3moe(&SMALL, F16, false), KvType::Q8_0);
    let cache = g.find("cache_k_l1").unwrap();
    let write = ops_named(&g, "set_rows").find(|o| o.inputs[0] == cache).unwrap();
    let stored = g.tensor(write.outputs[0]);
    assert_eq!(stored.kind, TensorKind::View { of: cache, byte_offset: 0, nb: [34, 68, 68 * 256, 68 * 256] });
    // The read view and its permute alias the cache too, so the planner gives none of
    // them memory; and the read comes after the write.
    let k = g.tensor(g.find("k-1").unwrap());
    assert!(matches!(k.kind, TensorKind::View { of, .. } if of == cache));
    let k_perm = g.tensor(g.find("k-1#1").unwrap());
    assert!(matches!(k_perm.kind, TensorKind::View { of, .. } if of == cache));
    assert_eq!(k_perm.dims, vec![64, 256], "[head width, cells], with one KV head dropped as a trailing 1");
    let pos = |id| g.ops.iter().position(|o| o.outputs.contains(&id)).unwrap();
    assert!(pos(write.outputs[0]) < pos(g.find("k-1").unwrap()));
}

#[test]
fn impossible_options_are_refused() {
    let model = testutil::qwen3(&SMALL, F16, false).model();
    let layout = builtin("qwen3").unwrap();
    for bad in [
        BindOptions { n_outputs: 9, ..opts(KvType::F16) },
        BindOptions { n_tokens: 0, ..opts(KvType::F16) },
        BindOptions { n_ctx: 4, ..opts(KvType::F16) },
    ] {
        assert!(matches!(layout.bind(&model, &bad), Err(LayoutError::Options(_))), "{bad:?}");
    }
    // Every output row: the last layer gathers all of them, and the logits are per token.
    let all = layout.bind(&model, &BindOptions { n_outputs: 8, ..opts(KvType::F16) }).unwrap();
    assert_eq!(all.tensor(all.outputs[0]).dims, vec![32, 8]);
}

#[test]
fn binding_is_deterministic() {
    let h = testutil::qwen3moe(&SMALL, Q8_0, true);
    let a = serde_json::to_string(&bind("qwen3moe", &h, KvType::Q4_0)).unwrap();
    let b = serde_json::to_string(&bind("qwen3moe", &h, KvType::Q4_0)).unwrap();
    assert_eq!(a, b);
    assert!(a.contains("\"kv_k\":\"q4_0\""));
}

#[test]
fn layout_mistakes_name_the_place() {
    let bad_op = TINY.replace(r#"op = "mul_mat""#, r#"op = "matmul""#);
    let err = Layout::from_toml(&bad_op).unwrap_err().to_string();
    assert!(err.contains("block op 0") && err.contains("mul_mat"), "{err}");
    let bad_param = TINY.replace(r#"out = "x"
"#, "out = \"x\"\neps = 1e-6\n");
    assert!(Layout::from_toml(&bad_param).unwrap_err().to_string().contains("takes no param"));
    let bad_symbol = TINY.replace(r#"["n_embd", "n_vocab"]"#, r#"["n_embd", "vocab"]"#);
    assert!(Layout::from_toml(&bad_symbol).unwrap_err().to_string().contains("unknown name `vocab`"));
    // A misspelled has() would be false everywhere, so it is refused wherever it appears.
    let bad_has = TINY.replace(r#"dims = ["n_tokens"] }"#, r#"dims = ["n_tokens"], when = "has(blk.{L}.typo.weight)" }"#);
    assert!(Layout::from_toml(&bad_has).unwrap_err().to_string().contains("has(blk.{L}.typo.weight)"));
    let bad_version = TINY.replace("schema_version = 1", "schema_version = 2");
    assert!(Layout::from_toml(&bad_version).unwrap_err().to_string().contains("version 2"));
    // A [32, 32] weight times the 8 token ids: ggml_can_mul_mat would refuse it.
    let bad_shape = TINY.replace(r#"in = ["blk.{L}.w.weight", "x"]"#, r#"in = ["blk.{L}.w.weight", "inp_tokens"]"#);
    let err = Layout::from_toml(&bad_shape).unwrap().bind(&tiny(), &opts(KvType::F16)).unwrap_err();
    assert!(matches!(&err, LayoutError::Shape { at, detail } if at == "block op 0 (mul_mat), layer 0" && detail.contains("a.ne0 == b.ne0")), "{err}");
}

#[test]
fn the_readme_documents_every_symbol_and_op() {
    let readme = include_str!("README.md");
    for (name, _) in SYMBOLS {
        assert!(readme.contains(&format!("`{name}`")), "README does not mention symbol {name}");
    }
    for name in ops::names() {
        assert!(readme.contains(&format!("`{name}`")), "README does not mention op {name}");
    }
}

/// Bind a real model header, when one is given. Set `DOVATA_TEST_HEADER` to a GGUF
/// file, or the first tens of megabytes of one fetched with an HTTP range request, as
/// long as its tensor index is complete.
#[test]
fn a_real_header_binds_its_builtin_layout() {
    let Ok(path) = std::env::var("DOVATA_TEST_HEADER") else {
        eprintln!("skipped: set DOVATA_TEST_HEADER to a GGUF header to bind a real model");
        return;
    };
    let model = crate::load::Model::open(std::path::Path::new(&path)).unwrap_or_else(|e| panic!("{path}: {e}"));
    let layout = for_model(&model, None).unwrap_or_else(|e| panic!("{e}"));
    let opts = BindOptions { n_ctx: 32768, n_tokens: 512, n_outputs: 1, kv_k: KvType::Q4_0, kv_v: KvType::Q4_0 };
    let g = layout.bind(&model, &opts).unwrap_or_else(|e| panic!("{path}: {e}"));
    check_invariants(&g);
    let weights = g.tensors.iter().filter(|t| matches!(t.kind, TensorKind::Weight { .. })).count();
    let kv: u64 = g.tensors.iter().filter(|t| t.kind == TensorKind::Kv).map(|t| t.bytes).sum();
    eprintln!(
        "{path}: {} ({}), {} layers, {weights} weights, {} ops, {} tensors, KV {} bytes at n_ctx {}",
        model.architecture(),
        model.meta().name().unwrap_or("?"),
        g.n_layer,
        g.ops.len(),
        g.tensors.len(),
        kv,
        opts.n_ctx
    );
    assert_eq!(weights, model.tensors().count());
    // The bound KV cache is exactly what ModelDims prices, which matches llama.cpp.
    // ModelDims reads one head count for every layer, so a file with per-layer head
    // counts, such as deci's, is priced layer by layer instead. An MLA model keeps no V
    // cache (llama_kv_cache's has_v is !is_mla), which ModelDims does not know, so its V
    // half is left out of the price.
    let expected = per_layer_kv_bytes(&model, &opts).unwrap_or_else(|| {
        let mut dims = crate::load::ModelDims::from_gguf(model.meta()).unwrap();
        if g.find("cache_v_l0").is_none() {
            dims.value_length = Some(0);
        }
        dims.kv_bytes_per_token(opts.kv_k, opts.kv_v) * u64::from(opts.n_ctx)
    });
    assert_eq!(kv, expected);
    if model.architecture() == "qwen3moe" && g.n_layer == 48 {
        // Qwen3-Coder-30B-A3B: 579 tensors, and llama.cpp's own KV buffer is 864 MiB at
        // 32768 tokens of q4_0.
        assert_eq!(weights, 579);
        assert_eq!(kv, 864 * 1024 * 1024);
    }
    if model.architecture() == "deci" && model.meta().name().is_some_and(|n| n.contains("51B")) {
        // Llama-3.1-Nemotron-51B: 54 of its 80 layers attend, with 371 KV heads in all, so
        // llama.cpp's KV buffer is 1669.5 MiB at 32768 tokens of q4_0.
        assert_eq!(g.tensors.iter().filter(|t| t.kind == TensorKind::Kv).count(), 2 * 54);
        assert_eq!(ops_named(&g, "flash_attn_ext").count(), 54);
        assert_eq!(kv, 371 * 128 * 2 * 32768 * 18 / 32);
    }
    if model.architecture() == "llama4" && model.meta().u64("llama4.attention.sliding_window").is_none() {
        // Llama 4 Scout and Maverick: three of every four layers attend in chunks, and
        // every interleave_moe_layer_step-th layer (Scout 1, Maverick 2) has experts.
        let reads = |name: &str| g.find(name).map_or(0, |id| g.ops.iter().filter(|o| o.inputs.contains(&id)).count());
        assert_eq!(reads("attn_inp_kq_mask_swa"), g.n_layer as usize / 4 * 3);
        let step = model.meta().u64("llama4.interleave_moe_layer_step").unwrap();
        assert_eq!(ops_named(&g, "sigmoid").count() as u64, u64::from(g.n_layer) / step);
    }
}

/// Purpose: This function prices a KV cache layer by layer from a header whose head
/// counts are per-layer arrays.
///
/// Inputs:
/// - `model`: the opened GGUF model.
/// - `opts`: the bind options, for the cache length and types.
///
/// Outputs: the KV bytes, or `None` when the header gives one head count for every
/// layer, which `ModelDims` prices.
fn per_layer_kv_bytes(model: &crate::load::Model, opts: &BindOptions) -> Option<u64> {
    use crate::gguf::Value;
    use crate::load::dims::kv_elem_bytes_x32;
    let meta = model.meta();
    let key = |suffix: &str| format!("{}.{suffix}", model.architecture());
    let is_array = |suffix: &str| matches!(meta.get(&key(suffix)), Some(Value::Array(_)));
    if !is_array("attention.head_count") && !is_array("attention.head_count_kv") {
        return None;
    }
    let at = |suffix: &str, layer: usize| match meta.get(&key(suffix))? {
        Value::Array(items) => items.get(layer)?.as_u64(),
        other => other.as_u64(),
    };
    let n_embd = meta.u64(&key("embedding_length"))?;
    let heads = at("attention.head_count", 0).filter(|h| *h > 0)?;
    let head_k = meta.u64(&key("attention.key_length")).unwrap_or(n_embd / heads);
    let head_v = meta.u64(&key("attention.value_length")).unwrap_or(n_embd / heads);
    // llama_kv_cache's constructor makes one K and one V cache of n_embd_k_gqa(il) by
    // n_ctx per layer, zero wide where the layer has no KV heads.
    let mut x32 = 0;
    for layer in 0..usize::try_from(meta.u64(&key("block_count"))?).ok()? {
        let kv = at("attention.head_count_kv", layer).or_else(|| at("attention.head_count", layer))?;
        x32 += kv * (head_k * kv_elem_bytes_x32(opts.kv_k) + head_v * kv_elem_bytes_x32(opts.kv_v));
    }
    Some(x32 / 32 * u64::from(opts.n_ctx))
}

// ---- deepseek2 ----

use super::testutil::{Deepseek2, DEEPSEEK_V3, DS2_EMBD, DS2_HEAD, DS2_KV_LORA, DS2_NOPE, DS2_ROT, DS2_V};

/// Purpose: This function counts one layer's ops of one kind.
///
/// Inputs:
/// - `g`: the bound graph.
/// - `layer`: the layer index.
/// - `op`: the op name.
///
/// Outputs: the count.
fn layer_count(g: &BoundGraph, layer: u32, op: &str) -> usize {
    g.layer_ops(layer).filter(|o| o.op == op).count()
}

/// Purpose: This function binds the built-in deepseek2 layout to a test header of one
/// variant.
///
/// Inputs:
/// - `v`: the deepseek2 variant.
/// - `ty`: the type of the large weights.
/// - `kv`: the KV cache type.
///
/// Outputs: the bound graph, checked for the facts every bound graph must have. It
/// panics on an error.
fn bind_deepseek2(v: &Deepseek2, ty: u32, kv: KvType) -> BoundGraph {
    let g = bind("deepseek2", &testutil::deepseek2(v, ty), kv);
    check_invariants(&g);
    g
}

#[test]
fn builtin_deepseek2_binds_dense_then_moe_layers() {
    let g = bind_deepseek2(&DEEPSEEK_V3, F16, KvType::F16);
    assert_eq!(g.n_layer, 3);
    // Layer 0 is dense: one SwiGLU FFN and no experts.
    assert_eq!(layer_count(&g, 0, "mul_mat_id"), 0);
    assert_eq!(layer_count(&g, 0, "swiglu"), 1);
    assert_eq!(layer_count(&g, 0, "sigmoid"), 0);
    // The rest route to experts, gated by sigmoid with normalized weights, and add the
    // shared experts' dense FFN.
    for l in 1..3 {
        assert_eq!(layer_count(&g, l, "mul_mat_id"), 3);
        assert_eq!(layer_count(&g, l, "swiglu"), 2, "routed and shared experts");
        assert_eq!(layer_count(&g, l, "sigmoid"), 1);
        assert_eq!(layer_count(&g, l, "sum_rows"), 1, "expert_weights_norm");
    }
    assert_eq!(ops_named(&g, "scale").next().unwrap().params["s"], Param::Float(2.5), "expert_weights_scale");
    // The selection bias steers the top-k, and the weights still come from the unbiased
    // probabilities.
    let sort = g.layer_ops(1).find(|o| o.op == "argsort").unwrap();
    assert_eq!(sort.inputs[0], g.find("selection_probs-1").unwrap());
    let weights = g.layer_ops(1).find(|o| o.op == "get_rows" && g.tensor(o.outputs[0]).name == "ffn_moe_weights-1").unwrap();
    assert_eq!(g.storage(weights.inputs[0]), g.find("ffn_moe_probs-1").unwrap());
    // The last layer runs its experts over the one output row.
    assert_eq!(g.tensor(g.find("ffn_moe_topk-2").unwrap()).dims, vec![2]);
    assert_eq!(g.tensor(g.outputs[0]).dims, vec![32]);
}

#[test]
fn deepseek2_mla_keeps_one_k_cache_and_reads_v_from_it() {
    let g = bind_deepseek2(&DEEPSEEK_V3, F16, KvType::Q8_0);
    // One cache per layer, K only, with a row of kv_lora_rank + n_rot values per cell.
    assert!(g.find("cache_v_l0").is_none() && g.find("inp_v_idxs").is_none());
    let cache = g.find("cache_k_l1").unwrap();
    assert_eq!(g.tensor(cache).dims, vec![DS2_KV_LORA + DS2_ROT, 256]);
    // V is the first kv_lora_rank values of every K row, a view of the same cache.
    let v = g.tensor(g.find("v-1").unwrap());
    assert!(matches!(v.kind, TensorKind::View { of, byte_offset: 0, .. } if of == cache));
    assert_eq!(v.dims, vec![DS2_KV_LORA, 1, 256]);
    // Q is the absorbed part then the rope part, per head; K is one head of the same
    // width, so attention is MQA.
    assert_eq!(g.tensor(g.find("Qcur-1").unwrap()).dims, vec![DS2_KV_LORA + DS2_ROT, DS2_HEAD, 8]);
    assert_eq!(g.tensor(g.find("Kcur-1").unwrap()).dims, vec![DS2_KV_LORA + DS2_ROT, 1, 8]);
    assert_eq!(ops_named(&g, "concat").count(), 2 * 3);
    // Attention gives kv_lora_rank values per head, and attn_v_b decompresses them to
    // the V head width before the output projection.
    assert_eq!(g.tensor(g.find("__fattn__-1").unwrap()).dims, vec![DS2_KV_LORA, DS2_HEAD, 8]);
    assert_eq!(g.tensor(g.find("kqv_out-1").unwrap()).dims, vec![DS2_V * DS2_HEAD, 8]);
    assert_eq!(g.tensor(g.find("attn_out-1").unwrap()).dims, vec![DS2_EMBD, 8]);
    // The K-only path builds no Hadamard rotation, though this cache is quantized and its
    // row divides by 64.
    assert!(g.find("attn_inp_k_rot").is_none());
    assert!(g.ops.iter().all(|o| o.params.get("hint") != Some(&Param::Int(1))));
    // An f32 cache is cast for flash attention twice per layer: K, and V, which has the
    // K cache's type.
    let f32_cache = bind_deepseek2(&DEEPSEEK_V3, F16, KvType::F32);
    assert_eq!(ops_named(&f32_cache, "cpy").count(), 2 * 3);
}

#[test]
fn deepseek2_low_rank_q_is_optional() {
    let pair = bind_deepseek2(&DEEPSEEK_V3, F16, KvType::F16);
    let lite = bind_deepseek2(&Deepseek2 { n_layer: 26, q_lora_rank: 0, ..DEEPSEEK_V3 }, F16, KvType::F16);
    let reads_q = |g: &BoundGraph| -> Vec<String> {
        g.layer_ops(0).flat_map(|o| o.inputs.iter().map(|i| g.tensor(*i).name.clone())).filter(|n| n.contains(".attn_q")).collect()
    };
    assert_eq!(reads_q(&pair), ["blk.0.attn_q_a.weight", "blk.0.attn_q_a_norm.weight", "blk.0.attn_q_b.weight"]);
    assert_eq!(reads_q(&lite), ["blk.0.attn_q.weight"]);
    // Either way Q splits into the same non-rope and rope parts per head.
    for g in [&pair, &lite] {
        assert_eq!(g.tensor(g.find("q_pe-0").unwrap()).dims, vec![DS2_ROT, DS2_HEAD, 8]);
        assert_eq!(g.tensor(g.find("q_nope-0").unwrap()).dims, vec![DS2_NOPE, DS2_HEAD, 8]);
    }
    // A model that is not a lite variant must state its rank, as llama.cpp requires.
    let unstated = testutil::deepseek2(&Deepseek2 { q_lora_rank: 0, ..DEEPSEEK_V3 }, F16);
    let err = builtin("deepseek2").unwrap().bind(&unstated.model(), &opts(KvType::F16)).unwrap_err();
    assert!(err.to_string().contains("deepseek2.attention.q_lora_rank"), "{err}");
}

#[test]
fn deepseek2_legacy_kv_b_file_takes_the_kv_path() {
    // An older file, with attn_kv_b whole and no MLA head widths, in Q8_0.
    let g = bind_deepseek2(&Deepseek2 { mla: false, ..DEEPSEEK_V3 }, Q8_0, KvType::Q8_0);
    // An ordinary K and V cache, with n_head heads each.
    assert_eq!(g.tensor(g.find("cache_k_l0").unwrap()).dims, vec![(DS2_NOPE + DS2_ROT) * DS2_HEAD, 256]);
    assert_eq!(g.tensor(g.find("cache_v_l0").unwrap()).dims, vec![DS2_V * DS2_HEAD, 256]);
    assert!(g.find("fattn_mla-0").is_none());
    // K's one rope head is repeated to every head.
    assert_eq!(ops_named(&g, "repeat").count(), 3);
    assert_eq!(g.tensor(g.find("Kcur-0").unwrap()).dims, vec![DS2_NOPE + DS2_ROT, DS2_HEAD, 8]);
    // This path rotates a quantized cache: K's head width of 64 divides by 64, and V's
    // of 32 does not. Q and K are rotated in every layer.
    assert!(g.find("attn_inp_k_rot").is_some() && g.find("attn_inp_v_rot").is_none());
    assert_eq!(g.ops.iter().filter(|o| o.params.get("hint") == Some(&Param::Int(1))).count(), 2 * 3);
}

#[test]
fn deepseek2_v2_files_default_to_softmax() {
    // A DeepSeek V2 file: no gating key, no bias, no normalization, a scale of 1.
    let v2 = Deepseek2 { gating: None, probs_bias: false, weights_norm: false, weights_scale: 1.0, ..DEEPSEEK_V3 };
    let g = bind_deepseek2(&v2, F16, KvType::F16);
    assert_eq!((ops_named(&g, "soft_max").count(), ops_named(&g, "sigmoid").count()), (2, 0));
    // Without the bias, the top-k reads the probabilities themselves, and the weights go
    // to the experts as they are.
    let sort = g.layer_ops(1).find(|o| o.op == "argsort").unwrap();
    assert_eq!(sort.inputs[0], g.find("ffn_moe_probs-1").unwrap());
    assert_eq!((ops_named(&g, "sum_rows").count(), ops_named(&g, "scale").count()), (0, 0));
}

#[test]
fn deepseek2_expert_groups_mask_the_unchosen_groups() {
    for bias in [true, false] {
        let g = bind_deepseek2(&Deepseek2 { n_expert_groups: 4, probs_bias: bias, ..DEEPSEEK_V3 }, F16, KvType::F16);
        assert_eq!(ops_named(&g, "fill").count(), 2, "one per MoE layer");
        assert_eq!(ops_named(&g, "fill").next().unwrap().params["c"], Param::Float(f64::NEG_INFINITY));
        // [experts per group, groups, tokens], and the chosen groups: [groups used, tokens].
        assert_eq!(g.tensor(g.find("selection_groups-1").unwrap()).dims, vec![2, 4, 8]);
        assert_eq!(g.tensor(g.find("expert_groups-1#1").unwrap()).dims, vec![2, 8]);
        // The groups are scored from the biased probabilities when there are any.
        let grouped = g.layer_ops(1).find(|o| o.op == "reshape" && g.tensor(o.outputs[0]).name == "selection_groups-1").unwrap();
        let source = if bias { "selection_probs-1" } else { "ffn_moe_probs-1" };
        assert_eq!(grouped.inputs[0], g.find(source).unwrap());
        // The top-k over experts reads the masked scores.
        let sort = g.layer_ops(1).find(|o| o.op == "argsort" && g.tensor(o.outputs[0]).name == "ffn_moe_argsort-1").unwrap();
        assert_eq!(g.storage(sort.inputs[0]), g.find("selection_masked-1").unwrap());
    }
}

#[test]
fn deepseek2_fused_gate_up_experts_split_by_views() {
    let g = bind_deepseek2(&Deepseek2 { gate_up: true, ..DEEPSEEK_V3 }, F16, KvType::F16);
    assert_eq!(layer_count(&g, 1, "mul_mat_id"), 2);
    let fused = g.find("ffn_moe_gate_up-1").unwrap();
    let up = g.tensor(g.find("ffn_moe_up-1").unwrap());
    assert!(matches!(up.kind, TensorKind::View { of, byte_offset, .. } if of == fused && byte_offset == 32 * 4), "{:?}", up.kind);
}

#[test]
fn deepseek2_yarn_scales_kq_and_mistral_scales_q_by_temperature() {
    let g = bind_deepseek2(&Deepseek2 { temp_scale: 0.1, ..DEEPSEEK_V3 }, F16, KvType::F16);
    // YaRN factor 40 with DeepSeek V3's yarn_log_multiplier: rope's attn_factor cancels
    // the YaRN magnitude, and kq_scale applies mscale = 1 + 0.1 ln 40 twice instead.
    let ln40 = 40f32.ln();
    let float = |p: &Param| match p {
        Param::Float(v) => *v,
        other => panic!("{other:?}"),
    };
    let attn = float(&ops_named(&g, "rope").next().unwrap().params["attn_factor"]);
    assert!((attn - f64::from(1.0 / (1.0 + 0.1 * ln40))).abs() < 1e-6, "{attn}");
    let mscale = f64::from(1.0 + 0.1 * ln40);
    let scale = float(&ops_named(&g, "flash_attn_ext").next().unwrap().params["scale"]);
    #[allow(clippy::cast_precision_loss)]
    let head = ((DS2_NOPE + DS2_ROT) as f64).sqrt();
    assert!((scale - mscale * mscale / head).abs() < 1e-6, "{scale}");
    // The temperature: one scale per token, multiplied into Q in every layer.
    let temp = g.find("inp_attn_scale").unwrap();
    assert_eq!(g.tensor(temp).dims, vec![1, 1, 8]);
    assert_eq!(g.ops.iter().filter(|o| o.op == "mul" && o.inputs.contains(&temp)).count(), 3);
    assert!(bind_deepseek2(&DEEPSEEK_V3, F16, KvType::F16).find("inp_attn_scale").is_none());
}
