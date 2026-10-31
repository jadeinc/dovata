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
fn every_op_name_exists_in_ggml_h() {
    // The enums, not the API functions: a layout op is one graph node type, and ggml's
    // node types are the `ggml_op` values, with unary and glu nodes named by their
    // sub-op. See the docs in ops.rs for why function names would not do.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/llama/bindings-sys/llama.cpp/ggml/include/ggml.h");
    let header = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("cannot read {}: {e}. Check out the llama.cpp submodule: git submodule update --init src/llama/bindings-sys/llama.cpp", path.display())
    });
    let enum_values = |name: &str, prefix: &str| -> BTreeSet<String> {
        let start = header.find(&format!("enum {name} {{")).unwrap_or_else(|| panic!("ggml.h has no enum {name}"));
        let body = &header[start..start + header[start..].find("};").expect("the enum ends")];
        body.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .filter_map(|w| w.strip_prefix(prefix))
            .filter(|w| *w != "COUNT")
            .map(str::to_owned)
            .collect()
    };
    let op = enum_values("ggml_op", "GGML_OP_");
    let unary = enum_values("ggml_unary_op", "GGML_UNARY_OP_");
    let glu = enum_values("ggml_glu_op", "GGML_GLU_OP_");
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
        assert!(set.contains(value), "layout op {name} names {value}, which ggml.h does not have");
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
    let dims = crate::load::ModelDims::from_gguf(model.meta()).unwrap();
    assert_eq!(kv, dims.kv_bytes_per_token(opts.kv_k, opts.kv_v) * u64::from(opts.n_ctx));
    if model.architecture() == "qwen3moe" && g.n_layer == 48 {
        // Qwen3-Coder-30B-A3B: 579 tensors, and llama.cpp's own KV buffer is 864 MiB at
        // 32768 tokens of q4_0.
        assert_eq!(weights, 579);
        assert_eq!(kv, 864 * 1024 * 1024);
    }
}
