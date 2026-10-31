//! Author: Joe Ricks
//! File: src/quantize/fixture.rs
//! Description: Builds a tiny F32 `llama` model for the quantize tests.

use std::path::{Path, PathBuf};

use crate::gguf::write::Writer;
use crate::gguf::Value;

pub(crate) const N_VOCAB: u32 = 32;
pub(crate) const N_EMBD: u32 = 64;
pub(crate) const N_HEAD: u32 = 4;
pub(crate) const N_HEAD_KV: u32 = 2;
pub(crate) const HEAD_DIM: u32 = N_EMBD / N_HEAD;
pub(crate) const N_FF: u32 = 128;
pub(crate) const N_LAYER: u32 = 2;

/// Purpose: This function sets every metadata key of the tiny test model except
/// `general.file_type`.
///
/// Inputs:
/// - `w`: the GGUF writer to add the keys to.
///
/// Outputs: none.
pub(crate) fn llama_metadata(w: &mut Writer) {
    let s = |v: &str| Value::String(v.to_owned());
    let mut tokens = vec!["<unk>".to_owned(), "<s>".to_owned(), "</s>".to_owned(), "<0x0A>".to_owned()];
    tokens.extend(('a'..='z').map(|c| format!("\u{2581}{c}")));
    tokens.extend(["\u{2581}the".to_owned(), "\u{2581}of".to_owned()]);
    assert_eq!(tokens.len(), N_VOCAB as usize);
    // LLAMA_TOKEN_TYPE_: 1 normal, 2 unknown, 3 control, 6 byte.
    let types = (0..N_VOCAB).map(|i| Value::I32(match i {
        0 => 2,
        1 | 2 => 3,
        3 => 6,
        _ => 1,
    }));

    w.kv("general.architecture", s("llama"))
        .kv("general.name", s("dovata tiny llama"))
        .kv("llama.context_length", Value::U32(128))
        .kv("llama.embedding_length", Value::U32(N_EMBD))
        .kv("llama.block_count", Value::U32(N_LAYER))
        .kv("llama.feed_forward_length", Value::U32(N_FF))
        .kv("llama.attention.head_count", Value::U32(N_HEAD))
        .kv("llama.attention.head_count_kv", Value::U32(N_HEAD_KV))
        .kv("llama.rope.dimension_count", Value::U32(HEAD_DIM))
        .kv("llama.rope.freq_base", Value::F32(10_000.0))
        .kv("llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5))
        .kv("llama.vocab_size", Value::U32(N_VOCAB))
        .kv("tokenizer.ggml.model", s("llama"))
        .kv("tokenizer.ggml.tokens", Value::Array(tokens.into_iter().map(Value::String).collect()))
        .kv("tokenizer.ggml.scores", Value::Array((0..N_VOCAB).map(|i| Value::F32(-(i as f32))).collect()))
        .kv("tokenizer.ggml.token_type", Value::Array(types.collect()))
        .kv("tokenizer.ggml.bos_token_id", Value::U32(1))
        .kv("tokenizer.ggml.eos_token_id", Value::U32(2))
        .kv("tokenizer.ggml.unknown_token_id", Value::U32(0));
}

/// Purpose: This function lists every tensor of the tiny test model with its shape.
///
/// Inputs: none.
///
/// Outputs: each tensor's name and shape, in the order llama.cpp's converter writes them.
pub(crate) fn tensor_shapes() -> Vec<(String, Vec<u64>)> {
    let (e, v, q, kv, ff) =
        (u64::from(N_EMBD), u64::from(N_VOCAB), u64::from(HEAD_DIM * N_HEAD), u64::from(HEAD_DIM * N_HEAD_KV), u64::from(N_FF));
    let mut t = vec![("token_embd.weight".to_owned(), vec![e, v])];
    for i in 0..N_LAYER {
        for (name, dims) in [
            ("attn_norm", vec![e]),
            ("attn_q", vec![e, q]),
            ("attn_k", vec![e, kv]),
            ("attn_v", vec![e, kv]),
            ("attn_output", vec![q, e]),
            ("ffn_norm", vec![e]),
            ("ffn_gate", vec![e, ff]),
            ("ffn_down", vec![ff, e]),
            ("ffn_up", vec![e, ff]),
        ] {
            t.push((format!("blk.{i}.{name}.weight"), dims));
        }
    }
    t.push(("output_norm.weight".to_owned(), vec![e]));
    t.push(("output.weight".to_owned(), vec![e, v]));
    t
}

/// Purpose: This function generates the F32 values of one test tensor.
///
/// Inputs:
/// - `name`: the tensor's name, which seeds the values.
/// - `n`: how many values.
///
/// Outputs: the values; all ones for a norm tensor.
fn values(name: &str, n: u64) -> Vec<f32> {
    if name.contains("norm") {
        return vec![1.0; n as usize];
    }
    let mut state = name.bytes().fold(0x9E37_79B9_7F4A_7C15_u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3));
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1
        })
        .collect()
}

/// Purpose: This function adds a list of tensors to a GGUF writer as F32.
///
/// Inputs:
/// - `w`: the GGUF writer.
/// - `shapes`: each tensor's name and shape.
///
/// Outputs: none.
fn add_tensors<'a>(w: &mut Writer, shapes: impl IntoIterator<Item = &'a (String, Vec<u64>)>) {
    for (name, dims) in shapes {
        w.f32_tensor(name.as_str(), dims, &values(name, dims.iter().product()));
    }
}

/// Purpose: This function builds the tiny test model as one GGUF file in memory.
///
/// Inputs: none.
///
/// Outputs: a [`Writer`] holding the model.
pub(crate) fn tiny_llama() -> Writer {
    let mut w = Writer::new();
    llama_metadata(&mut w);
    w.kv("general.file_type", Value::U32(0));
    add_tensors(&mut w, &tensor_shapes());
    w
}

/// Purpose: This function writes the one-file tiny test model to disk.
///
/// Inputs:
/// - `path`: the file to write.
///
/// Outputs: none; panics when the file cannot be written.
pub(crate) fn write_tiny_llama(path: &Path) {
    tiny_llama().write(path).unwrap();
}

/// Purpose: This function sets the `split.*` metadata keys on one part of the test model.
///
/// Inputs:
/// - `w`: the part's GGUF writer.
/// - `no`: the part's number, from 0.
/// - `count`: how many parts there are.
/// - `tensors`: the tensor count across all parts.
///
/// Outputs: none.
fn split_keys(w: &mut Writer, no: u16, count: u16, tensors: usize) {
    w.kv("split.no", Value::U16(no))
        .kv("split.count", Value::U16(count))
        .kv("split.tensors.count", Value::I32(i32::try_from(tensors).unwrap()));
}

/// Purpose: This function builds the tiny test model as two split parts in memory.
///
/// Inputs: none.
///
/// Outputs: the two parts' file bytes, part 1 first.
pub(crate) fn split_headers() -> (Vec<u8>, Vec<u8>) {
    let shapes = tensor_shapes();
    let (first, second) = shapes.split_at(shapes.len() / 2);
    let mut p0 = Writer::new();
    llama_metadata(&mut p0);
    p0.kv("general.file_type", Value::U32(0));
    split_keys(&mut p0, 0, 2, shapes.len());
    add_tensors(&mut p0, first);
    let mut p1 = Writer::new();
    split_keys(&mut p1, 1, 2, shapes.len());
    add_tensors(&mut p1, second);
    (p0.to_bytes().unwrap(), p1.to_bytes().unwrap())
}

/// Purpose: This function writes the two-part tiny test model into a directory.
///
/// Inputs:
/// - `dir`: the directory.
/// - `stem`: the model name the part file names start with.
///
/// Outputs: both paths, part 1 first; panics when a file cannot be written.
pub(crate) fn write_tiny_llama_split(dir: &Path, stem: &str) -> Vec<PathBuf> {
    let (p0, p1) = split_headers();
    let paths = vec![dir.join(format!("{stem}-00001-of-00002.gguf")), dir.join(format!("{stem}-00002-of-00002.gguf"))];
    std::fs::write(&paths[0], p0).unwrap();
    std::fs::write(&paths[1], p1).unwrap();
    paths
}

/// Purpose: This function creates a fresh, empty directory for one test under the system
/// temp directory.
///
/// Inputs:
/// - `test`: a short name for the test.
///
/// Outputs: the directory's path.
pub(crate) fn scratch_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("dovata-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn the_fixture_preflights_as_one_model() {
    let m = crate::load::Model::from_headers(&[("m.gguf".into(), tiny_llama().to_bytes().unwrap())]).unwrap();
    assert_eq!(m.architecture(), "llama");
    assert_eq!(m.tensors().count(), tensor_shapes().len());
    let dims = crate::load::ModelDims::from_gguf(m.meta()).unwrap();
    assert_eq!((dims.layers, dims.heads, dims.kv_heads), (N_LAYER, N_HEAD, N_HEAD_KV));
    assert_eq!(dims.key_dim(), HEAD_DIM);

    let (p0, p1) = split_headers();
    let split = crate::load::Model::from_headers(&[("m-00001-of-00002.gguf".into(), p0), ("m-00002-of-00002.gguf".into(), p1)]).unwrap();
    assert_eq!(split.part_count(), 2);
    assert_eq!(split.tensors().count(), tensor_shapes().len());
}
