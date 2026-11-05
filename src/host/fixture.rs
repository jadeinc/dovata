//! Author: Joe Ricks
//! File: src/host/fixture.rs
//! Description: Writes a tiny llama GGUF with f32 weights, and a `tokenizer.json` with
//! the same vocabulary beside it, into a temporary directory, for the host's tests. The library's tests and the process test both include it, each
//! giving it the GGUF writer as `super::Writer` and `super::Value`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use super::{Value, Writer};

/// Embedding width.
const N_EMBD: u64 = 64;
/// FFN width.
const N_FF: u64 = 96;
/// Tokens in the vocabulary.
const N_VOCAB: u64 = 32;

/// Purpose: This struct holds a temporary directory, removed when it is dropped.
#[derive(Debug)]
pub struct TempDir {
    /// The directory.
    pub path: PathBuf,
}

impl TempDir {
    /// Purpose: This function makes a new, empty temporary directory with a short path,
    /// so a socket inside it fits a socket path's length limit.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the directory.
    pub fn new() -> TempDir
    {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        loop {
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("dvh-{}-{n}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return TempDir { path },
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("{}: {e}", path.display()),
            }
        }
    }
}

impl Drop for TempDir {
    /// Purpose: This function removes the directory and everything in it.
    ///
    /// Inputs: none.
    ///
    /// Outputs: none.
    fn drop(&mut self)
    {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Purpose: This function writes the fixture model: two llama layers, 64 wide, every
/// weight f32 with values that differ from tensor to tensor, and a word-level
/// `tokenizer.json` over the same 32 tokens beside it, where dovata looks first.
///
/// Inputs:
/// - `path`: the file to write.
///
/// Outputs: none; it panics when a file cannot be written.
pub fn write_llama(
    path: &Path,
)
{
    let mut w = Writer::new();
    let s = |v: &str| Value::String(v.into());
    w.kv("general.architecture", s("llama"))
        .kv("general.name", s("host fixture"))
        .kv("llama.block_count", Value::U32(2))
        .kv("llama.context_length", Value::U32(256))
        .kv("llama.embedding_length", Value::U32(N_EMBD as u32))
        .kv("llama.feed_forward_length", Value::U32(N_FF as u32))
        .kv("llama.attention.head_count", Value::U32(2))
        .kv("llama.attention.head_count_kv", Value::U32(1))
        .kv("llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5))
        .kv("llama.rope.freq_base", Value::F32(10_000.0))
        .kv("tokenizer.ggml.model", s("gpt2"))
        .kv("tokenizer.ggml.tokens", Value::Array((0..N_VOCAB).map(|i| s(&format!("t{i}"))).collect()));
    let head = N_EMBD / 2;
    let mut salt = 0u64;
    let mut t = |w: &mut Writer, name: &str, dims: &[u64]| {
        salt += 1;
        let n: u64 = dims.iter().product();
        let values: Vec<f32> = (0..n).map(|i| ((i * 7 + salt * 13) % 251) as f32 / 251.0 - 0.5).collect();
        w.f32_tensor(name, dims, &values);
    };
    t(&mut w, "token_embd.weight", &[N_EMBD, N_VOCAB]);
    t(&mut w, "output_norm.weight", &[N_EMBD]);
    t(&mut w, "output.weight", &[N_EMBD, N_VOCAB]);
    for l in 0..2 {
        t(&mut w, &format!("blk.{l}.attn_norm.weight"), &[N_EMBD]);
        t(&mut w, &format!("blk.{l}.attn_q.weight"), &[N_EMBD, N_EMBD]);
        t(&mut w, &format!("blk.{l}.attn_k.weight"), &[N_EMBD, head]);
        t(&mut w, &format!("blk.{l}.attn_v.weight"), &[N_EMBD, head]);
        t(&mut w, &format!("blk.{l}.attn_output.weight"), &[N_EMBD, N_EMBD]);
        t(&mut w, &format!("blk.{l}.ffn_norm.weight"), &[N_EMBD]);
        t(&mut w, &format!("blk.{l}.ffn_gate.weight"), &[N_EMBD, N_FF]);
        t(&mut w, &format!("blk.{l}.ffn_down.weight"), &[N_FF, N_EMBD]);
        t(&mut w, &format!("blk.{l}.ffn_up.weight"), &[N_EMBD, N_FF]);
    }
    w.write(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let vocab: serde_json::Map<String, serde_json::Value> = (0..N_VOCAB).map(|i| (format!("t{i}"), serde_json::json!(i))).collect();
    let tokenizer = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
        "pre_tokenizer": { "type": "WhitespaceSplit" }, "post_processor": null, "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "t0" }
    });
    let json_path = path.with_file_name("tokenizer.json");
    std::fs::write(&json_path, tokenizer.to_string()).unwrap_or_else(|e| panic!("{}: {e}", json_path.display()));
}
