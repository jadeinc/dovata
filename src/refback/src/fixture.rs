//! Author: Joe Ricks
//! File: src/refback/src/fixture.rs
//! Description: Writes a tiny but complete llama GGUF file, header and tensor data, for
//! refback's tests.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use dovata::gguf::write::Writer;
use dovata::gguf::Value;

use crate::quant::{self, F16, F32, Q4_0, Q8_0};

/// Layers in the fixture.
pub const N_LAYER: u32 = 2;
/// Embedding width.
pub const N_EMBD: u64 = 64;
/// Attention heads.
pub const N_HEAD: u64 = 2;
/// KV heads.
pub const N_HEAD_KV: u64 = 1;
/// FFN width. A multiple of 32, so a row of `ffn_down` is whole Q4_0 and Q8_0 blocks.
pub const N_FF: u64 = 96;
/// Tokens in the vocabulary.
pub const N_VOCAB: u64 = 32;

/// The types the large weights take, in turn.
const ROTATION: [u32; 4] = [Q4_0, Q8_0, F16, F32];

/// Purpose: This struct holds a fixture model file in a temporary directory, which is
/// removed when the fixture is dropped.
#[derive(Debug)]
pub struct Fixture {
    /// The temporary directory.
    pub dir: PathBuf,
    /// The model file in it.
    pub path: PathBuf,
}

impl Fixture {
    /// Purpose: This function writes the llama fixture into a new temporary directory.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the [`Fixture`], or an I/O error's text.
    pub fn llama() -> Result<Fixture, String>
    {
        let dir = temp_dir()?;
        let path = dir.join("refback-llama.gguf");
        let fixture = Fixture { dir, path };
        write_llama(&fixture.path)?;
        Ok(fixture)
    }
}

impl Drop for Fixture {
    /// Purpose: This function removes the fixture's temporary directory.
    ///
    /// Inputs: none.
    ///
    /// Outputs: none.
    fn drop(&mut self)
    {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Purpose: This function makes a new, empty directory under the system's temporary
/// directory.
///
/// Inputs: none.
///
/// Outputs: the directory's path, or an I/O error's text.
fn temp_dir() -> Result<PathBuf, String>
{
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.subsec_nanos());
    loop {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("refback-{}-{nanos}-{n}", std::process::id()));
        match std::fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("{}: {e}", dir.display())),
        }
    }
}

/// Purpose: This function writes the llama fixture model to a file.
///
/// Inputs:
/// - `path`: the file to write.
///
/// Outputs: `Ok(())`, or the writer's error text.
pub fn write_llama(
    path: &Path,
) -> Result<(), String>
{
    let mut w = Writer::new();
    let s = |v: &str| Value::String(v.into());
    w.kv("general.architecture", s("llama"))
        .kv("general.name", s("refback fixture"))
        .kv("llama.block_count", Value::U32(N_LAYER))
        .kv("llama.context_length", Value::U32(256))
        .kv("llama.embedding_length", Value::U32(N_EMBD as u32))
        .kv("llama.feed_forward_length", Value::U32(N_FF as u32))
        .kv("llama.attention.head_count", Value::U32(N_HEAD as u32))
        .kv("llama.attention.head_count_kv", Value::U32(N_HEAD_KV as u32))
        .kv("llama.attention.layer_norm_rms_epsilon", Value::F32(1e-5))
        .kv("llama.rope.freq_base", Value::F32(10_000.0))
        .kv("tokenizer.ggml.model", s("gpt2"))
        .kv("tokenizer.ggml.tokens", Value::Array((0..N_VOCAB).map(|i| s(&format!("t{i}"))).collect()));

    let head = N_EMBD / N_HEAD;
    let kv = head * N_HEAD_KV;
    let mut n = 0usize;
    let mut big = |w: &mut Writer, name: &str, dims: &[u64]| {
        let ty = ROTATION[n % ROTATION.len()];
        w.tensor(name, dims, ty, data(ty, dims, n as u64));
        n += 1;
    };
    let norm = |w: &mut Writer, name: &str, salt: u64| {
        w.tensor(name, &[N_EMBD], F32, data(F32, &[N_EMBD], 1000 + salt));
    };

    big(&mut w, "token_embd.weight", &[N_EMBD, N_VOCAB]);
    norm(&mut w, "output_norm.weight", 0);
    big(&mut w, "output.weight", &[N_EMBD, N_VOCAB]);
    for l in 0..u64::from(N_LAYER) {
        norm(&mut w, &format!("blk.{l}.attn_norm.weight"), 1 + 2 * l);
        big(&mut w, &format!("blk.{l}.attn_q.weight"), &[N_EMBD, head * N_HEAD]);
        big(&mut w, &format!("blk.{l}.attn_k.weight"), &[N_EMBD, kv]);
        big(&mut w, &format!("blk.{l}.attn_v.weight"), &[N_EMBD, kv]);
        big(&mut w, &format!("blk.{l}.attn_output.weight"), &[head * N_HEAD, N_EMBD]);
        norm(&mut w, &format!("blk.{l}.ffn_norm.weight"), 2 + 2 * l);
        big(&mut w, &format!("blk.{l}.ffn_gate.weight"), &[N_EMBD, N_FF]);
        big(&mut w, &format!("blk.{l}.ffn_down.weight"), &[N_FF, N_EMBD]);
        big(&mut w, &format!("blk.{l}.ffn_up.weight"), &[N_EMBD, N_FF]);
    }
    w.write(path).map_err(|e| e.to_string())
}

/// Purpose: This function returns the next value of a deterministic pseudo-random stream.
///
/// Inputs:
/// - `state`: the stream's state, advanced in place.
///
/// Outputs: 64 random bits, from splitmix64.
fn next(
    state: &mut u64,
) -> u64
{
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Purpose: This function makes one tensor's data in its ggml encoding.
///
/// Inputs:
/// - `ty`: the `ggml_type`: F32, F16, Q8_0 or Q4_0.
/// - `dims`: the shape, fastest-varying first.
/// - `salt`: makes each tensor's values different.
///
/// Outputs: the encoded bytes.
fn data(
    ty: u32,
    dims: &[u64],
    salt: u64,
) -> Vec<u8>
{
    let elems: u64 = dims.iter().product();
    let mut st = salt.wrapping_mul(0x0123_4567_89ab_cdef) ^ 0x5eed;
    // A value in [-0.5, 0.5), in steps of 1/1024.
    let value = |st: &mut u64| (next(st) % 1024) as f32 / 1024.0 - 0.5;
    let mut b = Vec::new();
    match ty {
        F32 => (0..elems).for_each(|_| b.extend_from_slice(&value(&mut st).to_le_bytes())),
        F16 => (0..elems).for_each(|_| b.extend_from_slice(&quant::f32_to_f16(value(&mut st)).to_le_bytes())),
        // block_q8_0: an f16 scale, then 32 signed quants in -127..=127.
        Q8_0 => {
            for _ in 0..elems / quant::QK as u64 {
                let d = (1 + next(&mut st) % 8) as f32 / 1024.0;
                b.extend_from_slice(&quant::f32_to_f16(d).to_le_bytes());
                b.extend((0..quant::QK).map(|_| ((next(&mut st) % 255) as i32 - 127) as i8 as u8));
            }
        }
        // block_q4_0: an f16 scale, then 16 bytes of two nibbles each.
        Q4_0 => {
            for _ in 0..elems / quant::QK as u64 {
                let d = (1 + next(&mut st) % 8) as f32 / 128.0;
                b.extend_from_slice(&quant::f32_to_f16(d).to_le_bytes());
                b.extend((0..quant::QK / 2).map(|_| next(&mut st) as u8));
            }
        }
        _ => unreachable!("the fixture writes only F32, F16, Q8_0 and Q4_0"),
    }
    b
}
