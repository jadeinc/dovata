//! KV-composition spikes (tensor-level RLM, Phases A + C0).
//!
//! * `selftest_kv_roundtrip` (Phase A): prove a sequence's KV can be serialized to a
//!   CPU buffer (`state_seq_get_data_ext`) and restored (`state_seq_set_data_ext`) so
//!   that decoding continues IDENTICALLY — the block save/restore primitive, no C
//!   surgery.
//! * `selftest_kv_compose` (C0 spike): prove two SEPARATELY-computed blocks can be
//!   placed into ONE sequence's attention window at distinct positions using only
//!   existing ops — `kv_cache_seq_add` (RoPE-correct position shift) + same-stream
//!   `copy_kv_cache_seq` (metadata-only under a UNIFIED KV cache) — then a single
//!   decode that attends over both. If this works, multi-block composition needs no
//!   `is_full` fork surgery.
//!
//! Both are gated by env vars in `run` and reuse the loaded model/context.

use super::build_formatted_prompt;
use llama_cpp_2::{
    context::LlamaContext,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatTemplate, LlamaModel},
    sampling::LlamaSampler,
    token::LlamaToken,
    LlamaStateSeqFlags,
};

/// Serialize sequence `seq_id`'s full KV state to a byte buffer.
fn save_seq(ctx: &LlamaContext, seq_id: i32) -> Vec<u8> {
    let size = ctx.state_seq_get_size_ext(seq_id, LlamaStateSeqFlags::empty());
    let mut buf = vec![0u8; size];
    // SAFETY: buf has `size` bytes, the size the engine reported it needs.
    let n = unsafe { ctx.state_seq_get_data_ext(buf.as_mut_ptr(), seq_id, LlamaStateSeqFlags::empty()) };
    buf.truncate(n);
    buf
}

/// Restore a previously-saved KV state into sequence `seq_id`.
fn load_seq(ctx: &mut LlamaContext, bytes: &[u8], seq_id: i32) -> bool {
    // SAFETY: `bytes` came from save_seq on a compatible context.
    unsafe { ctx.state_seq_set_data_ext(bytes, seq_id, LlamaStateSeqFlags::empty()) }
}

/// Chunked prefill of `tokens` at positions `base_pos..` on `seq`. Logits only on
/// the final token when `logits_last`.
fn prefill(
    ctx: &mut LlamaContext,
    tokens: &[LlamaToken],
    base_pos: i32,
    seq: i32,
    logits_last: bool,
) -> Result<(), String> {
    let n_batch = ctx.n_batch() as usize;
    let mut pos = 0usize;
    while pos < tokens.len() {
        let end = (pos + n_batch).min(tokens.len());
        let mut b = LlamaBatch::new(end - pos, 1);
        for i in pos..end {
            let want = logits_last && i == tokens.len() - 1;
            b.add(tokens[i], base_pos + i as i32, &[seq], want).map_err(|e| format!("prefill add: {e}"))?;
        }
        ctx.decode(&mut b).map_err(|e| format!("prefill decode: {e}"))?;
        pos = end;
    }
    Ok(())
}

/// Decode one token with logits enabled (used to seed generation after a prefill
/// that didn't request logits).
fn feed_token(ctx: &mut LlamaContext, tok: LlamaToken, pos: i32, seq: i32) -> Result<(), String> {
    let mut b = LlamaBatch::new(1, 1);
    b.add(tok, pos, &[seq], true).map_err(|e| format!("feed add: {e}"))?;
    ctx.decode(&mut b).map_err(|e| format!("feed decode: {e}"))
}

/// Greedily generate up to `k` tokens from the CURRENT logits (requires a prior
/// decode that requested logits on its last token). Feeds each sampled token back
/// at `next_pos..` on `seq`.
fn gen_loop(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    seq: i32,
    mut next_pos: i32,
    k: usize,
) -> Result<String, String> {
    let mut sampler = LlamaSampler::greedy();
    let mut decoder = encoding_rs::UTF_8.new_decoder_without_bom_handling();
    let mut batch = LlamaBatch::new(1, 1);
    let mut out = String::new();
    for _ in 0..k {
        let tok = sampler.sample(ctx, -1);
        if model.is_eog_token(tok) {
            break;
        }
        out.push_str(&model.token_to_piece(tok, &mut decoder, true, None).map_err(|e| format!("piece: {e}"))?);
        batch.clear();
        batch.add(tok, next_pos, &[seq], true).map_err(|e| format!("gen add: {e}"))?;
        ctx.decode(&mut batch).map_err(|e| format!("gen decode: {e}"))?;
        next_pos += 1;
    }
    Ok(out)
}

/// Phase A: KV state save → restore round-trip. Prefill a block, save it, generate a
/// reference continuation; then clear, restore the block, generate again — the two
/// continuations must be byte-identical (greedy) iff the KV round-tripped.
pub(crate) fn selftest_kv_roundtrip(model: &LlamaModel, ctx: &mut LlamaContext, tmpl: &LlamaChatTemplate) {
    let mut run = || -> Result<(), String> {
        let prompt = build_formatted_prompt("In one sentence, what is the capital of France?", model, tmpl)?;
        let toks = model.str_to_token(&prompt, AddBos::Always).map_err(|e| format!("tok: {e}"))?;
        if toks.len() < 2 {
            return Err("prompt too short".into());
        }
        let n = toks.len();
        let block = &toks[..n - 1];
        let last = toks[n - 1];
        let last_pos = (n - 1) as i32;

        // Reference: prefill block on seq 0, save it, then continue.
        ctx.clear_kv_cache();
        prefill(ctx, block, 0, 0, false)?;
        let bytes = save_seq(ctx, 0);
        eprintln!("jaded: [kv-roundtrip] saved {} bytes for {} block tokens", bytes.len(), n - 1);
        feed_token(ctx, last, last_pos, 0)?;
        let ref_text = gen_loop(model, ctx, 0, last_pos + 1, 16)?;

        // Restore the block into a clean cache and continue identically.
        ctx.clear_kv_cache();
        let ok = load_seq(ctx, &bytes, 0);
        eprintln!("jaded: [kv-roundtrip] restore ok={ok}");
        feed_token(ctx, last, last_pos, 0)?;
        let test_text = gen_loop(model, ctx, 0, last_pos + 1, 16)?;

        eprintln!("jaded: [kv-roundtrip] ref  = {:?}", ref_text.trim());
        eprintln!("jaded: [kv-roundtrip] test = {:?}", test_text.trim());
        eprintln!("jaded: [kv-roundtrip] MATCH = {}", ref_text.trim() == test_text.trim());
        ctx.clear_kv_cache();
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("jaded: [kv-roundtrip] error: {e}");
    }
}

/// C0 spike: compose two independently-computed blocks into one sequence's attention
/// window via existing ops (no `is_full` surgery), then answer a question that needs
/// BOTH. Requires a unified KV cache (`JADE_KV_UNIFIED=1`) so the seq_cp is a cheap
/// same-stream metadata op. Success = the answer contains both blocks' facts.
pub(crate) fn selftest_kv_compose(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let mut run = || -> Result<(), String> {
        // Two independent facts as raw blocks (no chat template) + a question.
        let a = model.str_to_token("Note A: the secret animal is a platypus.\n", AddBos::Always)
            .map_err(|e| format!("tok a: {e}"))?;
        let b = model.str_to_token("Note B: the secret number is 42.\n", AddBos::Never)
            .map_err(|e| format!("tok b: {e}"))?;
        let q = model.str_to_token(
            "Using the notes above, answer: what is the secret animal and the secret number? Answer: ",
            AddBos::Never,
        ).map_err(|e| format!("tok q: {e}"))?;
        let la = a.len() as i32;
        let lb = b.len() as i32;

        ctx.clear_kv_cache();
        // Block A: native on seq 0 at [0, la).
        prefill(ctx, &a, 0, 0, false)?;
        // Block B: computed IN ISOLATION on seq 1 at [0, lb).
        prefill(ctx, &b, 0, 1, false)?;
        // Relocate B to [la, la+lb) (RoPE-correct), then splice its cells into seq 0
        // (same-stream metadata copy under a unified cache).
        ctx.kv_cache_seq_add(1, Some(0), Some(lb as u32), la).map_err(|e| format!("seq_add: {e}"))?;
        ctx.copy_kv_cache_seq(1, 0, Some(la as u32), Some((la + lb) as u32)).map_err(|e| format!("seq_cp: {e}"))?;
        eprintln!("jaded: [kv-compose] composed block A[0..{la}) + block B[{la}..{}) into seq 0", la + lb);

        // Append the question on seq 0 after both blocks and decode.
        let qbase = la + lb;
        prefill(ctx, &q, qbase, 0, true)?;
        let ans = gen_loop(model, ctx, 0, qbase + q.len() as i32, 40)?;
        let lc = ans.to_lowercase();
        let has_animal = lc.contains("platypus");
        let has_number = lc.contains("42");
        eprintln!("jaded: [kv-compose] ANSWER: {:?}", ans.trim());
        eprintln!("jaded: [kv-compose] platypus={has_animal} 42={has_number} -> {}",
            if has_animal && has_number { "COMPOSE OK (both blocks visible)" } else { "INCOMPLETE" });
        ctx.clear_kv_cache();
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("jaded: [kv-compose] error: {e}");
    }
}
