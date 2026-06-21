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
use std::collections::{HashMap, HashSet};
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

// ── KV-composition RLM engine (Phases B + C + D) ───────────────────────────────
//
// Full context lives as CPU-RAM token blocks. Per query: retrieve top-k blocks by
// IDF-weighted token overlap (Phase B — non-agentic, zero LLM calls), precompute
// each block's KV on a scratch sequence, splice them into ONE sequence at distinct
// positions (Phase C — `seq_add` + same-stream `seq_cp`, needs `kv_unified`), then
// a single decode answers over the composed window. Optional CacheBlend-style
// in-context recompute of a fraction of each block (Phase D) restores cross-block
// attention. NO agentic PICK/COMBINE calls, NO per-branch re-prompting.

/// One context block held as token ids (the full corpus lives in RAM as these).
pub(crate) struct Block {
    pub(crate) ids: Vec<i32>,
    pub(crate) text: String,
}

/// The corpus: blocks + a built-in IDF retriever.
pub(crate) struct BlockStore {
    pub(crate) blocks: Vec<Block>,
}

impl BlockStore {
    /// Tokenize each text into a block (no BOS — blocks are mid-context fragments).
    pub(crate) fn from_texts(model: &LlamaModel, texts: &[String]) -> Result<Self, String> {
        let mut blocks = Vec::with_capacity(texts.len());
        for t in texts {
            let ids = model
                .str_to_token(t, AddBos::Never)
                .map_err(|e| format!("tokenize block: {e}"))?
                .iter()
                .map(|tok| tok.0)
                .collect();
            blocks.push(Block { ids, text: t.clone() });
        }
        Ok(BlockStore { blocks })
    }

    /// Retrieve the top-`k` block indices for `query_ids` by IDF-weighted shared-token
    /// score (non-agentic; no model call). Falls back to the first `k` if nothing
    /// overlaps. (Lexical for now; a semantic embedder can drop in here later.)
    fn retrieve(&self, query_ids: &[i32], k: usize) -> Vec<usize> {
        let n = self.blocks.len();
        if n == 0 {
            return Vec::new();
        }
        let sets: Vec<HashSet<i32>> = self.blocks.iter().map(|b| b.ids.iter().copied().collect()).collect();
        let mut df: HashMap<i32, usize> = HashMap::new();
        for s in &sets {
            for &t in s {
                *df.entry(t).or_insert(0) += 1;
            }
        }
        let q: HashSet<i32> = query_ids.iter().copied().collect();
        let mut scored: Vec<(f32, usize)> = (0..n)
            .map(|i| {
                let score: f32 = q
                    .iter()
                    .filter(|t| sets[i].contains(t))
                    .map(|t| (n as f32 / *df.get(t).unwrap_or(&1) as f32).ln().max(0.0))
                    .sum();
                (score, i)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let picked: Vec<usize> = scored.iter().filter(|(s, _)| *s > 0.0).take(k).map(|(_, i)| *i).collect();
        if picked.is_empty() {
            (0..n.min(k)).collect()
        } else {
            picked
        }
    }
}

/// Splice one precomputed block (ids) into `seq 0` of `ctx` at `[off, off+len)` using
/// a scratch sequence: prefill in isolation, `seq_add` to the offset (RoPE-correct),
/// same-stream `seq_cp` into seq 0. Requires a unified KV cache. Returns `len`.
fn splice_block(ctx: &mut LlamaContext, ids: &[i32], off: i32, scratch: i32) -> Result<i32, String> {
    let toks: Vec<LlamaToken> = ids.iter().map(|&x| LlamaToken(x)).collect();
    let len = toks.len() as i32;
    prefill(ctx, &toks, 0, scratch, false)?;
    ctx.kv_cache_seq_add(scratch, Some(0), Some(len as u32), off).map_err(|e| format!("seq_add: {e}"))?;
    ctx.copy_kv_cache_seq(scratch, 0, Some(off as u32), Some((off + len) as u32)).map_err(|e| format!("seq_cp: {e}"))?;
    Ok(len)
}

/// Answer `question` over `store` by KV composition. Retrieve top-`k`, compose into
/// one sequence, optionally CacheBlend-recompute the trailing `recompute_frac` of
/// each block in-context, then a single decode. Returns (answer, selected indices).
/// **Requires the context to be created with `kv_unified=true`.**
pub(crate) fn compose_and_answer(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    store: &BlockStore,
    question: &str,
    k: usize,
    recompute_frac: f32,
) -> Result<(String, Vec<usize>), String> {
    // ── Phase B: retrieve (non-agentic) ────────────────────────────────────────
    let q_ids: Vec<i32> = model
        .str_to_token(question, AddBos::Never)
        .map_err(|e| format!("tok query: {e}"))?
        .iter()
        .map(|t| t.0)
        .collect();
    let selected = store.retrieve(&q_ids, k);
    eprintln!("jaded: [compose] retrieved blocks {selected:?} of {}", store.blocks.len());

    // ── Phase C: compose selected blocks into one sequence ──────────────────────
    ctx.clear_kv_cache();
    let prefix = model
        .str_to_token("You are given several notes. Use them to answer the question.\n\n", AddBos::Always)
        .map_err(|e| format!("tok prefix: {e}"))?;
    prefill(ctx, &prefix, 0, 0, false)?;
    let mut off = prefix.len() as i32;
    // record each composed block's [start,end) for the optional recompute pass.
    let mut spans: Vec<(usize, i32, i32)> = Vec::new(); // (block_idx, start, len)
    for (j, &bi) in selected.iter().enumerate() {
        let start = off;
        let len = splice_block(ctx, &store.blocks[bi].ids, off, (j + 1) as i32)?;
        spans.push((bi, start, len));
        off += len;
    }

    // ── Phase D: CacheBlend-lite (EXPERIMENTAL, default OFF). Recompute the trailing
    // fraction of each block IN CONTEXT so those tokens attend to preceding blocks.
    // NOTE: this positional heuristic is NOT real CacheBlend — proper HKVD token
    // selection needs per-layer KV-deviation access (ggml/fork work), and this naive
    // version did not help (even hurt) in testing, so callers pass 0.0. Kept as a knob.
    if recompute_frac > 0.0 {
        for &(bi, start, len) in spans.iter().skip(1) {
            let m = ((len as f32 * recompute_frac).ceil() as i32).clamp(1, len);
            let rstart = start + (len - m); // recompute the last m tokens of the block
            // evict those cells, then re-prefill the tokens at their composed
            // positions — now attending to everything before them in seq 0.
            ctx.clear_kv_cache_seq(Some(0), Some(rstart as u32), Some((start + len) as u32))
                .map_err(|e| format!("blend clear: {e}"))?;
            let toks: Vec<LlamaToken> = store.blocks[bi].ids[(len - m) as usize..].iter().map(|&x| LlamaToken(x)).collect();
            prefill(ctx, &toks, rstart, 0, false)?;
        }
        eprintln!("jaded: [compose] cacheblend recompute ~{:.0}% of {} blocks", recompute_frac * 100.0, spans.len().saturating_sub(1));
    }

    // ── single decode over the composed window ──────────────────────────────────
    let qtext = format!("\nQuestion: {question}\nAnswer: ");
    let qtoks = model.str_to_token(&qtext, AddBos::Never).map_err(|e| format!("tok q: {e}"))?;
    prefill(ctx, &qtoks, off, 0, true)?;
    let ans = gen_loop(model, ctx, 0, off + qtoks.len() as i32, 96)?;
    ctx.clear_kv_cache();
    Ok((ans, selected))
}

/// End-to-end composition demo (gated by `JADE_COMPOSE_DEMO`; needs `JADE_KV_UNIFIED=1`):
/// a needle-in-haystack answered by retrieve → compose → single decode — no agentic
/// calls, no re-prompting. Compares to the agentic baseline conceptually.
pub(crate) fn selftest_compose_demo(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let needle = "XQ-42";
    let n_sections: usize = std::env::var("JADE_RLM_DEMO_SECTIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    let topics = ["Introduction","Hardware","Boot","Networking","Filesystem","Processes","Logging","Power","Updates","Telemetry","Scheduling","Memory","Drivers","Crypto","Backups","Glossary"];
    let filler = "This section documents routine operational details with no special codes or secrets. ".repeat(20);
    let needle_at = n_sections / 2;
    let mut texts: Vec<String> = Vec::new();
    for i in 0..n_sections {
        if i == needle_at {
            texts.push(format!("Section {i}: Access Credentials. The secret access code is {needle}. Rotate it quarterly. {filler}"));
        } else {
            let t = topics[i % topics.len()];
            texts.push(format!("Section {i}: {t}. This section describes {t}. {filler}"));
        }
    }
    let store = match BlockStore::from_texts(model, &texts) {
        Ok(s) => s,
        Err(e) => { eprintln!("jaded: [compose-demo] build error: {e}"); return; }
    };
    let question = "What is the secret access code?";
    eprintln!("jaded: [compose-demo] {} blocks, question: {question:?}", store.blocks.len());
    let t = std::time::Instant::now();
    match compose_and_answer(model, ctx, &store, question, 4, 0.0) {
        Ok((ans, sel)) => {
            let ms = t.elapsed().as_millis();
            eprintln!("jaded: [compose-demo] selected {sel:?}");
            eprintln!("jaded: [compose-demo] ANSWER: {:?}", ans.trim());
            eprintln!("jaded: [compose-demo] needle {needle} found: {}  ({ms}ms, ONE decode, zero agentic calls)", ans.contains(needle));
        }
        Err(e) => eprintln!("jaded: [compose-demo] error: {e}"),
    }
}

/// Multi-block + CacheBlend test (gated by `JADE_COMPOSE_MULTI`; needs `JADE_KV_UNIFIED=1`):
/// the answer requires TWO different blocks, so composition must place both in the
/// window. Runs once WITHOUT CacheBlend recompute and once WITH, to show the fidelity
/// knob. Success = both facts present in the answer.
pub(crate) fn selftest_compose_multi(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let filler = "This section has routine details and no codes. ".repeat(20);
    let mut texts: Vec<String> = Vec::new();
    for i in 0..20 {
        texts.push(match i {
            7 => format!("Section 7: Launch. The launch code is ALPHA-9. {filler}"),
            13 => format!("Section 13: Abort. The abort code is OMEGA-3. {filler}"),
            _ => format!("Section {i}: Misc. {filler}"),
        });
    }
    let store = match BlockStore::from_texts(model, &texts) {
        Ok(s) => s,
        Err(e) => { eprintln!("jaded: [compose-multi] build error: {e}"); return; }
    };
    let question = "What are the launch code and the abort code?";
    for &frac in &[0.0f32, 0.15] {
        match compose_and_answer(model, ctx, &store, question, 4, frac) {
            Ok((ans, sel)) => {
                let lc = ans.to_lowercase();
                let both = lc.contains("alpha-9") && lc.contains("omega-3");
                eprintln!("jaded: [compose-multi] recompute={frac} selected={sel:?} both={both}");
                eprintln!("jaded: [compose-multi]   ANSWER: {:?}", ans.trim());
            }
            Err(e) => eprintln!("jaded: [compose-multi] error (frac={frac}): {e}"),
        }
    }
}
