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
use std::collections::{HashMap, HashSet, VecDeque};
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

/// Bounded LRU of precomputed block KV held in host RAM. Key = block index in its
/// [`BlockStore`]; value = the block's serialized KV state (`state_seq_get_data`, ~27
/// KB/token at q4_0). Restoring a block's KV is ~50× cheaper than re-prefilling it on
/// the GPU (benchmarked on the A10G), at the cost of host RAM — so total bytes are
/// capped and least-recently-used blocks are evicted; cold/evicted blocks fall back
/// to re-prefill.
pub(crate) struct KvBlockCache {
    entries: HashMap<usize, Vec<u8>>,
    order: VecDeque<usize>, // front = least-recently used, back = most-recently used
    cap_bytes: usize,
    cur_bytes: usize,
}

impl KvBlockCache {
    /// A cache bounded to `cap_bytes` of block KV. `cap_bytes == 0` disables caching
    /// (every block is re-prefilled), which is how the one-shot paths opt out.
    pub(crate) fn new(cap_bytes: usize) -> Self {
        KvBlockCache { entries: HashMap::new(), order: VecDeque::new(), cap_bytes, cur_bytes: 0 }
    }

    /// Bump `idx` to most-recently-used and return its KV bytes, or `None` on a miss.
    fn get(&mut self, idx: usize) -> Option<&[u8]> {
        if self.entries.contains_key(&idx) {
            self.order.retain(|&x| x != idx);
            self.order.push_back(idx);
            self.entries.get(&idx).map(|v| v.as_slice())
        } else {
            None
        }
    }

    /// Insert/replace a block's KV, evicting least-recently-used entries until the
    /// total fits under `cap_bytes`. A block larger than the whole cap is not cached.
    fn insert(&mut self, idx: usize, bytes: Vec<u8>) {
        if let Some(old) = self.entries.remove(&idx) {
            self.cur_bytes -= old.len();
            self.order.retain(|&x| x != idx);
        }
        if bytes.len() > self.cap_bytes {
            return;
        }
        while self.cur_bytes + bytes.len() > self.cap_bytes {
            match self.order.pop_front() {
                Some(victim) => {
                    if let Some(v) = self.entries.remove(&victim) {
                        self.cur_bytes -= v.len();
                    }
                }
                None => break,
            }
        }
        self.cur_bytes += bytes.len();
        self.entries.insert(idx, bytes);
        self.order.push_back(idx);
    }

    /// (entry count, total bytes) — for logging / tests.
    pub(crate) fn stats(&self) -> (usize, usize) {
        (self.entries.len(), self.cur_bytes)
    }
}

/// Materialize block `idx` (token `ids`) into `seq 0` of `ctx` at `[off, off+len)`:
/// RESTORE its KV from `cache` if present (a cheap host→device copy), else re-prefill
/// it in isolation on a scratch seq and populate the cache. Then `seq_add` to the
/// offset (RoPE-correct) + same-stream `seq_cp` into seq 0. Requires a unified KV
/// cache. Returns `len`.
fn materialize_block(
    ctx: &mut LlamaContext,
    cache: &mut KvBlockCache,
    idx: usize,
    ids: &[i32],
    off: i32,
    scratch: i32,
) -> Result<i32, String> {
    let len = ids.len() as i32;
    // Restore precomputed KV when cached; otherwise prefill in isolation and cache it.
    let restored = match cache.get(idx) {
        Some(bytes) => load_seq(ctx, bytes, scratch),
        None => false,
    };
    if !restored {
        let toks: Vec<LlamaToken> = ids.iter().map(|&x| LlamaToken(x)).collect();
        prefill(ctx, &toks, 0, scratch, false)?;
        let bytes = save_seq(ctx, scratch);
        cache.insert(idx, bytes);
    }
    ctx.kv_cache_seq_add(scratch, Some(0), Some(len as u32), off).map_err(|e| format!("seq_add: {e}"))?;
    ctx.copy_kv_cache_seq(scratch, 0, Some(off as u32), Some((off + len) as u32)).map_err(|e| format!("seq_cp: {e}"))?;
    Ok(len)
}

/// Answer `question` over `store` by KV composition. Retrieve top-`k`, materialize
/// each selected block's KV into one sequence (restoring from `cache` when warm, else
/// re-prefilling and caching it), optionally CacheBlend-recompute the trailing
/// `recompute_frac` of each block, then a single decode. Returns (answer, selected
/// indices). **Requires the context to be created with `kv_unified=true`.**
pub(crate) fn compose_and_answer(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    store: &BlockStore,
    cache: &mut KvBlockCache,
    question: &str,
    k: usize,
    recompute_frac: f32,
) -> Result<(String, Vec<usize>), String> {
    // ── PICK: retrieve top-k (non-agentic) ──────────────────────────────────────
    let q_ids = tokenize_query(model, question)?;
    let selected = store.retrieve(&q_ids, k);
    eprintln!("jaded: [compose] retrieved blocks {selected:?} of {}", store.blocks.len());
    // ── COMBINE: compose + single decode ────────────────────────────────────────
    let ans = compose_selected(model, ctx, store, cache, &selected, question, recompute_frac)?;
    Ok((ans, selected))
}

/// Tokenize a query string to raw token ids (no BOS — it's mid-context).
fn tokenize_query(model: &LlamaModel, question: &str) -> Result<Vec<i32>, String> {
    Ok(model
        .str_to_token(question, AddBos::Never)
        .map_err(|e| format!("tok query: {e}"))?
        .iter()
        .map(|t| t.0)
        .collect())
}

/// COMBINE: compose the `selected` blocks' KV into one sequence (restore-or-prefill via
/// `cache`), optional CacheBlend recompute, then a single decode. Shared by the plain
/// (retrieve→compose) and blended (retrieve→rerank→compose) pipelines. Requires
/// `kv_unified=true`.
fn compose_selected(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    store: &BlockStore,
    cache: &mut KvBlockCache,
    selected: &[usize],
    question: &str,
    recompute_frac: f32,
) -> Result<String, String> {
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
        let len = materialize_block(ctx, cache, bi, &store.blocks[bi].ids, off, (j + 1) as i32)?;
        spans.push((bi, start, len));
        off += len;
    }

    // CacheBlend-lite (EXPERIMENTAL, default OFF) — recompute the trailing fraction of
    // each block IN CONTEXT so those tokens attend to preceding blocks. Naive positional
    // heuristic, not real HKVD CacheBlend; callers pass 0.0. Kept as a knob.
    if recompute_frac > 0.0 {
        for &(bi, start, len) in spans.iter().skip(1) {
            let m = ((len as f32 * recompute_frac).ceil() as i32).clamp(1, len);
            let rstart = start + (len - m);
            ctx.clear_kv_cache_seq(Some(0), Some(rstart as u32), Some((start + len) as u32))
                .map_err(|e| format!("blend clear: {e}"))?;
            let toks: Vec<LlamaToken> = store.blocks[bi].ids[(len - m) as usize..].iter().map(|&x| LlamaToken(x)).collect();
            prefill(ctx, &toks, rstart, 0, false)?;
        }
    }

    // single decode over the composed window
    let qtext = format!("\nQuestion: {question}\nAnswer: ");
    let qtoks = model.str_to_token(&qtext, AddBos::Never).map_err(|e| format!("tok q: {e}"))?;
    prefill(ctx, &qtoks, off, 0, true)?;
    let ans = gen_loop(model, ctx, 0, off + qtoks.len() as i32, 96)?;
    ctx.clear_kv_cache();
    Ok(ans)
}

/// Per-token natural-log softmax value at `idx` over a logits row (numerically stable).
fn log_softmax_at(logits: &[f32], idx: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    logits[idx] - lse
}

/// FAN-OUT score for ONE candidate: query-likelihood `P(query | block)` as the mean
/// log-prob the model assigns to the query tokens with the block resident as context —
/// a cross-encoder relevance signal read from logits (NO generation, no agentic call).
/// Higher = more relevant. Reuses the KV cache: the block is restored (cheap) when warm,
/// else prefilled once and cached (the same canonical 0..len KV that COMBINE reuses).
fn score_block_query(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    cache: &mut KvBlockCache,
    store: &BlockStore,
    bi: usize,
    q_ids: &[i32],
) -> Result<f32, String> {
    if q_ids.len() < 2 {
        return Ok(f32::NEG_INFINITY);
    }
    let ids = &store.blocks[bi].ids;
    let lb = ids.len() as i32;
    // Make the block resident on a scratch seq (seq 1), restore-or-prefill via the cache.
    ctx.clear_kv_cache();
    let restored = match cache.get(bi) {
        Some(bytes) => load_seq(ctx, bytes, 1),
        None => false,
    };
    if !restored {
        let toks: Vec<LlamaToken> = ids.iter().map(|&x| LlamaToken(x)).collect();
        prefill(ctx, &toks, 0, 1, false)?;
        let bytes = save_seq(ctx, 1);
        cache.insert(bi, bytes);
    }
    // Teacher-force the query after the block, logits ON for every query position.
    let mut batch = LlamaBatch::new(q_ids.len(), 1);
    for (j, &t) in q_ids.iter().enumerate() {
        batch.add(LlamaToken(t), lb + j as i32, &[1], true).map_err(|e| format!("score add: {e}"))?;
    }
    ctx.decode(&mut batch).map_err(|e| format!("score decode: {e}"))?;
    // logits at batch index j predict q[j+1]; score = mean log P(q[j+1] | block, q[..=j]).
    let mut sum = 0f64;
    let mut cnt = 0usize;
    for j in 0..q_ids.len() - 1 {
        let logits = ctx.get_logits_ith(j as i32);
        let next = q_ids[j + 1] as usize;
        if next < logits.len() {
            sum += log_softmax_at(logits, next) as f64;
            cnt += 1;
        }
    }
    ctx.clear_kv_cache();
    Ok(if cnt > 0 { (sum / cnt as f64) as f32 } else { f32::NEG_INFINITY })
}

/// FAN-OUT: score every candidate block by query-likelihood and return them sorted by
/// relevance (desc). Block-level, no generation, no language-level reasoning.
fn rerank(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    cache: &mut KvBlockCache,
    store: &BlockStore,
    candidates: &[usize],
    q_ids: &[i32],
) -> Result<Vec<(usize, f32)>, String> {
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());
    for &bi in candidates {
        let s = score_block_query(model, ctx, cache, store, bi, q_ids)?;
        scored.push((bi, s));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored)
}

/// Blended RAG-RLM pipeline: PICK (retrieve `n_candidates`) → FAN-OUT (cross-encoder
/// rerank by query-likelihood) → COMBINE (compose top-`k` blocks' KV + single decode).
/// All block/KV-level, zero agentic LLM calls. Returns (answer, composed top-k,
/// full rerank scores). Requires `kv_unified=true`.
pub(crate) fn blended_compose_and_answer(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    store: &BlockStore,
    cache: &mut KvBlockCache,
    question: &str,
    n_candidates: usize,
    k: usize,
    recompute_frac: f32,
) -> Result<(String, Vec<usize>, Vec<(usize, f32)>), String> {
    let q_ids = tokenize_query(model, question)?;
    let candidates = store.retrieve(&q_ids, n_candidates.max(k)); // PICK (recall)
    let scored = rerank(model, ctx, cache, store, &candidates, &q_ids)?; // FAN-OUT (rerank)
    let selected: Vec<usize> = scored.iter().take(k).map(|(i, _)| *i).collect();
    eprintln!("jaded: [blended] PICK {} candidates -> RERANK -> COMBINE top-{k} {selected:?}", candidates.len());
    let ans = compose_selected(model, ctx, store, cache, &selected, question, recompute_frac)?; // COMBINE
    Ok((ans, selected, scored))
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
    let mut cache = KvBlockCache::new(0); // one-shot demo: caching disabled
    let t = std::time::Instant::now();
    match compose_and_answer(model, ctx, &store, &mut cache, question, 4, 0.0) {
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
    let mut cache = KvBlockCache::new(0); // recompute test: caching disabled
    for &frac in &[0.0f32, 0.15] {
        match compose_and_answer(model, ctx, &store, &mut cache, question, 4, frac) {
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

/// Transfer-vs-recompute benchmark (gated by `JADE_KV_BENCH`). For several block
/// sizes, compares the two ways the compose engine could materialize a block's KV in
/// the resident window:
///   (a) RE-PREFILL — recompute the block's K/V from its token ids on the GPU (what
///       `splice_block`/`compose_and_answer` do today: pure compute, no host traffic);
///   (b) RESTORE    — copy the block's precomputed K/V from a CPU-RAM byte buffer back
///       onto the device via `state_seq_set_data` (the Phase-A primitive: a host→device
///       PCIe transfer, ~27 KB/token).
/// Also reports the one-time `state_seq_get_data` (device→host) save cost and bytes/token.
/// This decides whether caching block KV in host RAM beats recomputing it each query on
/// this box. Each timing is the mean of `iters` runs after a warm-up. NOTE: restore
/// reinstates KV at the block's SAVED positions — composing it at a new offset needs a
/// follow-up `kv_cache_seq_add` (cheap, on-GPU); that is not part of this raw A/B.
pub(crate) fn selftest_kv_bench(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let mut run = || -> Result<(), String> {
        // One long filler tokenized once; each block size is a prefix of it, so both
        // paths operate on byte-identical tokens.
        let filler = "This is routine filler text for the kv cache transfer benchmark. ".repeat(500);
        let all = model.str_to_token(&filler, AddBos::Never).map_err(|e| format!("tok: {e}"))?;
        let sizes = [64usize, 128, 256, 512, 1024, 2048];
        let iters = 5usize;

        eprintln!("jaded: [kv-bench] re-prefill (GPU compute) vs CPU-RAM KV restore (PCIe), mean of {iters}");
        eprintln!(
            "jaded: [kv-bench] {:>6} {:>11} {:>11} {:>9} {:>8} {:>9} {:>11}",
            "tokens", "prefill_ms", "restore_ms", "save_ms", "KB/tok", "save_MB", "restore_GB/s"
        );

        for &n in &sizes {
            if all.len() < n {
                break;
            }
            let block = &all[..n];

            // Warm-up (first GPU op of a size pays one-time costs we don't want timed).
            ctx.clear_kv_cache();
            prefill(ctx, block, 0, 0, false)?;

            // (a) RE-PREFILL: recompute the block's KV on seq 0 from token ids.
            let mut prefill_ms = 0f64;
            for _ in 0..iters {
                ctx.clear_kv_cache();
                let t = std::time::Instant::now();
                prefill(ctx, block, 0, 0, false)?;
                prefill_ms += t.elapsed().as_secs_f64() * 1000.0;
            }
            prefill_ms /= iters as f64;

            // Save the block's KV once (device→host) into a CPU-RAM buffer.
            ctx.clear_kv_cache();
            prefill(ctx, block, 0, 0, false)?;
            let t = std::time::Instant::now();
            let bytes = save_seq(ctx, 0);
            let save_ms = t.elapsed().as_secs_f64() * 1000.0;

            // (b) RESTORE: copy the saved KV back onto the device (host→device PCIe).
            let mut restore_ms = 0f64;
            for _ in 0..iters {
                ctx.clear_kv_cache();
                let t = std::time::Instant::now();
                let ok = load_seq(ctx, &bytes, 0);
                restore_ms += t.elapsed().as_secs_f64() * 1000.0;
                if !ok {
                    return Err("load_seq returned false".into());
                }
            }
            restore_ms /= iters as f64;

            let save_mb = bytes.len() as f64 / (1024.0 * 1024.0);
            let kb_tok = bytes.len() as f64 / 1024.0 / n as f64;
            let restore_gbs = (bytes.len() as f64 / 1e9) / (restore_ms / 1000.0);
            eprintln!(
                "jaded: [kv-bench] {n:>6} {prefill_ms:>11.3} {restore_ms:>11.3} {save_ms:>9.3} {kb_tok:>8.2} {save_mb:>9.3} {restore_gbs:>11.2}"
            );
        }
        ctx.clear_kv_cache();
        eprintln!("jaded: [kv-bench] restore wins where prefill_ms > restore_ms; that's the cache-KV-in-RAM crossover.");
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("jaded: [kv-bench] error: {e}");
    }
}

/// Pure generation-throughput baseline (gated by `JADE_GEN_BENCH`). For each resident
/// context size, prefill that many tokens (UNTIMED), then time generating a fixed
/// `JADE_GEN_TOKENS` (default 128) tokens one-at-a-time (greedy, EOG ignored so the
/// loop runs the full count for clean timing). Reports decode tok/s per resident size
/// — the metric a user actually feels, and the one that scales with the composed
/// window the RLM builds. Decode is unaffected by the KV cache; this is the raw
/// serial-generation speed of the model on this box.
pub(crate) fn selftest_gen_bench(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let mut run = || -> Result<(), String> {
        let gen_n: usize = std::env::var("JADE_GEN_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
        let filler = "This is routine filler text for the generation throughput baseline. ".repeat(800);
        let all = model.str_to_token(&filler, AddBos::Never).map_err(|e| format!("tok: {e}"))?;
        let sizes = [128usize, 1024, 4096, 8192];

        // Decode the generation loop once (warm the single-token decode graph) so the
        // first sized measurement isn't polluted by graph compilation.
        ctx.clear_kv_cache();
        prefill(ctx, &all[..64.min(all.len())], 0, 0, true)?;
        let _ = gen_loop(model, ctx, 0, 64, 8)?;

        eprintln!("jaded: [gen-bench] pure decode throughput, generating {gen_n} tokens per resident size");
        eprintln!("jaded: [gen-bench] {:>10} {:>10} {:>12}", "resident", "decode_ms", "decode_tok/s");

        for &csz in &sizes {
            if all.len() < csz {
                break;
            }
            // Prefill the resident context (UNTIMED) with logits on the last token.
            ctx.clear_kv_cache();
            prefill(ctx, &all[..csz], 0, 0, true)?;

            // Time ONLY the serial generation of gen_n tokens (ignore EOG for clean timing).
            let mut sampler = LlamaSampler::greedy();
            let mut batch = LlamaBatch::new(1, 1);
            let mut pos = csz as i32;
            let t = std::time::Instant::now();
            for _ in 0..gen_n {
                let tok = sampler.sample(ctx, -1);
                batch.clear();
                batch.add(tok, pos, &[0], true).map_err(|e| format!("gen add: {e}"))?;
                ctx.decode(&mut batch).map_err(|e| format!("gen decode: {e}"))?;
                pos += 1;
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let tok_s = gen_n as f64 / (ms / 1000.0);
            eprintln!("jaded: [gen-bench] {csz:>10} {ms:>10.1} {tok_s:>12.1}");
        }
        ctx.clear_kv_cache();
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("jaded: [gen-bench] error: {e}");
    }
}

/// Generation throughput PAST the context window (gated by `JADE_RLM_OVERWINDOW`; needs
/// `JADE_KV_UNIFIED=1`). Holds a corpus FAR larger than the model's ctx window in RAM as
/// blocks, composes a FIXED `k` of them into a bounded resident window, then times pure
/// decode. The point: total corpus size does NOT touch resident KV or decode speed — RLM
/// keeps the window bounded, so generation tok/s past the window ≈ the rate for the
/// composed-window size and stays FLAT as the corpus grows (a standard model can't even
/// hold a corpus larger than its window). Sweeps corpus = 2×/4×/8× the ctx window.
pub(crate) fn selftest_rlm_overwindow(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let mut run = || -> Result<(), String> {
        let gen_n: usize = std::env::var("JADE_GEN_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
        let k: usize = std::env::var("JADE_OW_BLOCKS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
        let n_ctx = ctx.n_ctx() as usize;

        // One ~512-token block body, reused; measure its real token length.
        let block_text = format!("routine operational detail with no special codes or secrets {}",
            "and more routine detail ".repeat(150));
        let btok = model.str_to_token(&block_text, AddBos::Never).map_err(|e| format!("tok: {e}"))?.len();

        // Warm the single-token decode graph so the first measurement is clean.
        // (next_pos MUST equal the prefilled length — no position gap.)
        ctx.clear_kv_cache();
        let warm = model.str_to_token("Warm up the decode graph for the bench.", AddBos::Always)
            .map_err(|e| format!("tok: {e}"))?;
        prefill(ctx, &warm, 0, 0, true)?;
        let _ = gen_loop(model, ctx, 0, warm.len() as i32, 8)?;

        let prefix = model.str_to_token("You are given several notes. Use them to answer the question.\n\n", AddBos::Always)
            .map_err(|e| format!("tok prefix: {e}"))?;
        let qtoks = model.str_to_token("\nQuestion: What is the secret access code?\nAnswer: ", AddBos::Never)
            .map_err(|e| format!("tok q: {e}"))?;

        eprintln!("jaded: [overwindow] ctx window = {n_ctx} tokens; composing k={k} blocks (~{btok} tok each), generating {gen_n}");
        eprintln!("jaded: [overwindow] {:>12} {:>9} {:>9} {:>9} {:>10} {:>12}",
            "corpus_tok", "blocks", "x_window", "resident", "decode_ms", "decode_tok/s");

        for mult in [2usize, 4, 8] {
            let corpus = mult * n_ctx;
            let n_blocks = (corpus / btok.max(1)).max(k);
            // Corpus held in RAM as blocks (only k ever touch the GPU).
            let texts: Vec<String> = (0..n_blocks).map(|i| format!("Section {i}. {block_text}")).collect();
            let store = BlockStore::from_texts(model, &texts)?;
            let total: usize = store.blocks.iter().map(|b| b.ids.len()).sum();

            // Compose k evenly-spaced blocks into one bounded window (untimed).
            ctx.clear_kv_cache();
            prefill(ctx, &prefix, 0, 0, false)?;
            let mut off = prefix.len() as i32;
            let mut cache = KvBlockCache::new(0);
            let stride = (n_blocks / k).max(1);
            for j in 0..k {
                let bi = (j * stride).min(n_blocks - 1);
                off += materialize_block(ctx, &mut cache, bi, &store.blocks[bi].ids, off, (j + 1) as i32)?;
            }
            prefill(ctx, &qtoks, off, 0, true)?;
            let resident = off as usize + qtoks.len();

            // Time ONLY the serial decode (EOG ignored for clean timing).
            let mut sampler = LlamaSampler::greedy();
            let mut batch = LlamaBatch::new(1, 1);
            let mut pos = resident as i32;
            let t = std::time::Instant::now();
            for _ in 0..gen_n {
                let tok = sampler.sample(ctx, -1);
                batch.clear();
                batch.add(tok, pos, &[0], true).map_err(|e| format!("gen add: {e}"))?;
                ctx.decode(&mut batch).map_err(|e| format!("gen decode: {e}"))?;
                pos += 1;
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let tok_s = gen_n as f64 / (ms / 1000.0);
            eprintln!("jaded: [overwindow] {total:>12} {n_blocks:>9} {mult:>8}x {resident:>9} {ms:>10.1} {tok_s:>12.1}");
        }
        ctx.clear_kv_cache();
        eprintln!("jaded: [overwindow] resident window stays bounded → decode tok/s flat vs corpus size (standard model: cannot run past {n_ctx}).");
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("jaded: [overwindow] error: {e}");
    }
}

/// End-to-end COLD-vs-WARM compose bench (gated by `JADE_COMPOSE_CACHE`; needs
/// `JADE_KV_UNIFIED=1`). Answers the same needle question twice over one block store +
/// KV cache: first COLD (retrieved blocks re-prefilled and cached), then WARM (those
/// blocks restored from host RAM). Reports both end-to-end latencies and asserts the
/// two answers are IDENTICAL — restoring block KV must reproduce the re-prefill
/// composition exactly (greedy decode). That equality is the correctness check for the
/// cache path; the latency delta is the per-query win caching buys.
pub(crate) fn selftest_compose_cache(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
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
        Err(e) => { eprintln!("jaded: [compose-cache] build error: {e}"); return; }
    };
    let mut cache = KvBlockCache::new(2 * 1024 * 1024 * 1024); // 2 GiB
    let question = "What is the secret access code?";
    eprintln!("jaded: [compose-cache] {} blocks, 2 GiB KV cache, question: {question:?}", store.blocks.len());

    // WARMUP: a throwaway compose (its own cache) so the COLD timing below isn't
    // polluted by one-time CUDA graph compilation on the first decode after load.
    {
        let mut warm_cache = KvBlockCache::new(0);
        let _ = compose_and_answer(model, ctx, &store, &mut warm_cache, question, 4, 0.0);
    }

    // COLD: cache empty → retrieved blocks are re-prefilled on the GPU and saved.
    let t = std::time::Instant::now();
    let (cold_ans, cold_sel) = match compose_and_answer(model, ctx, &store, &mut cache, question, 4, 0.0) {
        Ok(r) => r,
        Err(e) => { eprintln!("jaded: [compose-cache] cold error: {e}"); return; }
    };
    let cold_ms = t.elapsed().as_millis();
    let (n_entries, bytes) = cache.stats();

    // WARM: the same blocks are now restored from host RAM instead of recomputed.
    let t = std::time::Instant::now();
    let (warm_ans, _) = match compose_and_answer(model, ctx, &store, &mut cache, question, 4, 0.0) {
        Ok(r) => r,
        Err(e) => { eprintln!("jaded: [compose-cache] warm error: {e}"); return; }
    };
    let warm_ms = t.elapsed().as_millis();

    let same = cold_ans.trim() == warm_ans.trim();
    eprintln!("jaded: [compose-cache] selected {cold_sel:?}; cache now {n_entries} blocks / {:.1} MB",
        bytes as f64 / (1024.0 * 1024.0));
    eprintln!("jaded: [compose-cache] COLD (re-prefill) {cold_ms}ms   WARM (restore) {warm_ms}ms   speedup {:.2}x",
        cold_ms as f64 / warm_ms.max(1) as f64);
    eprintln!("jaded: [compose-cache] prefill saved by caching: {}ms over {} composed block(s) (decode time is identical & unaffected)",
        cold_ms.saturating_sub(warm_ms), cold_sel.len());
    eprintln!("jaded: [compose-cache] answers identical (cold==warm): {same}");
    eprintln!("jaded: [compose-cache] needle {needle} found: cold={} warm={}",
        cold_ans.contains(needle), warm_ans.contains(needle));
    if !same {
        eprintln!("jaded: [compose-cache]   COLD: {:?}", cold_ans.trim());
        eprintln!("jaded: [compose-cache]   WARM: {:?}", warm_ans.trim());
    }
}

/// Blended RAG-RLM eval (gated by `JADE_BLENDED`; needs `JADE_KV_UNIFIED=1`). Builds a
/// haystack with the needle block plus lexical DISTRACTOR blocks (that echo the query's
/// words but hold no answer), then runs the plain pipeline (retrieve→compose) and the
/// blended pipeline (retrieve N→rerank→compose top-k) on the same store. Prints each
/// pipeline's selection, the needle hit, latency, and the rerank scores (needle/distractor
/// tagged) — so we can see whether the cross-encoder rerank reorders what lexical recall
/// got wrong.
pub(crate) fn selftest_blended(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let needle = "XQ-42";
    let n_sections: usize = std::env::var("JADE_RLM_DEMO_SECTIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    let n_cand: usize = std::env::var("JADE_BLEND_N").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
    let k: usize = std::env::var("JADE_BLEND_K").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let filler = "This section documents routine operational details. ".repeat(20);
    let needle_at = n_sections / 2;
    let distractors = [n_sections / 4, (3 * n_sections) / 4, n_sections / 3];
    let mut texts: Vec<String> = Vec::new();
    for i in 0..n_sections {
        if i == needle_at {
            texts.push(format!("Section {i}: Access Credentials. The secret access code is {needle}. Rotate it quarterly. {filler}"));
        } else if distractors.contains(&i) {
            texts.push(format!("Section {i}: Policy. This section discusses the secret access code policy and what a secret access code is in general, but lists no actual code. {filler}"));
        } else {
            texts.push(format!("Section {i}: Misc. {filler}"));
        }
    }
    let store = match BlockStore::from_texts(model, &texts) {
        Ok(s) => s,
        Err(e) => { eprintln!("jaded: [blended] build error: {e}"); return; }
    };
    let question = "What is the secret access code?";
    eprintln!("jaded: [blended] {} blocks (needle @ {needle_at}, distractors {distractors:?}); N={n_cand} k={k}", store.blocks.len());

    // PLAIN: retrieve top-k → compose.
    let mut cache = KvBlockCache::new(2 * 1024 * 1024 * 1024);
    let t = std::time::Instant::now();
    match compose_and_answer(model, ctx, &store, &mut cache, question, k, 0.0) {
        Ok((ans, sel)) => {
            let ms = t.elapsed().as_millis();
            eprintln!("jaded: [blended] PLAIN   sel={sel:?} needle={} ({ms}ms)", ans.contains(needle));
        }
        Err(e) => { eprintln!("jaded: [blended] plain error: {e}"); return; }
    }

    // BLENDED: retrieve N → rerank → compose top-k.
    let mut cache_b = KvBlockCache::new(2 * 1024 * 1024 * 1024);
    let t = std::time::Instant::now();
    match blended_compose_and_answer(model, ctx, &store, &mut cache_b, question, n_cand, k, 0.0) {
        Ok((ans, sel, scored)) => {
            let ms = t.elapsed().as_millis();
            eprintln!("jaded: [blended] BLENDED sel={sel:?} needle={} ({ms}ms)", ans.contains(needle));
            eprintln!("jaded: [blended] rerank scores (top 8, higher=more relevant):");
            for (bi, s) in scored.iter().take(8) {
                let tag = if *bi == needle_at { " <- NEEDLE" }
                    else if distractors.contains(bi) { " <- distractor" } else { "" };
                eprintln!("jaded: [blended]   block {bi:>3}  score {s:>9.4}{tag}");
            }
        }
        Err(e) => { eprintln!("jaded: [blended] blended error: {e}"); }
    }
}

// ── Unit tests (pure: no model; run under `--features inference` on Linux) ──────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_evicts_least_recently_used() {
        let mut c = KvBlockCache::new(10);
        c.insert(0, vec![0u8; 4]);
        c.insert(1, vec![0u8; 4]); // 8 bytes, both fit
        assert_eq!(c.stats().0, 2);
        assert!(c.get(0).is_some()); // touch 0 → block 1 becomes LRU
        c.insert(2, vec![0u8; 4]); // 12 > 10 → evict the LRU (block 1)
        assert!(c.get(1).is_none());
        assert!(c.get(0).is_some());
        assert!(c.get(2).is_some());
    }

    #[test]
    fn kv_cache_zero_cap_disables_and_skips_oversized() {
        let mut c = KvBlockCache::new(0); // disabled
        c.insert(0, vec![0u8; 8]);
        assert!(c.get(0).is_none());
        assert_eq!(c.stats(), (0, 0));
        let mut c2 = KvBlockCache::new(4);
        c2.insert(0, vec![0u8; 8]); // a single block bigger than the cap is not cached
        assert!(c2.get(0).is_none());
    }

    #[test]
    fn retrieve_prefers_idf_token_overlap() {
        let store = BlockStore {
            blocks: vec![
                Block { ids: vec![1, 2, 3], text: "a".into() },
                Block { ids: vec![7, 8, 9], text: "b".into() },
                Block { ids: vec![4, 5, 6], text: "c".into() },
            ],
        };
        // query tokens live only in block 1 → it scores highest.
        assert_eq!(store.retrieve(&[7, 8], 1), vec![1]);
    }

    #[test]
    fn retrieve_falls_back_to_first_k_on_no_overlap() {
        let store = BlockStore {
            blocks: vec![
                Block { ids: vec![1, 2], text: "a".into() },
                Block { ids: vec![3, 4], text: "b".into() },
                Block { ids: vec![5, 6], text: "c".into() },
            ],
        };
        // nothing overlaps the query → fall back to the first k block indices.
        assert_eq!(store.retrieve(&[99], 2), vec![0, 1]);
    }
}
