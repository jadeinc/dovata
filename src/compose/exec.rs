//! Tensor-level RLM engine: KV-cache composition.
//!
//! The full corpus lives in CPU RAM as token blocks ([`BlockStore`]). Per query the
//! engine retrieves the relevant blocks, splices their precomputed K/V into ONE
//! attention window, and answers in a SINGLE decode — no per-branch re-prompting and
//! no agentic decision calls. Two pipelines share the same COMBINE step:
//!
//! * [`compose_and_answer`] — PICK (IDF retrieve top-k) → COMBINE (compose + decode).
//! * [`blended_compose_and_answer`] — PICK (retrieve N) → FAN-OUT (cross-encoder
//!   rerank by query-likelihood) → COMBINE (compose top-k + decode).
//!
//! The mechanics rest on three llama.cpp primitives validated on the fork: a
//! sequence's K/V serializes to a host buffer and restores byte-identically
//! ([`save_seq`]/[`load_seq`]); `kv_cache_seq_add` relocates a block to new positions
//! RoPE-correctly; and same-stream `copy_kv_cache_seq` (metadata-only under a UNIFIED
//! KV cache) splices it into the target sequence. So composition needs no `is_full`
//! fork surgery and no CUDA kernels — it is pure KV-cache cell management.
//!
//! Everything here requires the serving context to be created with `kv_unified=true`.
//! The model-free pieces (retrieval, the LRU KV cache) are unit-tested at the bottom.
//! The model-bound paths have no automated coverage: they used to be driven by
//! env-gated selftests, which went away with the daemon startup phase they hung off.
//! Verifying a change to them means running a real prompt on the GPU box.

use std::collections::{HashMap, HashSet, VecDeque};
use crate::sample::chain;
use crate::tokenize::exec::PieceDecoder;
use llama_cpp_2::{
    context::LlamaContext,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel},
    token::LlamaToken,
    LlamaStateSeqFlags,
};

/// Serialize sequence `seq_id`'s full KV state to a byte buffer.
pub(crate) fn save_seq(ctx: &LlamaContext, seq_id: i32) -> Vec<u8> {
    let size = ctx.state_seq_get_size_ext(seq_id, LlamaStateSeqFlags::empty());
    let mut buf = vec![0u8; size];
    // SAFETY: buf has `size` bytes, the size the engine reported it needs.
    let n = unsafe { ctx.state_seq_get_data_ext(buf.as_mut_ptr(), seq_id, LlamaStateSeqFlags::empty()) };
    buf.truncate(n);
    buf
}

/// Restore a previously-saved KV state into sequence `seq_id`.
pub(crate) fn load_seq(ctx: &mut LlamaContext, bytes: &[u8], seq_id: i32) -> bool {
    // SAFETY: `bytes` came from save_seq on a compatible context.
    unsafe { ctx.state_seq_set_data_ext(bytes, seq_id, LlamaStateSeqFlags::empty()) }
}

/// Chunked prefill of `tokens` at positions `base_pos..` on `seq`. Logits only on
/// the final token when `logits_last`.
pub(crate) fn prefill(
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

/// Greedily generate up to `k` tokens from the CURRENT logits (requires a prior
/// decode that requested logits on its last token). Feeds each sampled token back
/// at `next_pos..` on `seq`.
pub(crate) fn gen_loop(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    seq: i32,
    mut next_pos: i32,
    k: usize,
) -> Result<String, String> {
    let mut sampler = chain::greedy();
    let mut decoder = PieceDecoder::new();
    let mut batch = LlamaBatch::new(1, 1);
    let mut out = String::new();
    for _ in 0..k {
        let tok = sampler.sample(ctx, -1);
        if model.is_eog_token(tok) {
            break;
        }
        out.push_str(&decoder.piece(model, tok)?);
        batch.clear();
        batch.add(tok, next_pos, &[seq], true).map_err(|e| format!("gen add: {e}"))?;
        ctx.decode(&mut batch).map_err(|e| format!("gen decode: {e}"))?;
        next_pos += 1;
    }
    Ok(out)
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
            blocks.push(Block { ids });
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
pub(crate) fn materialize_block(
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
/// re-prefilling and caching it), then a single decode. Returns (answer, selected
/// indices). **Requires the context to be created with `kv_unified=true`.**
pub(crate) fn compose_and_answer(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    store: &BlockStore,
    cache: &mut KvBlockCache,
    question: &str,
    k: usize,
) -> Result<(String, Vec<usize>), String> {
    // ── PICK: retrieve top-k (non-agentic) ──────────────────────────────────────
    let q_ids = tokenize_query(model, question)?;
    let selected = store.retrieve(&q_ids, k);
    eprintln!("dovata: [compose] retrieved blocks {selected:?} of {}", store.blocks.len());
    // ── COMBINE: compose + single decode ────────────────────────────────────────
    let ans = compose_selected(model, ctx, store, cache, &selected, question)?;
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
/// `cache`), then a single decode. Shared by the plain (retrieve→compose) and blended
/// (retrieve→rerank→compose) pipelines. Requires `kv_unified=true`.
fn compose_selected(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    store: &BlockStore,
    cache: &mut KvBlockCache,
    selected: &[usize],
    question: &str,
) -> Result<String, String> {
    ctx.clear_kv_cache();
    // Lay down the instruction prefix at positions [0, off), then splice each selected
    // block's KV in immediately after it, packing the composed window contiguously.
    let prefix = model
        .str_to_token("You are given several notes. Use them to answer the question.\n\n", AddBos::Always)
        .map_err(|e| format!("tok prefix: {e}"))?;
    prefill(ctx, &prefix, 0, 0, false)?;
    let mut off = prefix.len() as i32;
    for (j, &bi) in selected.iter().enumerate() {
        // Scratch seq (j+1) holds the block in isolation before it's relocated to `off`
        // and copied into seq 0; advance `off` past it so the next block follows on.
        off += materialize_block(ctx, cache, bi, &store.blocks[bi].ids, off, (j + 1) as i32)?;
    }

    // Append the question after the composed blocks and run ONE decode over the window.
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
    ctx: &mut LlamaContext,
    cache: &mut KvBlockCache,
    store: &BlockStore,
    candidates: &[usize],
    q_ids: &[i32],
) -> Result<Vec<(usize, f32)>, String> {
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(candidates.len());
    for &bi in candidates {
        let s = score_block_query(ctx, cache, store, bi, q_ids)?;
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
) -> Result<(String, Vec<usize>, Vec<(usize, f32)>), String> {
    let q_ids = tokenize_query(model, question)?;
    let candidates = store.retrieve(&q_ids, n_candidates.max(k)); // PICK (recall)
    let scored = rerank(ctx, cache, store, &candidates, &q_ids)?; // FAN-OUT (rerank)
    let selected: Vec<usize> = scored.iter().take(k).map(|(i, _)| *i).collect();
    eprintln!("dovata: [blended] PICK {} candidates -> RERANK -> COMBINE top-{k} {selected:?}", candidates.len());
    let ans = compose_selected(model, ctx, store, cache, &selected, question)?; // COMBINE
    Ok((ans, selected, scored))
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
                Block { ids: vec![1, 2, 3] },
                Block { ids: vec![7, 8, 9] },
                Block { ids: vec![4, 5, 6] },
            ],
        };
        // query tokens live only in block 1 → it scores highest.
        assert_eq!(store.retrieve(&[7, 8], 1), vec![1]);
    }

    #[test]
    fn retrieve_falls_back_to_first_k_on_no_overlap() {
        let store = BlockStore {
            blocks: vec![
                Block { ids: vec![1, 2] },
                Block { ids: vec![3, 4] },
                Block { ids: vec![5, 6] },
            ],
        };
        // nothing overlaps the query → fall back to the first k block indices.
        assert_eq!(store.retrieve(&[99], 2), vec![0, 1]);
    }
}
