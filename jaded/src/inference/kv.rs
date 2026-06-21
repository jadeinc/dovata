//! KV-cache optimizations shared by the decode engines.
//!
//! Three primitives, all just manipulate the resident KV cache to avoid redundant
//! prefill compute:
//! * [`kv_cache_type`]   — map our pure [`KvType`] onto llama.cpp's enum.
//! * [`reuse_prefix`]    — multi-turn prefix reuse: keep the leading tokens a new
//!                         prompt shares with what's already resident, evict the
//!                         rest (the standard single-shot path's main speedup).
//! * [`common_prefix_len`] + [`prefill_shared`] — shared-prefix batching: prefill a
//!                         common prefix ONCE and copy its KV into N sequences, so
//!                         each only prefills its unique suffix (concurrent / mass
//!                         prompting).

use jaded::config::KvType;
use llama_cpp_2::{
    context::{params::KvCacheType, LlamaContext},
    llama_batch::LlamaBatch,
    token::LlamaToken,
};

/// Map our pure [`KvType`] onto llama.cpp's `KvCacheType`.
pub(crate) fn kv_cache_type(t: KvType) -> KvCacheType {
    match t {
        KvType::F16 => KvCacheType::F16,
        KvType::BF16 => KvCacheType::BF16,
        KvType::F32 => KvCacheType::F32,
        KvType::Q8_0 => KvCacheType::Q8_0,
        KvType::Q5_1 => KvCacheType::Q5_1,
        KvType::Q5_0 => KvCacheType::Q5_0,
        KvType::Q4_1 => KvCacheType::Q4_1,
        KvType::Q4_0 => KvCacheType::Q4_0,
    }
}

/// Multi-turn KV prefix reuse for sequence 0. Compares the new prompt `tokens`
/// against the tokens currently resident (`kv_tokens`), keeps the shared leading
/// prefix, and evicts the divergent suffix from the cache. Returns the prefix
/// length (the caller prefills only `tokens[prefix_len..]`).
///
/// The last cached token is never reused — it was the previous response's
/// generation seed and its logits are stale.
pub(crate) fn reuse_prefix(
    ctx: &mut LlamaContext,
    kv_tokens: &mut Vec<LlamaToken>,
    tokens: &[LlamaToken],
) -> Result<usize, String> {
    let n_prompt = tokens.len();
    let prefix_len = tokens
        .iter()
        .zip(kv_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count()
        .saturating_sub(1)
        .min(n_prompt.saturating_sub(1));

    if prefix_len > 0 {
        // Evict only the suffix beyond the shared prefix.
        ctx.clear_kv_cache_seq(Some(0), Some(prefix_len as u32), None)
            .map_err(|e| format!("kv_cache_seq clear failed: {e}"))?;
        kv_tokens.truncate(prefix_len);
        eprintln!(
            "jaded: kv prefix reuse {prefix_len}/{n_prompt} tokens ({} new)",
            n_prompt - prefix_len
        );
    } else {
        ctx.clear_kv_cache();
        kv_tokens.clear();
    }
    Ok(prefix_len)
}

/// Longest common leading token prefix across all sequences (0 if empty). Finds the
/// shared portion of a set of prompts to hand to [`prefill_shared`] for shared-prefix
/// mass prompting. Retained as batching infrastructure: it has no in-tree caller since
/// the agentic RLM fan-out (its previous user) was removed in favor of KV composition.
#[allow(dead_code)]
pub(crate) fn common_prefix_len(seqs: &[Vec<LlamaToken>]) -> usize {
    let first = match seqs.first() {
        Some(f) => f,
        None => return 0,
    };
    let mut len = first.len();
    for s in &seqs[1..] {
        let mut k = 0;
        while k < len && k < s.len() && s[k].0 == first[k].0 {
            k += 1;
        }
        len = k;
        if len == 0 {
            break;
        }
    }
    len
}

/// Prefill a `shared` token prefix ONCE on a template sequence (`id = n_subs`),
/// then copy its KV into each of the `n_subs` sub-sequences — so each sub only
/// prefills its own unique suffix at `base_pos = shared.len()`. No-op if empty.
pub(crate) fn prefill_shared(
    ctx: &mut LlamaContext,
    shared: &[LlamaToken],
    n_subs: usize,
) -> Result<(), String> {
    if shared.is_empty() {
        return Ok(());
    }
    let n_batch = ctx.n_batch() as usize;
    let template_seq = n_subs as i32;
    let mut pos = 0usize;
    while pos < shared.len() {
        let end = (pos + n_batch).min(shared.len());
        let mut b = LlamaBatch::new(end - pos, 1);
        for i in pos..end {
            b.add(shared[i], i as i32, &[template_seq], false)
                .map_err(|e| format!("shared prefill add: {e}"))?;
        }
        ctx.decode(&mut b).map_err(|e| format!("shared prefill decode: {e}"))?;
        pos = end;
    }
    // Full-sequence copy (this fork's seq_cp only supports whole buffers): the
    // template seq holds *exactly* the shared prefix, so a full copy = the prefix.
    for i in 0..n_subs {
        ctx.copy_kv_cache_seq(template_seq, i as i32, None, None)
            .map_err(|e| format!("shared kv copy: {e}"))?;
    }
    Ok(())
}
