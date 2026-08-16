//! Continuous-batching concurrent engine: run N sub-inferences against ONE resident
//! model/context, with every `ctx.decode` packing tokens from every live sequence
//! into one forward pass (the RLM fan-out latency win). All scheduling bookkeeping
//! is in the pure [`crate::batch::sched::Scheduler`]; this is the thin llama-bound driver
//! that translates its plans into batches and feeds sampled tokens back.

use crate::kv::exec as kv;
use crate::sample::chain;
use crate::tokenize::exec::PieceDecoder;
use crate::sample::control::DecodeControl;
use crate::batch::sched::Scheduler;
use llama_cpp_2::{
    context::LlamaContext,
    llama_batch::LlamaBatch,
    model::LlamaModel,
    sampling::LlamaSampler,
    token::LlamaToken,
};

/// One sub-inference to run on its own sequence id.
pub(crate) struct SubInfer {
    /// Already-tokenized prompt suffix (the caller does any prefix reuse first).
    pub(crate) tokens: Vec<LlamaToken>,
    /// Absolute KV position the suffix starts at (0 for an independent sub-call;
    /// >0 when a shared prefix was pre-copied into this sequence's KV).
    pub(crate) base_pos: i32,
    /// Optional GBNF for constrained decoding (grammar sampler built per sequence).
    pub(crate) grammar: Option<String>,
    /// Per-sequence decode state machine (anchors / stop / depth tracking).
    pub(crate) control: DecodeControl,
    /// Generated-token cap for this sub-call.
    pub(crate) max_new: usize,
}

/// Run all `subs` concurrently against `ctx`, returning each one's generated text
/// (indexed parallel to `subs`). One `ctx.decode` per tick advances every live
/// sequence; per-sequence logits are sampled at their batch positions.
pub(crate) fn run_concurrent(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    subs: Vec<SubInfer>,
) -> Result<Vec<String>, String> {
    run_concurrent_inner(model, ctx, &[], subs)
}

/// Like [`run_concurrent`] but first prefills a SHARED token prefix ONCE and copies
/// its KV into every sub-sequence (cross-call prefix reuse). Each sub's `tokens`
/// must then be only its UNIQUE suffix with `base_pos = shared.len()`. `shared`
/// empty = no prefix sharing (the plain concurrent path).
pub(crate) fn run_concurrent_inner(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    shared: &[LlamaToken],
    subs: Vec<SubInfer>,
) -> Result<Vec<String>, String> {
    let n = subs.len();
    let n_batch = ctx.n_batch() as usize;
    // The whole KV cache (n_ctx cells) is the budget shared across sequences.
    let kv_budget = ctx.n_ctx() as usize;
    let mut sched = Scheduler::new(n_batch, kv_budget);

    let mut outputs = vec![String::new(); n];
    // Per-sequence UTF-8 decoders — a multi-byte char split across two BPE tokens
    // of the SAME sequence must not interleave with another sequence's bytes.
    let mut decoders: Vec<PieceDecoder> = (0..n).map(|_| PieceDecoder::new()).collect();
    // One free sampler shared by all unconstrained samples (no per-seq state).
    let sampling = chain::Params::default();
    let mut free_sampler = chain::free(sampling);
    // Per-sequence grammar samplers (stateful grammar stacks), kept alive for the
    // whole run — freeing one while the context is live crashes llama.cpp.
    let mut grammar_samplers: Vec<Option<LlamaSampler>> = Vec::with_capacity(n);

    // Start every sequence from a clean KV cache, then (if any) seed the shared
    // prefix once and copy it into each sub-sequence.
    ctx.clear_kv_cache();
    kv::prefill_shared(ctx, shared, n)?;

    for (i, sub) in subs.into_iter().enumerate() {
        let gs = match sub.grammar {
            Some(ref gbnf) => Some(
                chain::grammar(model, gbnf, sampling)
                    .map_err(|e| format!("sub {i}: {e}"))?,
            ),
            None => None,
        };
        grammar_samplers.push(gs);
        let toks: Vec<i32> = sub.tokens.iter().map(|t| t.0).collect();
        if !sched.admit(i as i32, toks, sub.base_pos, sub.control, sub.max_new) {
            return Err(format!(
                "sub {i} rejected by KV admission (budget {kv_budget} tokens exceeded)"
            ));
        }
    }

    // Drive the scheduler until every sequence has retired.
    while !sched.is_idle() {
        let plan = sched.plan();
        if plan.is_empty() {
            break; // all live sequences are awaiting a sample we already produced
        }
        // Translate the plan into one batch; tag each token with its sequence id.
        let mut batch = LlamaBatch::new(plan.entries.len(), n as i32);
        for e in &plan.entries {
            batch
                .add(LlamaToken(e.token), e.pos, &[e.seq_id], e.want_logits)
                .map_err(|err| format!("batch add failed: {err}"))?;
        }
        ctx.decode(&mut batch).map_err(|err| format!("decode failed: {err}"))?;

        // The k-th sampleable output sits at the k-th logits-enabled batch position.
        let logit_pos: Vec<usize> = plan
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.want_logits.then_some(i))
            .collect();

        for (k, slot) in plan.sample_order.iter().enumerate() {
            let sid = slot.seq_id as usize;
            let idx = logit_pos[k] as i32; // batch position of this sequence's logits
            let token = if slot.use_grammar {
                grammar_samplers[sid].as_mut().expect("grammar sampler present").sample(ctx, idx)
            } else {
                free_sampler.sample(ctx, idx)
            };
            let is_eog = model.is_eog_token(token);
            let piece = if is_eog {
                String::new()
            } else {
                decoders[sid].piece(model, token)?
            };
            let ev = sched.accept(slot.seq_id, token.0, &piece, is_eog);
            if let Some(text) = ev.emit {
                outputs[sid].push_str(&text);
            }
        }
    }

    // Keep grammar samplers alive until after the loop (the sharp edge), then reset.
    drop(grammar_samplers);
    ctx.clear_kv_cache();
    Ok(outputs)
}
