//! Standard single-shot inference: one sequence, KV prefix reuse, constrained
//! decode. This is the daemon's default request path — `handle_connection` streams
//! each token out as it's produced.

use super::build_formatted_prompt;
use super::kv;
use jade_model_profile::ChatTemplate;
use jade_protocol::InferenceRequest;
use jaded::control::DecodeControl;
use llama_cpp_2::{
    context::LlamaContext,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel},
    sampling::LlamaSampler,
    token::LlamaToken,
};

/// Generate a response for `req` on sequence 0, invoking `on_token` with each
/// chunk of client-safe text. Returns total tokens used (prompt + generated).
pub(crate) fn infer_streaming(
    req: &InferenceRequest,
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    tmpl: &ChatTemplate,
    kv_tokens: &mut Vec<LlamaToken>,
    mut on_token: impl FnMut(&str),
) -> Result<u64, String> {
    let mut decoder = encoding_rs::UTF_8.new_decoder_without_bom_handling();

    let prompt = build_formatted_prompt(&req.prompt, tmpl)?;

    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| format!("tokenization failed: {e}"))?;

    let n_prompt = tokens.len();
    let ctx_size = ctx.n_ctx();
    if n_prompt >= ctx_size as usize {
        return Err(format!("prompt ({n_prompt} tokens) exceeds context size ({ctx_size})"));
    }

    // Reuse the shared leading prefix already resident in the KV cache; only the
    // divergent suffix needs prefilling (the multi-turn-chat speedup).
    let prefix_len = kv::reuse_prefix(ctx, kv_tokens, &tokens)?;

    // Chunked prefill: process only the new suffix tokens at their absolute positions.
    // We start at prefix_len (the shared prefix is already in the KV cache) and feed
    // the model the new tokens in batches of at most n_batch at a time.
    let n_batch = ctx.n_batch() as usize;
    let last = n_prompt - 1; // index of the final prompt token (needs logits)
    let mut pos = prefix_len;
    while pos < n_prompt {
        // this chunk covers [pos, chunk_end); never more than n_batch tokens
        let chunk_end = (pos + n_batch).min(n_prompt);
        let mut batch = LlamaBatch::new(chunk_end - pos, 1);
        for i in pos..chunk_end {
            // add token i at absolute position i, sequence 0. The 4th arg asks for
            // logits ONLY on the very last prompt token — that's the one we sample
            // the first generated token from; the rest are just context.
            batch.add(tokens[i], i as i32, &[0], i == last)
                .map_err(|e| format!("batch add failed: {e}"))?;
        }
        // run the forward pass for this chunk, populating the KV cache
        ctx.decode(&mut batch).map_err(|e| format!("prefill failed at chunk {pos}..{chunk_end}: {e}"))?;
        pos = chunk_end;
    }

    // Record the prompt tokens in kv_tokens (suffix appended to retained prefix).
    kv_tokens.extend_from_slice(&tokens[prefix_len..]);

    // Free sampler used for unconstrained tokens (and all tokens once grammar is retired).
    let mut free_sampler = LlamaSampler::chain_simple(vec![
        LlamaSampler::temp(0.6),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::dist(0),
    ]);

    // Grammar sampler (constrained decoding). Kept allocated until after the
    // generation loop — freeing a grammar sampler while the context is active
    // crashes llama.cpp.
    let mut anchor_sampler: Option<LlamaSampler> = if let Some(ref gbnf) = req.grammar {
        match LlamaSampler::grammar(model, gbnf, "root") {
            Ok(g) => Some(LlamaSampler::chain_simple(vec![
                LlamaSampler::temp(0.6),
                g,
                LlamaSampler::min_p(0.05, 1),
                LlamaSampler::dist(0),
            ])),
            Err(e) => return Err(format!("GBNF grammar error: {e:?}")),
        }
    } else {
        None
    };

    // All anchor / stop-anchor / depth-tracking decisions live in this pure state
    // machine (see jaded::control). The engine only asks it which sampler to use
    // and what text to emit.
    let mut control = DecodeControl::new(req);

    // cap on new tokens: whichever is smaller — the room left in the context
    // window, or what the request asked for.
    let max_new = (ctx_size as usize).saturating_sub(n_prompt).min(req.max_tokens as usize);
    let mut n_gen = 0usize;
    let mut gen_batch = LlamaBatch::new(1, 1); // we decode one new token at a time

    loop {
        // Pick the sampler the control machine tells us to use this step, and
        // sample the next token from the most recent logits (index -1).
        //
        // LlamaSampler::sample() already accepts the token internally
        // (llama_sampler_sample calls llama_sampler_accept at the end). Do NOT
        // call sampler.accept() again — double-accepting empties the grammar
        // stacks and causes an abort on the next sample.
        let token = if control.use_grammar() {
            // constrained step: use the grammar sampler, then tell control it ran
            // (so it can retire the grammar after this first anchored token).
            let gsamp = anchor_sampler.as_mut().unwrap();
            let t = gsamp.sample(&ctx, -1);
            control.notice_grammar_sampled();
            t
        } else {
            // free step: ordinary temperature/min-p sampling
            free_sampler.sample(&ctx, -1)
        };

        // stop conditions: the model emitted an end-of-generation token, or we hit
        // the token budget. Either way we don't emit this token.
        if model.is_eog_token(token) || n_gen >= max_new {
            break;
        }

        // turn the token id into its text fragment (a "piece"); the decoder handles
        // multi-token UTF-8 sequences split across pieces.
        let piece = model.token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| format!("token to str failed: {e}"))?;
        n_gen += 1;

        // run the piece through the control machine; it returns the text that's
        // actually safe to stream (may be None while buffering a stop anchor).
        if let Some(text) = control.on_piece(&piece) {
            on_token(&text);
        }

        // record this token as now resident in the KV cache (for next-request
        // prefix reuse).
        kv_tokens.push(token);

        // feed the just-generated token back in as the next position so the model
        // can produce the following token. position = n_prompt + n_gen - 1.
        gen_batch.clear();
        let gen_pos = (n_prompt + n_gen - 1) as i32;
        gen_batch.add(token, gen_pos, &[0], true)
            .map_err(|e| format!("batch add failed: {e}"))?;
        ctx.decode(&mut gen_batch).map_err(|e| format!("decode step failed: {e}"))?;
    }

    // Flush anything buffered for stop-anchor prefix detection that never
    // completed (EOG, max_new, or depth-tracker retirement): it is real output.
    if let Some(tail) = control.finish() {
        on_token(&tail);
    }

    // Keep the grammar sampler alive until here (see allocation comment above).
    drop(anchor_sampler);

    Ok((n_prompt + n_gen) as u64)
}
