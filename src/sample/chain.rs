//! The sampler chain: one definition of what "sampling" means in this engine.
//!
//! Every request path needs two samplers — a free one for unconstrained tokens, and,
//! when the request carries a GBNF grammar, a constrained one for the span the grammar
//! governs. Both used to be built inline, and the same four lines
//!
//! ```text
//! LlamaSampler::chain_simple(vec![temp(0.6), min_p(0.05, 1), dist(0)])
//! ```
//!
//! were copy-pasted, constants and all, into four places across two request paths. The
//! numbers had to agree for the paths to behave alike, and nothing made them agree —
//! change the temperature for the standard path and the batch path silently kept the
//! old one. That is the same shape as the tool-call delimiter drift between the engine
//! profile and its Jade package, which `gen-profile-package.py` exists to prevent, and
//! it gets the same fix: one owner.
//!
//! What is *not* here is the decision of when to use which sampler, or when the grammar
//! may be retired. That is [`super::control`], and it is pure — it reasons about the
//! text the tokens spell, not about llama.

use llama_cpp_2::{model::LlamaModel, sampling::LlamaSampler};

/// Sampling parameters. Not yet request-controlled — these were hardcoded at each of
/// the four call sites, and collecting them here is the step that makes surfacing them
/// on `InferenceRequest` a small change rather than a four-file one.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub temperature: f32,
    /// Minimum probability, relative to the most likely token.
    pub min_p: f32,
    /// Minimum tokens `min_p` may leave in the running.
    pub min_keep: usize,
    /// RNG seed. 0 means llama picks one.
    pub seed: u32,
}

impl Default for Params {
    fn default() -> Self {
        // The values every call site carried. Preserved exactly: this refactor is not
        // the place to change what the engine samples.
        Self { temperature: 0.6, min_p: 0.05, min_keep: 1, seed: 0 }
    }
}

/// The unconstrained chain: temperature, then `min_p`, then a distribution draw.
///
/// Also the chain used for *every* token once a grammar has been retired, which is why
/// it is built even for a constrained request.
#[must_use]
pub fn free(p: Params) -> LlamaSampler {
    LlamaSampler::chain_simple(vec![
        LlamaSampler::temp(p.temperature),
        LlamaSampler::min_p(p.min_p, p.min_keep),
        LlamaSampler::dist(p.seed),
    ])
}

/// The constrained chain: the same, with a GBNF grammar spliced in after temperature.
///
/// Position matters. The grammar filters the candidate set, so it has to see logits that
/// temperature has already shaped and hand its survivors to `min_p` and the draw. Moving
/// it changes which tokens are reachable, quietly.
///
/// The returned sampler holds llama-side grammar state and must outlive the generation
/// loop — freeing one while its context is still active crashes llama.cpp.
///
/// # Errors
/// If llama cannot compile the GBNF.
pub fn grammar(model: &LlamaModel, gbnf: &str, p: Params) -> Result<LlamaSampler, String> {
    let g = LlamaSampler::grammar(model, gbnf, "root")
        .map_err(|e| format!("GBNF grammar error: {e:?}"))?;
    Ok(LlamaSampler::chain_simple(vec![
        LlamaSampler::temp(p.temperature),
        g,
        LlamaSampler::min_p(p.min_p, p.min_keep),
        LlamaSampler::dist(p.seed),
    ]))
}

/// Greedy: always the most likely token, no randomness.
///
/// The composition engine uses this. Its answers are read back and compared across
/// runs, so a sampled reply would make a real regression and ordinary variance look the
/// same.
#[must_use]
pub fn greedy() -> LlamaSampler {
    LlamaSampler::greedy()
}
