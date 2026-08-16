//! Pure multi-sequence decode scheduler — the concurrency core of the RLM engine.
//!
//! The single-request loop in `inference.rs` decodes ONE sequence at a time (seq
//! id hardcoded `&[0]`). To run many sub-inferences against one resident model —
//! the latency win behind RLM fan-out — the daemon must *continuously batch*:
//! every `llama_decode` step packs tokens from EVERY live sequence into one
//! forward pass, and per-sequence logits are sampled by batch output offset.
//!
//! All of that scheduling is pure bookkeeping — which sequence feeds which token
//! at which position, which batch outputs are sampleable and to whom, and when a
//! sequence's KV may be admitted or must be retired. So it lives here, free of
//! llama.cpp, and is unit-tested without a model (the same discipline as
//! `control.rs`). The Linux glue (`inference.rs`) owns only the three llama-bound
//! steps, once per tick:
//!   1. translate a [`BatchPlan`] into `LlamaBatch::add` + `ctx.decode`,
//!   2. for each [`SampleSlot`] of [`BatchPlan::sample_order`] (in order = batch
//!      output offset), sample that offset with the sequence's grammar-or-free
//!      sampler (the slot says which) and `token_to_piece` the result,
//!   3. feed each result back via [`Scheduler::accept`].
//!
//! Tokens are plain `i32` ids here; the glue wraps/unwraps `LlamaToken` at the
//! boundary (that type is Linux-only).
//!
//! Scope: this is Phase 1 — N *independent* concurrent sub-inferences. The
//! recurse-action / park-resume that makes a parent wait on its children
//! (Phase 2) layers on top of `accept`'s `finished` signal and is not here yet.

use crate::sample::control::DecodeControl;

/// llama.cpp sequence id (the KV partition a token belongs to).
pub type SeqId = i32;
/// A token id — plain integer in the pure layer; glue wraps it as `LlamaToken`.
pub type TokenId = i32;

/// One token to place in the next decode batch. The glue does
/// `batch.add(LlamaToken(token), pos, &[seq_id], want_logits)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanEntry {
    pub seq_id: SeqId,
    pub token: TokenId,
    pub pos: i32,
    /// Ask llama for this token's logits (we sample the *next* token from them).
    pub want_logits: bool,
    /// Sample those logits with the grammar sampler (vs. the free sampler).
    /// Only meaningful when `want_logits`.
    pub use_grammar: bool,
}

/// A sampleable batch output, in batch-offset order: `sample_order[i]` is produced
/// at logits offset `i`. The glue samples offset `i` and calls `accept` for `seq_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleSlot {
    pub seq_id: SeqId,
    pub use_grammar: bool,
}

/// The work for one decode step.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchPlan {
    /// Every token to feed this step, across all live sequences.
    pub entries: Vec<PlanEntry>,
    /// The sampleable outputs, ordered by batch offset.
    pub sample_order: Vec<SampleSlot>,
}

impl BatchPlan {
    /// Nothing to decode this tick (all sequences idle or awaiting their sample).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// What one accepted sample did to its sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqEvent {
    /// Text to stream to this sequence's client (already passed through the
    /// sequence's `DecodeControl`; `None` while buffering a stop anchor).
    pub emit: Option<String>,
    /// The sequence retired this step (EOG, token budget, or flushed tail). Its KV
    /// reservation has been released; no further plan entries will be produced.
    pub finished: bool,
}

/// Per-sequence decode state. Private — the engine only sees plans and events.
struct Seq {
    id: SeqId,
    /// The suffix this sequence must prefill, at absolute positions
    /// `base_pos + i`. (The caller does any prefix reuse / `copy_kv_cache_seq`
    /// before admitting and passes only the not-yet-resident tokens.)
    prompt: Vec<TokenId>,
    base_pos: i32,
    control: DecodeControl,
    max_new: usize,
    /// Generated tokens so far (the token-budget counter).
    n_gen: usize,
    /// Prompt tokens already placed into a batch (prefill progress).
    placed: usize,
    /// False until the first generated token is accepted; once true the sequence
    /// feeds `pending` back each step instead of prompt tokens.
    generating: bool,
    /// The just-sampled token to feed next (valid while `generating`).
    pending: TokenId,
    /// Absolute position for the next fed token.
    next_pos: i32,
    /// A logits request for this sequence is in flight — skip it in `plan` until
    /// its `accept` arrives, so we never double-feed.
    awaiting: bool,
    /// Retired; produces no entries and releases its reservation once.
    done: bool,
    /// Worst-case KV tokens reserved for this seq (prompt + max_new), released on
    /// retirement. Admission is gated on the sum of these.
    kv_reserve: usize,
}

/// Continuous-batching scheduler over one resident model / one context.
pub struct Scheduler {
    seqs: Vec<Seq>,
    /// Max tokens per decode batch (llama `n_batch`).
    n_batch: usize,
    /// Max total reserved KV tokens — the concurrency ceiling (free VRAM / KB per
    /// token). Admission refuses sequences that would exceed it.
    kv_budget: usize,
    kv_reserved: usize,
}

impl Scheduler {
    pub fn new(n_batch: usize, kv_budget: usize) -> Self {
        Scheduler { seqs: Vec::new(), n_batch: n_batch.max(1), kv_budget, kv_reserved: 0 }
    }

    /// Try to admit a new sequence. `prompt` is the (already prefix-reduced) token
    /// suffix to prefill at `base_pos..`; `control` is its decode state machine;
    /// `max_new` caps generated tokens. Returns `false` (admitting nothing) if its
    /// worst-case KV reservation (`prompt + max_new`) would exceed the budget — the
    /// caller should queue it and retry after a sequence retires.
    pub fn admit(
        &mut self,
        id: SeqId,
        prompt: Vec<TokenId>,
        base_pos: i32,
        control: DecodeControl,
        max_new: usize,
    ) -> bool {
        // Reserve worst-case KV: any pre-resident prefix (base_pos, e.g. a
        // shared-prefix copied in) + the suffix this seq prefills + its generation.
        // A sequence with no tokens to prefill can never produce logits (nothing to
        // sample from) — reject it rather than underflow the prefill bookkeeping.
        if prompt.is_empty() {
            return false;
        }
        let reserve = base_pos.max(0) as usize + prompt.len() + max_new;
        // Admission control: never accept KV we can't hold. (A single sequence
        // larger than the whole budget is always rejected — caller must shrink it.)
        if self.kv_reserved + reserve > self.kv_budget {
            return false;
        }
        self.kv_reserved += reserve;
        self.seqs.push(Seq {
            id,
            prompt,
            base_pos,
            control,
            max_new,
            n_gen: 0,
            placed: 0,
            generating: false,
            pending: 0,
            next_pos: 0,
            awaiting: false,
            done: false,
            kv_reserve: reserve,
        });
        true
    }

    /// No live sequences — the engine can block for new requests.
    pub fn is_idle(&self) -> bool {
        self.seqs.iter().all(|s| s.done)
    }

    /// Number of sequences still decoding.
    pub fn active(&self) -> usize {
        self.seqs.iter().filter(|s| !s.done).count()
    }

    /// Assemble the next decode batch: pack tokens from every sequence that is
    /// neither done nor awaiting a sample, up to `n_batch` total. A prefilling
    /// sequence contributes its next chunk (logits only on its final prompt token);
    /// a generating sequence contributes its one pending fed-back token. Mutates
    /// prefill progress / awaiting flags — the produced entries are assumed to be
    /// decoded (a decode failure aborts the whole request anyway).
    pub fn plan(&mut self) -> BatchPlan {
        let mut plan = BatchPlan::default();
        let mut budget = self.n_batch;

        for seq in self.seqs.iter_mut() {
            if seq.done || seq.awaiting || budget == 0 {
                continue;
            }
            if !seq.generating {
                // ── Prefill: place the next chunk of prompt tokens ──────────────
                let last = seq.prompt.len() - 1; // prompt is non-empty by contract
                let take = (seq.prompt.len() - seq.placed).min(budget);
                for k in 0..take {
                    let idx = seq.placed + k;
                    let is_last = idx == last;
                    plan.entries.push(PlanEntry {
                        seq_id: seq.id,
                        token: seq.prompt[idx],
                        pos: seq.base_pos + idx as i32,
                        // logits only on the final prompt token — that's the one we
                        // sample the first generated token from.
                        want_logits: is_last,
                        use_grammar: is_last && seq.control.use_grammar(),
                    });
                }
                seq.placed += take;
                budget -= take;
                // Reached the final prompt token → its logits are sampleable; park
                // this sequence until `accept` delivers the first generated token.
                if seq.placed == seq.prompt.len() {
                    seq.awaiting = true;
                    plan.sample_order.push(SampleSlot {
                        seq_id: seq.id,
                        use_grammar: seq.control.use_grammar(),
                    });
                }
            } else {
                // ── Generate: feed the one pending token back, request its logits ─
                plan.entries.push(PlanEntry {
                    seq_id: seq.id,
                    token: seq.pending,
                    pos: seq.next_pos,
                    want_logits: true,
                    use_grammar: seq.control.use_grammar(),
                });
                budget -= 1;
                seq.awaiting = true;
                plan.sample_order.push(SampleSlot {
                    seq_id: seq.id,
                    use_grammar: seq.control.use_grammar(),
                });
            }
        }
        plan
    }

    /// Feed back one sampled token for an awaiting sequence (the glue sampled it at
    /// this sequence's batch offset and decoded the piece). Advances the sequence's
    /// state and returns what to stream + whether it retired. `is_eog` is the
    /// model's end-of-generation verdict for `token` (Linux-only, so passed in).
    pub fn accept(&mut self, seq_id: SeqId, token: TokenId, piece: &str, is_eog: bool) -> SeqEvent {
        let seq = self
            .seqs
            .iter_mut()
            .find(|s| s.id == seq_id && !s.done)
            .expect("accept for unknown/retired sequence");
        debug_assert!(seq.awaiting, "accept without an in-flight sample");
        seq.awaiting = false;

        // Stop conditions, checked on the SAMPLED token *before* emitting it — an
        // EOG token or a token past the budget is discarded (matches the original
        // engine loop). The buffered stop-anchor tail, if any, is real output.
        if is_eog || seq.n_gen >= seq.max_new {
            let tail = seq.control.clone().finish();
            self.retire(seq_id);
            return SeqEvent { emit: tail, finished: true };
        }

        // The grammar sampler enforced this token iff the plan said so; tell the
        // control machine, so an anchor-span grammar retires after its first token.
        // (use_grammar is stable between this seq's plan-read and accept — nothing
        // touched its control in between.)
        if seq.control.use_grammar() {
            seq.control.notice_grammar_sampled();
        }
        seq.n_gen += 1;
        let emit = seq.control.on_piece(piece);

        // Advance the feed cursor. First generated token sits at the position right
        // after the prompt; subsequent ones step by one.
        if !seq.generating {
            seq.generating = true;
            seq.next_pos = seq.base_pos + seq.prompt.len() as i32;
        } else {
            seq.next_pos += 1;
        }
        seq.pending = token;

        SeqEvent { emit, finished: false }
    }

    /// Mark a sequence done and release its KV reservation (idempotent per seq).
    fn retire(&mut self, seq_id: SeqId) {
        if let Some(seq) = self.seqs.iter_mut().find(|s| s.id == seq_id && !s.done) {
            seq.done = true;
            self.kv_reserved = self.kv_reserved.saturating_sub(seq.kv_reserve);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ovata_infer_protocol::InferenceRequest;

    fn req(grammar: Option<&str>, anchor: Option<&str>, stop: Option<&str>) -> InferenceRequest {
        InferenceRequest {
            prompt: "p".into(),
            model: "m".into(),
            max_tokens: 256,
            grammar: grammar.map(|s| s.to_owned()),
            anchor: anchor.map(|s| s.to_owned()),
            stop_anchor: stop.map(|s| s.to_owned()),
            count_only: false,
            stats_only: false,
            health_only: false,
            keep_anchors: false,
            trust: 0,
            rlm: false,
        }
    }

    fn ctrl() -> DecodeControl {
        DecodeControl::new(&req(None, None, None))
    }

    // ── Single sequence: mirrors the current single-request loop ────────────────

    #[test]
    fn single_seq_prefill_then_generate_to_eog() {
        let mut s = Scheduler::new(8, 100);
        assert!(s.admit(0, vec![10, 11, 12], 0, ctrl(), 16));

        // Prefill: all 3 tokens fit in one batch; only the last requests logits.
        let p = s.plan();
        assert_eq!(p.entries.len(), 3);
        assert_eq!(p.entries[0], PlanEntry { seq_id: 0, token: 10, pos: 0, want_logits: false, use_grammar: false });
        assert_eq!(p.entries[2], PlanEntry { seq_id: 0, token: 12, pos: 2, want_logits: true, use_grammar: false });
        assert_eq!(p.sample_order, vec![SampleSlot { seq_id: 0, use_grammar: false }]);

        // Sample the first generated token (id 100, piece "A").
        let ev = s.accept(0, 100, "A", false);
        assert_eq!(ev, SeqEvent { emit: Some("A".into()), finished: false });

        // Next tick feeds token 100 back at position 3 (right after the prompt).
        let p = s.plan();
        assert_eq!(p.entries, vec![PlanEntry { seq_id: 0, token: 100, pos: 3, want_logits: true, use_grammar: false }]);
        let ev = s.accept(0, 101, "B", false);
        assert_eq!(ev.emit, Some("B".into()));

        // Token 101 fed at position 4; then the model emits EOG → finish, no emit.
        let p = s.plan();
        assert_eq!(p.entries[0].pos, 4);
        let ev = s.accept(0, 0, "", true);
        assert_eq!(ev, SeqEvent { emit: None, finished: true });
        assert!(s.is_idle());
    }

    // ── Two sequences share one batch (the concurrency proof) ───────────────────

    #[test]
    fn two_seqs_packed_into_one_batch_and_advance_independently() {
        let mut s = Scheduler::new(16, 100);
        assert!(s.admit(0, vec![1, 2], 0, ctrl(), 8));
        assert!(s.admit(1, vec![7, 8, 9], 0, ctrl(), 8));

        // One plan packs BOTH sequences' prefills; sample_order maps offsets→seqs.
        let p = s.plan();
        assert_eq!(p.entries.len(), 5); // 2 + 3 prompt tokens
        assert_eq!(
            p.sample_order,
            vec![SampleSlot { seq_id: 0, use_grammar: false }, SampleSlot { seq_id: 1, use_grammar: false }]
        );
        // seq 0's last prompt token is at offset 0, seq 1's at offset... the 2nd
        // want_logits entry. Verify positions are per-sequence (both start at 0).
        let logit_entries: Vec<_> = p.entries.iter().filter(|e| e.want_logits).collect();
        assert_eq!(logit_entries[0].seq_id, 0);
        assert_eq!(logit_entries[0].pos, 1);
        assert_eq!(logit_entries[1].seq_id, 1);
        assert_eq!(logit_entries[1].pos, 2);

        // Sample both; they advance independently.
        s.accept(0, 50, "x", false);
        s.accept(1, 60, "y", false);
        let p = s.plan();
        assert_eq!(p.entries.len(), 2); // one fed-back token each
        assert_eq!(p.entries[0], PlanEntry { seq_id: 0, token: 50, pos: 2, want_logits: true, use_grammar: false });
        assert_eq!(p.entries[1], PlanEntry { seq_id: 1, token: 60, pos: 3, want_logits: true, use_grammar: false });

        // Retire seq 0; seq 1 keeps going alone.
        s.accept(0, 0, "", true);
        s.accept(1, 61, "z", false);
        assert_eq!(s.active(), 1);
        let p = s.plan();
        assert_eq!(p.entries.len(), 1);
        assert_eq!(p.entries[0].seq_id, 1);
    }

    // ── n_batch splits a long prefill across ticks (no logits until the last) ────

    #[test]
    fn long_prefill_is_chunked_by_n_batch() {
        let mut s = Scheduler::new(2, 100); // tiny batch
        assert!(s.admit(0, vec![1, 2, 3, 4, 5], 0, ctrl(), 4));

        let p = s.plan(); // tokens 0,1
        assert_eq!(p.entries.len(), 2);
        assert!(p.sample_order.is_empty(), "no logits until the final prompt token");
        let p = s.plan(); // tokens 2,3
        assert_eq!(p.entries.len(), 2);
        assert!(p.sample_order.is_empty());
        let p = s.plan(); // token 4 (last) → logits
        assert_eq!(p.entries.len(), 1);
        assert_eq!(p.entries[0].pos, 4);
        assert!(p.entries[0].want_logits);
        assert_eq!(p.sample_order.len(), 1);
    }

    // ── KV admission control = the concurrency ceiling ──────────────────────────

    #[test]
    fn admission_refuses_when_kv_budget_exceeded() {
        // budget 10; each seq reserves prompt(3) + max_new(4) = 7.
        let mut s = Scheduler::new(8, 10);
        assert!(s.admit(0, vec![1, 2, 3], 0, ctrl(), 4)); // reserves 7, total 7
        assert!(!s.admit(1, vec![4, 5, 6], 0, ctrl(), 4), "second would push to 14 > 10");

        // Retire seq 0 → reservation freed → the second now fits.
        let _ = s.plan();
        s.accept(0, 0, "", true);
        assert!(s.admit(1, vec![4, 5, 6], 0, ctrl(), 4));
    }

    // ── max_new caps generated tokens ───────────────────────────────────────────

    #[test]
    fn admit_rejects_empty_prompt() {
        // A seq with no tokens to prefill can't produce logits — must be refused.
        let mut s = Scheduler::new(8, 100);
        assert!(!s.admit(0, vec![], 0, ctrl(), 8));
        assert!(s.is_idle());
    }

    #[test]
    fn max_new_one_emits_exactly_one_token() {
        let mut s = Scheduler::new(8, 100);
        assert!(s.admit(0, vec![1], 0, ctrl(), 1));
        let _ = s.plan(); // prefill the single prompt token (logits on it)
        // First sample: n_gen 0 < 1 → emitted.
        let ev = s.accept(0, 90, "first", false);
        assert_eq!(ev.emit, Some("first".into()));
        assert!(!ev.finished);
        // Feed it back; next sample: n_gen 1 >= 1 → discarded, finish.
        let _ = s.plan();
        let ev = s.accept(0, 91, "second", false);
        assert_eq!(ev, SeqEvent { emit: None, finished: true });
    }

    // ── Grammar sequence: plan flags the sampler, accept retires the grammar ─────

    #[test]
    fn grammar_anchor_span_flows_through_plan_and_accept() {
        // Anchor-span grammar: enforce one token after the anchor, then retire.
        let mut s = Scheduler::new(8, 100);
        let c = DecodeControl::new(&req(Some(r#"root ::= "{" [^}]* "}""#), Some("<t>"), Some("</t>")));
        assert!(s.admit(0, vec![1, 2], 0, c, 16));

        // Prefill: grammar not yet triggered (anchor "<t>" hasn't been emitted), so
        // the last prompt token's logits are sampled with the FREE sampler.
        let p = s.plan();
        assert_eq!(p.sample_order, vec![SampleSlot { seq_id: 0, use_grammar: false }]);

        // The model emits the anchor — grammar arms.
        s.accept(0, 50, "<t>", false);
        let p = s.plan();
        assert_eq!(p.sample_order, vec![SampleSlot { seq_id: 0, use_grammar: true }], "grammar armed after anchor");

        // Sample one grammar token "{" → accept retires the grammar (anchor-span).
        s.accept(0, 51, "{", false);
        let p = s.plan();
        assert_eq!(p.sample_order, vec![SampleSlot { seq_id: 0, use_grammar: false }], "grammar retired after first token");
    }
}
