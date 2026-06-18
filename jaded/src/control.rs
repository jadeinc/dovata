//! Pure constrained-decoding control state machine.
//!
//! This is the logic that used to live tangled inside the generation loop in
//! `inference.rs` — anchor-span triggering, split-token stop-anchor buffering,
//! and JSON depth tracking to retire the grammar. Pulled out here so it can be
//! unit-tested without llama.cpp, a GPU, or a model.
//!
//! ## How the engine drives it
//!
//! Per generated token:
//!   1. [`DecodeControl::use_grammar`] — pick grammar vs free sampler *before* sampling.
//!   2. [`DecodeControl::notice_grammar_sampled`] — call iff the grammar sampler was used.
//!   3. [`DecodeControl::on_piece`] — feed the decoded token text; emit what it returns.
//!
//! Once generation ends:
//!   4. [`DecodeControl::finish`] — flush any buffered tail (an incomplete stop anchor).
//!
//! ## The two grammar lifetimes
//!
//! * **Full-grammar primitives** (int/float/bool — detected heuristically by
//!   `[0-9]` / `"true"` in the GBNF, with *no* start anchor): the grammar runs
//!   for the entire output and is never retired.
//! * **Anchor-span complex types** (struct/dict/array behind a start anchor like
//!   `<tool>`): the grammar enforces only the first token, then the depth tracker
//!   retires it the instant the top-level JSON object/array closes — so the model
//!   emits the `stop_anchor` (e.g. `</tool>`) and EOG without grammar overhead.

use jade_protocol::InferenceRequest;

#[derive(Debug, Clone)]
pub struct DecodeControl {
    /// Start of the grammar-enforcement span; `None` = enforce from token 1.
    anchor: Option<String>,
    /// True once `anchor` has been seen in the output (or immediately if no anchor).
    anchor_triggered: bool,
    /// End of the grammar-enforcement span; stripped from client output unless
    /// `keep_anchors` is set.
    stop_anchor: Option<String>,
    /// True once `stop_anchor` has fired (grammar retired, emit freely after).
    stop_anchor_seen: bool,
    /// When set, the `stop_anchor` boundary is made observable in-band rather than
    /// stripped: it is emitted to the client and, if the model retired the grammar
    /// via JSON depth-close without emitting it, synthesized at the close point.
    keep_anchors: bool,
    /// True once the closing boundary has been emitted (either the model's own
    /// `stop_anchor` or a synthesized one). Guards against double-emitting when the
    /// model emits `</tool>` after we already synthesized the boundary.
    boundary_emitted: bool,
    /// All decoded text so far — used only for anchor substring detection.
    emitted_so_far: String,
    /// Text buffered while confirming it is not (part of) the stop anchor.
    stop_pending: String,
    /// True when the grammar covers the full output (primitive, no anchor).
    anchor_is_full_grammar: bool,
    /// Whether the grammar sampler should still be used.
    anchor_active: bool,
    /// Whether JSON depth tracking is retiring the grammar at span close.
    do_depth_track: bool,
    json_depth: usize,
    json_in_string: bool,
    json_escape: bool,
}

impl DecodeControl {
    pub fn new(req: &InferenceRequest) -> Self {
        let has_grammar = req.grammar.is_some();
        let anchor = req.anchor.clone();
        let has_anchor = anchor.is_some();
        // A full-primitive grammar (covers the whole output) must never have its
        // grammar retired by depth tracking — guard via this heuristic. Only
        // meaningful when there is no start anchor.
        let anchor_is_full_grammar = anchor.is_none()
            && req
                .grammar
                .as_deref()
                .map(|g| g.contains("[0-9]") || g.contains("\"true\""))
                .unwrap_or(false);
        DecodeControl {
            anchor_triggered: anchor.is_none(),
            anchor,
            stop_anchor: req.stop_anchor.clone(),
            stop_anchor_seen: false,
            keep_anchors: req.keep_anchors,
            boundary_emitted: false,
            emitted_so_far: String::new(),
            stop_pending: String::new(),
            anchor_is_full_grammar,
            anchor_active: has_grammar,
            // Depth tracking only when a start anchor gates an anchor-span grammar.
            do_depth_track: has_grammar && !anchor_is_full_grammar && has_anchor,
            json_depth: 0,
            json_in_string: false,
            json_escape: false,
        }
    }

    /// Whether the next token should be sampled with the grammar sampler.
    pub fn use_grammar(&self) -> bool {
        self.anchor_active && self.anchor_triggered
    }

    /// Call immediately after sampling *with the grammar sampler*. In anchor-span
    /// mode the grammar enforces only the first token, then is retired (the depth
    /// tracker takes over). Full-primitive grammars keep running.
    pub fn notice_grammar_sampled(&mut self) {
        if !self.anchor_is_full_grammar && self.anchor.is_some() {
            self.anchor_active = false;
        }
    }

    /// Feed the decoded text of one generated token. Returns the text (if any)
    /// to forward to the client, and updates internal state (anchor triggering,
    /// grammar retirement, depth tracking).
    pub fn on_piece(&mut self, piece: &str) -> Option<String> {
        self.emitted_so_far.push_str(piece);

        // Start-anchor detection: enable grammar enforcement once it appears.
        if !self.anchor_triggered {
            if let Some(a) = self.anchor.clone() {
                if self.emitted_so_far.contains(&a) {
                    self.anchor_triggered = true;
                    // Guard: if the token that completed the anchor already opened
                    // the JSON body, retire the grammar now so it doesn't force a
                    // second `{` and produce `<tool>{{...}}`.
                    if self.do_depth_track && self.anchor_active {
                        let a_end = self
                            .emitted_so_far
                            .find(&a)
                            .map(|p| p + a.len())
                            .unwrap_or(self.emitted_so_far.len());
                        let after = &self.emitted_so_far[a_end..];
                        if after.contains('{') || after.contains('[') {
                            self.anchor_active = false;
                        }
                    }
                }
            }
        }

        let mut emit = self.process_stop_anchor(piece);

        // Depth tracking: retire grammar when the top-level JSON span closes.
        // Only after the anchor has triggered — preamble braces must not count.
        if self.do_depth_track && self.anchor_triggered {
            for ch in piece.chars() {
                if self.json_escape {
                    self.json_escape = false;
                    continue;
                }
                match ch {
                    '\\' if self.json_in_string => self.json_escape = true,
                    '"' => self.json_in_string = !self.json_in_string,
                    '{' | '[' if !self.json_in_string => self.json_depth += 1,
                    '}' | ']' if !self.json_in_string => {
                        if self.json_depth > 0 {
                            self.json_depth -= 1;
                        }
                        if self.json_depth == 0 {
                            self.anchor_active = false;
                            self.do_depth_track = false;
                            // Make the span boundary observable: the model frequently
                            // retires the grammar at the closing brace and stops (or
                            // hits EOG) WITHOUT ever emitting the stop anchor, so a
                            // client relying on `</tool>` to delimit the span would
                            // never see it. Synthesize it here. `stop_pending` keeps
                            // buffering afterward so a model-emitted duplicate is
                            // stripped (see `process_stop_anchor`).
                            //
                            // Note: appended at piece end, so in the rare case the
                            // closing brace is fused with trailing text in one BPE
                            // token (`}foo`), the boundary lands after that text.
                            if self.keep_anchors && !self.boundary_emitted && !self.stop_anchor_seen {
                                if let Some(stop) = self.stop_anchor.clone() {
                                    let mut out = emit.take().unwrap_or_default();
                                    out.push_str(&stop);
                                    emit = Some(out);
                                    self.boundary_emitted = true;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        emit
    }

    /// Prefix-aware stop-anchor handling. Buffers text until certain it is not
    /// (part of) the stop anchor — prevents split-token leaks where e.g. `</`
    /// and `tool>` arrive in separate tokens. Returns the confirmed-safe text.
    fn process_stop_anchor(&mut self, piece: &str) -> Option<String> {
        let stop = match self.stop_anchor.clone() {
            None => return Some(piece.to_owned()),
            Some(s) => s,
        };
        if self.stop_anchor_seen {
            // Grammar already retired — emit immediately, no buffering needed.
            return Some(piece.to_owned());
        }
        self.stop_pending.push_str(piece);
        if let Some(stop_pos) = self.stop_pending.find(&stop) {
            // Stop anchor complete — retire grammar.
            let after_pos = stop_pos + stop.len();
            let emit = if self.keep_anchors {
                if self.boundary_emitted {
                    // We already synthesized the boundary at depth-close; this is the
                    // model's own duplicate anchor — strip just the anchor, keep the
                    // text on either side.
                    let mut out = self.stop_pending[..stop_pos].to_owned();
                    out.push_str(&self.stop_pending[after_pos..]);
                    (!out.is_empty()).then_some(out)
                } else {
                    // First boundary — emit everything including the anchor, in-band,
                    // so the client can delimit the span by string match.
                    self.boundary_emitted = true;
                    Some(self.stop_pending.clone())
                }
            } else {
                // Legacy: strip the anchor and everything after it in this buffer.
                (stop_pos > 0).then(|| self.stop_pending[..stop_pos].to_owned())
            };
            self.stop_pending.clear();
            self.stop_anchor_seen = true;
            self.anchor_active = false;
            self.do_depth_track = false;
            emit
        } else {
            // Keep the longest suffix that is a strict prefix of `stop`; emit the rest.
            let mut keep = 0usize;
            for (start, _) in self.stop_pending.char_indices() {
                let suffix = &self.stop_pending[start..];
                if stop.starts_with(suffix) {
                    keep = suffix.len();
                    break;
                }
            }
            let emit_len = self.stop_pending.len() - keep;
            if emit_len > 0 {
                let emit = self.stop_pending[..emit_len].to_owned();
                self.stop_pending = self.stop_pending[emit_len..].to_owned();
                Some(emit)
            } else {
                None
            }
        }
    }

    /// Flush any buffered tail after generation ends (an incomplete stop anchor
    /// that never completed is real output: EOG / max_tokens / depth-close).
    pub fn finish(self) -> Option<String> {
        if self.stop_pending.is_empty() {
            None
        } else {
            Some(self.stop_pending)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        }
    }

    /// Feed a sequence of pieces, returning the concatenated client output.
    fn drive(ctrl: &mut DecodeControl, pieces: &[&str]) -> String {
        let mut out = String::new();
        for p in pieces {
            if let Some(s) = ctrl.on_piece(p) {
                out.push_str(&s);
            }
        }
        out
    }

    // ── No constraints — plain passthrough ──────────────────────────────────

    #[test]
    fn unconstrained_passes_pieces_through() {
        let mut c = DecodeControl::new(&req(None, None, None));
        assert!(!c.use_grammar());
        assert_eq!(drive(&mut c, &["Hello", ", ", "world"]), "Hello, world");
        assert_eq!(c.finish(), None);
    }

    // ── Grammar lifetimes ───────────────────────────────────────────────────

    #[test]
    fn full_primitive_grammar_never_retires() {
        // Bool grammar, no anchor → full grammar, stays active the whole way.
        let mut c = DecodeControl::new(&req(Some(r#"root ::= "true" | "false""#), None, None));
        assert!(c.use_grammar());
        c.on_piece("true");
        assert!(c.use_grammar(), "primitive grammar must stay active");
    }

    #[test]
    fn anchor_span_grammar_retires_after_first_token() {
        let mut c = DecodeControl::new(&req(Some(r#"root ::= "{" [^}]* "}""#), Some("<tool>"), Some("</tool>")));
        // Before the anchor appears, grammar is gated off.
        assert!(!c.use_grammar());
        c.on_piece("Let me check. ");
        assert!(!c.use_grammar());
        // Anchor appears → grammar enables.
        c.on_piece("<tool>");
        assert!(c.use_grammar());
        // Engine samples one grammar token and reports it; grammar retires.
        c.notice_grammar_sampled();
        assert!(!c.use_grammar());
    }

    #[test]
    fn anchor_completing_token_that_opens_json_retires_immediately() {
        // The token that completes "<tool>" also contains "{" — grammar must
        // retire on detection so it doesn't force a duplicate "{".
        let mut c = DecodeControl::new(&req(Some(r#"root ::= "{" [^}]* "}""#), Some("<tool>"), Some("</tool>")));
        c.on_piece("<tool>{");
        assert!(!c.use_grammar(), "grammar should retire when anchor token already opened JSON");
    }

    // ── Depth tracking ──────────────────────────────────────────────────────

    #[test]
    fn depth_tracker_retires_grammar_when_json_closes() {
        let mut c = DecodeControl::new(&req(Some(r#"root ::= "{" [^}]* "}""#), Some("<tool>"), Some("</tool>")));
        c.on_piece("<tool>");
        c.notice_grammar_sampled();
        c.on_piece(r#"{"k": "#);
        c.on_piece(r#"{"n": 1}"#); // nested open+close, depth back to 1
        assert!(c.do_depth_track || !c.do_depth_track); // still tracking until top closes
        c.on_piece("}"); // top-level close → depth 0
        assert!(!c.use_grammar());
    }

    #[test]
    fn braces_inside_json_strings_do_not_close_span() {
        let mut c = DecodeControl::new(&req(Some(r#"root ::= "{" [^}]* "}""#), Some("<tool>"), Some("</tool>")));
        c.on_piece("<tool>");
        c.notice_grammar_sampled();
        // A "}" inside a string must not be counted as a close.
        c.on_piece(r#"{"msg": "a } brace"#);
        // depth is still 1 (the string brace was ignored); not yet closed.
        c.on_piece(r#""}"#); // close string then close object
        assert!(!c.use_grammar());
    }

    // ── Stop-anchor buffering ───────────────────────────────────────────────

    #[test]
    fn stop_anchor_split_across_tokens_not_leaked() {
        let mut c = DecodeControl::new(&req(None, None, Some("</tool>")));
        // Body emitted, then the stop anchor arrives in two BPE pieces.
        let out = drive(&mut c, &[r#"{"x":1}"#, "</", "tool>"]);
        assert_eq!(out, r#"{"x":1}"#, "stop anchor must not leak to the client");
        assert_eq!(c.finish(), None);
    }

    #[test]
    fn stop_anchor_in_single_token_stripped() {
        let mut c = DecodeControl::new(&req(None, None, Some("</tool>")));
        let out = drive(&mut c, &[r#"{"x":1}</tool>"#]);
        assert_eq!(out, r#"{"x":1}"#);
    }

    #[test]
    fn text_after_stop_anchor_emitted_freely() {
        let mut c = DecodeControl::new(&req(None, None, Some("</tool>")));
        let out = drive(&mut c, &["a", "</tool>", "trailing"]);
        assert_eq!(out, "atrailing");
    }

    #[test]
    fn buffered_prefix_that_is_not_stop_anchor_is_flushed() {
        // "</" looks like the start of "</tool>" but generation ends first.
        let mut c = DecodeControl::new(&req(None, None, Some("</tool>")));
        let mut out = String::new();
        if let Some(s) = c.on_piece("done</") {
            out.push_str(&s);
        }
        // "done" is safe to emit; "</" is held back pending confirmation.
        assert_eq!(out, "done");
        // Generation ends — the held suffix is real output.
        assert_eq!(c.finish(), Some("</".to_owned()));
    }

    #[test]
    fn partial_false_alarm_then_continue() {
        // "<" then a non-matching char: the "<" must be released, not dropped.
        let mut c = DecodeControl::new(&req(None, None, Some("</tool>")));
        let out = drive(&mut c, &["a<", "b"]);
        assert_eq!(out, "a<b");
    }

    // ── keep_anchors: observable tool-span boundary ───────────────────────────
    //
    // With keep_anchors the closing boundary is made visible in-band so a client
    // can delimit `<tool>…</tool>` by string match. The daemon synthesizes the
    // boundary at JSON depth-close (the model often retires the grammar at `}`
    // and never emits `</tool>`), and strips a model-emitted duplicate.

    fn req_ka(grammar: Option<&str>, anchor: Option<&str>, stop: Option<&str>) -> InferenceRequest {
        let mut r = req(grammar, anchor, stop);
        r.keep_anchors = true;
        r
    }

    const TOOL_GBNF: &str = r#"root ::= "{" [^}]* "}""#;

    #[test]
    fn keep_anchors_synthesizes_boundary_at_depth_close() {
        // Model emits the JSON and stops at `}` — never emits </tool>.
        let mut c = DecodeControl::new(&req_ka(Some(TOOL_GBNF), Some("<tool>"), Some("</tool>")));
        let out = drive(&mut c, &["<tool>", r#"{"tool_name": "get_weather", "city": "SF"}"#]);
        assert_eq!(out, r#"<tool>{"tool_name": "get_weather", "city": "SF"}</tool>"#);
        assert_eq!(c.finish(), None);
    }

    #[test]
    fn keep_anchors_no_double_when_close_and_anchor_split() {
        // `}` arrives, boundary is synthesized, then the model emits its own
        // </tool> in a later token — it must be stripped, not doubled.
        let mut c = DecodeControl::new(&req_ka(Some(TOOL_GBNF), Some("<tool>"), Some("</tool>")));
        let out = drive(&mut c, &["<tool>", r#"{"x": 1"#, "}", "</tool>"]);
        assert_eq!(out, r#"<tool>{"x": 1}</tool>"#);
        assert_eq!(c.finish(), None);
    }

    #[test]
    fn keep_anchors_no_double_when_close_and_anchor_fused() {
        // `}</tool>` in one token: process_stop_anchor includes the anchor and
        // retires depth tracking, so no synthesis happens.
        let mut c = DecodeControl::new(&req_ka(Some(TOOL_GBNF), Some("<tool>"), Some("</tool>")));
        let out = drive(&mut c, &["<tool>", r#"{"x": 1}</tool>"#]);
        assert_eq!(out, r#"<tool>{"x": 1}</tool>"#);
        assert_eq!(c.finish(), None);
    }

    #[test]
    fn keep_anchors_free_text_after_boundary_flows() {
        // After the synthesized boundary, trailing free text is emitted normally.
        let mut c = DecodeControl::new(&req_ka(Some(TOOL_GBNF), Some("<tool>"), Some("</tool>")));
        let out = drive(&mut c, &["<tool>", r#"{"x": 1}"#, " all done"]);
        assert_eq!(out, r#"<tool>{"x": 1}</tool> all done"#);
        assert_eq!(c.finish(), None);
    }

    #[test]
    fn keep_anchors_false_does_not_synthesize_at_depth_close() {
        // Legacy: no keep_anchors → boundary is stripped, none synthesized.
        let mut c = DecodeControl::new(&req(Some(TOOL_GBNF), Some("<tool>"), Some("</tool>")));
        let out = drive(&mut c, &["<tool>", r#"{"x": 1}"#]);
        assert_eq!(out, r#"<tool>{"x": 1}"#);
        assert_eq!(c.finish(), None);
    }
}
