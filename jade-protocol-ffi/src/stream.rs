//! Pure client-side token-stream parser.
//!
//! The daemon (with `keep_anchors`) guarantees an anchored tool span appears
//! in-band as `<open>{…json…}<close>` (default `<tool>…</tool>`). This parser
//! consumes the token text incrementally and splits it into two kinds of event:
//! free **content** text, and complete **tool calls** (the JSON between the
//! anchors, with `tool_name` lifted out). The anchor markers can be split across
//! BPE tokens, so it buffers the same way the daemon does on the emit side.
//!
//! No sockets, no FFI — fully unit-tested on any platform.

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Free text outside any tool span.
    Content(String),
    /// A complete tool call. `name` is the `tool_name` field (empty if absent);
    /// `args` is the verbatim trimmed JSON object (still including `tool_name`).
    ToolCall { name: String, args: String },
}

pub struct ToolStreamParser {
    open: String,
    close: String,
    /// When false, tool-span detection is off — every token is passed through as content.
    enabled: bool,
    in_tool: bool,
    buf: String,
}

impl ToolStreamParser {
    /// Tool-span detection active, using the given open/close delimiters.
    pub fn new(open: String, close: String) -> Self {
        ToolStreamParser { open, close, enabled: true, in_tool: false, buf: String::new() }
    }

    /// Pass-through mode: tokens are emitted as content, no span detection.
    pub fn disabled() -> Self {
        ToolStreamParser {
            open: String::new(),
            close: String::new(),
            enabled: false,
            in_tool: false,
            buf: String::new(),
        }
    }

    /// Feed one token's text; returns any complete events it produced.
    pub fn push_token(&mut self, tok: &str) -> Vec<Event> {
        let mut out = Vec::new();
        if !self.enabled {
            if !tok.is_empty() {
                out.push(Event::Content(tok.to_owned()));
            }
            return out;
        }
        self.buf.push_str(tok);
        loop {
            if !self.in_tool {
                if let Some(p) = self.buf.find(&self.open) {
                    if p > 0 {
                        out.push(Event::Content(self.buf[..p].to_owned()));
                    }
                    self.buf = self.buf[p + self.open.len()..].to_owned();
                    self.in_tool = true;
                    continue;
                }
                // Open marker not present: emit the safe prefix, hold back the
                // longest suffix that could still grow into `open`.
                let keep = held_suffix_len(&self.buf, &self.open);
                let emit_len = self.buf.len() - keep;
                if emit_len > 0 {
                    out.push(Event::Content(self.buf[..emit_len].to_owned()));
                    self.buf = self.buf[emit_len..].to_owned();
                }
                break;
            } else {
                if let Some(p) = self.buf.find(&self.close) {
                    out.push(parse_tool_call(&self.buf[..p]));
                    self.buf = self.buf[p + self.close.len()..].to_owned();
                    self.in_tool = false;
                    continue;
                }
                // Inside a tool span but the close marker hasn't arrived — keep
                // buffering (a partial JSON body is not a usable event).
                break;
            }
        }
        out
    }

    /// Flush at end of stream. A held content tail becomes content; an
    /// unterminated tool body is parsed defensively (the daemon should have
    /// closed it, but never drop data).
    pub fn finish(&mut self) -> Vec<Event> {
        let mut out = Vec::new();
        if !self.enabled {
            return out;
        }
        let tail = std::mem::take(&mut self.buf);
        if self.in_tool {
            if !tail.trim().is_empty() {
                out.push(parse_tool_call(&tail));
            }
        } else if !tail.is_empty() {
            out.push(Event::Content(tail));
        }
        out
    }
}

/// Longest suffix of `buf` that is a prefix of `pat` (0 when none).
fn held_suffix_len(buf: &str, pat: &str) -> usize {
    for (start, _) in buf.char_indices() {
        let suffix = &buf[start..];
        if pat.starts_with(suffix) {
            return suffix.len();
        }
    }
    0
}

fn parse_tool_call(json: &str) -> Event {
    let trimmed = json.trim();
    let name = serde_json::from_str::<serde_json::Value>(trimmed)
        .ok()
        .and_then(|v| v.get("tool_name").and_then(|n| n.as_str()).map(str::to_owned))
        .unwrap_or_default();
    Event::ToolCall { name, args: trimmed.to_owned() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parser() -> ToolStreamParser {
        ToolStreamParser::new("<tool>".into(), "</tool>".into())
    }

    /// Feed pieces, then finish; return the flat event list.
    fn drive(p: &mut ToolStreamParser, pieces: &[&str]) -> Vec<Event> {
        let mut evs = Vec::new();
        for piece in pieces {
            evs.extend(p.push_token(piece));
        }
        evs.extend(p.finish());
        evs
    }

    #[test]
    fn plain_content_passes_through() {
        let mut p = parser();
        assert_eq!(
            drive(&mut p, &["Hello, ", "world"]),
            vec![Event::Content("Hello, ".into()), Event::Content("world".into())],
        );
    }

    #[test]
    fn disabled_passes_everything_as_content() {
        let mut p = ToolStreamParser::disabled();
        assert_eq!(
            drive(&mut p, &["<tool>", "{\"x\":1}", "</tool>"]),
            vec![
                Event::Content("<tool>".into()),
                Event::Content("{\"x\":1}".into()),
                Event::Content("</tool>".into()),
            ],
        );
    }

    #[test]
    fn single_tool_call_extracted() {
        let mut p = parser();
        let evs = drive(&mut p, &["<tool>", r#"{"tool_name": "get_weather", "city": "SF"}"#, "</tool>"]);
        assert_eq!(
            evs,
            vec![Event::ToolCall {
                name: "get_weather".into(),
                args: r#"{"tool_name": "get_weather", "city": "SF"}"#.into(),
            }],
        );
    }

    #[test]
    fn content_before_and_after_tool_call() {
        let mut p = parser();
        let evs = drive(&mut p, &["Let me check. ", "<tool>", r#"{"tool_name":"t"}"#, "</tool>", " Done."]);
        assert_eq!(
            evs,
            vec![
                Event::Content("Let me check. ".into()),
                Event::ToolCall { name: "t".into(), args: r#"{"tool_name":"t"}"#.into() },
                Event::Content(" Done.".into()),
            ],
        );
    }

    #[test]
    fn open_marker_split_across_tokens() {
        // "<too" then "l>" — the partial "<too" must be held, not leaked as content.
        let mut p = parser();
        let evs = drive(&mut p, &["a", "<too", "l>", r#"{"tool_name":"t"}"#, "</tool>"]);
        assert_eq!(
            evs,
            vec![
                Event::Content("a".into()),
                Event::ToolCall { name: "t".into(), args: r#"{"tool_name":"t"}"#.into() },
            ],
        );
    }

    #[test]
    fn close_marker_split_across_tokens() {
        let mut p = parser();
        let evs = drive(&mut p, &["<tool>", r#"{"tool_name":"t"}"#, "</", "tool>", "after"]);
        assert_eq!(
            evs,
            vec![
                Event::ToolCall { name: "t".into(), args: r#"{"tool_name":"t"}"#.into() },
                Event::Content("after".into()),
            ],
        );
    }

    #[test]
    fn tool_call_fused_in_one_token() {
        let mut p = parser();
        let evs = drive(&mut p, &[r#"<tool>{"tool_name":"t","a":1}</tool>"#]);
        assert_eq!(
            evs,
            vec![Event::ToolCall { name: "t".into(), args: r#"{"tool_name":"t","a":1}"#.into() }],
        );
    }

    #[test]
    fn unterminated_tool_span_recovered_at_finish() {
        // Daemon normally closes the span; if it doesn't, don't drop the JSON.
        let mut p = parser();
        let evs = drive(&mut p, &["<tool>", r#"{"tool_name":"t"}"#]);
        assert_eq!(
            evs,
            vec![Event::ToolCall { name: "t".into(), args: r#"{"tool_name":"t"}"#.into() }],
        );
    }

    #[test]
    fn missing_tool_name_yields_empty_name() {
        let mut p = parser();
        let evs = drive(&mut p, &["<tool>", r#"{"x":1}"#, "</tool>"]);
        assert_eq!(evs, vec![Event::ToolCall { name: String::new(), args: r#"{"x":1}"#.into() }]);
    }

    #[test]
    fn held_partial_open_at_finish_is_content() {
        // A "<too" that never completes is real output at end of stream.
        let mut p = parser();
        let evs = drive(&mut p, &["done<too"]);
        assert_eq!(evs, vec![Event::Content("done".into()), Event::Content("<too".into())]);
    }
}
