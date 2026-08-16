/// Prompt rendering for the messages wire format.
///
/// The envelope itself — its marker key, parsing, and building — belongs to
/// `ovata-infer-protocol`, which the language repo also depends on, so both sides
/// share one definition. What stays here is the part that is *not* wire contract:
/// turning parsed turns into a specific model's chat text.
///
/// Separated from inference.rs so it can be compiled and tested on all
/// platforms (inference.rs is Linux + --features inference only).

use crate::model::ChatTemplate;
use ovata_infer_protocol::envelope::{self, Message};

/// Render a request prompt into the model's chat format using the model profile's
/// chat template. A `_ovata_infer_protocol` messages object becomes a multi-turn chat
/// (system turn first when present); anything else is wrapped as a single user
/// turn. The trailing generation prompt is always appended so the model continues
/// as the assistant.
///
/// This replaces the daemon's former call into llama.cpp's `apply_chat_template`,
/// which the vendored fork removed — the template now lives in the profile as data
/// ([`ChatTemplate`]). Pure (no model/tokenizer), so it is unit-tested here.
pub fn format_prompt(prompt_str: &str, tmpl: &ChatTemplate) -> String {
    match extract_envelope(prompt_str) {
        Some((system, messages)) => {
            let mut turns: Vec<(&str, &str)> = Vec::with_capacity(messages.len() + 1);
            if !system.is_empty() {
                turns.push(("system", system.as_str()));
            }
            for m in &messages {
                turns.push((m.role.as_str(), m.content.as_str()));
            }
            tmpl.render(turns, true)
        }
        // Not the messages protocol → a single user turn.
        None => tmpl.render([("user", prompt_str)], true),
    }
}

/// Parse a messages envelope into (system_prompt, messages), normalized for the
/// model's chat template.
///
/// Returns `None` if the string is not an envelope (i.e. it is a plain single-turn
/// prompt that should be wrapped as a user message).
///
/// The wire-level parse is [`envelope::parse`]; the normalization on top of it is
/// this function's reason to exist. Tool role handling: `role == "tool"` is passed
/// through unchanged, but the content is wrapped in `<tool_response>` tags because
/// Qwen3's embedded Jinja2 template passes `message.content` verbatim inside
/// `<|im_start|>tool\n{content}\n<|im_end|>`. That wrapper is model-specific, which
/// is why it lives here rather than in the shared protocol crate.
pub fn extract_envelope(prompt_str: &str) -> Option<(String, Vec<Message>)> {
    let env = envelope::parse(prompt_str)?;

    let messages = env
        .messages
        .into_iter()
        .map(|m| {
            // special case: a `tool` result. We KEEP the role as "tool" (do NOT remap
            // to "user") and wrap the body in <tool_response>…</tool_response>. This
            // is load-bearing for Qwen3 — its chat template only links a tool result
            // to the preceding tool call when it sees the tool role + this wrapper.
            if m.role == "tool" {
                let content = format!("<tool_response>\n{}\n</tool_response>", m.content);
                Message { role: m.role, content }
            } else {
                // every other role (user/assistant/system) passes through untouched
                m
            }
        })
        .collect();

    Some((env.system, messages))
}

/// Strip a stop-anchor string from a response buffer in-place.
/// If the anchor is not present the buffer is unchanged.
pub fn strip_stop_anchor(buf: &mut String, stop: &str) {
    // find the first occurrence of the stop string; if present, chop everything
    // from that point on (the stop marker and anything after it). If it's not
    // found, find() is None and we leave the buffer exactly as-is.
    if let Some(pos) = buf.find(stop) {
        buf.truncate(pos);
    }
}


/// Render a request prompt into the model's chat format.
///
/// A `Result` for the caller's convenience — every request path does this as one step
/// of a `and_then` chain over tokenization, which can fail. The rendering itself
/// cannot; [`format_prompt`] above is total.
///
/// This lived in the dispatch file, which meant the router owned prompt rendering.
/// Rendering is decode's job; the router's job is choosing a path.
pub fn build_formatted_prompt(prompt_str: &str, tmpl: &ChatTemplate) -> Result<String, String> {
    Ok(format_prompt(prompt_str, tmpl))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build test envelopes through the shared builder rather than hand-writing the
    /// JSON — a literal here would reintroduce the second spelling of the marker key
    /// that moving the envelope into the protocol crate was meant to eliminate.
    fn proto(system: &str, messages: &[(&str, &str)]) -> String {
        let msgs: Vec<Message> = messages
            .iter()
            .map(|(r, c)| Message { role: (*r).to_owned(), content: (*c).to_owned() })
            .collect();
        envelope::build(system, &msgs).to_string()
    }

    // ── Protocol detection ────────────────────────────────────────────────────

    #[test]
    fn plain_text_is_not_protocol() {
        assert!(extract_envelope("Hello, what's the weather?").is_none());
    }

    #[test]
    fn json_without_protocol_field_is_not_protocol() {
        assert!(extract_envelope(r#"{"role":"user","content":"hi"}"#).is_none());
    }

    #[test]
    fn envelope_detected() {
        let json = proto("sys", &[("user", "hi")]);
        assert!(extract_envelope(&json).is_some());
    }

    // ── System field ──────────────────────────────────────────────────────────

    #[test]
    fn system_prompt_extracted() {
        let json = proto("You are Omar", &[("user", "hi")]);
        let (sys, _) = extract_envelope(&json).unwrap();
        assert_eq!(sys, "You are Omar");
    }

    #[test]
    fn empty_system_is_ok() {
        let json = proto("", &[("user", "hi")]);
        let (sys, _) = extract_envelope(&json).unwrap();
        assert_eq!(sys, "");
    }

    // ── Role and content passthrough ──────────────────────────────────────────

    #[test]
    fn user_role_preserved() {
        let json = proto("", &[("user", "Hello")]);
        let (_, msgs) = extract_envelope(&json).unwrap();
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content, "Hello");
    }

    #[test]
    fn assistant_role_preserved() {
        let json = proto("", &[("user", "ping"), ("assistant", "pong")]);
        let (_, msgs) = extract_envelope(&json).unwrap();
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].content, "pong");
    }

    // ── Tool role — the critical regression ───────────────────────────────────

    #[test]
    fn tool_role_is_not_remapped_to_user() {
        let json = proto("", &[
            ("user", "What's the weather?"),
            ("assistant", r#"<tool>{"tool_name":"get_weather","city":"SF"}</tool>"#),
            ("tool", "SF: 26°C and sunny"),
        ]);
        let (_, msgs) = extract_envelope(&json).unwrap();
        assert_eq!(
            msgs[2].role, "tool",
            "tool response must use role='tool', not 'user' — Qwen3 only \
             associates a tool result with the prior call when it sees \
             <|im_start|>tool"
        );
    }

    #[test]
    fn tool_content_wrapped_in_tool_response_tags() {
        let json = proto("", &[("tool", "SF: 26°C and sunny")]);
        let (_, msgs) = extract_envelope(&json).unwrap();
        assert_eq!(
            msgs[0].content,
            "<tool_response>\nSF: 26°C and sunny\n</tool_response>"
        );
    }

    #[test]
    fn tool_response_roundtrip_full_sequence() {
        let json = proto(
            "You are an assistant with tools",
            &[
                ("user", "What's the weather in SF?"),
                ("assistant", r#"<tool>{"tool_name":"get_weather","city":"San Francisco"}</tool>"#),
                ("tool", "Temperature: 26°C, Condition: Sunny"),
            ],
        );
        let (sys, msgs) = extract_envelope(&json).unwrap();
        assert_eq!(sys, "You are an assistant with tools");
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0], Message { role: "user".into(), content: "What's the weather in SF?".into() });
        assert_eq!(msgs[1].role, "assistant");
        assert!(msgs[1].content.contains("<tool>"));
        assert_eq!(msgs[2].role, "tool");
        assert!(msgs[2].content.starts_with("<tool_response>"));
        assert!(msgs[2].content.contains("Temperature: 26°C, Condition: Sunny"));
        assert!(msgs[2].content.ends_with("</tool_response>"));
    }

    // ── Multi-turn conversation without tools ─────────────────────────────────

    #[test]
    fn multi_turn_no_tools() {
        let json = proto("", &[
            ("user", "What is 2+2?"),
            ("assistant", "4"),
            ("user", "And 3+3?"),
        ]);
        let (_, msgs) = extract_envelope(&json).unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[2].role, "user");
        assert_eq!(msgs[2].content, "And 3+3?");
    }

    // ── Prompt formatting (chat template) ─────────────────────────────────────

    #[test]
    fn format_plain_prompt_is_single_user_turn() {
        let t = ChatTemplate::default();
        assert_eq!(
            format_prompt("What is 2+2?", &t),
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn format_messages_with_system_renders_full_chatml() {
        let json = proto("You are Omar", &[("user", "hi"), ("assistant", "hey"), ("user", "bye")]);
        let t = ChatTemplate::default();
        assert_eq!(
            format_prompt(&json, &t),
            "<|im_start|>system\nYou are Omar<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n\
             <|im_start|>assistant\nhey<|im_end|>\n\
             <|im_start|>user\nbye<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn format_empty_system_omits_system_turn() {
        let json = proto("", &[("user", "hi")]);
        let t = ChatTemplate::default();
        assert_eq!(
            format_prompt(&json, &t),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    // The tool role must render as `<|im_start|>tool` with the <tool_response>
    // wrapper intact — the load-bearing Qwen3 behavior the protocol depends on.
    #[test]
    fn format_tool_turn_uses_tool_role_and_wrapper() {
        let json = proto("", &[
            ("user", "weather?"),
            ("assistant", r#"<tool_call>{"tool_name":"get_weather"}</tool_call>"#),
            ("tool", "26C sunny"),
        ]);
        let t = ChatTemplate::default();
        let out = format_prompt(&json, &t);
        assert!(out.contains("<|im_start|>tool\n<tool_response>\n26C sunny\n</tool_response><|im_end|>\n"),
            "got: {out}");
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    // ── Stop anchor stripping ─────────────────────────────────────────────────

    #[test]
    fn stop_anchor_stripped() {
        let mut buf = "some text</tool> extra content".to_string();
        strip_stop_anchor(&mut buf, "</tool>");
        assert_eq!(buf, "some text");
    }

    #[test]
    fn stop_anchor_absent_is_noop() {
        let mut buf = "clean response with no stop anchor".to_string();
        strip_stop_anchor(&mut buf, "</tool>");
        assert_eq!(buf, "clean response with no stop anchor");
    }

    #[test]
    fn stop_anchor_at_start_clears_buffer() {
        let mut buf = "</tool>trailing".to_string();
        strip_stop_anchor(&mut buf, "</tool>");
        assert_eq!(buf, "");
    }

    #[test]
    fn stop_anchor_at_end() {
        let mut buf = r#"{"tool_name":"get_weather"}</tool>"#.to_string();
        strip_stop_anchor(&mut buf, "</tool>");
        assert_eq!(buf, r#"{"tool_name":"get_weather"}"#);
    }

    // ── parse_tag equivalent — JSON extraction from <tool> response ───────────
    //
    // Mirrors the logic in programs/lib/tools.jde parse_tag(). Tests verify that
    // jade-tree's stop anchor and depth-tracker produce inputs that parse correctly.

    fn parse_tag(response: &str) -> Option<serde_json::Value> {
        if !response.contains("<tool>") {
            return None;
        }
        let after = response.split("<tool>").nth(1)?.trim();
        let json_str = if let Some(before_close) = after.split("</tool>").next() {
            before_close.trim()
        } else {
            after
        };
        serde_json::from_str(json_str).ok()
    }

    #[test]
    fn parse_tag_with_no_closing_tag() {
        // Depth-tracker stops at '}' — no </tool> emitted.
        let v = parse_tag(r#"<tool>{"tool_name":"get_weather","city":"SF"}"#).unwrap();
        assert_eq!(v["tool_name"], "get_weather");
        assert_eq!(v["city"], "SF");
    }

    #[test]
    fn parse_tag_with_closing_tag() {
        // Stop anchor fires — </tool> present but must be stripped before json.parse.
        let v = parse_tag(r#"<tool>{"tool_name":"get_weather","city":"SF"}</tool>"#).unwrap();
        assert_eq!(v["tool_name"], "get_weather");
    }

    #[test]
    fn parse_tag_with_preamble_text() {
        // Model outputs text before the tool call.
        let v = parse_tag(r#"Let me check the weather. <tool>{"tool_name":"get_weather","city":"SF"}"#).unwrap();
        assert_eq!(v["tool_name"], "get_weather");
    }

    #[test]
    fn parse_tag_no_tool_tag_returns_none() {
        assert!(parse_tag("The weather is sunny.").is_none());
    }

    #[test]
    fn parse_tag_invalid_json_returns_none() {
        assert!(parse_tag("<tool>not json at all").is_none());
    }
}
