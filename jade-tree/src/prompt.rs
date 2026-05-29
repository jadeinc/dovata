/// Pure protocol-parsing logic for the jade messages wire format.
///
/// Separated from inference.rs so it can be compiled and tested on all
/// platforms (inference.rs is Linux + --features inference only).

use serde_json::Value;

#[derive(Debug, PartialEq, Clone)]
pub struct JadeMessage {
    pub role: String,
    pub content: String,
}

/// Parse a jade messages protocol payload into (system_prompt, messages).
///
/// Returns `None` if the string is not in jade protocol format (i.e. it is a
/// plain single-turn prompt that should be wrapped as a user message).
///
/// Tool role handling: `role == "tool"` is passed through unchanged. The
/// content is wrapped in `<tool_response>` tags because Qwen3's embedded
/// Jinja2 template passes `message.content` verbatim inside
/// `<|im_start|>tool\n{content}\n<|im_end|>`.
pub fn extract_jade_protocol(prompt_str: &str) -> Option<(String, Vec<JadeMessage>)> {
    if !prompt_str.starts_with('{') {
        return None;
    }
    let v: Value = serde_json::from_str(prompt_str).ok()?;
    if v.get("_jade_protocol").and_then(|p| p.as_str()) != Some("messages") {
        return None;
    }
    let system = v.get("system")
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_owned();
    let raw_msgs = v.get("messages").and_then(|m| m.as_array())?;
    let mut messages = Vec::with_capacity(raw_msgs.len());
    for msg in raw_msgs {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
        let (final_role, final_content) = if role == "tool" {
            (role.to_owned(), format!("<tool_response>\n{content}\n</tool_response>"))
        } else {
            (role.to_owned(), content.to_owned())
        };
        messages.push(JadeMessage { role: final_role, content: final_content });
    }
    Some((system, messages))
}

/// Strip a stop-anchor string from a response buffer in-place.
/// If the anchor is not present the buffer is unchanged.
pub fn strip_stop_anchor(buf: &mut String, stop: &str) {
    if let Some(pos) = buf.find(stop) {
        buf.truncate(pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proto(system: &str, messages: &[(&str, &str)]) -> String {
        let msgs: Vec<Value> = messages
            .iter()
            .map(|(r, c)| serde_json::json!({"role": r, "content": c}))
            .collect();
        serde_json::json!({
            "_jade_protocol": "messages",
            "system": system,
            "messages": msgs,
        })
        .to_string()
    }

    // ── Protocol detection ────────────────────────────────────────────────────

    #[test]
    fn plain_text_is_not_protocol() {
        assert!(extract_jade_protocol("Hello, what's the weather?").is_none());
    }

    #[test]
    fn json_without_protocol_field_is_not_protocol() {
        assert!(extract_jade_protocol(r#"{"role":"user","content":"hi"}"#).is_none());
    }

    #[test]
    fn jade_protocol_detected() {
        let json = proto("sys", &[("user", "hi")]);
        assert!(extract_jade_protocol(&json).is_some());
    }

    // ── System field ──────────────────────────────────────────────────────────

    #[test]
    fn system_prompt_extracted() {
        let json = proto("You are Omar", &[("user", "hi")]);
        let (sys, _) = extract_jade_protocol(&json).unwrap();
        assert_eq!(sys, "You are Omar");
    }

    #[test]
    fn empty_system_is_ok() {
        let json = proto("", &[("user", "hi")]);
        let (sys, _) = extract_jade_protocol(&json).unwrap();
        assert_eq!(sys, "");
    }

    // ── Role and content passthrough ──────────────────────────────────────────

    #[test]
    fn user_role_preserved() {
        let json = proto("", &[("user", "Hello")]);
        let (_, msgs) = extract_jade_protocol(&json).unwrap();
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content, "Hello");
    }

    #[test]
    fn assistant_role_preserved() {
        let json = proto("", &[("user", "ping"), ("assistant", "pong")]);
        let (_, msgs) = extract_jade_protocol(&json).unwrap();
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
        let (_, msgs) = extract_jade_protocol(&json).unwrap();
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
        let (_, msgs) = extract_jade_protocol(&json).unwrap();
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
        let (sys, msgs) = extract_jade_protocol(&json).unwrap();
        assert_eq!(sys, "You are an assistant with tools");
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0], JadeMessage { role: "user".into(), content: "What's the weather in SF?".into() });
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
        let (_, msgs) = extract_jade_protocol(&json).unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[2].role, "user");
        assert_eq!(msgs[2].content, "And 3+3?");
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

    fn parse_tag(response: &str) -> Option<Value> {
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
