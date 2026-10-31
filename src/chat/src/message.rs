//! One turn of a conversation, in the shape chat templates read.
//!
//! Templates are written against the OpenAI-style message dicts `transformers` passes
//! them: `role` and `content`, plus whatever else a turn carries, such as `tool_calls` on
//! an assistant turn or `reasoning_content` for a thinking model. [`Message`] keeps the
//! two fields every template reads as fields and everything else in a map, so a new
//! model's extra key needs no change here.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// One turn of a conversation.
///
/// Serializes to the dict a template sees as `message`: `role`, `content`, and every
/// entry of `extra` at the same level. It deserializes from the same shape, so an
/// OpenAI-style JSON message list can be read straight into a `Vec<Message>`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// `system`, `user`, `assistant`, `tool`, or any role the model's template knows.
    pub role: String,
    /// The text of the turn. `None` serializes to an explicit null rather than being
    /// left out, because templates tell the two apart: DeepSeek V3 renders an assistant
    /// turn's tool calls only when `content is none`, and a turn with an empty string
    /// instead silently loses them. A missing or null `content` in JSON reads as `None`.
    #[serde(default)]
    pub content: Option<String>,
    /// Every other key, such as `tool_calls`, `tool_call_id`, `name` or
    /// `reasoning_content`. Must not hold `role` or `content`; rendering refuses a
    /// message that does, since the two copies would disagree.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Message {
    /// A turn with the given role and text.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Message {
        Message { role: role.into(), content: Some(content.into()), extra: Map::new() }
    }

    /// A system turn.
    pub fn system(content: impl Into<String>) -> Message {
        Message::new("system", content)
    }

    /// A user turn.
    pub fn user(content: impl Into<String>) -> Message {
        Message::new("user", content)
    }

    /// An assistant turn with text.
    pub fn assistant(content: impl Into<String>) -> Message {
        Message::new("assistant", content)
    }

    /// An assistant turn that calls tools and says nothing, with `content` null.
    ///
    /// `calls` is the OpenAI-style list, each entry
    /// `{"type": "function", "function": {"name": ..., "arguments": ...}}`. Templates
    /// differ on whether `arguments` should be a JSON string or an object: DeepSeek V3
    /// concatenates it as a string, Qwen3-Coder iterates it as a map, and Qwen3 takes
    /// either. Pass what the model's template expects.
    pub fn tool_calls(calls: Value) -> Message {
        let mut m = Message { role: "assistant".into(), content: None, extra: Map::new() };
        m.extra.insert("tool_calls".into(), calls);
        m
    }

    /// A tool's result, answering a call.
    pub fn tool(content: impl Into<String>) -> Message {
        Message::new("tool", content)
    }

    /// Add an extra key, such as `tool_call_id` or `reasoning_content`.
    #[must_use]
    pub fn with(mut self, key: impl Into<String>, value: Value) -> Message {
        self.extra.insert(key.into(), value);
        self
    }

    /// Refuse an `extra` map that shadows a field. With `serde(flatten)` it would
    /// serialize a second `role` or `content` key, and which one a template saw would
    /// depend on map internals.
    pub(crate) fn check(&self) -> Result<(), String> {
        for field in ["role", "content"] {
            if self.extra.contains_key(field) {
                return Err(format!("extra holds `{field}`, which is a field of Message"));
            }
        }
        Ok(())
    }
}
