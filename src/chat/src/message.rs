//! Author: Joe Ricks
//! File: src/chat/src/message.rs
//! Description: Defines `Message`, one turn of a conversation.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Purpose: This struct holds one turn of a conversation.
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
    /// Purpose: This function makes a turn with any role.
    ///
    /// Inputs:
    /// - `role`: the role, such as `user`.
    /// - `content`: the turn's text.
    ///
    /// Outputs: the new `Message`.
    pub fn new(
        role: impl Into<String>,
        content: impl Into<String>,
    ) -> Message
    {
        Message { role: role.into(), content: Some(content.into()), extra: Map::new() }
    }

    /// Purpose: This function makes a system turn.
    ///
    /// Inputs:
    /// - `content`: the system prompt.
    ///
    /// Outputs: the new `Message`.
    pub fn system(
        content: impl Into<String>,
    ) -> Message
    {
        Message::new("system", content)
    }

    /// Purpose: This function makes a user turn.
    ///
    /// Inputs:
    /// - `content`: what the person said.
    ///
    /// Outputs: the new `Message`.
    pub fn user(
        content: impl Into<String>,
    ) -> Message
    {
        Message::new("user", content)
    }

    /// Purpose: This function makes an assistant turn with text.
    ///
    /// Inputs:
    /// - `content`: the model's reply.
    ///
    /// Outputs: the new `Message`.
    pub fn assistant(
        content: impl Into<String>,
    ) -> Message
    {
        Message::new("assistant", content)
    }

    /// Purpose: This function makes an assistant turn that calls tools, with null
    /// content.
    ///
    /// Inputs:
    /// - `calls`: the OpenAI-style list of tool calls.
    ///
    /// Outputs: the new `Message`.
    pub fn tool_calls(
        calls: Value,
    ) -> Message
    {
        let mut m = Message { role: "assistant".into(), content: None, extra: Map::new() };
        m.extra.insert("tool_calls".into(), calls);
        m
    }

    /// Purpose: This function makes a tool turn that holds a tool's result.
    ///
    /// Inputs:
    /// - `content`: the tool's result.
    ///
    /// Outputs: the new `Message`.
    pub fn tool(
        content: impl Into<String>,
    ) -> Message
    {
        Message::new("tool", content)
    }

    /// Purpose: This function adds an extra key to a turn, such as `tool_call_id`.
    ///
    /// Inputs:
    /// - `key`: the key's name.
    /// - `value`: its value.
    ///
    /// Outputs: the same `Message` with the key set.
    #[must_use]
    pub fn with(
        mut self,
        key: impl Into<String>,
        value: Value,
    ) -> Message
    {
        self.extra.insert(key.into(), value);
        self
    }

    /// Purpose: This function checks that a turn's extra keys do not repeat `role` or
    /// `content`.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `Ok(())`, or an error naming the repeated key.
    pub(crate) fn check(&self) -> Result<(), String>
    {
        for field in ["role", "content"] {
            if self.extra.contains_key(field) {
                return Err(format!("extra holds `{field}`, which is a field of Message"));
            }
        }
        Ok(())
    }
}
