//! Swappable, per-model token profiles for JADE LLM inference.
//!
//! ## Why this exists
//!
//! Tool-call delimiters (`<tool>…</tool>`), turn-taking markers, and other
//! special tokens are properties of the **model**, not of Jade or the wire
//! protocol. They ship with the `.gguf` and differ between models. Hardcoding
//! them anywhere — in Jade programs, in the protocol, in the daemon — means
//! swapping the model silently breaks tool calling.
//!
//! A [`ModelProfile`] captures that model-specific vocabulary as data. The wire
//! protocol stays model-agnostic (it carries opaque text + frames); the language
//! layer (`use llm`) loads the profile for whatever model the daemon reports and
//! uses it to construct requests and interpret the streamed special tokens. Swap
//! the model → swap the profile file → no Jade program changes.
//!
//! ## Format
//!
//! Profiles are TOML. See `profiles/qwen3-coder-30b.toml` for a worked example.
//! Load one with [`ModelProfile::from_toml`], or use a [`Registry`] to pick the
//! right profile for a model name reported by the daemon's `Meta` frame.

use serde::Deserialize;

mod stream;
pub use stream::{Event, ToolStreamParser};

/// A model's token/turn-taking vocabulary.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelProfile {
    /// Format version, for forward-compatible evolution.
    pub schema_version: u32,
    /// Human-readable model name this profile targets.
    pub model: String,
    /// Glob patterns (against the daemon-reported model name) this profile
    /// applies to. `*` matches any sequence. First matching profile wins.
    #[serde(default, rename = "match")]
    pub match_patterns: Vec<String>,
    /// How the model emits a tool invocation.
    pub tool_call: ToolCall,
    /// How a tool result is rendered back into the conversation.
    pub tool_result: ToolResult,
    /// Additional special-token spans the model may stream (e.g. reasoning),
    /// surfaced as structured events instead of raw content.
    #[serde(default, rename = "span")]
    pub spans: Vec<Span>,
}

/// Delimiters + parsing convention for a tool call the model emits.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolCall {
    /// Opening delimiter, e.g. `<tool>`.
    pub open: String,
    /// Closing delimiter, e.g. `</tool>`.
    pub close: String,
    /// JSON field inside the call object naming the tool, e.g. `tool_name`.
    #[serde(default = "default_name_field")]
    pub name_field: String,
}

/// How a tool result is presented back to the model on the next turn.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolResult {
    /// Message role to use for the result (e.g. `tool`).
    pub role: String,
    /// Text prepended to the result content.
    #[serde(default)]
    pub wrap_prefix: String,
    /// Text appended to the result content.
    #[serde(default)]
    pub wrap_suffix: String,
}

/// A generic special-token span the model streams (beyond the tool call).
#[derive(Debug, Clone, Deserialize)]
pub struct Span {
    /// Semantic label, e.g. `reasoning`.
    pub tag: String,
    /// Opening delimiter, e.g. `<think>`.
    pub open: String,
    /// Closing delimiter, e.g. `</think>`.
    pub close: String,
}

fn default_name_field() -> String {
    "tool_name".to_owned()
}

impl ModelProfile {
    /// Parse and validate a profile from TOML.
    pub fn from_toml(s: &str) -> Result<Self, ProfileError> {
        let p: ModelProfile = toml::from_str(s).map_err(|e| ProfileError::Parse(e.to_string()))?;
        p.validate()?;
        Ok(p)
    }

    /// Reject structurally-valid-but-meaningless profiles (empty delimiters).
    pub fn validate(&self) -> Result<(), ProfileError> {
        if self.tool_call.open.is_empty() || self.tool_call.close.is_empty() {
            return Err(ProfileError::Invalid("tool_call.open/close must be non-empty"));
        }
        if self.tool_result.role.is_empty() {
            return Err(ProfileError::Invalid("tool_result.role must be non-empty"));
        }
        for s in &self.spans {
            if s.open.is_empty() || s.close.is_empty() || s.tag.is_empty() {
                return Err(ProfileError::Invalid("span tag/open/close must be non-empty"));
            }
        }
        Ok(())
    }

    /// Whether this profile applies to `model_name` (any `match` glob matches,
    /// or the exact `model` name matches).
    pub fn matches(&self, model_name: &str) -> bool {
        self.model == model_name
            || self.match_patterns.iter().any(|pat| glob_match(pat, model_name))
    }

    /// Render a raw tool result into the form the model expects in its next turn.
    pub fn wrap_tool_result(&self, content: &str) -> String {
        format!("{}{}{}", self.tool_result.wrap_prefix, content, self.tool_result.wrap_suffix)
    }
}

/// A set of profiles, selectable by model name.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    profiles: Vec<ModelProfile>,
}

/// Profiles compiled into the crate (the defaults shipped in `profiles/`).
const BUILTIN_PROFILES: &[&str] = &[include_str!("../profiles/qwen3-coder-30b.toml")];

impl Registry {
    /// A registry of the profiles bundled with this crate.
    pub fn builtin() -> Self {
        let mut profiles = Vec::new();
        for raw in BUILTIN_PROFILES {
            // Bundled profiles are validated by a unit test, so an error here is a
            // build-time bug; skip rather than panic in production.
            if let Ok(p) = ModelProfile::from_toml(raw) {
                profiles.push(p);
            }
        }
        Registry { profiles }
    }

    /// Add a profile (e.g. one loaded from disk to override a builtin). Newly
    /// added profiles take precedence in [`Registry::select`].
    pub fn push(&mut self, profile: ModelProfile) {
        self.profiles.insert(0, profile);
    }

    /// The first profile matching `model_name`, if any.
    pub fn select(&self, model_name: &str) -> Option<&ModelProfile> {
        self.profiles.iter().find(|p| p.matches(model_name))
    }
}

/// `*`-glob match (no other metacharacters). ASCII byte comparison — model names
/// are ASCII identifiers.
fn glob_match(pattern: &str, text: &str) -> bool {
    let (p, t) = (pattern.as_bytes(), text.as_bytes());
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut mark) = (None, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(sp) = star {
            pi = sp + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

#[derive(Debug)]
pub enum ProfileError {
    Parse(String),
    Invalid(&'static str),
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "profile parse error: {e}"),
            Self::Invalid(m) => write!(f, "invalid profile: {m}"),
        }
    }
}

impl std::error::Error for ProfileError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_profiles_parse_and_validate() {
        // Guards every bundled profile — a malformed one fails the build's tests.
        for raw in BUILTIN_PROFILES {
            ModelProfile::from_toml(raw).expect("bundled profile must be valid");
        }
        assert!(!Registry::builtin().profiles.is_empty());
    }

    #[test]
    fn qwen_profile_fields() {
        let reg = Registry::builtin();
        let p = reg.select("Qwen3-Coder-30B").expect("qwen profile selected");
        assert_eq!(p.tool_call.open, "<tool_call>");
        assert_eq!(p.tool_call.close, "</tool_call>");
        assert_eq!(p.tool_call.name_field, "tool_name");
        assert_eq!(p.tool_result.role, "tool");
    }

    #[test]
    fn qwen_wraps_tool_result() {
        let reg = Registry::builtin();
        let p = reg.select("Qwen3-Coder-30B").unwrap();
        assert_eq!(p.wrap_tool_result("26C"), "<tool_response>\n26C\n</tool_response>");
    }

    #[test]
    fn select_matches_glob_and_misses_unknown() {
        let reg = Registry::builtin();
        assert!(reg.select("Qwen3-Coder-30B-Instruct").is_some());
        assert!(reg.select("some-other-model").is_none());
    }

    #[test]
    fn glob_basics() {
        assert!(glob_match("Qwen3*", "Qwen3-Coder-30B"));
        assert!(glob_match("*Coder*", "Qwen3-Coder-30B"));
        assert!(glob_match("exact", "exact"));
        assert!(!glob_match("Qwen3*", "Llama-3"));
        assert!(!glob_match("abc", "abcd"));
        assert!(glob_match("*", "anything"));
    }

    #[test]
    fn user_profile_overrides_builtin() {
        let mut reg = Registry::builtin();
        let custom = ModelProfile::from_toml(
            r#"
            schema_version = 1
            model = "Qwen3-Coder-30B"
            match = ["Qwen3-Coder*"]
            [tool_call]
            open = "<call>"
            close = "</call>"
            [tool_result]
            role = "tool"
        "#,
        )
        .unwrap();
        reg.push(custom);
        assert_eq!(reg.select("Qwen3-Coder-30B").unwrap().tool_call.open, "<call>");
    }

    #[test]
    fn validate_rejects_empty_delimiters() {
        let r = ModelProfile::from_toml(
            r#"
            schema_version = 1
            model = "x"
            [tool_call]
            open = ""
            close = "</tool>"
            [tool_result]
            role = "tool"
        "#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn parser_from_profile_splits_tool_call() {
        let reg = Registry::builtin();
        let p = reg.select("Qwen3-Coder-30B").unwrap();
        let mut parser = ToolStreamParser::from_profile(p);
        let mut evs = parser.push_token("Sure. ");
        evs.extend(parser.push_token(r#"<tool_call>{"tool_name":"get_weather","city":"SF"}</tool_call>"#));
        evs.extend(parser.finish());
        assert_eq!(evs[0], Event::Content("Sure. ".into()));
        assert_eq!(
            evs[1],
            Event::ToolCall {
                name: "get_weather".into(),
                args: r#"{"tool_name":"get_weather","city":"SF"}"#.into(),
            },
        );
    }

    #[test]
    fn spans_parse_when_present() {
        let p = ModelProfile::from_toml(
            r#"
            schema_version = 1
            model = "thinker"
            [tool_call]
            open = "<tool>"
            close = "</tool>"
            [tool_result]
            role = "tool"
            [[span]]
            tag = "reasoning"
            open = "<think>"
            close = "</think>"
        "#,
        )
        .unwrap();
        assert_eq!(p.spans.len(), 1);
        assert_eq!(p.spans[0].tag, "reasoning");
    }
}
