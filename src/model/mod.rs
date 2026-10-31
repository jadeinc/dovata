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
//! protocol stays model-agnostic (it carries opaque text + frames) — and as of
//! protocol v3 it no longer reports a model name at all: the `Meta` frame names
//! the *provider*, not the model. So the profile is selected where the model
//! identity is actually known — daemon-side, from the `.gguf`'s own metadata —
//! and used to construct requests and interpret the streamed special tokens.
//! Swap the model → swap the profile file → no Jade program changes.
//!
//! ## Two identifiers, in that order
//!
//! Selection happens at boot in [`crate::load::preflight`], which reads the GGUF header
//! before the model loads and calls [`Registry::select_for`] with both identifiers
//! the file carries. `general.name` is tried first: it is free text, but a name glob
//! names one model, so it is the specific answer. `general.architecture` is tried
//! second: it is the controlled vocabulary llama.cpp itself dispatches on, stable
//! across publishers and requants, so it catches a model nobody has written a profile
//! for and lands it on its family's conventions. Matching nothing is a refused boot,
//! not a silent fallback — see the preflight docs.
//!
//! ## Format
//!
//! Profiles are TOML. See `profiles/qwen3-coder-30b.toml` for a worked example.
//! Load one with [`ModelProfile::from_toml`], or use a [`Registry`] to pick the
//! right profile for the model name the daemon reads from the loaded `.gguf`
//! (the wire no longer carries a model name — `Meta` reports the provider).

use serde::Deserialize;

/// A model's token/turn-taking vocabulary.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelProfile {
    /// Format version, for forward-compatible evolution.
    pub schema_version: u32,
    /// Human-readable model name this profile targets.
    pub model: String,
    /// Glob patterns (against the daemon's loaded model name) this profile
    /// applies to. `*` matches any sequence. First matching profile wins.
    #[serde(default, rename = "match")]
    pub match_patterns: Vec<String>,
    /// GGUF `general.architecture` strings this profile serves, e.g. `qwen3moe`.
    ///
    /// This is the *family* fallback, tried only after every profile's name globs
    /// have missed. Architecture is a controlled vocabulary llama.cpp dispatches on
    /// and is stable across publishers and requants, where `general.name` is free
    /// text — so a model nobody has written a profile for still lands on its
    /// family's conventions instead of on bare ChatML. Exact matching, no globs.
    #[serde(default)]
    pub arch: Vec<String>,
    /// How the model emits a tool invocation.
    pub tool_call: ToolCall,
    /// How a tool result is rendered back into the conversation.
    pub tool_result: ToolResult,
    /// Additional special-token spans the model may stream (e.g. reasoning),
    /// surfaced as structured events instead of raw content.
    #[serde(default, rename = "span")]
    pub spans: Vec<Span>,
    /// Context-length limits for this model: the native attention window and the
    /// measured RLM (KV-composition) efficient frontier. Absent → both unset.
    #[serde(default)]
    pub limits: Limits,
    /// How a chat (system + turns) is rendered into the model's flat prompt
    /// string. Absent → ChatML (the [`ChatTemplate`] default), so existing
    /// profiles and unknown models keep working.
    #[serde(default)]
    pub chat_template: ChatTemplate,
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

/// Context-length limits for a model. Both fields are token counts; `0` means
/// "unset" (the daemon falls back to its env/config default).
///
/// * `context_window` — the native attention window the daemon serves below which
///   a prompt is answered with full-context inference (highest fidelity).
/// * `rlm_max_context` — the **efficient frontier** of the blended KV-composition
///   (RLM+RAG) engine: the largest corpus, in tokens, over which it answers
///   reliably (measured with the RULER-style frontier harness). A request whose
///   corpus exceeds this is rejected rather than answered with degraded accuracy.
///   This is documented here so Jade programmers know each model's usable limit.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Limits {
    #[serde(default)]
    pub context_window: u32,
    #[serde(default)]
    pub rlm_max_context: u32,
}

impl Limits {
    /// The RLM efficient frontier in tokens, or `None` when unset (`0`).
    pub fn rlm_max_context(&self) -> Option<u32> {
        (self.rlm_max_context > 0).then_some(self.rlm_max_context)
    }

    /// Whether a corpus of `corpus_tokens` is within this model's RLM frontier.
    /// `true` when the limit is unset (enforcement deferred to the daemon default).
    pub fn rlm_corpus_allowed(&self, corpus_tokens: u32) -> bool {
        self.rlm_max_context().map_or(true, |max| corpus_tokens <= max)
    }
}

/// How a chat (a system prompt + role/content turns) is rendered into the flat
/// prompt string the model is tokenized from. This used to be llama.cpp's
/// `apply_chat_template`; the vendored fork moved templating out of the engine, so
/// it lives here as model data instead. The strings are the model's literal
/// special-token markers — they are tokenized downstream with `parse_special`, so
/// e.g. `<|im_start|>` becomes a single special token, not raw characters.
///
/// The default is **ChatML** (Qwen/OpenAI style), byte-identical to llama.cpp's
/// `LLM_CHAT_TEMPLATE_CHATML`, so a profile with no `[chat_template]` block — or no
/// profile at all — still formats correctly.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ChatTemplate {
    /// Emitted before each turn's role, e.g. `<|im_start|>`.
    pub message_prefix: String,
    /// Between role and content, e.g. `"\n"`.
    pub role_content_sep: String,
    /// Emitted after each turn's content, e.g. `<|im_end|>\n`.
    pub message_suffix: String,
    /// Appended once at the end to elicit the assistant's reply, e.g.
    /// `<|im_start|>assistant\n`.
    pub generation_prompt: String,
    /// Optional leading marker prepended once (a BOS-like prefix). Empty for ChatML.
    pub bos: String,
}

impl Default for ChatTemplate {
    fn default() -> Self {
        // ChatML — matches llama.cpp's LLM_CHAT_TEMPLATE_CHATML exactly:
        //   per turn:  <|im_start|>{role}\n{content}<|im_end|>\n
        //   then:      <|im_start|>assistant\n   (the generation prompt)
        ChatTemplate {
            message_prefix: "<|im_start|>".to_owned(),
            role_content_sep: "\n".to_owned(),
            message_suffix: "<|im_end|>\n".to_owned(),
            generation_prompt: "<|im_start|>assistant\n".to_owned(),
            bos: String::new(),
        }
    }
}

impl ChatTemplate {
    /// Render `turns` (each `(role, content)`) into the model's prompt string. When
    /// `add_generation_prompt` is set, the trailing assistant cue is appended so the
    /// model continues as the assistant. Pure string assembly — no tokenizer needed.
    pub fn render<'a, I>(&self, turns: I, add_generation_prompt: bool) -> String
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let mut out = self.bos.clone();
        for (role, content) in turns {
            out.push_str(&self.message_prefix);
            out.push_str(role);
            out.push_str(&self.role_content_sep);
            out.push_str(content);
            out.push_str(&self.message_suffix);
        }
        if add_generation_prompt {
            out.push_str(&self.generation_prompt);
        }
        out
    }
}

fn default_name_field() -> String {
    "tool_name".to_owned()
}

impl ModelProfile {
    /// Parse and validate a profile from TOML.
    pub fn from_toml(s: &str) -> Result<Self, ProfileError> {
        // two steps: deserialize the TOML into the struct, then run our own
        // semantic checks (serde only guarantees the shape, not that delimiters
        // are non-empty).
        let p: ModelProfile = toml::from_str(s).map_err(|e| ProfileError::Parse(e.to_string()))?;
        p.validate()?;
        Ok(p)
    }

    /// Reject structurally-valid-but-meaningless profiles (empty delimiters).
    pub fn validate(&self) -> Result<(), ProfileError> {
        // a tool call needs both an opening AND closing delimiter to be parseable
        if self.tool_call.open.is_empty() || self.tool_call.close.is_empty() {
            return Err(ProfileError::Invalid("tool_call.open/close must be non-empty"));
        }
        // a result needs a role to render back into the conversation
        if self.tool_result.role.is_empty() {
            return Err(ProfileError::Invalid("tool_result.role must be non-empty"));
        }
        // each optional span likewise needs a label and both delimiters
        for s in &self.spans {
            if s.open.is_empty() || s.close.is_empty() || s.tag.is_empty() {
                return Err(ProfileError::Invalid("span tag/open/close must be non-empty"));
            }
        }
        // when both context limits are set, the RLM frontier must be at least the
        // native window — RLM only ever serves prompts that DON'T fit natively.
        if self.limits.context_window > 0
            && self.limits.rlm_max_context > 0
            && self.limits.rlm_max_context < self.limits.context_window
        {
            return Err(ProfileError::Invalid("limits.rlm_max_context must be >= limits.context_window"));
        }
        Ok(())
    }

    /// Whether this profile applies to `model_name` (any `match` glob matches,
    /// or the exact `model` name matches).
    pub fn matches(&self, model_name: &str) -> bool {
        // exact name match is the cheap path; otherwise try each glob pattern
        self.model == model_name
            || self.match_patterns.iter().any(|pat| glob_match(pat, model_name))
    }

    /// Render a raw tool result into the form the model expects in its next turn.
    pub fn wrap_tool_result(&self, content: &str) -> String {
        // sandwich the raw result between the model's configured prefix/suffix
        // (e.g. <tool_response>\n … \n</tool_response> for Qwen3)
        format!("{}{}{}", self.tool_result.wrap_prefix, content, self.tool_result.wrap_suffix)
    }
}

/// A set of profiles, selectable by model name.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    profiles: Vec<ModelProfile>,
}

/// Profiles compiled into the crate (the defaults shipped in `profiles/`).
const BUILTIN_PROFILES: &[&str] = &[
    include_str!("profiles/qwen3-coder-30b.toml"),
    include_str!("profiles/qwen3.6-35b-a3b.toml"),
];

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
        // first match wins; since push() inserts at the front, user-added profiles
        // are checked before builtins.
        self.profiles.iter().find(|p| p.matches(model_name))
    }

    /// Select on the two identifiers a GGUF actually carries: `general.name` and
    /// `general.architecture`.
    ///
    /// Name first, architecture second, and the order is the whole point. A name glob
    /// is specific — it names one model — while an architecture names a whole family,
    /// so trying arch first would let a family profile shadow the exact profile written
    /// for the model in hand. Falling back to the family is still much better than the
    /// alternative: bare ChatML with no tool-call delimiters, which fails silently at
    /// the first tool call rather than loudly at boot.
    ///
    /// Returns the profile and which identifier picked it, so the daemon can say so.
    pub fn select_for(&self, name: Option<&str>, arch: Option<&str>) -> Option<(&ModelProfile, Selector)> {
        if let Some(n) = name {
            if let Some(p) = self.profiles.iter().find(|p| p.matches(n)) {
                return Some((p, Selector::Name));
            }
        }
        if let Some(a) = arch {
            if let Some(p) = self.profiles.iter().find(|p| p.arch.iter().any(|x| x == a)) {
                return Some((p, Selector::Architecture));
            }
        }
        None
    }
}

/// Which GGUF identifier picked a profile in [`Registry::select_for`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selector {
    /// `general.name` matched the profile's `model` or one of its `match` globs.
    Name,
    /// `general.architecture` matched one of the profile's `arch` entries — the
    /// family fallback.
    Architecture,
}

impl Selector {
    /// A phrase for a startup log line.
    pub fn describe(self) -> &'static str {
        match self {
            Selector::Name => "matched general.name",
            Selector::Architecture => "matched general.architecture (family fallback)",
        }
    }
}

/// `*`-glob match (no other metacharacters). ASCII byte comparison — model names
/// are ASCII identifiers.
fn glob_match(pattern: &str, text: &str) -> bool {
    // classic linear-time wildcard match with backtracking. We walk the text once;
    // when a `*` is in play we remember where it was so we can "give it one more
    // character" if the rest of the pattern fails to line up.
    let (p, t) = (pattern.as_bytes(), text.as_bytes());
    let (mut pi, mut ti) = (0usize, 0usize); // cursors into pattern and text
    // `star` = pattern index of the most recent '*'; `mark` = the text index we'd
    // resume from after letting that '*' absorb one more char.
    let (mut star, mut mark) = (None, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == t[ti]) {
            // literal characters match — advance both cursors
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            // hit a '*': record it and tentatively let it match zero chars (don't
            // advance the text yet), moving past the '*' in the pattern.
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(sp) = star {
            // mismatch, but there's a '*' to fall back on: rewind the pattern to
            // just after that '*' and let it swallow one more text char.
            pi = sp + 1;
            mark += 1;
            ti = mark;
        } else {
            // mismatch with no '*' to back off to → not a match
            return false;
        }
    }
    // text exhausted: any trailing '*'s in the pattern match the empty string
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    // it's a match only if we also consumed the entire pattern
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

    // A model profile is two files — this TOML, compiled into the engine, and a
    // sibling `.jde` package published to jadelang.org that parses tool calls out
    // of a reply. They ship on separate pipelines and nothing at runtime links
    // them, so they drifted: this side said the tool-name key was `tool_name`,
    // the Jade side said `name`. `tools/scripts/gen-profile-package.py` now
    // generates the Jade side's constants from here; this test is what makes
    // forgetting to run it fail rather than ship.
    #[test]
    fn jade_packages_match_their_toml() {
        // CARGO_MANIFEST_DIR is the repo root: this crate's package root is the root,
        // and its capabilities are the children of src/.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/model/profiles");
        let mut checked = 0;
        for entry in std::fs::read_dir(&dir).expect("profiles dir readable") {
            let toml_path = entry.expect("dir entry").path();
            if toml_path.extension().is_none_or(|e| e != "toml") {
                continue;
            }
            // Not every model has a published language-side package, and the
            // engine works fine without one.
            let jde_path = toml_path.with_extension("jde");
            if !jde_path.exists() {
                continue;
            }
            checked += 1;

            let raw = std::fs::read_to_string(&toml_path).expect("profile readable");
            let p = ModelProfile::from_toml(&raw).expect("profile valid");
            let jde = std::fs::read_to_string(&jde_path).expect("jade package readable");
            let name = jde_path.file_name().unwrap().to_string_lossy();

            for (konst, value) in [
                ("TOOL_OPEN ", &p.tool_call.open),
                ("TOOL_CLOSE", &p.tool_call.close),
                ("NAME_FIELD", &p.tool_call.name_field),
            ] {
                let expected = format!("let {konst} = {}", jade_string(value));
                assert!(
                    jde.contains(&expected),
                    "{name} is out of date with {}: expected `{expected}`.\n\
                     Run tools/scripts/gen-profile-package.py and commit the result.",
                    toml_path.file_name().unwrap().to_string_lossy(),
                );
            }
        }
        assert!(checked > 0, "no profile had a sibling .jde — did the profiles move?");
    }

    /// A Jade double-quoted string literal, matching `jade_string` in
    /// `tools/scripts/gen-profile-package.py`.
    fn jade_string(s: &str) -> String {
        let escaped = s
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\t', "\\t")
            .replace('\r', "\\r");
        format!("\"{escaped}\"")
    }

    #[test]
    fn qwen_wraps_tool_result() {
        let reg = Registry::builtin();
        let p = reg.select("Qwen3-Coder-30B").unwrap();
        assert_eq!(p.wrap_tool_result("26C"), "<tool_response>\n26C\n</tool_response>");
    }

    // ── Chat template rendering ───────────────────────────────────────────────

    // The default MUST be byte-identical to llama.cpp's LLM_CHAT_TEMPLATE_CHATML,
    // because the daemon used to format via apply_chat_template(chatml) and we must
    // not change the prompt the model sees.
    #[test]
    fn default_chat_template_is_chatml_byte_identical() {
        let t = ChatTemplate::default();
        let out = t.render(
            [("system", "You are Omar"), ("user", "hi"), ("assistant", "hello"), ("user", "bye")],
            true,
        );
        assert_eq!(
            out,
            "<|im_start|>system\nYou are Omar<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n\
             <|im_start|>assistant\nhello<|im_end|>\n\
             <|im_start|>user\nbye<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn render_without_generation_prompt_omits_assistant_cue() {
        let t = ChatTemplate::default();
        let out = t.render([("user", "hi")], false);
        assert_eq!(out, "<|im_start|>user\nhi<|im_end|>\n");
    }

    #[test]
    fn qwen_profile_chat_template_renders_chatml() {
        let reg = Registry::builtin();
        let p = reg.select("Qwen3-Coder-30B").unwrap();
        let out = p.chat_template.render([("user", "ping")], true);
        assert_eq!(out, "<|im_start|>user\nping<|im_end|>\n<|im_start|>assistant\n");
    }

    // A profile with no [chat_template] block falls back to the ChatML default.
    #[test]
    fn absent_chat_template_block_defaults_to_chatml() {
        let toml = r#"
            schema_version = 1
            model = "X"
            [tool_call]
            open = "<t>"
            close = "</t>"
            [tool_result]
            role = "tool"
        "#;
        let p = ModelProfile::from_toml(toml).unwrap();
        assert_eq!(p.chat_template.message_prefix, "<|im_start|>");
        assert_eq!(p.chat_template.render([("user", "q")], true),
            "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n");
    }

    // A partial [chat_template] block fills missing fields from the ChatML default.
    #[test]
    fn partial_chat_template_block_fills_defaults() {
        let toml = r#"
            schema_version = 1
            model = "Y"
            [tool_call]
            open = "<t>"
            close = "</t>"
            [tool_result]
            role = "tool"
            [chat_template]
            generation_prompt = "ASSISTANT: "
        "#;
        let p = ModelProfile::from_toml(toml).unwrap();
        // overridden field
        assert_eq!(p.chat_template.generation_prompt, "ASSISTANT: ");
        // defaulted fields
        assert_eq!(p.chat_template.message_prefix, "<|im_start|>");
        assert_eq!(p.chat_template.render([("user", "q")], true),
            "<|im_start|>user\nq<|im_end|>\nASSISTANT: ");
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
    fn limits_parse_and_gate_corpus() {
        let p = ModelProfile::from_toml(
            r#"
            schema_version = 1
            model = "m"
            [tool_call]
            open = "<tool>"
            close = "</tool>"
            [tool_result]
            role = "tool"
            [limits]
            context_window = 32768
            rlm_max_context = 262144
        "#,
        )
        .unwrap();
        assert_eq!(p.limits.context_window, 32768);
        assert_eq!(p.limits.rlm_max_context(), Some(262144));
        assert!(p.limits.rlm_corpus_allowed(262144));
        assert!(!p.limits.rlm_corpus_allowed(262145));
    }

    #[test]
    fn limits_absent_defaults_to_unset_and_allows_everything() {
        let reg = Registry::builtin();
        let p = reg.select("Qwen3-Coder-30B").unwrap();
        // The bundled profile may or may not set limits; an unset limit allows any corpus.
        if p.limits.rlm_max_context().is_none() {
            assert!(p.limits.rlm_corpus_allowed(u32::MAX));
        }
    }

    #[test]
    fn validate_rejects_rlm_frontier_below_window() {
        let r = ModelProfile::from_toml(
            r#"
            schema_version = 1
            model = "m"
            [tool_call]
            open = "<tool>"
            close = "</tool>"
            [tool_result]
            role = "tool"
            [limits]
            context_window = 32768
            rlm_max_context = 16384
        "#,
        );
        assert!(r.is_err());
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
