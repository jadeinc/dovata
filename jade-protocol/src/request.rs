//! Request side of the wire protocol: client → daemon `InferenceRequest`.
//!
//! On-wire layout: `[4-byte little-endian length][JSON body]`. One request per
//! frame; the client sends them sequentially over a single Unix-socket connection.
//! Most fields are `#[serde(default)]` so older/simpler clients can omit them.

use serde::{Deserialize, Serialize};

/// Trust level for an inference request prompt. Carried over the wire so the
/// daemon can log provenance; runtime-side enforcement (refusing tainted strings
/// at `sh.exec`/`http.get`/`fs.read`) lives in the C runtime, not in jade-tree.
pub mod trust {
    pub const TRUSTED: u8 = 0;
    pub const TAINTED: u8 = 1;
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    pub model: String,
    pub max_tokens: u32,
    /// GBNF grammar for constrained decoding. `None` (or absent from JSON) = unconstrained.
    #[serde(default)]
    pub grammar: Option<String>,
    /// Start of the grammar-enforcement span. Grammar is applied only after the model emits
    /// this string. `None` = enforce grammar from the very first token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<String>,
    /// End of the grammar-enforcement span. Grammar enforcement ends when this string appears;
    /// the stop string itself is not emitted to the client, and inference continues freely.
    /// `None` = enforce grammar through the very last token (no early retirement).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_anchor: Option<String>,
    /// If true, tokenize the prompt and return the token count without running generation.
    #[serde(default)]
    pub count_only: bool,
    /// If true, return the daemon-wide cumulative token counter without touching the model.
    #[serde(default)]
    pub stats_only: bool,
    /// If true, return a [`crate::Health`] report (one `Json` frame then `Done`) without
    /// touching the model. A cheap liveness/readiness probe.
    #[serde(default)]
    pub health_only: bool,
    /// When true, the daemon makes the `stop_anchor` boundary observable in-band:
    /// it does not strip the `stop_anchor` from client output, and it guarantees the
    /// boundary appears exactly once at span close (synthesizing it if the model
    /// retired the grammar via JSON depth-close without emitting it). This lets a
    /// client deterministically delimit an anchored span (e.g. `<tool>…</tool>`) by
    /// pure string parsing. Absent/false = legacy behavior (the `stop_anchor` is
    /// stripped). Defaults to false for backward compatibility with old clients.
    #[serde(default)]
    pub keep_anchors: bool,
    /// Provenance of the prompt string. `0` (TRUSTED) means program-author-controlled;
    /// `1` (TAINTED) means it derived from an LLM, network, or other untrusted source.
    /// Absent from JSON defaults to TRUSTED for backward compatibility with old clients.
    #[serde(default)]
    pub trust: u8,
    /// RLM mode. When true, `prompt` carries a JSON payload instead of text:
    ///   - upload:  `{"upload":[["summary","body"],...]}` → daemon stores the context
    ///     and replies with a `Json` frame `{"handle":N}` then `Done`.
    ///   - query:   `{"handle":N,"question":"...","max_spans":K}` or inline
    ///     `{"sections":[["summary","body"],...],"question":"...","max_spans":K}`
    ///     → answered by in-process recursive decomposition (PICK→fan-out→COMBINE),
    ///     streamed as `Token` frames then `Done`.
    /// Default false → ordinary single-shot inference (path unchanged).
    #[serde(default)]
    pub rlm: bool,
}

impl InferenceRequest {
    /// Encode as length-prefixed JSON for writing to the Unix socket.
    pub fn encode(&self) -> Result<Vec<u8>, serde_json::Error> {
        // serialize the struct to JSON bytes, then prepend a 4-byte LE length so
        // the reader knows exactly how many bytes the body is.
        let json = serde_json::to_vec(self)?;
        let len = json.len() as u32;
        let mut buf = Vec::with_capacity(4 + json.len());
        buf.extend_from_slice(&len.to_le_bytes()); // 4-byte header
        buf.extend_from_slice(&json);              // JSON body
        Ok(buf)
    }

    /// Decode from a length-prefixed JSON buffer.
    /// Returns `Err(RequestDecodeError::Incomplete)` if the buffer is too short.
    pub fn decode(buf: &[u8]) -> Result<Self, RequestDecodeError> {
        // need the 4-byte length header before we can know the body size
        if buf.len() < 4 {
            return Err(RequestDecodeError::Incomplete);
        }
        // read the declared body length...
        let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        // ...and make sure the whole body has actually arrived yet
        if buf.len() < 4 + len {
            return Err(RequestDecodeError::Incomplete);
        }
        // parse exactly the body slice. Note we only read `len` bytes, so any extra
        // bytes after this frame (e.g. the start of the next request) are ignored.
        serde_json::from_slice(&buf[4..4 + len]).map_err(RequestDecodeError::Json)
    }
}

#[derive(Debug)]
pub enum RequestDecodeError {
    Incomplete,
    Json(serde_json::Error),
}

impl std::fmt::Display for RequestDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Incomplete => write!(f, "incomplete request buffer"),
            Self::Json(e) => write!(f, "JSON decode: {e}"),
        }
    }
}

impl std::error::Error for RequestDecodeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(prompt: &str) -> InferenceRequest {
        InferenceRequest {
            prompt: prompt.into(),
            model: "m".into(),
            max_tokens: 256,
            grammar: None,
            anchor: None,
            stop_anchor: None,
            count_only: false,
            stats_only: false,
            health_only: false,
            keep_anchors: false,
            trust: trust::TRUSTED,
            rlm: false,
        }
    }

    #[test]
    fn request_roundtrip() {
        let r = InferenceRequest {
            prompt: "hello world".into(),
            model: "llama4-scout".into(),
            max_tokens: 256,
            grammar: None,
            anchor: None,
            stop_anchor: None,
            count_only: false,
            stats_only: false,
            health_only: false,
            keep_anchors: false,
            trust: trust::TRUSTED,
            rlm: false,
        };
        let decoded = InferenceRequest::decode(&r.encode().unwrap()).unwrap();
        assert_eq!(decoded.prompt, r.prompt);
        assert_eq!(decoded.model, r.model);
        assert_eq!(decoded.max_tokens, r.max_tokens);
    }

    #[test]
    fn request_roundtrip_with_grammar() {
        let grammar = r#"root ::= "true" | "false""#.to_owned();
        let r = InferenceRequest {
            prompt: "true or false".into(),
            model: "llama4-scout".into(),
            max_tokens: 8,
            grammar: Some(grammar.clone()),
            anchor: None,
            stop_anchor: None,
            count_only: false,
            stats_only: false,
            health_only: false,
            keep_anchors: false,
            trust: trust::TRUSTED,
            rlm: false,
        };
        let decoded = InferenceRequest::decode(&r.encode().unwrap()).unwrap();
        assert_eq!(decoded.grammar, Some(grammar));
    }

    #[test]
    fn request_roundtrip_with_anchor() {
        let grammar = r#"root ::= "true" | "false""#.to_owned();
        let r = InferenceRequest {
            prompt: "Answer: true or false?".into(),
            model: "llama4-scout".into(),
            max_tokens: 8,
            grammar: Some(grammar.clone()),
            anchor: Some("Answer: ".into()),
            stop_anchor: None,
            count_only: false,
            stats_only: false,
            health_only: false,
            keep_anchors: false,
            trust: trust::TRUSTED,
            rlm: false,
        };
        let decoded = InferenceRequest::decode(&r.encode().unwrap()).unwrap();
        assert_eq!(decoded.grammar, Some(grammar));
        assert_eq!(decoded.anchor, Some("Answer: ".into()));
    }

    #[test]
    fn request_roundtrip_with_stop_anchor() {
        let grammar = r#"root ::= "{" [^}]* "}""#.to_owned();
        let r = InferenceRequest {
            prompt: "call a tool".into(),
            model: "m".into(),
            max_tokens: 64,
            grammar: Some(grammar.clone()),
            anchor: Some("<tool>".into()),
            stop_anchor: Some("</tool>".into()),
            count_only: false,
            stats_only: false,
            health_only: false,
            keep_anchors: false,
            trust: trust::TRUSTED,
            rlm: false,
        };
        let decoded = InferenceRequest::decode(&r.encode().unwrap()).unwrap();
        assert_eq!(decoded.anchor, Some("<tool>".into()));
        assert_eq!(decoded.stop_anchor, Some("</tool>".into()));
    }

    #[test]
    fn request_roundtrip_with_keep_anchors() {
        let mut r = req("call a tool");
        r.anchor = Some("<tool>".into());
        r.stop_anchor = Some("</tool>".into());
        r.keep_anchors = true;
        let decoded = InferenceRequest::decode(&r.encode().unwrap()).unwrap();
        assert!(decoded.keep_anchors);
    }

    #[test]
    fn keep_anchors_absent_from_json_defaults_to_false() {
        let json = br#"{"prompt":"hi","model":"m","max_tokens":10}"#;
        let len = json.len() as u32;
        let mut buf = len.to_le_bytes().to_vec();
        buf.extend_from_slice(json);
        let decoded = InferenceRequest::decode(&buf).unwrap();
        assert!(!decoded.keep_anchors);
    }

    #[test]
    fn request_roundtrip_with_health_only() {
        let mut r = req("");
        r.health_only = true;
        let decoded = InferenceRequest::decode(&r.encode().unwrap()).unwrap();
        assert!(decoded.health_only);
    }

    #[test]
    fn health_only_absent_from_json_defaults_to_false() {
        let json = br#"{"prompt":"hi","model":"m","max_tokens":10}"#;
        let len = json.len() as u32;
        let mut buf = len.to_le_bytes().to_vec();
        buf.extend_from_slice(json);
        let decoded = InferenceRequest::decode(&buf).unwrap();
        assert!(!decoded.health_only);
    }

    #[test]
    fn anchor_absent_from_json_defaults_to_none() {
        let json = br#"{"prompt":"hi","model":"m","max_tokens":10,"grammar":"root ::= \"ok\""}"#;
        let len = json.len() as u32;
        let mut buf = len.to_le_bytes().to_vec();
        buf.extend_from_slice(json);
        let decoded = InferenceRequest::decode(&buf).unwrap();
        assert_eq!(decoded.anchor, None);
        assert_eq!(decoded.stop_anchor, None);
    }

    #[test]
    fn grammar_absent_from_json_defaults_to_none() {
        let json = br#"{"prompt":"hi","model":"m","max_tokens":10}"#;
        let len = json.len() as u32;
        let mut buf = len.to_le_bytes().to_vec();
        buf.extend_from_slice(json);
        let decoded = InferenceRequest::decode(&buf).unwrap();
        assert_eq!(decoded.grammar, None);
    }

    #[test]
    fn incomplete_buffer_returns_error() {
        assert!(matches!(
            InferenceRequest::decode(&[0x00, 0x00, 0x03]),
            Err(RequestDecodeError::Incomplete)
        ));
    }

    #[test]
    fn empty_buffer_returns_incomplete() {
        assert!(matches!(
            InferenceRequest::decode(&[]),
            Err(RequestDecodeError::Incomplete)
        ));
    }

    #[test]
    fn partial_payload_returns_incomplete() {
        let full = req("test").encode().unwrap();
        assert!(matches!(
            InferenceRequest::decode(&full[..5]),
            Err(RequestDecodeError::Incomplete)
        ));
    }

    #[test]
    fn empty_prompt_roundtrip() {
        let decoded = InferenceRequest::decode(&req("no history").encode().unwrap()).unwrap();
        assert_eq!(decoded.prompt, "no history");
    }

    #[test]
    fn request_with_special_characters() {
        let prompt = "Explain <memory> & \"unsafe\" in Rust — what's the difference?";
        let decoded = InferenceRequest::decode(&req(prompt).encode().unwrap()).unwrap();
        assert_eq!(decoded.prompt, prompt);
    }

    #[test]
    fn request_roundtrip_with_trust() {
        let mut r = req("untrusted content");
        r.trust = trust::TAINTED;
        let decoded = InferenceRequest::decode(&r.encode().unwrap()).unwrap();
        assert_eq!(decoded.trust, trust::TAINTED);
    }

    #[test]
    fn trust_absent_from_json_defaults_to_trusted() {
        let json = br#"{"prompt":"hi","model":"m","max_tokens":10}"#;
        let len = json.len() as u32;
        let mut buf = len.to_le_bytes().to_vec();
        buf.extend_from_slice(json);
        let decoded = InferenceRequest::decode(&buf).unwrap();
        assert_eq!(decoded.trust, trust::TRUSTED);
    }

    #[test]
    fn extra_bytes_after_message_ignored() {
        let mut buf = req("test").encode().unwrap();
        buf.extend_from_slice(b"\xDE\xAD\xBE\xEF");
        let decoded = InferenceRequest::decode(&buf).unwrap();
        assert_eq!(decoded.prompt, "test");
    }
}
