//! OpenAI provider for the ovata inference protocol.
//!
//! Built as a `cdylib` (`.so`) the daemon dlopens at runtime; implements
//! [`Provider`] and exports the C-ABI via [`export_provider!`]. Each request is one
//! blocking chat-completions call, emitted as `Token` frames (chunked under the wire
//! limit by [`FrameSink::emit_text`]) then `Done`. Token-by-token SSE streaming is a
//! future refinement.
//!
//! Supported request features: `prompt`, `model`, `max_tokens`. Constrained decoding
//! (`grammar`/`anchor`/`stop_anchor`) and `rlm` mode have no chat-completions
//! equivalent and are rejected with an `Error` frame rather than silently ignored.
//! `count_only` is also rejected: OpenAI exposes no server-side tokenizer endpoint
//! (tokenization is client-side via tiktoken), so we cannot honor it truthfully.

use ovata_infer_protocol::{export_provider, Frame, FrameSink, InferenceRequest, Provider};
use serde::Deserialize;

const DEFAULT_BASE_URL: &str = "https://api.openai.com";
const DEFAULT_MODEL: &str = "gpt-4o";
/// `max_tokens` used when a request leaves it `0` (unspecified).
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Provider config, passed as JSON through `from_config`. The daemon owns the API
/// key and defaults — never the request path.
#[derive(Deserialize)]
struct Config {
    api_key: String,
    /// Model used when a request leaves `model` empty.
    #[serde(default = "default_model")]
    default_model: String,
    /// API base URL, overridable for gateways/proxies. Defaults to the public API.
    #[serde(default = "default_base_url")]
    base_url: String,
    /// `max_tokens` used when a request leaves it `0` (unspecified).
    #[serde(default = "default_max_tokens")]
    max_tokens_default: u32,
    /// HTTP timeout; defaults to 60s.
    #[serde(default)]
    timeout_secs: Option<u64>,
}

fn default_model() -> String {
    DEFAULT_MODEL.to_string()
}
fn default_base_url() -> String {
    DEFAULT_BASE_URL.to_string()
}
fn default_max_tokens() -> u32 {
    DEFAULT_MAX_TOKENS
}

pub struct OpenAi {
    api_key: String,
    default_model: String,
    base_url: String,
    max_tokens_default: u32,
    client: reqwest::blocking::Client,
}

impl Provider for OpenAi {
    const NAME: &'static str = "openai";

    fn from_config(config: &[u8]) -> Result<Self, String> {
        let cfg: Config =
            serde_json::from_slice(config).map_err(|e| format!("openai: bad config: {e}"))?;
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(cfg.timeout_secs.unwrap_or(60)))
            .build()
            .map_err(|e| format!("openai: http client: {e}"))?;
        Ok(OpenAi {
            api_key: cfg.api_key,
            default_model: cfg.default_model,
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            max_tokens_default: cfg.max_tokens_default,
            client,
        })
    }

    fn infer(&self, req: &InferenceRequest, sink: &mut dyn FrameSink) {
        if let Err(msg) = reject_unsupported(req) {
            sink.emit(Frame::Error(msg));
            return;
        }
        match self.complete(req) {
            Ok((text, tokens_used)) => {
                sink.emit_text(&text);
                sink.emit(Frame::Done { tokens_used });
            }
            Err(msg) => sink.emit(Frame::Error(msg)),
        }
    }
}

/// Reject request features a remote HTTP provider cannot honor. Returning an error
/// (rather than ignoring the field) keeps a constrained-decoding, RLM, or token-count
/// client from silently receiving a plain generated completion instead.
fn reject_unsupported(req: &InferenceRequest) -> Result<(), String> {
    if req.rlm {
        return Err("openai: rlm mode is not supported by a remote provider".into());
    }
    if req.grammar.is_some() {
        return Err(
            "openai: grammar/constrained decoding is not supported by a remote provider".into(),
        );
    }
    if req.count_only {
        return Err(
            "openai: count_only is not supported — OpenAI has no server-side tokenizer endpoint"
                .into(),
        );
    }
    Ok(())
}

impl OpenAi {
    /// The model for this request: its own `model`, else the configured default.
    fn model<'a>(&'a self, req: &'a InferenceRequest) -> &'a str {
        if req.model.is_empty() { &self.default_model } else { &req.model }
    }

    /// The `max_tokens` to send: the request's, or the configured default when `0`.
    fn max_tokens(&self, req: &InferenceRequest) -> u32 {
        if req.max_tokens == 0 { self.max_tokens_default } else { req.max_tokens }
    }

    /// One blocking chat-completions call → (completion text, total tokens).
    fn complete(&self, req: &InferenceRequest) -> Result<(String, u64), String> {
        let body = serde_json::json!({
            "model": self.model(req),
            "max_tokens": self.max_tokens(req),
            "messages": [{ "role": "user", "content": req.prompt }],
        });

        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| format!("openai: request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(format!("openai: API returned HTTP {status}: {text}"));
        }

        let json: serde_json::Value =
            resp.json().map_err(|e| format!("openai: bad response body: {e}"))?;

        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .ok_or("openai: unexpected response format")?
            .to_string();

        let tokens_used = json["usage"]["total_tokens"].as_u64().unwrap_or(0);

        Ok((text, tokens_used))
    }
}

export_provider!(OpenAi);

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    /// A one-shot mock HTTP server. Accepts a single connection, captures the raw
    /// request head (request line + headers) and body, replies with `status` +
    /// `resp_body`, and sends `(head, body)` back over the channel. Runs on a
    /// background thread so the blocking client can talk to it in-process.
    fn mock_once(status: u16, resp_body: &'static str) -> (String, mpsc::Receiver<(String, String)>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = Vec::new();
            let mut tmp = [0u8; 4096];
            let (head, body_start) = loop {
                let n = stream.read(&mut tmp).unwrap();
                if n == 0 {
                    return;
                }
                buf.extend_from_slice(&tmp[..n]);
                if let Some(pos) = find(&buf, b"\r\n\r\n") {
                    break (String::from_utf8_lossy(&buf[..pos]).to_string(), pos + 4);
                }
            };
            let want = content_length(&head);
            while buf.len() < body_start + want {
                let n = stream.read(&mut tmp).unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            let body = String::from_utf8_lossy(&buf[body_start..body_start + want]).to_string();
            tx.send((head, body)).unwrap();
            let resp = format!(
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                resp_body.len(),
                resp_body
            );
            stream.write_all(resp.as_bytes()).unwrap();
        });
        (base, rx)
    }

    fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
        hay.windows(needle.len()).position(|w| w == needle)
    }

    fn content_length(head: &str) -> usize {
        head.lines()
            .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap_or(0)))
            .unwrap_or(0)
    }

    /// A minimal in-memory sink capturing frames for assertions.
    struct VecSink(Vec<Frame>);
    impl FrameSink for VecSink {
        fn emit(&mut self, frame: Frame) {
            self.0.push(frame);
        }
    }

    fn provider(base_url: &str, extra: &str) -> OpenAi {
        let cfg = format!(r#"{{"api_key":"sk-test","base_url":"{base_url}"{extra}}}"#);
        OpenAi::from_config(cfg.as_bytes()).unwrap()
    }

    #[test]
    fn exports_match_the_abi_and_name() {
        assert_eq!(ovata_provider_abi_version(), ovata_infer_protocol::PROVIDER_ABI_VERSION);
        // SAFETY: name symbols return a static &str's ptr/len.
        let name = unsafe {
            std::slice::from_raw_parts(ovata_provider_name_ptr(), ovata_provider_name_len())
        };
        assert_eq!(name, b"openai");
        assert_eq!(<OpenAi as Provider>::NAME, "openai");
    }

    #[test]
    fn from_config_parses_without_touching_the_network() {
        let cfg = br#"{"api_key":"sk-test","default_model":"gpt-x"}"#;
        assert!(OpenAi::from_config(cfg).is_ok());
    }

    #[test]
    fn from_config_requires_api_key() {
        assert!(OpenAi::from_config(b"{}").is_err());
    }

    #[test]
    fn infer_sends_expected_request_and_parses_completion() {
        let (base, rx) = mock_once(
            200,
            r#"{"choices":[{"message":{"role":"assistant","content":"hi there"}}],"usage":{"total_tokens":9}}"#,
        );
        let p = provider(&base, "");
        let req = InferenceRequest { prompt: "hello".into(), model: "gpt-x".into(), max_tokens: 32, ..Default::default() };
        let mut sink = VecSink(Vec::new());
        p.infer(&req, &mut sink);

        let (head, body) = rx.recv().unwrap();
        assert!(head.starts_with("POST /v1/chat/completions "), "head: {head}");
        assert!(head.to_ascii_lowercase().contains("authorization: bearer sk-test"));
        assert!(body.contains("\"model\":\"gpt-x\""), "body: {body}");
        assert!(body.contains("\"max_tokens\":32"), "body: {body}");
        assert!(body.contains("\"content\":\"hello\""), "body: {body}");

        assert!(matches!(&sink.0[0], Frame::Token(t) if t == "hi there"));
        assert!(matches!(sink.0[1], Frame::Done { tokens_used: 9 }));
    }

    #[test]
    fn unspecified_max_tokens_uses_the_configured_default() {
        let (base, rx) = mock_once(
            200,
            r#"{"choices":[{"message":{"content":"x"}}],"usage":{"total_tokens":2}}"#,
        );
        let p = provider(&base, r#","max_tokens_default":1234"#);
        let req = InferenceRequest { prompt: "hello".into(), max_tokens: 0, ..Default::default() };
        let mut sink = VecSink(Vec::new());
        p.infer(&req, &mut sink);
        let (_head, body) = rx.recv().unwrap();
        assert!(body.contains("\"max_tokens\":1234"), "body: {body}");
    }

    #[test]
    fn grammar_rlm_and_count_only_are_rejected_without_a_network_call() {
        let p = provider("http://127.0.0.1:1", ""); // unroutable: proves no request is made
        for req in [
            InferenceRequest { prompt: "x".into(), grammar: Some("root ::= \"a\"".into()), ..Default::default() },
            InferenceRequest { prompt: "x".into(), rlm: true, ..Default::default() },
            InferenceRequest { prompt: "x".into(), count_only: true, ..Default::default() },
        ] {
            let mut sink = VecSink(Vec::new());
            p.infer(&req, &mut sink);
            assert_eq!(sink.0.len(), 1);
            assert!(matches!(sink.0[0], Frame::Error(_)));
        }
    }
}
