//! OpenAI provider for the ovata inference protocol.
//!
//! Built as a `cdylib` (`.so`) the daemon loads at runtime. Implements
//! [`Provider`] and exports the C-ABI via [`export_provider!`]. Provenance: the
//! OpenAI HTTP backend formerly staged in the daemon (`inference/openai.rs`),
//! rewritten from an async `InferenceBackend` returning a whole `InferenceResponse`
//! into a synchronous `Provider` that streams `Frame`s.
//!
//! Non-streaming: one blocking chat-completions request, emitted as `Token` frames
//! (chunked under the wire limit by [`FrameSink::emit_text`]) then `Done`. SSE
//! streaming is a future refinement.

use ovata_infer_protocol::{export_provider, Frame, FrameSink, InferenceRequest, Provider};
use serde::Deserialize;

/// Provider config, passed as JSON through `from_config`. The daemon owns the API
/// key and default model — never the request path.
#[derive(Deserialize)]
struct Config {
    api_key: String,
    /// Model used when a request leaves `model` empty.
    #[serde(default = "default_model")]
    default_model: String,
    /// HTTP timeout; defaults to 60s.
    #[serde(default)]
    timeout_secs: Option<u64>,
}

fn default_model() -> String {
    "gpt-4o".to_string()
}

pub struct OpenAi {
    api_key: String,
    default_model: String,
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
        Ok(OpenAi { api_key: cfg.api_key, default_model: cfg.default_model, client })
    }

    fn infer(&self, req: &InferenceRequest, sink: &mut dyn FrameSink) {
        match self.complete(req) {
            Ok((text, tokens_used)) => {
                sink.emit_text(&text);
                sink.emit(Frame::Done { tokens_used });
            }
            Err(msg) => sink.emit(Frame::Error(msg)),
        }
    }
}

impl OpenAi {
    /// One blocking chat-completions call → (completion text, total tokens).
    fn complete(&self, req: &InferenceRequest) -> Result<(String, u64), String> {
        let model = if req.model.is_empty() { &self.default_model } else { &req.model };
        let mut body = serde_json::json!({
            "model": model,
            "messages": [{ "role": "user", "content": req.prompt }],
        });
        // Forward a token cap only when the client set one (0 = unspecified).
        if req.max_tokens > 0 {
            body["max_tokens"] = serde_json::json!(req.max_tokens);
        }

        let resp = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
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
}
