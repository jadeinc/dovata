//! Anthropic provider for the ovata inference protocol.
//!
//! Built as a `cdylib` (`.so`) the daemon loads at runtime. Implements
//! [`Provider`] and exports the C-ABI via [`export_provider!`]. Provenance: the
//! Anthropic HTTP backend formerly staged in the daemon (`inference/anthropic.rs`),
//! rewritten from an async `InferenceBackend` returning a whole `InferenceResponse`
//! into a synchronous `Provider` that streams `Frame`s.
//!
//! This is a non-streaming implementation: it makes one blocking request and emits
//! the completion as `Token` frames (chunked under the wire limit by
//! [`FrameSink::emit_text`]) followed by `Done`. Real token-by-token SSE streaming
//! is a future refinement.

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
    "claude-sonnet-5".to_string()
}

pub struct Anthropic {
    api_key: String,
    default_model: String,
    client: reqwest::blocking::Client,
}

impl Provider for Anthropic {
    const NAME: &'static str = "anthropic";

    fn from_config(config: &[u8]) -> Result<Self, String> {
        let cfg: Config =
            serde_json::from_slice(config).map_err(|e| format!("anthropic: bad config: {e}"))?;
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(cfg.timeout_secs.unwrap_or(60)))
            .build()
            .map_err(|e| format!("anthropic: http client: {e}"))?;
        Ok(Anthropic { api_key: cfg.api_key, default_model: cfg.default_model, client })
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

impl Anthropic {
    /// One blocking Messages API call → (completion text, total tokens).
    fn complete(&self, req: &InferenceRequest) -> Result<(String, u64), String> {
        let model = if req.model.is_empty() { &self.default_model } else { &req.model };
        let body = serde_json::json!({
            "model": model,
            // Anthropic requires max_tokens; the protocol default of 0 is invalid.
            "max_tokens": req.max_tokens.max(1),
            "messages": [{ "role": "user", "content": req.prompt }],
        });

        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| format!("anthropic: request failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(format!("anthropic: API returned HTTP {status}: {text}"));
        }

        let json: serde_json::Value =
            resp.json().map_err(|e| format!("anthropic: bad response body: {e}"))?;

        let text = json["content"][0]["text"]
            .as_str()
            .ok_or("anthropic: unexpected response format")?
            .to_string();

        let tokens_used = json["usage"]["input_tokens"].as_u64().unwrap_or(0)
            + json["usage"]["output_tokens"].as_u64().unwrap_or(0);

        Ok((text, tokens_used))
    }
}

export_provider!(Anthropic);

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
        assert_eq!(name, b"anthropic");
        assert_eq!(<Anthropic as Provider>::NAME, "anthropic");
    }

    #[test]
    fn from_config_parses_without_touching_the_network() {
        // Building the client makes no request; infer() is what would.
        let cfg = br#"{"api_key":"sk-test","default_model":"claude-x"}"#;
        assert!(Anthropic::from_config(cfg).is_ok());
    }

    #[test]
    fn from_config_requires_api_key() {
        assert!(Anthropic::from_config(b"{}").is_err());
    }
}
