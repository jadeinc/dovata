//! Anthropic HTTP inference backend — MOVED from the Jade language repo.
//!
//! Provenance: `jadelang/src/llm/anthropic.rs`. It was removed from the language
//! as part of making the language a pure wire-protocol client; all inference
//! backends now live in the daemon behind one socket.
//!
//! NOT YET WIRED. This file is staged for integration, not compiled — it is not
//! declared in any `mod`. To bring it into the daemon:
//!   * replace the language types (`InferenceBackend` / `InferenceResponse` /
//!     `JadeError` / `Span`) with the daemon's request/response/error types;
//!     the request is `ovata_infer_protocol::InferenceRequest`.
//!   * add an async HTTP client dependency (`reqwest`) to `jaded/Cargo.toml`.
//!   * register it as a route target in `route.rs` selected by config/provider.
//!   * move the API key out of the request path and into the daemon's own
//!     config (`jaded/src/config.rs`), which now owns provider + key + model.

use std::sync::Arc;

use tokio::sync::Semaphore;

use super::{InferenceBackend, InferenceRequest, InferenceResponse};
use crate::frontend::error::{JadeError, Result, Span};

pub struct AnthropicBackend {
    api_key: String,
    default_model: String,
    client: reqwest::Client,
    semaphore: Option<Arc<Semaphore>>,
}

impl AnthropicBackend {
    pub fn new(api_key: &str, default_model: &str, max_parallel: Option<usize>) -> Result<Self> {
        Ok(AnthropicBackend {
            api_key: api_key.to_string(),
            default_model: default_model.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .map_err(|e| JadeError::InferenceError {
                    message: format!("failed to build HTTP client: {e}"),
                    span: Span { line: 0, col: 0 },
                })?,
            semaphore: max_parallel.map(|n| Arc::new(Semaphore::new(n))),
        })
    }
}

#[async_trait::async_trait]
impl InferenceBackend for AnthropicBackend {
    async fn infer(&self, req: InferenceRequest, span: Span) -> Result<InferenceResponse> {
        let _permit = if let Some(sem) = &self.semaphore {
            Some(sem.acquire().await.map_err(|e| JadeError::InferenceError {
                message: format!("semaphore error: {e}"),
                span,
            })?)
        } else {
            None
        };

        let model = if req.model.is_empty() { &self.default_model } else { &req.model };

        let body = serde_json::json!({
            "model": model,
            "max_tokens": req.max_tokens,
            "messages": [{ "role": "user", "content": req.prompt }],
        });

        let response = self.client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e: reqwest::Error| JadeError::InferenceError { message: e.to_string(), span })?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            return Err(JadeError::InferenceError {
                message: format!("Anthropic API returned HTTP {}: {}", status, body_text),
                span,
            });
        }

        let json: serde_json::Value = response.json()
            .await
            .map_err(|e: reqwest::Error| JadeError::InferenceError { message: e.to_string(), span })?;

        let text = json["content"][0]["text"]
            .as_str()
            .ok_or_else(|| JadeError::InferenceError {
                message: "unexpected response format from Anthropic API".to_string(),
                span,
            })?
            .trim()
            .to_string();

        let tokens_used = (json["usage"]["input_tokens"].as_u64().unwrap_or(0)
            + json["usage"]["output_tokens"].as_u64().unwrap_or(0)) as i64;

        Ok(InferenceResponse { text, tokens_used })
    }
}
