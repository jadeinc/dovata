//! dovata — JADE OS inference daemon (host binary).
//!
//! The host: it owns the Unix socket at `~/.jade/llm.sock`, frames
//! `InferenceRequest`s off the wire, answers `health`/`stats`, and routes each request
//! to a provider. The **local** llama.cpp engine (`dovata::engine`) is compiled in and
//! served in-process (Linux); **remote** HTTP backends (`anthropic`, `openai`) are
//! loaded from `.so` packages (see [`dovata::providers`]).
//!
//! ```bash
//! # Local engine (Linux, on by default) — configure the model via JADE_MODEL_PATH:
//! JADE_MODEL_PATH=/models/qwen3.gguf cargo run --release --features cuda
//! # Add remote providers:
//! JADE_PROVIDERS=/path/libovata_provider_anthropic.so cargo run --release
//! ```

mod host;

fn jade_sock_path() -> String {
    // socket lives under the user's home so a normal (non-root) daemon can bind it;
    // fall back to /root's home if HOME somehow isn't set (e.g. bare init).
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_owned());
    format!("{home}/.jade/llm.sock")
}

fn main() {
    host::run(jade_sock_path());
}
