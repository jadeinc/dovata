//! dovata — JADE OS inference daemon (host binary).
//!
//! A thin host: it owns the Unix socket at `~/.jade/llm.sock`, frames
//! `InferenceRequest`s off the wire, answers `health`/`stats`, and routes each
//! request to a provider loaded from a `.so` package (see [`dovata::providers`] and
//! the `providers/` crates). All inference — the local llama.cpp backend
//! (`provider-dovata`) or a remote HTTP backend (`provider-anthropic`,
//! `provider-openai`) — lives in providers, so this binary links no model backend
//! and builds on any platform.
//!
//! ```bash
//! JADE_PROVIDERS=/path/libovata_provider_dovata.so cargo run --release
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
