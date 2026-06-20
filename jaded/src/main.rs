//! jaded — JADE OS inference daemon (binary).
//!
//! Listens on `~/.jade/llm.sock` and serves `InferenceRequest`s from JADE
//! programs, streaming `Frame` tokens back over the same socket. The pure core
//! (config, prompt parsing, response cache, decode control) lives in the
//! `jaded` library and is tested on any platform; this binary adds the
//! llama.cpp + io_uring backend behind the `inference` feature.
//!
//! ```bash
//! cargo build --release --features inference   # production (Linux)
//! cargo build --release --features cuda        # production + NVIDIA GPU
//! ```
//!
//! Without `--features inference` the binary builds (so `cargo test` runs the
//! library suite anywhere) but has no backend and exits with an error.

// Guard: the backend pulls in llama.cpp + io_uring, which only build on Linux.
// We narrow the error to "inference feature AND not linux" so the default (no-
// backend) build and the library tests still compile everywhere, including macOS.
#[cfg(all(feature = "inference", not(target_os = "linux")))]
compile_error!("the `inference` backend (llama.cpp + io_uring) only builds on Linux");

// The backend module only exists when the feature is on (Linux production build).
#[cfg(feature = "inference")]
mod inference;

#[cfg(feature = "inference")]
fn jade_sock_path() -> String {
    // socket lives under the user's home so a normal (non-root) daemon can bind
    // it; fall back to /root's home if HOME somehow isn't set (e.g. bare init).
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_owned());
    format!("{home}/.jade/llm.sock")
}

fn main() {
    // Production path: hand control to the backend's run loop (binds the socket,
    // loads the model, serves requests forever). Only compiled with the feature.
    #[cfg(feature = "inference")]
    inference::run(jade_sock_path());

    // Stub path: no backend was compiled in. Print why and exit non-zero so this
    // build is never mistaken for a working daemon — its only real purpose is to
    // let `cargo test` exercise the pure library on any platform.
    #[cfg(not(feature = "inference"))]
    {
        eprintln!("JADE OS — jaded v{}", env!("CARGO_PKG_VERSION"));
        eprintln!("jaded: built without --features inference — no backend available");
        eprintln!("jaded: rebuild with --features inference (Linux) for the llama.cpp backend");
        std::process::exit(1);
    }
}
