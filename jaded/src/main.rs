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

#[cfg(all(feature = "inference", not(target_os = "linux")))]
compile_error!("the `inference` backend (llama.cpp + io_uring) only builds on Linux");

#[cfg(feature = "inference")]
mod inference;

#[cfg(feature = "inference")]
fn jade_sock_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_owned());
    format!("{home}/.jade/llm.sock")
}

fn main() {
    #[cfg(feature = "inference")]
    inference::run(jade_sock_path());

    #[cfg(not(feature = "inference"))]
    {
        eprintln!("JADE OS — jaded v{}", env!("CARGO_PKG_VERSION"));
        eprintln!("jaded: built without --features inference — no backend available");
        eprintln!("jaded: rebuild with --features inference (Linux) for the llama.cpp backend");
        std::process::exit(1);
    }
}
