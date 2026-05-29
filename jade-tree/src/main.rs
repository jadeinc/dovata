//! jade-tree — JADE OS primary interaction layer
//!
//! Listens on `~/.jade/llm.sock` and serves `InferenceRequest`s from
//! jadec, streaming `Frame` tokens back over the same socket.
//!
//! ## Features
//!
//! | Feature     | Use case                                        |
//! |-------------|-------------------------------------------------|
//! | `inference` | Production: llama.cpp LLM backend               |
//! | `cuda`      | NVIDIA GPU offload (implies `inference`)        |
//! | `dummy`     | Dev/QEMU: canned responses, no llama.cpp        |
//!
//! ```bash
//! cargo build --release --features inference          # production
//! cargo build --release --features cuda               # production + GPU
//! cargo build --release --features dummy \            # dev/QEMU
//!   --target x86_64-unknown-linux-musl
//! ```

#[cfg(all(not(target_os = "linux"), not(feature = "dummy")))]
compile_error!("jade-tree only runs on Linux — use --features dummy for dev builds");

mod prompt;

#[cfg(feature = "dummy")]
mod dummy;

#[cfg(all(not(feature = "dummy"), feature = "inference"))]
mod inference;

fn jade_sock_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_owned());
    format!("{home}/.jade/llm.sock")
}

fn main() {
    #[cfg(feature = "dummy")]
    dummy::run(jade_sock_path());

    #[cfg(all(not(feature = "dummy"), feature = "inference"))]
    inference::run(jade_sock_path());

    #[cfg(all(not(feature = "dummy"), not(feature = "inference")))]
    {
        eprintln!("JADE OS — jade-tree v{}", env!("CARGO_PKG_VERSION"));
        eprintln!("jade-tree: built without --features inference — no backend available");
        eprintln!("jade-tree: rebuild with --features inference or --features dummy");
        std::process::exit(1);
    }
}
