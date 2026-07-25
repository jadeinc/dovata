//! Minimal one-shot smoke test for the in-process local engine.
//!
//! Connects to a running `dovata` over its Unix socket, sends ONE ordinary
//! plain-prompt request, and prints the `Meta` provider name plus the streamed
//! tokens. Proves the `ProviderHandle::Local` path (engine compiled into the
//! daemon, no FFI hop) loads the model and decodes end to end.
//!
//! Run on the box AFTER the daemon reports "listening on":
//!   JADE_SOCK=$HOME/.jade/llm.sock cargo run --release --example smoke_client -p dovata

use ovata_infer_protocol::{Frame, InferenceRequest};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;

fn main() -> std::io::Result<()> {
    let sock = std::env::var("JADE_SOCK")
        .unwrap_or_else(|_| format!("{}/.jade/llm.sock", std::env::var("HOME").unwrap()));
    let prompt = std::env::args().nth(1).unwrap_or_else(|| {
        "In one short sentence, what is a Unix domain socket?".to_string()
    });

    let mut s = UnixStream::connect(&sock)?;
    let req = InferenceRequest { prompt, max_tokens: 64, ..Default::default() };
    s.write_all(&req.encode().expect("encode"))?;

    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 8192];
    let mut provider = String::new();
    let mut text = String::new();
    loop {
        loop {
            match Frame::decode(&buf) {
                Ok((Frame::Meta { provider: p }, c)) => { provider = p; buf.drain(..c); }
                Ok((Frame::Token(t), c)) => { text.push_str(&t); buf.drain(..c); }
                Ok((Frame::Json(_), c)) => { buf.drain(..c); }
                Ok((Frame::Done { tokens_used }, _)) => {
                    println!("provider (Meta) : {provider}");
                    println!("tokens_used     : {tokens_used}");
                    println!("completion      : {}", text.trim());
                    assert_eq!(provider, "dovata", "expected Meta provider 'dovata'");
                    assert!(!text.trim().is_empty(), "expected non-empty completion");
                    assert!(tokens_used > 0, "expected tokens_used > 0");
                    println!("SMOKE OK");
                    return Ok(());
                }
                Ok((Frame::Error(e), _)) => {
                    eprintln!("daemon error: {e}");
                    std::process::exit(1);
                }
                Err(ovata_infer_protocol::FrameError::Incomplete) => break,
                Err(e) => { eprintln!("frame decode: {e}"); std::process::exit(1); }
            }
        }
        let n = s.read(&mut tmp)?;
        if n == 0 { eprintln!("eof before Done"); std::process::exit(1); }
        buf.extend_from_slice(&tmp[..n]);
    }
}
