//! RLM socket-client smoke test (Phase 3).
//!
//! Talks to a running `jaded` over its Unix socket using the real wire protocol:
//!   1. UPLOAD a large needle-in-haystack context  → receive a handle,
//!   2. QUERY by handle                            → receive the answer,
//!   3. QUERY inline (sections in the request)      → receive the answer,
//! and checks the needle appears in both answers. Proves a program can drive the
//! in-process RLM purely over the socket (no model code in the client).
//!
//! Run on the box AFTER starting the daemon:
//!   JADE_SOCK=$HOME/.jade/llm.sock cargo run --example rlm_client -p jaded

use jade_protocol::{Frame, InferenceRequest};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;

/// Read frames until Done; return (concatenated Token text, last Json payload).
fn read_response(s: &mut UnixStream) -> std::io::Result<(String, Option<String>)> {
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 8192];
    let mut text = String::new();
    let mut json: Option<String> = None;
    loop {
        loop {
            match Frame::decode(&buf) {
                Ok((Frame::Done { .. }, c)) => {
                    buf.drain(..c);
                    return Ok((text, json));
                }
                Ok((Frame::Error(e), _)) => {
                    return Err(std::io::Error::other(format!("daemon error: {e}")));
                }
                Ok((Frame::Token(t), c)) => {
                    text.push_str(&t);
                    buf.drain(..c);
                }
                Ok((Frame::Json(j), c)) => {
                    json = Some(j);
                    buf.drain(..c);
                }
                Ok((_, c)) => {
                    buf.drain(..c); // Meta, etc.
                }
                Err(jade_protocol::FrameError::Incomplete) => break,
                Err(e) => return Err(std::io::Error::other(format!("frame decode: {e}"))),
            }
        }
        let n = s.read(&mut tmp)?;
        if n == 0 {
            return Err(std::io::Error::other("eof before Done"));
        }
        buf.extend_from_slice(&tmp[..n]);
    }
}

/// Send one rlm request (the JSON payload rides in `prompt`).
fn rlm(sock: &str, payload: String) -> std::io::Result<(String, Option<String>)> {
    let mut s = UnixStream::connect(sock)?;
    let req = InferenceRequest { prompt: payload, rlm: true, max_tokens: 256, ..Default::default() };
    s.write_all(&req.encode().expect("encode"))?;
    read_response(&mut s)
}

fn main() {
    let sock = std::env::var("JADE_SOCK")
        .unwrap_or_else(|_| format!("{}/.jade/llm.sock", std::env::var("HOME").unwrap()));
    let needle = "XQ-42";

    // Build a haystack JSON array of ["summary","body"] sections (no quotes in the
    // content, so manual JSON is safe). Total >> context window.
    let n = 50usize;
    let needle_at = n / 2;
    let filler = "routine operational detail ".repeat(60);
    let mut secs: Vec<String> = Vec::new();
    for i in 0..n {
        if i == needle_at {
            secs.push(format!(
                "[\"Section {i}: Access Credentials\",\"Section {i}: Access Credentials. \
                 The secret access code is {needle}. {filler}\"]"
            ));
        } else {
            secs.push(format!(
                "[\"Section {i}: Topic {i}\",\"Section {i}: general routine details. {filler}\"]"
            ));
        }
    }
    let sections_json = format!("[{}]", secs.join(","));
    let question = "What is the secret access code?";

    // 1. UPLOAD → handle
    let (_, json) = rlm(&sock, format!("{{\"upload\":{sections_json}}}")).expect("upload");
    let json = json.expect("upload should return a handle JSON");
    println!("upload -> {json}");
    let handle: u64 = json.chars().filter(|c| c.is_ascii_digit()).collect::<String>().parse().expect("handle");

    // 2. QUERY by handle
    let (ans1, _) = rlm(&sock, format!(
        "{{\"handle\":{handle},\"question\":\"{question}\",\"max_spans\":4}}"
    )).expect("query by handle");
    println!("query(handle={handle}) ANSWER: {:?}", ans1.trim());
    println!("  needle found: {}", ans1.contains(needle));

    // 3. QUERY inline (sections in the request itself)
    let (ans2, _) = rlm(&sock, format!(
        "{{\"sections\":{sections_json},\"question\":\"{question}\",\"max_spans\":4}}"
    )).expect("inline query");
    println!("query(inline) ANSWER: {:?}", ans2.trim());
    println!("  needle found: {}", ans2.contains(needle));

    let ok = ans1.contains(needle) && ans2.contains(needle);
    println!("RLM SOCKET TEST: {}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
