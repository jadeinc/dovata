//! Protocol-level latency / throughput benchmark for the jaded daemon.
//!
//! Talks directly to the Unix socket using `ovata-infer-protocol` (no Jade runtime in
//! the path), so the numbers reflect the daemon itself. Measures:
//!   * TTFT  — request send → first `Token` frame (prefill + first decode step)
//!   * decode tok/s — generated tokens after the first, over the time to `Done`
//! `Done.tokens_used` (= n_prompt + n_gen) is used to recover the prompt length.
//!
//! Run on the box where jaded is listening:
//!   cargo run --release --example bench -- [sock_path] [iters]
//! Defaults: $HOME/.jade/llm.sock, 5 iters/scenario.
//!
//! Prompts carry a per-iteration nonce so the daemon's response cache never hits.

use ovata_infer_protocol::{Frame, InferenceRequest};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::time::Instant;

struct Sample {
    ttft_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    gen_frames: u64,
    total_tokens: u64,
}

fn count_only(sock: &str, prompt: &str) -> std::io::Result<u64> {
    let mut s = UnixStream::connect(sock)?;
    let req = InferenceRequest {
        prompt: prompt.to_owned(),
        model: "bench".to_owned(),
        max_tokens: 1,
        grammar: None,
        anchor: None,
        stop_anchor: None,
        count_only: true,
        stats_only: false,
        health_only: false,
        keep_anchors: false,
        trust: 0,
        rlm: false,
    };
    s.write_all(&req.encode().expect("encode"))?;
    read_done_tokens(&mut s)
}

fn read_done_tokens(s: &mut UnixStream) -> std::io::Result<u64> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        loop {
            match Frame::decode(&buf) {
                Ok((Frame::Done { tokens_used }, _)) => return Ok(tokens_used),
                Ok((Frame::Error(e), _)) => return Err(err(e)),
                Ok((_, consumed)) => {
                    buf.drain(..consumed);
                }
                Err(ovata_infer_protocol::FrameError::Incomplete) => break,
                Err(e) => return Err(err(format!("{e}"))),
            }
        }
        let n = s.read(&mut tmp)?;
        if n == 0 {
            return Err(err("eof before Done"));
        }
        buf.extend_from_slice(&tmp[..n]);
    }
}

fn run_once(sock: &str, prompt: &str, max_tokens: u32) -> std::io::Result<Sample> {
    let mut s = UnixStream::connect(sock)?;
    let req = InferenceRequest {
        prompt: prompt.to_owned(),
        model: "bench".to_owned(),
        max_tokens,
        grammar: None,
        anchor: None,
        stop_anchor: None,
        count_only: false,
        stats_only: false,
        health_only: false,
        keep_anchors: false,
        trust: 0,
        rlm: false,
    };
    let bytes = req.encode().expect("encode");

    let t0 = Instant::now();
    s.write_all(&bytes)?;

    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    let mut t_first: Option<Instant> = None;
    let mut gen_frames = 0u64;

    loop {
        loop {
            match Frame::decode(&buf) {
                Ok((frame, consumed)) => {
                    buf.drain(..consumed);
                    match frame {
                        Frame::Meta { .. } | Frame::Json(_) => {}
                        Frame::Token(_) => {
                            if t_first.is_none() {
                                t_first = Some(Instant::now());
                            }
                            gen_frames += 1;
                        }
                        Frame::Done { tokens_used } => {
                            let t_done = Instant::now();
                            let tf = t_first.unwrap_or(t_done);
                            return Ok(Sample {
                                ttft_ms: ms(t0, tf),
                                decode_ms: ms(tf, t_done),
                                total_ms: ms(t0, t_done),
                                gen_frames,
                                total_tokens: tokens_used,
                            });
                        }
                        Frame::Error(e) => return Err(err(e)),
                    }
                }
                Err(ovata_infer_protocol::FrameError::Incomplete) => break,
                Err(e) => return Err(err(format!("{e}"))),
            }
        }
        let n = s.read(&mut tmp)?;
        if n == 0 {
            return Err(err("eof before Done"));
        }
        buf.extend_from_slice(&tmp[..n]);
    }
}

fn ms(a: Instant, b: Instant) -> f64 {
    b.saturating_duration_since(a).as_secs_f64() * 1000.0
}

fn err(m: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, m.into())
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if xs.is_empty() {
        return 0.0;
    }
    xs[xs.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sock = args.get(1).cloned().unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
        format!("{home}/.jade/llm.sock")
    });
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    println!("jaded bench — socket {sock}, {iters} iters/scenario\n");

    // (label, prompt, max_tokens)
    let scenarios: &[(&str, &str, u32)] = &[
        ("ttft (1 tok)", "Write a haiku about the sea.", 1),
        ("decode (128 tok)", "Write a detailed paragraph about the ocean.", 128),
        ("decode (512 tok)", "Write a long, detailed essay about the ocean.", 512),
    ];

    // Warm up one connection/request first.
    let _ = run_once(&sock, "Say hi.", 8);

    println!(
        "{:<18} {:>10} {:>14} {:>10} {:>10}",
        "scenario", "TTFT ms", "decode tok/s", "gen tok", "total ms"
    );
    println!("{}", "-".repeat(66));

    for (label, base_prompt, max_tokens) in scenarios {
        let mut ttfts = Vec::new();
        let mut tps = Vec::new();
        let mut gen = 0u64;
        let mut total_ms = Vec::new();
        let mut prompt_tokens = 0u64;
        let mut failed = None;

        for i in 0..iters {
            let prompt = format!("{base_prompt} (run {i})"); // nonce → no cache hit
            if i == 0 {
                prompt_tokens = count_only(&sock, &prompt).unwrap_or(0);
            }
            match run_once(&sock, &prompt, *max_tokens) {
                Ok(s) => {
                    ttfts.push(s.ttft_ms);
                    total_ms.push(s.total_ms);
                    gen = s.gen_frames;
                    // decode tok/s = (generated - 1) / decode_time
                    if s.decode_ms > 0.0 && s.gen_frames > 1 {
                        tps.push((s.gen_frames - 1) as f64 / (s.decode_ms / 1000.0));
                    }
                    let _ = s.total_tokens;
                }
                Err(e) => {
                    failed = Some(e.to_string());
                    break;
                }
            }
        }

        if let Some(e) = failed {
            println!("{label:<18} FAILED: {e}");
            continue;
        }
        println!(
            "{:<18} {:>10.1} {:>14.1} {:>10} {:>10.1}",
            label,
            median(ttfts),
            median(tps),
            gen,
            median(total_ms),
        );
        let _ = prompt_tokens;
    }

    println!("\nnote: TTFT = prefill + first token; decode tok/s excludes the first token.");
    println!("prompt-token counts available via count_only (printed per-scenario in -v builds).");
}
