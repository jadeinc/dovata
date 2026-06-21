//! llama.cpp + io_uring inference backend (Linux, `--features inference`).
//!
//! This file is the backend ROOT: it owns the model/context lifecycle, the
//! Unix-socket server loop, the async io_uring token writer, and request
//! dispatch. The actual decode engines live in focused submodules:
//!
//! * [`kv`]          — KV-cache optimizations: cache-type mapping, multi-turn
//!                     prefix reuse, and shared-prefix prefill+copy.
//! * [`standard`]    — single-shot inference (one sequence, constrained decode).
//! * [`concurrency`] — the continuous-batching engine (N sequences, one decode
//!                     per tick) over the pure [`jaded::sched::Scheduler`].
//! * [`rlm`]         — Recursive Language Model: context store, model-driven
//!                     PICK→fan-out→COMBINE recursion, and its socket dispatch.
//!
//! The platform-independent logic (anchors/stop/grammar, config, response cache,
//! scheduler) lives in the `jaded` library and is unit-tested on any platform;
//! everything here is llama.cpp-bound and Linux-only.

mod kv;
mod standard;
mod concurrency;
mod rlm;
mod compose;

use io_uring::{opcode, types, IoUring};
use jade_protocol::{Frame, InferenceRequest};
use jaded::cache::PromptCache;
use jaded::config::{Config, KvType};
use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel},
    token::LlamaToken,
};
use std::io::{Read, Write};
use std::num::NonZeroU32;
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Daemon-wide cumulative generated-token counter (served by `stats_only`).
static GLOBAL_TOKENS: AtomicU64 = AtomicU64::new(0);

/// Process start, set once at the top of [`run`]. Backs `Health::uptime_secs`.
static START: OnceLock<Instant> = OnceLock::new();

const QUEUE_DEPTH: u32 = 128;

// ── Async token writer (io_uring) ─────────────────────────────────────────────
//
// Submits each Frame::Token write to the io_uring submission queue and returns
// immediately, letting the GPU start on the next token while the kernel drains
// the write in parallel. flush() must be called before sending the Done frame
// to ensure all token bytes are visible to the client.

struct IoUringWriter {
    ring: IoUring,
    fd: RawFd,
    // Buffers whose data pointers are live in the ring. Never shrunk until
    // flush() confirms all completions — the kernel holds raw pointers into
    // these heap allocations between submit and completion.
    buffers: Vec<Vec<u8>>,
    in_flight: usize,
}

impl IoUringWriter {
    fn new(fd: RawFd) -> std::io::Result<Self> {
        Ok(Self {
            ring: IoUring::new(QUEUE_DEPTH)?,
            fd,
            buffers: Vec::new(),
            in_flight: 0,
        })
    }

    fn write(&mut self, data: Vec<u8>) -> std::io::Result<()> {
        // Drain one completion before pushing when the ring is full.
        // (back-pressure: never queue more than QUEUE_DEPTH outstanding writes.)
        if self.in_flight >= QUEUE_DEPTH as usize {
            self.reap(1)?;
        }
        // take ownership of the bytes and keep them alive in self.buffers — the
        // kernel will read from this exact heap allocation asynchronously, so it
        // must NOT move or free until the completion is reaped.
        self.buffers.push(data);
        let buf = self.buffers.last().unwrap();
        let op = opcode::Write::new(types::Fd(self.fd), buf.as_ptr(), buf.len() as u32)
            .build()
            .user_data((self.buffers.len() - 1) as u64);
        // SAFETY: buf lives in self.buffers and is not removed until flush()
        // confirms the completion, so the pointer outlives the io_uring op.
        unsafe { self.ring.submission().push(&op).expect("SQ full"); }
        self.ring.submit()?;
        self.in_flight += 1;
        Ok(())
    }

    fn reap(&mut self, min: usize) -> std::io::Result<()> {
        // block until at least `min` writes have completed, then process every
        // completion that's ready.
        self.ring.submit_and_wait(min)?;
        for cqe in self.ring.completion() {
            // a negative result is a kernel errno for that write — surface it.
            if cqe.result() < 0 {
                return Err(std::io::Error::from_raw_os_error(-cqe.result()));
            }
            // one fewer write outstanding
            self.in_flight -= 1;
        }
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // wait for ALL outstanding writes to land so the client has every token...
        if self.in_flight > 0 {
            self.reap(self.in_flight)?;
        }
        // ...only now is it safe to drop the buffers (kernel is done with them).
        // MUST be called before sending the Done frame.
        self.buffers.clear();
        Ok(())
    }
}

pub fn run(sock_path: String) {
    let _ = START.set(Instant::now());
    eprintln!("JADE OS — jaded v{}", env!("CARGO_PKG_VERSION"));

    let cfg = Config::from_env();
    if let Err(e) = cfg.validate() {
        eprintln!("jaded: ERROR — {e}");
        std::process::exit(1);
    }
    // Warn when a KV-cache type fell back to q8_0 because it was unrecognized.
    for (label, var) in [("K", "JADE_KV_TYPE_K"), ("V", "JADE_KV_TYPE_V")] {
        if let Ok(s) = std::env::var(var) {
            if !KvType::is_known(&s) {
                eprintln!("jaded: unknown KV cache type '{s}' for {label}, falling back to q8_0");
            }
        }
    }

    eprintln!("jaded: loading model: {}", cfg.model_path);
    eprintln!(
        "jaded: ctx_size={} gpu_layers={} n_batch={} n_ubatch={} kv_k={:?} kv_v={:?} flash_attn={}",
        cfg.ctx_size, cfg.gpu_layers, cfg.n_batch, cfg.n_ubatch, cfg.kv_type_k, cfg.kv_type_v, cfg.flash_attn
    );

    let backend = LlamaBackend::init().unwrap_or_else(|e| {
        eprintln!("jaded: llama backend init failed: {e}");
        std::process::exit(1);
    });

    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(cfg.gpu_layers)
        .with_use_mlock(true);
    let model = LlamaModel::load_from_file(&backend, &cfg.model_path, &model_params)
        .unwrap_or_else(|e| {
            eprintln!("jaded: failed to load model from {}: {e}", cfg.model_path);
            std::process::exit(1);
        });

    eprintln!("jaded: model ready");

    let model_name = model.meta_val_str("general.name").unwrap_or_else(|_| {
        std::path::Path::new(&cfg.model_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_owned()
    });
    eprintln!("jaded: model name: {model_name}");

    let chat_template = model.chat_template(None).unwrap_or_else(|e| {
        eprintln!("jaded: WARNING — could not read chat template from GGUF: {e}");
        eprintln!("jaded: falling back to ChatML");
        LlamaChatTemplate::new("chatml").expect("chatml fallback template")
    });
    eprintln!("jaded: chat template loaded");

    let flash_attn_policy = if cfg.flash_attn {
        llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED
    } else {
        llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED
    };

    let mut ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(cfg.ctx_size).unwrap()))
        .with_n_batch(cfg.n_batch)
        .with_n_ubatch(cfg.n_ubatch)
        // Reserve room for N concurrent sequences (RLM fan-out / parallel serving);
        // the shared KV cache is partitioned across them by sequence id.
        .with_n_seq_max(cfg.n_seq_max)
        .with_type_k(kv::kv_cache_type(cfg.kv_type_k))
        .with_type_v(kv::kv_cache_type(cfg.kv_type_v))
        .with_flash_attention_policy(flash_attn_policy);
    // A unified KV buffer puts all sequences in one stream, so cross-sequence seq_cp
    // becomes a cheap metadata-only op (no `is_full` block) — the path multi-block KV
    // composition needs. Since composition is now the default RLM engine, unified is
    // ON by default; set JADE_KV_UNIFIED=0/off/false to opt out (e.g. to A/B the
    // standard path). Disabling it makes the default (compose) RLM path abort, so the
    // legacy agentic engine must be selected explicitly (`"compose": false`).
    let kv_unified = match std::env::var("JADE_KV_UNIFIED") {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false"),
        Err(_) => true,
    };
    if kv_unified {
        ctx_params = ctx_params.with_kv_unified(true);
        eprintln!("jaded: kv_unified enabled (default; set JADE_KV_UNIFIED=0 to disable)");
    } else {
        eprintln!("jaded: kv_unified DISABLED — default (compose) RLM path unavailable");
    }
    let mut ctx = model.new_context(&backend, ctx_params).unwrap_or_else(|e| {
        eprintln!("jaded: context init failed: {e}");
        std::process::exit(1);
    });
    eprintln!("jaded: context ready (ctx_size={})", cfg.ctx_size);

    // Continuous-batching engine self-test: set JADE_RLM_SELFTEST=1 to run a batch of
    // sub-inferences through the multi-sequence scheduler against the just-loaded
    // model, print outputs + timing, and exit. The normal daemon path is untouched.
    if std::env::var("JADE_RLM_SELFTEST").is_ok() {
        rlm::selftest_rlm(&model, &mut ctx, &chat_template);
        return;
    }
    // KV-composition Phase A: state-seq KV save/restore round-trip.
    if std::env::var("JADE_KV_ROUNDTRIP").is_ok() {
        compose::selftest_kv_roundtrip(&model, &mut ctx, &chat_template);
        return;
    }
    // KV-composition C0 spike: compose two precomputed blocks into one sequence.
    if std::env::var("JADE_KV_COMPOSE").is_ok() {
        compose::selftest_kv_compose(&model, &mut ctx, &chat_template);
        return;
    }
    // KV-composition RLM demo: retrieve -> compose -> single decode (needs JADE_KV_UNIFIED=1).
    if std::env::var("JADE_COMPOSE_DEMO").is_ok() {
        compose::selftest_compose_demo(&model, &mut ctx, &chat_template);
        return;
    }
    // Multi-block composition + CacheBlend test (needs JADE_KV_UNIFIED=1).
    if std::env::var("JADE_COMPOSE_MULTI").is_ok() {
        compose::selftest_compose_multi(&model, &mut ctx, &chat_template);
        return;
    }
    // KV transfer-vs-recompute benchmark: GPU re-prefill vs CPU-RAM KV restore (PCIe).
    if std::env::var("JADE_KV_BENCH").is_ok() {
        compose::selftest_kv_bench(&model, &mut ctx, &chat_template);
        return;
    }
    // End-to-end cold-vs-warm compose bench: KV-cache restore vs re-prefill per query.
    if std::env::var("JADE_COMPOSE_CACHE").is_ok() {
        compose::selftest_compose_cache(&model, &mut ctx, &chat_template);
        return;
    }
    // Pure generation-throughput baseline: decode tok/s vs resident context size.
    if std::env::var("JADE_GEN_BENCH").is_ok() {
        compose::selftest_gen_bench(&model, &mut ctx, &chat_template);
        return;
    }
    // Generation tok/s PAST the ctx window via RLM: corpus >> window, decode stays flat.
    if std::env::var("JADE_RLM_OVERWINDOW").is_ok() {
        compose::selftest_rlm_overwindow(&model, &mut ctx, &chat_template);
        return;
    }
    // Blended RAG-RLM: PICK (retrieve) -> FAN-OUT (cross-encoder rerank) -> COMBINE.
    if std::env::var("JADE_BLENDED").is_ok() {
        compose::selftest_blended(&model, &mut ctx, &chat_template);
        return;
    }

    let mut cache = PromptCache::new(cfg.cache_size);
    eprintln!("jaded: prompt cache size={}", cfg.cache_size);

    std::fs::create_dir_all(std::path::Path::new(&sock_path).parent().unwrap())
        .expect("create socket dir");
    let _ = std::fs::remove_file(&sock_path);
    let listener = UnixListener::bind(&sock_path).unwrap_or_else(|e| {
        eprintln!("jaded: could not bind {sock_path}: {e}");
        std::process::exit(1);
    });

    // Tracks the token sequence currently resident in the KV cache so we can
    // reuse the shared prefix across consecutive requests instead of re-prefilling
    // the entire conversation history on every turn.
    let mut kv_tokens: Vec<LlamaToken> = Vec::new();

    eprintln!("jaded: listening on {sock_path}");
    listen(&listener, &model, &mut ctx, &mut cache, &chat_template, &mut kv_tokens, &model_name);
}

fn listen(listener: &UnixListener, model: &LlamaModel, ctx: &mut LlamaContext, cache: &mut PromptCache, tmpl: &LlamaChatTemplate, kv_tokens: &mut Vec<LlamaToken>, model_name: &str) {
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(e) = handle_connection(&mut stream, model, ctx, cache, tmpl, kv_tokens, model_name) {
                    eprintln!("jaded: connection error: {e}");
                }
            }
            Err(e) => eprintln!("jaded: accept error: {e}"),
        }
    }
}

fn handle_connection(
    stream: &mut UnixStream,
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    cache: &mut PromptCache,
    tmpl: &LlamaChatTemplate,
    kv_tokens: &mut Vec<LlamaToken>,
    model_name: &str,
) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::new();      // accumulates raw bytes off the socket
    let mut tmp = [0u8; 4096];              // scratch read buffer
    let mut request_count: u64 = 0;        // just for the close-log line

    // outer loop: one pass per request. A single client connection sends many
    // requests back-to-back over its lifetime, processed strictly in order.
    loop {
        // inner loop: keep reading from the socket until `buf` holds at least one
        // complete request frame ([4-byte len][JSON]), then return it.
        let req = loop {
            match InferenceRequest::decode(&buf) {
                Ok(req) => {
                    // a full frame decoded — consume its bytes from the front of
                    // buf (4 length bytes + the JSON payload) and hand it back.
                    let payload_len =
                        u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
                    buf.drain(..4 + payload_len);
                    break req;
                }
                Err(jade_protocol::RequestDecodeError::Incomplete) => {
                    // not enough bytes yet — read more. n == 0 means EOF (client
                    // hung up cleanly), so this connection is done.
                    let n = stream.read(&mut tmp)?;
                    if n == 0 {
                        eprintln!("jaded: connection closed after {request_count} requests");
                        return Ok(());
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                Err(e) => {
                    // garbage on the wire — we can't resync, so drop the connection.
                    eprintln!("jaded: malformed request: {e}");
                    return Ok(());
                }
            }
        };

        request_count += 1;

        // TODO(trust): daemon-side enforcement hook — currently the runtime is
        // the sole gatekeeper (refuses tainted strings at sh.exec/http.get/fs.read).
        // jaded only records the trust level for observability.

        if req.health_only {
            let health = jade_protocol::Health {
                status: "ok".to_owned(),
                model: model_name.to_owned(),
                model_loaded: true, // we only serve once the model is loaded
                uptime_secs: START.get().map(|s| s.elapsed().as_secs()).unwrap_or(0),
                protocol_version: jade_protocol::PROTOCOL_VERSION,
            };
            let json = serde_json::to_string(&health).unwrap_or_else(|_| "{}".to_owned());
            stream.write_all(&Frame::Json(json).encode())?;
            stream.write_all(&Frame::Done { tokens_used: 0 }.encode())?;
            continue;
        }

        if req.stats_only {
            let total = GLOBAL_TOKENS.load(Ordering::Relaxed);
            stream.write_all(&Frame::Done { tokens_used: total }.encode())?;
            continue;
        }

        if req.count_only {
            let formatted = match build_formatted_prompt(&req.prompt, model, tmpl) {
                Ok(s) => s,
                Err(e) => {
                    stream.write_all(&Frame::Error(e).encode())?;
                    continue;
                }
            };
            let tokens = match model.str_to_token(&formatted, AddBos::Always) {
                Ok(t) => t,
                Err(e) => {
                    stream.write_all(&Frame::Error(format!("tokenize failed: {e}")).encode())?;
                    continue;
                }
            };
            stream.write_all(&Frame::Done { tokens_used: tokens.len() as u64 }.encode())?;
            continue;
        }

        // RLM mode: in-process recursive decomposition over a large (possibly
        // pre-uploaded) context. Separate path from single-shot inference.
        if req.rlm {
            rlm::handle_rlm(&req, model, ctx, stream, model_name)?;
            continue;
        }

        eprintln!(
            "jaded: request — model={} max_tokens={} prompt_len={} trust={}",
            req.model, req.max_tokens, req.prompt.len(), req.trust
        );

        // first frame of every real response is Meta (the serving model name),
        // sent before any tokens so the client knows what answered.
        stream.write_all(&Frame::Meta { model: model_name.to_owned() }.encode())?;

        // token frames go out through the async io_uring writer (overlaps the next
        // token's compute with the kernel draining this token's bytes).
        let mut writer = IoUringWriter::new(stream.as_raw_fd())?;

        // response cache: identical request → replay the stored text as one token
        // frame and skip the model entirely.
        let cache_key = PromptCache::key(&req);
        if let Some((cached_text, tokens_used)) = cache.get(cache_key) {
            eprintln!("jaded: cache hit (key={cache_key:016x})");
            writer.write(Frame::Token(cached_text.clone()).encode())?;
            writer.flush()?; // flush before Done so the client sees the bytes
            GLOBAL_TOKENS.fetch_add(*tokens_used, Ordering::Relaxed);
            stream.write_all(&Frame::Done { tokens_used: *tokens_used }.encode())?;
            continue;
        }

        // cache miss: actually run the model. The closure fires once per generated
        // token — we both stream it to the client AND accumulate it so we can cache
        // the full response afterward.
        let mut response_buf = String::new();
        let result = standard::infer_streaming(&req, model, ctx, tmpl, kv_tokens, |token| {
            response_buf.push_str(token);
            writer.write(Frame::Token(token.to_owned()).encode()).ok();
        });

        // make sure every streamed token byte has landed before the Done/Error frame
        writer.flush()?;

        match result {
            Ok(tokens_used) => {
                // Clean the cached copy to match what was streamed. When keep_anchors
                // is set the stream intentionally includes the boundary (control.rs
                // emits/synthesizes it), so the cached replay must keep it too.
                if !req.keep_anchors {
                    if let Some(stop) = req.stop_anchor.as_deref() {
                        jaded::prompt::strip_stop_anchor(&mut response_buf, stop);
                    }
                }
                GLOBAL_TOKENS.fetch_add(tokens_used, Ordering::Relaxed);
                cache.insert(cache_key, response_buf, tokens_used);
                stream.write_all(&Frame::Done { tokens_used }.encode())?;
            }
            Err(msg) => {
                eprintln!("jaded: inference error: {msg}");
                stream.write_all(&Frame::Error(msg).encode())?;
            }
        }
    }
}

/// Render a request prompt into the model's chat format. A `_jade_protocol`
/// messages object becomes a multi-turn chat; anything else is a single user turn.
/// Shared by the standard path, `count_only`, and the RLM sub-call builder.
pub(crate) fn build_formatted_prompt(
    prompt_str: &str,
    model: &LlamaModel,
    tmpl: &LlamaChatTemplate,
) -> Result<String, String> {
    if let Some((system, messages)) = jaded::prompt::extract_jade_protocol(prompt_str) {
        let mut chat: Vec<LlamaChatMessage> = Vec::new();
        if !system.is_empty() {
            chat.push(LlamaChatMessage::new("system".to_owned(), system)
                .map_err(|e| format!("chat message error: {e}"))?);
        }
        for msg in messages {
            chat.push(LlamaChatMessage::new(msg.role, msg.content)
                .map_err(|e| format!("chat message error: {e}"))?);
        }
        return model.apply_chat_template(tmpl, &chat, true)
            .map_err(|e| format!("chat template error: {e}"));
    }
    // Single-turn: wrap as a user message and apply the model's template.
    let msg = LlamaChatMessage::new("user".to_owned(), prompt_str.to_owned())
        .map_err(|e| format!("chat message error: {e}"))?;
    model.apply_chat_template(tmpl, &[msg], true)
        .map_err(|e| format!("chat template error: {e}"))
}
