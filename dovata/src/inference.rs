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
//!                     per tick) over the pure [`dovata::sched::Scheduler`].
//! * [`compose`]     — Recursive Language Model engine: tensor-level KV-cache
//!                     composition (retrieve → splice block KV → one decode).
//! * [`rlm`]         — RLM socket dispatch: context-handle upload + query routing
//!                     onto the [`compose`] engine.
//! * [`bench`]       — env-gated developer self-tests and benchmarks (not a request
//!                     path); [`bench::dispatch`] runs one and exits if requested.
//!
//! The platform-independent logic (anchors/stop/grammar, config, response cache,
//! scheduler) lives in the `dovata` library and is unit-tested on any platform;
//! everything here is llama.cpp-bound and Linux-only.

mod kv;
mod standard;
mod concurrency;
mod rlm;
mod compose;
mod bench;

use io_uring::{opcode, types, IoUring};
use ovata_infer_protocol::{Frame, InferenceRequest};
use dovata::cache::PromptCache;
use jade_model_profile::ChatTemplate;
use dovata::config::{Config, KvType};
use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, AddBos, LlamaChatTemplate, LlamaModel},
    token::LlamaToken,
};
use std::io::{Read, Write};
use std::num::NonZeroU32;
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// FFI to the index-driven MoE expert-offload driver. `moe_driver_init` is a C
/// symbol compiled into libllama (the `llama-cpp/moe-driver` lib); it mirrors
/// `moe_driver_config` in `moe_driver.h` and `MoeOffload` in `config.rs`. The
/// `build_moe_ffn` seam (also in libllama) calls the rest of the driver on decode.
#[repr(C)]
struct MoeDriverConfig {
    enabled: bool,
    pin: i32,
    cache: i32,
    order_path: *const std::os::raw::c_char,
}

extern "C" {
    /// Returns 0 on success; non-zero leaves the driver inactive (stock path).
    fn moe_driver_init(
        cfg: *const MoeDriverConfig,
        n_layers: i32,
        n_experts: i32,
        n_expert_used: i32,
        expert_order_flat: *const i32,
    ) -> i32;
}

/// Daemon-wide cumulative generated-token counter (served by `stats_only`).
static GLOBAL_TOKENS: AtomicU64 = AtomicU64::new(0);

/// Process start, set once at the top of [`run`]. Backs `Health::uptime_secs`.
static START: OnceLock<Instant> = OnceLock::new();

const QUEUE_DEPTH: u32 = 128;

// ── Over-context routing constants ─────────────────────────────────────────────
//
// A request whose formatted prompt leaves fewer than NATIVE_GEN_HEADROOM tokens of
// window is treated as "over context" and answered by the blended KV-composition
// engine instead of native full-context inference (which would have no room to
// generate, or not fit at all). Native handles everything that fits — highest fidelity.
const NATIVE_GEN_HEADROOM: u32 = 512;
/// Fallback RLM frontier = this × the native window, when no profile/env limit is set.
const RLM_FRONTIER_WINDOW_MULT: u32 = 8;
/// Defaults for the over-context blended pipeline (overridable via env): retrieve
/// `RLM_DEFAULT_N` candidates, rerank, compose the top `RLM_DEFAULT_K`.
const RLM_DEFAULT_N: usize = 12;
const RLM_DEFAULT_K: usize = 4;

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
    eprintln!("JADE OS — dovata v{}", env!("CARGO_PKG_VERSION"));

    let cfg = Config::from_env();
    if let Err(e) = cfg.validate() {
        eprintln!("dovata: ERROR — {e}");
        std::process::exit(1);
    }
    // Warn when a KV-cache type fell back to q8_0 because it was unrecognized.
    for (label, var) in [("K", "JADE_KV_TYPE_K"), ("V", "JADE_KV_TYPE_V")] {
        if let Ok(s) = std::env::var(var) {
            if !KvType::is_known(&s) {
                eprintln!("dovata: unknown KV cache type '{s}' for {label}, falling back to q8_0");
            }
        }
    }

    eprintln!("dovata: loading model: {}", cfg.model_path);
    eprintln!(
        "dovata: ctx_size={} gpu_layers={} n_batch={} n_ubatch={} kv_k={:?} kv_v={:?} flash_attn={}",
        cfg.ctx_size, cfg.gpu_layers, cfg.n_batch, cfg.n_ubatch, cfg.kv_type_k, cfg.kv_type_v, cfg.flash_attn
    );

    let backend = LlamaBackend::init().unwrap_or_else(|e| {
        eprintln!("dovata: llama backend init failed: {e}");
        std::process::exit(1);
    });

    // Pinned because a buft override stores a self-referential pointer into the
    // params (see LlamaModelParams::add_cpu_buft_override).
    let mut model_params = Box::pin(
        LlamaModelParams::default()
            .with_n_gpu_layers(cfg.gpu_layers)
            .with_use_mlock(true),
    );
    // Baseline whole-layer expert offload (stock `--n-cpu-moe`): move the first N
    // layers' expert tensors to CPU. This is the dumb offload the index-driven
    // moe driver is benchmarked against; config.validate() guarantees it isn't
    // combined with JADE_MOE_DRIVER.
    //
    // CRITICAL: add_cpu_buft_override stores a raw pointer into this CString's
    // bytes; llama.cpp reads it (std::regex) at load time. The CString MUST
    // outlive load_from_file, so bind it in this scope (not inside the `if`).
    // Index-driven MoE driver: keep ALL expert layers in host RAM (the driver's
    // page source) via the stock buft override; the driver allocates the small
    // GPU resident cache and pages experts in. add_cpu_moe_override uses a static
    // pattern literal, so no CString lifetime concern. Mutually exclusive with
    // the JADE_CPU_MOE_LAYERS baseline (enforced by Config::validate).
    if cfg.moe.enabled {
        model_params.as_mut().add_cpu_moe_override();
        eprintln!("dovata: MoE driver — all expert layers -> host RAM (page source)");
    }
    let _cpu_moe_pattern: Option<std::ffi::CString> = match cfg.cpu_moe_pattern() {
        Some(pattern) => {
            let cpattern =
                std::ffi::CString::new(pattern).expect("cpu-moe pattern has no interior NUL");
            model_params.as_mut().add_cpu_buft_override(&cpattern);
            eprintln!(
                "dovata: baseline cpu-moe offload — first {} layers' experts -> CPU",
                cfg.cpu_moe_layers
            );
            Some(cpattern)
        }
        None => None,
    };
    let model = LlamaModel::load_from_file(&backend, &cfg.model_path, &model_params)
        .unwrap_or_else(|e| {
            eprintln!("dovata: failed to load model from {}: {e}", cfg.model_path);
            std::process::exit(1);
        });

    eprintln!("dovata: model ready");

    let model_name = model.meta_val_str("general.name").unwrap_or_else(|_| {
        std::path::Path::new(&cfg.model_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_owned()
    });
    eprintln!("dovata: model name: {model_name}");

    // Activate the index-driven MoE driver (the expert tensors are now host-
    // resident from the override above). moe_driver_init lives in libllama (the
    // moe-driver lib compiled into it); it sets up the per-layer residency policy
    // and marks the driver active so the build_moe_ffn seam engages on decode.
    // The resident GPU cache is allocated lazily on the first decode.
    if cfg.moe.enabled {
        let order_path = cfg.moe.order_path.as_deref().unwrap_or_default();
        match dovata::moe::ExpertOrder::load(order_path) {
            Ok(order) => {
                let mc = MoeDriverConfig {
                    enabled: true,
                    pin: cfg.moe.pin as i32,
                    cache: cfg.moe.cache as i32,
                    order_path: std::ptr::null(),
                };
                // n_expert_used is a loose validation hint (the driver reads the
                // real top-k from the tensor shape at decode); Qwen3-30B-A3B = 8.
                let rc = unsafe {
                    moe_driver_init(
                        &mc,
                        order.n_layers as i32,
                        order.n_experts as i32,
                        8,
                        order.flat().as_ptr(),
                    )
                };
                if rc == 0 {
                    eprintln!(
                        "dovata: MoE driver ACTIVE — pin={} cache={} ({}/{} resident/layer, {} layers)",
                        cfg.moe.pin,
                        cfg.moe.cache,
                        cfg.moe.pin + cfg.moe.cache,
                        order.n_experts,
                        order.n_layers
                    );
                } else {
                    eprintln!("dovata: MoE driver init failed (rc={rc}) — running stock path");
                }
            }
            Err(e) => eprintln!("dovata: MoE order load failed ({e}) — running stock path"),
        }
    }

    let chat_template = model.chat_template(None).unwrap_or_else(|e| {
        eprintln!("dovata: WARNING — could not read chat template from GGUF: {e}");
        eprintln!("dovata: falling back to ChatML");
        LlamaChatTemplate::new("chatml").expect("chatml fallback template")
    });
    eprintln!("dovata: chat template loaded");

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
    // becomes a cheap metadata-only op (no `is_full` block) — exactly what multi-block
    // KV composition needs. The RLM engine is composition, so unified is ON by default;
    // set JADE_KV_UNIFIED=0/off/false to opt out (e.g. to A/B the standard path).
    // Disabling it makes the `rlm` request path abort at compose time.
    let kv_unified = match std::env::var("JADE_KV_UNIFIED") {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false"),
        Err(_) => true,
    };
    if kv_unified {
        ctx_params = ctx_params.with_kv_unified(true);
        eprintln!("dovata: kv_unified enabled (default; set JADE_KV_UNIFIED=0 to disable)");
    } else {
        eprintln!("dovata: kv_unified DISABLED — the rlm (compose) request path is unavailable");
    }
    let mut ctx = model.new_context(&backend, ctx_params).unwrap_or_else(|e| {
        eprintln!("dovata: context init failed: {e}");
        std::process::exit(1);
    });
    eprintln!("dovata: context ready (ctx_size={})", cfg.ctx_size);

    // Developer self-tests / benchmarks: if a JADE_* selftest var is set, run that one
    // routine against the just-loaded model and exit WITHOUT serving (see `bench`). The
    // normal daemon path below is untouched when no such var is set.
    if bench::dispatch(&model, &mut ctx, &chat_template) {
        return;
    }

    let mut cache = PromptCache::new(cfg.cache_size);
    eprintln!("dovata: prompt cache size={}", cfg.cache_size);

    std::fs::create_dir_all(std::path::Path::new(&sock_path).parent().unwrap())
        .expect("create socket dir");
    let _ = std::fs::remove_file(&sock_path);
    let listener = UnixListener::bind(&sock_path).unwrap_or_else(|e| {
        eprintln!("dovata: could not bind {sock_path}: {e}");
        std::process::exit(1);
    });

    // Tracks the token sequence currently resident in the KV cache so we can
    // reuse the shared prefix across consecutive requests instead of re-prefilling
    // the entire conversation history on every turn.
    let mut kv_tokens: Vec<LlamaToken> = Vec::new();

    // The RLM efficient frontier (max corpus tokens) for the over-context path,
    // resolved once: the active model's profile, else env, else a multiple of the
    // window. A request whose corpus exceeds this is rejected (see `answer_over_context`).
    let rlm_max = rlm_max_context(&model_name, cfg.ctx_size);
    eprintln!("dovata: rlm frontier (max corpus tokens) = {rlm_max}");

    // Prompt formatting is profile-driven (llama.cpp's apply_chat_template was removed
    // in the vendored fork). Resolve the active model's chat template once.
    let chat_tmpl = resolve_chat_template(&model_name);

    eprintln!("dovata: listening on {sock_path}");
    listen(&listener, &model, &mut ctx, &mut cache, &chat_tmpl, &mut kv_tokens, &model_name, rlm_max);
}

/// The chat template for the active model: the matching profile's `[chat_template]`
/// block, else the ChatML default. The vendored llama.cpp fork removed
/// `apply_chat_template`, so formatting is profile data now (see `prompt::format_prompt`).
fn resolve_chat_template(model_name: &str) -> ChatTemplate {
    jade_model_profile::Registry::builtin()
        .select(model_name)
        .map(|p| p.chat_template.clone())
        .unwrap_or_default()
}

/// The RLM efficient-frontier limit (max corpus tokens) for the active model: env
/// `JADE_RLM_MAX_CONTEXT` (operator override) if set, else the model profile's
/// `rlm_max_context`, else [`RLM_FRONTIER_WINDOW_MULT`]× the native window. This is the
/// limit beyond which the over-context path rejects a request rather than answer with
/// degraded accuracy.
fn rlm_max_context(model_name: &str, ctx_size: u32) -> u32 {
    // operator override wins (e.g. to A/B the limit or test the reject path)
    if let Some(n) = std::env::var("JADE_RLM_MAX_CONTEXT").ok().and_then(|v| v.trim().parse::<u32>().ok()) {
        if n > 0 {
            return n;
        }
    }
    if let Some(p) = jade_model_profile::Registry::builtin().select(model_name) {
        if let Some(m) = p.limits.rlm_max_context() {
            return m;
        }
    }
    ctx_size.saturating_mul(RLM_FRONTIER_WINDOW_MULT)
}

#[allow(clippy::too_many_arguments)]
fn listen(listener: &UnixListener, model: &LlamaModel, ctx: &mut LlamaContext, cache: &mut PromptCache, tmpl: &ChatTemplate, kv_tokens: &mut Vec<LlamaToken>, model_name: &str, rlm_max: u32) {
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(e) = handle_connection(&mut stream, model, ctx, cache, tmpl, kv_tokens, model_name, rlm_max) {
                    eprintln!("dovata: connection error: {e}");
                }
            }
            Err(e) => eprintln!("dovata: accept error: {e}"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_connection(
    stream: &mut UnixStream,
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    cache: &mut PromptCache,
    tmpl: &ChatTemplate,
    kv_tokens: &mut Vec<LlamaToken>,
    model_name: &str,
    rlm_max: u32,
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
                Err(ovata_infer_protocol::RequestDecodeError::Incomplete) => {
                    // not enough bytes yet — read more. n == 0 means EOF (client
                    // hung up cleanly), so this connection is done.
                    let n = stream.read(&mut tmp)?;
                    if n == 0 {
                        eprintln!("dovata: connection closed after {request_count} requests");
                        return Ok(());
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                Err(e) => {
                    // garbage on the wire — we can't resync, so drop the connection.
                    eprintln!("dovata: malformed request: {e}");
                    return Ok(());
                }
            }
        };

        request_count += 1;

        // TODO(trust): daemon-side enforcement hook — currently the runtime is
        // the sole gatekeeper (refuses tainted strings at sh.exec/http.get/fs.read).
        // dovata only records the trust level for observability.

        if req.health_only {
            let health = ovata_infer_protocol::Health {
                status: "ok".to_owned(),
                model: model_name.to_owned(),
                model_loaded: true, // we only serve once the model is loaded
                uptime_secs: START.get().map(|s| s.elapsed().as_secs()).unwrap_or(0),
                protocol_version: ovata_infer_protocol::PROTOCOL_VERSION,
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
            let formatted = match build_formatted_prompt(&req.prompt, tmpl) {
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

        // RLM mode: explicit, pre-chunked / context-handle decomposition over a large
        // context. Separate path from single-shot inference (now blended-backed).
        if req.rlm {
            rlm::handle_rlm(&req, model, ctx, stream, model_name)?;
            continue;
        }

        // Over-context routing: tokenize the formatted prompt up front (cheap vs.
        // inference) to decide the path. A prompt that won't fit the native window
        // with room to generate is answered by the blended KV-composition engine;
        // everything that fits falls through to native full-context inference below.
        match build_formatted_prompt(&req.prompt, tmpl)
            .and_then(|f| model.str_to_token(&f, AddBos::Always).map_err(|e| format!("tokenize failed: {e}")))
        {
            Ok(toks) if toks.len() as u32 + NATIVE_GEN_HEADROOM > ctx.n_ctx() => {
                eprintln!(
                    "dovata: over-context ({} prompt tokens, window {}) → blended RLM compose",
                    toks.len(), ctx.n_ctx()
                );
                answer_over_context(&req, model, ctx, stream, model_name, rlm_max)?;
                continue;
            }
            Ok(_) => {} // fits → native path below
            Err(e) => {
                stream.write_all(&Frame::Error(e).encode())?;
                continue;
            }
        }

        eprintln!(
            "dovata: request — model={} max_tokens={} prompt_len={} trust={}",
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
            eprintln!("dovata: cache hit (key={cache_key:016x})");
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
                        dovata::prompt::strip_stop_anchor(&mut response_buf, stop);
                    }
                }
                GLOBAL_TOKENS.fetch_add(tokens_used, Ordering::Relaxed);
                cache.insert(cache_key, response_buf, tokens_used);
                stream.write_all(&Frame::Done { tokens_used }.encode())?;
            }
            Err(msg) => {
                eprintln!("dovata: inference error: {msg}");
                stream.write_all(&Frame::Error(msg).encode())?;
            }
        }
    }
}

/// Answer an over-context request with the blended KV-composition engine. The prompt
/// is split into a retrievable corpus + a question (`dovata::route`), chunked into
/// blocks, and — if the corpus is within the model's RLM frontier (`rlm_max`) — answered
/// by retrieve→rerank→compose→single-decode. A corpus beyond the frontier is rejected
/// with an `Error` frame (the documented limit) rather than answered with degraded
/// accuracy. Emits `Meta` → `Token`(answer) → `Done`, mirroring the explicit `rlm` path.
fn answer_over_context(
    req: &InferenceRequest,
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    stream: &mut UnixStream,
    model_name: &str,
    rlm_max: u32,
) -> std::io::Result<()> {
    // Split prompt → (corpus blocks, question) and tokenize the corpus into a store.
    let input = dovata::route::derive_rlm_input(&req.prompt, dovata::route::DEFAULT_CHUNK_CHARS);
    let store = match compose::BlockStore::from_texts(model, &input.corpus) {
        Ok(s) => s,
        Err(e) => {
            stream.write_all(&Frame::Error(format!("over-context chunking failed: {e}")).encode())?;
            return Ok(());
        }
    };
    let corpus_tokens: usize = store.blocks.iter().map(|b| b.ids.len()).sum();

    // Frontier gate: reject a corpus larger than this model's measured RLM limit so
    // we never silently answer past the point accuracy is known to hold.
    if corpus_tokens as u32 > rlm_max {
        let msg = format!(
            "context corpus ({corpus_tokens} tokens) exceeds this model's RLM limit \
             ({rlm_max} tokens); reduce the context (see the model profile's limits.rlm_max_context)"
        );
        eprintln!("dovata: [over-context] REJECT — {msg}");
        stream.write_all(&Frame::Error(msg).encode())?;
        return Ok(());
    }

    // Defaults overridable via env, mirroring the bench knobs.
    let n_cand = std::env::var("JADE_RLM_N").ok().and_then(|v| v.parse().ok()).unwrap_or(RLM_DEFAULT_N);
    let k = std::env::var("JADE_RLM_K").ok().and_then(|v| v.parse().ok()).unwrap_or(RLM_DEFAULT_K);

    // One-shot request → ephemeral (cap-0) KV cache; PICK→rerank→COMBINE→one decode.
    let mut kv = compose::KvBlockCache::new(0);
    match compose::blended_compose_and_answer(model, ctx, &store, &mut kv, &input.question, n_cand, k.max(1)) {
        Ok((answer, picked, _scored)) => {
            eprintln!(
                "dovata: [over-context] {} blocks ({corpus_tokens} tok) → composed {picked:?}",
                store.blocks.len()
            );
            stream.write_all(&Frame::Meta { model: model_name.to_owned() }.encode())?;
            stream.write_all(&Frame::Token(answer).encode())?;
            stream.write_all(&Frame::Done { tokens_used: 0 }.encode())?;
        }
        Err(e) => {
            stream.write_all(&Frame::Error(format!("over-context compose: {e}")).encode())?;
        }
    }
    Ok(())
}

/// Render a request prompt into the model's chat format. A `_ovata_infer_protocol`
/// messages object becomes a multi-turn chat; anything else is a single user turn.
/// Shared by the standard path, `count_only`, and the RLM sub-call builder.
///
/// Formatting is profile-driven via the model's [`ChatTemplate`] — the vendored
/// llama.cpp fork removed `apply_chat_template`. The real work is the pure, tested
/// `prompt::format_prompt`; this stays `Result` so the call sites are unchanged
/// (rendering itself is now infallible).
pub(crate) fn build_formatted_prompt(
    prompt_str: &str,
    tmpl: &ChatTemplate,
) -> Result<String, String> {
    Ok(dovata::prompt::format_prompt(prompt_str, tmpl))
}
