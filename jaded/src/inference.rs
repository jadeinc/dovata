//! llama.cpp + io_uring inference backend (Linux, `--features inference`).
//!
//! Owns the model/context, the Unix-socket server loop, prefill with KV-cache
//! prefix reuse, and the per-token generation loop. The fragile control logic
//! (anchors, stop-anchor buffering, JSON depth tracking) lives in the platform-
//! independent [`jaded::control::DecodeControl`]; configuration in
//! [`jaded::config::Config`]; the response cache in [`jaded::cache::PromptCache`].

use io_uring::{opcode, types, IoUring};
use jade_protocol::{Frame, InferenceRequest};
use jaded::cache::PromptCache;
use jaded::config::{Config, KvType};
use jaded::control::DecodeControl;
use jaded::sched::Scheduler;
use llama_cpp_2::{
    context::{params::{KvCacheType, LlamaContextParams}, LlamaContext},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use std::io::{Read, Write};
use std::num::NonZeroU32;
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static GLOBAL_TOKENS: AtomicU64 = AtomicU64::new(0);

/// Process start, set once at the top of [`run`]. Backs `Health::uptime_secs`.
static START: OnceLock<Instant> = OnceLock::new();

const QUEUE_DEPTH: u32 = 128;

/// Map our pure [`KvType`] onto llama.cpp's `KvCacheType`.
fn kv_cache_type(t: KvType) -> KvCacheType {
    match t {
        KvType::F16 => KvCacheType::F16,
        KvType::BF16 => KvCacheType::BF16,
        KvType::F32 => KvCacheType::F32,
        KvType::Q8_0 => KvCacheType::Q8_0,
        KvType::Q5_1 => KvCacheType::Q5_1,
        KvType::Q5_0 => KvCacheType::Q5_0,
        KvType::Q4_1 => KvCacheType::Q4_1,
        KvType::Q4_0 => KvCacheType::Q4_0,
    }
}

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

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(cfg.ctx_size).unwrap()))
        .with_n_batch(cfg.n_batch)
        .with_n_ubatch(cfg.n_ubatch)
        // Reserve room for N concurrent sequences (RLM fan-out / parallel serving);
        // the shared KV cache is partitioned across them by sequence id.
        .with_n_seq_max(cfg.n_seq_max)
        .with_type_k(kv_cache_type(cfg.kv_type_k))
        .with_type_v(kv_cache_type(cfg.kv_type_v))
        .with_flash_attention_policy(flash_attn_policy);
    let mut ctx = model.new_context(&backend, ctx_params).unwrap_or_else(|e| {
        eprintln!("jaded: context init failed: {e}");
        std::process::exit(1);
    });
    eprintln!("jaded: context ready (ctx_size={})", cfg.ctx_size);

    // RLM concurrent-engine self-test: set JADE_RLM_SELFTEST=1 to run a batch of
    // sub-inferences through the multi-sequence scheduler against the just-loaded
    // model, print outputs + timing, and exit. The normal daemon path is untouched.
    if std::env::var("JADE_RLM_SELFTEST").is_ok() {
        selftest_rlm(&model, &mut ctx, &chat_template);
        return;
    }
    // End-to-end RLM demo (needle-in-haystack over a large RAM-held document).
    if std::env::var("JADE_RLM_DEMO").is_ok() {
        selftest_rlm_demo(&model, &mut ctx, &chat_template);
        return;
    }
    // Latency A/B: standard single-shot vs RLM on a doc that fits the window.
    if std::env::var("JADE_RLM_COMPARE").is_ok() {
        selftest_rlm_compare(&model, &mut ctx, &chat_template);
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
            handle_rlm(&req, model, ctx, tmpl, stream, model_name)?;
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
        let result = infer_streaming(&req, model, ctx, tmpl, kv_tokens, |token| {
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

fn build_formatted_prompt(
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

fn infer_streaming(
    req: &InferenceRequest,
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    tmpl: &LlamaChatTemplate,
    kv_tokens: &mut Vec<LlamaToken>,
    mut on_token: impl FnMut(&str),
) -> Result<u64, String> {
    let mut decoder = encoding_rs::UTF_8.new_decoder_without_bom_handling();

    let prompt = build_formatted_prompt(&req.prompt, model, tmpl)?;

    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| format!("tokenization failed: {e}"))?;

    let n_prompt = tokens.len();
    let ctx_size = ctx.n_ctx();
    if n_prompt >= ctx_size as usize {
        return Err(format!("prompt ({n_prompt} tokens) exceeds context size ({ctx_size})"));
    }

    // Find how many leading tokens the new prompt shares with the current KV cache.
    // We can skip re-prefilling the shared prefix and only process the new suffix,
    // which is the main cost driver for multi-turn chat (system prompt + history grows
    // by one turn per message but was being fully re-prefilled every time).
    let prefix_len = tokens.iter()
        .zip(kv_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count()
        // Never reuse the very last cached token — it was the generation seed for
        // the previous response and its logits are stale.
        .saturating_sub(1)
        .min(n_prompt.saturating_sub(1));

    if prefix_len > 0 {
        // Evict only the suffix beyond the shared prefix.
        ctx.clear_kv_cache_seq(Some(0), Some(prefix_len as u32), None)
            .map_err(|e| format!("kv_cache_seq clear failed: {e}"))?;
        kv_tokens.truncate(prefix_len);
        eprintln!("jaded: kv prefix reuse {prefix_len}/{} tokens ({} new)",
            n_prompt, n_prompt - prefix_len);
    } else {
        ctx.clear_kv_cache();
        kv_tokens.clear();
    }

    // Chunked prefill: process only the new suffix tokens at their absolute positions.
    // We start at prefix_len (the shared prefix is already in the KV cache) and feed
    // the model the new tokens in batches of at most n_batch at a time.
    let n_batch = ctx.n_batch() as usize;
    let last = n_prompt - 1; // index of the final prompt token (needs logits)
    let mut pos = prefix_len;
    while pos < n_prompt {
        // this chunk covers [pos, chunk_end); never more than n_batch tokens
        let chunk_end = (pos + n_batch).min(n_prompt);
        let mut batch = LlamaBatch::new(chunk_end - pos, 1);
        for i in pos..chunk_end {
            // add token i at absolute position i, sequence 0. The 4th arg asks for
            // logits ONLY on the very last prompt token — that's the one we sample
            // the first generated token from; the rest are just context.
            batch.add(tokens[i], i as i32, &[0], i == last)
                .map_err(|e| format!("batch add failed: {e}"))?;
        }
        // run the forward pass for this chunk, populating the KV cache
        ctx.decode(&mut batch).map_err(|e| format!("prefill failed at chunk {pos}..{chunk_end}: {e}"))?;
        pos = chunk_end;
    }

    // Record the prompt tokens in kv_tokens (suffix appended to retained prefix).
    kv_tokens.extend_from_slice(&tokens[prefix_len..]);

    // Free sampler used for unconstrained tokens (and all tokens once grammar is retired).
    let mut free_sampler = LlamaSampler::chain_simple(vec![
        LlamaSampler::temp(0.6),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::dist(0),
    ]);

    // Grammar sampler (constrained decoding). Kept allocated until after the
    // generation loop — freeing a grammar sampler while the context is active
    // crashes llama.cpp.
    let mut anchor_sampler: Option<LlamaSampler> = if let Some(ref gbnf) = req.grammar {
        match LlamaSampler::grammar(model, gbnf, "root") {
            Ok(g) => Some(LlamaSampler::chain_simple(vec![
                LlamaSampler::temp(0.6),
                g,
                LlamaSampler::min_p(0.05, 1),
                LlamaSampler::dist(0),
            ])),
            Err(e) => return Err(format!("GBNF grammar error: {e:?}")),
        }
    } else {
        None
    };

    // All anchor / stop-anchor / depth-tracking decisions live in this pure state
    // machine (see jaded::control). The engine only asks it which sampler to use
    // and what text to emit.
    let mut control = DecodeControl::new(req);

    // cap on new tokens: whichever is smaller — the room left in the context
    // window, or what the request asked for.
    let max_new = (ctx_size as usize).saturating_sub(n_prompt).min(req.max_tokens as usize);
    let mut n_gen = 0usize;
    let mut gen_batch = LlamaBatch::new(1, 1); // we decode one new token at a time

    loop {
        // Pick the sampler the control machine tells us to use this step, and
        // sample the next token from the most recent logits (index -1).
        //
        // LlamaSampler::sample() already accepts the token internally
        // (llama_sampler_sample calls llama_sampler_accept at the end). Do NOT
        // call sampler.accept() again — double-accepting empties the grammar
        // stacks and causes an abort on the next sample.
        let token = if control.use_grammar() {
            // constrained step: use the grammar sampler, then tell control it ran
            // (so it can retire the grammar after this first anchored token).
            let gsamp = anchor_sampler.as_mut().unwrap();
            let t = gsamp.sample(&ctx, -1);
            control.notice_grammar_sampled();
            t
        } else {
            // free step: ordinary temperature/min-p sampling
            free_sampler.sample(&ctx, -1)
        };

        // stop conditions: the model emitted an end-of-generation token, or we hit
        // the token budget. Either way we don't emit this token.
        if model.is_eog_token(token) || n_gen >= max_new {
            break;
        }

        // turn the token id into its text fragment (a "piece"); the decoder handles
        // multi-token UTF-8 sequences split across pieces.
        let piece = model.token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| format!("token to str failed: {e}"))?;
        n_gen += 1;

        // run the piece through the control machine; it returns the text that's
        // actually safe to stream (may be None while buffering a stop anchor).
        if let Some(text) = control.on_piece(&piece) {
            on_token(&text);
        }

        // record this token as now resident in the KV cache (for next-request
        // prefix reuse).
        kv_tokens.push(token);

        // feed the just-generated token back in as the next position so the model
        // can produce the following token. position = n_prompt + n_gen - 1.
        gen_batch.clear();
        let gen_pos = (n_prompt + n_gen - 1) as i32;
        gen_batch.add(token, gen_pos, &[0], true)
            .map_err(|e| format!("batch add failed: {e}"))?;
        ctx.decode(&mut gen_batch).map_err(|e| format!("decode step failed: {e}"))?;
    }

    // Flush anything buffered for stop-anchor prefix detection that never
    // completed (EOG, max_new, or depth-tracker retirement): it is real output.
    if let Some(tail) = control.finish() {
        on_token(&tail);
    }

    // Keep the grammar sampler alive until here (see allocation comment above).
    drop(anchor_sampler);

    Ok((n_prompt + n_gen) as u64)
}

// ── RLM concurrent engine (continuous batching) ────────────────────────────────
//
// The single-shot `infer_streaming` above decodes ONE sequence. `run_concurrent`
// runs N sub-inferences at once against the SAME resident model/context, driven by
// the pure `jaded::sched::Scheduler`: every `ctx.decode` packs tokens from every
// live sequence into one forward pass (the RLM fan-out latency win). This is the
// Phase-1 primitive the recurse-action (Phase 2) calls; the daemon's normal
// request path does not use it yet.

/// One sub-inference to run on its own sequence id.
struct SubInfer {
    /// Already-tokenized prompt suffix (the caller does any prefix reuse first).
    tokens: Vec<LlamaToken>,
    /// Absolute KV position the suffix starts at (0 for an independent sub-call;
    /// >0 when a shared prefix was pre-copied into this sequence's KV).
    base_pos: i32,
    /// Optional GBNF for constrained decoding (grammar sampler built per sequence).
    grammar: Option<String>,
    /// Per-sequence decode state machine (anchors / stop / depth tracking).
    control: DecodeControl,
    /// Generated-token cap for this sub-call.
    max_new: usize,
}

/// Run all `subs` concurrently against `ctx`, returning each one's generated text
/// (indexed parallel to `subs`). One `ctx.decode` per tick advances every live
/// sequence; per-sequence logits are sampled at their batch positions.
fn run_concurrent(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    subs: Vec<SubInfer>,
) -> Result<Vec<String>, String> {
    let n = subs.len();
    let n_batch = ctx.n_batch() as usize;
    // The whole KV cache (n_ctx cells) is the budget shared across sequences.
    let kv_budget = ctx.n_ctx() as usize;
    let mut sched = Scheduler::new(n_batch, kv_budget);

    let mut outputs = vec![String::new(); n];
    // Per-sequence UTF-8 decoders — a multi-byte char split across two BPE tokens
    // of the SAME sequence must not interleave with another sequence's bytes.
    let mut decoders: Vec<_> =
        (0..n).map(|_| encoding_rs::UTF_8.new_decoder_without_bom_handling()).collect();
    // One free sampler shared by all unconstrained samples (no per-seq state).
    let mut free_sampler = LlamaSampler::chain_simple(vec![
        LlamaSampler::temp(0.6),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::dist(0),
    ]);
    // Per-sequence grammar samplers (stateful grammar stacks), kept alive for the
    // whole run — freeing one while the context is live crashes llama.cpp.
    let mut grammar_samplers: Vec<Option<LlamaSampler>> = Vec::with_capacity(n);

    // Start every sequence from a clean KV cache.
    ctx.clear_kv_cache();

    for (i, sub) in subs.into_iter().enumerate() {
        let gs = if let Some(ref gbnf) = sub.grammar {
            match LlamaSampler::grammar(model, gbnf, "root") {
                Ok(g) => Some(LlamaSampler::chain_simple(vec![
                    LlamaSampler::temp(0.6),
                    g,
                    LlamaSampler::min_p(0.05, 1),
                    LlamaSampler::dist(0),
                ])),
                Err(e) => return Err(format!("GBNF grammar error (sub {i}): {e:?}")),
            }
        } else {
            None
        };
        grammar_samplers.push(gs);
        let toks: Vec<i32> = sub.tokens.iter().map(|t| t.0).collect();
        if !sched.admit(i as i32, toks, sub.base_pos, sub.control, sub.max_new) {
            return Err(format!(
                "sub {i} rejected by KV admission (budget {kv_budget} tokens exceeded)"
            ));
        }
    }

    // Drive the scheduler until every sequence has retired.
    while !sched.is_idle() {
        let plan = sched.plan();
        if plan.is_empty() {
            break; // all live sequences are awaiting a sample we already produced
        }
        // Translate the plan into one batch; tag each token with its sequence id.
        let mut batch = LlamaBatch::new(plan.entries.len(), n as i32);
        for e in &plan.entries {
            batch
                .add(LlamaToken(e.token), e.pos, &[e.seq_id], e.want_logits)
                .map_err(|err| format!("batch add failed: {err}"))?;
        }
        ctx.decode(&mut batch).map_err(|err| format!("decode failed: {err}"))?;

        // The k-th sampleable output sits at the k-th logits-enabled batch position.
        let logit_pos: Vec<usize> = plan
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.want_logits.then_some(i))
            .collect();

        for (k, slot) in plan.sample_order.iter().enumerate() {
            let sid = slot.seq_id as usize;
            let idx = logit_pos[k] as i32; // batch position of this sequence's logits
            let token = if slot.use_grammar {
                grammar_samplers[sid].as_mut().expect("grammar sampler present").sample(ctx, idx)
            } else {
                free_sampler.sample(ctx, idx)
            };
            let is_eog = model.is_eog_token(token);
            let piece = if is_eog {
                String::new()
            } else {
                model
                    .token_to_piece(token, &mut decoders[sid], true, None)
                    .map_err(|err| format!("token to str failed: {err}"))?
            };
            let ev = sched.accept(slot.seq_id, token.0, &piece, is_eog);
            if let Some(text) = ev.emit {
                outputs[sid].push_str(&text);
            }
        }
    }

    // Keep grammar samplers alive until after the loop (the sharp edge), then reset.
    drop(grammar_samplers);
    ctx.clear_kv_cache();
    Ok(outputs)
}

/// Self-test for the concurrent engine: build a handful of sub-inferences, run them
/// both sequentially (baseline) and concurrently (one batch/tick), and print
/// outputs + wall-clock so the fan-out speedup and correctness are visible. Gated
/// by `JADE_RLM_SELFTEST`. Check `nvidia-smi` alongside to confirm VRAM tracks
/// live KV rather than scaling with the number of sub-calls.
fn selftest_rlm(model: &LlamaModel, ctx: &mut LlamaContext, tmpl: &LlamaChatTemplate) {
    let prompts = [
        "Reply with exactly one word: the capital of France.",
        "Reply with exactly one word: a primary color.",
        "Reply with exactly one word: the largest planet.",
        "Reply with exactly one word: the opposite of hot.",
        "Reply with exactly one word: a fruit that is yellow.",
        "Reply with exactly one word: the chemical symbol for water.",
        "Reply with exactly one word: the number after ten.",
        "Reply with exactly one word: a animal that barks.",
    ];
    let max_new = 24usize;

    // Build a SubInfer per prompt (tokenize through the model's chat template).
    let build_subs = || -> Result<Vec<SubInfer>, String> {
        prompts
            .iter()
            .map(|p| {
                let formatted = build_formatted_prompt(p, model, tmpl)?;
                let tokens = model
                    .str_to_token(&formatted, AddBos::Always)
                    .map_err(|e| format!("tokenize failed: {e}"))?;
                let req = InferenceRequest {
                    prompt: (*p).to_owned(),
                    max_tokens: max_new as u32,
                    ..Default::default()
                };
                Ok(SubInfer { tokens, base_pos: 0, grammar: None, control: DecodeControl::new(&req), max_new })
            })
            .collect()
    };

    eprintln!("jaded: [selftest] {} sub-inferences, max_new={max_new}", prompts.len());

    // Sequential baseline: one sub per run_concurrent call, summed wall-clock.
    let seq_start = Instant::now();
    let mut seq_outputs = Vec::new();
    match build_subs() {
        Ok(subs) => {
            for sub in subs {
                match run_concurrent(model, ctx, vec![sub]) {
                    Ok(mut o) => seq_outputs.push(o.pop().unwrap_or_default()),
                    Err(e) => { eprintln!("jaded: [selftest] sequential error: {e}"); return; }
                }
            }
        }
        Err(e) => { eprintln!("jaded: [selftest] build error: {e}"); return; }
    }
    let seq_ms = seq_start.elapsed().as_millis();

    // Concurrent: all subs in one run (one decode per tick advances all).
    let con_start = Instant::now();
    let con_outputs = match build_subs() {
        Ok(subs) => match run_concurrent(model, ctx, subs) {
            Ok(o) => o,
            Err(e) => { eprintln!("jaded: [selftest] concurrent error: {e}"); return; }
        },
        Err(e) => { eprintln!("jaded: [selftest] build error: {e}"); return; }
    };
    let con_ms = con_start.elapsed().as_millis();

    eprintln!("jaded: [selftest] ── outputs (concurrent) ──");
    for (i, (p, o)) in prompts.iter().zip(con_outputs.iter()).enumerate() {
        eprintln!("  [{i}] {p}\n       -> {:?}", o.trim());
    }
    eprintln!("jaded: [selftest] sequential={seq_ms}ms  concurrent={con_ms}ms  speedup={:.2}x",
        seq_ms as f64 / con_ms.max(1) as f64);
    // Sanity: concurrent outputs should match sequential (same greedy-ish sampling
    // path, modulo the shared free sampler's RNG — compare trimmed first words).
    let matches = seq_outputs.iter().zip(con_outputs.iter())
        .filter(|(a, b)| a.trim() == b.trim()).count();
    eprintln!("jaded: [selftest] concurrent==sequential on {matches}/{} outputs", prompts.len());

    // Concurrency sweep: latency + throughput vs N concurrent sub-inferences (the
    // A10G knee). Latency staying ~flat as N grows = continuous batching working.
    eprintln!("jaded: [selftest] ── concurrency sweep (N concurrent, max_new={max_new}) ──");
    for &nn in &[1usize, 2, 4, 8, 16] {
        let mut subs = Vec::with_capacity(nn);
        let mut ok = true;
        for k in 0..nn {
            match build_sub(model, tmpl, prompts[k % prompts.len()], None, max_new) {
                Ok(s) => subs.push(s),
                Err(_) => { ok = false; break; }
            }
        }
        if !ok { continue; }
        let t = Instant::now();
        if run_concurrent(model, ctx, subs).is_err() { continue; }
        let ms = t.elapsed().as_millis().max(1);
        eprintln!("  N={nn:2}  {ms:4}ms  throughput {:.1} sub-inferences/s", nn as f64 * 1000.0 / ms as f64);
    }
}

// ── RLM: context store + model-driven recursion (Phase 1d + Phase 2) ───────────
//
// The headline win: reason over a context far larger than the model window / VRAM.
// The full context lives in CPU RAM as sliceable spans (`ContextStore`); only the
// spans the ROOT model chooses to look at are ever tokenized into a prompt and put
// on the GPU. The driver is model-driven recursion in three in-process steps (zero
// socket roundtrips):
//   PICK    — the root sees a structural MAP of the spans + the question and emits
//             (grammar-forced) a JSON array of the span ids to examine;
//   FAN-OUT — those spans run as CONCURRENT sub-inferences (`run_concurrent`), each
//             answering the question from its excerpt alone — the latency win;
//   COMBINE — the root sees the per-span findings and writes the final answer.
// Depth-1 here; a sub-call is itself a `run_rlm` away from genuine deeper recursion.

/// A large context held in CPU RAM, addressable by span. Each span carries a short
/// `summary` (shown to the root in the map) and its full `body` text (only fetched
/// into a prompt if the root selects it).
#[derive(Clone)]
struct ContextStore {
    summaries: Vec<String>,
    bodies: Vec<String>,
}

impl ContextStore {
    fn from_sections(sections: &[(String, String)]) -> Self {
        ContextStore {
            summaries: sections.iter().map(|(s, _)| s.clone()).collect(),
            bodies: sections.iter().map(|(_, b)| b.clone()).collect(),
        }
    }
    fn len(&self) -> usize {
        self.bodies.len()
    }
    /// The structural map the root navigates: one line per span, `id: summary`.
    fn map_text(&self) -> String {
        self.summaries
            .iter()
            .enumerate()
            .map(|(i, s)| format!("{i}: {s}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// GBNF forcing a JSON array of integers, e.g. `[2, 5]`. Contains `[0-9]`, so
/// `DecodeControl` treats it as a full-output primitive grammar (never retired).
const SPAN_ARRAY_GBNF: &str = r#"root ::= "[" ws (num (ws "," ws num)*)? ws "]"
num ::= [0-9]+
ws ::= [ \t\n]*"#;

/// Pull the integers out of a model-emitted span array like `[3, 7]`.
fn parse_span_ids(s: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_digit() {
            cur.push(ch);
        } else if !cur.is_empty() {
            if let Ok(n) = cur.parse::<usize>() {
                out.push(n);
            }
            cur.clear();
        }
    }
    if let Ok(n) = cur.parse::<usize>() {
        out.push(n);
    }
    out
}

/// Build one [`SubInfer`] from a plain prompt: apply the chat template, tokenize,
/// and attach a fresh `DecodeControl` (+ optional grammar).
fn build_sub(
    model: &LlamaModel,
    tmpl: &LlamaChatTemplate,
    prompt: &str,
    grammar: Option<&str>,
    max_new: usize,
) -> Result<SubInfer, String> {
    let formatted = build_formatted_prompt(prompt, model, tmpl)?;
    let tokens = model
        .str_to_token(&formatted, AddBos::Always)
        .map_err(|e| format!("tokenize failed: {e}"))?;
    let req = InferenceRequest {
        prompt: prompt.to_owned(),
        max_tokens: max_new as u32,
        grammar: grammar.map(|g| g.to_owned()),
        ..Default::default()
    };
    Ok(SubInfer {
        tokens,
        base_pos: 0,
        grammar: grammar.map(|g| g.to_owned()),
        control: DecodeControl::new(&req),
        max_new,
    })
}

/// Answer `question` over `store` by model-driven recursion (PICK → FAN-OUT →
/// COMBINE). Returns the final answer plus the span ids the root chose (for
/// observability). `max_spans` caps fan-out width (the concurrency ceiling).
fn run_rlm(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    tmpl: &LlamaChatTemplate,
    question: &str,
    store: &ContextStore,
    max_spans: usize,
) -> Result<(String, Vec<usize>), String> {
    // ── PICK: root chooses which spans to examine (grammar-forced array) ────────
    let pick_prompt = format!(
        "You are navigating a large document split into numbered sections:\n{}\n\n\
         Question: {question}\n\n\
         Reply with ONLY a JSON array of the section numbers most likely to contain \
         the answer (e.g. [2, 5]). Choose at most {max_spans}.",
        store.map_text()
    );
    let pick_sub = build_sub(model, tmpl, &pick_prompt, Some(SPAN_ARRAY_GBNF), 48)?;
    let t_pick = Instant::now();
    let pick_out = run_concurrent(model, ctx, vec![pick_sub])?.pop().unwrap_or_default();
    let pick_ms = t_pick.elapsed().as_millis();
    let mut picked: Vec<usize> = parse_span_ids(&pick_out)
        .into_iter()
        .filter(|&i| i < store.len())
        .collect();
    picked.dedup();
    picked.truncate(max_spans);
    // Fallback: if the model emitted nothing usable, scan the first few spans.
    if picked.is_empty() {
        picked = (0..store.len().min(max_spans)).collect();
    }
    eprintln!("jaded: [rlm] PICK -> spans {picked:?} (raw {:?})", pick_out.trim());

    // ── FAN-OUT: one concurrent sub-inference per chosen span ───────────────────
    let mut subs = Vec::with_capacity(picked.len());
    for &i in &picked {
        let p = format!(
            "Document excerpt (section {i}):\n{}\n\n\
             Question: {question}\n\
             Answer using ONLY this excerpt. If the answer is not present here, reply \
             exactly: NOT FOUND.",
            store.bodies[i]
        );
        subs.push(build_sub(model, tmpl, &p, None, 80)?);
    }
    let t_fan = Instant::now();
    let findings = run_concurrent(model, ctx, subs)?;
    let fan_ms = t_fan.elapsed().as_millis();
    for (k, &i) in picked.iter().enumerate() {
        eprintln!("jaded: [rlm] span {i} -> {:?}", findings[k].trim());
    }

    // ── COMBINE: root writes the final answer from the findings ─────────────────
    let mut combine = format!("Question: {question}\n\nFindings from the document:\n");
    for (k, &i) in picked.iter().enumerate() {
        combine.push_str(&format!("- (section {i}) {}\n", findings[k].trim()));
    }
    combine.push_str("\nUsing only these findings, give the final answer concisely.");
    let combine_sub = build_sub(model, tmpl, &combine, None, 120)?;
    let t_comb = Instant::now();
    let answer = run_concurrent(model, ctx, vec![combine_sub])?.pop().unwrap_or_default();
    let comb_ms = t_comb.elapsed().as_millis();
    eprintln!(
        "jaded: [rlm] timing: pick={pick_ms}ms fan-out({} spans)={fan_ms}ms combine={comb_ms}ms",
        picked.len()
    );
    Ok((answer, picked))
}

// ── RLM over the socket: context handles + request dispatch (Phase 3) ──────────

/// Server-side context store: a program uploads a large context ONCE and gets a
/// handle, then issues many cheap `rlm` queries against it (no re-sending the doc).
static RLM_STORES: OnceLock<Mutex<HashMap<u64, ContextStore>>> = OnceLock::new();
static RLM_NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

fn rlm_stores() -> &'static Mutex<HashMap<u64, ContextStore>> {
    RLM_STORES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Store an uploaded context, returning its handle.
fn store_context(store: ContextStore) -> u64 {
    let h = RLM_NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    rlm_stores().lock().unwrap().insert(h, store);
    h
}

/// Fetch a clone of a stored context (cloned so the lock isn't held during the
/// long-running inference that follows).
fn get_context(h: u64) -> Option<ContextStore> {
    rlm_stores().lock().unwrap().get(&h).cloned()
}

/// Parse a JSON array of `["summary","body"]` pairs into sections.
fn parse_sections(v: &serde_json::Value) -> Vec<(String, String)> {
    v.as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let pair = e.as_array()?;
                    Some((pair.get(0)?.as_str()?.to_owned(), pair.get(1)?.as_str()?.to_owned()))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Handle an `rlm` request: `prompt` is a JSON payload — an upload (store a context,
/// reply with its handle) or a query (answer a question over a stored/inline context
/// via `run_rlm`, streamed as one Token frame then Done).
fn handle_rlm(
    req: &InferenceRequest,
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    tmpl: &LlamaChatTemplate,
    stream: &mut UnixStream,
    model_name: &str,
) -> std::io::Result<()> {
    let payload: serde_json::Value = match serde_json::from_str(&req.prompt) {
        Ok(v) => v,
        Err(e) => {
            stream.write_all(&Frame::Error(format!("rlm: malformed payload JSON: {e}")).encode())?;
            return Ok(());
        }
    };

    // UPLOAD: store the context, return its handle.
    if let Some(up) = payload.get("upload") {
        let sections = parse_sections(up);
        if sections.is_empty() {
            stream.write_all(&Frame::Error("rlm: upload had no [summary,body] sections".into()).encode())?;
            return Ok(());
        }
        let h = store_context(ContextStore::from_sections(&sections));
        eprintln!("jaded: [rlm] uploaded context handle={h} ({} sections)", sections.len());
        stream.write_all(&Frame::Json(format!("{{\"handle\":{h}}}")).encode())?;
        stream.write_all(&Frame::Done { tokens_used: 0 }.encode())?;
        return Ok(());
    }

    // QUERY: resolve the context (by handle or inline), then run the RLM.
    let question = payload.get("question").and_then(|q| q.as_str()).unwrap_or("");
    let max_spans = payload.get("max_spans").and_then(|m| m.as_u64()).unwrap_or(4) as usize;
    let store = if let Some(h) = payload.get("handle").and_then(|h| h.as_u64()) {
        match get_context(h) {
            Some(s) => s,
            None => {
                stream.write_all(&Frame::Error(format!("rlm: unknown context handle {h}")).encode())?;
                return Ok(());
            }
        }
    } else if let Some(secs) = payload.get("sections") {
        ContextStore::from_sections(&parse_sections(secs))
    } else {
        stream.write_all(&Frame::Error("rlm: query needs a \"handle\" or inline \"sections\"".into()).encode())?;
        return Ok(());
    };

    stream.write_all(&Frame::Meta { model: model_name.to_owned() }.encode())?;
    match run_rlm(model, ctx, tmpl, question, &store, max_spans.max(1)) {
        Ok((answer, picked)) => {
            eprintln!("jaded: [rlm] query answered (spans {picked:?})");
            stream.write_all(&Frame::Token(answer).encode())?;
            stream.write_all(&Frame::Done { tokens_used: 0 }.encode())?;
        }
        Err(e) => {
            stream.write_all(&Frame::Error(format!("rlm: {e}")).encode())?;
        }
    }
    Ok(())
}

/// End-to-end RLM demo (gated by `JADE_RLM_DEMO`): a needle-in-haystack over a
/// document far larger than any single prompt would want, proving the root can
/// navigate by map, fan out concurrently into the right span, and answer — with the
/// full document held in RAM, never stuffed into one context.
fn selftest_rlm_demo(model: &LlamaModel, ctx: &mut LlamaContext, tmpl: &LlamaChatTemplate) {
    // Build a haystack large enough that the WHOLE document cannot fit in the
    // context window — proving RLM reasons beyond the model's window. Each section
    // body is padded with filler; the needle lives in the "Access Credentials" one.
    let needle = "XQ-42";
    let n_sections: usize = std::env::var("JADE_RLM_DEMO_SECTIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let topics = [
        "Introduction", "Hardware Overview", "Boot Sequence", "Networking",
        "Filesystem Layout", "Process Model", "Logging", "Power Management",
        "Update Channels", "Telemetry", "Scheduling", "Memory Management",
        "Device Drivers", "Cryptography", "Backups", "Glossary",
    ];
    // ~1000 tokens of filler per section: a FEW big sections sum past the window
    // while each section's body — and the short map — still fit one sequence's KV
    // slot (n_ctx / n_seq_max). This is the RLM shape: large content, small map.
    let filler = "This section documents routine operational details with no special \
                  codes or secrets. ".repeat(60);
    let needle_at = n_sections / 2; // put the credentials section in the middle
    let mut sections: Vec<(String, String)> = Vec::new();
    for i in 0..n_sections {
        if i == needle_at {
            sections.push((
                format!("Section {i}: Access Credentials"),
                format!("Section {i}: Access Credentials. For maintenance, operators authenticate \
                         with the device. The secret access code is {needle}. Rotate it quarterly. {filler}"),
            ));
        } else {
            let t = topics[i % topics.len()];
            sections.push((
                format!("Section {i}: {t}"),
                format!("Section {i}: {t}. This section describes {t} in general terms. {filler}"),
            ));
        }
    }
    let store = ContextStore::from_sections(&sections);
    let question = "What is the secret access code?";

    // Report how big the whole document is vs the context window — the point is it
    // does NOT fit, yet RLM answers by selecting only the relevant span.
    let ctx_size = ctx.n_ctx();
    let full_doc: String = store.bodies.join("\n\n");
    let full_tokens = model.str_to_token(&full_doc, AddBos::Always).map(|t| t.len()).unwrap_or(0);
    eprintln!(
        "jaded: [rlm-demo] {} sections, ~{full_tokens} tokens total vs ctx_size={ctx_size} \
         ({}), question: {question:?}",
        store.len(),
        if full_tokens as u32 > ctx_size { "DOES NOT FIT — RLM required" } else { "fits" }
    );
    let start = Instant::now();
    match run_rlm(model, ctx, tmpl, question, &store, 4) {
        Ok((answer, picked)) => {
            let ms = start.elapsed().as_millis();
            eprintln!("jaded: [rlm-demo] picked spans {picked:?}");
            eprintln!("jaded: [rlm-demo] ANSWER: {:?}", answer.trim());
            let ok = answer.contains(needle);
            eprintln!("jaded: [rlm-demo] needle {needle} found in answer: {ok}  ({ms}ms)");
        }
        Err(e) => eprintln!("jaded: [rlm-demo] error: {e}"),
    }
}

/// Latency A/B (gated by `JADE_RLM_COMPARE`): standard single-shot (stuff the WHOLE
/// document into one prompt → one inference) vs RLM (PICK→fan-out→COMBINE), on a
/// SMALL document that fits one sequence's KV slot so both are even possible. Prints
/// the RLM overhead — the cost you pay on small contexts for the machinery that lets
/// large contexts work at all (where standard isn't an option).
fn selftest_rlm_compare(model: &LlamaModel, ctx: &mut LlamaContext, tmpl: &LlamaChatTemplate) {
    let needle = "XQ-42";
    let topics = [
        "Introduction", "Hardware", "Boot", "Networking", "Filesystem", "Processes",
        "Access Credentials", "Logging", "Power", "Updates", "Telemetry", "Glossary",
    ];
    let mut sections: Vec<(String, String)> = Vec::new();
    for (i, t) in topics.iter().enumerate() {
        let body = if *t == "Access Credentials" {
            format!("Section {i}: Access Credentials. The secret access code is {needle}. Rotate it quarterly.")
        } else {
            format!("Section {i}: {t}. Routine details about {t} with no secrets.")
        };
        sections.push((format!("Section {i}: {t}"), body));
    }
    let store = ContextStore::from_sections(&sections);
    let question = "What is the secret access code?";

    let full: String = sections.iter().map(|(s, b)| format!("{s}\n{b}")).collect::<Vec<_>>().join("\n\n");
    let full_toks = model.str_to_token(&full, AddBos::Always).map(|t| t.len()).unwrap_or(0);

    // STANDARD baseline: whole document + question in a single prompt, one inference.
    let std_prompt = format!("Document:\n{full}\n\nQuestion: {question}\nAnswer concisely.");
    let std_sub = match build_sub(model, tmpl, &std_prompt, None, 120) {
        Ok(s) => s,
        Err(e) => { eprintln!("jaded: [compare] build error: {e}"); return; }
    };
    let t = Instant::now();
    let std_ans = match run_concurrent(model, ctx, vec![std_sub]) {
        Ok(mut o) => o.pop().unwrap_or_default(),
        Err(e) => { eprintln!("jaded: [compare] standard error: {e}"); return; }
    };
    let std_ms = t.elapsed().as_millis().max(1);

    // RLM over the same document.
    let t = Instant::now();
    let (rlm_ans, _) = match run_rlm(model, ctx, tmpl, question, &store, 4) {
        Ok(r) => r,
        Err(e) => { eprintln!("jaded: [compare] rlm error: {e}"); return; }
    };
    let rlm_ms = t.elapsed().as_millis().max(1);

    eprintln!("jaded: [compare] doc {} sections (~{full_toks} tokens, fits the window)", sections.len());
    eprintln!("jaded: [compare] STANDARD {std_ms}ms -> {:?}", std_ans.trim());
    eprintln!("jaded: [compare] RLM      {rlm_ms}ms -> {:?}", rlm_ans.trim());
    eprintln!(
        "jaded: [compare] RLM latency vs standard: {:.2}x  (+{}ms absolute)",
        rlm_ms as f64 / std_ms as f64,
        rlm_ms.saturating_sub(std_ms)
    );
}
