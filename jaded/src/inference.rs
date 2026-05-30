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
use std::sync::atomic::{AtomicU64, Ordering};

static GLOBAL_TOKENS: AtomicU64 = AtomicU64::new(0);

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
        if self.in_flight >= QUEUE_DEPTH as usize {
            self.reap(1)?;
        }
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
        self.ring.submit_and_wait(min)?;
        for cqe in self.ring.completion() {
            if cqe.result() < 0 {
                return Err(std::io::Error::from_raw_os_error(-cqe.result()));
            }
            self.in_flight -= 1;
        }
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.in_flight > 0 {
            self.reap(self.in_flight)?;
        }
        self.buffers.clear();
        Ok(())
    }
}

pub fn run(sock_path: String) {
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
        .with_type_k(kv_cache_type(cfg.kv_type_k))
        .with_type_v(kv_cache_type(cfg.kv_type_v))
        .with_flash_attention_policy(flash_attn_policy);
    let mut ctx = model.new_context(&backend, ctx_params).unwrap_or_else(|e| {
        eprintln!("jaded: context init failed: {e}");
        std::process::exit(1);
    });
    eprintln!("jaded: context ready (ctx_size={})", cfg.ctx_size);

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
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];
    let mut request_count: u64 = 0;

    loop {
        let req = loop {
            match InferenceRequest::decode(&buf) {
                Ok(req) => {
                    let payload_len =
                        u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
                    buf.drain(..4 + payload_len);
                    break req;
                }
                Err(jade_protocol::RequestDecodeError::Incomplete) => {
                    let n = stream.read(&mut tmp)?;
                    if n == 0 {
                        eprintln!("jaded: connection closed after {request_count} requests");
                        return Ok(());
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                Err(e) => {
                    eprintln!("jaded: malformed request: {e}");
                    return Ok(());
                }
            }
        };

        request_count += 1;

        // TODO(trust): daemon-side enforcement hook — currently the runtime is
        // the sole gatekeeper (refuses tainted strings at sh.exec/http.get/fs.read).
        // jaded only records the trust level for observability.

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

        eprintln!(
            "jaded: request — model={} max_tokens={} prompt_len={} trust={}",
            req.model, req.max_tokens, req.prompt.len(), req.trust
        );

        stream.write_all(&Frame::Meta { model: model_name.to_owned() }.encode())?;

        let mut writer = IoUringWriter::new(stream.as_raw_fd())?;

        let cache_key = PromptCache::key(&req);
        if let Some((cached_text, tokens_used)) = cache.get(cache_key) {
            eprintln!("jaded: cache hit (key={cache_key:016x})");
            writer.write(Frame::Token(cached_text.clone()).encode())?;
            writer.flush()?;
            GLOBAL_TOKENS.fetch_add(*tokens_used, Ordering::Relaxed);
            stream.write_all(&Frame::Done { tokens_used: *tokens_used }.encode())?;
            continue;
        }

        let mut response_buf = String::new();
        let result = infer_streaming(&req, model, ctx, tmpl, kv_tokens, |token| {
            response_buf.push_str(token);
            writer.write(Frame::Token(token.to_owned()).encode()).ok();
        });

        writer.flush()?;

        match result {
            Ok(tokens_used) => {
                if let Some(stop) = req.stop_anchor.as_deref() {
                    jaded::prompt::strip_stop_anchor(&mut response_buf, stop);
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
    let n_batch = ctx.n_batch() as usize;
    let last = n_prompt - 1;
    let mut pos = prefix_len;
    while pos < n_prompt {
        let chunk_end = (pos + n_batch).min(n_prompt);
        let mut batch = LlamaBatch::new(chunk_end - pos, 1);
        for i in pos..chunk_end {
            batch.add(tokens[i], i as i32, &[0], i == last)
                .map_err(|e| format!("batch add failed: {e}"))?;
        }
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

    let max_new = (ctx_size as usize).saturating_sub(n_prompt).min(req.max_tokens as usize);
    let mut n_gen = 0usize;
    let mut gen_batch = LlamaBatch::new(1, 1);

    loop {
        // LlamaSampler::sample() already accepts the token internally
        // (llama_sampler_sample calls llama_sampler_accept at the end). Do NOT
        // call sampler.accept() again — double-accepting empties the grammar
        // stacks and causes an abort on the next sample.
        let token = if control.use_grammar() {
            let gsamp = anchor_sampler.as_mut().unwrap();
            let t = gsamp.sample(&ctx, -1);
            control.notice_grammar_sampled();
            t
        } else {
            free_sampler.sample(&ctx, -1)
        };

        if model.is_eog_token(token) || n_gen >= max_new {
            break;
        }

        let piece = model.token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| format!("token to str failed: {e}"))?;
        n_gen += 1;

        if let Some(text) = control.on_piece(&piece) {
            on_token(&text);
        }

        kv_tokens.push(token);

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
