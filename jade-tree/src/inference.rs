use encoding_rs;
use io_uring::{opcode, types, IoUring};
use jade_protocol::{Frame, InferenceRequest};
use llama_cpp_2::{
    context::{params::{KvCacheType, LlamaContextParams}, LlamaContext},
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, LlamaChatMessage, LlamaChatTemplate},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::num::NonZeroU32;
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicU64, Ordering};
static GLOBAL_TOKENS: AtomicU64 = AtomicU64::new(0);

const DEFAULT_MODEL_PATH: &str = "/jade/inference/model.gguf";
const DEFAULT_CTX_SIZE: u32 = 32_768;
const DEFAULT_GPU_LAYERS: u32 = 999;
const DEFAULT_CACHE_SIZE: usize = 256;
const DEFAULT_N_BATCH: u32 = 2048;
const DEFAULT_N_UBATCH: u32 = 512;
const DEFAULT_KV_TYPE: &str = "q4_0";
const QUEUE_DEPTH: u32 = 128;

fn parse_kv_type(s: &str) -> KvCacheType {
    match s.to_ascii_lowercase().as_str() {
        "f16"  => KvCacheType::F16,
        "bf16" => KvCacheType::BF16,
        "f32"  => KvCacheType::F32,
        "q8_0" => KvCacheType::Q8_0,
        "q5_1" => KvCacheType::Q5_1,
        "q5_0" => KvCacheType::Q5_0,
        "q4_1" => KvCacheType::Q4_1,
        "q4_0" => KvCacheType::Q4_0,
        other  => {
            eprintln!("jade-tree: unknown KV cache type '{other}', falling back to q8_0");
            KvCacheType::Q8_0
        }
    }
}

// ── Prompt cache ──────────────────────────────────────────────────────────────

struct PromptCache {
    map: HashMap<u64, (String, u64)>,
    order: std::collections::VecDeque<u64>,
    max_entries: usize,
}

impl PromptCache {
    fn new(max_entries: usize) -> Self {
        PromptCache {
            map: HashMap::new(),
            order: std::collections::VecDeque::new(),
            max_entries,
        }
    }

    fn key(req: &InferenceRequest) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        req.prompt.hash(&mut h);
        req.model.hash(&mut h);
        req.max_tokens.hash(&mut h);
        req.grammar.hash(&mut h);
        req.anchor.hash(&mut h);
        req.stop_anchor.hash(&mut h);
        h.finish()
    }

    fn get(&self, key: u64) -> Option<&(String, u64)> {
        self.map.get(&key)
    }

    fn insert(&mut self, key: u64, text: String, tokens_used: u64) {
        if self.max_entries == 0 || self.map.contains_key(&key) {
            return;
        }
        if self.map.len() >= self.max_entries {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
        self.map.insert(key, (text, tokens_used));
        self.order.push_back(key);
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
    eprintln!("JADE OS — jade-tree v{}", env!("CARGO_PKG_VERSION"));

    let model_path = std::env::var("JADE_MODEL_PATH")
        .unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_owned());
    let ctx_size: u32 = std::env::var("JADE_CTX_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_CTX_SIZE);
    let gpu_layers: u32 = std::env::var("JADE_GPU_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_GPU_LAYERS);
    let n_batch: u32 = std::env::var("JADE_N_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N_BATCH);
    let n_ubatch: u32 = std::env::var("JADE_N_UBATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_N_UBATCH);
    let kv_type_k = parse_kv_type(
        &std::env::var("JADE_KV_TYPE_K").unwrap_or_else(|_| DEFAULT_KV_TYPE.to_owned()),
    );
    let kv_type_v = parse_kv_type(
        &std::env::var("JADE_KV_TYPE_V").unwrap_or_else(|_| DEFAULT_KV_TYPE.to_owned()),
    );
    let flash_attn = !matches!(
        std::env::var("JADE_FLASH_ATTN").as_deref(),
        Ok("0") | Ok("off") | Ok("false")
    );

    let v_is_quantized = !matches!(
        kv_type_v,
        KvCacheType::F16 | KvCacheType::BF16 | KvCacheType::F32
    );
    if v_is_quantized && !flash_attn {
        eprintln!(
            "jade-tree: ERROR — quantized V cache (kv_v={kv_type_v:?}) requires flash attention, \
             but JADE_FLASH_ATTN is disabled. Either set JADE_KV_TYPE_V=f16 or remove \
             JADE_FLASH_ATTN=off."
        );
        std::process::exit(1);
    }

    eprintln!("jade-tree: loading model: {model_path}");
    eprintln!(
        "jade-tree: ctx_size={ctx_size} gpu_layers={gpu_layers} \
         n_batch={n_batch} n_ubatch={n_ubatch} \
         kv_k={kv_type_k:?} kv_v={kv_type_v:?} flash_attn={flash_attn}"
    );

    let backend = LlamaBackend::init().unwrap_or_else(|e| {
        eprintln!("jade-tree: llama backend init failed: {e}");
        std::process::exit(1);
    });

    let model_params = LlamaModelParams::default()
        .with_n_gpu_layers(gpu_layers)
        .with_use_mlock(true);
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .unwrap_or_else(|e| {
            eprintln!("jade-tree: failed to load model from {model_path}: {e}");
            std::process::exit(1);
        });

    eprintln!("jade-tree: model ready");

    let model_name = model.meta_val_str("general.name").unwrap_or_else(|_| {
        std::path::Path::new(&model_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_owned()
    });
    eprintln!("jade-tree: model name: {model_name}");

    let chat_template = model.chat_template(None).unwrap_or_else(|e| {
        eprintln!("jade-tree: WARNING — could not read chat template from GGUF: {e}");
        eprintln!("jade-tree: falling back to ChatML");
        LlamaChatTemplate::new("chatml").expect("chatml fallback template")
    });
    eprintln!("jade-tree: chat template loaded");

    let flash_attn_policy = if flash_attn {
        llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED
    } else {
        llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED
    };

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(ctx_size).unwrap()))
        .with_n_batch(n_batch)
        .with_n_ubatch(n_ubatch)
        .with_type_k(kv_type_k)
        .with_type_v(kv_type_v)
        .with_flash_attention_policy(flash_attn_policy);
    let mut ctx = model.new_context(&backend, ctx_params).unwrap_or_else(|e| {
        eprintln!("jade-tree: context init failed: {e}");
        std::process::exit(1);
    });
    eprintln!("jade-tree: context ready (ctx_size={ctx_size})");

    let cache_size: usize = std::env::var("JADE_CACHE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_CACHE_SIZE);
    let mut cache = PromptCache::new(cache_size);
    eprintln!("jade-tree: prompt cache size={cache_size}");

    std::fs::create_dir_all(std::path::Path::new(&sock_path).parent().unwrap())
        .expect("create socket dir");
    let _ = std::fs::remove_file(&sock_path);
    let listener = UnixListener::bind(&sock_path).unwrap_or_else(|e| {
        eprintln!("jade-tree: could not bind {sock_path}: {e}");
        std::process::exit(1);
    });

    // Tracks the token sequence currently resident in the KV cache so we can
    // reuse the shared prefix across consecutive requests instead of re-prefilling
    // the entire conversation history on every turn.
    let mut kv_tokens: Vec<LlamaToken> = Vec::new();

    eprintln!("jade-tree: listening on {sock_path}");
    listen(&listener, &model, &mut ctx, &mut cache, &chat_template, &mut kv_tokens, &model_name);
}

fn listen(listener: &UnixListener, model: &LlamaModel, ctx: &mut LlamaContext, cache: &mut PromptCache, tmpl: &LlamaChatTemplate, kv_tokens: &mut Vec<LlamaToken>, model_name: &str) {
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(e) = handle_connection(&mut stream, model, ctx, cache, tmpl, kv_tokens, model_name) {
                    eprintln!("jade-tree: connection error: {e}");
                }
            }
            Err(e) => eprintln!("jade-tree: accept error: {e}"),
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
                        eprintln!(
                            "jade-tree: connection closed after {request_count} requests"
                        );
                        return Ok(());
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                Err(e) => {
                    eprintln!("jade-tree: malformed request: {e}");
                    return Ok(());
                }
            }
        };

        request_count += 1;

        // TODO(trust): daemon-side enforcement hook — currently the runtime is
        // the sole gatekeeper (refuses tainted strings at sh.exec/http.get/fs.read).
        // jade-tree only records the trust level for observability.

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
                    stream.write_all(
                        &Frame::Error(format!("tokenize failed: {e}")).encode(),
                    )?;
                    continue;
                }
            };
            stream.write_all(
                &Frame::Done { tokens_used: tokens.len() as u64 }.encode(),
            )?;
            continue;
        }

        eprintln!(
            "jade-tree: request — model={} max_tokens={} prompt_len={} trust={}",
            req.model, req.max_tokens, req.prompt.len(), req.trust
        );

        stream.write_all(&Frame::Meta { model: model_name.to_owned() }.encode())?;

        let mut writer = IoUringWriter::new(stream.as_raw_fd())?;

        let cache_key = PromptCache::key(&req);
        if let Some((cached_text, tokens_used)) = cache.get(cache_key) {
            eprintln!("jade-tree: cache hit (key={cache_key:016x})");
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
                    if let Some(pos) = response_buf.find(stop) {
                        response_buf.truncate(pos);
                    }
                }
                GLOBAL_TOKENS.fetch_add(tokens_used, Ordering::Relaxed);
                cache.insert(cache_key, response_buf, tokens_used);
                stream.write_all(&Frame::Done { tokens_used }.encode())?;
            }
            Err(msg) => {
                eprintln!("jade-tree: inference error: {msg}");
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
    if let Some((system, messages)) = crate::prompt::extract_jade_protocol(prompt_str) {
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
        eprintln!("jade-tree: kv prefix reuse {prefix_len}/{} tokens ({} new)",
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

    // Free sampler used for all unconstrained tokens (and all tokens once the anchor is dropped).
    let mut free_sampler = LlamaSampler::chain_simple(vec![
        LlamaSampler::temp(0.6),
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::dist(0),
    ]);

    // Anchor sampler: grammar forces the first token ({, [, true/false, etc.), then is
    // dropped entirely — zero masking overhead after token 1 for complex types.
    // For primitives (int/float/bool) the grammar runs for the full output; anchor_sampler
    // is never taken to None and free_sampler is never used.
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

    // String anchor: grammar enforcement is deferred until the model emits req.anchor.
    // anchor_triggered = true from the start when no string anchor is configured.
    let anchor_str = req.anchor.as_deref();
    let mut anchor_triggered = anchor_str.is_none();
    // Stop anchor: ends grammar enforcement when this string appears; inference continues.
    let stop_anchor_str = req.stop_anchor.as_deref();
    let mut emitted_so_far = String::new();
    // Tokens buffered while we wait to confirm they aren't (part of) the stop anchor.
    let mut stop_pending = String::new();
    // Set to true once the stop anchor has been detected and grammar retired.
    let mut stop_anchor_seen = false;

    // Primitives (int/float/bool): grammar runs for the full output.
    // Complex types (struct/dict/array): grammar enforces only the first token
    // ({, [) then is retired — zero masking overhead after token 1.
    // Crucially: we keep anchor_sampler allocated until after the loop to avoid
    // crashing llama.cpp by freeing a grammar sampler while the context is active.
    //
    // When a string anchor is set (e.g. "<tool>"), the grammar is NEVER a full
    // primitive grammar — it enforces the anchored suffix (JSON tool call) and the
    // depth tracker retires grammar exactly when the JSON closes, letting the model
    // emit the stop_anchor (e.g. "</tool>") freely and then EOG.
    // Without this guard, a GBNF containing "[0-9]" (e.g. the num rule in the tool
    // JSON grammar) would falsely disable depth tracking.
    let anchor_is_full_grammar = anchor_str.is_none() && req.grammar.as_deref().map(|g| {
        g.contains("[0-9]") || g.contains("\"true\"")
    }).unwrap_or(false);

    let max_new = (ctx_size as usize).saturating_sub(n_prompt).min(req.max_tokens as usize);
    let mut n_gen = 0usize;
    let mut gen_batch = LlamaBatch::new(1, 1);
    // anchor_active: whether the grammar sampler should still be used for sampling.
    // For primitives: stays true forever. For complex types: false after first token.
    let mut anchor_active = anchor_sampler.is_some();

    // Depth tracking for anchor-span grammars: retires grammar once the top-level
    // JSON object/array closes, so the model continues freely (emitting stop_anchor,
    // then EOG) without grammar enforcement running on post-JSON tokens.
    // Only active when a start anchor is set — without one, grammar enforcement
    // covers the full response and must not be retired by depth tracking.
    let mut do_depth_track = anchor_sampler.is_some() && !anchor_is_full_grammar && anchor_str.is_some();
    let mut json_depth = 0usize;
    let mut json_in_string = false;
    let mut json_escape = false;

    'gen_loop: loop {
        // LlamaSampler::sample() already accepts the token internally (llama_sampler_sample
        // calls llama_sampler_accept at the end). Do NOT call sampler.accept() again here —
        // double-accepting empties the grammar stacks and causes an abort on the next sample.
        let token = if anchor_active && anchor_triggered {
            let gsamp = anchor_sampler.as_mut().unwrap();
            let t = gsamp.sample(&ctx, -1);
            if !anchor_is_full_grammar && anchor_str.is_some() {
                // Anchor-span mode: retire grammar after first token; depth tracker
                // monitors the remaining JSON. Without a start anchor, grammar stays
                // active through the full response (until stop_anchor or EOG).
                anchor_active = false;
            }
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

        // Accumulate output for anchor detection (both start and stop).
        emitted_so_far.push_str(&piece);

        // Detect start anchor to enable grammar enforcement.
        if !anchor_triggered {
            if let Some(a) = anchor_str {
                if emitted_so_far.contains(a) {
                    anchor_triggered = true;
                    // Guard: if this BPE token also started the JSON body (e.g. the token
                    // that completed "<tool>" already contained "{"), retire the grammar
                    // sampler now so it doesn't force a second "{" on the next token,
                    // which would produce invalid JSON like `<tool>{{...}}`.
                    if do_depth_track && anchor_active {
                        let a_end = emitted_so_far.find(a).map(|p| p + a.len()).unwrap_or(emitted_so_far.len());
                        if emitted_so_far[a_end..].contains('{') || emitted_so_far[a_end..].contains('[') {
                            anchor_active = false;
                        }
                    }
                }
            }
        }

        // Prefix-aware stop anchor buffering: accumulate into stop_pending and only
        // call on_token for content confirmed not to be (part of) the stop anchor.
        // Prevents split-token leaks where e.g. "</" and "tool>" arrive separately.
        // Once the stop anchor fires, grammar is retired and we emit freely thereafter.
        if let Some(stop) = stop_anchor_str {
            if stop_anchor_seen {
                // Grammar already retired — emit immediately, no need to buffer.
                on_token(&piece);
            } else {
                stop_pending.push_str(&piece);
                if let Some(stop_pos) = stop_pending.find(stop) {
                    // Stop anchor complete — emit everything before it, retire grammar.
                    if stop_pos > 0 {
                        on_token(&stop_pending[..stop_pos]);
                    }
                    stop_pending.clear();
                    stop_anchor_seen = true;
                    anchor_active = false;
                    do_depth_track = false;
                    // No break — inference continues freely after the stop anchor.
                } else {
                    // Find the longest suffix of stop_pending that is a strict prefix of
                    // stop — keep it buffered; emit the confirmed-safe leading portion.
                    let mut keep = 0usize;
                    for (start, _) in stop_pending.char_indices() {
                        let suffix = &stop_pending[start..];
                        if stop.starts_with(suffix) {
                            keep = suffix.len();
                            break;
                        }
                    }
                    let emit_len = stop_pending.len() - keep;
                    if emit_len > 0 {
                        on_token(&stop_pending[..emit_len]);
                        stop_pending = stop_pending[emit_len..].to_string();
                    }
                }
            }
        } else {
            on_token(&piece);
        }

        // Depth tracking: break as soon as the top-level JSON object/array closes.
        // Only track after anchor_triggered — preamble braces must not count.
        if do_depth_track && anchor_triggered {
            for ch in piece.chars() {
                if json_escape { json_escape = false; continue; }
                match ch {
                    '\\' if json_in_string => { json_escape = true; }
                    '"' => { json_in_string = !json_in_string; }
                    '{' | '[' if !json_in_string => { json_depth += 1; }
                    '}' | ']' if !json_in_string => {
                        if json_depth > 0 { json_depth -= 1; }
                        if json_depth == 0 {
                            // JSON span closed — retire grammar enforcement.
                            // Inference continues; model will naturally emit stop_anchor
                            // (e.g. </tool>) then EOG.
                            anchor_active = false;
                            do_depth_track = false;
                        }
                    }
                    _ => {}
                }
            }
        }

        kv_tokens.push(token);

        gen_batch.clear();
        let gen_pos = (n_prompt + n_gen - 1) as i32;
        gen_batch.add(token, gen_pos, &[0], true)
            .map_err(|e| format!("batch add failed: {e}"))?;
        ctx.decode(&mut gen_batch).map_err(|e| format!("decode step failed: {e}"))?;
    }

    // Flush anything buffered for stop-anchor prefix detection that never completed
    // (EOG, max_new, or depth-tracker break — the buffered content is real output).
    if !stop_pending.is_empty() {
        on_token(&stop_pending);
    }

    Ok((n_prompt + n_gen) as u64)
}

