//! The llama.cpp inference backend — the half of this crate that touches the GPU.
//!
//! [`Dovata`] implements [`Provider`], and the crate ships it as a `cdylib` that the
//! calling program `dlopen`s once and then calls in-process. There is no host and no
//! socket: this module owns the model and context lifecycle and dispatches requests
//! straight onto them, and the caller owns everything around that. The decode engines
//! live in focused submodules:
//!
//! * [`kv`]          — KV-cache optimizations (cache-type mapping, prefix reuse).
//! * [`standard`]    — single-shot inference (one sequence, constrained decode).
//! * `concurrency`   — continuous-batching engine over the pure `crate::sched`
//!                     (feature `batch`).
//! * `compose`       — RLM engine: tensor-level KV-cache composition (feature `compose`).
//! * `rlm`           — RLM request dispatch onto `compose` (feature `compose`).
//!
//! `kv` and `standard` are unconditional: every request path needs them. The other
//! three are the llama-bound halves of capabilities that can be compiled out whole.
//! When one is off, `infer` refuses the requests it would have served — see the
//! degradation arms below — rather than falling through to a path that answers the
//! wrong question.
//!
//! The platform-independent logic (anchors/stop/grammar, config, response cache,
//! scheduler) lives in the `dovata` library and is unit-tested on any platform;
//! everything here is llama.cpp-bound and Linux-only.
//!
//! ## Linux-only / verification note
//!
//! This module links vendored llama.cpp and does not build on macOS, so it is behind
//! `#[cfg(target_os = "linux")]` and nothing here is exercised by a Mac `cargo test`.
//! It reached its current shape by moving the daemon's old `inference*.rs` here and
//! swapping the socket/io_uring output boundary for a [`FrameSink`]. Points to
//! re-check on a Linux build: the `unsafe impl Send/Sync for Dovata` justification,
//! the `Box::leak` `'static` model/backend, and the llama-cpp-2 API surface.

mod kv;
mod standard;
// The continuous-batching driver is retained as general infrastructure for concurrent
// and mass prompting, but nothing reaches it any more. Its only caller was the
// env-gated selftest module, which was deleted once dropping the daemon left it with
// no startup phase to hang off. Wire it to an explicit entry point before relying on it.
#[cfg(feature = "batch")]
#[allow(dead_code)]
mod concurrency;
#[cfg(feature = "compose")]
mod rlm;
#[cfg(feature = "compose")]
mod compose;

use std::num::NonZeroU32;
use std::sync::Mutex;

use ovata_infer_protocol::{Frame, FrameSink, InferenceRequest, Provider};

use crate::cache::PromptCache;
use crate::config::{Config, KvType};
use crate::profile::ChatTemplate;
use llama_cpp_2::{
    context::{params::LlamaContextParams, LlamaContext},
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    token::LlamaToken,
};

// ── Over-context routing constants ─────────────────────────────────────────────
/// Tokens of generation room a prompt must leave inside the native window before the
/// request is treated as over-context. Unconditional: it decides *whether* to route to
/// compose, which is a question worth answering even when compose is not compiled in.
const NATIVE_GEN_HEADROOM: u32 = 512;
/// Fallback RLM frontier = this × the native window when no profile/env limit is set.
#[cfg(feature = "compose")]
const RLM_FRONTIER_WINDOW_MULT: u32 = 8;
/// Defaults for the over-context blended pipeline (overridable via env).
#[cfg(feature = "compose")]
const RLM_DEFAULT_N: usize = 12;
#[cfg(feature = "compose")]
const RLM_DEFAULT_K: usize = 4;

// ── The provider ───────────────────────────────────────────────────────────────

/// The mutable engine state, guarded by a `Mutex` so the `Send + Sync` `Provider`
/// can serialize the (single-threaded) llama context across requests.
struct Engine {
    ctx: LlamaContext<'static>,
    cache: PromptCache,
    /// Token sequence resident in the KV cache, for cross-request prefix reuse.
    kv_tokens: Vec<LlamaToken>,
}

/// The local llama.cpp provider.
pub struct Dovata {
    /// Leaked to `'static` (loaded once, lives for the process) so the context can
    /// borrow it while living in the same struct.
    model: &'static LlamaModel,
    engine: Mutex<Engine>,
    /// Profile-driven chat template for prompt formatting.
    chat_tmpl: ChatTemplate,
    /// GGUF model name — used for profile/limit selection, never sent on the wire.
    model_name: String,
    /// RLM efficient-frontier limit (max corpus tokens) for the over-context path.
    /// Only the compose path reads it, so without that capability the field would be
    /// written at load and never read.
    #[cfg(feature = "compose")]
    rlm_max: u32,
}

// SAFETY: every access to the raw llama context/model goes through `engine`'s Mutex,
// and the daemon calls `infer` from one thread at a time. The leaked `LlamaModel` is
// only read (llama models are safe for concurrent read). No llama pointer is touched
// concurrently. (Verify on the Linux build — llama-cpp-2 does not mark these Send.)
unsafe impl Send for Dovata {}
unsafe impl Sync for Dovata {}

impl Provider for Dovata {
    const NAME: &'static str = "dovata";

    fn from_config(_config: &[u8]) -> Result<Self, String> {
        // Config comes from the JADE_* environment; the passed config bytes are
        // reserved for future per-load overrides.
        let cfg = Config::from_env();
        cfg.validate().map_err(|e| e.to_string())?;

        // Vet the model file before llama.cpp spends minutes on it, and pick the
        // profile. This runs here because `from_config` is the only entry point — the
        // library is loaded and constructed in one step by whoever dlopens it, so
        // there is no earlier place to stand. Returning `Err` fails the load, which
        // the loader surfaces as a null handle rather than a half-built provider.
        preflight(&cfg)?;
        for (label, var) in [("K", "JADE_KV_TYPE_K"), ("V", "JADE_KV_TYPE_V")] {
            if let Ok(s) = std::env::var(var) {
                if !KvType::is_known(&s) {
                    eprintln!("dovata: unknown KV cache type '{s}' for {label}, falling back to q8_0");
                }
            }
        }

        eprintln!("dovata: loading model: {}", cfg.model_path);

        // Leaked to 'static: the daemon loads one provider for the process lifetime,
        // and the context must borrow the backend + model from inside `Dovata`.
        let backend: &'static LlamaBackend = Box::leak(Box::new(
            LlamaBackend::init().map_err(|e| format!("llama backend init failed: {e}"))?,
        ));

        // Pinned because a buft override stores a self-referential pointer into the
        // params (see LlamaModelParams::add_cpu_buft_override).
        let mut model_params = Box::pin(
            LlamaModelParams::default()
                .with_n_gpu_layers(cfg.gpu_layers)
                .with_use_mlock(true),
        );
        if cfg.moe.enabled {
            model_params.as_mut().add_cpu_moe_override();
            eprintln!("dovata: MoE driver — all expert layers -> host RAM (page source)");
        }
        // CRITICAL: add_cpu_buft_override stores a raw pointer into this CString's
        // bytes; it MUST outlive load_from_file, so bind it in this scope.
        let _cpu_moe_pattern: Option<std::ffi::CString> = match cfg.cpu_moe_pattern() {
            Some(pattern) => {
                let cpattern = std::ffi::CString::new(pattern)
                    .map_err(|_| "cpu-moe pattern has an interior NUL".to_string())?;
                model_params.as_mut().add_cpu_buft_override(&cpattern);
                eprintln!(
                    "dovata: baseline cpu-moe offload — first {} layers' experts -> CPU",
                    cfg.cpu_moe_layers
                );
                Some(cpattern)
            }
            None => None,
        };

        let model_owned = LlamaModel::load_from_file(backend, &cfg.model_path, &model_params)
            .map_err(|e| format!("failed to load model from {}: {e}", cfg.model_path))?;
        let model: &'static LlamaModel = Box::leak(Box::new(model_owned));
        eprintln!("dovata: model ready");

        let model_name = model.meta_val_str("general.name").unwrap_or_else(|_| {
            std::path::Path::new(&cfg.model_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_owned()
        });
        eprintln!("dovata: model name: {model_name}");

        // Activate the index-driven MoE driver (expert tensors are host-resident now).
        //
        // Every failure here fails the load. That is a change from the original, which
        // logged and continued on a stock path: a run with the driver inactive produces
        // identical output and uses roughly three times the VRAM, so the symptom is an
        // OOM much later, in a place that says nothing about expert offload. The config
        // asked for offload; not getting it is an error, not a fallback.
        #[cfg(feature = "moe")]
        if cfg.moe.enabled {
            let order_path = cfg.moe.order_path.as_deref().unwrap_or_default();
            let order = crate::moe::ExpertOrder::load(order_path)
                .map_err(|e| format!("MoE expert order load failed ({order_path}): {e}"))?;
            crate::moe::ffi::init(&cfg.moe, &order)
                .map_err(|e| format!("MoE driver activation failed: {e}"))?;
            eprintln!(
                "dovata: MoE driver ACTIVE — pin={} cache={} ({}/{} resident/layer, {} layers)",
                cfg.moe.pin,
                cfg.moe.cache,
                cfg.moe.pin + cfg.moe.cache,
                order.n_experts,
                order.n_layers
            );
        }
        // Without the `moe` capability there is nothing to activate — and nothing to
        // check for here, because `cfg.validate()` above already refused the load with
        // a message naming the missing capability. That rule lives in `config` so it is
        // testable without a GPU.

        let flash_attn_policy = if cfg.flash_attn {
            llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED
        } else {
            llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED
        };
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(cfg.ctx_size).ok_or("ctx_size must be non-zero")?))
            .with_n_batch(cfg.n_batch)
            .with_n_ubatch(cfg.n_ubatch)
            .with_n_seq_max(cfg.n_seq_max)
            .with_type_k(kv::kv_cache_type(cfg.kv_type_k))
            .with_type_v(kv::kv_cache_type(cfg.kv_type_v))
            .with_flash_attention_policy(flash_attn_policy);
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
        let ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| format!("context init failed: {e}"))?;
        eprintln!("dovata: context ready (ctx_size={})", cfg.ctx_size);

        #[cfg(feature = "compose")]
        let rlm_max = {
            let m = rlm_max_context(&model_name, cfg.ctx_size);
            eprintln!("dovata: rlm frontier (max corpus tokens) = {m}");
            m
        };
        let chat_tmpl = resolve_chat_template(&model_name);
        let cache = PromptCache::new(cfg.cache_size);
        eprintln!("dovata: prompt cache size={}", cfg.cache_size);

        Ok(Dovata {
            model,
            engine: Mutex::new(Engine { ctx, cache, kv_tokens: Vec::new() }),
            chat_tmpl,
            model_name,
            #[cfg(feature = "compose")]
            rlm_max,
        })
    }

    fn infer(&self, req: &InferenceRequest, sink: &mut dyn FrameSink) {
        let mut eng = match self.engine.lock() {
            Ok(g) => g,
            Err(_) => {
                sink.emit(Frame::Error("dovata: engine lock poisoned".into()));
                return;
            }
        };
        let Engine { ctx, cache, kv_tokens } = &mut *eng;
        let model = self.model;
        let tmpl = &self.chat_tmpl;

        // count_only: tokenize the formatted prompt and report the count.
        if req.count_only {
            match build_formatted_prompt(&req.prompt, tmpl)
                .and_then(|f| model.str_to_token(&f, AddBos::Always).map_err(|e| format!("tokenize failed: {e}")))
            {
                Ok(tokens) => sink.emit(Frame::Done { tokens_used: tokens.len() as u64 }),
                Err(e) => sink.emit(Frame::Error(e)),
            }
            return;
        }

        // RLM mode: explicit context-handle / pre-chunked decomposition.
        if req.rlm {
            #[cfg(feature = "compose")]
            {
                rlm::handle_rlm(req, model, ctx, sink);
            }
            #[cfg(not(feature = "compose"))]
            {
                sink.emit(Frame::Error(
                    "this request asks for the rlm path, but the engine was built \
                     without the `compose` capability. Rebuild with --features compose."
                        .into(),
                ));
            }
            return;
        }

        // Over-context routing: a prompt that won't fit the native window with room to
        // generate is answered by the blended KV-composition engine.
        match build_formatted_prompt(&req.prompt, tmpl)
            .and_then(|f| model.str_to_token(&f, AddBos::Always).map_err(|e| format!("tokenize failed: {e}")))
        {
            Ok(toks) if toks.len() as u32 + NATIVE_GEN_HEADROOM > ctx.n_ctx() => {
                #[cfg(feature = "compose")]
                {
                    eprintln!(
                        "dovata: over-context ({} prompt tokens, window {}) → blended RLM compose",
                        toks.len(),
                        ctx.n_ctx()
                    );
                    answer_over_context(req, model, ctx, sink, self.rlm_max);
                }
                // Falling through to the native path here would send llama.cpp a prompt
                // longer than its window, and the answer would come back derived from
                // whatever fragment survived — right-looking and wrong. Refuse instead.
                #[cfg(not(feature = "compose"))]
                {
                    sink.emit(Frame::Error(format!(
                        "prompt is {} tokens against a {} token window, and this engine \
                         was built without the `compose` capability that answers \
                         over-context requests. Shorten the prompt, raise JADE_CTX_SIZE, \
                         or rebuild with --features compose.",
                        toks.len(),
                        ctx.n_ctx()
                    )));
                }
                return;
            }
            Ok(_) => {} // fits → native path below
            Err(e) => {
                sink.emit(Frame::Error(e));
                return;
            }
        }

        eprintln!(
            "dovata: request — model={} max_tokens={} prompt_len={} trust={}",
            req.model, req.max_tokens, req.prompt.len(), req.trust
        );

        // Response cache: identical request → replay stored text, skip the model.
        let cache_key = PromptCache::key(req);
        if let Some((cached_text, tokens_used)) = cache.get(cache_key) {
            eprintln!("dovata: cache hit (key={cache_key:016x})");
            let tokens_used = *tokens_used;
            sink.emit_text(cached_text);
            sink.emit(Frame::Done { tokens_used });
            return;
        }

        // Cache miss: run the model. The closure fires once per generated token — we
        // both stream it and accumulate it so the full response can be cached.
        let mut response_buf = String::new();
        let result = standard::infer_streaming(req, model, ctx, tmpl, kv_tokens, |token| {
            response_buf.push_str(token);
            sink.emit(Frame::Token(token.to_owned()));
        });

        match result {
            Ok(tokens_used) => {
                if !req.keep_anchors {
                    if let Some(stop) = req.stop_anchor.as_deref() {
                        crate::prompt::strip_stop_anchor(&mut response_buf, stop);
                    }
                }
                cache.insert(cache_key, response_buf, tokens_used);
                sink.emit(Frame::Done { tokens_used });
            }
            Err(msg) => {
                eprintln!("dovata: inference error: {msg}");
                sink.emit(Frame::Error(msg));
            }
        }
    }
}

/// Answer an over-context request with the blended KV-composition engine. Emits
/// `Token`(answer) → `Done`, or an `Error` frame. (The `Meta` frame is the host's.)
#[cfg(feature = "compose")]
fn answer_over_context(
    req: &InferenceRequest,
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    sink: &mut dyn FrameSink,
    rlm_max: u32,
) {
    let input = crate::route::derive_rlm_input(&req.prompt, crate::route::DEFAULT_CHUNK_CHARS);
    let store = match compose::BlockStore::from_texts(model, &input.corpus) {
        Ok(s) => s,
        Err(e) => {
            sink.emit(Frame::Error(format!("over-context chunking failed: {e}")));
            return;
        }
    };
    let corpus_tokens: usize = store.blocks.iter().map(|b| b.ids.len()).sum();

    if corpus_tokens as u32 > rlm_max {
        let msg = format!(
            "context corpus ({corpus_tokens} tokens) exceeds this model's RLM limit \
             ({rlm_max} tokens); reduce the context (see the model profile's limits.rlm_max_context)"
        );
        eprintln!("dovata: [over-context] REJECT — {msg}");
        sink.emit(Frame::Error(msg));
        return;
    }

    let n_cand = std::env::var("JADE_RLM_N").ok().and_then(|v| v.parse().ok()).unwrap_or(RLM_DEFAULT_N);
    let k = std::env::var("JADE_RLM_K").ok().and_then(|v| v.parse().ok()).unwrap_or(RLM_DEFAULT_K);

    let mut kv = compose::KvBlockCache::new(0);
    match compose::blended_compose_and_answer(model, ctx, &store, &mut kv, &input.question, n_cand, k.max(1)) {
        Ok((answer, picked, _scored)) => {
            eprintln!(
                "dovata: [over-context] {} blocks ({corpus_tokens} tok) → composed {picked:?}",
                store.blocks.len()
            );
            sink.emit_text(&answer);
            sink.emit(Frame::Done { tokens_used: 0 });
        }
        Err(e) => sink.emit(Frame::Error(format!("over-context compose: {e}"))),
    }
}

/// Render a request prompt into the model's chat format. Shared by the standard
/// path, `count_only`, and the RLM sub-call builder. The real work is the pure,
/// tested `crate::prompt::format_prompt`.
pub(crate) fn build_formatted_prompt(prompt_str: &str, tmpl: &ChatTemplate) -> Result<String, String> {
    Ok(crate::prompt::format_prompt(prompt_str, tmpl))
}

/// Read the model's GGUF header, pick its profile, and publish the choice — or refuse
/// the load with a message that names the fix.
///
/// This moved out of the daemon's host when the socket went away. It is the same policy
/// it always was: everything a GGUF can tell us is decided before llama.cpp touches the
/// file, so a bad path, an unusable file, or a model no profile matches costs a second
/// instead of a full load. `JADE_ALLOW_UNPROFILED=1` downgrades the no-profile refusal
/// to a warning; see `crate::preflight`.
fn preflight(cfg: &Config) -> Result<(), String> {
    let allow_unprofiled = std::env::var("JADE_ALLOW_UNPROFILED")
        .map(|v| matches!(v.trim(), "1" | "on" | "true"))
        .unwrap_or(false);

    let pf = crate::preflight::Preflight::read(
        &cfg.model_path,
        &crate::profile::Registry::builtin(),
        allow_unprofiled,
    )
    .map_err(|e| format!("model preflight failed\n  {e}"))?;

    for line in pf.summary() {
        eprintln!("{line}");
    }
    if pf.ctx_exceeds_native(cfg.ctx_size) {
        eprintln!(
            "dovata: WARNING JADE_CTX_SIZE={} exceeds the model's native window of {}",
            cfg.ctx_size,
            pf.context_length.unwrap_or(0)
        );
    }
    crate::preflight::publish(&pf);
    Ok(())
}

/// The profile the engine is serving.
///
/// The startup preflight already chose one — from the same GGUF, before the load, and
/// keyed on architecture as well as name — so read its decision rather than making a
/// second one that could differ. The name-glob path stays as the fallback for the cases
/// where the preflight never ran: unit tests, and the `JADE_*` self-test hijacks that
/// construct an engine directly without going through the host.
fn active_profile(model_name: &str) -> Option<&'static crate::profile::ModelProfile> {
    crate::preflight::selected().or_else(|| {
        // Leaked deliberately: the builtin registry is process-lifetime data, and this
        // fallback runs at most once per process.
        let reg: &'static crate::profile::Registry = Box::leak(Box::new(crate::profile::Registry::builtin()));
        reg.select(model_name)
    })
}

/// The chat template for the active model: the matching profile's `[chat_template]`
/// block, else the ChatML default.
fn resolve_chat_template(model_name: &str) -> ChatTemplate {
    active_profile(model_name).map(|p| p.chat_template.clone()).unwrap_or_default()
}

/// The RLM efficient-frontier limit (max corpus tokens) for the active model.
#[cfg(feature = "compose")]
fn rlm_max_context(model_name: &str, ctx_size: u32) -> u32 {
    if let Some(n) = std::env::var("JADE_RLM_MAX_CONTEXT").ok().and_then(|v| v.trim().parse::<u32>().ok()) {
        if n > 0 {
            return n;
        }
    }
    if let Some(p) = active_profile(model_name) {
        if let Some(m) = p.limits.rlm_max_context() {
            return m;
        }
    }
    ctx_size.saturating_mul(RLM_FRONTIER_WINDOW_MULT)
}

