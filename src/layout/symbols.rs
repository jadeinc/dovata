//! The closed set of names a layout's expressions may use, and where each one comes from.
//!
//! ## Why the set is closed
//!
//! A layout may say `n_embd_head_k * n_head` or `eps = "f_norm_rms_eps"`, but it can
//! never say which GGUF key that is, nor give a number in its place. Every hparam is
//! read here, by reference, from the model's own header, so the same layout serves every
//! model of its architecture, and a model with an unusual epsilon or rope base gets its
//! own value without anyone editing the layout. An open set, where a layout names GGUF
//! keys directly, would let one layout quietly hardcode one model's numbers.
//!
//! The names are llama.cpp's own: the `LLAMA_LOAD_LOCALS` macro in `src/llama-model.h`
//! (`n_embd`, `n_embd_k_gqa`, `n_rot`, ...), the `llama_hparams` fields (`f_norm_rms_eps`,
//! `n_ff_exp`), and the `llm_graph_context` members the graph code passes to
//! `ggml_rope_ext` (`freq_base`, `ext_factor`, `n_ctx_orig`, ...). A layout line then
//! reads the way the C++ call it came from reads.
//!
//! ## Where each value comes from
//!
//! Each symbol below is read the way llama.cpp reads it at the pinned commit: the same
//! key, the same default when the key is absent, and per layer where llama.cpp keeps a
//! per-layer array (`get_key_or_arr`). Outside a block, a per-layer symbol reads layer
//! 0, which is what llama.cpp's `n_head()` with no argument does.
//!
//! A few symbols describe the bind options rather than the model: the context and batch
//! sizes, and the KV cache types. The KV rotation symbols (`attn_rot_k` and friends)
//! repeat the rule in `llama_kv_cache`'s constructor that turns on a Hadamard rotation
//! of Q, K and V when the cache is quantized.
//!
//! Some hparams are read by only some architectures' `load_arch_hparams`. A symbol for
//! one of those (`f_attention_scale`, for example) would give a value llama.cpp never
//! reads for the others, so none is offered yet. Add one when a layout for an
//! architecture that reads it is written, and say which in its row.

use crate::gguf::{Gguf, Value as GValue};
use crate::load::Model;

use super::bind::BindOptions;
use super::expr::Value;

/// Every symbol, with where it comes from. Kept as data so the README can be checked
/// against it and an unknown name can be answered with the list.
pub const SYMBOLS: &[(&str, &str)] = &[
    // The model's shape.
    ("n_layer", "`<arch>.block_count`"),
    ("n_embd", "`<arch>.embedding_length`"),
    ("n_head", "`<arch>.attention.head_count`, per layer when it is an array; 0 when absent"),
    ("n_head_kv", "`<arch>.attention.head_count_kv`, per layer when it is an array; `n_head` when absent"),
    ("n_embd_head_k", "`<arch>.attention.key_length`; `n_embd / n_head` when absent"),
    ("n_embd_head_v", "`<arch>.attention.value_length`; `n_embd / n_head` when absent"),
    ("n_embd_k_gqa", "`n_embd_head_k * n_head_kv`, per layer"),
    ("n_embd_v_gqa", "`n_embd_head_v * n_head_kv`, per layer"),
    ("n_rot", "`<arch>.rope.dimension_count`; `n_embd_head_k` when absent"),
    ("n_ff", "`<arch>.feed_forward_length`, per layer when it is an array; 0 when absent"),
    ("n_expert", "`<arch>.expert_count`; 0 when absent"),
    ("n_expert_used", "`<arch>.expert_used_count`; 0 when absent"),
    ("n_ff_exp", "`<arch>.expert_feed_forward_length`; 0 when absent"),
    ("n_vocab", "the length of `tokenizer.ggml.tokens`, else `<arch>.vocab_size`"),
    ("n_ctx_train", "`<arch>.context_length`"),
    // Hparams passed to ops.
    ("f_norm_rms_eps", "`<arch>.attention.layer_norm_rms_epsilon`; required when used"),
    ("f_norm_eps", "`<arch>.attention.layer_norm_epsilon`; required when used"),
    ("freq_base", "`<arch>.rope.freq_base`; 10000 when absent"),
    ("freq_scale", "1 / `<arch>.rope.scaling.factor` (or `rope.scale_linear`); 1 when absent or when `rope.scaling.type` is none"),
    ("ext_factor", "1 when `<arch>.rope.scaling.type` is yarn, else 0"),
    ("attn_factor", "the YaRN magnitude correction from llama-context.cpp, times `<arch>.rope.scaling.attn_factor`"),
    ("beta_fast", "32, llama_hparams' default"),
    ("beta_slow", "1, llama_hparams' default"),
    ("n_ctx_orig", "`<arch>.rope.scaling.original_context_length`; `n_ctx_train` when absent"),
    // The bind options.
    ("n_ctx", "BindOptions::n_ctx, the KV cache length"),
    ("n_kv", "the KV cells one step attends over; planned at its maximum, n_ctx"),
    ("n_tokens", "BindOptions::n_tokens, the tokens in one step"),
    ("n_outputs", "BindOptions::n_outputs, the tokens whose logits the step returns"),
    ("kv_k_quantized", "the K cache type is block-quantized"),
    ("kv_v_quantized", "the V cache type is block-quantized"),
    ("kv_k_f32", "the K cache type is f32"),
    ("kv_v_f32", "the V cache type is f32"),
    ("attn_rot_k", "llama_kv_cache rotates Q and K: K cache quantized and `n_embd_head_k % 64 == 0`"),
    ("attn_rot_v", "llama_kv_cache rotates V: V cache quantized and `n_embd_head_v % 64 == 0`"),
    ("n_attn_rot_k", "the K rotation's size: the largest power of two from 64 that divides n_embd_head_k"),
    ("n_attn_rot_v", "the V rotation's size: 64"),
];

/// Whether `name` is in the closed set.
pub(crate) fn is_symbol(name: &str) -> bool {
    SYMBOLS.iter().any(|(n, _)| *n == name)
}

/// Resolves symbols against one model and one set of bind options.
pub(crate) struct Hparams<'m> {
    meta: &'m Gguf,
    arch: &'m str,
    opts: BindOptions,
    n_layer: u32,
}

impl<'m> Hparams<'m> {
    pub(crate) fn new(model: &'m Model, opts: &BindOptions) -> Result<Self, String> {
        let meta = model.meta();
        let arch = model.architecture();
        let n_layer = meta
            .u64(&format!("{arch}.block_count"))
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| format!("the model has no `{arch}.block_count`"))?;
        Ok(Hparams { meta, arch, opts: *opts, n_layer })
    }

    pub(crate) fn n_layer(&self) -> u32 {
        self.n_layer
    }

    fn key(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.arch)
    }

    /// An integer key, taking the layer's element when the key is an array.
    fn int_key(&self, suffix: &str, layer: Option<u32>) -> Option<i64> {
        let v = match self.meta.get(&self.key(suffix))? {
            GValue::Array(items) => items.get(layer.unwrap_or(0) as usize)?.as_u64()?,
            other => other.as_u64()?,
        };
        i64::try_from(v).ok()
    }

    fn required_int(&self, suffix: &str, layer: Option<u32>) -> Result<i64, String> {
        self.int_key(suffix, layer).ok_or_else(|| format!("the model has no `{}`", self.key(suffix)))
    }

    fn float_key(&self, suffix: &str) -> Option<f64> {
        match self.meta.get(&self.key(suffix))? {
            GValue::F32(v) => Some(f64::from(*v)),
            GValue::F64(v) => Some(*v),
            #[allow(clippy::cast_precision_loss)]
            other => other.as_u64().map(|v| v as f64),
        }
    }

    fn n_head(&self, layer: Option<u32>) -> i64 {
        self.int_key("attention.head_count", layer).unwrap_or(0)
    }

    fn head_dim(&self, suffix: &str) -> Result<i64, String> {
        if let Some(v) = self.int_key(suffix, None) {
            return Ok(v);
        }
        // llama_model_base::load_hparams: n_embd / n_head(), layer 0.
        let heads = self.n_head(None);
        if heads == 0 {
            return Ok(0);
        }
        Ok(self.required_int("embedding_length", None)? / heads)
    }

    fn n_head_kv(&self, layer: Option<u32>) -> i64 {
        self.int_key("attention.head_count_kv", layer).unwrap_or_else(|| self.n_head(layer))
    }

    /// The rope scaling type, as `llama_rope_scaling_type_from_string` reads it.
    fn rope_scaling(&self) -> &str {
        self.meta.string(&self.key("rope.scaling.type")).unwrap_or("linear")
    }

    /// `cparams.rope_freq_scale`, computed in f32 as llama.cpp computes it.
    fn freq_scale(&self) -> f32 {
        if self.rope_scaling() == "none" {
            return 1.0;
        }
        #[allow(clippy::cast_possible_truncation)]
        let factor = self.float_key("rope.scaling.factor").or_else(|| self.float_key("rope.scale_linear")).unwrap_or(0.0) as f32;
        if factor == 0.0 { 1.0 } else { 1.0 / factor }
    }

    fn ext_factor(&self) -> f32 {
        if self.rope_scaling() == "yarn" { 1.0 } else { 0.0 }
    }

    /// `cparams.yarn_attn_factor`, from the block in `llama_context`'s constructor.
    /// `rope_yarn_log_mul` is read only by deepseek2, so its branch is not taken here.
    fn attn_factor(&self) -> f32 {
        let mut f = 1.0f32;
        if self.ext_factor() != 0.0 {
            let factor = 1.0 / self.freq_scale();
            let mscale = if factor <= 1.0 { 1.0 } else { 0.1 * factor.ln() + 1.0 };
            f = mscale;
            f *= 1.0 / (1.0 + 0.1 * factor.ln());
        }
        #[allow(clippy::cast_possible_truncation)]
        let attn = self.float_key("rope.scaling.attn_factor").unwrap_or(1.0) as f32;
        f * attn
    }

    fn rot(&self, quantized: bool, head: i64) -> bool {
        quantized && head > 0 && head % 64 == 0
    }

    /// Resolve one symbol, at a layer when inside a block.
    pub(crate) fn value(&self, name: &str, layer: Option<u32>) -> Result<Value, String> {
        let int = |v: i64| Ok(Value::Int(v));
        let float = |v: f64| Ok(Value::Float(v));
        let flag = |v: bool| Ok(Value::Bool(v));
        match name {
            "n_layer" => int(i64::from(self.n_layer)),
            "n_embd" => int(self.required_int("embedding_length", None)?),
            "n_head" => int(self.n_head(layer)),
            "n_head_kv" => int(self.n_head_kv(layer)),
            "n_embd_head_k" => int(self.head_dim("attention.key_length")?),
            "n_embd_head_v" => int(self.head_dim("attention.value_length")?),
            "n_embd_k_gqa" => int(self.head_dim("attention.key_length")? * self.n_head_kv(layer)),
            "n_embd_v_gqa" => int(self.head_dim("attention.value_length")? * self.n_head_kv(layer)),
            "n_rot" => match self.int_key("rope.dimension_count", None) {
                Some(v) => int(v),
                None => int(self.head_dim("attention.key_length")?),
            },
            "n_ff" => int(self.int_key("feed_forward_length", layer).unwrap_or(0)),
            "n_expert" => int(self.int_key("expert_count", None).unwrap_or(0)),
            "n_expert_used" => int(self.int_key("expert_used_count", None).unwrap_or(0)),
            "n_ff_exp" => int(self.int_key("expert_feed_forward_length", None).unwrap_or(0)),
            "n_vocab" => {
                if let Some(n) = self.meta.get("tokenizer.ggml.tokens").and_then(GValue::len) {
                    return int(i64::try_from(n).map_err(|_| "vocabulary too large")?);
                }
                int(self.required_int("vocab_size", None).map_err(|_| "the model has neither tokenizer.ggml.tokens nor a vocab_size".to_string())?)
            }
            "n_ctx_train" => int(self.required_int("context_length", None)?),
            "f_norm_rms_eps" => float(
                self.float_key("attention.layer_norm_rms_epsilon")
                    .ok_or_else(|| format!("the model has no `{}`", self.key("attention.layer_norm_rms_epsilon")))?,
            ),
            "f_norm_eps" => float(
                self.float_key("attention.layer_norm_epsilon")
                    .ok_or_else(|| format!("the model has no `{}`", self.key("attention.layer_norm_epsilon")))?,
            ),
            "freq_base" => float(self.float_key("rope.freq_base").unwrap_or(10000.0)),
            "freq_scale" => float(f64::from(self.freq_scale())),
            "ext_factor" => float(f64::from(self.ext_factor())),
            "attn_factor" => float(f64::from(self.attn_factor())),
            "beta_fast" => float(32.0),
            "beta_slow" => float(1.0),
            "n_ctx_orig" => match self.int_key("rope.scaling.original_context_length", None) {
                Some(v) if v != 0 => int(v),
                _ => int(self.required_int("context_length", None)?),
            },
            "n_ctx" | "n_kv" => int(i64::from(self.opts.n_ctx)),
            "n_tokens" => int(i64::from(self.opts.n_tokens)),
            "n_outputs" => int(i64::from(self.opts.n_outputs)),
            "kv_k_quantized" => flag(self.opts.kv_k.is_quantized()),
            "kv_v_quantized" => flag(self.opts.kv_v.is_quantized()),
            "kv_k_f32" => flag(self.opts.kv_k == crate::load::KvType::F32),
            "kv_v_f32" => flag(self.opts.kv_v == crate::load::KvType::F32),
            "attn_rot_k" => flag(self.rot(self.opts.kv_k.is_quantized(), self.head_dim("attention.key_length")?)),
            "attn_rot_v" => flag(self.rot(self.opts.kv_v.is_quantized(), self.head_dim("attention.value_length")?)),
            "n_attn_rot_k" => {
                // llama_kv_cache::build_input_k_rot: double from 64 while it still divides.
                let head = self.head_dim("attention.key_length")?;
                if !self.rot(self.opts.kv_k.is_quantized(), head) {
                    return int(0);
                }
                let mut n = 64;
                loop {
                    n *= 2;
                    if head % n != 0 {
                        break;
                    }
                }
                int(n / 2)
            }
            "n_attn_rot_v" => int(if self.rot(self.opts.kv_v.is_quantized(), self.head_dim("attention.value_length")?) { 64 } else { 0 }),
            other => Err(format!("`{other}` is not a layout symbol")),
        }
    }
}
