//! Author: Joe Ricks
//! File: src/layout/symbols.rs
//! Description: Lists the symbols layout expressions may use, and reads their values from
//! a model's GGUF header and the bind options.

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

/// Purpose: This function tells whether a name is a layout symbol.
///
/// Inputs:
/// - `name`: the name.
///
/// Outputs: `true` when it is.
pub(crate) fn is_symbol(name: &str) -> bool {
    SYMBOLS.iter().any(|(n, _)| *n == name)
}

/// Purpose: This struct reads layout symbol values for one model and its bind options.
pub(crate) struct Hparams<'m> {
    meta: &'m Gguf,
    arch: &'m str,
    opts: BindOptions,
    n_layer: u32,
}

impl<'m> Hparams<'m> {
    /// Purpose: This function builds the symbol reader for one model and one set of bind
    /// options.
    ///
    /// Inputs:
    /// - `model`: the opened GGUF model.
    /// - `opts`: the bind options.
    ///
    /// Outputs: the reader, or an error when the model has no block count.
    pub(crate) fn new(model: &'m Model, opts: &BindOptions) -> Result<Self, String> {
        let meta = model.meta();
        let arch = model.architecture();
        let n_layer = meta
            .u64(&format!("{arch}.block_count"))
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| format!("the model has no `{arch}.block_count`"))?;
        Ok(Hparams { meta, arch, opts: *opts, n_layer })
    }

    /// Purpose: This function returns the model's layer count.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the layer count.
    pub(crate) fn n_layer(&self) -> u32 {
        self.n_layer
    }

    /// Purpose: This function builds a full GGUF key from a key suffix.
    ///
    /// Inputs:
    /// - `suffix`: the key after the architecture name, such as `block_count`.
    ///
    /// Outputs: the full key, such as `qwen3.block_count`.
    fn key(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.arch)
    }

    /// Purpose: This function reads an integer GGUF key, taking one layer's entry when
    /// the key is an array.
    ///
    /// Inputs:
    /// - `suffix`: the key after the architecture name, such as `block_count`.
    /// - `layer`: the layer index, or `None` for layer 0.
    ///
    /// Outputs: the value, or `None` when the key is missing.
    fn int_key(&self, suffix: &str, layer: Option<u32>) -> Option<i64> {
        let v = match self.meta.get(&self.key(suffix))? {
            GValue::Array(items) => items.get(layer.unwrap_or(0) as usize)?.as_u64()?,
            other => other.as_u64()?,
        };
        i64::try_from(v).ok()
    }

    /// Purpose: This function reads an integer GGUF key that must be present.
    ///
    /// Inputs:
    /// - `suffix`: the key after the architecture name, such as `block_count`.
    /// - `layer`: the layer index, or `None` for layer 0.
    ///
    /// Outputs: the value, or an error naming the missing key.
    fn required_int(&self, suffix: &str, layer: Option<u32>) -> Result<i64, String> {
        self.int_key(suffix, layer).ok_or_else(|| format!("the model has no `{}`", self.key(suffix)))
    }

    /// Purpose: This function reads a GGUF key as a float.
    ///
    /// Inputs:
    /// - `suffix`: the key after the architecture name, such as `block_count`.
    ///
    /// Outputs: the value, or `None` when the key is missing.
    fn float_key(&self, suffix: &str) -> Option<f64> {
        match self.meta.get(&self.key(suffix))? {
            GValue::F32(v) => Some(f64::from(*v)),
            GValue::F64(v) => Some(*v),
            #[allow(clippy::cast_precision_loss)]
            other => other.as_u64().map(|v| v as f64),
        }
    }

    /// Purpose: This function returns the number of attention heads at a layer.
    ///
    /// Inputs:
    /// - `layer`: the layer index, or `None` for layer 0.
    ///
    /// Outputs: the head count, or 0 when the key is missing.
    fn n_head(&self, layer: Option<u32>) -> i64 {
        self.int_key("attention.head_count", layer).unwrap_or(0)
    }

    /// Purpose: This function returns the width of one attention head.
    ///
    /// Inputs:
    /// - `suffix`: the head width key to read.
    ///
    /// Outputs: the width, or an error.
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

    /// Purpose: This function returns the number of KV heads at a layer.
    ///
    /// Inputs:
    /// - `layer`: the layer index, or `None` for layer 0.
    ///
    /// Outputs: the KV head count.
    fn n_head_kv(&self, layer: Option<u32>) -> i64 {
        self.int_key("attention.head_count_kv", layer).unwrap_or_else(|| self.n_head(layer))
    }

    /// Purpose: This function returns the model's rope scaling type.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the type's name, `linear` when the key is missing.
    fn rope_scaling(&self) -> &str {
        self.meta.string(&self.key("rope.scaling.type")).unwrap_or("linear")
    }

    /// Purpose: This function returns rope's frequency scale.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the scale.
    fn freq_scale(&self) -> f32 {
        if self.rope_scaling() == "none" {
            return 1.0;
        }
        #[allow(clippy::cast_possible_truncation)]
        let factor = self.float_key("rope.scaling.factor").or_else(|| self.float_key("rope.scale_linear")).unwrap_or(0.0) as f32;
        if factor == 0.0 { 1.0 } else { 1.0 / factor }
    }

    /// Purpose: This function returns rope's YaRN extrapolation factor.
    ///
    /// Inputs: none.
    ///
    /// Outputs: 1 for YaRN scaling, else 0.
    fn ext_factor(&self) -> f32 {
        if self.rope_scaling() == "yarn" { 1.0 } else { 0.0 }
    }

    /// Purpose: This function returns rope's attention magnitude factor.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the factor.
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

    /// Purpose: This function tells whether llama.cpp rotates a KV cache's heads by a
    /// Hadamard matrix.
    ///
    /// Inputs:
    /// - `quantized`: whether the cache type is quantized.
    /// - `head`: the head width.
    ///
    /// Outputs: `true` when the rotation applies.
    fn rot(&self, quantized: bool, head: i64) -> bool {
        quantized && head > 0 && head % 64 == 0
    }

    /// Purpose: This function returns the value of one layout symbol.
    ///
    /// Inputs:
    /// - `name`: the symbol.
    /// - `layer`: the layer index, or `None` outside the block.
    ///
    /// Outputs: the value, or an error.
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
