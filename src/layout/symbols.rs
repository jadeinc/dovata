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
    // Interleaved experts, NoPE layers and chunked attention, added with llama4. The
    // rules marked llama4 are llama4's load_arch_hparams; other architectures set them
    // their own way, so those symbols refuse other architectures until a layout ports
    // that architecture's rule.
    ("n_moe_layer_step", "`<arch>.interleave_moe_layer_step`; required when used"),
    ("n_no_rope_layer_step", "llama4: `n_layer` when `<arch>.attention.sliding_window` is 0, else 4, llama_hparams' default"),
    ("n_swa", "llama4: 8192, which llama4 sets over the file's value; 0 when `<arch>.attention.sliding_window` is 0"),
    ("is_swa", "per layer. llama4: `L % p < p - 1`, p = `<arch>.attention.sliding_window_pattern` (4 when absent); false when n_swa is 0"),
    ("freq_base_swa", "llama4: `<arch>.rope.freq_base_swa`; `freq_base` when absent"),
    ("freq_scale_swa", "llama4: 1 / `<arch>.rope.scaling.factor` (or `rope.scale_linear`); 1 when absent; not reset by scaling type none"),
    ("n_ctx_swa", "the SWA cache's length: n_ctx, the size llama.cpp gives it with swa_full"),
    ("n_kv_swa", "the SWA cache cells one step attends over; planned at its maximum, n_ctx_swa"),
    // Hparams deepseek2's load_arch_hparams reads, for MLA attention, leading dense
    // blocks and expert routing.
    ("n_layer_dense_lead", "`<arch>.leading_dense_block_count`; 0 when absent"),
    ("q_lora_rank", "`<arch>.attention.q_lora_rank`, required; 0 for deepseek2's lite variants, which never read it"),
    ("kv_lora_rank", "`<arch>.attention.kv_lora_rank`; required when used"),
    ("is_mla", "`<arch>.attention.key_length_mla` and `value_length_mla` are both set and not 0"),
    ("n_embd_head_k_mla", "`<arch>.attention.key_length_mla` when `is_mla`, else `n_embd_head_k`"),
    ("n_embd_head_v_mla", "`<arch>.attention.value_length_mla` when `is_mla`, else `n_embd_head_v`"),
    ("n_expert_shared", "`<arch>.expert_shared_count`; required when used"),
    ("n_expert_groups", "`<arch>.expert_group_count`; 0 when absent"),
    ("n_group_used", "`<arch>.expert_group_used_count`; 0 when absent"),
    ("expert_gating_func", "`<arch>.expert_gating_func`: 1 softmax, 2 sigmoid; deepseek2's default when absent or 0"),
    ("expert_weights_norm", "`<arch>.expert_weights_norm`; false when absent"),
    ("expert_weights_scale", "`<arch>.expert_weights_scale`; 0 when absent"),
    ("f_attn_temp_scale", "`<arch>.attention.temperature_scale`; 0 when absent"),
    ("mscale", "deepseek2.cpp's YaRN factor for kq_scale, from `attn_factor`, `freq_scale` and `<arch>.rope.scaling.yarn_log_multiplier`"),
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
            let log_mul = self.rope_yarn_log_mul();
            if log_mul != 0.0 {
                // llama_context's rope_yarn_log_mul branch: get_mscale(factor, mscale)
                // over get_mscale(factor, mscale_all_dims), where deepseek2 takes mscale
                // to be mscale_all_dims unless that is 1.
                let get_mscale = |scale: f32, m: f32| if scale <= 1.0 { 1.0 } else { 0.1 * m * scale.ln() + 1.0 };
                let mscale = if self.arch == "deepseek2" && log_mul != 1.0 { log_mul } else { 1.0 };
                f = get_mscale(factor, mscale) / get_mscale(factor, log_mul);
            } else {
                let mscale = if factor <= 1.0 { 1.0 } else { 0.1 * factor.ln() + 1.0 };
                f = mscale;
            }
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

    // ---- llama4: chunked attention and NoPE layers ----

    /// Purpose: This function reads an integer GGUF key that must hold one value, as
    /// llama.cpp's scalar `get_key` and `get_key_or_arr(kid, uint32_t &)` read it.
    ///
    /// Inputs:
    /// - `suffix`: the key after the architecture name, such as `attention.sliding_window`.
    ///
    /// Outputs: the value, or `None` when the key is missing or is an array.
    fn scalar_key(&self, suffix: &str) -> Option<i64> {
        match self.meta.get(&self.key(suffix))? {
            GValue::Array(_) => None,
            other => i64::try_from(other.as_u64()?).ok(),
        }
    }

    /// Purpose: This function refuses a symbol whose rule dovata has ported only from
    /// llama4's load_arch_hparams.
    ///
    /// Inputs:
    /// - `name`: the symbol.
    ///
    /// Outputs: `Ok(())` for a llama4 model, or an error naming what to port.
    fn llama4_only(&self, name: &str) -> Result<(), String> {
        if self.arch == "llama4" {
            return Ok(());
        }
        Err(format!("`{name}` is read only for llama4 so far; port {}'s rule from its load_arch_hparams into symbols.rs", self.arch))
    }

    /// Purpose: This function tells whether a llama4 model uses chunked attention, which it
    /// does unless the file sets `attention.sliding_window` to 0.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `true` for chunked attention, LLAMA_SWA_TYPE_CHUNKED in llama.cpp.
    fn llama4_chunked(&self) -> bool {
        self.scalar_key("attention.sliding_window") != Some(0)
    }

    /// Purpose: This function tells whether a llama4 layer uses chunked attention.
    ///
    /// Inputs:
    /// - `layer`: the layer index, or `None` for layer 0.
    ///
    /// Outputs: `true` for a chunked layer.
    fn llama4_is_swa(&self, layer: Option<u32>) -> bool {
        if !self.llama4_chunked() {
            return false;
        }
        // get_key_or_arr(SLIDING_WINDOW_PATTERN, swa_period, false) keeps 4 for an
        // absent key or an array; set_swa_pattern(swa_period) then marks the layers.
        let period = self.scalar_key("attention.sliding_window_pattern").unwrap_or(4);
        let il = i64::from(layer.unwrap_or(0));
        period == 0 || il % period < period - 1
    }

    /// Purpose: This function returns llama_hparams' rope_freq_scale_train, the scale
    /// before llama_context resets it for scaling type none.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the scale.
    fn rope_freq_scale_train(&self) -> f32 {
        #[allow(clippy::cast_possible_truncation)]
        let factor = self.float_key("rope.scaling.factor").or_else(|| self.float_key("rope.scale_linear")).unwrap_or(0.0) as f32;
        if factor == 0.0 { 1.0 } else { 1.0 / factor }
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
            // llama4: interleaved experts, NoPE layers and chunked attention.
            "n_moe_layer_step" => int(
                self.scalar_key("interleave_moe_layer_step").ok_or_else(|| format!("the model has no `{}`", self.key("interleave_moe_layer_step")))?,
            ),
            "n_no_rope_layer_step" => {
                self.llama4_only(name)?;
                int(if self.llama4_chunked() { 4 } else { i64::from(self.n_layer) })
            }
            "n_swa" => {
                self.llama4_only(name)?;
                int(if self.llama4_chunked() { 8192 } else { 0 })
            }
            "is_swa" => {
                self.llama4_only(name)?;
                flag(self.llama4_is_swa(layer))
            }
            "freq_base_swa" => {
                self.llama4_only(name)?;
                // rope_freq_base_train_swa: llama_hparams' 10000 unless the attention is
                // chunked, when it starts at rope_freq_base_train.
                if !self.llama4_chunked() {
                    return float(10000.0);
                }
                float(self.float_key("rope.freq_base_swa").unwrap_or_else(|| self.float_key("rope.freq_base").unwrap_or(10000.0)))
            }
            "freq_scale_swa" => {
                self.llama4_only(name)?;
                // rope_freq_scale_train_swa: llama_hparams' 1 unless the attention is
                // chunked, when it is rope_freq_scale_train.
                float(if self.llama4_chunked() { f64::from(self.rope_freq_scale_train()) } else { 1.0 })
            }
            "n_ctx_swa" | "n_kv_swa" => int(i64::from(self.opts.n_ctx)),
            // deepseek2's hparams; the helpers are in the impl block below.
            "n_layer_dense_lead" => int(self.int_key("leading_dense_block_count", None).unwrap_or(0)),
            "q_lora_rank" => int(self.q_lora_rank()?),
            "kv_lora_rank" => int(self.required_int("attention.kv_lora_rank", None)?),
            "is_mla" => flag(self.is_mla()),
            // llama_hparams::n_embd_head_k_mla and _v_mla: the MLA widths, or the plain ones.
            "n_embd_head_k_mla" if self.is_mla() => int(self.required_int("attention.key_length_mla", None)?),
            "n_embd_head_k_mla" => int(self.head_dim("attention.key_length")?),
            "n_embd_head_v_mla" if self.is_mla() => int(self.required_int("attention.value_length_mla", None)?),
            "n_embd_head_v_mla" => int(self.head_dim("attention.value_length")?),
            "n_expert_shared" => int(self.required_int("expert_shared_count", None)?),
            "n_expert_groups" => int(self.int_key("expert_group_count", None).unwrap_or(0)),
            "n_group_used" => int(self.int_key("expert_group_used_count", None).unwrap_or(0)),
            "expert_gating_func" => int(self.expert_gating_func()),
            "expert_weights_norm" => flag(self.bool_key("expert_weights_norm").unwrap_or(false)),
            "expert_weights_scale" => float(self.float_key("expert_weights_scale").unwrap_or(0.0)),
            "f_attn_temp_scale" => float(self.float_key("attention.temperature_scale").unwrap_or(0.0)),
            "mscale" => float(f64::from(self.mscale())),
            other => Err(format!("`{other}` is not a layout symbol")),
        }
    }
}

// deepseek2's hparams, read the way `llama_model_deepseek2::load_arch_hparams` reads
// them at llama.cpp commit 52b3df0.
impl Hparams<'_> {
    /// Purpose: This function reads a GGUF key as a truth value.
    ///
    /// Inputs:
    /// - `suffix`: the key after the architecture name, such as `expert_weights_norm`.
    ///
    /// Outputs: the value, or `None` when the key is missing or not a bool.
    fn bool_key(&self, suffix: &str) -> Option<bool> {
        match self.meta.get(&self.key(suffix))? {
            GValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Purpose: This function counts the vocabulary the way deepseek2's load_arch_hparams
    /// does, from `vocab_size` first and the token list second.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the vocabulary size, or 0 when the file states neither.
    fn deepseek2_n_vocab(&self) -> i64 {
        self.int_key("vocab_size", None)
            .or_else(|| self.meta.get("tokenizer.ggml.tokens").and_then(GValue::len).and_then(|n| i64::try_from(n).ok()))
            .unwrap_or(0)
    }

    /// Purpose: This function tells whether a deepseek2 model is one of the variants
    /// llama.cpp calls lite, such as DeepSeek-V2-Lite and GigaChat3-10B-A1.8B.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `true` for 27 or 26 layers, or 48 layers with a vocabulary of 128256.
    fn deepseek2_is_lite(&self) -> bool {
        self.n_layer == 27 || self.n_layer == 26 || (self.n_layer == 48 && self.deepseek2_n_vocab() == 128_256)
    }

    /// Purpose: This function returns the rank of the low-rank Q projection.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the rank, 0 for a deepseek2 lite variant, or an error when the key is
    /// missing.
    fn q_lora_rank(&self) -> Result<i64, String> {
        // load_arch_hparams skips the key for the lite variants, so n_lora_q stays 0
        // even when the file has it.
        if self.arch == "deepseek2" && self.deepseek2_is_lite() {
            return Ok(0);
        }
        self.required_int("attention.q_lora_rank", None)
    }

    /// Purpose: This function tells whether the model stores MLA's head widths, which is
    /// how llama_hparams::is_mla decides.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `true` when both MLA head widths are set and not 0.
    fn is_mla(&self) -> bool {
        let k = self.int_key("attention.key_length_mla", None).unwrap_or(0);
        let v = self.int_key("attention.value_length_mla", None).unwrap_or(0);
        k != 0 && v != 0
    }

    /// Purpose: This function returns the expert gating function's
    /// llama_expert_gating_func_type value.
    ///
    /// Inputs: none.
    ///
    /// Outputs: 1 for softmax, 2 for sigmoid, or the file's value.
    fn expert_gating_func(&self) -> i64 {
        let v = self.int_key("expert_gating_func", None).unwrap_or(0);
        if v != 0 || self.arch != "deepseek2" {
            return v;
        }
        // deepseek2 files that predate the key: GLM 4.7 Lite is sigmoid, the DeepSeek V2
        // and V2.5 files are softmax.
        if (self.n_layer == 47 || self.n_layer == 48) && self.deepseek2_n_vocab() == 154_880 { 2 } else { 1 }
    }

    /// Purpose: This function returns llama_hparams' rope_yarn_log_mul.
    ///
    /// Inputs: none.
    ///
    /// Outputs: deepseek2's `rope.scaling.yarn_log_multiplier` over 0.1, or 0 when the key
    /// is missing or the architecture is not deepseek2.
    fn rope_yarn_log_mul(&self) -> f32 {
        if self.arch != "deepseek2" {
            return 0.0;
        }
        // load_arch_hparams cancels the 0.1 the convert script multiplies in.
        #[allow(clippy::cast_possible_truncation)]
        self.float_key("rope.scaling.yarn_log_multiplier").map_or(0.0, |v| v as f32 / 0.1)
    }

    /// Purpose: This function returns the factor deepseek2's graph squares into kq_scale.
    ///
    /// Inputs: none.
    ///
    /// Outputs: deepseek2.cpp's `mscale`, computed in f32 as llama.cpp computes it.
    fn mscale(&self) -> f32 {
        // First undo the YaRN correction llama_context put into attn_factor, then apply
        // the model's own mscale_all_dim.
        let freq_scale = self.freq_scale();
        let attn_factor_org = self.attn_factor() * (1.0 + 0.1 * (1.0 / freq_scale).ln());
        attn_factor_org * (1.0 + 0.1 * self.rope_yarn_log_mul() * (1.0 / freq_scale).ln())
    }
}
