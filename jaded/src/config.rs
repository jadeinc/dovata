//! Daemon configuration, parsed from the environment.
//!
//! Pure and platform-independent: the KV-cache type is modeled by our own
//! [`KvType`] enum (the binary maps it to llama.cpp's `KvCacheType`), so the
//! parsing and the flash-attention validation rule are unit-testable without
//! pulling in the inference backend.

pub const DEFAULT_MODEL_PATH: &str = "/jade/inference/model.gguf";
pub const DEFAULT_CTX_SIZE: u32 = 32_768;
pub const DEFAULT_GPU_LAYERS: u32 = 999;
pub const DEFAULT_CACHE_SIZE: usize = 256;
pub const DEFAULT_N_BATCH: u32 = 2048;
pub const DEFAULT_N_UBATCH: u32 = 512;
/// Max concurrent sequences the context can hold (RLM fan-out / parallel serving).
/// The shared KV cache (`ctx_size` cells) is partitioned across these at runtime.
pub const DEFAULT_N_SEQ_MAX: u32 = 16;
pub const DEFAULT_KV_TYPE: KvType = KvType::Q4_0;

/// KV-cache quantization type. Mirrors the subset of llama.cpp's `KvCacheType`
/// the daemon supports; the binary maps this to the llama.cpp enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvType {
    F16,
    BF16,
    F32,
    Q8_0,
    Q5_1,
    Q5_0,
    Q4_1,
    Q4_0,
}

impl KvType {
    /// Parse a `JADE_KV_TYPE_*` string. Unknown values fall back to `Q8_0`
    /// (the safest quantized type) and the fallback is signalled to the caller.
    pub fn parse(s: &str) -> Self {
        // normalize case first so "Q4_0", "q4_0", "Q4_0" all match the same arm
        match s.to_ascii_lowercase().as_str() {
            // each accepted spelling maps to its enum variant
            "f16" => KvType::F16,
            "bf16" => KvType::BF16,
            "f32" => KvType::F32,
            "q8_0" => KvType::Q8_0,
            "q5_1" => KvType::Q5_1,
            "q5_0" => KvType::Q5_0,
            "q4_1" => KvType::Q4_1,
            "q4_0" => KvType::Q4_0,
            // anything we don't recognize: fall back to Q8_0 rather than crash.
            // Q8_0 is the safe choice — high quality and always legal.
            _ => KvType::Q8_0,
        }
    }

    /// Whether parsing `s` produced an exact match (vs. the `Q8_0` fallback).
    pub fn is_known(s: &str) -> bool {
        // same accepted set as parse() — used so callers can warn the user that
        // their JADE_KV_TYPE_* value was a typo (silently fell back) vs. valid
        matches!(
            s.to_ascii_lowercase().as_str(),
            "f16" | "bf16" | "f32" | "q8_0" | "q5_1" | "q5_0" | "q4_1" | "q4_0"
        )
    }

    /// Quantized types require flash attention for the V cache.
    pub fn is_quantized(self) -> bool {
        // the three float types are the *only* non-quantized ones; everything
        // else (the q* family) is quantized. Phrasing it as "not a float type"
        // means new q* variants are automatically treated as quantized.
        !matches!(self, KvType::F16 | KvType::BF16 | KvType::F32)
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub model_path: String,
    pub ctx_size: u32,
    pub gpu_layers: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    /// Max concurrent sequences (continuous-batching fan-out). See [`DEFAULT_N_SEQ_MAX`].
    pub n_seq_max: u32,
    pub kv_type_k: KvType,
    pub kv_type_v: KvType,
    pub flash_attn: bool,
    pub cache_size: usize,
}

impl Config {
    /// Build a config from the process environment.
    pub fn from_env() -> Self {
        // delegate to the testable getter form, feeding it the real environment.
        // std::env::var returns Result; .ok() turns it into the Option the
        // getter expects (None = var unset).
        Self::from_getter(|k| std::env::var(k).ok())
    }

    /// Build a config from an arbitrary key→value getter (testable).
    pub fn from_getter(get: impl Fn(&str) -> Option<String>) -> Self {
        // small helper: look up a key, try to parse it as u32, and fall back to
        // the default `d` if the var is unset OR doesn't parse. This is why a
        // garbage value like JADE_CTX_SIZE=abc quietly becomes the default
        // instead of crashing the daemon at boot.
        let parse_u32 = |k: &str, d: u32| get(k).and_then(|s| s.parse().ok()).unwrap_or(d);
        Config {
            // string field: use the env value, or own a copy of the default path
            model_path: get("JADE_MODEL_PATH").unwrap_or_else(|| DEFAULT_MODEL_PATH.to_owned()),
            // numeric fields all follow the same parse-or-default pattern
            ctx_size: parse_u32("JADE_CTX_SIZE", DEFAULT_CTX_SIZE),
            gpu_layers: parse_u32("JADE_GPU_LAYERS", DEFAULT_GPU_LAYERS),
            n_batch: parse_u32("JADE_N_BATCH", DEFAULT_N_BATCH),
            n_ubatch: parse_u32("JADE_N_UBATCH", DEFAULT_N_UBATCH),
            n_seq_max: parse_u32("JADE_N_SEQ_MAX", DEFAULT_N_SEQ_MAX),
            // KV types: if the var is set, parse it (unknown → Q8_0); if unset,
            // use the default type. .map(..).unwrap_or(..) keeps "unset" distinct
            // from "set but unknown".
            kv_type_k: get("JADE_KV_TYPE_K").map(|s| KvType::parse(&s)).unwrap_or(DEFAULT_KV_TYPE),
            kv_type_v: get("JADE_KV_TYPE_V").map(|s| KvType::parse(&s)).unwrap_or(DEFAULT_KV_TYPE),
            // flash attention defaults ON. We only turn it OFF for the explicit
            // falsey spellings; any other value (or unset) leaves it enabled.
            flash_attn: !matches!(get("JADE_FLASH_ATTN").as_deref(), Some("0") | Some("off") | Some("false")),
            // cache size parsed directly (usize, so it can't reuse parse_u32)
            cache_size: get("JADE_CACHE_SIZE").and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_CACHE_SIZE),
        }
    }

    /// Reject configurations llama.cpp cannot run: a quantized V cache requires
    /// flash attention. Returns a human-readable error the daemon prints before
    /// exiting.
    pub fn validate(&self) -> Result<(), String> {
        // the one hard rule: a quantized V cache only works when flash attention
        // is on. If the user asked for both a quantized V type AND flash_attn
        // off, that's an impossible combo — refuse to boot with a message that
        // tells them exactly which two settings conflict and how to fix it.
        if self.kv_type_v.is_quantized() && !self.flash_attn {
            return Err(format!(
                "quantized V cache (kv_v={:?}) requires flash attention, but JADE_FLASH_ATTN \
                 is disabled. Either set JADE_KV_TYPE_V=f16 or remove JADE_FLASH_ATTN=off.",
                self.kv_type_v
            ));
        }
        // every other combination is legal
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn cfg(pairs: &[(&str, &str)]) -> Config {
        let map: HashMap<String, String> =
            pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        Config::from_getter(|k| map.get(k).cloned())
    }

    #[test]
    fn defaults_when_env_empty() {
        let c = cfg(&[]);
        assert_eq!(c.model_path, DEFAULT_MODEL_PATH);
        assert_eq!(c.ctx_size, DEFAULT_CTX_SIZE);
        assert_eq!(c.gpu_layers, DEFAULT_GPU_LAYERS);
        assert_eq!(c.cache_size, DEFAULT_CACHE_SIZE);
        assert_eq!(c.kv_type_k, DEFAULT_KV_TYPE);
        assert!(c.flash_attn);
    }

    #[test]
    fn overrides_parsed() {
        let c = cfg(&[
            ("JADE_MODEL_PATH", "/tmp/m.gguf"),
            ("JADE_CTX_SIZE", "8192"),
            ("JADE_KV_TYPE_V", "f16"),
        ]);
        assert_eq!(c.model_path, "/tmp/m.gguf");
        assert_eq!(c.ctx_size, 8192);
        assert_eq!(c.kv_type_v, KvType::F16);
    }

    #[test]
    fn unparseable_numbers_fall_back_to_default() {
        let c = cfg(&[("JADE_CTX_SIZE", "not-a-number")]);
        assert_eq!(c.ctx_size, DEFAULT_CTX_SIZE);
    }

    #[test]
    fn n_seq_max_default_and_override() {
        assert_eq!(cfg(&[]).n_seq_max, DEFAULT_N_SEQ_MAX);
        assert_eq!(cfg(&[("JADE_N_SEQ_MAX", "32")]).n_seq_max, 32);
    }

    #[test]
    fn flash_attn_disabled_by_falsey_values() {
        for v in ["0", "off", "false"] {
            assert!(!cfg(&[("JADE_FLASH_ATTN", v)]).flash_attn, "{v} should disable");
        }
        assert!(cfg(&[("JADE_FLASH_ATTN", "1")]).flash_attn);
    }

    #[test]
    fn unknown_kv_type_falls_back_to_q8_0() {
        assert_eq!(KvType::parse("wat"), KvType::Q8_0);
        assert!(!KvType::is_known("wat"));
        assert!(KvType::is_known("Q4_0"));
    }

    #[test]
    fn quantized_v_without_flash_attn_is_rejected() {
        let c = cfg(&[("JADE_KV_TYPE_V", "q4_0"), ("JADE_FLASH_ATTN", "off")]);
        assert!(c.validate().is_err());
    }

    #[test]
    fn quantized_v_with_flash_attn_is_ok() {
        let c = cfg(&[("JADE_KV_TYPE_V", "q4_0")]);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn f16_v_without_flash_attn_is_ok() {
        let c = cfg(&[("JADE_KV_TYPE_V", "f16"), ("JADE_FLASH_ATTN", "off")]);
        assert!(c.validate().is_ok());
    }
}
