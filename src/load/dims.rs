//! A model's shape, read from its GGUF header, and what one token of KV cache costs.
//!
//! The planner sizes the KV cache from these numbers: one key and one value entry per KV
//! head per layer for every token of context. The formula is
//!
//! ```text
//!   bytes/token = layers * kv_heads * (key_dim * k_elem_bytes + value_dim * v_elem_bytes)
//! ```
//!
//! It is arithmetic over numbers read from the header, so it is pure and unit-tested on
//! any machine. Up to 2.x this file also guessed a generation cap from the VRAM left
//! after the weights; 3.0 plans the KV cache exactly instead, so that estimate is gone.

/// The element type of the KV cache. The subset of `ggml_type` that llama.cpp accepts
/// for a K or V cache, with K and V chosen separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q5_1,
    Q5_0,
    Q4_1,
    Q4_0,
}

impl KvType {
    /// Every KV type, in the order a reader would list them.
    pub const ALL: [KvType; 8] = [
        KvType::F32,
        KvType::F16,
        KvType::BF16,
        KvType::Q8_0,
        KvType::Q5_1,
        KvType::Q5_0,
        KvType::Q4_1,
        KvType::Q4_0,
    ];

    /// Parse a name such as `q4_0`, in any case. `None` for anything else: 2.x fell back
    /// to `q8_0` on a typo, which hid the typo.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name().eq_ignore_ascii_case(s))
    }

    /// The lowercase name llama.cpp's `--cache-type-k` flag uses.
    pub fn name(self) -> &'static str {
        match self {
            KvType::F32 => "f32",
            KvType::F16 => "f16",
            KvType::BF16 => "bf16",
            KvType::Q8_0 => "q8_0",
            KvType::Q5_1 => "q5_1",
            KvType::Q5_0 => "q5_0",
            KvType::Q4_1 => "q4_1",
            KvType::Q4_0 => "q4_0",
        }
    }

    /// The `ggml_type` tag, the same numbering [`crate::gguf::TensorInfo::kind`] uses.
    pub fn ggml_type(self) -> u32 {
        match self {
            KvType::F32 => 0,
            KvType::F16 => 1,
            KvType::Q4_0 => 2,
            KvType::Q4_1 => 3,
            KvType::Q5_0 => 6,
            KvType::Q5_1 => 7,
            KvType::Q8_0 => 8,
            KvType::BF16 => 30,
        }
    }

    /// Whether the type is block-quantized. llama.cpp needs flash attention for a
    /// quantized V cache, so a layout without `flash_attn_ext` cannot use one.
    pub fn is_quantized(self) -> bool {
        !matches!(self, KvType::F16 | KvType::BF16 | KvType::F32)
    }
}

impl ModelDims {
    /// Read a model's shape out of its GGUF header.
    ///
    /// Keys are namespaced by architecture (`qwen3moe.block_count`), so the
    /// architecture is read first and the rest built from it. `head_count_kv` is
    /// absent on models without grouped-query attention and falls back to
    /// `head_count`, which is what that absence means.
    ///
    /// `None` when a required key is missing, the same condition preflight refuses.
    #[must_use]
    pub fn from_gguf(model: &crate::gguf::Gguf) -> Option<Self> {
        let arch = model.architecture()?;
        let u32_key = |suffix: &str| -> Option<u32> {
            model.u64(&format!("{arch}.{suffix}")).and_then(|v| u32::try_from(v).ok())
        };
        let heads = u32_key("attention.head_count")?;
        Some(ModelDims {
            layers: u32_key("block_count")?,
            heads,
            kv_heads: u32_key("attention.head_count_kv").unwrap_or(heads),
            embedding_length: u32_key("embedding_length")?,
            key_length: u32_key("attention.key_length"),
            value_length: u32_key("attention.value_length"),
        })
    }
}

/// Bytes per element in the KV cache.
///
/// The quantized types store blocks of 32 weights plus their scales, so the cost per
/// element is fractional: `Q4_0` is 16 bytes of nibbles plus one f16 scale over 32
/// elements, or 18/32. Returned scaled by 32 to stay in integer arithmetic — a
/// rounding error here is multiplied by every layer and every token.
#[must_use]
pub fn kv_elem_bytes_x32(kv: KvType) -> u64 {
    match kv {
        KvType::F32 => 4 * 32,
        KvType::F16 | KvType::BF16 => 2 * 32,
        KvType::Q8_0 => 34,  // 32 bytes + f16 scale
        KvType::Q5_1 => 24,  // 20 bytes + scale + min
        KvType::Q5_0 => 22,  // 20 bytes + scale
        KvType::Q4_1 => 20,  // 16 bytes + scale + min
        KvType::Q4_0 => 18,  // 16 bytes + scale
    }
}

/// The shape of a model, as read from its GGUF header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDims {
    /// `<arch>.block_count` — the number of transformer layers.
    pub layers: u32,
    /// `<arch>.attention.head_count` — query heads.
    pub heads: u32,
    /// `<arch>.attention.head_count_kv` — key/value heads. Equal to `heads` on a
    /// model without grouped-query attention, and much smaller with it. Using
    /// `heads` in its place would overstate the per-token cost several times over,
    /// which is the difference between a sensible cap and a pointlessly tiny one.
    pub kv_heads: u32,
    /// `<arch>.embedding_length`.
    pub embedding_length: u32,
    /// `<arch>.attention.key_length` — the width of one key head.
    ///
    /// Deriving this as `embedding_length / heads` is wrong, and wrong quietly. It
    /// happens to hold for many models and does not for Qwen3-Coder-30B, which
    /// declares 128 where the division gives 64 — so the estimate came out at exactly
    /// half of llama.cpp's own KV buffer. The header states the number; read it.
    ///
    /// `None` when the header omits it, in which case the division is the best
    /// available guess.
    pub key_length: Option<u32>,
    /// `<arch>.attention.value_length` — the width of one value head. Usually equal to
    /// `key_length`, but the header keeps them separate and so does this.
    pub value_length: Option<u32>,
}

impl ModelDims {
    /// Width of one head, when the header does not say. Only a fallback: see
    /// [`ModelDims::key_length`] for why the division cannot be trusted on its own.
    #[must_use]
    pub fn derived_head_dim(&self) -> u32 {
        if self.heads == 0 {
            return 0;
        }
        self.embedding_length / self.heads
    }

    /// Width of one key head — declared if the header declares it, derived otherwise.
    #[must_use]
    pub fn key_dim(&self) -> u32 {
        self.key_length.unwrap_or_else(|| self.derived_head_dim())
    }

    /// Width of one value head — declared if the header declares it, derived otherwise.
    #[must_use]
    pub fn value_dim(&self) -> u32 {
        self.value_length.unwrap_or_else(|| self.derived_head_dim())
    }

    /// Bytes of KV cache one token occupies, across every layer.
    ///
    /// The keys and values are two separate caches, one entry per KV head per layer,
    /// each `head_dim` elements wide. They are typed *independently*, so this takes
    /// both types and sums them rather than doubling one. A common deployment
    /// quantizes the values harder than the keys, and treating them as one type
    /// misestimates every figure downstream.
    #[must_use]
    pub fn kv_bytes_per_token(&self, kv_k: KvType, kv_v: KvType) -> u64 {
        let heads = u64::from(self.layers) * u64::from(self.kv_heads);
        // x32 numerator and denominator cancel; integer arithmetic throughout.
        let k = heads * u64::from(self.key_dim()) * kv_elem_bytes_x32(kv_k);
        let v = heads * u64::from(self.value_dim()) * kv_elem_bytes_x32(kv_v);
        (k + v) / 32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3-Coder-30B's shape, from its GGUF header: 48 layers, 32 query heads,
    /// 4 KV heads (grouped-query), 2048 embedding.
    fn qwen() -> ModelDims {
        ModelDims {
            layers: 48,
            heads: 32,
            kv_heads: 4,
            embedding_length: 2048,
            key_length: Some(128),
            value_length: Some(128),
        }
    }

    #[test]
    fn head_width_is_read_not_derived() {
        assert_eq!(qwen().key_dim(), 128);
        assert_eq!(qwen().derived_head_dim(), 64);
    }

    #[test]
    fn grouped_query_attention_is_not_ignored() {
        // Using query heads in place of KV heads would overstate the per-token cost
        // 8x for this model, and the planner would reserve 8x the KV memory.
        let gqa = qwen();
        let no_gqa = ModelDims { kv_heads: gqa.heads, ..gqa };
        assert_eq!(
            no_gqa.kv_bytes_per_token(KvType::F16, KvType::F16) / gqa.kv_bytes_per_token(KvType::F16, KvType::F16),
            8
        );
    }

    #[test]
    fn kv_bytes_per_token_is_the_documented_formula() {
        // 2 caches * 48 layers * 4 kv heads * 128 head_dim = 49152 elements, f16 = 2 bytes.
        assert_eq!(qwen().kv_bytes_per_token(KvType::F16, KvType::F16), 49_152 * 2);
        // q4_0 is 18 bytes per 32 elements.
        assert_eq!(qwen().kv_bytes_per_token(KvType::Q4_0, KvType::Q4_0), 49_152 * 18 / 32);
    }

    #[test]
    fn dims_come_out_of_a_real_gguf_header() {
        // Built the way preflight's own fixtures build one, so the key names here are
        // the key names a real file carries.
        let bytes = crate::load::preflight::tests::header("qwen3moe", Some("Qwen3-Coder-30B"));
        let model = crate::gguf::Gguf::parse(&bytes).expect("fixture parses");
        let dims = ModelDims::from_gguf(&model).expect("dims readable");
        assert_eq!(dims.layers, 48);
        assert_eq!(dims.heads, 32);
        assert_eq!(dims.embedding_length, 2048);
        // The fixture declares no head_count_kv, so it must fall back to head_count
        // rather than to zero — a zero here would divide the per-token cost away.
        assert_eq!(dims.kv_heads, 32);
    }

    /// The per-token cost is checkable against a number llama.cpp prints, so check it.
    ///
    /// Loading Qwen3-Coder-30B at `ctx_size=32768` with a q4_0 cache reports:
    ///
    /// ```text
    /// llama_kv_cache: CUDA0 KV buffer size = 864.00 MiB
    /// ```
    ///
    /// 864 MiB over 32768 tokens is 27648 bytes per token. This test failed at
    /// exactly half that until `key_length` was read from the header instead of
    /// derived from `embedding_length / heads` — the derivation gives 64 where this
    /// model declares 128. Being off by a clean factor of two is what made it
    /// findable; a subtler error would have looked plausible.
    #[test]
    fn the_per_token_cost_matches_llama_cpps_own_kv_buffer() {
        const OBSERVED_KV_BYTES: u64 = 864 * 1024 * 1024;
        const OBSERVED_CTX: u64 = 32768;
        assert_eq!(
            qwen().kv_bytes_per_token(KvType::Q4_0, KvType::Q4_0),
            OBSERVED_KV_BYTES / OBSERVED_CTX,
        );
    }

    #[test]
    fn keys_and_values_are_costed_separately() {
        // K and V are typed independently, and a common deployment quantizes the
        // values harder than the keys. Doubling one type would misprice that case in
        // both directions.
        let d = qwen();
        let both_f16 = d.kv_bytes_per_token(KvType::F16, KvType::F16);
        let mixed = d.kv_bytes_per_token(KvType::F16, KvType::Q4_0);
        let both_q4 = d.kv_bytes_per_token(KvType::Q4_0, KvType::Q4_0);
        assert!(both_q4 < mixed && mixed < both_f16, "{both_q4} < {mixed} < {both_f16}");
        // The mixed case is exactly the sum of one side of each.
        assert_eq!(mixed, both_f16 / 2 + both_q4 / 2);
    }

    #[test]
    fn a_malformed_header_does_not_divide_by_zero() {
        let dims = ModelDims { layers: 0, heads: 0, kv_heads: 0, embedding_length: 0, key_length: None, value_length: None };
        assert_eq!(dims.key_dim(), 0);
        assert_eq!(dims.kv_bytes_per_token(KvType::F16, KvType::F16), 0);
    }

    #[test]
    fn kv_types_parse_by_name_and_refuse_typos() {
        assert_eq!(KvType::parse("Q4_0"), Some(KvType::Q4_0));
        assert_eq!(KvType::parse("bf16"), Some(KvType::BF16));
        assert_eq!(KvType::parse("q4_O"), None, "a typo must not fall back silently");
        for k in KvType::ALL {
            assert_eq!(KvType::parse(k.name()), Some(k));
        }
    }

    #[test]
    fn kv_element_cost_matches_the_gguf_type_table() {
        // Two tables describe the same block formats; they must agree.
        for k in KvType::ALL {
            let (elems, bytes) = crate::gguf::ggml_type_layout(k.ggml_type()).unwrap();
            assert_eq!(kv_elem_bytes_x32(k), bytes * 32 / elems, "{}", k.name());
        }
    }
}
