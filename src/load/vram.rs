//! Sizing the generation cap from what VRAM is actually left after the weights.
//!
//! A Jade program cannot say how long a reply may be. `max_tokens` went with the
//! daemon, and `InferRequest` carries only `input`, `grammar`, `anchor` and
//! `stop_anchor`, so the engine has to pick the number itself. The first attempt was
//! a flat 2048, which is a guess dressed as a default: too small on a card with room
//! to spare, and too large on one already full of weights.
//!
//! The number that actually constrains a reply is how much KV cache fits. Weights
//! are resident for the life of the process, so whatever they do not occupy is what
//! is left to hold tokens, and each token costs a fixed number of bytes decided by
//! the model's shape and the cache's element type.
//!
//! ```text
//!   headroom      = vram - weights - overhead
//!   bytes/token   = layers * kv_heads * (key_dim * k_elem_bytes + value_dim * v_elem_bytes)
//!   capacity      = headroom / bytes/token
//! ```
//!
//! Everything here is arithmetic over numbers read from the GGUF header, so it is
//! pure and unit-tested on any machine — no CUDA, no model, no llama.cpp. The engine
//! still clamps the result to what the context window leaves after the prompt, so
//! this is an upper bound rather than a promise.
//!
//! ## What this deliberately does not model
//!
//! The estimate assumes the weights are resident in VRAM. With MoE expert offload
//! that is false and pessimistic: cold experts live in host RAM and only a pinned
//! working set sits on the card, so the file size overstates VRAM use — sometimes by
//! more than half. A model larger than the budget therefore does not mean "will not
//! run"; it means this estimator cannot see how much of it is resident, and it falls
//! back rather than returning zero. Set `JADE_MAX_TOKENS` when the offload driver is
//! doing the real work and this arithmetic no longer describes the machine.

use crate::load::config::KvType;

impl ModelDims {
    /// Read a model's shape out of its GGUF header.
    ///
    /// Keys are namespaced by architecture (`qwen3moe.block_count`), so the
    /// architecture is read first and the rest built from it. `head_count_kv` is
    /// absent on models without grouped-query attention and falls back to
    /// `head_count`, which is what that absence means.
    ///
    /// `None` when a required key is missing — the same condition preflight already
    /// refuses a load for, so in practice this is reached only with an override.
    #[must_use]
    pub fn from_gguf(model: &gguf::Gguf) -> Option<Self> {
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

/// Assumed VRAM when nothing says otherwise, in bytes. 16 GiB is the smaller of the
/// two cards this engine targets, so guessing it is the safe direction: a 24 GiB card
/// gets a conservative cap rather than an over-commit that fails at allocation.
pub const DEFAULT_VRAM_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// Held back for the CUDA context, compute buffers, and fragmentation. Not a
/// measured figure — a deliberately generous slice, because over-committing VRAM
/// fails at allocation time with an error that names none of this.
pub const DEFAULT_OVERHEAD_BYTES: u64 = 1024 * 1024 * 1024;

/// Floor for the returned cap. Below roughly this a reply is too short to be worth
/// serving, so returning it would turn a sizing problem into a mysteriously truncated
/// answer. Hitting the floor means the arithmetic no longer describes the machine —
/// see the MoE note above.
pub const MIN_MAX_TOKENS: u32 = 256;

/// Ceiling for the returned cap. A card with room for a million tokens of KV does not
/// mean a single reply should run that long; past this, a runaway generation costs
/// more than it can be worth, and a caller who genuinely wants more sets the override.
pub const MAX_MAX_TOKENS: u32 = 8192;

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
    /// each `head_dim` elements wide. They are quantized *independently* —
    /// `JADE_KV_TYPE_K` and `JADE_KV_TYPE_V` are separate settings — so this takes
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

/// What the card has, and what is already spoken for.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub vram_bytes: u64,
    pub weights_bytes: u64,
    pub overhead_bytes: u64,
}

impl Budget {
    /// Bytes left for KV cache, or `None` when the weights and overhead already
    /// exceed the card. `None` is not "will not run" — see the MoE note above.
    #[must_use]
    pub fn headroom(&self) -> Option<u64> {
        self.vram_bytes
            .checked_sub(self.weights_bytes)?
            .checked_sub(self.overhead_bytes)
    }
}

/// How many tokens of KV cache fit in the headroom, clamped to a usable range.
///
/// Returns `MIN_MAX_TOKENS` when the model does not visibly fit, rather than zero: a
/// zero cap generates an empty reply, which reads as the model saying nothing instead
/// of as a sizing problem. That exact failure is what prompted this module.
#[must_use]
pub fn estimate_max_tokens(dims: ModelDims, kv_k: KvType, kv_v: KvType, budget: Budget) -> u32 {
    let per_token = dims.kv_bytes_per_token(kv_k, kv_v);
    if per_token == 0 {
        return MIN_MAX_TOKENS;
    }
    let Some(headroom) = budget.headroom() else {
        return MIN_MAX_TOKENS;
    };
    let capacity = headroom / per_token;
    let capped = u32::try_from(capacity).unwrap_or(u32::MAX);
    capped.clamp(MIN_MAX_TOKENS, MAX_MAX_TOKENS)
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
        // 8x for this model, and the resulting cap would be pointlessly small.
        let gqa = qwen();
        let no_gqa = ModelDims { kv_heads: gqa.heads, ..gqa };
        assert_eq!(
            no_gqa.kv_bytes_per_token(KvType::F16, KvType::F16) / gqa.kv_bytes_per_token(KvType::F16, KvType::F16),
            8
        );
    }

    #[test]
    fn kv_bytes_per_token_is_the_documented_formula() {
        // 2 * 48 layers * 4 kv heads * 64 head_dim = 24576 elements, f16 = 2 bytes.
        assert_eq!(qwen().kv_bytes_per_token(KvType::F16, KvType::F16), 49_152 * 2);
        // q4_0 is 18 bytes per 32 elements.
        assert_eq!(qwen().kv_bytes_per_token(KvType::Q4_0, KvType::Q4_0), 49_152 * 18 / 32);
    }

    #[test]
    fn a_quantized_cache_holds_more_than_an_unquantized_one() {
        let budget = Budget {
            vram_bytes: DEFAULT_VRAM_BYTES,
            weights_bytes: 8 * 1024 * 1024 * 1024,
            overhead_bytes: DEFAULT_OVERHEAD_BYTES,
        };
        let f16 = estimate_max_tokens(qwen(), KvType::F16, KvType::F16, budget);
        let q4 = estimate_max_tokens(qwen(), KvType::Q4_0, KvType::Q4_0, budget);
        assert!(q4 >= f16, "q4_0 ({q4}) should fit at least as many tokens as f16 ({f16})");
    }

    #[test]
    fn heavier_weights_leave_room_for_fewer_tokens() {
        let mk = |weights| Budget {
            vram_bytes: DEFAULT_VRAM_BYTES,
            weights_bytes: weights,
            overhead_bytes: DEFAULT_OVERHEAD_BYTES,
        };
        // Small enough that neither result is pinned to the ceiling.
        let dims = ModelDims { layers: 80, heads: 64, kv_heads: 64, embedding_length: 8192, key_length: None, value_length: None };
        let light = estimate_max_tokens(dims, KvType::F16, KvType::F16, mk(2 * 1024 * 1024 * 1024));
        let heavy = estimate_max_tokens(dims, KvType::F16, KvType::F16, mk(12 * 1024 * 1024 * 1024));
        assert!(light > heavy, "light={light} heavy={heavy}");
    }

    #[test]
    fn a_model_larger_than_the_card_falls_back_rather_than_returning_zero() {
        // The production case: an 18 GiB model against the assumed 16 GiB. Zero would
        // generate an empty reply and read as the model saying nothing.
        let budget = Budget {
            vram_bytes: DEFAULT_VRAM_BYTES,
            weights_bytes: 18 * 1024 * 1024 * 1024,
            overhead_bytes: DEFAULT_OVERHEAD_BYTES,
        };
        assert!(budget.headroom().is_none());
        assert_eq!(estimate_max_tokens(qwen(), KvType::F16, KvType::F16, budget), MIN_MAX_TOKENS);
    }

    #[test]
    fn the_result_stays_inside_the_usable_range() {
        // A tiny model on a big card would otherwise return a cap in the millions.
        let budget = Budget {
            vram_bytes: 80 * 1024 * 1024 * 1024,
            weights_bytes: 1024 * 1024 * 1024,
            overhead_bytes: DEFAULT_OVERHEAD_BYTES,
        };
        let dims = ModelDims { layers: 4, heads: 8, kv_heads: 1, embedding_length: 512, key_length: None, value_length: None };
        assert_eq!(estimate_max_tokens(dims, KvType::Q4_0, KvType::Q4_0, budget), MAX_MAX_TOKENS);
    }

    #[test]
    fn dims_come_out_of_a_real_gguf_header() {
        // Built the way preflight's own fixtures build one, so the key names here are
        // the key names a real file carries.
        let bytes = crate::load::preflight::tests::header("qwen3moe", Some("Qwen3-Coder-30B"));
        let model = gguf::Gguf::parse(&bytes).expect("fixture parses");
        let dims = ModelDims::from_gguf(&model).expect("dims readable");
        assert_eq!(dims.layers, 48);
        assert_eq!(dims.heads, 32);
        assert_eq!(dims.embedding_length, 2048);
        // The fixture declares no head_count_kv, so it must fall back to head_count
        // rather than to zero — a zero here would divide the per-token cost away.
        assert_eq!(dims.kv_heads, 32);
    }

    /// The whole estimate is checkable against a number llama.cpp prints, so check it.
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
    fn the_estimate_matches_llama_cpps_own_kv_buffer() {
        const OBSERVED_KV_BYTES: u64 = 864 * 1024 * 1024;
        const OBSERVED_CTX: u64 = 32768;
        assert_eq!(
            qwen().kv_bytes_per_token(KvType::Q4_0, KvType::Q4_0),
            OBSERVED_KV_BYTES / OBSERVED_CTX,
        );
    }

    #[test]
    fn keys_and_values_are_costed_separately() {
        // JADE_KV_TYPE_K and JADE_KV_TYPE_V are independent settings, and a common
        // deployment quantizes the values harder than the keys. Doubling one type
        // would misprice that case in both directions.
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
        let budget = Budget {
            vram_bytes: DEFAULT_VRAM_BYTES,
            weights_bytes: 0,
            overhead_bytes: DEFAULT_OVERHEAD_BYTES,
        };
        assert_eq!(estimate_max_tokens(dims, KvType::F16, KvType::F16, budget), MIN_MAX_TOKENS);
    }
}
