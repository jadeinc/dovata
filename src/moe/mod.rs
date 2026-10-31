//! MoE expert ranking: which experts each layer of a mixture-of-experts model uses most.
//!
//! Loads and validates the per-layer `expert_order` asset, a usage-ranked permutation of
//! expert ids made offline from routing counts. In 3.0 the planner will read it to decide
//! which experts stay in VRAM when a MoE model does not fit there whole. That case is
//! deferred, so nothing calls this yet.
//!
//! Up to 2.x this module was the Rust side of a C++ and CUDA pager that moved cold experts
//! into VRAM while the model ran, and it flattened the ranking for that driver over FFI.
//! The pager went with the rest of the runtime; see `docs/moe-expert-offload.md` and
//! `main:src/moe/driver/`. The validation stayed, because it is the part that is easy to
//! get wrong without noticing.
//!
//! The capability is behind the `moe` cargo feature, so a build without it proves nothing
//! else depends on the ranking.

use serde_json::Value;

/// The expert ranking for every layer: `order[layer]` is the expert ids of that
/// layer, hottest-first. Pinned = the first `pin`, cache = the next `cache`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertOrder {
    pub n_layers: usize,
    pub n_experts: usize,
    /// Row-major flatten: `flat[layer * n_experts + rank]`.
    flat: Vec<i32>,
}

impl ExpertOrder {
    /// Parse and validate the asset JSON. Rejects wrong layer or expert counts, and any
    /// layer that is not an exact permutation of `0..n_experts`: a non-permutation leaves
    /// one expert unreachable and places another twice, and the model still answers,
    /// worse, with nothing logged.
    ///
    /// Parsed via `serde_json::Value` because the asset is tiny and the validation, not
    /// the deserialization, is the part that matters.
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let v: Value = serde_json::from_slice(bytes).map_err(|e| format!("expert_order parse: {e}"))?;
        let usize_field = |k: &str| -> Result<usize, String> {
            v.get(k)
                .and_then(Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| format!("expert_order: missing/invalid '{k}'"))
        };
        let n_layers = usize_field("n_layers")?;
        let n_experts = usize_field("n_experts")?;
        let layers = v
            .get("expert_order")
            .and_then(Value::as_array)
            .ok_or_else(|| "expert_order: missing 'expert_order' array".to_string())?;
        if layers.len() != n_layers {
            return Err(format!("expert_order has {} layers, header says {n_layers}", layers.len()));
        }
        let mut flat = Vec::with_capacity(n_layers * n_experts);
        // `seen` is reused per layer to check the permutation property in O(n).
        let mut seen = vec![false; n_experts];
        for (li, layer) in layers.iter().enumerate() {
            let row = layer
                .as_array()
                .ok_or_else(|| format!("layer {li}: not an array"))?;
            if row.len() != n_experts {
                return Err(format!("layer {li}: {} experts, header says {n_experts}", row.len()));
            }
            seen.iter_mut().for_each(|s| *s = false);
            for cell in row {
                let e = cell.as_i64().ok_or_else(|| format!("layer {li}: non-integer expert id"))?;
                if e < 0 || e as usize >= n_experts {
                    return Err(format!("layer {li}: expert id {e} out of range"));
                }
                if seen[e as usize] {
                    return Err(format!("layer {li}: expert id {e} appears twice"));
                }
                seen[e as usize] = true;
                flat.push(e as i32);
            }
            // every id in range + no duplicates + correct count ⇒ exact permutation
        }
        Ok(ExpertOrder { n_layers, n_experts, flat })
    }

    /// Load from a file path.
    pub fn load(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("expert_order read {path}: {e}"))?;
        Self::from_json(&bytes)
    }

    /// Row-major `[n_layers * n_experts]` slice: `flat[layer * n_experts + rank]`.
    pub fn flat(&self) -> &[i32] {
        &self.flat
    }

    /// The hottest-first expert ids for one layer.
    pub fn layer(&self, layer: usize) -> &[i32] {
        let base = layer * self.n_experts;
        &self.flat[base..base + self.n_experts]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal valid asset: 2 layers, 4 experts, distinct permutations.
    const GOOD: &str = r#"{"n_layers":2,"n_experts":4,
        "expert_order":[[3,1,0,2],[0,2,1,3]]}"#;

    #[test]
    fn parses_and_flattens_row_major() {
        let o = ExpertOrder::from_json(GOOD.as_bytes()).unwrap();
        assert_eq!(o.n_layers, 2);
        assert_eq!(o.n_experts, 4);
        assert_eq!(o.flat(), &[3, 1, 0, 2, 0, 2, 1, 3]);
        assert_eq!(o.layer(0), &[3, 1, 0, 2]);
        assert_eq!(o.layer(1), &[0, 2, 1, 3]);
    }

    #[test]
    fn rejects_wrong_layer_count() {
        let j = r#"{"n_layers":3,"n_experts":4,"expert_order":[[0,1,2,3]]}"#;
        assert!(ExpertOrder::from_json(j.as_bytes()).is_err());
    }

    #[test]
    fn rejects_wrong_expert_count() {
        let j = r#"{"n_layers":1,"n_experts":4,"expert_order":[[0,1,2]]}"#;
        assert!(ExpertOrder::from_json(j.as_bytes()).is_err());
    }

    #[test]
    fn rejects_duplicate_expert() {
        let j = r#"{"n_layers":1,"n_experts":4,"expert_order":[[0,1,1,3]]}"#;
        assert!(ExpertOrder::from_json(j.as_bytes()).unwrap_err().contains("twice"));
    }

    #[test]
    fn rejects_out_of_range_expert() {
        let j = r#"{"n_layers":1,"n_experts":4,"expert_order":[[0,1,2,9]]}"#;
        assert!(ExpertOrder::from_json(j.as_bytes()).unwrap_err().contains("out of range"));
    }

    // The real shipped asset must load and be a valid 48×128 permutation set.
    #[test]
    fn real_asset_is_valid() {
        // CARGO_MANIFEST_DIR is the repo root: this crate's package root is the root,
        // and its capabilities are the children of src/.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/moe/assets/qwen3-coder-30b.expert-order.json"
        );
        let o = ExpertOrder::load(path).expect("ship asset loads");
        assert_eq!(o.n_layers, 48);
        assert_eq!(o.n_experts, 128);
        assert_eq!(o.flat().len(), 48 * 128);
    }
}
