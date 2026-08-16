//! MoE expert offload — the whole capability's Rust side.
//!
//! Loads and validates the per-layer `expert_order` asset (the usage-ranked
//! expert permutation produced offline; see `docs/moe-expert-offload.md`) and
//! flattens it for hand-off to the C `moe_driver_init` over FFI. Keeping the
//! parse + validation here means the C++ driver never touches JSON, and the
//! bug-prone "is this a valid permutation" check is unit-tested on any platform.
//!
//! This file is the pure half and builds anywhere. [`ffi`] is the thin
//! declaration over the C driver, Linux-only. The paging machinery itself is
//! C++ and CUDA in `src/infer-engine/moe-driver/`, compiled into libllama by the
//! driver's CMake wiring.
//!
//! The whole capability is behind the `moe` cargo feature. Turning it off drops
//! this module, the FFI, and the llama.cpp seam together — and makes a config
//! that asks for offload fail the load rather than run stock, because a driver
//! that quietly did not engage looks exactly like one that did, at three times
//! the VRAM.

#[cfg(target_os = "linux")]
pub(crate) mod ffi;

use serde_json::Value;

/// The expert ranking for every layer: `order[layer]` is the expert ids of that
/// layer, hottest-first. Pinned = the first `pin`, cache = the next `cache`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertOrder {
    pub n_layers: usize,
    pub n_experts: usize,
    /// Row-major flatten: `flat[layer * n_experts + rank]`. Ready for FFI.
    flat: Vec<i32>,
}

impl ExpertOrder {
    /// Parse + validate the asset JSON. Rejects anything the driver couldn't
    /// safely consume: wrong layer/expert counts, or a layer that isn't an exact
    /// permutation of `0..n_experts` (a non-permutation would mean some expert is
    /// unreachable and another double-mapped — silent wrong answers).
    ///
    /// Parsed via `serde_json::Value` (no serde-derive dep): the asset is tiny
    /// and the validation, not deserialization, is the load-bearing part.
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

    /// Load from a file path (the `JADE_MOE_ORDER_PATH` value).
    pub fn load(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("expert_order read {path}: {e}"))?;
        Self::from_json(&bytes)
    }

    /// Row-major `[n_layers * n_experts]` slice for `moe_driver_init` over FFI.
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
            "/src/model/profiles/qwen3-coder-30b.expert-order.json"
        );
        let o = ExpertOrder::load(path).expect("ship asset loads");
        assert_eq!(o.n_layers, 48);
        assert_eq!(o.n_experts, 128);
        assert_eq!(o.flat().len(), 48 * 128);
    }
}
