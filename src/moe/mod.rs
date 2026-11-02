//! Author: Joe Ricks
//! File: src/moe/mod.rs
//! Description: Loads and checks the ranking of each MoE layer's experts by use.

use serde_json::Value;

/// Purpose: This struct holds the expert ranking of every layer of a MoE model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertOrder {
    pub n_layers: usize,
    pub n_experts: usize,
    /// Row-major flatten: `flat[layer * n_experts + rank]`.
    flat: Vec<i32>,
}

impl ExpertOrder {
    /// Purpose: This function parses an `expert_order` asset from JSON and checks it.
    ///
    /// Inputs:
    /// - `bytes`: the asset's JSON text.
    ///
    /// Outputs: the [`ExpertOrder`], or an error when a count is wrong or a layer is not
    /// an exact permutation.
    pub fn from_json(
        bytes: &[u8],
    ) -> Result<Self, String>
    {
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

    /// Purpose: This function reads an `expert_order` asset file and checks it.
    ///
    /// Inputs:
    /// - `path`: the asset file.
    ///
    /// Outputs: the [`ExpertOrder`], or an error.
    pub fn load(
        path: &str,
    ) -> Result<Self, String>
    {
        let bytes = std::fs::read(path).map_err(|e| format!("expert_order read {path}: {e}"))?;
        Self::from_json(&bytes)
    }

    /// Purpose: This function returns every layer's expert ranking as one flat list,
    /// layer after layer.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `n_layers * n_experts` expert ids.
    pub fn flat(&self) -> &[i32]
    {
        &self.flat
    }

    /// Purpose: This function returns one layer's expert ids, most used first.
    ///
    /// Inputs:
    /// - `layer`: the layer's index, from 0.
    ///
    /// Outputs: the layer's expert ids; panics when `layer` is out of range.
    pub fn layer(
        &self,
        layer: usize,
    ) -> &[i32]
    {
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
    fn parses_and_flattens_row_major()
    {
        let o = ExpertOrder::from_json(GOOD.as_bytes()).unwrap();
        assert_eq!(o.n_layers, 2);
        assert_eq!(o.n_experts, 4);
        assert_eq!(o.flat(), &[3, 1, 0, 2, 0, 2, 1, 3]);
        assert_eq!(o.layer(0), &[3, 1, 0, 2]);
        assert_eq!(o.layer(1), &[0, 2, 1, 3]);
    }

    #[test]
    fn rejects_wrong_layer_count()
    {
        let j = r#"{"n_layers":3,"n_experts":4,"expert_order":[[0,1,2,3]]}"#;
        assert!(ExpertOrder::from_json(j.as_bytes()).is_err());
    }

    #[test]
    fn rejects_wrong_expert_count()
    {
        let j = r#"{"n_layers":1,"n_experts":4,"expert_order":[[0,1,2]]}"#;
        assert!(ExpertOrder::from_json(j.as_bytes()).is_err());
    }

    #[test]
    fn rejects_duplicate_expert()
    {
        let j = r#"{"n_layers":1,"n_experts":4,"expert_order":[[0,1,1,3]]}"#;
        assert!(ExpertOrder::from_json(j.as_bytes()).unwrap_err().contains("twice"));
    }

    #[test]
    fn rejects_out_of_range_expert()
    {
        let j = r#"{"n_layers":1,"n_experts":4,"expert_order":[[0,1,2,9]]}"#;
        assert!(ExpertOrder::from_json(j.as_bytes()).unwrap_err().contains("out of range"));
    }

    // The real shipped asset must load and be a valid 48×128 permutation set.
    #[test]
    fn real_asset_is_valid()
    {
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
