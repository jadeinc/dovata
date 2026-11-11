//! Author: Joe Ricks
//! File: src/memplan/graph.rs
//! Description: Plans a model from its layout: binds the layout, works out each tensor's
//! lifetime from op order, places every tensor, and puts the ops into the plan.

use std::collections::BTreeMap;

use serde::Serialize;

use super::address::{place, Plan, PlanError, PlanTensor, Role};
use crate::layout::{BindOptions, BoundGraph, Layout, Param, TensorKind};
use crate::load::Model;
use crate::target::Target;
use crate::tokenize::TokenizerPlan;

/// Purpose: This struct holds the forward step a plan's tensors belong to: its ops in
/// run order, by tensor name, so the Jade compiler can generate one kernel per op.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Step {
    /// `general.architecture`.
    pub arch: String,
    /// The step's sizes and KV cache types.
    pub options: BindOptions,
    /// How many times the layout's block ran.
    pub n_layer: u32,
    /// The tensors the program reads after the step, such as `result_output`.
    pub outputs: Vec<String>,
    /// Every op, in run order.
    pub ops: Vec<StepOp>,
}

/// Purpose: This struct holds one ggml op of a planned step, with its tensors named as in
/// the plan's address table.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StepOp {
    /// The ggml op in lower case, such as `mul_mat`.
    pub op: String,
    /// Input tensor names, in ggml API argument order.
    pub inputs: Vec<String>,
    /// Output tensor names. Every ggml op has exactly one today.
    pub outputs: Vec<String>,
    /// The layer, for an op of the layout's block.
    pub layer: Option<u32>,
    /// Every parameter as a concrete value, as [`crate::layout::BoundOp`] holds it.
    pub params: BTreeMap<String, Param>,
}

/// Purpose: This function plans a model on one target from its layout.
///
/// Inputs:
/// - `model`: the opened GGUF model.
/// - `layout`: the model's layout, from [`crate::layout::for_model`].
/// - `opts`: the step's sizes and KV cache types.
/// - `target`: the machine to plan for.
/// - `tokenizer`: the model's tokenizer, from [`crate::tokenize::describe`], which the
///   plan carries and its table hash covers. `None` plans without one.
///
/// Outputs: the [`Plan`] with its step and tokenizer, or a [`PlanError`] saying why the
/// model cannot be bound or placed.
pub fn plan(
    model: &Model,
    layout: &Layout,
    opts: &BindOptions,
    target: &Target,
    tokenizer: Option<&TokenizerPlan>,
) -> Result<Plan, PlanError>
{
    let graph = layout.bind(model, opts).map_err(PlanError::Layout)?;
    plan_graph(model, &graph, target, tokenizer)
}

/// Purpose: This function plans an already bound layout on one target.
///
/// Inputs:
/// - `model`: the model the graph was bound to.
/// - `graph`: the bound forward step.
/// - `target`: the machine to plan for.
/// - `tokenizer`: the model's tokenizer, or `None`.
///
/// Outputs: the [`Plan`] with its step and tokenizer, or a [`PlanError`].
pub fn plan_graph(
    model: &Model,
    graph: &BoundGraph,
    target: &Target,
    tokenizer: Option<&TokenizerPlan>,
) -> Result<Plan, PlanError>
{
    check_weights(model, graph)?;
    let others = tensors_to_place(graph)?;
    let mut plan = place(model, &others, target)?;
    let name = |id: &crate::layout::TensorId| graph.tensor(*id).name.clone();
    plan.step = Some(Step {
        arch: graph.arch.clone(),
        options: graph.options,
        n_layer: graph.n_layer,
        outputs: graph.outputs.iter().map(name).collect(),
        ops: graph
            .ops
            .iter()
            .map(|op| StepOp {
                op: op.op.to_owned(),
                inputs: op.inputs.iter().map(name).collect(),
                outputs: op.outputs.iter().map(name).collect(),
                layer: op.layer,
                params: op.params.clone(),
            })
            .collect(),
    });
    plan.tokenizer = tokenizer.cloned();
    plan.seal(model)?;
    Ok(plan)
}

/// Purpose: This function checks that a bound graph's weights are the model's tensors, in
/// file order.
///
/// Inputs:
/// - `model`: the model.
/// - `graph`: the graph bound to it.
///
/// Outputs: `Ok(())`, or [`PlanError::Invariant`] when the graph was bound to another
/// model.
fn check_weights(
    model: &Model,
    graph: &BoundGraph,
) -> Result<(), PlanError>
{
    let mismatch = |detail: String| PlanError::Invariant(format!("the bound graph does not match the model: {detail}"));
    if graph.arch != model.architecture() {
        return Err(mismatch(format!("it is for {}, the model is {}", graph.arch, model.architecture())));
    }
    for (i, t) in model.tensors().enumerate() {
        match graph.tensors.get(i) {
            Some(g) if g.name == t.info.name && matches!(g.kind, TensorKind::Weight { .. }) => {}
            _ => return Err(mismatch(format!("tensor {i} should be the weight {}", t.info.name))),
        }
    }
    Ok(())
}

/// Purpose: This function lists a bound graph's non-weight tensors with the role and
/// lifetime the planner needs.
///
/// Inputs:
/// - `graph`: the bound forward step.
///
/// Outputs: one [`PlanTensor`] per input, activation, KV, state and view tensor, in graph
/// order, or [`PlanError::Invariant`] when the graph has more ops than fit in a `u32`.
fn tensors_to_place(
    graph: &BoundGraph,
) -> Result<Vec<PlanTensor>, PlanError>
{
    let n_ops = u32::try_from(graph.ops.len()).map_err(|_| PlanError::Invariant("more than 2^32 ops".into()))?;
    let end = n_ops.saturating_sub(1);

    // A tensor is alive from the first op that touches its memory to the last. A view
    // touches the memory of the tensor it views, so its uses count for that tensor.
    let mut first: Vec<Option<u32>> = vec![None; graph.tensors.len()];
    let mut last: Vec<Option<u32>> = vec![None; graph.tensors.len()];
    for (i, op) in (0u32..).zip(&graph.ops) {
        for id in op.inputs.iter().chain(&op.outputs) {
            let s = graph.storage(*id).index();
            first[s] = Some(first[s].map_or(i, |f| f.min(i)));
            last[s] = Some(last[s].map_or(i, |l| l.max(i)));
        }
    }
    // The program reads the outputs after the step, so their memory stays alive to the
    // end of it.
    let mut is_output = vec![false; graph.tensors.len()];
    for (i, t) in graph.tensors.iter().enumerate() {
        if t.output || graph.outputs.iter().any(|o| o.index() == i) {
            let s = graph.storage(crate::layout::TensorId(i as u32)).index();
            is_output[s] = true;
            last[s] = Some(end);
        }
    }

    let mut out = Vec::new();
    for (i, t) in graph.tensors.iter().enumerate() {
        let role = match &t.kind {
            TensorKind::Weight { .. } => continue,
            TensorKind::Input => Role::Input,
            TensorKind::Activation if is_output[i] => Role::Output,
            TensorKind::Activation => Role::Scratch,
            TensorKind::Kv => Role::Kv,
            TensorKind::State => Role::State,
            TensorKind::View { of, byte_offset, nb } => {
                Role::View { of: graph.tensor(*of).name.clone(), byte_offset: *byte_offset, nb: *nb }
            }
        };
        // Inputs are written before the step starts.
        let first = if role == Role::Input { 0 } else { first[i].unwrap_or(0) };
        out.push(PlanTensor {
            name: t.name.clone(),
            role,
            dims: t.dims.clone(),
            ggml_type: t.ggml_type,
            bytes: t.bytes,
            first,
            last: last[i].unwrap_or(first).max(first),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::testutil::{qwen3moe, SMALL};
    use crate::layout::{builtin, for_model};
    use crate::load::KvType;
    use crate::target::RegionKind;

    const F16: u32 = 1;

    /// Purpose: This function returns small bind options for the test models.
    ///
    /// Inputs:
    /// - `kv`: the K and V cache type.
    ///
    /// Outputs: the options.
    fn opts(
        kv: KvType,
    ) -> BindOptions
    {
        BindOptions { n_ctx: 256, n_tokens: 8, n_outputs: 1, kv_k: kv, kv_v: kv }
    }

    /// Purpose: This function plans the small synthetic qwen3moe model on one target.
    ///
    /// Inputs:
    /// - `target`: the machine to plan for.
    ///
    /// Outputs: the model, its bound graph and the plan.
    fn small(
        target: &Target,
    ) -> (Model, BoundGraph, Plan)
    {
        let model = qwen3moe(&SMALL, F16, false).model();
        let layout = builtin("qwen3moe").unwrap();
        let graph = layout.bind(&model, &opts(KvType::Q8_0)).unwrap();
        let plan = plan_graph(&model, &graph, target, None).unwrap();
        (model, graph, plan)
    }

    #[test]
    fn every_bound_tensor_gets_one_entry()
    {
        let (_, graph, plan) = small(&Target::apple(1 << 30, 1 << 30));
        assert_eq!(plan.entries.len(), graph.tensors.len());
        for t in &graph.tensors {
            assert!(plan.entry(&t.name).is_some(), "{} has no entry", t.name);
        }
    }

    #[test]
    fn the_step_names_only_planned_tensors()
    {
        let (_, graph, plan) = small(&Target::cpu(1 << 30));
        let step = plan.step.as_ref().unwrap();
        assert_eq!(step.ops.len(), graph.ops.len());
        assert_eq!(step.outputs, ["result_output"]);
        for op in &step.ops {
            for n in op.inputs.iter().chain(&op.outputs) {
                assert!(plan.entry(n).is_some(), "{} reads or writes {n}, which has no entry", op.op);
            }
        }
    }

    #[test]
    fn outputs_live_to_the_end_and_inputs_from_the_start()
    {
        let (_, graph, plan) = small(&Target::cuda(1 << 30));
        let end = u32::try_from(graph.ops.len() - 1).unwrap();
        let out = plan.entry("result_output").unwrap();
        assert_eq!(out.role, Role::Output);
        assert_eq!(out.live.unwrap().1, end);
        let tokens = plan.entry("inp_tokens").unwrap();
        assert_eq!(tokens.live.unwrap().0, 0);
    }

    #[test]
    fn kv_is_persistent_and_scratch_reuses_memory()
    {
        let (_, graph, plan) = small(&Target::apple(1 << 30, 1 << 30));
        let kv = plan.entries.iter().filter(|e| e.role == Role::Kv).count();
        assert_eq!(kv, 2 * graph.n_layer as usize);
        assert!(plan.entries.iter().filter(|e| e.role == Role::Kv).all(|e| e.live.is_none()));
        let scratch: u64 = plan.entries.iter().filter(|e| matches!(e.role, Role::Scratch | Role::Input | Role::Output)).map(|e| e.bytes).sum();
        let arena = plan.regions.last().unwrap();
        assert_eq!(arena.kind, RegionKind::Unified);
        assert!(arena.bytes < scratch / 2, "arena {} for {} bytes of step tensors", arena.bytes, scratch);
    }

    #[test]
    fn views_sit_inside_the_tensor_they_view()
    {
        let (_, graph, plan) = small(&Target::cuda(1 << 30));
        for t in &graph.tensors {
            if let TensorKind::View { of, byte_offset, .. } = t.kind {
                let (v, base) = (plan.entry(&t.name).unwrap(), plan.entry(&graph.tensor(of).name).unwrap());
                assert_eq!((v.region, v.offset), (base.region, base.offset + byte_offset), "{}", t.name);
            }
        }
    }

    #[test]
    fn planning_is_deterministic()
    {
        let a = small(&Target::apple(1 << 30, 1 << 30)).2;
        let b = small(&Target::apple(1 << 30, 1 << 30)).2;
        assert_eq!(a.to_json(), b.to_json());
        assert_eq!(a.table_hash, b.table_hash);
    }

    #[test]
    fn a_layout_error_comes_back_as_a_plan_error()
    {
        let model = qwen3moe(&SMALL, F16, false).model();
        let layout = builtin("qwen3moe").unwrap();
        let err = plan(&model, &layout, &BindOptions { n_tokens: 0, ..opts(KvType::F16) }, &Target::cpu(1 << 30), None).unwrap_err();
        assert!(matches!(err, PlanError::Layout(_)), "{err:?}");
    }

    #[test]
    fn a_graph_bound_to_another_model_is_refused()
    {
        let model = qwen3moe(&SMALL, F16, false).model();
        let tied = qwen3moe(&SMALL, F16, true).model();
        let graph = builtin("qwen3moe").unwrap().bind(&tied, &opts(KvType::F16)).unwrap();
        assert!(matches!(plan_graph(&model, &graph, &Target::cpu(1 << 30), None), Err(PlanError::Invariant(_))));
    }

    /// Plans a real model on all three targets when `DOVATA_TEST_HEADER` names a GGUF
    /// file or the first tens of megabytes of one.
    #[test]
    fn a_real_header_plans_on_every_target()
    {
        let Ok(path) = std::env::var("DOVATA_TEST_HEADER") else {
            eprintln!("skipped: set DOVATA_TEST_HEADER to a GGUF header to plan a real model");
            return;
        };
        let model = Model::open(std::path::Path::new(&path)).unwrap_or_else(|e| panic!("{path}: {e}"));
        let layout = for_model(&model, None).unwrap_or_else(|e| panic!("{e}"));
        let o = BindOptions { n_ctx: 32768, n_tokens: 512, n_outputs: 1, kv_k: KvType::Q8_0, kv_v: KvType::Q8_0 };
        for target in [Target::apple(1 << 40, 1 << 34), Target::cpu(1 << 40), Target::cuda(1 << 40)] {
            let p = plan(&model, &layout, &o, &target, None).unwrap_or_else(|e| panic!("{path} on {target:?}: {e}"));
            eprintln!("{path} on {:?}: {} regions, {} entries, {:?}", target.site(), p.regions.len(), p.entries.len(), p.totals);
        }
    }
}
