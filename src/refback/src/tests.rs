//! Author: Joe Ricks
//! File: src/refback/src/tests.rs
//! Description: Runs refback on the fixture model's plans, true and tampered, on every
//! target this build covers.

use std::path::PathBuf;

use dovata::layout::{self, BindOptions};
use dovata::load::{KvType, Model};
use dovata::memplan::{Backing, Plan, Role};
use dovata::target::Target;

use crate::fixture::Fixture;
use crate::{quant, run, Check, Report, StartError};

/// Purpose: This function returns the small step the tests plan.
///
/// Inputs: none.
///
/// Outputs: 8 tokens, 1 with logits, a 64-token f16 KV cache.
fn opts() -> BindOptions
{
    BindOptions { n_ctx: 64, n_tokens: 8, n_outputs: 1, kv_k: KvType::F16, kv_v: KvType::F16 }
}

/// Purpose: This function lists the targets this build can check.
///
/// Inputs: none.
///
/// Outputs: the CPU, and with Metal an Apple target whose 48 KiB buffer limit splits the
/// fixture's weights into several windows.
fn targets() -> Vec<Target>
{
    #[allow(unused_mut)]
    let mut t = vec![Target::cpu(1 << 30)];
    #[cfg(all(feature = "metal", target_os = "macos"))]
    t.push(Target::Apple { unified_bytes: 1 << 30, max_buffer_bytes: 48 << 10, page_bytes: crate::metal::page_bytes() as u64 });
    t
}

/// Purpose: This function writes the fixture model and plans it on one target.
///
/// Inputs:
/// - `target`: the machine to plan for.
///
/// Outputs: the fixture, which must outlive the run, its part files and the plan.
fn planned(
    target: Target,
) -> (Fixture, Vec<PathBuf>, Plan)
{
    let f = Fixture::llama().unwrap();
    let model = Model::open(&f.path).unwrap();
    let layout = layout::for_model(&model, None).unwrap();
    let plan = dovata::memplan::plan(&model, &layout, &opts(), &target, None).unwrap();
    let parts = model.paths.clone();
    (f, parts, plan)
}

/// Purpose: This function finds an entry's index in a plan by tensor name.
///
/// Inputs:
/// - `plan`: the plan.
/// - `name`: the tensor's name.
///
/// Outputs: the index; panics when there is no such entry.
fn at(
    plan: &Plan,
    name: &str,
) -> usize
{
    plan.entries.iter().position(|e| e.tensor == name).unwrap_or_else(|| panic!("no entry {name}"))
}

/// Purpose: This function formats a report's failures for an assertion message.
///
/// Inputs:
/// - `r`: the report.
///
/// Outputs: one failure per line.
fn failures(
    r: &Report,
) -> String
{
    r.failures().iter().map(|o| o.to_string()).collect::<Vec<_>>().join("\n")
}

#[test]
fn a_true_plan_passes_every_check()
{
    for target in targets() {
        let (_f, parts, plan) = planned(target);
        let r = run(&parts, &plan).unwrap();
        assert!(r.passed(), "{target:?}:\n{}", failures(&r));
        let count = |c: Check| r.outcomes.iter().filter(|o| o.check == c).count();
        let weights = plan.entries.iter().filter(|e| e.role == Role::Weight).count();
        assert_eq!(count(Check::Identity), 1);
        assert_eq!(count(Check::Region), plan.regions.len());
        assert_eq!(count(Check::Bytes), weights);
        assert_eq!(count(Check::Sharing), 1);
        // Every 2-D weight is a covered type, and each type appears.
        let two_d: Vec<_> = plan.entries.iter().filter(|e| e.role == Role::Weight && e.dims.len() == 2).collect();
        assert_eq!(count(Check::Matvec), two_d.len());
        for ty in [quant::F32, quant::F16, quant::Q8_0, quant::Q4_0] {
            assert!(two_d.iter().any(|e| e.ggml_type == ty), "no 2-D weight of type {ty}");
        }
        if let Target::Apple { .. } = target {
            let windows = plan.regions.iter().filter(|r| matches!(r.backing, Backing::File { .. })).count();
            assert!(windows > 1, "the Apple test should map several windows, not {windows}");
        }
    }
}

#[test]
fn two_swapped_weights_fail_and_nothing_else_does()
{
    for target in targets() {
        let (_f, parts, mut plan) = planned(target);
        // Two weights of one window, so each lands inside the other's region.
        let (a, b) = {
            let w: Vec<usize> = (0..plan.entries.len()).filter(|&i| plan.entries[i].role == Role::Weight && plan.entries[i].dims.len() == 2).collect();
            let a = w[0];
            let b = *w[1..].iter().find(|&&j| plan.entries[j].region == plan.entries[a].region).expect("two weights share a window");
            (a, b)
        };
        let (oa, ob) = (plan.entries[a].offset, plan.entries[b].offset);
        plan.entries[a].offset = ob;
        plan.entries[b].offset = oa;
        let r = run(&parts, &plan).unwrap();
        let mut both = vec![plan.entries[a].tensor.as_str(), plan.entries[b].tensor.as_str()];
        both.sort_unstable();
        assert_eq!(r.failed(Check::Bytes), both, "{target:?}:\n{}", failures(&r));
        assert_eq!(r.failed(Check::Matvec), both, "{target:?}:\n{}", failures(&r));
        assert!(r.failures().iter().all(|o| both.contains(&o.subject.as_str())), "{target:?}:\n{}", failures(&r));
    }
}

#[test]
fn an_offset_shifted_by_one_byte_fails_alignment()
{
    for target in targets() {
        let (_f, parts, mut plan) = planned(target);
        let w = at(&plan, "blk.0.ffn_up.weight");
        let s = at(&plan, "ffn_inp-0");
        assert_eq!(plan.entries[s].role, Role::Scratch);
        plan.entries[w].offset += 1;
        plan.entries[s].offset += 1;
        let r = run(&parts, &plan).unwrap();
        assert_eq!(r.failed(Check::Align), ["blk.0.ffn_up.weight", "ffn_inp-0"], "{target:?}:\n{}", failures(&r));
        assert_eq!(r.failed(Check::Bytes), ["blk.0.ffn_up.weight"]);
    }
}

#[test]
fn a_changed_header_hash_or_size_refuses_to_start()
{
    let (_f, parts, plan) = planned(Target::cpu(1 << 30));
    let mut hash = plan.clone();
    let h = &mut hash.model.parts[0].identity.header_sha256;
    *h = h.chars().rev().collect();
    assert!(matches!(run(&parts, &hash), Err(StartError::Identity(_))));
    let mut size = plan.clone();
    size.model.parts[0].identity.file_bytes += 1;
    assert!(matches!(run(&parts, &size), Err(StartError::Identity(_))));
    assert!(run(&parts, &plan).is_ok());
}

#[test]
fn two_scratch_tensors_alive_together_at_one_offset_fail_sharing()
{
    for target in targets() {
        let (_f, parts, mut plan) = planned(target);
        // attn_norm-0 is read by the Q, K and V matmuls; the Q matmul writes Qcur-0
        // while attn_norm-0 is still needed by K and V.
        let step = plan.step.clone().unwrap();
        let q = step.ops.iter().find(|o| o.op == "mul_mat" && o.inputs.iter().any(|n| n == "attn_norm-0")).unwrap();
        let (x, y) = (at(&plan, "attn_norm-0"), at(&plan, &q.outputs[0]));
        let (lx, ly) = (plan.entries[x].live.unwrap(), plan.entries[y].live.unwrap());
        assert!(lx.0 <= ly.1 && ly.0 <= lx.1, "{lx:?} and {ly:?} should overlap");
        assert_eq!(plan.entries[x].region, plan.entries[y].region);
        // Move Y onto X, and Y's views with it, as a planner bug would.
        let (from, to) = (plan.entries[y].offset, plan.entries[x].offset);
        let y_name = plan.entries[y].tensor.clone();
        for e in plan.entries.iter_mut() {
            let views_y = matches!(&e.role, Role::View { of, .. } if *of == y_name);
            if e.tensor == y_name || views_y {
                e.offset = e.offset - from + to;
            }
        }
        let r = run(&parts, &plan).unwrap();
        let failed = r.failed(Check::Sharing);
        assert!(failed.contains(&"attn_norm-0"), "{target:?}:\n{}", failures(&r));
        assert!(r.failures().iter().all(|o| o.check == Check::Sharing), "{target:?}:\n{}", failures(&r));
    }
}

#[test]
fn a_cuda_plan_is_refused_until_cuda_is_covered()
{
    let (_f, parts, plan) = planned(Target::cuda(1 << 30));
    assert!(matches!(run(&parts, &plan), Err(StartError::Unsupported(_))));
}
