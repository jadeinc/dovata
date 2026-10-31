# moe/ — expert ranking

Which experts each layer of a MoE model uses most. In 3.0 the planner will use it to
decide which experts stay resident in VRAM and which stay in host memory.

## What it was, and what is left

Up to 2.x this directory was the whole expert-offload capability: a C++ and CUDA driver
that paged cold experts into VRAM while the model ran, a patch series that hooked it
into llama.cpp, and the FFI between them. All of that ran at inference time, so it went
with the rest of the runtime on the 3.0 branch. It is on `main` under `src/moe/driver/`,
and the design is in [`docs/moe-expert-offload.md`](../../docs/moe-expert-offload.md).

What is left is the part that never ran a model: the loader for the `expert_order`
asset, in `mod.rs`.

The driver's residency policy, `LayerResidency` in `main:src/moe/driver/include/moe_policy.h`,
is the one piece of the old driver the planner still wants. It decided which experts
are pinned and which get cache slots. The 3.0 plan ports it to Rust, together with its
test cases from `main:src/moe/driver/test/moe_policy_test.cpp`.

## Why the asset is validated so strictly

The `expert_order` asset is a per-layer permutation of expert ids, hottest first, made
offline from usage counts. `ExpertOrder::from_json` checks the layer count, the expert
count, that every id is in range, and that each layer is an exact permutation.

A non-permutation leaves one expert unreachable and another placed twice. The model
would still answer, worse, with nothing logged. Catching it here means it fails in a
unit test on a laptop, not on a GPU box.

## Who uses it

Nothing calls it yet. The planner will, once it exists. The real asset for Qwen3-Coder,
48 layers by 128 experts, sits in [`../model/profiles/`](../model/profiles/README.md),
and the tests load it.
