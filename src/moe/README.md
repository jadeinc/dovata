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
asset, in `mod.rs`. The validation stayed because it is the part that is easy to get
wrong without noticing. `ExpertOrder::flat` still gives the ranking in the row-major
form the 2.x driver took over FFI, `flat[layer * n_experts + rank]`.

The driver's residency policy, `LayerResidency` in `main:src/moe/driver/include/moe_policy.h`,
is the one piece of the old driver the planner still wants. It decided which experts
are pinned and which get cache slots: the first `pin` experts of a layer's ranking were
pinned in VRAM, and the next `cache` got cache slots. The 3.0 plan ports it to Rust,
together with its test cases from `main:src/moe/driver/test/moe_policy_test.cpp`.

The capability is behind the `moe` cargo feature, so a build without it proves that
nothing else depends on the ranking.

## Why the asset is validated so strictly

The `expert_order` asset is a per-layer permutation of expert ids, hottest first, made
offline from usage counts. `ExpertOrder::from_json` checks the layer count, the expert
count, that every id is in range, and that each layer is an exact permutation.

A non-permutation leaves one expert unreachable and another placed twice. The model
would still answer, worse, with nothing logged. Catching it here means it fails in a
unit test on a laptop, not on a GPU box.

The loader parses the JSON into a `serde_json::Value` and walks it by hand, instead of
deserializing into a typed struct, because the asset is tiny and the validation, not
the deserialization, is the part that matters.

## Who uses it

Nothing calls it yet. The planner will, once the case of a MoE model that does not fit
in VRAM is designed. The real asset for Qwen3-Coder-30B, 48 layers by 128 experts, is
`assets/qwen3-coder-30b.expert-order.json`, and the tests load it. It moved here from
the 2.x model profiles, which 3.0 dropped.
