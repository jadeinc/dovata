# MoE expert offload — index-driven pin + dynamic VRAM cache

**Status:** planned (design locked, not yet implemented).
**Goal:** raise the native context window of Qwen3-Coder-30B-A3B to **256k tokens on a
single RTX 3090 (24 GiB, driving a display)** by offloading cold MoE experts to host RAM,
guided by a measured per-layer expert-usage index. Lossless (same math, only weight
placement changes); the only cost is decode speed when a token routes to a non-resident
expert, which the cache minimizes.

This composes with the blended RAG-RLM engine (see `blended-rag-rlm.md`): a 256k native
window under the measured ~8× RLM frontier puts effective context in the ~2M range
(re-measure `JADE_FRONTIER` at the new window to confirm the multiplier holds).

## 1. Why offload is required, and why static pinning is not enough

A 3090 cannot hold the full model + a 256k KV cache:

```
full experts 15.9 + non-expert weights 1.3 + KV(256k, q4_0 @27KB/tok) 6.75
  + compute ~1.0 + desktop reserve ~2.0  =  ~27 GiB  >  24 GiB
```

So we must move some expert weight off the GPU. **How much** depends on the expert-usage
distribution — and the measured distribution for this model is **near-uniform**, not the
heavily-skewed shape an earlier (mislabeled adaptive) metric suggested. Ground truth,
computed directly from `expert_counts` in the offline study
(`moe-routing-offload-experiments/results/expert_usage.json`; entropy 6.37 / 7.0 bits,
Gini 0.50):

| pinned K / layer | % of experts | static top-K coverage (mean) | worst layer |
|------------------|-------------:|-----------------------------:|------------:|
| 16 (12.5%)       |              | 37%                          | 29%         |
| 32 (25%)         |              | 58%                          | 49%         |
| 43 (⅓)           |              | 69%                          | 60%         |
| 64 (50%)         |              | 85%                          | 77%         |
| 86 (⅔ kept)      |              | 94%                          | 90%         |

**Implication:** pinning the globally-hottest ⅓ of experts covers only ~69% of routing —
a ~31% cold tail. Computing 31% of every token's experts on the CPU would gut decode
throughput. Therefore a **static partition alone is insufficient at aggressive offload**,
and a **dynamic VRAM cache** that captures the *per-context* working set is required. The
locality study supports this: 42% token-to-token expert reuse, and a per-layer working set
that saturates around ~51 experts.

## 2. Operating point (3090, 256k)

- `JADE_CTX_SIZE = 262144`. KV ≈ 6.75 GiB at q4_0.
- **Resident experts = pinned H + cache C per layer.** Provisional **H=16, C=48**
  (64 resident = 50% offload) — the validated knee from the offline `phase2_latency` model
  (~8% decode overhead) and large enough to hold the ~51-expert working set. ≈ 8 GiB of
  experts in VRAM → ~5 GiB free after the desktop at 256k. H/C are tunables: 256k's slack
  lets us grow C if measured miss-rate is high, or shrink toward 43-resident for maximum
  headroom.
- Cold experts (~64/layer) are **pinned in host RAM** → the workstation needs **≥ 32 GiB
  system RAM**.

> Note: at 256k specifically the cache is partly insurance — a large static pin (H≈104,
> ~98% coverage) would also fit. Its real payoff is letting the *same* engine scale down to
> smaller GPUs and up to larger contexts without a rewrite. Phase 1 alone (large pin, no
> paging) is therefore a valid ship if Phase 2 slips.

## 3. Architecture — isolated driver + thin seam

We do **not** edit llama.cpp internals in place. All offload logic lives in a standalone
module; the vendored fork gets only a small, guarded seam so we can **abort cleanly**
(stock behavior) and **re-sync with upstream** (re-apply a handful of lines, not a sprawling
diff).

```
llama-cpp/
  moe-driver/            # NEW — standalone C++/CUDA static lib, OUTSIDE llama.cpp/ subtree
    CMakeLists.txt       #       own build target
    moe_driver.h         #       pure-ggml + C-ABI surface (no llama internal types)
    graph.cpp            #       dual mul_mat_id graph emission (split tensors)
    cache.cu / cache.cpp #       cache directory + async host→VRAM copy engine + kernels
    policy.cpp           #       LRU/admission state machine (pure; mirrored by Rust tests)
  ffi/llama.cpp/         # vendored upstream — touched only at the seam below
```

**The seam (≈3 guarded call sites + build wiring), each a passthrough when the driver is off:**
- `src/llama-graph.cpp :: build_moe_ffn` — `if (moe_driver_active()) return
  moe_driver_build_ffn(ctx0, gate_exps, up_exps, down_exps, cur, ids, weights, ...);`
- `src/llama-model.cpp :: load_tensors` — per `blk.%d.ffn_{gate,up,down}_exps` tensor, ask
  the driver for the split/placement plan (resident GPU buft vs cold CPU buft).
- one init/teardown call at context create/free for the cache-engine lifecycle.
- `add_subdirectory(moe-driver)` + link, behind CMake option `JADE_MOE_DRIVER=ON`.

The seam adapts llama's internal handles into **public ggml types** at the boundary, so the
driver never depends on llama internals and survives upstream rebases.

### Mechanism: paging, NOT dual `mul_mat_id`

Reading the fork settles the mechanism. The expert matmul in
`llama-graph.cpp::build_moe_ffn` is a **single** `build_lora_mm_id(exps, cur,
selected_experts)` (→ `ggml_mul_mat_id`) over `selected_experts[n_expert_used=8,
n_tokens]`. A naive "second `mul_mat_id` over a CPU cold tensor" does **not** work: that op
still computes all 8 ids/token (the non-cold ~7 masked to zero weight), wasting ~7 CPU
expert-matmuls/layer ≈ ~1 GiB/token of wasted host bandwidth → it dominates latency. ggml
has no data-dependent control flow to compute "only the cold picks."

So cold experts are **paged into VRAM**, not computed on CPU. The expert tensor in VRAM is
the **resident** tensor of `H+C` experts (pinned + cache slots). Per token per layer:

1. the router produces `selected_experts` (forced to a host-readable buffer — it is tiny,
   8×n_tokens i32);
2. a **custom op** (`ggml_map_custom`) between the router and the FFN runs a host callback
   that, via `LayerResidency`: classifies each id resident/cold, **async-copies each cold
   expert host→VRAM into a cache slot** (LRU evict), updates the slot map, and writes the
   **remapped slot-ids** into its output tensor;
3. a **single GPU `mul_mat_id`** runs over the resident tensor using the remapped ids — all
   experts are now resident, all compute stays on GPU.

The ggml graph shape stays static (tensor sizes fixed); only the cache *contents* and the
remapped `ids` change per token. This is **lossless** — verifiable by logit equivalence vs
the full-resident model. The cost is the per-layer host sync + the host→VRAM copies, which
the offline `phase2_latency` model covers (`fetch_us≈80`, `concurrency=4` → ~8% at H16/C48).

Each `ffn_{gate,up,down}_exps` is permuted by the usage index at load (router/gate rows
permuted to match so ids stay consistent) and the cold experts kept in host RAM as the page
source.

## 4. Preprocessing (pure, macOS-testable)

`top32_experts_per_layer.json` has only 32 entries/layer; we need the full ranking. Generate
`expert_order[48][128]` (descending by `expert_counts`) from `expert_usage.json` and ship it
as a model-profile sidecar asset, read at load. Config (enable, H, C, order-path) flows from
`dovata` through the `ffi`/`api` crates into `moe_driver` init — nothing hardcoded.

New env / profile knobs: `JADE_MOE_DRIVER` (on/off), `JADE_MOE_PIN` (H), `JADE_MOE_CACHE`
(C), `JADE_MOE_ORDER_PATH`.

## 5. Phase 1 — static partition + CPU cold tail

Foundation for the cache, and **ships 256k on its own** (large pin, no paging → ~98%
coverage, ~2% CPU cold).
- **driver/graph.cpp**: dual masked `mul_mat_id` (GPU resident + CPU cold) + weighted
  combine; id remap into the two partitions.
- **driver placement plan** consumed by the `load_tensors` seam: permute + split each
  `ffn_*_exps`, permute router/gate rows to match.
- **Rust surface**: thread H / order-path / enable through `ffi` + `api` into `dovata`
  (`Config::from_env` + model profile).
- **Verify**: logits match full-resident within fp tolerance (box); VRAM + tok/s measured;
  index parsing + id-remap logic unit-tested pure on macOS.

## 6. Phase 2 — dynamic VRAM cache (the hard part)

Turn the C region into an LRU cache so the pin can shrink to the ⅔-offload point without the
cold tail exploding.
- **driver/cache**: per-layer slot↔expert directory; on a same-token miss, CPU-compute now +
  enqueue host→VRAM copy on a dedicated stream (offline model: `fetch_us≈80`,
  `concurrency=4`), LRU evict. Working set migrates into VRAM over a generation; CPU-miss
  rate decays.
- **token-level id→slot remap** fed as the `ids` input each step (small host gather).
- **driver/policy.cpp** (admission, LRU, pin-protection) is **pure** → unit-tested on macOS;
  only the CUDA copy/gather is box-only.
- **Verify** measured overhead vs the `phase2_latency` model (~8% at H16/C48) and miss-rate
  behavior on *long* generations (the offline locality data is on 18-token prompts and is
  pessimistic for the long-context regime).

## 6b. Baseline measured (whole-layer offload) — 2026-06-21

Measured on the A10G box (23028 MiB, headless), Qwen3-Coder-30B-A3B Q4, q4_0 KV,
flash-attn on, via `JADE_CPU_MOE_LAYERS` (stock buft override, wired in this session) +
`JADE_GEN_BENCH`. This is the **dumb baseline** the index-driven engine must beat.

**Decode throughput (tok/s, 128 generated):**

| offload            | resident 128 | 1024 | 4096 | 8192 |
|--------------------|-------------:|-----:|-----:|-----:|
| none (all on GPU)  | 131.7        | 122.0| 106.5| 90.6 |
| first 14 layers (29%) → CPU | 41.5 | 46.2 | 44.1 | 41.2 |

→ **whole-layer offload of 29% of layers costs a 2.2–3.2× decode slowdown** — every token
routes 100% through CPU for those layers. (Decode rate is ~flat across resident size, so the
hit is per-token expert compute, not attention.)

**256k native window — feasibility (14-layer offload):** CUDA0 model 12592 MiB, CPU experts
5257 MiB, KV(256k, q4_0) 6912 MiB, compute 792 MiB → **peak 20621 / 23028 MiB (~2.4 GiB
free)**, decode ~41 tok/s. So **256k native context fits on a 24 GiB card** with ~29% layer
offload — the headline feasibility result.

**Accuracy:** whole-layer offload is numerically lossless; with offload on, the RULER-style
rerank still ranks the needle #1 and the plain path retrieves it (the blended-answer gap is
the pre-existing compose-framing issue, unrelated to offload).

**Implication / target:** the naive baseline pays 120–220% decode overhead because it moves
*whole layers*. The index-driven engine pins the hot experts so only the cold tail (~31% at a
⅓ pin, paged via the LRU cache) touches the slow path — target ~8% overhead per the
`phase2_latency` model. That delta is the entire point of the smart engine.

> Wiring note: `add_cpu_buft_override` stores a raw pointer into the pattern `CString`; it
> MUST outlive `load_from_file`. A first cut scoped the `CString` to the `if let` block → it
> dropped pre-load, the regex read freed memory, and **nothing offloaded** (caught by
> identical tok/s). Fixed by binding it for the function scope. See `inference.rs`.

## 6c. Smart engine built + measured (2026-06-21) — host-sync bottleneck found

The full index-driven engine is implemented and runs end-to-end on the A10G:
- `moe-driver` lib (policy + driver + ggml hook + CUDA pager), compiled into the
  **ggml-cuda** target; the only vendored seam is the `build_moe_ffn` delegation +
  a `GGML_OP_CUSTOM` CUDA dispatch + a capture-disable line.
- Mechanism: experts kept host-resident (CPU buft override); driver allocates the
  VRAM resident tensors (H+C/layer) and pages cold experts on the CUDA stream.

**Two real bugs found and fixed by running it:** (1) `tensors` vector unsized in
init → segfault; (2) the pager's `cudaStreamSynchronize` ran *during CUDA-graph
capture* → abort, fixed by disabling capture for `GGML_OP_CUSTOM`.

**Measured decode (resident 128, pin16/cache48):**

| variant | tok/s |
|---|---|
| no offload (baseline) | 131.7 |
| 14-layer whole-layer offload (baseline) | 41.5 |
| smart engine, `ggml_map_custom` (CPU op) | 27.6 |
| smart engine, `GGML_OP_CUSTOM` (on-stream) | 24.7 |

**Finding: the smart engine is NOT yet faster than the baseline, and the graph
break was not the whole story.** Moving the op onto the CUDA backend removed the
~96 per-layer backend-split syncs, but decode is still ~25 tok/s because the
pager does a **`cudaStreamSynchronize` per layer** — it must read the router's
`selected_experts` back to the host to run the residency policy, which serializes
the GPU 48×/token (the same order of stall the backend split caused).

**The fast path requires fully device-side paging (no host sync):** a CUDA kernel
that reads `selected_experts` on-device, maps slots on-device, and copies cold
experts on-device from a **pinned/mapped** host pool. That needs the expert pool
host-pinned (~16 GiB) — **infeasible on this 15 GiB A10G box**, but feasible on:
- the **RTX 3090 target** (more system RAM → pin the pool), or
- a **low-offload operating point** (keep ~100/layer resident → ~3.5 GiB cold
  pool, pinnable even on 15 GiB) — which is also exactly what 256k needs (§2/§6b).

So: engine built, correct, runs; the host-sync version is host-sync-bound at
~25 tok/s; beating the baseline is gated on the device-side no-sync pager, which
is the next build and is best measured on the actual 3090 (RAM) target.

## 7. Risks / notes

- **The dual-`mul_mat_id` + dynamic `ids` remap is the core risk** — surgery on the hottest
  kernel path. Phase 1 de-risks it before any paging.
- **Not zero-touch**: the graph/load paths are internal, so a thin seam is the floor. Kept
  minimal + guarded for clean abort and easy upstream sync.
- **System RAM** ≥ 32 GiB for the host-resident cold experts.
- **Re-run `JADE_FRONTIER` at 256k** to confirm the RLM multiplier (the ~2M figure).
- Box is billable — batch the equivalence + VRAM + tok/s runs into one session.
