# moe-driver — index-driven MoE expert offload

A small C++/CUDA static library that keeps a Mixture-of-Experts model running on a GPU
too small to hold all of it. The hot experts stay pinned in VRAM, a cache holds recently
used ones, and everything else lives in host RAM and gets paged in on demand.

This is ours, not vendored — it sits inside `src/moe/` with the capability that owns it, and the llama.cpp build
pulls it in with `add_subdirectory()`.

## The problem it solves

Qwen3-Coder-30B-A3B with a 256k KV cache does not fit on a 24 GiB card: full experts
15.9 GiB, non-expert weights 1.3, KV at q4_0 6.75, compute ~1, desktop reserve ~2 — about
27 GiB. But a 30B-total MoE only activates ~3B per token, and expert usage is heavily
skewed, so most of that resident weight is idle most of the time.

The driver exploits the skew. An offline measurement ranks every layer's experts by how
often routing selects them (`src/model/profiles/*.expert-order.json`). The top H experts
per layer are pinned, C more slots hold an LRU cache, and the rest are computed on CPU
for the current token and asynchronously promoted into a cache slot for next time. The
math is unchanged — the same weights, just placed differently — so this is lossless. The
only cost is decode speed when a token routes somewhere non-resident, which is exactly
what the cache minimizes. Full design in `docs/moe-expert-offload.md`.

## Why the seam looks the way it does

Two constraints shaped it.

The **header speaks pure C and ggml types only** — no llama internals, no C++ in the
public surface. `moe_driver.h` is the single header the vendored llama.cpp includes, so a
rebase onto newer upstream does not drag the driver along with it. When the driver is
inactive every seam call is a no-op and llama.cpp runs stock.

The **policy is split from the plumbing**, the same discipline as the daemon's pure core.
Deciding which expert occupies which slot and what to evict is bug-prone bookkeeping with
no need for a GPU, so it lives in `moe_policy.cpp` with no CUDA and no ggml, and is unit
tested on any platform including macOS. Only the paging and graph emission need hardware.

## What each file does

**Headers** (`include/`)

- `moe_driver.h` — the public C ABI. Init once per model load, `build_ffn` per layer
  during graph build, free at teardown. `moe_driver_config` mirrors `MoeOffload` in
  `src/load/config.rs`.
- `moe_policy.h` — `LayerResidency`: the per-layer slot map. Slots `0..H` are pinned,
  `H..H+C` are the LRU cache, everything else is cold.
- `moe_driver_internal.h` — state shared between the host core and the graph hook. Not
  public. Holds ggml/CUDA handles as opaque `void*` so the host side stays ggml-free.

**Sources** (`src/`)

- `moe_policy.cpp` — the residency and eviction logic. Pure host C++.
- `moe_driver.cpp` — host orchestration behind the C ABI. Also ggml-free.
- `graph.cpp` — the `build_moe_ffn` hook. Needs ggml headers, so it is compiled only
  inside the llama.cpp build.
- `page.cu` — the CUDA copy engine that moves expert weights host↔VRAM.
- `moe_stub.cpp` — no-op implementations for builds without the driver.

**Test** (`test/`)

- `moe_policy_test.cpp` — exercises the residency policy: pinning, cache fills, eviction
  order, permutation validity.

## Who uses it

llama.cpp calls it, and only when active — the seam in our fork delegates to these entry
points behind the `JADE_MOE_DRIVER` cmake option. Its configuration comes from
`dovata::config::MoeOffload` (parsed from `JADE_MOE_*` env), and the expert ranking is
loaded and validated by `dovata::moe` before being flattened and passed over FFI, so the
C++ side never parses JSON.

## Building and testing

The policy tests build standalone on any platform, no ggml and no CUDA:

```sh
cmake -S . -B build -DMOE_DRIVER_STANDALONE=ON
cmake --build build
./build/moe_driver_test
```

In a real build it comes in through llama.cpp's `add_subdirectory()`, where the `ggml`
target exists and `graph.cpp` is added to the source list.

## Gotcha

The driver is **off** unless `JADE_MOE_DRIVER` is truthy at runtime, so existing
deployments are byte-for-byte unaffected. The default operating point is `pin=16`,
`cache=48` per layer — the validated `phase2_latency` point. Also note that request
routing keys off the live runtime context size, not the `context_window` figure in the
model profile; the profile field is a documented contract, not what the daemon reads.
