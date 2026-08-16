# bundle/ — the MoE capability's delta to llama.cpp

Three patches and a manifest. Together they are everything this capability changes in
the inference engine, and they are the reason this repo no longer carries a fork.

## Why this exists

llama.cpp used to be a fork: `joericks1998/llama.cpp`, tracked as a submodule, four
files ahead of upstream. The delta was small, and that was the problem — it was small
*and invisible*. Nobody could see from this repo what we had changed upstream or why.
Every upstream bump was a rebase against a branch whose contents you had to go read
somewhere else. And the change belonged to one capability, which nothing recorded.

Now the submodule is stock `ggml-org/llama.cpp` at a pin, and the delta lives here, in
the directory of the capability that needs it. `build.rs` stages a copy of the pinned
tree, applies these patches to the copy, and points cmake at it. Reviewing what we do
to llama.cpp is now reading three files in this repo.

## What each patch does

- **`0001-llama-graph-build-moe-ffn-seam.patch`** — the seam itself. Inside
  `build_moe_ffn`, when the driver is active, swap the host-resident expert tensors for
  its VRAM-resident ones and remap `selected_experts` to resident slot ids; the stock
  matmul below then runs unchanged. The remap has to happen *after* the routing weights
  are gathered and *before* the matmul, because the weights were gathered with the
  original expert ids.

  It also defines `moe_driver_seam_present()`. See below.

- **`0002-ggml-cuda-custom-op-pager-dispatch.patch`** — the CUDA side. Dispatches
  `GGML_OP_CUSTOM` to the pager so paging stays on the CUDA stream instead of forcing a
  per-layer backend split, declares the op supported, and opts it out of CUDA graph
  capture (it does host-side classification and a data-dependent async copy, none of
  which is legal during capture).

- **`0003-cmake-compile-and-link-the-driver.patch`** — compiles the driver's four
  sources into `ggml-cuda`, and on non-CUDA builds compiles `moe_stub.cpp` into
  `libllama` so the seam's symbols resolve without a driver.

## The counted paths, and why they had to go

The fork's CMake hunks reached out of the llama.cpp tree by counting directories:

```cmake
set(MOE_DRIVER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../moe-driver)
target_include_directories(llama PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../moe-driver/include)
```

Five levels from one file, three from another. Both hardcode exactly how deep llama.cpp
sits inside *this* repo, so moving either directory silently breaks the build — and
staging the tree into `OUT_DIR`, which is the whole mechanism here, changes the depth by
construction.

The patches now read `${JADE_MOE_DRIVER_DIR}`, an absolute path `build.rs` passes as a
`-D`. The patched CMakeLists `message(FATAL_ERROR ...)` if it is unset, so a bundle
wired up wrong fails at configure time rather than producing a `libllama` with no driver
in it. The precedent is already in `build.rs`: `CMAKE_POSITION_INDEPENDENT_CODE` is set
there rather than in the fork, because the requirement comes from how *we* consume
llama.cpp, not from llama.cpp.

## Three guards against the quietest failure

The seam is a no-op when the driver is inactive. That is correct — a build without
expert offload must run stock — and it means *the patch never applied* and *the driver
is correctly disabled* look identical at run time. A build that silently skipped the
patch answers every prompt correctly and uses about three times the VRAM, so the first
symptom is an OOM somewhere that never mentions experts.

So there are three:

1. **`build.rs` panics** if any patch fails to apply, naming the patch and the tree.
   This is the one that fires in practice.
2. **`moe_driver_seam_present()`** is defined by patch 0001, beside the seam it reports
   on, so the symbol exists exactly when the seam does. It is declared on the Rust side
   only (`src/moe/ffi.rs`) and deliberately *not* in `moe_driver.h` — the
   non-CUDA build compiles `moe_stub.cpp`, which implements that whole header, and a
   stub definition would satisfy the check while proving nothing.
3. **`crate::moe::ffi::init` calls it** before `moe_driver_init`, and fails the library
   load if it does not return 1.

## Keeping it applying

`tools/scripts/check-bundles.py` stages a worktree at the pinned commit and runs
`git apply --check` for every patch. It takes seconds, needs no compiler, and CI runs it
before the CUDA build so a pin that moved out from under a bundle fails immediately
rather than half an hour in.

It also rejects any patch touching `convert_hf_to_gguf.py`, `gguf-py/`,
`tools/quantize/`, or `src/llama-quant.cpp`. That rule is what keeps *one pin, not two*
true now that there are two consumers of it: the engine links upstream **plus these
patches**, while `tools/scripts/quantize-model.py` builds its tools from **stock**
upstream at the same commit. They agree only while nothing here changes how a GGUF is
produced or read. Break that and the symptom is a published `model.gguf` the engine
refuses, discovered after hours on a GPU box.

## Bumping upstream

Move the submodule pointer, run the checker, and expect to regenerate rather than
rebase — `build_moe_ffn` and the `ggml_cuda_compute_forward` switch are both places
upstream moves. Then update `upstream` in `bundle.toml` to the new commit; the checker
warns when the two disagree, because a stale record is worse than none.

Do it as its own change. De-forking and bumping are different jobs, and the patch series
is the thing that tells you whether the bump is safe.
