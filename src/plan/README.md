# plan/ — where every tensor lives

The planner takes a model, the tensors a layout says one step needs, and a target
machine, and gives back an address table: every tensor's region and byte offset. The
Jade compiler bakes those offsets into the program as constants. At startup the program
sets up each region once, then addresses every tensor as its region's base plus a
constant.

## Why region and offset

An absolute address cannot be known at build time. The driver picks where `cuMemAlloc`
lands, ASLR moves every mapping, `cuMemAddressReserve` may ignore the address it is
asked for, and `MTLBuffer.gpuAddress` is read-only. So the table works the way a linker's
relocations do. On Metal the program binds `(buffer, offset)` instead of adding.

## What goes where

| target | weights | KV cache and state | scratch |
|---|---|---|---|
| Apple | the mapped GGUF, in windows no larger than one Metal buffer | `unified` | `unified` |
| CPU | the mapped GGUF, one window per part file | `host` | `host` |
| CUDA | copied into `vram` | `vram` | `vram` |

On Apple and CPU the weights are never copied. The program maps each part file once.
On Apple, Metal reads the mapping through no-copy buffers, the way ggml's Metal backend
does (`ggml-metal-device.m`). A Metal buffer has a size limit, `maxBufferLength`, so the
mapping is bound as several windows, and every tensor lies wholly inside one. On CPU
there is one window per file starting at byte 0, so every offset is the file offset.

On CUDA everything is copied into VRAM once at startup. A model that does not fit is
refused. Placing part of it in host memory, including paging MoE experts, is deferred
until that design is settled with the user. One GPU only.

Nothing moves while the model runs.

## Memory reuse

The KV cache and recurrent state live for the whole program and get their own regions,
packed in layout order, which is layer order. Inputs, scratch and outputs live for one
step and share one arena: a tensor can take bytes another tensor used earlier in the
step, as long as the two are never alive at the same time. Lifetimes come from op
order in the bound layout. Views own no memory; they sit inside the tensor they view.

## Trusting the table

The offsets are compiled in, so a wrong one is a kernel reading the wrong bytes with no
error. Three things guard against that.

- The planner checks its own output before returning it: alignment, bounds, mapped
  entries at their file offsets, no overlap except scratch that is never alive together,
  and one weight entry per GGUF tensor. A violation is reported as a dovata bug.
- The plan records each part file's name, the SHA-256 of its header, and its exact
  size. The program checks these at startup and refuses a different file.
- `table_hash` is a SHA-256 over the whole plan. The same model, layout and target give
  the same hash, so a rebuild that changes the constants is visible.

## Files

- `target.rs` — `Target`, the machine as the caller describes it; dovata does not probe
  hardware, since the compiler may build on a laptop for a GPU box. Also `ExecSite` and
  `RegionKind`.
- `window.rs` — splitting a mapped file into page-aligned windows no larger than one GPU
  buffer.
- `arena.rs` — greedy-by-size placement of per-step tensors by lifetime, and a checker
  for it. Deterministic: ties break by first use, then by name.
- `address.rs` — `place`, which builds the regions and entries for a target and checks
  the result, and the `Plan` types that serialize to the JSON the compiler reads.
- `mod.rs` — re-exports.

## Status

`place` takes a neutral list of non-weight tensors. The step that turns a bound layout
into that list, and puts the ops into the plan, follows once the layout format in
[`src/layout/`](../layout/README.md) is reviewed.

## Who uses it

The `dovata-plan` CLI and the Jade compiler call it. `refback`, the test-only reference
backend, will set up the regions for real and check every entry against the GGUF bytes.
