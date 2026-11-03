# memplan/ — where every tensor lives in memory

The name is short for memory plan: the table that says where each tensor sits in memory
on one machine. The table's type is `Plan`, and `dovata plan` is the command that prints
it.

The planner takes a model, its layout, the size of one forward step, and a target
machine, and gives back an address table: every tensor's region and byte offset. It
leans on three neighbors: [`src/target/`](../target/README.md) describes the machine,
[`src/window/`](../window/README.md) fits a mapped file into GPU buffers, and
[`src/arena/`](../arena/README.md) packs short-lived tensors into shared memory. The
Jade compiler bakes those offsets into the program as constants. At startup the program
sets up each region once, then addresses every tensor as its region's base plus a
constant.

```rust
let model = dovata::load::Model::open(path)?;
let layout = dovata::layout::for_model(&model, None)?;
let step = BindOptions { n_ctx: 32768, n_tokens: 512, n_outputs: 1, kv_k: KvType::Q8_0, kv_v: KvType::Q8_0 };
let plan = dovata::memplan::plan(&model, &layout, &step, &Target::apple(unified, max_buffer))?;
```

The plan also carries the step itself: every op in run order, its inputs and outputs
named as in the address table, and its parameters. The compiler generates one kernel per
op from that list and finds each tensor it names in the table.

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

On Apple and CPU the weights are never copied. The program maps each part file once and
reads every weight where it sits in the file. On Apple the mapping is bound to Metal as
several windows, each no larger than one Metal buffer, and every tensor lies wholly
inside one. On CPU there is one window per file starting at byte 0, so every offset is
the file offset. How the windows are cut is in [`src/window/`](../window/README.md).

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
How the arena packs is in [`src/arena/`](../arena/README.md).

Every entry in an allocated region starts at a multiple of the region's alignment, and
the program must allocate the region's base at that alignment too. The alignment for
each kind of memory is in [`src/target/`](../target/README.md).

`plan` binds the layout, then works out each tensor's lifetime from op order: a tensor
is alive from the first op that touches its memory to the last. Reading or writing a
view counts as touching the tensor it views. Inputs are alive from the start of the
step, since the program writes them before it runs, and outputs to the end of it, since
the program reads them after. `place` then takes the non-weight tensors as a list, each
with its range of ops. The weights come from the model itself, so every GGUF tensor
gets a weight entry whether or not the layout names it.

A view's entry carries its byte strides, ggml's `nb`, besides the tensor it views and
its byte offset. Views such as a permuted K are not contiguous, and a kernel needs the
strides to read one.

## Trusting the table

The offsets are compiled in, so a wrong one is a kernel reading the wrong bytes with no
error. Three things guard against that.

- The planner checks its own output with `Plan::check` before returning it: region ids
  dense from 0, every offset a multiple of its entry's alignment, every entry inside its
  region, mapped entries at their file offset minus their window's start, no shared
  bytes except between scratch tensors that are never alive together or between a view
  and the tensor it views, one weight entry per GGUF tensor, and a `table_hash` that
  matches. A violation is reported as a dovata bug, and the plan is not returned.
- The plan records each part file's name, the SHA-256 of its header, and its exact
  size. The program checks these at startup and refuses a different file.
- `table_hash` is a SHA-256 over the whole plan. The same model, layout and target give
  the same hash, so a rebuild that changes the constants is visible.

## Files

- `address.rs` — `place`, which builds the regions and entries for a target and checks
  the result, and the `Plan` types that serialize to the JSON the compiler reads.
- `graph.rs` — `plan` and `plan_graph`, which turn a bound layout into the list `place`
  takes, with each tensor's lifetime, and put the step's ops into the plan as `Step`.
- `mod.rs` — re-exports.

`a_real_header_plans_on_every_target` in `graph.rs` plans a real model when
`DOVATA_TEST_HEADER` names a GGUF file or its first tens of megabytes:

```sh
DOVATA_TEST_HEADER=model.head.gguf cargo test -p dovata --lib a_real_header -- --nocapture
```

## Who uses it

The `dovata plan` command and the Jade compiler call it. `refback`, the test-only
reference backend in [`src/refback/`](../refback/README.md), sets up the regions for
real and checks every entry against the GGUF bytes.
