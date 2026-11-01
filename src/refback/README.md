# refback: a plan's memory, set up for real and checked

`refback` is a test-only reference backend for dovata. It takes a model file and a plan,
sets up the plan's memory the way the compiled program will at startup, and checks that
every tensor is where the plan says it is. It is never published and never grows into a
runtime: it has no layout ops, no tokenizer and no attention math. Its only kernel is a
matvec, used to read weights the way a GPU kernel reads them.

## Why it exists

The Jade compiler bakes a plan's offsets into the program as constants. A wrong offset
is a kernel reading the wrong bytes, with no error and only a worse answer. dovata's own
guard, `Plan::check`, checks the table against itself: dense region ids, aligned
offsets, entries inside their regions, no forbidden overlap. That proves the table is
consistent. It cannot prove the table is right, because it shares the planner's model of
memory.

refback checks the plan against things outside the planner: the bytes of the model file,
real mappings and allocations, and a walk over the step's ops. Its checks do not call
`Plan::check`, so a test can change the plan's public fields after planning and see that
refback, on its own, catches the change.

## What each check proves

`run` in `lib.rs` runs them in this order and returns one `Outcome` per check.

1. *Identity.* Each part file's size on disk and the SHA-256 of its header must equal
   `plan.model.parts`. `dovata::load::Model::from_parts` and `Model::identity` compute
   the hash. A different file is refused before anything is mapped, with
   `StartError::Identity`.
2. *Regions.* Each region is set up once. On the CPU, each part file is mapped once,
   read-only, with memmap2; a file-backed region's base is the mapping plus
   `file_start`, and an allocated region is a host allocation of `region.bytes` at
   `region.align`. On Apple, an allocated region is a shared `MTLBuffer`, and a
   file-backed window is wrapped with `newBufferWithBytesNoCopy`, its length rounded up
   to whole pages as ggml's `ggml_backend_metal_buffer_from_ptr` does; the buffer's
   contents must be the mapping's own address. Every base must meet its region's
   alignment. A region that cannot be set up as described, such as a window past the end
   of its file, is refused with `StartError::Region`.
3. *Alignment.* Every entry that owns memory has an offset that is a multiple of its
   alignment, and so is its address, base plus offset. Views own no memory and are
   skipped.
4. *Bytes.* Every weight's bytes at its planned address equal the file's bytes at its
   `source.file_offset`. The file's bytes are read with a positioned read, `pread`, not
   through the mapping, so a mapping at the wrong base cannot agree with itself.
5. *Matvec.* For every 2-D weight of type F32, F16, Q8_0 or Q4_0, y = W·x is computed with
   a fixed x, reading W from its planned address: plain Rust on the CPU, and on Apple a
   Metal shader with W bound at its region's buffer and offset. The result is compared
   with an f64 reference computed from the file's bytes. The bound is the worst error an
   f32 kernel can have in any summation order, `(cols + 2) * 2^-23 * sum(|w * x|)`, so a
   pass is exact up to rounding and a wrong address fails by far. Other types get the
   bytes check only.
6. *Sharing.* The step's ops are walked in run order. Every tensor that owns memory gets
   a byte pattern of its own, a hash of its name and the byte's position: inputs at the
   step's start, the KV cache and recurrent state once, and a scratch or output tensor at
   the first op that touches it. At every read, directly or through a view, the bytes the
   read covers must still be the owner's pattern. A view's bytes are found from its own
   entry and its strides `nb`, and compared with the owner's pattern at `byte_offset`
   plus the same strides. The outputs are read once more after the last op, as the
   program reads them. A view is never filled, since it owns nothing; its owner is.
   This proves the scratch arena's reuse and the region packing are safe: if two tensors
   that are alive together were given the same bytes, the one written later has its
   pattern where the other's should be.

The tests in `tests.rs` tamper with a plan after planning and check that refback catches
each change by itself:

- two weights' offsets swapped: exactly those two fail, in the bytes and matvec checks;
- an offset shifted by one byte: alignment fails, for a weight and for a scratch tensor;
- a part's `header_sha256` or `file_bytes` changed: refback refuses to start;
- two scratch tensors that are alive together moved to one offset: the sharing check
  fails, naming the tensor that was overwritten;
- a CUDA plan: refused, since CUDA is not covered yet.

Each runs on the CPU, and with the `metal` feature also on an Apple target whose buffer
limit is set to 48 KiB, so the fixture's weights are split over several windows and the
no-copy buffers start past byte 0.

## Files

| File | What it does |
|---|---|
| `lib.rs` | `run`, and the result types: `Check`, `Outcome`, `Report` and `StartError`. |
| `identity.rs` | The identity check: part count, header hash and size on disk. |
| `regions.rs` | `Memory`: maps each part file, sets up each region once, checks each base, and gives bounds-checked access to region bytes. The `unsafe` code for raw pointers into mappings and allocations is here. |
| `check.rs` | The alignment, bytes and matvec checks. |
| `quant.rs` | Decoding F32, F16, Q8_0 and Q4_0 as ggml does, with citations to `ggml-common.h` and `ggml-quants.c`; the CPU matvec; the f64 reference and its bound. |
| `sharing.rs` | The walk over the step's ops, and the byte patterns. |
| `metal.rs` | Behind the `metal` feature: the Metal device, shared and no-copy buffers, and the matvec dispatch. |
| `matvec.metal` | The four matvec kernels, one thread per row, reading W one byte at a time so any offset is safe to read. Compiled when the GPU is set up. |
| `fixture.rs` | A tiny but complete llama GGUF, header and data, written into a temporary directory. Two layers, 64-wide, with every 2-D weight F32, F16, Q8_0 or Q4_0 in turn, written directly with dovata's GGUF writer; no quantizer runs. |
| `tests.rs` | The true-plan and tampered-plan tests above. |
| `main.rs` | The dev command line, `refback`. |

## Running it

```sh
# The tests, on the CPU, then on the CPU and Metal
cargo test -p refback
cargo test -p refback --features metal

# A real model: plans it with dovata's built-in layout for this machine's limits, with
# dovata plan's default step, and prints one pass or fail line per check
cargo run --release -p refback --features metal -- model.gguf --target cpu
cargo run --release -p refback --features metal -- model.gguf --target apple
```

The command exits with 1 when any check fails or refback refuses to start, and with 2
for a bad command line. A release build checks a 135M model in about a second.

## What it does not cover

- *CUDA.* The Mac Mini this runs on has no NVIDIA GPU, so a CUDA plan is refused in
  `Memory::set_up`. CUDA goes in `regions.rs` as a third `Owner`, a `cuMemAlloc` region:
  weights are copied in from the file at setup, the bytes check reads them back with a
  device-to-host copy, and the matvec runs as a device kernel in a `cuda.rs` beside
  `metal.rs`. The other checks need no change.
- *Other weight types.* K-quants, I-quants and the rest get the bytes check only. A
  matvec over them would need their decoders, which proves nothing more about addresses.
- *Weights of more than two dims,* such as MoE expert stacks, get the bytes check only.
- *Compute.* The sharing check proves where tensors live and that their lifetimes are
  honoured. It does not run any op, so it says nothing about what the kernels compute.
- *More than one step.* The walk covers one forward step. The KV cache and state live in
  their own regions and are filled once, so a later step reuses nothing new.
- *A write nothing reads.* An overlap is seen at the overwritten tensor's next read. If
  the last op that touches it writes it rather than reads it, the overlap is not seen.
- *Weight data beyond the header hash.* Identity hashes the header and checks the size,
  as the program does. The bytes check then reads every weight, so changed weight data
  is still caught here.

## Who uses it

Only tests and people. `cargo test -p refback` runs it on the fixture, and the dev
command runs it on a real model. It depends on dovata's planner, layouts, GGUF reader and
writer, and, for the command line, its `machine` capability; nothing depends on it.
