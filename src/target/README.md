# target: the machine a plan is for

`target` describes the machine a compiled program will run on, in the terms the planner
needs: how much memory a plan may use, the largest single GPU buffer, and the page size.
That description is `Target`, which has three kinds:

```rust
Target::apple(unified_bytes, max_buffer_bytes) // Apple Silicon, 16 KiB pages
Target::cpu(ram_bytes)                         // CPU only, 4 KiB pages
Target::cuda(vram_bytes)                       // one NVIDIA GPU
```

It also holds `ExecSite`, where the compiled program runs the ops: on the CPU, as
Metal shaders or as CUDA kernels. And it holds `RegionKind`, the kinds of memory a
plan's regions can be: a window of the mapped model file, host memory, VRAM or Apple's
unified memory.

## Why it exists

The planner never reads hardware, because the Jade compiler may build on a laptop for a
GPU box. So the machine is an input that the code calling the planner describes, and a
plan is only as right as what it was told. `target` is that description and nothing
more.

## Why it sits outside `memplan/` and `machine/`

Four capabilities share it. `memplan/` plans for it. `machine/` builds one for the current
machine through `Machine::target`. `cli/` builds one from the command line. `refback/`
sets up real memory for one. Inside `memplan/`, it made the planner look like it owned the
machine's description. Inside `machine/`, it would go away when the `machine` feature is
off, and the planner needs it in every build.

## Kinds of memory and their alignment

Every entry in an allocated region starts at a multiple of the region's alignment, and
the program must allocate the region's base at that alignment too.
`RegionKind::alloc_align` gives it:

| kind | alignment | why |
|---|---|---|
| `vram` | 256 bytes | `cuMemAlloc` returns 256-byte-aligned memory, and 256 covers every vector width a kernel loads |
| `unified` | 256 bytes | the same reason, for Metal |
| `host` | 64 bytes | one cache line |
| `mmap` | 1 byte | a mapped window has no alignment of its own; its entries sit at the GGUF's file offsets, which are aligned to the file's `general.alignment` |

## Files

- `mod.rs`: `Target` and its builders `apple`, `cpu` and `cuda`; `Target::site`;
  `ExecSite`; `RegionKind` and `alloc_align`.

## Who uses it

`memplan/` takes a `Target` and records it in every plan. `machine/` builds one from the
current machine, `cli/` prints and parses them for `dovata machine target` and
`dovata plan`, and `refback/` sets up each region by its `RegionKind`.
