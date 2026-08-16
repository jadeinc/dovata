# moe/ — expert offload

Keep a MoE model's cold experts in host RAM and page them into VRAM as the router asks
for them, so a 30B-total model runs in the VRAM its ~3B active parameters need.

The design is in [`docs/moe-expert-offload.md`](../../docs/moe-expert-offload.md).

## The one capability that reaches into llama.cpp

Which is why all of it is here, in four forms:

| | |
|---|---|
| `mod.rs` | the pure half: loads and validates the per-layer `expert_order` asset |
| `ffi.rs` | the C ABI declaration, and the one `unsafe` call |
| [`driver/`](driver/README.md) | the C++ and CUDA that does the paging |
| [`driver/bundle/`](driver/bundle/README.md) | its patch series against llama.cpp |

Those four used to live in three unrelated places, and the fourth was invisible — the
llama.cpp seam existed only as commits on a fork, so nothing in this repo recorded that
the change belonged to this capability, or that it existed at all.

## Why the Rust side parses the JSON

The `expert_order` asset is a per-layer permutation of expert ids, hottest first. It is
validated here — right count, in range, no duplicates, an exact permutation — so the C++
never touches JSON and the bug-prone check is unit-tested on a laptop.

A non-permutation would leave one expert unreachable and another double-mapped. The
model would still answer, worse, with nothing logged.

## Failing loudly is the whole design

A driver that does not engage produces *identical output* and roughly three times the
VRAM. The first symptom is an OOM somewhere that never mentions experts, which is why
every layer here refuses rather than falls back:

- `Config::validate` refuses a build without the `moe` capability but with
  `JADE_MOE_DRIVER` set.
- A failed asset load, or a non-zero `moe_driver_init`, fails the library load. Both used
  to log and continue on the stock path.
- `ffi::init` first calls `moe_driver_seam_present`, so a llama.cpp built without the
  seam is caught before the driver is initialised into a graph that will never call it.

## Coverage

The pure loader is well tested. Nothing else here is: the driver's policy has its own
host-only gtest-free binary (see `driver/README.md`), and the paging itself is verified
by a before-and-after VRAM comparison on the GPU box, because no eval covers it.
