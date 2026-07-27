# jade-gpu-mon — live GPU monitor

A terminal dashboard showing VRAM use, memory bandwidth, clocks, temperature, and
utilization, refreshed twice a second. Run it in a second pane while the daemon serves
requests.

## Why it exists

Most of the daemon's hard numbers are memory numbers. Whether the KV cache fits, whether
MoE expert paging is thrashing, whether a context size is survivable on a given card —
all of it is answered by watching VRAM and bandwidth during a real request, not by
reasoning about it beforehand. `nvidia-smi` polling in a loop works but scrolls; this
gives a stable readout you can watch while something else has your attention.

It is deliberately trivial and self-contained. It declares `[workspace]` in its own
`Cargo.toml`, which excludes it from the root workspace — it depends on NVML and would
otherwise force every `cargo build --workspace` to care about an NVIDIA library it does
not need.

## Files

- `src/main.rs` — the whole program: NVML polling, the bar rendering, and the crossterm
  alternate-screen loop. Quit with `q` or Ctrl-C.

## Who uses it

Nobody, programmatically. It is a human tool, run next to `dovata` on the GPU box. It
depends on `nvml-wrapper` and `crossterm`, and on nothing else in this repo — no protocol
types, no daemon code.

## Building and running

Because it is its own workspace, build it from inside its directory, on a machine with
an NVIDIA driver:

```sh
cd src/jade-gpu-mon && cargo run --release
```

It will not run on a Mac — NVML needs the NVIDIA driver present.
