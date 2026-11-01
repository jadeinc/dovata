# hw: the current machine's memory and GPUs

This capability reads the machine dovata is running on: how much memory it has, its page
size, its GPUs and their memory limits, and how all of that memory is being used right
now. `hw::machine()` returns the capacity as a `Machine`, and `hw::usage(process)`
returns a snapshot of use as a `Usage`. The `dovata hw` commands in
[`src/cli/`](../cli/README.md) print both.

## Why it exists

Most questions about running a model are memory questions: whether the weights fit,
whether the KV cache fits at a given context length, and whether the system starts
compressing or swapping while a program runs. `hw` answers them for the machine at hand.
It replaces `jade-gpu-mon`, a 2.x terminal dashboard that only worked on NVIDIA through
NVML and lived outside the workspace.

It is separate from the planner on purpose. The planner never reads hardware, because
the Jade compiler may build on a laptop for a different machine; it takes a
`plan::Target` the caller describes. `Machine::target` is the bridge: it describes the
current machine as a `Target`, for when the build machine is the machine the program
runs on.

## How each value is read

dovata runs each system's own tools and reads their output, the same approach
`quantize` takes with `llama-quantize`. Nothing is linked except Apple's Metal
framework, and there is no `unsafe` code.

| Value | macOS | Linux |
|---|---|---|
| memory size | `sysctl -n hw.memsize` | `MemTotal` in `/proc/meminfo` |
| page size | `sysctl -n hw.pagesize` | `getconf PAGESIZE` |
| processor | `sysctl -n machdep.cpu.brand_string` | `model name` in `/proc/cpuinfo` |
| system version | `sw_vers -productVersion` | `uname -sr` |
| memory use | `vm_stat` | `/proc/meminfo` |
| swap | `sysctl -n vm.swapusage` | `SwapTotal` and `SwapFree` in `/proc/meminfo` |
| pressure | `memory_pressure -Q` | `/proc/pressure/memory` |
| a program's memory | `ps -A -o pid=,rss=,comm=` | the same |
| NVIDIA GPUs | `nvidia-smi`, when installed | the same |

On macOS the tools are called by their full paths, such as `/usr/bin/vm_stat`, so a
program of the same name earlier on `PATH` is never run instead. The Linux tools vary in
location between distributions, so they are found on `PATH`.

### What "used" means

On macOS, used memory is app memory plus wired plus compressed. App memory is
`vm_stat`'s anonymous pages less its purgeable pages. File-backed pages are reported
separately as cached, since the system drops them when it needs room. On Linux, used is
`MemTotal` less `MemAvailable`, the kernel's own estimate of what a new program could
get. On both, available is total less used.

### Pressure

The two systems report pressure differently, so `Pressure` keeps them apart.
`memory_pressure -Q` on macOS prints the share of memory free, where lower is worse.
Linux's pressure stall information prints the share of recent time in which some task
waited for memory; dovata reads the 10-second average of the `some` line, where higher
is worse. A Linux kernel without pressure stall information has no reading.

### Metal's limits

No command-line tool reports a Mac GPU's memory limits, so `metal.rs` asks Metal through
the objc2-metal crate, which is Rust bindings to Apple's framework. It reads
`recommendedMaxWorkingSetSize`, the memory the GPU can use before performance suffers,
and `maxBufferLength`, the largest single buffer. The planner's Apple target needs both:
its region size is the working set, and a mapped GGUF is split into windows no larger
than the largest buffer. On a 24 GB Mac Mini with an M4, they are 19,069,665,280 and
14,302,248,960 bytes.

### NVIDIA GPUs

`nvidia-smi --query-gpu=... --format=csv,noheader,nounits` gives each GPU's index, name,
total, used and free memory in MiB, load, temperature, and shader and memory clocks. A
field `nvidia-smi` cannot read prints as `[N/A]` or `[Not Supported]`, and becomes no
value. When `nvidia-smi` is not installed there are no NVIDIA GPUs. When it is installed
but fails, as it does without a loaded driver, that is an error, since the GPUs are
there but cannot be read.

### What it does not show

An Apple GPU's load, clocks and temperature come only from `powermetrics`, which needs
`sudo`, so `hw` does not report them. A Mac reports no NVIDIA GPU, and a Mac Mini cannot
check the NVIDIA path on real hardware; its parsing is tested on sample output instead.

## Files

- `mod.rs` holds the types, `Machine`, `MetalLimits`, `Gpu`, `Usage`, `MemoryUse`,
  `Pressure`, `ProcessUse` and `HwError`, and the pure parsers, one per tool. They build
  and test on any machine. With the `hw` feature off, it also holds the `machine()` and
  `usage()` that return `NotBuilt`.
- `probe.rs` is the edge, built only with the `hw` feature. It runs the tools for the
  current system and passes their output to the parsers.
- `metal.rs` reads Metal's limits. It is built only on macOS with the `hw` feature.
- `testdata/` holds the tool output the parser tests read. The macOS files were saved
  from a Mac Mini with an M4 on 2026-10-31. `macos-ps.txt` is in the real `ps` format
  with made-up processes. The Linux and `nvidia-smi` files are written by hand in the
  formats those tools print, since no Linux or NVIDIA machine was at hand.

## Testing

```sh
# The parsers, and on macOS or Linux two tests that read this machine for real
cargo test -p dovata --lib hw

# Without the feature: machine() and usage() return NotBuilt
cargo test -p dovata --lib --no-default-features hw
```

## Who uses it

The `dovata hw` commands. The Jade compiler can call `hw::machine()` and
`Machine::target` to plan for the machine it is building on.
