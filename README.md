# dovata: the JADE OS model memory and inference layout planner

A Rust library the Jade compiler uses at build time. dovata reads a model's GGUF file
and a layout file describing its compute graph, decides where every tensor lives on the
target machine, and quantizes the weights with existing open-source tools. It hands the
compiler that placement as data: for each tensor, a memory region and a byte offset into
it. The compiler generates the GPU code from that. dovata never runs a model.

The first targets are NVIDIA GPUs on Linux, Apple Silicon on macOS, and CPU only.

## What changed in 3.0

Up to 2.x, dovata was a runtime: a provider library a running Jade program loaded to do
inference through llama.cpp, with request paths, sampling, a wire protocol, and remote
providers next to it. That code is on `main` and in the `v2.*` tags.

3.0 moves the work to build time. The work is GPU-bound, and the compiler is what should
emit the machine code, so a runtime engine between the program and the GPU was overhead.
The interface is a Rust crate the compiler takes as a cargo dependency; there is no
protocol and no C ABI.

## Where the rewrite stands

The runtime has been removed. dovata now has these parts:

| | |
|---|---|
| [`src/gguf/`](src/gguf/README.md) | GGUF header reader and the `gguf-inspect` CLI: metadata, the tensor index, each tensor's size and file offset, split parts. No llama.cpp. |
| [`src/load/`](src/load/README.md) | preflight, which refuses a file that cannot be planned; `Model`, which treats a split GGUF as one model; the model's shape and KV cost |
| [`src/layout/`](src/layout/README.md) | layouts: each architecture's compute graph as data in ggml's op names, and the binder that resolves one against a model. A built-in layout is one supported architecture. |
| [`src/hf_hub/`](src/hf_hub/README.md) | getting a model: a local path, or one GGUF from the Hugging Face Hub, all parts of a split one. Never quantizes; the code default is the unquantized file. |
| [`src/quantize/`](src/quantize/README.md) | quantizing by running llama.cpp's `llama-quantize` program, only when a program such as the Jade compiler asks |
| [`src/target/`](src/target/README.md) | the machine a plan is for: its memory, largest GPU buffer and page size, as Apple, CPU or CUDA, and the kinds of memory a region can be |
| [`src/window/`](src/window/README.md) | splits a mapped model file into page-aligned windows no larger than one GPU buffer, so every weight fits in one |
| [`src/arena/`](src/arena/README.md) | packs one step's short-lived tensors into shared memory, reusing bytes between tensors that are never alive together |
| [`src/memplan/`](src/memplan/README.md) | the planner: binds a layout, works out each tensor's lifetime, gives every tensor a region and offset on one target using `window` and `arena`, checks and hashes the result, and carries the step's ops for the compiler |
| [`src/moe/`](src/moe/README.md) | MoE expert ranking: which experts each layer uses most |
| [`src/chat/`](src/chat/README.md) | `dovata-chat`, the optional tokenizer and chat-template package for Jade users. Its own crate; dovata never depends on it. |
| [`src/machine/`](src/machine/README.md) | the current machine: its memory, page size, GPUs and their limits, and how memory is used right now |
| [`src/cli/`](src/cli/README.md) | `dovata`, the command line: `dovata machine` reports this machine, and `dovata plan` plans a model, downloading or quantizing it first when asked |
| [`src/refback/`](src/refback/README.md) | `refback`, the test-only reference backend: sets up a plan's memory for real and checks every tensor against the model file. Its own crate, never published. |

Still to come: CUDA in `refback`, which needs an NVIDIA machine to test on; placing
part of a model in host memory when it does not fit in VRAM, which waits on a design;
the hybrid architectures with recurrent state, such as Qwen3.6; and CI.

## Why each capability sits where it does

Every child of `src/` is one capability, named for what it does, and the ones that can
be compiled out sit behind a cargo feature. They are flat rather than nested, because
the grouping that helps a reader is the table above, not a directory tree. A flat tree
invites one question about each directory: why is it here, and not inside a neighbor?

- `load/` is separate from `gguf/` because it is policy. The GGUF reader reports what
  any file says, even a truncated one, and `load/` decides whether that is enough to
  plan from.
- `gguf/` has no dependencies, because it has to run on a laptop against a model no GPU
  has touched. `load::preflight` reads every model through it, and `gguf-inspect`, a
  binary of this package, is its command-line tool.
- `hf_hub/` is a peer of `load/`, not part of it, because it is the one capability that
  talks to the network. Its pure half, which picks the file to download, always builds;
  the download itself is behind the `hf-hub` feature.
- `quantize/` is never called implicitly. Fetching and planning never quantize, so a
  program that wants a smaller model, such as the Jade compiler, calls it directly. The
  quant-type table always builds; running `llama-quantize` is behind the `quantize`
  feature. llama.cpp is not in this repo: `llama-quantize` must be installed on the
  machine, and it is the only part of llama.cpp dovata uses.
- `layout/` is separate from the planner because a layout is the contract with the Jade
  compiler's kernels, and the planner only decides where its tensors live. Each built-in
  layout is one architecture dovata supports.
- `memplan/` is separate from `layout/` because the same layout plans differently on every
  target machine. It hands the compiler an address table of region and offset pairs.
- `target/` is separate from `memplan/` because four capabilities share it: `memplan/`,
  `machine/`, `cli/` and `refback/`. It is not in `machine/` because `machine/` can be
  compiled out, and the planner needs a target in every build.
- `window/` and `arena/` are separate from `memplan/` because each is one packing method,
  a pure function of sizes and offsets that knows nothing about models. The planner
  decides what goes where; `window` decides how a mapped file fits in GPU buffers, and
  `arena` decides how short-lived tensors share bytes.
- `moe/` is behind the `moe` feature. The planner will use its ranking to decide which
  experts stay resident in VRAM.
- `machine/` is separate from `memplan/` because the planner never reads hardware: the Jade
  compiler may build on one machine for another. `machine` reads only the machine it
  runs on, and `Machine::target` turns that into a planner target when the two are the
  same. Its parsers always build; running the system tools is behind the `machine`
  feature.
- `cli/` is the one place that prints for a person. The library returns values, and the
  `dovata` program formats them.
- `refback/` is a separate crate because it is the one part that runs real memory: it
  maps files, allocates buffers and talks to Metal, which takes unsafe code and Apple's
  framework. Neither belongs in a library the compiler links.

Outside `src/`:

| | |
|---|---|
| `docs/` | `moe-expert-offload.md`, the design of the 2.x expert pager |

## Building and testing

```sh
# The library. Runs anywhere; no model and no GPU. The quantize tests also run
# llama.cpp's llama-quantize, and skip when it is not installed: brew install llama.cpp
cargo test -p dovata

# With every capability compiled out, the check that the boundaries are real
cargo test -p dovata --lib --no-default-features

# Look at a model file
cargo run --bin gguf-inspect -- /path/to/model.gguf

# This machine's memory and GPUs, and their use
cargo run --bin dovata -- machine info
cargo run --bin dovata -- machine watch

# Plan a model for this machine
cargo run --bin dovata -- plan /path/to/model.gguf

# Check a plan on real memory: the CPU, then Metal on a Mac
cargo test -p refback
cargo test -p refback --features metal
```

The repo is pure Rust: nothing builds C or C++ code, so `cargo build --workspace` needs
only a Rust toolchain.

There is no CI on the 3.0 branch yet. The 2.x workflows released and published the
runtime, and went with it.
