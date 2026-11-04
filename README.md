# dovata: the JADE OS model memory and inference layout planner

A Rust library the Jade compiler uses at build time. dovata reads a model's GGUF file
and a layout file describing its compute graph, decides where every tensor lives on the
target machine, and quantizes the weights with existing open-source tools. It hands the
compiler that placement as data: for each tensor, a memory region and a byte offset into
it. The compiler generates the GPU code from that.

dovata is also the process that holds models in memory. `dovata start` starts it, with
no models; `dovata add` loads a model into it, from a file or a Hugging Face link; and
it runs until `dovata kill` stops it. There is one dovata process per user, so every
model's memory is managed in one place. A compiled Jade program maps a model's memory
into itself, every region where its offsets expect it, and runs its own kernels on it.
All of it works the same from the command line and from code. dovata never runs a
model's math.

The first targets are NVIDIA GPUs on Linux, Apple Silicon on macOS, and CPU only.

## What changed in 3.0

Up to 2.x, dovata was a runtime: a provider library a running Jade program loaded to do
inference through llama.cpp, with request paths, sampling, a wire protocol, and remote
providers next to it. That code is in the `v2.*` tags.

3.0 moves the work to build time. The work is GPU-bound, and the compiler is what should
emit the machine code, so a runtime engine between the program and the GPU was overhead.
The interface is a Rust crate the compiler takes as a cargo dependency, with no C ABI.

The process came back on 2026-11-03, without the request path. The 2.x daemon took a
message per inference step. The dovata process takes messages only to add and remove
models, report, hand out memory and stop; the program's kernels then use that memory directly. The
messages are dovata's own, between the library and the program of the same version, not
a protocol other projects implement.

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
| [`src/chat_template/`](src/chat_template/README.md) | `dovata-chat-template`, an optional package for Jade users: builds a model's prompt from the chat template in its GGUF, and turns the prompt into token ids and back. It does not run the model. Its own crate; dovata never depends on it. |
| [`src/machine/`](src/machine/README.md) | the current machine: its memory, page size, GPUs and their limits, and how memory is used right now |
| [`src/host/`](src/host/README.md) | the dovata process: holds every model added to it, loaded, checks each fits beside the others, reports how much is resident, and hands a model's memory to programs to map as file descriptors over a Unix socket; and the library calls that start it, add and remove models, map them and stop it |
| [`src/cli/`](src/cli/README.md) | `dovata`, the command line: `dovata start`, `add`, `remove`, `status` and `kill` run the dovata process and its models, `dovata machine` reports this machine, and `dovata plan` plans a model, downloading or quantizing it first when asked |
| [`src/refback/`](src/refback/README.md) | `refback`, the test-only reference backend: sets up a plan's memory for real and checks every tensor against the model file. Its own crate, never published. |

Still to come: CUDA in `refback` and in the dovata process, which need an NVIDIA
machine to test on; placing
part of a model in host memory when it does not fit in VRAM, which waits on a design;
and the hybrid architectures with recurrent state, such as Qwen3.6.

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
- `host/` is separate from `memplan/` because the planner is a pure function the
  compiler calls at build time, and `host` is a running process that maps memory and
  passes it between processes. It is not in `refback/` because `refback` checks a plan
  and is never published, and `host` is what ships. It is behind the `host` feature,
  and its unsafe calls are all in one file.
- `cli/` is the one place that prints for a person. The library returns values, and the
  `dovata` program formats them.
- `refback/` is a separate crate because it checks a plan through Metal: it builds
  Metal buffers and runs a shader, which takes Apple's framework and a good deal of
  unsafe code that only a checker needs. `host` maps memory too, but through libc
  alone, behind its feature, since it is part of what ships.

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

# Start the dovata process, hold a model in it, look at it, and stop it
cargo run --bin dovata -- start
cargo run --bin dovata -- add https://huggingface.co/unsloth/Qwen3-0.6B-GGUF --quant Q4_K_M
cargo run --bin dovata -- status
cargo run --bin dovata -- kill

# The process test: starts the real program, adds a model, maps it, and stops it
cargo test -p dovata --test host-process

# Check a plan on real memory: the CPU, then Metal on a Mac
cargo test -p refback
cargo test -p refback --features metal
```

The repo is pure Rust. One dependency, `ring`, which the Hugging Face downloads use for
HTTPS, compiles a little C, so a build needs a C compiler too. Xcode's command line tools
and every Linux distribution's build tools have one.

## Installing

The programs `dovata` and `gguf-inspect` are released for Apple Silicon Macs and for
x86_64 Linux. This installs the latest release into `/usr/local/bin`:

```sh
curl -fsSL https://raw.githubusercontent.com/jadeinc/dovata/main/install.sh | sh
```

`DOVATA_INSTALL_DIR` picks another directory. The script checks the download against
the release's `checksums.txt` before installing anything.

The Jade compiler takes the library as a cargo dependency on a release tag:

```toml
dovata = { git = "https://github.com/jadeinc/dovata", tag = "v3.0.1" }
```

## Releasing

A release is a version bump. Change `version` in `Cargo.toml`, run `cargo build` so
`Cargo.lock` follows, and merge to `main`. When CI passes on `main`, it tags the new
version and the release workflow publishes the programs. The details are in
[`.github/workflows/README.md`](.github/workflows/README.md).
