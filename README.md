# dovata: the JADE OS model memory and inference layout planner

A Rust library the Jade compiler uses at build time. dovata reads a model's GGUF file and
a layout file describing its compute graph, decides where every tensor lives on the
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

The runtime has been removed. What is left are the parts of 2.x that were already about
memory and the model file:

| | |
|---|---|
| [`src/gguf/`](src/gguf/README.md) | GGUF header reader and the `gguf-inspect` CLI: metadata, the tensor index, each tensor's size and file offset, split parts. No llama.cpp. |
| [`src/load/`](src/load/README.md) | preflight, which refuses a file that cannot be planned; `Model`, which treats a split GGUF as one model; the model's shape and KV cost |
| [`src/hub/`](src/hub/README.md) | getting a model: a local path, or one GGUF from the Hugging Face Hub, all parts of a split one. Never quantizes; the code default is the unquantized file. |
| [`src/quantize/`](src/quantize/README.md) | quantizing with llama.cpp's own `llama_model_quantize`, only when the caller asks |
| [`src/plan/`](src/plan/README.md) | the planner: every tensor's region and offset on one target, checked and hashed |
| [`src/moe/`](src/moe/README.md) | MoE expert ranking: which experts each layer uses most |
| [`src/chat/`](src/chat/README.md) | `dovata-chat`, the optional tokenizer and chat-template package for Jade users. Its own crate; dovata never depends on it. |
| [`src/llama/`](src/llama/README.md) | the bindgen'd llama.cpp C API over the stock upstream submodule, kept for quantization and as the reference for ggml's tensor formats |
| [`src/jade-gpu-mon/`](src/jade-gpu-mon/README.md) | a terminal dashboard for VRAM and bandwidth on the GPU box |

Still to come: the layout file format and the built-in layouts in `src/layout/`, the
step that feeds a bound layout to the planner, the `dovata-plan` CLI, and a small
reference backend. The reference backend is for
testing only. It places the tensors for real and runs one simple call over each, to
prove the weights land where the plan says on split memory: VRAM and host RAM together,
more than one GPU, or Apple's unified memory.

Outside `src/`:

| | |
|---|---|
| [`tools/`](tools/README.md) | the Python: `scripts/` local, `ec2/` for the GPU box |
| [`ci/`](ci/README.md) | pinned inputs for building the model |
| `docs/` | `moe-expert-offload.md`, the design of the 2.x expert pager |

## Building and testing

```sh
# The library. Runs anywhere; no model, no GPU, no llama.cpp.
cargo test -p dovata

# With every capability compiled out, the check that the boundaries are real
cargo test -p dovata --lib --no-default-features

# The bundle machinery under the llama.cpp bindings
cargo test -p bundle-kit

# Look at a model file
cargo run --bin gguf-inspect -- /path/to/model.gguf
```

Build by package name with `-p`. `src/llama/bindings-sys` compiles llama.cpp from source,
so `cargo build --workspace` needs the submodule checked out and cmake installed.

There is no CI on the 3.0 branch yet. The 2.x workflows released and published the
runtime, and went with it.
