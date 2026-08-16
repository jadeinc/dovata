# dovata — the JADE OS inference engine

A library, not a daemon. `dovata` is an `ovata-infer-protocol` provider built as a
`cdylib`, wrapping llama.cpp. A compiled Jade program `dlopen`s it once and calls `infer`
in-process for every `?p` dereference, so the chain is
`jadelang -> dovata.so -> nvidia.ko`, with no socket and no second process.

## Why there is no daemon

It used to be one: a Unix socket at `~/.jade/llm.sock`, and a host that read framed
requests and routed each to a provider.

The work is almost entirely GPU-bound, so a socket, a framing layer, and a second process
in front of it bought nothing and cost real time on every request — plus a second thing to
install, supervise, and explain. The engine was already reachable in-process, because it
implements `Provider` and that is all `export_provider!` needs, so the host came out
without the engine changing at all.

Dropping the process boundary does not mean every program pays for its own copy of the
model. Weights are `mmap`ed out of the GGUF, and MoE cold experts sit in the CPU backend's
default buffer type, so the kernel's page cache shares them across every process that maps
the same `model.gguf`. The model costs the machine once, not once per program.

The one job that genuinely wants a resident process is checking whether the GPU is
healthy. That lives in a separate daemon, outside this repo.

## How the source is organized

Every child of `src/` is one *capability* — a slice of the engine, named for what it does,
behind a cargo feature that can compile it out whole. They are flat: `ls src/` is a list,
and the grouping lives here rather than in a directory tree.

**The shape of a request.** Nothing reaches backwards.

```
dlopen -> load -> model + context built
per request: dispatch -> decode | compose | batch  -> frames out
             each built on tokenize + kv + sample
```

**Admission** — once, before any request

| | |
|---|---|
| [`src/load/`](src/load/README.md) | env, preflight, VRAM sizing. The only thing that can refuse the load. |
| [`src/model/`](src/model/README.md) | per-model vocabulary as data; the profiles themselves |
| [`src/gguf/`](src/gguf/README.md) | GGUF header reader + `gguf-inspect`. Vets a model on a laptop. |

**Primitives over the context** — shared by all three request paths

| | |
|---|---|
| [`src/tokenize/`](src/tokenize/README.md) | text to tokens, and the UTF-8 decoder that spans them |
| [`src/kv/`](src/kv/README.md) | prefix reuse, sequence ops, save/restore |
| [`src/sample/`](src/sample/README.md) | the sampler chain, and the constrained-decode machine |

**The three request paths**

| | |
|---|---|
| [`src/dispatch/`](src/dispatch/README.md) | the `Provider` impl and the routing decision |
| [`src/decode/`](src/decode/README.md) | the default: one sequence, streamed |
| [`src/compose/`](src/compose/README.md) | long-context answering by KV composition |
| [`src/batch/`](src/batch/README.md) | continuous batching. Compiles; nothing calls it. |

**Acceleration**

| | |
|---|---|
| [`src/moe/`](src/moe/README.md) | expert offload: Rust, the C++/CUDA driver, and its llama.cpp bundle |
| [`src/llama/`](src/llama/README.md) | the vendored bindings over stock upstream. The one directory that nests. |

**Providers and the wire**

| | |
|---|---|
| [`src/protocol/`](src/protocol/README.md) | the wire protocol (git submodule) |
| [`src/echo/`](src/echo/README.md) | the reference native provider. Copy it to write one. |
| [`src/anthropic/`](src/anthropic/README.md), [`src/openai/`](src/openai/README.md) | remote providers, written in Jade |

**Outside `src/`** — what you run, rather than what ships

| | |
|---|---|
| [`src/packaging/`](src/packaging/README.md) | the xtask that builds `dist/` |
| [`tools/`](tools/README.md) | all the Python: `scripts/` local, `ec2/` remote |
| [`tests/`](tests/README.md) | the end-to-end evals in Jade, plus their runner |
| [`ci/`](ci/README.md) | pinned model build inputs |
| `docs/` | `blended-rag-rlm.md` (KV composition, with measured numbers), `moe-expert-offload.md` |

## Two entry points, and only one of them reaches Jade

`export_provider!` emits the `ovata_provider_*` C ABI. That is the right surface for a
host that speaks plain C, and it is **not** what the Jade compiler calls. The compiler
loads a provider through its own native-package machinery: `dlopen`, then `jade_pkg_init`
to collect name-to-function-pointer bindings, then a binding called `infer`.

The two drifted apart quietly. The compiler dropped its dependency on the protocol's Rust
half in jade v1.1.30 and kept only `jade/infer.jde`, which declares the *messages* and
says nothing about the entry point — so both sides agreed on `InferRequest`, `Token`,
`Done` and `Error`, and disagreed on how to start the conversation. Loading the engine
from Jade failed with "missing `jade_pkg_init` symbol".

`src/jade_pkg.rs` is the conforming half, and both entry points sit in one library over
one `Provider` impl. Read its module docs before touching either: two things in it are
load-bearing and easy to get wrong.

## Building and testing

```sh
# The pure half. macOS or Linux; no model, no GPU, no llama.cpp.
cargo test                                    # the engine
cargo test -p ovata-infer-protocol -p ovata-provider-echo -p bundle-kit -p packaging

# One test
cargo test sample::control::tests::stop_anchor_split_across_tokens

# With every capability compiled out — the check that the boundaries are real
cargo test --lib --no-default-features

# The shipping library, with the GPU backend (Linux, NVIDIA)
cargo build --release --features cuda

# Vet a model file before a GPU box ever sees it
cargo run --bin gguf-inspect -- /path/to/model.gguf

# The dist/ bundle
cargo run -p packaging -- release --cuda
```

`cargo build --workspace` compiles llama.cpp from source and is effectively a Linux
operation.

## What a green test run does not tell you

Everything named `exec.rs` is `#[cfg(target_os = "linux")]`, and `src/kv/` and
`src/tokenize/` are llama-bound end to end. The suite is fast and broad and compiles none
of it.

That gap is narrower than it looks, though, and worth closing before you push:

```sh
python3 tools/scripts/check-engine.py
```

It copies the repo to a scratch directory, drops the platform gate in the copy, and runs
`cargo check` — so the engine is typechecked here, without slowing the default build or
touching your tree. llama.cpp builds fine on macOS; nothing in the engine's Linux-only
code is actually Linux-specific, only its dependency is.

What still needs the GPU box: anything about CUDA, and anything at run time. The evals in
[`tests/`](tests/README.md) are the real gate.
