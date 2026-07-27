# src/ — all first-party software

Everything this project writes or vendors lives under this one directory: the Rust
crates, the wire-protocol submodule, the Jade provider packages, the C++/CUDA driver,
and the Python tooling. Non-source directories stay at the repo root (`ci/`, `docs/`,
`dist/`, `.github/`).

## Why it is flat

The tree used to nest by kind — a `crates/` folder, a separate `llama-cpp/`
sub-workspace with its own lockfile, a `providers/` folder, a `scripts/` folder of bash.
That made simple questions hard: which lockfile governs this build, is this provider a
`.so` or a library, why does the daemon depend on a crate two levels up. The flattening
put every unit of software as a direct child of `src/`, named for itself, and folded the
inference-engine sub-workspace into the one root workspace so there is a single
`Cargo.lock`.

The organizing idea is **provider-first**. The daemon is a host that routes a request to
a provider; each provider is a peer directory here, whatever language it happens to be
written in. `dovata/` is the host plus the one provider compiled into it, `echo/` is the
reference native provider, `anthropic/` and `openai/` are providers written in Jade.

## What each subtree is

| Directory | What it is |
|---|---|
| `protocol/` | **Git submodule.** `ovata-infer-protocol` — the request/response types and frame codec both this daemon and the Jade language speak. Its own public repo. |
| `dovata/` | The inference daemon: socket host, routing, and the in-process llama.cpp engine. The main crate. |
| `echo/` | Smallest possible native provider. Doubles as the host loader's test fixture. |
| `anthropic/`, `openai/` | Remote-API providers, written in Jade (`.jde`), published as `.so` packages to jadelang.org. Not workspace members. |
| `infer-engine/` | The vendored llama.cpp bindings fork (`api/` safe wrapper, `ffi/` sys crate) plus `moe-driver/`, our C++/CUDA expert-offload driver. |
| `packaging/` | Build/packaging xtask — assembles the daemon and provider `.so`s into `dist/`. |
| `gguf/` | GGUF header reader + `gguf-inspect`. Vets a model file — architecture, hparams, profile match — without llama.cpp, on any machine. |
| `jade-gpu-mon/` | Standalone terminal VRAM/bandwidth monitor. Its own tiny workspace. |
| `scripts/` | Python developer tooling: model download, quantization, provider key setup, dev env. |
| `ec2-tests/` | Python harness that drives the EC2 GPU box (start, sync, build, run evals). |
| `ec2-jade-tests/` | The evals themselves — Jade programs asserting daemon mechanics end to end. |

## How they depend on each other

`protocol/` sits at the bottom and depends on nothing but serde; every other Rust piece
depends on it. `dovata/` depends on `protocol/` always, and on `infer-engine/api` +
`infer-engine/ffi` only on Linux. `echo/` and `packaging/` depend on `protocol/`.
`infer-engine/` depends on nothing here — it is vendored upstream code plus our driver.
The Jade packages depend on nothing in Rust; they conform to the protocol by matching its
frame shapes, and are checked by the evals in `ec2-jade-tests/`.

Nothing depends on `scripts/`, `ec2-tests/`, or `jade-gpu-mon/` — those are tools you run,
not code that is linked.

## Building from here

Commands run from the repo root, not from this directory. See the root `CLAUDE.md` for
the full set; the short version is that `cargo build --workspace` is a Linux operation
(it compiles llama.cpp) and on macOS you use targeted `-p` builds:

```sh
cargo test -p dovata -p ovata-infer-protocol -p ovata-provider-echo
```

## A note on the submodule

`protocol/` is a checkout of another repository. Edits there are commits to *that* repo,
not this one — change it there, push, then bump the submodule pointer here. The same goes
for `infer-engine/ffi/llama.cpp`, which points at our llama.cpp fork at a pinned commit.
