# src/ — the software this repo ships

Everything this project ships or vendors lives under this one directory: the Rust
crates, the wire-protocol submodule, the Jade provider packages, and the C++/CUDA
driver. Things you *run* rather than ship live at the repo root — `tools/` for the
Python tooling, `tests/` for the evals, plus `ci/`, `docs/`, `dist/`, and `.github/`.

## Why it is flat

The tree used to nest by kind — a `crates/` folder, a separate `llama-cpp/`
sub-workspace with its own lockfile, a `providers/` folder, a `scripts/` folder of bash.
That made simple questions hard: which lockfile governs this build, is this provider a
`.so` or a library, why does this crate depend on one two levels up. The flattening
put every unit of software as a direct child of `src/`, named for itself, and folded the
inference-engine sub-workspace into the one root workspace so there is a single
`Cargo.lock`.

## Why the tooling and the evals are not here

They were, for a while, and the flattening is why: pulling `scripts/` out of a nested
tree and dropping it next to the crates was the shortest move at the time. But it made
`src/` mean two things at once. Six of its children were software that gets compiled and
shipped; three were Python you run from a laptop and Jade files that only ever execute on
a GPU box. A reader landing here could not tell which was which, and the doc comment at
the top of `Cargo.toml` claimed a rule the tree did not follow.

So the line is now what it says: `src/` is what ships, `tools/` is what you run, `tests/`
is what you run it against. The move also let `_common.py` sit at the root of `tools/`,
shared by both halves of the Python instead of owned by one of them and reached into by
the other.

The organizing idea is *provider-first*. A Jade program loads a provider `.so` and calls
`infer` in-process; each provider is a peer directory here, whatever language it happens
to be written in. `dovata/` is the local llama.cpp engine, `echo/` is the reference
native provider, `anthropic/` and `openai/` are providers written in Jade.

## What each subtree is

| Directory | What it is |
|---|---|
| `protocol/` | *Git submodule.* `ovata-infer-protocol` — the request/response types, the frame codec, and the `Provider` trait plus its C ABI. Its own public repo. |
| `dovata/` | The local llama.cpp inference engine, shipped as a provider `.so` a Jade program loads in-process. The main crate. |
| `echo/` | Smallest possible native provider. The worked example you copy when writing one. |
| `anthropic/`, `openai/` | Remote-API providers, written in Jade (`.jde`), published as `.so` packages to jadelang.org. Not workspace members. |
| `infer-engine/` | The vendored llama.cpp bindings fork (`api/` safe wrapper, `ffi/` sys crate) plus `moe-driver/`, our C++/CUDA expert-offload driver. |
| `packaging/` | Build/packaging xtask — assembles the engine and provider `.so`s into `dist/`. |
| `gguf/` | GGUF header reader + `gguf-inspect`. Vets a model file — architecture, hparams, profile match — without llama.cpp, on any machine. |
| `jade-gpu-mon/` | Standalone terminal VRAM/bandwidth monitor. Its own tiny workspace, built by nothing here. |

## How they depend on each other

`protocol/` sits at the bottom and depends on nothing but serde; every other Rust piece
depends on it. `dovata/` depends on `protocol/` and `gguf/` always, and on
`infer-engine/api` + `infer-engine/ffi` only on Linux. `echo/` and `packaging/` depend on
`protocol/`.
`infer-engine/` depends on nothing here — it is vendored upstream code plus our driver.
The Jade packages depend on nothing in Rust; they conform to the protocol by matching its
frame shapes, and are checked by the evals in the repo-root `tests/`.

Nothing here depends on `tools/` or on `jade-gpu-mon/`. The arrow points the other way:
`tools/` reads `dovata/profiles/` to regenerate the Jade profile packages, and reads the
`infer-engine/ffi/llama.cpp` submodule pin when it quantizes a model.

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
