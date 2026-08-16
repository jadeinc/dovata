# dovata — the inference engine

`dovata` is the library a Jade program loads when it wants a model to answer something.
It is a *provider package*: an `impl Provider` from `ovata-infer-protocol`, exported
across the C ABI and built as a `cdylib`. The program `dlopen`s it once and calls `infer`
in-process. The chain is `jadelang -> dovata.so -> nvidia.ko`, which is the ordinary shape
for GPU work.

## Why there is no daemon anymore

`dovata` used to be a daemon. It bound a Unix socket at `~/.jade/llm.sock`, and a host
inside it read framed requests and routed each one to a provider. That is gone, and so are
`main.rs`, `host.rs`, and `providers.rs`.

The reason is that the work is almost entirely GPU-bound. Putting a socket, a framing
layer, and a second process in front of it bought nothing and cost real time on every
request, plus a second thing to install, supervise, and explain. The engine was already
reachable in-process, because it implements `Provider` and that is all `export_provider!`
needs, so the host came out without the engine changing at all.

The one job that genuinely wants a resident process is checking whether the GPU is
healthy. That is a separate concern and now lives in a separate daemon, outside this repo.

Dropping the process boundary does not mean every Jade program pays for its own copy of
the model. Weights are `mmap`ed out of the GGUF, and the MoE cold experts sit in the CPU
backend's default buffer type, so the kernel's page cache shares them across every process
that maps the same `model.gguf`. The model costs the machine once, not once per program.

## The shape of it: a thin llama.cpp edge over a large pure core

Everything except `src/engine/` is pure Rust with no llama.cpp dependency, so the config
parsing, the control machine, the scheduler, the routing, the preflight, and the profiles
all compile and unit-test on a laptop. `src/engine/` is `#[cfg(target_os = "linux")]`,
because it links vendored llama.cpp. On macOS the cdylib builds as an empty stub: there is
no engine to export.

Most of what looks like "inference logic" is not actually model-bound, which is why the
pure half is the big one. Constrained decoding is a state machine over token text.
Continuous batching is bookkeeping about which sequence feeds which token at which
position. Over-context routing is string splitting. All of it was once tangled inside the
generation loop and untestable; pulling it out is why `cargo test -p dovata` runs in well
under a second and still covers the tricky parts.

## What each file does

**Entry point**

- `src/lib.rs` — the library root. Its module list is the map of the pure core, and on
  Linux it ends with `export_provider!(engine::Dovata)`, which is what makes this crate's
  cdylib a provider package.
- `src/jade_pkg.rs` — the *other* entry point, and the one a Jade program actually uses.
  See below; it is easy to assume `export_provider!` is enough and it is not.

### Two entry points, and only one of them reaches Jade

`export_provider!` emits the `ovata_provider_*` C ABI. That is the right surface for a
host that speaks plain C, and it is **not** what the Jade compiler calls. The compiler
loads a provider through its own native-package machinery: `dlopen`, then
`jade_pkg_init` to collect name-to-function-pointer bindings, then a binding called
`infer` taking an `InferRequest` and returning an array of frames.

The two drifted apart quietly. The compiler dropped its dependency on the protocol's
Rust half in jade v1.1.30 and kept only `jade/infer.jde`, which declares the *messages*
and says nothing about the entry point. So both sides agreed on `InferRequest`, `Token`,
`Done` and `Error`, and disagreed on how to start the conversation — while the protocol's
own docs still said a compiled Jade program calls `ovata_provider_new`. Loading the
engine from Jade failed with "missing `jade_pkg_init` symbol", and the eval suite could
not run at all. ovata-infer-protocol v0.5.2 now names the compiler's loader as the
authority.

`src/jade_pkg.rs` is the conforming half. Both entry points sit in one library over one
`Provider` impl, so nothing about the engine changed and a non-Jade host keeps the ABI it
already links. The marshalling is pure and unit-tested on any machine — it is only the
engine call behind it that is Linux-only — and the tests pin the frame type names and
field names, because a rename there decodes as a bag of nils rather than failing loudly.

Two things in that file are load-bearing and easy to get wrong. Every byte of a returned
tree must come from libc `malloc`, because the compiler frees it with libc `free` and the
process holds two allocators. And `jade_pkg_abi_version` must report 2, the value ABI
structs cross under since jade v1.1.31; the compiler refuses a package it cannot talk to.

**The pure core** (no llama.cpp, tested everywhere)

- `src/config.rs` — reads the `JADE_*` environment into a `Config`. Models the KV-cache
  type as our own `KvType` enum so the flash-attention validation rule stays testable.
- `src/control.rs` — the constrained-decoding state machine: anchor spans, stop-anchor
  buffering across split tokens, JSON depth tracking to retire the grammar. The engine
  drives it once per generated token.
- `src/prompt.rs` — renders a parsed message envelope into a specific model's chat text.
  The envelope format itself belongs to the protocol crate; only the model-specific
  rendering lives here.
- `src/cache.rs` — bounded LRU cache of past *responses*, keyed by a hash of the request
  fields that determine output. Distinct from KV-cache prefix reuse, which happens during
  prefill.
- `src/sched.rs` — the multi-sequence continuous-batching scheduler. Decides which
  sequence contributes which tokens to each forward pass and when KV may be admitted or
  retired. Pure bookkeeping; the Linux glue only executes its plans.
- `src/route.rs` — when a prompt exceeds the context window, splits it into a retrievable
  *corpus* and a *question* for the KV-composition engine. No tokenizer needed.
- `src/moe.rs` — loads and validates the per-layer `expert_order` asset and flattens it
  for hand-off to the C driver over FFI, so the C++ side never parses JSON.
- `src/profile/` — per-model token vocabulary (tool-call delimiters, turn markers) as
  data, so swapping a `.gguf` does not break tool calling. `mod.rs` is the typed schema
  and the registry of bundled profiles; `stream.rs` is the profile-driven parser that
  splits a token stream into content and complete tool calls.
- `src/preflight.rs` — the load-time gate. Reads the model's GGUF header with the `gguf`
  crate *before* llama.cpp loads it, checks the metadata llama.cpp requires, selects the
  profile on architecture as well as name, and refuses with a specific message when
  something is wrong. Pure apart from one file read, so it is unit-tested on macOS against
  synthetic headers.

**The Linux engine**

- `src/engine/` — the llama.cpp-bound half. See its own [README](src/engine/README.md).

## Loading is where policy runs

`Dovata::from_config` in `src/engine/mod.rs` is the only entry point now. The library is
loaded and the provider constructed in one step by whoever `dlopen`s it, so there is no
earlier place to stand and no separate startup phase. Everything that used to happen
before the socket opened happens there: read the environment into a `Config`, validate it,
preflight the GGUF, select the profile, then load the model and build the context.

A failure returns `Err`, which fails the load. The loader sees a null handle instead of a
half-configured provider. That is a better outcome than the old one, where a bad model
took down a running daemon.

## Who uses it, and what it uses

Upstream, the **Jade language** is the client. A compiled Jade program loads the library
and serves every `?p` dereference through it, in its own process. The **JADE OS
bootloader** installs the release library at `/jade/inference/`. `src/packaging/` builds
and stages it; the repo-root `tests/` suite is run against it on the GPU box, which
`tools/ec2/` gets you onto.

Downstream, it depends on `src/protocol/` for every wire type and the `Provider` trait
(always), on `src/gguf/` for the preflight (always), and on `src/infer-engine/{api,ffi}`
for the model backend (Linux only).

### The protocol floor is declared twice, for two different failures

`src/protocol` is a submodule and the dependency is by path, so nothing about a path
dep alone stops the pointer moving somewhere the engine does not work.

`Cargo.toml` carries `version = ">=0.5.0"` next to the path. That is not redundant:
cargo checks the requirement against the submodule's own version, so rolling the
pointer *back* below the floor fails to resolve, with a message naming the versions,
rather than failing to compile somewhere further in. 0.5.0 is the floor because the
provider ABI gained `ovata_provider_protocol_version` there, which `packaging` reads
off each provider `.so` to validate it.

The tests in `protocol_compat` (`src/lib.rs`) pin the two numbers a semver bump would
otherwise move silently: `PROTOCOL_VERSION`, the wire version the engine speaks, and
`PROVIDER_ABI_VERSION`, what the shipped `.so` advertises. The crate's semver and the
wire version move independently — 0.4.0 and 0.5.1 both speak wire 3 — so the floor
cannot stand in for either. Changing one of those numbers should be deliberate, and
the test is what makes it so.

## Capabilities

The engine is built from capabilities: slices that compile out whole, the Rust half and
the llama.cpp seam together, each behind one cargo feature.

| Feature | The capability | Pure half | llama-bound half |
|---|---|---|---|
| `compose` | long-context answering by KV composition | `src/route.rs` | `src/engine/{compose,rlm}.rs` |
| `batch` | continuous batching across sequences | `src/sched.rs` | `src/engine/concurrency.rs` |
| `moe` | MoE expert offload to host RAM | `src/moe/mod.rs` | `src/moe/ffi.rs` + the C driver |

Every one is in `default`, so `cargo build -p dovata` is still the shipping artifact.
They are not there to make the build configurable. They are there because a boundary you
cannot switch off is not a boundary — if a capability's edges are real, the crate compiles
and passes its tests without it, and CI checks exactly that with
`cargo test -p dovata --lib --no-default-features`. The second reason arrives with the
llama.cpp bundles: a capability that patches llama.cpp needs one switch that gates its
patch series and its Rust caller together, or the two drift.

Turning one off makes the engine *refuse* the requests it can no longer serve, never
answer them worse. Without `compose`, an over-context prompt gets an `Error` frame instead
of being truncated to fit — a truncated prompt returns a confident answer to a question
nobody asked. Without `moe`, `JADE_MOE_DRIVER=1` fails the load instead of quietly running
stock, which produces the same output at roughly three times the VRAM and surfaces as an
OOM with no mention of experts.

## Configuration

All environment, all `JADE_*`, all read by `Config::from_env` when the library loads.

- `JADE_ALLOW_UNPROFILED` — load even when no profile matches the model. Off by default:
  an unprofiled model runs on bare ChatML, which generates text fine and mis-parses every
  tool call, so the engine refuses rather than fail silently. Set it when you are
  deliberately serving a model for plain completion.

The rest (`JADE_MODEL_PATH`, `JADE_CTX_SIZE`, `JADE_KV_TYPE_*`, `JADE_MOE_*`, …) are
parsed in `src/config.rs`. Read that file for the current list and defaults rather than
trusting a copy here.

## Loading against a new model

`JADE_MODEL_PATH` is the only thing that has to change, but the preflight decides whether
the load proceeds. It refuses a model whose metadata is incomplete, and refuses one no
profile matches. To see what it will say before you run the program:

```sh
cargo run -p gguf --bin gguf-inspect -- /path/to/model.gguf
```

If it reports no profile match, `--profile-stub` writes the starter TOML. See
[`src/gguf/README.md`](../gguf/README.md) for what llama.cpp requires of a model file and
why architecture, not name, is the identifier that matters.

## Building and testing

```sh
# Pure core — works on macOS and Linux, no model needed.
cargo test -p dovata

# One test
cargo test -p dovata control::tests::stop_anchor_split_across_tokens

# The shipping library, with the GPU backend (Linux, NVIDIA):
cargo build --release -p dovata --features cuda
```

`cargo test -p dovata` on macOS compiles no llama.cpp and needs no model.

## Gotchas

- **Nothing in `src/engine/` is covered by a Mac test run.** It is `cfg`'d out entirely.
  Treat a green macOS test suite as saying nothing about the engine.
- The `cuda` feature is a no-op off Linux — it forwards to the vendored llama-cpp-2,
  which is not compiled in on other targets.
- `src/engine/concurrency.rs` carries `#[allow(dead_code)]` and nothing calls it. Its
  only caller was the env-gated selftest module, which is gone; the continuous-batching
  driver is kept because the RLM fan-out design needs it, but treat it as unverified.
  See the [engine README](src/engine/README.md).
