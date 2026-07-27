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
and stages it; the `ec2-tests` harness drives the `ec2-jade-tests` evals against it.

Downstream, it depends on `src/protocol/` for every wire type and the `Provider` trait
(always), on `src/gguf/` for the preflight (always), and on `src/infer-engine/{api,ffi}`
for the model backend (Linux only).

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
- Some engine submodules (`concurrency`, `bench`) carry `#[allow(dead_code)]`: they are
  reached from env-gated self-tests, not the request path, so the compiler sees them as
  unused in a normal build.
- `examples/` still holds the old socket clients (`smoke_client.rs`, `rlm_client.rs`,
  `bench.rs`). They connect to `~/.jade/llm.sock`, which nothing binds anymore, so they
  have no server to talk to. Treat them as leftovers from the daemon era, not as a way to
  exercise the library.
