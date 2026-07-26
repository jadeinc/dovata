# dovata — the inference daemon

`dovata` is the daemon a Jade program talks to when it wants a model to answer something.
It binds a Unix socket at `~/.jade/llm.sock`, reads length-prefixed
`InferenceRequest`s off the wire, picks a provider to serve each one, and streams back
protocol frames until the request is done. It is both the library (`src/lib.rs`) and the
binary (`src/main.rs`) — the library half is what the unit tests exercise.

## The shape of it: a host plus providers

The daemon is deliberately split into a *host* and a set of *providers*.

The **host** (`src/main.rs`, `src/host.rs`) owns everything that is true regardless of
which model answers: the socket, the framing, the `health` and `stats` requests, the
`Meta` frame that names the provider, and the routing rules that choose one. It never
touches a model.

A **provider** is anything that implements `ovata_infer_protocol::Provider`. Two kinds
exist, and the difference is only in how they are loaded:

- The **local** provider is the llama.cpp engine in `src/engine/`. It is compiled *into*
  the daemon and called directly — the host holds it as `ProviderHandle::Local`, with no
  `dlopen` and no FFI hop. It is Linux-only because it links vendored llama.cpp.
- **Remote** providers are `.so` packages the host `dlopen`s through `src/providers.rs`,
  driven over a small C ABI. `echo` is the native reference one; the Anthropic and OpenAI
  providers are Jade packages that speak the same protocol.

That split is why the daemon builds on macOS at all. Everything except `src/engine/` is
pure Rust with no llama.cpp dependency, so the host, the routing, the control machine,
the scheduler, and the profiles all compile and unit-test on a laptop. On macOS you get a
host with no local engine; remote providers still load and serve.

## Why the pure core is so large

Most of what looks like "inference logic" is not actually model-bound, and putting it in
the pure half means it can be tested without a GPU, a model file, or Linux. Constrained
decoding is a state machine over token text. Continuous batching is bookkeeping about
which sequence feeds which token at which position. Over-context routing is string
splitting. All of it was once tangled inside the generation loop and untestable; pulling
it out is why `cargo test -p dovata` runs in well under a second and still covers the
tricky parts.

## What each file does

**Entry points**

- `src/main.rs` — the binary. Resolves the socket path under `$HOME` and calls `host::run`.
- `src/host.rs` — the server loop: accept, decode a request, answer `health`/`stats`
  directly, otherwise select a provider and stream its frames to the socket. Holds
  `select_provider_index`, the pure routing rule, so routing is unit-tested on its own.
- `src/lib.rs` — the library root; its module list is the map of the pure core.

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
- `src/providers.rs` — the host half of the plugin boundary: `dlopen`s a provider `.so`,
  checks its ABI version, and drives its C-ABI. Also defines `ProviderHandle`, the enum
  that lets the host treat the local engine and a plugin the same way.
- `src/profile/` — per-model token vocabulary (tool-call delimiters, turn markers) as
  data, so swapping a `.gguf` does not break tool calling. `mod.rs` is the typed schema
  and the registry of bundled profiles; `stream.rs` is the profile-driven parser that
  splits a token stream into content and complete tool calls.

**The Linux engine**

- `src/engine/` — the llama.cpp-bound half. See its own [README](src/engine/README.md).

**Everything else**

- `profiles/` — the bundled model profiles, one TOML per model plus a Jade counterpart.
  See its [README](profiles/README.md).
- `examples/` — socket clients you run against a live daemon: `smoke_client.rs` (one
  plain request, proves the local path decodes), `rlm_client.rs` (upload a large context,
  query it by handle and inline), `bench.rs` (TTFT and decode tok/s measured at the
  protocol level, no Jade runtime in the path).

## Who uses it, and what it uses

Upstream, the **Jade language** is the client — a compiled Jade program opens the socket
once and sends a request per `?p` dereference. The **JADE OS bootloader** installs the
release binary at `/jade/inference/`. `src/packaging/` builds and stages it; the
`ec2-tests` harness starts it and runs the `ec2-jade-tests` evals against it.

Downstream, it depends on `src/protocol/` for every wire type (always) and on
`src/infer-engine/{api,ffi}` for the model backend (Linux only).

## Configuration

All environment, all `JADE_*`. The host reads:

- `JADE_LOCAL_ENGINE` — set `0`/`off`/`false` to disable the in-process engine.
- `JADE_PROVIDERS` — comma-separated `.so` paths to load, each optionally
  `path.so@config.json`.
- `JADE_DEFAULT_PROVIDER` — who serves unrouted requests. Defaults to `dovata` if loaded.
- `JADE_MODEL_ROUTES` — `prefix=provider,...`; a request whose `model` starts with
  `prefix` goes to that provider, first match wins.

The engine's own settings (`JADE_MODEL_PATH`, `JADE_CTX_SIZE`, `JADE_KV_TYPE_*`,
`JADE_MOE_*`, …) are parsed in `src/config.rs` — read that file for the current list and
defaults rather than trusting a copy here.

## Building and testing

```sh
# Pure core — works on macOS and Linux, no model needed.
cargo build -p ovata-provider-echo      # the loader test needs this fixture present
cargo test -p dovata

# One test
cargo test -p dovata control::tests::stop_anchor_split_across_tokens

# The full daemon with the local engine (Linux, GPU):
JADE_MODEL_PATH=/jade/inference/model.gguf cargo run --release -p dovata --features cuda
```

`cargo test -p dovata` on macOS compiles no llama.cpp and needs no model. One test does
need a built artifact: `providers::tests::loads_echo_fixture_and_round_trips_frames`
`dlopen`s the `echo` library out of `target/debug/`, so build that crate first.

## Gotchas

- **Nothing in `src/engine/` is covered by a Mac test run.** It is `cfg`'d out entirely.
  Treat a green macOS test suite as saying nothing about the engine.
- The `cuda` feature is a no-op off Linux — it forwards to the vendored llama-cpp-2,
  which is not compiled in on other targets.
- Some engine submodules (`concurrency`, `bench`) carry `#[allow(dead_code)]`: they are
  reached from env-gated self-tests, not the request path, so the compiler sees them as
  unused in a normal build.
