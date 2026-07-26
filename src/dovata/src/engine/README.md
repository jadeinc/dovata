# engine/ — the local llama.cpp backend

This is the half of the daemon that actually runs a model. `Dovata` implements the
protocol's `Provider` trait, owns the llama.cpp backend, model, and context, and serves
requests in-process — the host holds it as `ProviderHandle::Local` and calls it directly,
with no `dlopen` and no FFI hop.

Everything here links vendored llama.cpp, so the whole module is behind
`#[cfg(target_os = "linux")]`. On macOS the daemon builds without it.

## Why the split from the rest of the crate

The daemon's logic used to live in one large `inference.rs` where the decode loop, the
grammar handling, the scheduling, and the llama.cpp calls were interleaved. That made the
interesting parts impossible to test — you needed a GPU and a model file to check whether
a stop sequence split across two tokens was handled correctly.

So the pure logic moved up into the crate root (`control.rs`, `sched.rs`, `route.rs`,
`config.rs`) where it is unit-tested on any platform, and what stayed here is only the
llama-bound execution: allocate a batch, call `decode`, sample, read the KV cache. The
rule of thumb when adding code is that if it can be written without touching a
`llama_cpp_2` type, it belongs upstairs, not here.

## The three request paths

A request lands in one of three places depending on its shape:

1. **Standard** (`standard.rs`) — the default. One sequence, KV prefix reuse against
   what is already resident, constrained decode, tokens streamed out as produced.
2. **RLM** (`rlm.rs` → `compose.rs`) — when the prompt exceeds the context window. The
   full corpus stays in CPU RAM as token blocks; per query the engine retrieves the
   relevant blocks, splices their precomputed K/V into one attention window, and answers
   in a *single* decode. There is no per-branch re-prompting and no agentic decision
   calls — the recursion lives in the KV cache, not in a prompt loop. See
   `docs/blended-rag-rlm.md`.
3. **Concurrent** (`concurrency.rs`) — continuous batching: N sub-inferences against one
   resident model, every `decode` packing tokens from every live sequence into one
   forward pass. All the scheduling decisions come from the pure `crate::sched`; this
   file just translates plans into batches and feeds sampled tokens back.

## What each file does

- `mod.rs` — `Dovata` itself: model/context lifecycle, the `Provider` impl, request
  dispatch to one of the paths above, and the FFI declaration for `moe_driver_init`.
- `kv.rs` — the KV-cache primitives the paths share. `kv_cache_type` maps our pure
  `KvType` onto llama.cpp's enum; `reuse_prefix` keeps the leading tokens a new prompt
  shares with the resident one and evicts the rest; `prefill_shared` prefills a common
  prefix once and copies its KV into N sequences.
- `standard.rs` — the single-shot path.
- `compose.rs` — the KV-composition engine. `BlockStore` holds the corpus; two pipelines
  share a COMBINE step — plain IDF retrieval, and a blended variant that retrieves N
  candidates then reranks them by cross-encoder query-likelihood before composing top-k.
- `rlm.rs` — RLM request dispatch: `StoredContext`, the upload/handle lifecycle, and the
  host-RAM `KvBlockCache` so an uploaded context is tokenized only once.
- `concurrency.rs` — the continuous-batching driver over `crate::sched`.
- `bench.rs` — developer self-tests and benchmarks, *not* a request path.

## The self-tests are a startup hijack

`bench.rs::dispatch` runs once at the end of daemon startup. If any of its `JADE_*`
variables is set it runs that one routine, prints to stderr, and the daemon **exits
without binding the socket**. With none set it returns `false` and the normal server path
proceeds.

That is worth knowing before you debug a daemon that "starts and immediately dies" — a
stray `JADE_KV_BENCH` in the environment does exactly that. The gates are
`JADE_RLM_SELFTEST`, `JADE_KV_ROUNDTRIP`, `JADE_KV_COMPOSE`, `JADE_COMPOSE_DEMO`,
`JADE_COMPOSE_MULTI`, `JADE_KV_BENCH`, `JADE_COMPOSE_CACHE`, `JADE_GEN_BENCH`,
`JADE_RLM_OVERWINDOW`, `JADE_BLENDED`, and `JADE_FRONTIER`, checked in that order.

## Who uses it, and what it uses

Only `crate::host` constructs it, through `ProviderHandle::Local`. It uses the pure
modules in the crate root (`control`, `sched`, `cache`, `config`, `profile`, `route`,
`moe`), `ovata-infer-protocol` for `Frame`/`FrameSink`/`InferenceRequest`, and
`llama-cpp-2` + `llama-cpp-sys-2` from `src/infer-engine/`. The MoE paging machinery it
calls over FFI lives in `src/infer-engine/moe-driver/`.

## Building and gotchas

There is no way to compile this on a Mac. It builds as part of the daemon on Linux:

```sh
cargo build --release -p dovata --features cuda
```

- `concurrency` and `bench` carry `#[allow(dead_code)]` — they are reached only from the
  env-gated self-tests, so a request-path build sees them as unused.
- `Dovata` carries a hand-written `unsafe impl Send/Sync` and leaks the model and backend
  to `'static` via `Box::leak`. Both are load-bearing for the context lifetime; re-read
  the justification in `mod.rs` before changing how the model is owned.
- The composition paths need the unified KV cache (`JADE_KV_UNIFIED`, on by default).
