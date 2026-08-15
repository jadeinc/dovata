# engine/ — the local llama.cpp backend

This is the half of the crate that actually runs a model. `Dovata` implements the
protocol's `Provider` trait and owns the llama.cpp backend, model, and context. It is what
`export_provider!` exports across the C ABI, so a Jade program that loads `dovata.so` is
calling straight into this code, in its own process.

Everything here links vendored llama.cpp, so the whole module is behind
`#[cfg(target_os = "linux")]`. On macOS the crate builds without it, as an empty cdylib.

## Why the split from the rest of the crate

The old logic used to live in one large `inference.rs` where the decode loop, the
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
  `from_config` is the load-time gate: environment into `Config`, validate, preflight the
  GGUF, pick the profile, then build the model and context. It is the only entry point, so
  anything that must happen before the first request happens there, and an `Err` fails the
  library load instead of producing a half-built provider.
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

## The self-tests are gone, and the JADE_* gates with them

There used to be a `bench.rs` here: a `dispatch` function that ran once at the end of
daemon startup, checked a list of `JADE_*` variables, and if one was set ran that
routine, printed to stderr, and exited before the socket was bound.

Dropping the daemon left it with no startup phase to hang off, and exiting the process
would now take the caller's program with it. So `dispatch` sat defined and never called,
and `JADE_KV_BENCH` and its siblings did nothing — 903 lines the compiler kept alive and
no path reached. It has been deleted. The measured numbers it produced are recorded in
[`docs/blended-rag-rlm.md`](../../../../docs/blended-rag-rlm.md), and the routines
themselves are recoverable from git history if you want to re-run one.

If you do bring one back, wire it to an explicit entry point rather than to a load-time
hijack. Deleting it also left `concurrency` with no caller at all — see the gotchas.

## Who uses it, and what it uses

`export_provider!` in `src/lib.rs` constructs it, on behalf of whoever loads the library.
It uses the pure modules in the crate root (`control`, `sched`, `cache`, `config`,
`profile`, `route`, `moe`, `preflight`), `ovata-infer-protocol` for
`Frame`/`FrameSink`/`InferenceRequest`, and `llama-cpp-2` + `llama-cpp-sys-2` from
`src/infer-engine/`. The MoE paging machinery it calls over FFI lives in
`src/infer-engine/moe-driver/`.

## Building and gotchas

There is no way to compile this on a Mac. It builds as part of the crate on Linux:

```sh
cargo build --release -p dovata --features cuda
```

- `concurrency` carries `#[allow(dead_code)]` and is now genuinely unreachable. The
  selftest module was its only caller, so deleting that left the continuous-batching
  driver with nothing calling it. It is kept because the RLM fan-out design depends on
  it and the pure scheduler underneath (`crate::sched`) is well covered, but treat it as
  unverified code until something calls it again.
- `Dovata` carries a hand-written `unsafe impl Send/Sync` and leaks the model and backend
  to `'static` via `Box::leak`. Both are load-bearing for the context lifetime; re-read
  the justification in `mod.rs` before changing how the model is owned.
- The composition paths need the unified KV cache (`JADE_KV_UNIFIED`, on by default).
