# load/ — admission

Everything that happens once, before any request: read the `JADE_*` environment into a
`Config`, vet the model file, pick its profile, and size the generation cap from the
VRAM left after the weights.

## Why it is its own capability

Because it is the only part of the engine that can decline to come up.

Load-time policy and per-request policy fail differently. A request that cannot be
served returns an `Error` frame and the engine carries on. A model file with incomplete
metadata, or one no profile matches, is not a bad request — it is a provider that should
not exist. `Provider::from_config` returns `Err`, the loader sees a null handle, and the
caller finds out immediately instead of receiving confident nonsense later.

That is worth keeping in one place, which is why it is separate from
[`dispatch/`](../dispatch/README.md) rather than the first half of it.

## What each file does

- `config.rs` — the `JADE_*` environment into a `Config`, and `validate()`. The KV cache
  type is modelled as our own `KvType` enum rather than llama's, so the
  flash-attention rule stays testable without a GPU. `validate()` is also where a config
  asking for a capability this binary was not built with is refused.
- `preflight.rs` — reads the GGUF header with [`crate::gguf`](../gguf/README.md) *before*
  llama.cpp opens the file, checks the metadata llama.cpp will require, and selects the
  profile on architecture as well as name. Refuses with a message that names the fix.
- `vram.rs` — estimates how many tokens can be generated from the KV cache that fits
  after the weights, using the model's own shape from the GGUF header. Exists because
  nothing sets `max_tokens` any more and `u32::default()` is 0, which the engine would
  otherwise read as "generate nothing".

## Gotchas

Pure apart from one file read, so all of it is unit-tested on macOS against synthetic
GGUF headers. `JADE_ALLOW_UNPROFILED=1` downgrades the no-profile refusal to a warning —
an unprofiled model generates text fine and mis-parses every tool call, which is why it
is off by default.
