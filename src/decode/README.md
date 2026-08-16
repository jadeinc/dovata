# decode/ — the default request path

One sequence, streamed. Render the prompt through the active model profile, check the
response cache, reuse whatever KV prefix is already resident, generate under the
constrained decoder, emit each chunk of client-safe text as it is produced.

Most requests land here. [`compose/`](../compose/README.md) takes the ones that do not
fit the window; [`batch/`](../batch/README.md) would take fan-out, if anything called it.

## What each file does

- `prompt.rs` — renders a parsed message envelope into a specific model's chat text. The
  envelope format belongs to the protocol crate; only the model-specific rendering is
  here, and the template comes from [`model/`](../model/README.md). Pure and tested.
  `build_formatted_prompt` lives here too — it used to sit in the dispatch file, which
  made the router own prompt rendering.
- `cache.rs` — a bounded LRU of past *responses*, keyed by a hash of the request fields
  that determine output. Distinct from KV prefix reuse, which happens during prefill and
  lives in [`kv/`](../kv/README.md); this one skips the model entirely.
- `exec.rs` — the llama-bound generation loop. Linux-only.

## How thin exec.rs is, and why

It allocates a batch, calls `decode`, samples, and hands each piece to the control
machine. Everything interesting around it belongs to a capability it borrows:
`tokenize` for text in and out, `kv` for prefix reuse, `sample` for the chain and for
deciding when the grammar retires.

That is the *pure core, thin edge* rule at capability scale. The tricky logic — a stop
anchor split across two tokens, a grammar that must retire at the right depth — is
reachable from a laptop test because none of it needed a model.

## Coverage

`prompt.rs` and `cache.rs` are unit-tested on any platform. `exec.rs` is not covered at
all: it is `cfg(target_os = "linux")`, and the evals in `tests/` are the only thing that
exercises it. A green macOS run says nothing about it.
