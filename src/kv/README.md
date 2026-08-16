# kv/ — the KV cache

Cache-type mapping, cross-request prefix reuse, sequence copies, and the per-sequence
save and restore the composition engine is built on.

## Why it is a capability and not part of decode

Because all three request paths use it, and each uses it differently.

- [`decode/`](../decode/README.md) reuses whatever prefix is already resident, so a
  multi-turn conversation does not re-prefill its own history.
- [`compose/`](../compose/README.md) saves a block's computed K/V and restores it later,
  which is the whole reason that engine is cheaper than re-prefilling — roughly 50x per
  block on an A10G.
- [`batch/`](../batch/README.md) drives the per-sequence operations that let N sequences
  share one context.

Owning it from any one of them would misstate who depends on what. It lived under the
old `engine/` directory next to the standard path, which read as decode's private helper
and was never true.

## What is here

`exec.rs`, and nothing else. A KV cache is a llama.cpp object, so unlike most
capabilities there is no useful pure half — no state machine to lift out, no policy that
can be decided without the context in hand.

That makes this one of the two directories a green macOS run says nothing at all about.
The other is [`tokenize/`](../tokenize/README.md).

## Gotchas

The composition paths need the unified KV cache (`JADE_KV_UNIFIED`, on by default). With
it off, `copy_kv_cache_seq` stops being metadata-only and the compose path is
unavailable — `load/` logs that at startup rather than failing, because everything else
still works.
