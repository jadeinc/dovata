# compose/ — long-context answering by KV composition

When a prompt does not fit the native window, this answers it: split the input into a
retrievable corpus and a question, retrieve the blocks that matter, and splice their
precomputed K/V into one attention window. One decode, no re-prefill.

The design and the measured numbers are in
[`docs/blended-rag-rlm.md`](../../docs/blended-rag-rlm.md).

## Why it is a peer of decode, not a mode of it

It is a different engine, not a flag. `decode` prefills a prompt and generates;
this retrieves, relocates KV RoPE-correctly, splices, and generates. The two share
`kv`, `sample` and `tokenize` and nothing else.

Restoring a block's K/V is roughly 50x cheaper per block than re-prefilling it on an
A10G, which is the entire reason the path exists.

## What each file does

- `route.rs` — the pure split: which part of a prompt is corpus and which is the
  question, and how the corpus is chunked. A messages prompt splits at the last user
  turn; a plain prompt at the trailing paragraph. No tokenizer needed, so it is tested
  on any machine.
- `exec.rs` — the composition engine: KV save and restore, `BlockStore` and IDF
  retrieval, the host-RAM `KvBlockCache`, and the blended rerank pipeline that scores
  candidates by query-likelihood before composing the top k. Linux-only.
- `rlm.rs` — request dispatch, and the store of uploaded context handles so a program
  can upload a large context once and ask many cheap questions against it. Linux-only.

## Turning it off refuses rather than truncates

Without the `compose` feature an over-context prompt gets an `Error` frame naming the
capability. The alternative — falling through to the native path — hands llama.cpp more
tokens than its window holds and returns an answer derived from whatever fragment
survived. Confident, plausible, and wrong.

## Coverage, and the honest gap

`route.rs` and the model-free half of `exec.rs` (retrieval, the cache LRU) are unit
tested. The composition paths themselves have **no automated coverage**: the env-gated
selftests that drove them went away with the daemon. `13_long_context` in `tests/` is
the only thing that exercises this end to end, and it needs the GPU box.

There is also a known capability gap, not a length one: near-duplicate fact
disambiguation fails at *every* corpus size including sub-window. See the design doc.
