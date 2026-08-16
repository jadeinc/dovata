# tokenize/ — text to tokens and back

Thin wrappers over llama's tokenizer, plus `PieceDecoder`.

## Why the decoder is the point

A token is a span of *bytes*, not of characters. A three-byte character can arrive as
the tail of one token and the head of the next, so decoding each token's bytes on their
own turns one character into two replacement characters. It shows up in non-Latin text
and emoji — which is to say, not in anything anyone tries locally.

The fix is a decoder that carries the incomplete tail across calls. Every request path
needs one, and each had built its own inline: one in the standard path, one in compose,
and a `Vec` of them in batch, one per sequence.

## The per-sequence rule

One decoder per sequence, never shared. A shared decoder across interleaved sequences
would splice one sequence's incomplete byte tail onto another's next token — exactly the
bug the decoder exists to prevent, reintroduced by sharing it. `batch/` holds a `Vec` for
this reason; the comment saying so used to live only there, and now lives with the type.

## What is here

`exec.rs`, Linux-only. There is no tokenizer without a model, so like
[`kv/`](../kv/README.md) this capability has no pure half and no macOS coverage.
