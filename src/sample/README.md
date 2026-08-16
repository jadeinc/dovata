# sample/ — choosing the next token

Two halves. `control.rs` decides what a token is *allowed* to be and when the rules can
be relaxed; `chain.rs` builds the llama sampler that draws it.

## Why it is a capability and not part of decode

The narrow reason is that `compose` and `batch` sample too, so decode does not own this.

The sharper reason is what the split fixed. The unconstrained chain

```rust
LlamaSampler::chain_simple(vec![temp(0.6), min_p(0.05, 1), dist(0)])
```

had been copy-pasted, constants included, into four places across two request paths.
Nothing made the four agree. Changing the temperature for the standard path would have
left the batch path on the old value, silently, and the only way to notice would be
comparing outputs between two paths nobody compares.

That is the same shape as the tool-call delimiter drift between a profile's TOML and its
Jade package, which `gen-profile-package.py` exists to prevent. Same fix here: one owner.

## What each file does

- `control.rs` — the pure constrained-decoding state machine, over the *text* the tokens
  spell rather than the tokens: anchor spans, stop anchors that arrive split across two
  tokens, JSON depth tracking so the grammar can be retired once the structured part of
  a reply closes. No model needed, so it is thoroughly tested on a laptop — and it is
  the single most bug-prone thing in the engine, which is why it was pulled out of the
  generation loop in the first place.
- `chain.rs` — `free`, `grammar`, and `greedy`, plus the `Params` that used to be four
  hardcoded literals. Linux-only.

## Two things that will bite you

Grammar position in the chain is load-bearing. It filters candidates, so it must see
logits temperature has already shaped and hand its survivors to `min_p` and the draw.
Moving it changes which tokens are reachable, and nothing will tell you.

A grammar sampler holds llama-side state and must outlive the generation loop. Freeing
one while its context is still active crashes llama.cpp.
