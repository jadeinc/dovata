# batch/ — continuous batching

N sequences contributing tokens to one forward pass, so a fan-out of sub-inferences
costs roughly one decode instead of N.

## Read this first: nothing calls it

`exec.rs` has no caller. Its only one was the env-gated selftest module, which was
deleted once dropping the daemon left it with no startup phase to hang off. It carries
`#[allow(dead_code)]` and it compiles, and that is all anyone can currently say about it.

It is kept because the RLM fan-out design depends on it and because the hard part —
the scheduler — is well covered. But treat the llama-bound half as **unverified**: it
has not run since the selftests were removed, and no eval reaches it. If you wire it to
something, verify it on the GPU box before believing it.

## What each file does

- `sched.rs` — the pure scheduler. Which sequence contributes which tokens to each
  forward pass, when KV may be admitted, when it must be retired, and what the
  concurrency ceiling is given free VRAM. All bookkeeping, no model, so it is
  thoroughly unit-tested on any machine — including the two-sequences-share-one-batch
  case that is the whole point.
- `exec.rs` — the thin llama-bound driver that turns the scheduler's plans into batches
  and feeds sampled tokens back. Linux-only.

The split is why the interesting half survives having no caller: `sched.rs` is tested
regardless, and `exec.rs` is small enough to re-read.

## Turning it off

Dropping the `batch` feature changes no behaviour, because no path reaches the driver.
It is in `default` so the shipped artifact keeps the code, and so that the day something
does call it, the capability already exists rather than needing to be re-created.
