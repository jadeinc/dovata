# bundle-kit — reading capability bundles

A small library the sys crate's `build.rs` uses to read *capability bundles*: the
per-capability patch series that let this repo build stock upstream llama.cpp plus our
deltas, instead of carrying a fork. See
[`../moe-driver/bundle/`](../moe-driver/bundle/README.md) for the one bundle that
exists and what it does.

## Why it is a crate

Because build scripts cannot be tested. Cargo runs them; `cargo test` does not. The sys
crate's `build.rs` is over 1200 lines and not one of them has ever been covered.

The logic in here is the part where being wrong is both expensive and silent:

- A manifest naming a patch that is not on disk. Skipping it would produce a build with
  half a seam in it, which for a seam that is inert when disabled reports no problem at
  all.
- A patch touching the GGUF converter or the quantizer. That breaks the invariant
  letting the engine (upstream + patches) and `quantize-model.py` (stock upstream) share
  one pin, and the symptom arrives as a published model the engine refuses.
- Two bundles setting the same cmake define to different values, which would otherwise
  resolve by argument order — a decision nobody made.

So they live where `cargo test` can reach them, and `build.rs` keeps only the filesystem
and process work: staging a worktree, shelling out to `git apply`, and panicking with
the patch name when that fails.

## What is here

`src/lib.rs`, and nothing else.

- `load_manifest` / `parse_manifest` — read a `bundle.toml` into a `Bundle`. Resolves
  the capability's source directory to an *absolute* path, because the patched
  CMakeLists is read from a staged copy whose depth relative to the bundle is not
  knowable here. That counted-relative-path fragility is the thing bundles exist to
  remove; see the bundle README.
- `patch_touches_forbidden_path` / `check_forbidden_paths` — the denylist, and the
  reason for it.
- `cmake_defines` — merge the enabled bundles' defines, erroring on a conflict.

Twelve tests, all on synthetic input except one: `the_real_moe_bundle_loads_and_is_allowed`
loads the shipped bundle from disk. That is the test that notices a typo in the real
`bundle.toml`, and it runs on a laptop with no llama.cpp and no GPU.

## Who uses it

Only `llama-cpp-sys-2`, as a `[build-dependencies]` entry. It is a workspace member so
its own tests run — a path dependency alone would build it and never test it.

`tools/scripts/check-bundles.py` does the same job for CI and reimplements the denylist
in Python rather than depending on this crate. Two copies of one list is a smell, and it
is deliberate: this one gates the *build*, that one gates *review*, and a patch reaching
CI having skipped the build check is exactly the case worth catching. If you add a path
to one, add it to both.
