# bundle-kit — reading capability bundles

A small library the sys crate's `build.rs` uses to read *capability bundles*: the
per-capability patch series that let this repo build stock upstream llama.cpp plus our
deltas, instead of carrying a fork. A bundle is a directory holding a `bundle.toml`, the
patch series it names, and optionally a pointer at the capability's native sources.

No bundle exists today. The only one was the MoE driver's, and it went with the runtime
MoE pager in the 3.0 removal. It lives on `main` at `src/moe/driver/bundle/`, and its
README there is the worked example of the format. The machinery stays so a future
capability can patch llama.cpp the same way without bringing back a fork.

## Why it is a crate

Because build scripts cannot be tested. Cargo runs them; `cargo test` does not. The sys
crate's `build.rs` is over 1200 lines and not one of them has ever been covered.

The logic in here is the part where being wrong is both expensive and silent:

- A manifest naming a patch that is not on disk. Skipping it would produce a build with
  half a seam in it, which for a seam that is inert when disabled reports no problem at
  all.
- A patch touching the GGUF converter or the quantizer. That breaks the invariant
  letting the patched build and `quantize-model.py` (stock upstream) share one pin, and
  the symptom arrives as a published model the patched build refuses.
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
  remove.
- `patch_touches_forbidden_path` / `check_forbidden_paths` — the denylist, and the
  reason for it.
- `cmake_defines` — merge the enabled bundles' defines, erroring on a conflict.

Twelve tests, all on synthetic input except one: `every_shipped_bundle_loads_and_is_allowed`
walks `src/` for every `bundle.toml` and loads each one from disk. That is the test that
notices a typo in a real manifest, and it runs on a laptop with no llama.cpp and no GPU.
With no bundle shipped it finds none and passes, which is the correct answer for the tree
as it stands.

## Who uses it

Only `llama-cpp-sys-2`, as a `[build-dependencies]` entry. It is a workspace member so
its own tests run — a path dependency alone would build it and never test it. The sys
crate's `build.rs` has an empty `enabled_bundles` today, so nothing is staged or patched.

`tools/scripts/check-bundles.py` used to do the same job for CI, with its own copy of the
denylist in Python. It went with the 3.0 runtime removal and lives on `main`, so this
crate is now the only copy of the denylist.
