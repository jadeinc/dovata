# packaging — the release xtask

A small Rust binary that builds the engine library and the native provider libraries,
validates them, and assembles `dist/` at the repo root. It is the single source of truth
for how a release is built and staged.

## Why an xtask instead of CI steps

The staging rules are fiddly and consequential: the engine must land under the exact
asset name the JADE OS bootloader fetches, with a checksum sidecar; every provider `.so`
must be a real `export_provider!` plugin at the current ABI version, not just a file that
happens to exist; and the manifest has to describe what actually shipped. Encoding that
in YAML means it only runs in CI, cannot be tested locally, and drifts from what a
developer builds by hand.

So the logic lives here as ordinary Rust, and `.github/workflows/release.yml` is a thin
wrapper that installs CUDA and calls `cargo run -p packaging -- release`. You can run
the same command locally and get the same bundle.

## What it does

1. Builds the engine and every provider cdylib in one `cargo build`.
2. Wipes and recreates `dist/`.
3. For each provider, opens the built library and reads its exported
   `ovata_provider_name_*` and `ovata_provider_abi_version` symbols. This both labels the
   file and proves it is a genuine plugin. An ABI mismatch warns rather than fails.
   It never calls `ovata_provider_new`, so no model or config is needed.
4. Stages the engine cdylib as `dovata-<arch>.so` plus a `.sha256` sidecar. Linux only.
5. Writes `dist/manifest.json` describing the ABI version, profile, arch, the engine
   asset, and every provider.

## Files

- `src/main.rs` — all of it. `Plan` captures the two modes, `assemble` runs the five
  steps above, and `read_provider_meta` is the unsafe symbol-reading helper.
- `Cargo.toml` — depends on `ovata-infer-protocol` for the provider ABI constant,
  `libloading` to read the symbols, and `sha2` so checksums are computed portably
  (the `release` path runs identically on macOS).

## The engine ships as a library, and is the one thing not validated

What ships is libraries, not a binary. There used to be a `dovata` daemon binary staged
here beside the `.so`, and the bundle carried the engine twice. That is gone. A Jade
program loads `dovata-<arch>.so` and calls `infer` in its own process, so the library is
the whole product.

The engine `.so` is staged without the `read_provider_meta` check every other provider
gets. That check `dlopen`s the library, and this one links CUDA, so it cannot load on a
builder with no NVIDIA driver, which the release runner is. Validating it there would fail
every release. Its ABI holds by construction instead: the same `export_provider!` on the
same trait impl. It is Linux-only, because the engine is; elsewhere the cdylib is an empty
stub and is skipped rather than shipped, since shipping something that looks like a
provider but exports nothing is worse than shipping nothing.

Read asset names out of `manifest.json` rather than hardcoding them. The OS install
layer and the language both need to find these files, and the manifest is the contract.
The engine lives under the `engine` key; the loadable providers under `providers`.

## What ships and what does not

Only the OS bundle is built here: the engine library plus the native `echo` provider. The
Anthropic and OpenAI providers are pure-Jade `.so` packages published on a different path
entirely — the `publish-providers` workflow builds them with the Jade toolchain and
serves them from jadelang.org.

## Usage

```sh
cargo run -p packaging -- build              # dev bundle, debug profile
cargo run -p packaging -- build --release    # dev bundle, release profile
cargo run -p packaging -- release            # release bundle: dovata-<arch>.so + sha256
cargo run -p packaging -- release --cuda     # ... with the engine's GPU backend
```

`--cuda` is release/Linux only — it forwards the `cuda` feature to the engine crate, which
is a no-op where the engine is not compiled in.

## Gotchas

`dist/` is gitignored and is deleted and recreated on every run. Do not put anything
there you want to keep. The crate locates the workspace root by walking up from its own
manifest directory, so it stays correct only as long as it lives at `<root>/src/packaging`
— moving it means fixing `workspace_root()`.
