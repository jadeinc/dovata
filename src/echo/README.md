# echo — the reference provider

The smallest thing that is a provider. It takes a request, emits the prompt back as one
token, and completes. No config, no network, no model.

## Why it exists

Two jobs, both worth keeping it around for.

It is the **worked example**. The provider boundary is a C ABI with six exported
symbols, an ABI version check, and panic containment at the crossing — all of which the
`export_provider!` macro generates. `src/lib.rs` shows exactly how little you write on
top of that: name the provider, accept a config, implement `infer`. Copy this directory
when adding a native provider.

It is also the **test fixture**. `dovata::providers` needs something real to `dlopen` to
prove the loader round-trips frames across the boundary, and using a genuine provider
rather than a mock means the test also checks that `export_provider!` emits what the
loader looks for. That is why the crate builds as both `cdylib` and `rlib`.

## Files

- `src/lib.rs` — the whole provider. `struct Echo`, an `impl Provider`, and one
  `export_provider!(Echo)` call.
- `Cargo.toml` — the only thing worth noting is `crate-type = ["cdylib", "rlib"]`.

## Who uses it

`dovata::providers::tests` loads the built library out of `target/debug/`, so
`cargo test -p dovata` fails with a clear message unless you have run
`cargo build -p ovata-provider-echo` first. `src/packaging/` includes it in every
bundle, reading its exported name and ABI version to validate the packaging path
end to end. It depends only on `ovata-infer-protocol`.

## Building

```sh
cargo build -p ovata-provider-echo
```

Producing `target/debug/libovata_provider_echo.{so,dylib}`. To watch a live daemon load
it, point `JADE_PROVIDERS` at that path.
