# echo — the reference provider

The smallest thing that is a provider. It takes a request, emits the prompt back as one
token, and completes. No config, no network, no model.

## Why it exists

It is the *worked example*. The provider boundary is a C ABI with six exported symbols,
an ABI version check, and panic containment at the crossing, all of which the
`export_provider!` macro generates. `src/lib.rs` shows exactly how little you write on
top of that: name the provider, accept a config, implement `infer`. Copy this directory
when adding a native provider.

It also gives the packaging path something real to check. `src/packaging/` reads every
provider `.so`'s exported name and ABI version to prove it is a genuine plugin, and `echo`
is the one it can always build and read on any machine, with no model and no GPU.

It used to be this repo's `dlopen` test fixture as well. The host that loaded providers
is gone, and so is that test, but the crate still builds as both `cdylib` and `rlib`,
which is what a provider package has to do.

## Files

- `src/lib.rs` — the whole provider. `struct Echo`, an `impl Provider`, and one
  `export_provider!(Echo)` call.
- `Cargo.toml` — the only thing worth noting is `crate-type = ["cdylib", "rlib"]`.

## Who uses it

`src/packaging/` includes it in every bundle and validates it on the way in. Anyone
writing a native provider reads it. It depends only on `ovata-infer-protocol`.

## Building

```sh
cargo build -p ovata-provider-echo
```

Producing `target/debug/libovata_provider_echo.{so,dylib}`.
