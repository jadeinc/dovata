# bindings — `llama-cpp-2`, the safe wrapper

The safe(-ish) Rust wrapper over [`../bindings-sys`](../bindings-sys/README.md): model,
context, batch, sampling, tokens, GGUF, grammar. Vendored from
[utilityai/llama-cpp-rs][readme] at 0.1.150; the crate name is kept because it is
upstream identity, even though the directory was renamed from `api/`.

> Upstream's own README is preserved below the line and still describes the crate
> accurately. Read this section first for what is different here.

[readme]: https://github.com/utilityai/llama-cpp-rs

## Why the `common` feature stays on, at ~2.2 MB

`common` pulls in llama.cpp's `common/` static library plus `wrapper_common.cpp`, and it
is an obvious candidate to drop — the engine's own code never calls
`json_schema_to_grammar`, `FitError`, `LlamaTokenDataArray`, or `LlamaSampler::accept`.
Grepping every capability for the `common`-gated surface finds nothing.

It stays anyway, because **`LlamaSampler::grammar` is gated on it**, and grammar-
constrained decoding is not optional here. It is how tool calls and structured output
work; `src/sample/chain.rs` calls it directly, and `05_grammar`, `07_tool_anchor` and
`12_grammar_retire` all exercise it.

The dependency runs deeper than the `#[cfg]`. `LlamaSampler::grammar` calls
`llama_rs_sampler_init_grammar`, which is defined in `wrapper_common.cpp` — compiled only
when `common` is on. Built both ways to confirm rather than reading attributes:

| | `--features common` | `--no-default-features` |
|---|---|---|
| `libllama_cpp_sys_2_common_wrapper.a` | present, 2.2 MB | absent |
| `llama_rs_sampler_init_grammar` in `bindings.rs` | 3 occurrences | 0 |

So dropping `common` is not a subtle regression to be caught by an eval. It is a compile
error at `chain::grammar`, and there is no non-`common` path to a grammar sampler in this
crate. The size saving is real and the feature is load-bearing; the feature wins.

Revisit only if grammar sampling moves to a path that does not route through
`wrapper_common.cpp`.

---

# llama-cpp-rs-2

[utilityai]: https://utilityai.ca

A wrapper around the [llama-cpp](https://github.com/ggerganov/llama.cpp/) library for rust.

# Info

This is part of the project powering all the LLMs at [utilityai], it is tightly coupled llama.cpp and mimics its API as
closly as possible while being safe in order to stay up to date.

# Dependencies

This uses bindgen to build the bindings to llama.cpp. This means that you need to have clang installed on your system.

If this is a problem for you, open an issue, and we can look into including the bindings. 

See [bindgen](https://rust-lang.github.io/rust-bindgen/requirements.html) for more information.

# Disclaimer

This crate is *not safe*. There is absolutly ways to misuse the llama.cpp API provided to create UB, please create an issue if you spot one. Do not use this code for tasks where UB is not acceptable.

This is not a simple library to use. In an ideal world a nice abstraction would be written on top of this crate to
provide an ergonomic API - the benefits of this crate over raw bindings is safety (and not much of it as that) and not much else.

We compensate for this shortcoming (we hope) by providing lots of examples and good documentation. Testing is a work in
progress.

# Contributing

Contributions are welcome. Please open an issue before starting work on a non-trivial PR.
