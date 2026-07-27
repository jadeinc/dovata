# gguf — reading a model file without llama.cpp

A dependency-free Rust crate that parses the GGUF metadata header, plus the
`gguf-inspect` CLI that prints it. Nothing here links llama.cpp, touches CUDA, or reads
tensor data, so it builds and runs on a laptop.

## Why it exists

Pointing the daemon at a different model is a one-variable change — set
`JADE_MODEL_PATH` and restart. Whether the new model *works* is decided by three things
you cannot see from the filename, and until this crate existed the only way to find out
was to build the CUDA daemon on the GPU box and watch it fail:

1. Does llama.cpp recognise the architecture the file declares.
2. Does the file carry the hyperparameters that architecture needs.
3. Will the daemon's profile registry match it, or silently fall back to bare ChatML.

A GGUF puts all of its metadata before the tensor data, so all three are answerable from
the first few megabytes. You can vet a model with an HTTP range request before
downloading 18 GB of it — which is exactly how `sample-qwen3-coder-30b.txt` was made.

## What llama.cpp actually requires

This is the part worth internalising, because our own profile matching does it
differently and that difference is the sharp edge.

*The architecture is a controlled vocabulary.* llama.cpp reads `general.architecture`,
looks the string up in its `LLM_ARCH_NAMES` table, and everything after that decision —
which hyperparameters are read, which tensor names are expected, which graph is built —
follows from the match. A string not in the table is the `unknown model architecture`
error, and no amount of correct weights gets past it. The vocabulary is family-level and
deliberately unglamorous: our production model declares `qwen3moe`, not
`Qwen3-Coder-30B`. A future Qwen release may well keep declaring `qwen3moe`, in which
case it loads on the engine we already ship.

*The hyperparameters are namespaced by that same string.* Once the arch is `qwen3moe`,
the loader reads `qwen3moe.block_count`, `qwen3moe.embedding_length`,
`qwen3moe.attention.head_count`, and so on. Miss a required one and the load throws.
MoE models add `expert_count`, `expert_used_count`, and `expert_feed_forward_length`.
The tokenizer comes out of the same header — `tokenizer.ggml.model`, the token array,
the BPE merges — and fails just as hard when incomplete.

*The tensor index has to agree.* Names must match the architecture's tensor mapping
(`blk.0.attn_q.weight` and friends) and shapes must match the hyperparameters. A file
with the right arch string and the wrong tensor names still fails.

*`general.name` is free text, and llama.cpp ignores it entirely.* It is whatever the
person who ran the conversion typed. That is the field the daemon's profile registry
globs against, which means the same model from two publishers can select two different
profiles, or none. The architecture is the stable identifier; the name is not. Worth
remembering when a profile mysteriously does not apply.

None of this is transcribed from the loader source, and this crate does not carry a copy
of the arch table — that table moves upstream every few releases and a stale copy here
would be a confident lie. `gguf-inspect` reports what the file claims and flags the keys
that are known to be load-bearing. The authoritative check is still loading the model
with the pinned engine.

## Files

- `src/lib.rs` — the parser. `Gguf::parse` over a byte slice, `Value` for the twelve
  metadata types plus arrays, `TensorInfo` for the index, and `Report`, which pairs the
  file against the keys llama.cpp reads and tells you which required ones are missing.
  Every read is bounds-checked and array lengths are sanity-checked against the input
  size, because "point this at a half-downloaded model" is a normal thing to do.
- `src/main.rs` — the `gguf-inspect` CLI. Reads at most 64 MB of the file.
- `sample-qwen3-coder-30b.txt` — real output for the production model, annotated. Read
  this first if you want the shape of the answer before running anything.

## Usage

```sh
# What is in this file, and will it load
cargo run -p gguf --bin gguf-inspect -- model.gguf

# Everything, including tokenizer.chat_template printed verbatim
cargo run -p gguf --bin gguf-inspect -- model.gguf --all

# Vet a model you have not downloaded
curl -sL -r 0-41943039 -o head.gguf "<huggingface resolve url>"
cargo run -p gguf --bin gguf-inspect -- head.gguf
```

Exit status is 0 when no required key is missing, 1 when one is, so it drops into a
script or a CI gate as-is.

## Adding a model, end to end

```sh
cargo run -p gguf --bin gguf-inspect -- new-model.gguf                    # 1. does it load
cargo run -p gguf --bin gguf-inspect -- new-model.gguf --profile-stub \
  > src/dovata/profiles/new-model.toml                                    # 2. starter profile
```

The stub arrives with `model`, `match`, and `[limits] context_window` already filled in
from the header, and with the tool-call and tool-result blocks left as explicit TODOs —
those tokens are a training convention, not a recorded field, so they cannot be read out
of the file. The stub appends the model's `tokenizer.chat_template` verbatim as a
comment, which is where the delimiters are actually written down. Fill them in, add the
file to `BUILTIN_PROFILES` in `src/dovata/src/profile/mod.rs`, and `cargo test -p dovata`
validates it. The language-side `.jde` twin is a separate file; see
`src/dovata/profiles/README.md`.

## Who uses it

`src/dovata` links it. `dovata::preflight` runs at daemon startup, before the engine
loads: it reads the header, checks the required keys, selects the model profile, and
refuses the boot with a specific message when something is wrong. That turns a mistyped
`JADE_MODEL_PATH` or an unprofiled model from a minutes-long load followed by a vague
failure into a one-second error naming the fix.

There is deliberately only *one* reader of this decision. The engine used to select a
profile itself after `LlamaModel::load_from_file` returned; now it reads the choice the
preflight published, so the profile that approved the boot is the profile that serves
requests. llama.cpp remains the only authority on whether a model loads — this crate
never claims otherwise, it just gets there first with a better error.

Everything else is still humans at a terminal. The obvious next users are
`src/packaging` (record the shipped model's arch and hyperparameters in
`manifest.json`) and `src/scripts/quantize-model.py` (assert a freshly quantized GGUF
declares what the pinned engine expects before it is uploaded).

The `KNOWN_PROFILE_GLOBS` constant in `src/main.rs` is a hand-copied mirror of the globs
in `src/dovata/profiles/*.toml`, kept so this crate builds standalone. Add a profile and
it needs updating here too, or `gguf-inspect` will under-report a match.
