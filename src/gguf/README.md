# gguf — reading a model file without llama.cpp

A dependency-free reader for the GGUF header, plus the `gguf-inspect` CLI that prints
it. Nothing here links llama.cpp, touches a GPU, or reads tensor data, so it builds and
runs on a laptop.

## Why it exists

Everything dovata decides about a model comes from this header. The planner needs every
tensor's name, shape, `ggml_type`, byte size and file offset to place it in memory. The
layout binder needs the architecture and hyperparameters to pick a layout and check it.
`fetch` needs `general.file_type` to confirm a download is the quantization it asked
for. None of that needs the tensor data.

A GGUF puts its metadata and its tensor index before the data, so all of it is
answerable from the first few megabytes. You can plan a model from an HTTP range request
before downloading 18 GB of it, which is how `sample-qwen3-coder-30b.txt` was made.

## What llama.cpp actually requires

*The architecture is a controlled vocabulary.* llama.cpp reads `general.architecture`,
looks the string up in its `LLM_ARCH_NAMES` table, and everything after that decision —
which hyperparameters are read, which tensor names are expected, which graph is built —
follows from the match. A string not in the table is the `unknown model architecture`
error, and no amount of correct weights gets past it. The vocabulary is family-level and
deliberately unglamorous: Qwen3-Coder-30B declares `qwen3moe`. dovata's built-in
layouts key on the same string, so a later release that keeps declaring `qwen3moe`
plans with the layout we already ship.

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
person who ran the conversion typed, and two publishers of the same model often
disagree on it. dovata ignores it too. 2.x selected model profiles on it, which is one
reason profiles went in 3.0.

None of this is transcribed from the loader source, and this crate does not carry a copy
of the arch table — that table moves upstream every few releases and a stale copy here
would be a confident lie. `gguf-inspect` reports what the file claims and flags the keys
that are known to be load-bearing. The authoritative check is still loading the model
with the pinned llama.cpp.

## Files

- `mod.rs` — the parser. `Gguf::parse` over a byte slice, `Value` for the twelve metadata
  types plus arrays, `TensorInfo` for the index, and `Report`, which pairs the file
  against the keys llama.cpp reads and lists the required ones that are missing. It also
  finds where the tensor data starts (`data_offset`, after `general.alignment` padding),
  sizes each tensor (`TensorInfo::bytes`, from `ggml_type_layout`), and reads the split
  keys and `general.file_type`. Every read is bounds-checked, because pointing this at a
  half-downloaded model is a normal thing to do.
- `main.rs` — the `gguf-inspect` CLI. Reads at most 64 MB of the file.
- `sample-qwen3-coder-30b.txt` — real output for Qwen3-Coder-30B at Q4_K_M, annotated.

## The type table

`ggml_type_layout` gives each `ggml_type`'s block: how many elements, how many bytes.
The numbers were read out of the vendored ggml through `ggml_get_type_traits`, not worked
out from struct definitions. A size wrong by a byte per block moves every tensor after
it, so the table is checked against a real file: on the Qwen3-Coder header above, the
last tensor ends exactly at the file's size on the Hub, 18,556,689,568 bytes. A type
newer than the table has no size, and a tensor of that type cannot be planned until the
table is extended.

## Usage

```sh
# What is in this file
cargo run --bin gguf-inspect -- model.gguf

# Everything, including tokenizer.chat_template printed verbatim
cargo run --bin gguf-inspect -- model.gguf --all

# Look at a model you have not downloaded
curl -sL -r 0-41943039 -o head.gguf "<huggingface resolve url>"
cargo run --bin gguf-inspect -- head.gguf
```

Exit status is 0 when no required key is missing and 1 when one is, so it drops into a
script or a CI gate as-is.

## Who uses it

[`src/load/`](../load/README.md) reads every model through it: preflight checks the
header, and `Model` gathers the parts of a split file. The planner and the layout binder
read tensors through `Model`, so this module is underneath everything dovata does.
