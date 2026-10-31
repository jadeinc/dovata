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

## The format

GGUF is a header of typed key and value metadata, followed by a tensor index and then
the tensor data, little-endian throughout. Versions 2 and 3 use 64-bit lengths and
counts. Version 1 used 32-bit ones and predates every model dovata plans, so `parse`
refuses it with a clear message instead of supporting it halfway.

Each tensor's offset in the index counts from the start of the data section, not from
the start of the file. The data section begins at the end of the index, rounded up to
`general.alignment`, or to 32 bytes when that key is absent, the
`GGUF_DEFAULT_ALIGNMENT` in the vendored `ggml/include/gguf.h`. gguf.cpp pads only when
there are tensors, so a file with none is not padded. `Gguf::file_offset` adds the two
offsets together, which is what a memory map needs.

A header cut short inside the tensor index still parses. The metadata is complete and
can be trusted; only the tensor list is partial, and `Gguf::truncated` says so. That is
the normal case for a range-fetched header, so it is not an error. A header cut short
inside the metadata is an error, since too little of the file was fetched to answer anything.

Converters write integers at whatever width they choose: one file stores
`block_count` as a `u32` and another as a `u64` or an `i32`. So `Value::as_u64` and
`Gguf::u64` read any unsigned or non-negative integer, and nothing reads a
hyperparameter by its declared type.

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

`Gguf::is_moe` decides whether a file is MoE from a non-zero `<arch>.expert_count`, not
from the architecture name, because several architectures cover both dense and MoE
variants. The MoE keys are required only when it says yes.

*The file type is a label, not the truth.* `general.file_type` names the quantization of
the file as a whole, such as 15 for `Q4_K_M`. Each tensor keeps its own `ggml_type`, and
a "Q4_K_M" file still holds F32 norms and Q6_K output weights. `Gguf::quant_mix` counts
the real mix, and `gguf-inspect` prints it.

*Split files.* A model uploaded in several files, written by llama.cpp's `gguf-split`,
carries `split.no`, `split.count` and `split.tensors.count` in every part. Only part 0
carries the model's metadata. A file that declares a `split.count` of 1 is treated as
one file.

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

`Report` is a guide in the same way. The keys any one architecture demands live in
llama.cpp's `llm_load_hparams` and change between releases. The report lists the core
that every llama-family architecture shares, plus the MoE keys and the tokenizer keys.
An architecture can demand more than that core, and none demands less. A clean report
means nothing obviously missing, never that the file is sure to load.

Whether llama.cpp supports an architecture and whether dovata can plan it are separate
questions. The pinned llama.cpp answers the first. The second is answered by whether a
built-in layout exists for the architecture, or a layout file is given to the planner.

## Files

- `mod.rs` — the parser. `Gguf::parse` over a byte slice, `Value` for the twelve metadata
  types plus arrays, `TensorInfo` for the index, and `Report`, which pairs the file
  against the keys llama.cpp reads and lists the required ones that are missing. It also
  finds where the tensor data starts (`data_offset`, after `general.alignment` padding),
  sizes each tensor (`TensorInfo::bytes`, from `ggml_type_layout`), and reads the split
  keys and `general.file_type`. Every read is bounds-checked, because pointing this at a
  half-downloaded model is a normal thing to do.
- `write.rs` — a small GGUF v3 writer: metadata, the tensor index, and aligned data.
  It exists for test fixtures, such as the tiny llama model the quantize tests feed to
  llama.cpp, and for `refback`'s fixtures later. Its output round-trips through `parse`.
  See "The writer" below.
- `main.rs` — the `gguf-inspect` CLI. It answers the first questions a person has about
  a new model file: what architecture it declares, whether it carries the
  hyperparameters that architecture needs, what it is quantized to, whether it is one
  part of a split model, and whether dovata has a built-in layout for it. It reads at
  most 64 MB of the file, which holds the header of every published model, so a model
  of many gigabytes is never pulled into memory to print forty lines.
- `sample-qwen3-coder-30b.txt` — real output for Qwen3-Coder-30B at Q4_K_M, annotated.

## The type table

`ggml_type_layout` gives each `ggml_type`'s block: how many elements, how many bytes.
The numbers were copied from `type_traits` in the vendored `ggml/src/ggml.c`, read
through `ggml_get_type_traits`, not worked out from struct definitions. Plain types such
as F32 are blocks of one element. A size wrong by a byte per block moves every tensor after
it, so with the `quantize` feature on, `type_table_matches_ggml` compares the table with
ggml's own `ggml_type_size` and `ggml_blck_size`. It was also checked against a real file: on the Qwen3-Coder header above, the
last tensor ends exactly at the file's size on the Hub, 18,556,689,568 bytes. A type
newer than the table has no size, and a tensor of that type cannot be planned until the
table is extended. Retired tags, such as 4, 5 and 31 to 33, have no size either.

`TensorInfo::bytes` sizes a tensor the way ggml's `ggml_nbytes` does for a contiguous
tensor. Quantized types store whole blocks, so the first dimension has to be a multiple
of the block size, and ggml refuses a row that is not. When the type is unknown, the row
does not divide into blocks, or the size overflows, the answer is `None` and not a
guess, because a guessed size would place the next tensor on top of this one.

`ggml_type_name` names only the types that appear in published GGUF files. The enum
grows upstream every few releases, so an unknown tag prints as `type ?` instead of a
guessed name: a wrong quant name in a report is worse than an honest unknown.

## The writer

dovata never makes a model anyone runs; llama.cpp's quantizer writes the real outputs.
`write.rs` exists for fixtures. The quantize tests need a small model that llama.cpp's
loader accepts, and `refback` will need models whose every byte is known, so it can
check that each tensor landed where the plan says. A binary fixture checked into the
repo is something nobody can review. Writing it from a few lines of Rust keeps it
readable and lets each test pick its own shapes.

The bytes follow `gguf_write_out` in the vendored `ggml/src/gguf.cpp`, format version 3:
the header, the metadata, the tensor index, zero padding up to the alignment, and then
each tensor's data, itself padded to the alignment. gguf.cpp's reader refuses a tensor
whose offset is not the running total of the padded sizes before it, so `Writer` works
the offsets out itself and never takes them from a test. The alignment is
`general.alignment` when set and 32 otherwise, and the writer refuses anything but a
`u32` power of two, the same rule the reader applies.

A few rules keep the output readable by any GGUF reader. Setting a key twice replaces
its value and keeps its place, because GGUF does not allow a key twice. Arrays must be
homogeneous, and an empty array has no element type to write, so both are refused.
A tensor's byte count is checked against its type and shape when the file is laid out,
not when the tensor is added, so the adding calls can be chained. Metadata keeps the
order it was set in and tensors keep the order they were added in, so the same calls
always give the same bytes.

It is small on purpose. A `Writer` holds everything in memory and `to_bytes` lays it
out. Fixtures are kilobytes; a real model is not something to build this way.

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
