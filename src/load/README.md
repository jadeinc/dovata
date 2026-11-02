# load/ — reading a model file

Everything dovata does starts here: read a model file's header, refuse it if it cannot
be planned, gather the parts of a split model into one value, and read the model's shape.

## Why it is its own capability

Because it is policy, and [`src/gguf/`](../gguf/README.md) is not. The GGUF reader
reports what any file says, even a truncated one, because `gguf-inspect` has to work on
a half-downloaded model. Planning needs more: every required key present and every
tensor in the index. This module is where that line is drawn.

All of it is pure except the file reads, `Preflight::read` and the reads inside
`Model::open` and `Model::from_parts`. The checks take byte slices, so they are
unit-tested on any machine against synthetic GGUF headers. Those headers all come from
one builder, `preflight::tests::header` and `header_with`, which the tests in `dims.rs`,
`model.rs` and `src/plan/address.rs` share. A second hand-written GGUF byte
builder is the kind of duplicate that drifts from the real format without anyone
noticing.

## Why preflight runs first

A GGUF stores its metadata and its tensor index ahead of the tensor data, so whether a
file is usable is decidable from its first few megabytes. Preflight reads that much,
checks the keys llama.cpp requires, and checks that the tensor index is complete. A file
that fails either check fails there, with a message naming the problem, instead of
failing later inside the planner with a message about a tensor.

Preflight also tells whether a requested context is longer than the model's
`<arch>.context_length`, through `Preflight::ctx_exceeds_native`. A longer context does
not fail. The model's answers degrade quietly once prompts get long, which is worth
reporting before the context is compiled into a program.

## Why a split model is one value

Large uploads come as several files, `<name>-00001-of-00003.gguf` and so on, written by
llama.cpp's `gguf-split`. A split model is still one model, so everything after `load`
sees one `Model` with one list of tensors, each tagged with the part it lives in. The
parts are never merged into one file. Each keeps its own file offsets, so on a CPU target
the planner maps each part as its own region and addresses its tensors at their offsets
in that file. A merge would copy hundreds of gigabytes to save nothing.

The first part passes preflight. The later parts carry no model metadata, only the
`split.*` keys and their share of the tensors, so preflight does not apply to them and
`Model::assemble` checks them instead: each must be a complete header whose `split.no`
and `split.count` match its place, the tensor counts must add up to
`split.tensors.count`, and no tensor name may appear twice. A mismatch means a part from
a different upload or a different quantization, which would otherwise plan fine and
compute garbage.

A split model can be opened two ways. `Model::open` takes the first part's path and
finds the others by llama.cpp's naming rule. `Model::from_parts` takes the list of files
the Hub download in `src/hf_hub/` returns, so nothing is guessed from names.
`Model::from_headers` builds the same value from header bytes in memory, for tests; its
paths are only labels.

`PartIdentity` records each part's header hash and exact size. The plan's offsets are
compiled in as constants, and mapping a different file would read the wrong bytes with
no error, so a compiled program checks these at startup and refuses a file other than
the one the plan was made from.

## Sizing the KV cache

The planner sizes the KV cache from the numbers `ModelDims` reads: one key and one value
entry per KV head per layer for every token of context.

```text
bytes/token = layers * kv_heads * (key_dim * k_elem_bytes + value_dim * v_elem_bytes)
```

It is arithmetic over numbers read from the header, so it is pure and unit-tested on any
machine.

*Keys are namespaced by architecture.* `ModelDims::from_gguf` reads
`general.architecture` first and builds each key from it, such as
`qwen3moe.block_count`. It returns `None` when a required key is missing, the same
condition preflight refuses. A missing `head_count_kv` means the model has no
grouped-query attention, so it falls back to `head_count`, which is what that absence
means.

*Keys and values are priced separately.* They are two caches, and llama.cpp types them
independently, so `kv_bytes_per_token` takes both types and sums them instead of
doubling one. A common deployment quantizes the values harder than the keys, and
treating them as one type would misprice every figure after it.

*Whole numbers only.* The quantized types store blocks of 32 elements plus their scales,
so the cost per element is fractional: `Q4_0` is 16 bytes of nibbles plus one f16 scale
over 32 elements, or 18/32 of a byte. `kv_elem_bytes_x32` returns the cost of 32
elements, so the arithmetic stays in integers. A rounding error here would be multiplied
by every layer and every token.

*The head width is read, not derived.* `embedding_length / heads` is only a fallback for
a header that does not state `attention.key_length`; the doc on
`ModelDims::key_length` says why it cannot be trusted on its own.

`KvType` lists the element types llama.cpp accepts for a K or V cache. `KvType::parse`
uses the names llama.cpp's `--cache-type-k` flag takes, ignores case, and returns `None`
for anything else. 2.x fell back to `q8_0` on a typo, which hid the typo.
`KvType::is_quantized` matters because llama.cpp needs flash attention for a quantized V
cache, so a layout without the `flash_attn_ext` op cannot use one.

## What each file does

- `preflight.rs` — reads up to 64 MB of the file, checks the metadata llama.cpp requires,
  and refuses a tensor index that is cut off or a file that is a later part of a split
  model. It does not decide which architectures dovata supports; a layout does that.
- `model.rs` — `Model`, one model in one or more files. A split GGUF, named
  `<name>-00001-of-00003.gguf` and so on, is opened from its first part; the others are
  found by llama.cpp's naming rule and checked to belong: right part numbers, tensor
  counts that add up, no tensor in two parts. The parts are never merged. Each tensor
  keeps the part it lives in and its offset in that part's file.
- `dims.rs` — `ModelDims`, the model's shape from its header, and `KvType`, the KV cache
  element type. `kv_bytes_per_token` prices one token of KV cache, checked against the
  864 MiB KV buffer llama.cpp reports for Qwen3-Coder-30B at 32768 tokens.

## What changed in 3.0

2.x read the `JADE_*` environment into a runtime config, picked a model profile, and
guessed a generation cap from the VRAM left after the weights. The config and the
profiles went with the runtime. The guess went because the planner sizes the KV cache
exactly.

## Who uses it

The layout binder and the planner take a `Model`. `gguf-inspect` uses the reader
directly, since it must work on files that preflight would refuse.
