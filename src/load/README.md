# load/ — reading a model file

Everything dovata does starts here: read a model file's header, refuse it if it cannot
be planned, gather the parts of a split model into one value, and read the model's shape.

## Why it is its own capability

Because it is policy, and [`src/gguf/`](../gguf/README.md) is not. The GGUF reader
reports what any file says, even a truncated one, because `gguf-inspect` has to work on
a half-downloaded model. Planning needs more: every required key present and every
tensor in the index. This module is where that line is drawn.

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
