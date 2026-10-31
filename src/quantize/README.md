# quantize: a smaller GGUF, made by llama.cpp's own quantizer

This capability turns a GGUF into a quantized copy, such as a BF16 model into Q4_K_M.
It calls `llama_model_quantize` from the vendored llama.cpp, the same function the
`llama-quantize` tool runs, so its output is byte for byte what that tool would write.

It only runs when a caller asks. Fetching a model never quantizes it, and planning
never quantizes it either. A caller who wants a smaller model calls
`quantize::quantize(input, output, &spec)` and gets a `QuantReport` back.

## Why it is built this way

*The formats belong to ggml.* The Jade compiler's kernels read ggml's block layouts,
and a "Q4_K_M" file is not one type but a mix that llama.cpp chooses per tensor: Q4_K
for most weights, Q6_K for some, F32 for the norms. A second quantizer would have to
match all of that exactly. Calling the real one is the only way to be sure it does.

*Quantizing is never a side effect.* A 30B model is tens of gigabytes read and written
and several minutes of work. If `fetch` or the planner did it on their own, a typo in a
type name could cost an hour and fill a disk. So nothing calls `quantize` except the
caller, and a repo that lacks the requested type is an error from `hub`, not a cue to
quantize.

*The pure half builds everywhere.* The type table, the checks, and the output naming
rule have no native code, so they are tested on any machine. Only the call into
llama.cpp needs the `quantize` cargo feature, which compiles llama.cpp through the sys
crate. With the feature off, `quantize()` still exists and returns `NotBuilt`.

## Files

- `mod.rs` holds the pure half. `Ftype` is the table of file types, one variant per
  `llama_ftype`, spelled the way `llama-quantize` spells them, with each variant's value
  equal to its `llama_ftype` number. `QuantSpec` says what to make, `QuantReport` says
  what was made, and `QuantError` says why nothing was. `default_output` names an output
  when the caller has none. `prepare` runs every check that needs no llama.cpp. With the
  feature off, this file also holds the `quantize()` that returns `NotBuilt`.
- `exec.rs` is the native edge, built only with the `quantize` feature. It opens the
  input, runs `prepare`, makes the output directory, calls `llama_backend_init` once and
  `llama_model_quantize` in dovata's only `unsafe` block, then opens the output again and
  checks that its `general.file_type` is the type asked for. Its tests quantize a real
  file and compare dovata's ggml type table with ggml's own.
- `fixture.rs` is test-only. It writes a tiny but complete `llama` model in F32 with
  `crate::gguf::write`: two layers, a 64-wide embedding, and a 32-token SentencePiece
  vocabulary, with every shape `load_arch_tensors` in llama.cpp expects. A test loads it
  in llama.cpp to prove that. It can also write the model as two split parts.

## How a call goes

A call runs in four steps, and every refusal happens before llama.cpp starts.

1. The input is opened as a `load::Model`. A split model is opened from its first part,
   and its other parts are found by llama.cpp's `-00001-of-00003.gguf` rule.
2. `prepare` checks the request against the model's header: the type, the input's
   current type, and where the output would go.
3. llama.cpp gets only the first part's path. Its loader finds the other parts with
   `llama_split_prefix` and `llama_split_path`, the same rule step 1 used, so it reads
   the files dovata already checked.
4. The output is opened as a `Model` and its `general.file_type` is compared with the
   request. A run that returned success but wrote something else is still an error.

llama.cpp logs to stderr while it works, one line per tensor, as `llama-quantize` does.
That log is left on, because a big model takes minutes and the log is the only sign of
progress, and because when llama.cpp fails, the reason is in it.

## What a user will see when it refuses

| Error | When | Message, in short |
|---|---|---|
| `NotBuilt` | the `quantize` feature is off | this build cannot quantize; enable the `quantize` cargo feature |
| `AlreadyAt` | the input already is the requested type | `x.gguf` is already Q8_0; use the file as it is |
| `Requantize` | the input holds quantized tensors and `allow_requantize` is off | already quantized; quantizing again loses quality; start from the unquantized file or set `allow_requantize` |
| `BadInput` | the input does not open, the output is an input file, the output is in the hub cache, the type is NVFP4, or the type needs an importance matrix | says which, and what to do instead |
| `Llama` | llama.cpp returned an error, or wrote a file with the wrong type | the reason is in llama.cpp's log on stderr |

*Already at the type is an error on purpose.* The alternatives are worse. Copying the
input to the output is the 60 GB accident the check exists to stop. Returning success
without writing the output would break a caller that deletes its output later, since it
would delete the input. The error carries the input's path, so a caller that is happy
with the file as it is can use it.

*Requantizing is allowed but never silent.* Quantizing a Q8_0 file to Q4_K_M loses more
than quantizing the BF16 original once. It needs `allow_requantize`, and the report's
`requantized` field says it happened. The check looks at the tensors, not only at
`general.file_type`, so a file without that key is judged correctly.

*Two kinds of type are refused up front.* NVFP4 files load in the pinned llama.cpp, but
its quantizer cannot write one. The types whose default tensor type needs an importance
matrix are IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS and Q2_K_S. llama.cpp
refuses those without a matrix, and `QuantSpec` has no way to pass one yet.

## Where the output goes

`default_output(input, ftype)` names the output `<stem>.<FTYPE>.gguf`. The stem is the
input's file name less `.gguf` and less any part suffix, so
`Qwen3-Coder-30B-A3B-Instruct-BF16-00001-of-00002.gguf` becomes
`Qwen3-Coder-30B-A3B-Instruct-BF16.Q4_K_M.gguf`. The source type stays in the name, so
copies made from different sources never share a name.

The output goes next to the input, except when the input is inside a Hugging Face hub
cache. That cache keeps each file once under `blobs/`, and lists each revision under
`snapshots/<commit>/` as symlinks into it. hf-hub owns both directories and may
overwrite or delete what is in them. So for an input at
`<cache>/models--org--repo/snapshots/<commit>/<dirs>/<file>`, the output goes to
`<cache>/models--org--repo/dovata/<commit>/<dirs>/<stem>.<FTYPE>.gguf`. It sits beside
`snapshots/` instead of inside it, it stays with the repo it came from, and deleting the
repo from the cache deletes it too.

`quantize()` enforces the rule for any output, not only the default one. An output
inside a cache's `snapshots/` or `blobs/` is refused, and the check runs again on the
resolved path, so a symlinked directory cannot get around it.

## Split models

dovata never merges a split model, and `QuantSpec::new` keeps that true for quantized
copies: `keep_split` starts on, so a model in three parts gives a copy in three parts,
named `<stem>.<FTYPE>-00001-of-00003.gguf` and so on. Each tensor stays in the part it
came from. `QuantReport::output` is the first part, which is the one to open. With
`keep_split` off, llama.cpp writes one file holding everything. A one-file input always
gives one file.

## How the tables stay right

The `Ftype` table is checked against the pinned llama.cpp by tests that read its
sources:

- `ftype_table_matches_llama_h` parses the `llama_ftype` enum in `include/llama.h`. Every
  `Ftype` must have its value there, and every value there must be an `Ftype`.
- `ftype_names_match_llama_quantize` parses `QUANT_OPTIONS` in
  `tools/quantize/quantize.cpp`, so every name is the one the tool takes.
- `targets_and_imatrix_needs_match_llama_quant` parses `src/llama-quant.cpp`, so the list
  of writable types and the list of types that need an importance matrix match it.
- `type_table_matches_ggml`, with the feature on, asks ggml itself for each type's block
  size, byte size, name and whether it is quantized, and compares them with
  `crate::gguf`'s table.

The first three need the llama.cpp submodule checked out. Without it they fail and say
so, because a drift check that passes when it cannot see its reference proves nothing.

## Testing

```sh
# Everything, including a real quantization of the fixture
cargo test -p dovata --lib quantize

# The pure half alone, as a build without llama.cpp sees it
cargo test -p dovata --lib --no-default-features quantize
```

## Who uses it

Nothing inside dovata calls `quantize`, by design. Its callers are the Jade compiler
and, once it is written, the `dovata-plan` command. The plan is for that command to
quantize only when a user asks for a type the Hub repo does not offer: it fetches the
unquantized model with `hub::fetch`, then passes it here with `default_output` as the
destination. The planned `refback` crate will use `crate::gguf::write` the way
`fixture.rs` does, to make its own small models.
