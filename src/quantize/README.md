# quantize: a smaller GGUF, made by llama.cpp's own quantizer

This capability turns a GGUF into a quantized copy, such as a BF16 model into Q4_K_M.
It runs llama.cpp's `llama-quantize` program, so its output is exactly what that tool
writes. llama.cpp is not part of this repo: the program must be installed on the
machine and on `PATH`, from `brew install llama.cpp` on macOS or a release build from
https://github.com/ggml-org/llama.cpp/releases on Linux. It is the only part of
llama.cpp dovata uses.

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

*Run the program, do not link the library.* An earlier version linked llama.cpp's C
API through a vendored copy of llama.cpp and a bindings crate, which made every build
compile llama.cpp with cmake. It was replaced because llama.cpp does not keep its ABI
stable: the `llama_model_quantize_params` struct changed 15 times in the history of
`include/llama.h`, most recently in April 2026 in PR #20346, so declarations written for
one release read the wrong bytes with another, with no error. The program's command
line has stayed stable, and a mismatch there fails with a usage message instead of
corrupting memory. dovata now has no C or C++ code and no `unsafe` block.

*The pure half builds everywhere.* The type table, the checks, and the output naming
rule are tested on any machine. Running the program needs the `quantize` cargo feature.
With the feature off, `quantize()` still exists and returns `NotBuilt`, so a program
compiled without it gets an error that names the feature. With the feature on and no
`llama-quantize` on `PATH`, it returns `NotInstalled`, which says how to install it.

## Files

- `mod.rs` holds the pure half. `Ftype` is the table of file types, one variant per
  `llama_ftype`, spelled the way `llama-quantize` spells them, with each variant's value
  equal to its `llama_ftype` number. `QuantSpec` says what to make, `QuantReport` says
  what was made, and `QuantError` says why nothing was. `default_output` names an output
  when the caller has none. `prepare` runs every check that needs no llama.cpp. With the
  feature off, this file also holds the `quantize()` that returns `NotBuilt`.
- `exec.rs` is the edge that runs the program, built only with the `quantize` feature.
  It opens the input, runs `prepare`, makes the output directory, finds
  `llama-quantize` on `PATH`, builds its arguments and runs it, then opens the output
  again and checks that its `general.file_type` is the type asked for. Its tests run the
  real program on the fixture, and they skip, with a message, when it is not installed.
- `fixture.rs` is test-only. It writes a tiny but complete `llama` model in F32 with
  `crate::gguf::write`: two layers, a 64-wide embedding, and a 32-token SentencePiece
  vocabulary, with every shape `load_arch_tensors` in llama.cpp expects. A test loaded
  it in llama.cpp to prove that until the library was unlinked; the quantize tests still
  run it through llama.cpp's loader. It can also write the model as two split parts. See
  "The fixture model" below.

## How a call goes

A call runs in four steps, and every refusal happens before llama.cpp starts.

1. The input is opened as a `load::Model`. A split model is opened from its first part,
   and its other parts are found by llama.cpp's `-00001-of-00003.gguf` rule.
2. `prepare` checks the request against the model's header: the type, the input's
   current type, and where the output would go.
3. `llama-quantize` gets only the first part's path. It calls `llama_model_quantize`,
   which builds a
   `llama_model_loader` with an empty split list, and the loader, seeing
   `split.count > 1`, names the other parts with `llama_split_prefix` and
   `llama_split_path`: `<prefix>-%05d-of-%05d.gguf` next to the first. That is the same
   rule step 1 used, so it reads the files dovata already checked. The loader also
   refuses to start from any part but the first, which `Model::open` has already
   enforced.
4. The output is opened as a `Model` and its `general.file_type` is compared with the
   request. A run that returned success but wrote something else is still an error.

An existing output file is overwritten, and a missing output directory is created.
llama.cpp writes each file's header last, so an output cut short by a crash or a full
disk has a zeroed header and never opens as a model. Before anything is written, the
output paths are checked a second time on the resolved paths, because a symlinked
directory can put an output inside the hub cache or make it another name for an input
file.

The program is called as `llama-quantize [--allow-requantize] [--keep-split] <input>
<output> <TYPE> [nthreads]`. Both paths are made absolute first. The reason is in
`try_parse_ftype` in llama.cpp's `tools/quantize/quantize.cpp`: the program decides
whether its second argument is the output or the type by trying to read it as a type,
and that includes `std::stoi`, so an output named `7b-model.gguf` would be read as type
7, Q8_0. An absolute path starts with `/` and never reads as a number. The thread count
is passed only when `QuantSpec::nthread` is set; otherwise the program uses one thread
per hardware thread.

llama.cpp logs one line per tensor while it works. dovata copies that log to its own
standard error line by line as it arrives, because a big model takes minutes and the
log is the only sign of progress. It also keeps the last 20 lines, and a failure's
`QuantError::Llama` quotes them, so the reason is in the error itself.

## What a user will see when it refuses

| Error | When | Message, in short |
|---|---|---|
| `NotBuilt` | the `quantize` feature is off | this build cannot quantize; enable the `quantize` cargo feature |
| `NotInstalled` | `llama-quantize` is not on `PATH` | install llama.cpp, with `brew install llama.cpp` on macOS or a release build on Linux |
| `AlreadyAt` | the input already is the requested type | `x.gguf` is already Q8_0; use the file as it is |
| `Requantize` | the input holds quantized tensors and `allow_requantize` is off | already quantized; quantizing again loses quality; start from the unquantized file or set `allow_requantize` |
| `BadInput` | the input does not open, the output is an input file, the output is in the hub cache, the type is NVFP4, or the type needs an importance matrix | says which, and what to do instead |
| `Llama` | `llama-quantize` failed, or wrote a file with the wrong type | the exit status and the last 20 lines of its log |

*Already at the type is an error on purpose.* The alternatives are worse. Copying the
input to the output is the 60 GB accident the check exists to stop. Returning success
without writing the output would break a caller that deletes its output later, since it
would delete the input. The error carries the input's path, so a caller that is happy
with the file as it is can use it.

*Requantizing is allowed but never silent.* Quantizing a Q8_0 file to Q4_K_M loses more
than quantizing the BF16 original once. It needs `allow_requantize`, and the report's
`requantized` field says it happened. The check looks at the tensors, not only at
`general.file_type`, so a file without that key is judged correctly. It uses the same
split as ggml's `ggml_is_quantized`: F32, F16, BF16, F64 and the integer types are plain,
and every other type, known or not, counts as quantized, so an unknown type is never
silently requantized.

*Two kinds of type are refused up front.* NVFP4 files load in llama.cpp, but
its quantizer cannot write one: `llama_ftype_get_default_type` has no case for it, so
asking for one fails. The types whose default tensor type needs an importance matrix are
IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS and Q2_K_S. llama.cpp refuses those
without a matrix, because the result would be unusable, and `QuantSpec` has no way to
pass one yet. The list is the types whose default tensor type is one that
`tensor_requires_imatrix` in `src/llama-quant.cpp` names, plus Q2_K_S, which that
function names by ftype.

## Type names

`Ftype` spells each type the way `llama-quantize` does, and `Ftype::parse` reads names
the same way, ignoring case. It takes the tool's three aliases too: Q3_K, Q4_K and Q5_K
mean the `_M` mix of each. NVFP4 is the one type `llama-quantize` does not offer; its
name is the one llama.cpp prints for it. An unknown name is an `UnknownFtype` error whose
message lists every real name. `Ftype::from_file_type` returns `None` for a
`general.file_type` value llama.cpp no longer writes, such as the removed Q4_0_4_4, and
for one newer than the table.

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
resolved path, so a symlinked directory cannot get around it. `default_output` itself
works on the path as given and never touches the disk. hf-hub names a repo's directory
`models--org--repo`, `datasets--…` or `spaces--…`, and that is how a path inside the
cache is recognized.

## Split models

dovata never merges a split model, and `QuantSpec::new` keeps that true for quantized
copies: `keep_split` starts on, so a model in three parts gives a copy in three parts,
named `<stem>.<FTYPE>-00001-of-00003.gguf` and so on. Each tensor stays in the part it
came from. `QuantReport::output` is the first part, which is the one to open. With
`keep_split` off, llama.cpp writes one file holding everything. A one-file input always
gives one file.

For a split output, llama.cpp is given the output path less `.gguf`, or less a whole
`-00001-of-00003.gguf`, the same name `llama-quantize --keep-split` passes, and it adds
the part suffixes itself.

## The fixture model

llama.cpp's quantizer runs its real model loader, which reads the architecture's
hyperparameters with exact GGUF types, so a fixture has to be a model llama.cpp would
load, not only a well-formed GGUF. `fixture.rs` follows llama.cpp at commit
52b3df0023659b142ce29f75c7a82cf437769c33:

- `load_hparams` in `src/llama-model.cpp` reads `context_length`, `embedding_length` and
  `block_count` as `u32`, and for `llama` refuses a `rope.dimension_count` that is not
  the head size.
- `llama_model_llama::load_arch_hparams` in `src/models/llama.cpp` requires
  `attention.layer_norm_rms_epsilon` as an `f32`.
- `load_arch_tensors` there fixes every shape: `token_embd` is `{n_embd, n_vocab}`,
  `attn_k` and `attn_v` are `{n_embd, head_dim * n_head_kv}`, and the FFN is `gate` and
  `up` at `{n_embd, n_ff}` with `down` at `{n_ff, n_embd}`.
- The vocabulary is a `llama` SentencePiece one, which needs `<0x0A>` for its newline
  token and unique token strings.

The widths are multiples of 32, so Q8_0 and Q4_0 quantize every weight without falling
back to another type. Norm tensors are all ones, as a freshly initialized model's are.
The other weights are small values from a fixed generator seeded by each tensor's name,
so tensors differ from each other and every run writes the same bytes. The tensors are
listed in the order llama.cpp's converter writes them. The split form is written the
way `gguf-split` writes one: part 1 holds the metadata and the first half of the
tensors, and part 2 holds only the split keys and the rest. Each test writes its files into
its own directory under the system temp directory, named with the test and the process
id, so tests running at the same time never share files.

## How the tables stay right

The tables are checked against the real llama.cpp, through the installed program and
through real files:

- `ftype_table_matches_llama_quantize_help` reads the type list `llama-quantize --help`
  prints, each line a number and a name such as `15  or  Q4_K_M`. Every name it lists
  must parse to the `Ftype` with that number, and an `Ftype` is listed exactly when
  `is_target` says the program can write it.
- `type_sizes_match_what_llama_quantize_writes` quantizes the fixture to every type that
  needs no importance matrix and checks dovata's size for every tensor against where the
  program put the next one: each tensor's data, padded to the file's alignment, must end
  exactly where the next tensor starts. The fixture's rows are 64 and 128 wide, so this
  covers the types with 32-wide blocks. Q1_0 is left out, since it needs rows a multiple
  of 128 and llama.cpp has no fallback for it.
- `a_real_headers_tensors_sit_end_to_end` in `src/load/model.rs` makes the same check on
  a real model's tensor index when `DOVATA_TEST_HEADER` names one, which covers the
  256-wide K-quant types. It passes on Qwen3-Coder-30B at Q4_K_M, Qwen3-0.6B at Q8_0 and
  Llama-3.2-1B at Q4_K_M.

Two checks from the linked version have no replacement. The table of which types need an
importance matrix was checked against `src/llama-quant.cpp`, and the non-writable types
against `include/llama.h`; both passed at commit 52b3df0 and are now plain data. A
newer llama.cpp that changes either shows up as a `QuantError::Llama` from the program.

## Testing

```sh
# Everything. The tests that run llama-quantize skip, with a message, when it is not
# on PATH, so install llama.cpp first for a full run.
cargo test -p dovata --lib quantize

# The pure half alone, as a build without the quantize feature sees it
cargo test -p dovata --lib --no-default-features quantize
```

## Who uses it

Nothing inside dovata calls `quantize`, by design. Its callers are the Jade compiler
and, once it is written, the `dovata plan` command. The plan is for that command to
quantize only when a user asks for a type the Hub repo does not offer: it fetches the
unquantized model with `hub::fetch`, then passes it here with `default_output` as the
destination. The planned `refback` crate will use `crate::gguf::write` the way
`fixture.rs` does, to make its own small models.
