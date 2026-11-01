# hub: getting a model's GGUF files, from disk or from the Hugging Face Hub

This capability turns a model source into the paths of one model's files on disk. The
source is a local path or a Hub link, such as
`https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`.
`hub::fetch(&source, want, &opts)` returns a `Fetched` holding every part
of the model, part 1 first, along with its type and, for the Hub, the exact commit.

It fetches exactly one model each time. A split model is one model, so `fetch` gets
every part of it. It never quantizes: a repo that lacks the type asked for is an error
that lists what the repo has. Whether the source is a path or a link, the files are
opened as one `load::Model` before `fetch` returns, so the Jade compiler never gets parts
that do not belong together.

## Why it is built this way

*One model, never a guess.* GGUF repos commonly hold twenty or more versions of the same
model, from BF16 at 61 GB down to 1-bit quants at 8 GB. Downloading the wrong one wastes
an hour. So the choice follows fixed rules, and when the rules cannot decide, `fetch`
stops and names the candidates instead of picking one.

*Never quantize as a side effect.* Making a missing type is the caller's decision,
through `quantize::quantize`. The `NotOffered` error says how. Quantizing takes minutes
and writes tens of gigabytes, so it never happens inside a fetch.

*The user always gives the link.* A Hub source carries the server it comes from, taken
from the link the user typed. There is no default server, a bare `org/repo` is refused,
and `HF_ENDPOINT` is not read, so nothing sends a download somewhere the user did not
name.

*Share the cache the Hugging Face tools use.* Files go into the standard hf-hub cache, so
a model already downloaded by `huggingface-cli` or the Python library is used as it is.

*Pure core, network edge.* Reading a source, grouping a repo's files and choosing one are
pure functions, tested against real file lists. Local paths always work. The download is
behind the `hub` cargo feature; without it, a Hub source is refused with a message that
names the feature.

## Files

- `mod.rs` holds everything but the download. `ModelSource` and its `parse`, `Want`,
  `Level`, `levels` to group a file list into models, `select` and `select_file` to
  choose one, the `FetchOptions`, `Fetched` and error types, `fetch` itself, which
  handles a local path and hands a Hub source to `fetch.rs`, and `offered`, which lists
  a repo's models without downloading any, so a caller can ask which one to fetch.
- `fetch.rs` is the network edge, built only with the `hub` feature. It lists the repo
  with hf-hub 0.4.3's blocking client over `ureq`, downloads every part of the chosen
  model at one commit, and checks the result. It also works out the cache directory and
  the token from the environment. The endpoint comes only from the source. The 0.4 line
  of hf-hub is used because its blocking client needs no tokio, so the Jade compiler's
  build does not pull tokio in; the note beside the dependency in `Cargo.toml` says the
  same.
- `testdata/` holds real file lists copied from the Hub API on 2026-10-31, one name per
  line, with the repo and commit at the top. The selection tests run against them: two
  unsloth repos with split BF16 models, one with `mmproj` files, TheBloke's Llama-2
  with no unquantized file, bartowski's lower-case names, Qwen's own split `fp16`
  upload, and a repo with safetensors only.

## Sources

`ModelSource::parse` reads these forms:

| Form | Example |
|---|---|
| a local path | `/models/x.gguf`, `./x.gguf`, `~/x.gguf` |
| a repo link | `https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` |
| a file link | `.../blob/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` |
| a download link | `.../resolve/main/BF16/Qwen3-Coder-30B-A3B-Instruct-BF16-00001-of-00002.gguf` |
| a revision link | `.../tree/b17cb02dd882d5b6ab62fc777ad2995f19668350` |
| a directory link | `.../tree/main/BF16` |

The rule is applied in this order. A URL is always the Hub. A string that starts with
`/`, `.` or `~` is always a path. A string that names something on disk is a path. A
bare `org/repo` that is not on disk is refused, and the error shows the full link it
most likely meant. Anything else is a path, and `fetch` then reports it as missing.

The endpoint is the link's host. `hf.co` and `www.huggingface.co` links use
`https://huggingface.co`, the host they redirect to, because hf-hub drops the token on a
redirect to another host. The endpoint is always `https`, even for an `http` link, so
the token never goes out in plain text.

A link that names a file takes exactly that model, with every part when it is split,
even when the link points at part 4 of 6. A directory link takes the one model inside
the directory.

A repo id is two names joined by `/`, each made of letters, digits, `-`, `_` and `.`,
with no `..`. A link may escape part of its path, as Hub links do for a revision such as
`refs%2Fpr%2F1`, and `parse` decodes it.

A local path gives that file and, for a split model, every other part; any part may be
given. `Model::open` opens a one-file model or a first part. For a later part, `fetch`
reads `split.count` from the part's header to name the rest. A file whose header does
not read is passed to `Model::open`, which reports the error. `Want` does not apply to a
local path.

## How one model is chosen

`levels` groups a repo's files into models, and `select` picks one by `Want`.

1. Only `.gguf` files count. A name that starts with `mmproj` is a vision projector and
   a name that contains `imatrix` is an importance matrix; both are skipped.
2. Split parts, named `<name>-00001-of-00003.gguf` and so on in one directory, form one
   model. unsloth puts them in a directory named for the type, such as `BF16/`. Files
   are grouped by directory, by name without the part suffix, and by part count.
3. The type comes from the file name. Every llama.cpp type name is tried, longest first,
   ignoring case. So are `llama-quantize`'s aliases Q3_K, Q4_K and Q5_K, and three
   spellings repos use: FP16, FP32 and MXFP4. A match must start the name or follow `-`,
   `.` or `_`, and must end the name or come before `-` or `.`. The end rule keeps `Q4_K`
   from matching inside `Q4_K_XL`, and `Q4_0` inside the retired `Q4_0_4_4`. When the
   file name has no type, its directory is tried.
4. A label that is not a llama type keeps its name and has no type: unsloth's
   `UD-Q4_K_XL` and `Q2_K_L`, bartowski's `Q4_K_L` and `Q6_K_L`. Anything after `UD-`
   has no type either, even `UD-IQ1_M`, because unsloth's Dynamic quants choose types per
   tensor and are not what `llama-quantize` makes under that name. All of these can
   still be fetched by naming the file.
5. `Want::Unquantized`, the default, takes BF16, then F16, then F32. BF16 is first
   because most models are trained in it, so it holds the weights exactly at half the
   size of F32. When the repo has none of the three, it takes the most precise type
   offered. A repo with one model and no type in its name gives that model.
6. `Want::Ftype(x)` takes the model of type x, or fails with `NotOffered`.

Against the real lists in `testdata/`:

| Repo | `Want::Unquantized` gives |
|---|---|
| `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` | `BF16/…-BF16-00001-of-00002.gguf` and its second part |
| `unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF` | all ten parts under `BF16/` |
| `unsloth/gemma-3-4b-it-GGUF` | `gemma-3-4b-it-BF16.gguf`, not `mmproj-BF16.gguf` |
| `bartowski/Qwen_Qwen3-8B-GGUF` | `Qwen_Qwen3-8B-bf16.gguf` |
| `Qwen/Qwen2.5-7B-Instruct-GGUF` | the four `fp16` parts, read as F16 |
| `TheBloke/Llama-2-7B-GGUF` | `llama-2-7b.Q8_0.gguf`, the most precise it has |
| `Qwen/Qwen3-8B` | the `NoGguf` error |

The precision order is `quantize::Ftype::ALL`: bits per weight, with close types in the
order their files come out for one real model.

## What a fetch from the Hub does

1. List the repo at the requested revision, `main` by default. The listing also gives
   the commit that revision points at.
2. Choose one model, by the named file or by `Want`.
3. Get every part at that commit. A part already in the cache is used as it is; any
   other is downloaded. Pinning the commit means a branch that moves in the middle of a
   ten-part download cannot mix two uploads. hf-hub 0.4.3 finds cached files only
   through `refs/<revision>`, so `fetch.rs` looks in `snapshots/<commit>/` first, where
   every tool puts them; otherwise a model cached under `main` would be downloaded again.
4. Open the parts as one `load::Model`, and refuse a file whose `general.file_type`
   disagrees with the type its name gave. A mislabeled upload would otherwise plan as
   one type and hold another. A file with no `general.file_type` cannot be checked, so
   it is accepted and `Fetched::ftype` is `None`.

`Fetched::revision` is the commit sha, even when the source named a branch, so a build
can record exactly what it used.

## Settings from the environment

The settings follow the Python `huggingface_hub`, so one setup serves both.

- The cache is `FetchOptions::cache_dir`, else `HF_HUB_CACHE`, else `$HF_HOME/hub`, else
  `$XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`.
- The token is `FetchOptions::token`, else `HF_TOKEN`, else the older
  `HUGGING_FACE_HUB_TOKEN`, else the file `huggingface-cli login` writes: the path in
  `HF_TOKEN_PATH`, else `token` in the `HF_HOME` directory, normally
  `~/.cache/huggingface/token`.
- The `HF_HOME` directory is `HF_HOME`, else `$XDG_CACHE_HOME/huggingface`, else
  `~/.cache/huggingface`, the same search the cache uses.
- An environment variable set to an empty string counts as unset. A token is trimmed,
  and a blank token counts as no token.

The cache holds each repo as `models--org--repo/{blobs,snapshots,refs}`, the layout
`huggingface-cli` and the Python library write, so a model either of them downloaded is
not downloaded again.

Reading the token is deliberate. Public repos need none, but gated ones such as Meta's
Llama do, and a build tool that ignored a login the user had already done would fail for
no reason they could see. The token goes only to the Hub endpoint, and printing a
`FetchOptions` with `{:?}` shows `<redacted>` in its place.

hf-hub draws a progress bar on stderr while it downloads, and hides it when stderr is
not a terminal, so a build log gets no bar.

## What a user will see when it fails

| Error | When | Message, in short |
|---|---|---|
| `NotBuilt` | a Hub source, with the `hub` feature off | no Hugging Face client in this build; enable the `hub` feature, or pass a local path |
| `BadSource` | a malformed repo id, an endpoint that is not `https`, or no cache directory | says which |
| `Select` with `NoGguf` | the repo has no GGUF model files | safetensors-only repos are not supported yet; convert with `convert_hf_to_gguf.py` or use a GGUF repo |
| `Select` with `NotOffered` | the repo lacks the type | lists what it offers, then: fetch with `Want::Unquantized` and pass it to `quantize::quantize` |
| `Select` with `Ambiguous` or `Unclear` | the rules fit several models, or none has a type | lists the candidates; name the file |
| `Select` with `NoSuchFile` or `NotAModel` | the named file is missing, or is a projector or imatrix | says which, and what the repo offers |
| `Hub` | the network or the Hub failed | the hf-hub error, with a hint for 401 and 403 (a gated or private repo: accept its terms, set `HF_TOKEN`), 404 and 429 |
| `Model` | the files do not open as one model | the `load::Model` error, such as a missing part |
| `Mismatch` | the header's type disagrees with the file name | names both; name the file to fetch it anyway |

`ModelSource::parse` has its own `SourceError` for a string it cannot read, such as a
bare `org/repo`, a link to another site, a dataset, or a file that is not a GGUF.

The hints for 401, 403, 404 and 429 come from the HTTP status in hf-hub's error text.
The ureq crate prints a status error as `<url>: status code 404`. Reading the status from
that text avoids taking ureq on as a direct dependency just to match its error type, and
when the text does not match, only the hint is lost.

## Testing

```sh
# The pure tests, which need no network
cargo test -p dovata --lib hub

# The one network test: downloads about 28 MB from ggml-org/tiny-llamas into a temp dir
cargo test -p dovata --lib -- --ignored fetches_a_small_public_gguf
```

## Who uses it

Nothing inside dovata's library calls `fetch`. Its callers are the Jade compiler and
the `dovata plan` command, which calls `offered` first and stops to ask for `--quant`
when a repo offers several models. They hand the parts to `load::Model::from_parts`,
and they call `quantize::quantize` themselves when the type they want is not offered.
