# scripts — developer tooling

Python scripts you run by hand to set up a machine, fetch or build a model, and configure
provider credentials. Stdlib only, no dependencies to install; they shell out to the same
system tools (`curl`, `git`, `cmake`, `aws`) the work actually needs.

These were bash. The flatten converted them to Python because the quantization pipeline in
particular had grown past what is reasonable in shell — argument handling, provenance
recording, preflight disk checks, and error paths that need to fail clearly rather than
half-run. Python keeps the flow readable; nothing here is a library.

## What each script does

- `_common.py` — shared helpers: the `info`/`ok`/`warn`/`die` logging, a `run` wrapper,
  checksum and disk-usage utilities, `.env` loading, and `ENV_ALIASES` /
  `normalize_env_aliases`, which reconcile the env file's variable names with the
  older `JADE_`-prefixed ones and with the AWS CLI's own spellings. Every other script
  adds this directory to `sys.path` and imports from it — including
  `src/ec2-tests/_ec2.py`, which reaches over for the alias map specifically so there
  is only one copy of it.
- `setup-dev.py` — creates a project-local `.venv` and installs the AWS CLI into it.
  Writes nothing globally. Credentials come from the environment, not from a config file.
- `download-model.py` — fetches a small CPU-capable GGUF (Llama-3.2-3B-Instruct-Q4_K_M,
  about 2 GB) to `models/model.gguf` for local development. Override with `MODEL_URL` and
  `MODEL_DEST`.
- `quantize-model.py` — the production model pipeline, and the serious one. Downloads (or
  takes locally-merged) weights, converts to GGUF f16, builds a code-calibrated importance
  matrix, and quantizes to Q4_K_M. Emits the model, its `.sha256`, and a `.buildinfo`
  provenance file into `dist/model`. Its `--from-url` mode is the shortcut for a model
  that is already quantized: it fetches a `.gguf` straight from a URL, verifies the
  header, and publishes it to the same S3 contract — no GPU, no llama.cpp build, no
  150 GB of scratch.
- `gen-profile-package.py` — regenerates the Jade-side model profile
  (`src/dovata/profiles/<model>.jde`) from the engine-side `<model>.toml`. A profile is
  two files describing one model, shipped on two different pipelines with no runtime link
  between them, and they drifted on the tool-call delimiters. The TOML is the source of
  truth now; this copies the three shared constants down into the marked region of the
  `.jde` and leaves the hand-written parser body alone. `--check` writes nothing and
  exits non-zero if a package is stale, which is what `publish-providers` runs before it
  builds. Needs Python 3.11+ for `tomllib`.
- `configure-provider.py` — writes `~/.jade/providers/<name>.json` at mode 0600 with an
  API key, which is where the Anthropic and OpenAI providers look when the corresponding
  `*_API_KEY` env var is not set. Prompts hidden if the key is not given as an argument.

## Two ways a model reaches S3

Both land at `s3://jade-os-release-bucket/models/model.gguf` with two sidecars,
`.sha256` and `.buildinfo`, so nothing downstream can tell them apart. What differs is
provenance, and the `.buildinfo` says which one you got. It ships next to the model
rather than staying on the build machine, because whoever ends up holding the file is
the person who needs to know where it came from.

The pipeline is for a model we build: it pins every input through
`ci/model-version.lock` and the result is reproducible. `--from-url` is for a model
someone else already quantized — it downloads, verifies, and republishes, and its
buildinfo records the source URL and states plainly that it is *not* reproducible from
the lock. Reach for the pipeline whenever the weights are ours, which is every
fine-tune; reach for `--from-url` when upstream has already shipped the exact GGUF you
want and quantizing it again would just burn GPU hours to arrive at the same file.

`--from-url` rewrites a HuggingFace `/blob/` link to `/resolve/`, so pasting the URL out
of the browser works. It refuses a sharded GGUF, since the contract is one file, and it
runs `gguf-inspect` before upload — an HTML error page saved as `model.gguf`, a
truncated transfer, or an architecture the pinned engine cannot load all fail here
rather than in S3.

## The quantization contract

`quantize-model.py` uses the **exact llama.cpp commit the daemon links**, read from the
`src/infer-engine/ffi/llama.cpp` submodule with `git ls-tree`. That is not a nicety — a
GGUF written by a different commit can be rejected outright for quant-type or
GGUF-version skew. The commit is read from the submodule rather than recorded anywhere,
so it cannot drift from what the daemon actually links; it used to be a hand-copied line
in `ci/model-version.lock`, and by the time it was removed the copy was two weeks stale.

`ci/model-version.lock` still holds the model repo, revision, and quant scheme. Bumping
a value there is the intended way to cut a new model release.

It also cannot run on a laptop. Peak disk is around 150 GB (safetensors ~60, f16 GGUF
~60, Q4_K_M ~18) and the imatrix pass needs a CUDA GPU. `src/ec2-tests/quantize.py` is
the one-command driver that starts the GPU box, syncs the repo, runs this over SSH, and
uploads in-region. The script preflights and fails fast rather than getting halfway.

## Who uses it

Humans, mostly. `src/ec2-tests/quantize.py` and `src/ec2-tests/first-run.py` invoke
`quantize-model.py` remotely. The provider READMEs in `src/anthropic/` and `src/openai/`
point at `configure-provider.py` as the way to install a key. Nothing imports these as a
library, and no build depends on them.

## Running

Always from the repo root, always with an explicit `python3`:

```sh
python3 src/scripts/setup-dev.py
python3 src/scripts/download-model.py
python3 src/scripts/configure-provider.py anthropic     # prompts for the key
python3 src/scripts/quantize-model.py --help
```

Each script's module docstring carries its full option list — read that rather than
trusting a summary here.
