# tools/ec2 — getting work onto the GPU box

Python scripts that drive the remote EC2 GPU instance from your laptop: start it, sync
the repo up, build, open a shell, stop it again. Stdlib only; they shell out to
`aws` and `ssh`.

The directory used to be called `ec2-tests`, which was misleading twice over — it holds
no tests, and the harness that *did* run the tests moved out to live with them. What is
left here is plumbing: everything whose job is to get code onto a machine that can run
it.

## Why it exists

The box used to be where the runtime engine was verified, since its llama.cpp backend
was Linux-only and its CUDA paths needed a real NVIDIA card. That engine went with the
3.0 runtime removal and lives on `main`. What still needs the box is quantization: the
pipeline wants around 150 GB of scratch disk and a CUDA GPU for the imatrix pass, and the
18 GB upload to S3 is faster in-region. The box is a `g5.xlarge` with an A10G, and these
scripts are the shortest path between an edit here and a result there.

They also exist to keep the box **stopped**. It costs money by the hour, and a workflow
where starting and stopping are one command each is a workflow where you actually stop it.

Each script used to be bash, and each one re-derived the repo root, sourced the env files,
located the AWS CLI, and resolved the hostname. That boilerplate is now in `_ec2.py` and
the individual scripts are a dozen lines.

## What each script does

- `_ec2.py` — the box-specific helpers: env loading, `aws` binary discovery (prefers the
  project `.venv`), hostname resolution with an `.ec2-host` cache, and `ssh` argv
  construction. Not run directly. The generic logging and env-alias helpers it builds on
  are one level up, in [`tools/_common.py`](../README.md).
- `start.py` — starts the instance and blocks until SSH answers.
- `stop.py` — stops it.
- `ssh.py` — an interactive shell, or `ssh.py <cmd...>` to run one command and exit.
- `first-run.py` — one-time provisioning, idempotent. Installs CUDA, LLVM 18, Rust,
  musl-tools, and the NVIDIA driver, then downloads the 30B model to
  `/jade/inference/model.gguf`. Run from your laptop it uploads itself to the box and
  re-execs there with `--local`.
- `quantize.py` — the manual driver for `tools/scripts/quantize-model.py`. Starts the box,
  rsyncs this repo, and runs the pipeline over SSH with CUDA on `PATH` and your AWS
  credentials forwarded, so the 18 GB upload to S3 happens in-region instead of round
  tripping through your laptop. Supports `--detached` (survives an SSH drop) and
  `--follow`.

## Where the eval harness and the REPL went

`run-tests.py` used to live here. It moved to a repo-root `tests/` next to the `.jde`
files it ran, because it was the runner for that suite and not GPU-box plumbing. That
suite went with the 3.0 runtime removal, since dovata no longer runs a model.

`repl.py` went with it. It opened a Jade REPL on the box backed by real inference, and
there is no inference left here to back it. Both live on `main`.

## Configuration

`load_env` reads an optional `.env` then `.ec2.env` from the repo root, and the scripts
then require these variables:

| Variable | Used for | Also accepted as |
|---|---|---|
| `INSTANCE_ID` | which instance to start/stop/describe | `JADE_INSTANCE_ID` |
| `AWS_REGION` | its AWS region | `JADE_REGION` |
| `SSH_KEY` | path to the `.pem`, relative to the repo root (see `keys/`) | `JADE_SSH_KEY` |
| `SSH_USER` | SSH login user | `JADE_SSH_USER` |
| `AWS_ACCESS_KEY_ID` | credentials, and what the AWS CLI itself reads | `AWS_ACCESS_KEY` |
| `AWS_SECRET_ACCESS_KEY` | ditto | — |

Three naming conventions have to coexist: the short names the env file uses, the
`JADE_`-prefixed ones the scripts used to read, and the AWS CLI's own spellings, which
are not ours to rename. Rather than pick a winner, `_common.ENV_ALIASES` maps each name
the code reads to the names accepted for it and `load_env` fills them in. Anything
already exported in your shell wins over anything in a file. That map is the single
definition — a second copy is how the code and the env file drifted apart before.

All of it is gitignored — the env files, the `keys/` directory, and the `.ec2-host`
hostname cache.

The file was called `.env.ec2` until the rename to `.ec2.env`. If you have an old
checkout that still dies with `.env.ec2 not found`, that is why; the contents did not
change, only the name.

## Who uses it

You do, by hand. `quantize.py` is the consumer of `tools/scripts/quantize-model.py`.
Nothing in the build depends on this directory.

## A typical session

```sh
python3 tools/ec2/start.py
python3 tools/ec2/ssh.py            # poke around, build
python3 tools/ec2/stop.py
```

For a model build, `quantize.py` does the start, sync and run for you.
