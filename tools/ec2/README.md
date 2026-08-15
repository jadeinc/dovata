# tools/ec2 — getting work onto the GPU box

Python scripts that drive the remote EC2 GPU instance from your laptop: start it, sync
the repo up, build, open a shell or a REPL, stop it again. Stdlib only; they shell out to
`aws` and `ssh`.

The directory used to be called `ec2-tests`, which was misleading twice over — it holds
no tests, and the harness that *does* run the tests now lives with them in the repo-root
[`tests/`](../../tests/README.md). What is left here is plumbing: everything whose job is
to get code onto a machine that can run it.

## Why it exists

The engine's interesting half does not run on a Mac. The llama.cpp backend is
Linux-only, the model is 18 GB, and the CUDA paths need a real NVIDIA card — so the
actual verification of anything in `src/dovata/src/engine/` happens on a `g5.xlarge`
(A10G), not in a local test run. These scripts are the shortest path between an edit here
and a result there.

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
- `repl.py` — opens a PTY and execs `jade repl` on the box, checking first that `jade` is
  installed and the LLM socket is live. Gives you an interactive Jade session backed by
  real inference. See the note under Configuration: the socket check is stale.
- `first-run.py` — one-time provisioning, idempotent. Installs CUDA, LLVM 18, Rust,
  musl-tools, and the NVIDIA driver, then downloads the 30B model to
  `/jade/inference/model.gguf`. Run from your laptop it uploads itself to the box and
  re-execs there with `--local`.
- `quantize.py` — the manual driver for `tools/scripts/quantize-model.py`. Starts the box,
  rsyncs this repo, and runs the pipeline over SSH with CUDA on `PATH` and your AWS
  credentials forwarded, so the 18 GB upload to S3 happens in-region instead of round
  tripping through your laptop. Supports `--detached` (survives an SSH drop) and
  `--follow`.

## Where the eval harness went

`run-tests.py` used to live here. It now sits in the repo-root
[`tests/`](../../tests/README.md) next to the `.jde` files it runs, because it is the
runner for that suite and not GPU-box plumbing: it takes no AWS credentials, resolves no
instance, and derives everything from its own path. It still runs *on* the box — you just
reach it through `ssh.py` like anything else.

The write-up of how it stages a provider `.so` now that there is no daemon to start moved
with it.

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

### `repl.py` still assumes a daemon

It checks that `~/.jade/llm.sock` is live before it execs `jade repl`. Nothing binds that
socket any more — the engine is a library a Jade program `dlopen`s in-process — so the
check can only fail. `run-tests.py` was ported off the same assumption; `repl.py` was not,
and it needs the same treatment before it works again.

## Who uses it

You do, by hand. `quantize.py` is the consumer of `tools/scripts/quantize-model.py`.
Nothing in the build depends on this directory.

## A typical session

```sh
python3 tools/ec2/start.py
python3 tools/ec2/ssh.py            # poke around, build
python3 tools/ec2/ssh.py 'cd ~/dovata && python3 tests/run-tests.py --build'
python3 tools/ec2/stop.py
```

`run-tests.py` derives the repo location from its own path, so it works the same whether
you run it locally, over SSH, or from a different `$HOME` on the box.
