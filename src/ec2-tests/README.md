# ec2-tests — the GPU box harness

Python scripts that drive the remote EC2 GPU instance from your laptop: start it, sync
the repo up, build, run the evals, open a shell or a REPL, stop it again. Stdlib only;
they shell out to `aws` and `ssh`.

## Why it exists

The daemon's interesting half does not run on a Mac. The local llama.cpp engine is
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

- `_ec2.py` — the shared helpers: env loading, `aws` binary discovery (prefers the
  project `.venv`), hostname resolution with an `.ec2-host` cache, and `ssh` argv
  construction. Not run directly.
- `start.py` — starts the instance and blocks until SSH answers.
- `stop.py` — stops it.
- `ssh.py` — an interactive shell, or `ssh.py <cmd...>` to run one command and exit.
- `repl.py` — opens a PTY and execs `jade repl` on the box, checking first that `jade` is
  installed and the LLM socket is live. Gives you an interactive Jade session backed by
  real inference.
- `run-tests.py` — the eval harness. Starts `dovata`, waits for its `listening on…` line,
  runs every `.jde` in `src/ec2-jade-tests/`, and reports pass/fail. Runs *on* the box.
- `first-run.py` — one-time provisioning, idempotent. Installs CUDA, LLVM 18, Rust,
  musl-tools, and the NVIDIA driver, then downloads the 30B model to
  `/jade/inference/model.gguf`.
- `quantize.py` — the manual driver for `src/scripts/quantize-model.py`. Starts the box,
  rsyncs this repo, and runs the pipeline over SSH with CUDA on `PATH` and your AWS
  credentials forwarded, so the 18 GB upload to S3 happens in-region instead of round
  tripping through your laptop. Supports `--detached` (survives an SSH drop) and
  `--follow`.

## Configuration

`load_env` reads an optional `.env` then `.ec2.env` from the repo root, and the scripts
then require these variables:

| Variable | Used for |
|---|---|
| `JADE_INSTANCE_ID` | which instance to start/stop/describe |
| `JADE_REGION` | its AWS region |
| `JADE_SSH_KEY` | path to the `.pem`, relative to the repo root (see `keys/`) |
| `JADE_SSH_USER` | SSH login user |

Plus AWS credentials in the environment. All of it is gitignored — the env files, the
`keys/` directory, and the `.ec2-host` hostname cache.

The file was called `.env.ec2` until the rename to `.ec2.env`. If you have an old
checkout that still dies with `.env.ec2 not found`, that is why; the contents did not
change, only the name.

## Who uses it

You do, by hand. `run-tests.py` is the consumer of `src/ec2-jade-tests/`; `quantize.py`
is the consumer of `src/scripts/quantize-model.py`. Nothing in the build depends on this
directory.

## A typical session

```sh
python3 src/ec2-tests/start.py
python3 src/ec2-tests/ssh.py            # poke around, build
python3 src/ec2-tests/run-tests.py --build
python3 src/ec2-tests/repl.py           # interactive Jade against live inference
python3 src/ec2-tests/stop.py
```

`run-tests.py` derives the repo location from its own path, so it works the same whether
you run it locally, over SSH, or from a different `$HOME` on the box.
