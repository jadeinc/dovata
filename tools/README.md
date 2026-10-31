# tools/ — the Python you run, not the code that ships

Everything here is a script a person invokes by hand. None of it is compiled, none of it
is linked, and nothing under `src/` imports it. It is stdlib-only Python that shells out
to the tools the work actually needs — `aws`, `ssh`, `git`, `curl`, `cmake`.

## Why it is a top-level directory

It used to live at `src/scripts/` and `src/ec2-tests/`, which put shell-outs and SSH
drivers in the same folder as the crates that get built into the shipping `.so`. That was
a leftover from flattening the tree: pulling the scripts out of an older nested layout and
dropping them beside the crates was the short move, and it stuck.

The cost was that `src/` stopped meaning anything. `tools/` restores the line — `src/` is
what ships, `tools/` is what you run against it.

## The two halves, and the file between them

- `_common.py` — the shared helpers: `info`/`ok`/`warn`/`die` logging, a `run` wrapper,
  checksum and disk-usage utilities, `.env` loading, and `ENV_ALIASES` /
  `normalize_env_aliases`, which reconcile the env file's variable names with the older
  `JADE_`-prefixed ones and with the AWS CLI's own spellings.
- [`scripts/`](scripts/README.md) — *local.* Set up a machine, fetch or build a model.
  Runs on your laptop or on a build host.
- [`ec2/`](ec2/README.md) — *remote.* Start the GPU box, sync to it, run things over SSH,
  stop it again. Every script here is a thin wrapper around `aws` and `ssh`.

`_common.py` sits at this level rather than inside `scripts/` on purpose. Both halves need
the alias map, and there is exactly one copy of it — two copies is precisely how the code
and the env file drifted apart the first time. When it lived in `scripts/`, the `ec2/`
half reached sideways into a sibling package to get it, which read as `ec2/` depending on
`scripts/` when really they are peers that share a floor. Now both halves insert the same
parent directory on `sys.path` and import the same module.

## The local/remote split is not the same as the laptop/box split

Two scripts cross the line, and it is worth knowing which:

- `ec2/quantize.py` is a *driver* for `scripts/quantize-model.py`. It starts the box,
  rsyncs the repo, and runs the local-half script over SSH — so the 18 GB upload to S3
  happens in-region instead of round-tripping through your laptop.
- `ec2/first-run.py` runs on both. Invoked from your laptop it uploads itself to the box
  and re-execs there with `--local`.

So `scripts/` is "code that does the work" and `ec2/` is "code that gets the work onto a
machine that can do it." A script in `scripts/` should never know an instance exists.

## Where the evals went

`run-tests.py`, the eval harness, is not here. It had moved to a repo-root `tests/`
alongside the `.jde` files it ran, and that whole suite went with the 3.0 runtime
removal: dovata no longer runs a model, so there is nothing for an end-to-end eval to
call. It lives on `main`.

## Running them

All paths are from the repo root.

```sh
python3 tools/scripts/setup-dev.py                    # project-local venv + aws CLI
python3 tools/scripts/download-model.py               # small CPU model for local dev
python3 tools/ec2/start.py                            # bring the GPU box up
python3 tools/ec2/ssh.py                              # shell on the box
python3 tools/ec2/stop.py                             # and put it away
```

Stopping the box is the one that matters. It bills by the hour, and the whole reason
these are one command each is that a two-command stop is a stop you skip.

## Who else runs this

Nothing, today. `.github/workflows/publish-providers.yml` used to run
`gen-profile-package.py --check` to fail a build whose Jade profile package had drifted
from its TOML, and `check-package-abi.py` to fail one that would publish packages the
Jade runtime refuses. Both scripts and the workflow went with the 3.0 runtime removal
and live on `main`. Everything here is invoked by a human.
