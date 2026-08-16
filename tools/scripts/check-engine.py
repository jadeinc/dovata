#!/usr/bin/env python3
"""Typecheck the Linux-only engine on a machine that is not Linux.

Everything under `src/*/exec.rs`, plus `src/dispatch/` and `src/moe/ffi.rs`, is behind
`#[cfg(target_os = "linux")]` because it links llama.cpp. On macOS the crate compiles
those modules out entirely, so `cargo test` is fast and broad and says *nothing* about
roughly half the engine. Every refactor of that half has historically been verified by
reading it and finding out on the GPU box.

It does not have to be that way. llama.cpp builds on macOS (Metal backend), and the
engine's Linux-only code is plain `llama-cpp-2` API — nothing in it is actually
Linux-specific. Given the dependency, it typechecks here.

So this script makes a scratch copy of the repo, removes the platform gate in the copy,
and runs `cargo check`. The working tree is never touched, and neither is the default
build: `cargo test` stays a sub-second command that compiles no C++.

What it catches: every renamed symbol, changed signature, wrong import, and bad `cfg`
in the half of the engine a Mac normally cannot see. That is the entire class of error
a large refactor produces.

What it does NOT catch: anything about CUDA (the copy builds the Metal/CPU backend and
the MoE driver's inert stub), and anything at run time. The evals in `tests/` on the GPU
box remain the real gate.

First run compiles llama.cpp and takes a few minutes; after that the scratch target dir
is warm and it is about a minute.

Usage:
  python3 tools/scripts/check-engine.py
  python3 tools/scripts/check-engine.py --keep     # leave the scratch copy for poking at
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import die, info, ok, warn  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GATE = '#[cfg(target_os = "linux")]'

# The dependency table that only exists on Linux, and the same entries unconditional.
TARGET_TABLE = '[target.\'cfg(target_os = "linux")\'.dependencies]'


def scratch_dir():
    """Where the copy lives. Outside the repo so cargo does not see two packages."""
    return Path(os.environ.get("TMPDIR", "/tmp")) / "dovata-engine-check"


def copy_repo(dst):
    """Copy the repo, skipping what a typecheck does not need.

    `target/` is the big one and is excluded; the scratch copy keeps its own. The
    llama.cpp submodule IS needed — the sys crate compiles it — so it is copied, and
    that is most of the time on a cold run.
    """
    def ignore(dirname, names):
        skip = set()
        for n in names:
            if n == "target" or n == ".git" or n == "dist" or n == "__pycache__":
                skip.add(n)
        return skip

    if dst.exists():
        # Keep the previous scratch target/ so repeat runs are fast; replace the source.
        keep = dst / "target"
        tmp_keep = None
        if keep.exists():
            tmp_keep = dst.parent / "dovata-engine-check-target"
            if tmp_keep.exists():
                shutil.rmtree(tmp_keep)
            shutil.move(str(keep), str(tmp_keep))
        shutil.rmtree(dst)
        shutil.copytree(ROOT, dst, ignore=ignore, symlinks=True)
        if tmp_keep is not None:
            shutil.move(str(tmp_keep), str(dst / "target"))
    else:
        shutil.copytree(ROOT, dst, ignore=ignore, symlinks=True)


def ungate(dst):
    """Remove the platform gate in the copy, and make the llama deps unconditional."""
    n_files = 0
    for p in (dst / "src").rglob("*.rs"):
        s = str(p)
        # Only our code: the vendored bindings gate on the target for real reasons.
        if "/llama/" in s or "/protocol/" in s:
            continue
        t = p.read_text()
        if GATE not in t:
            continue
        # `all()` is always true, so the module compiles. Kept as a cfg rather than
        # deleted so line numbers in any error match the real file.
        p.write_text(t.replace(GATE, "#[cfg(all())]"))
        n_files += 1

    manifest = dst / "Cargo.toml"
    t = manifest.read_text()
    if TARGET_TABLE not in t:
        die("could not find the Linux dependency table in Cargo.toml — has it moved?")
    manifest.write_text(t.replace(TARGET_TABLE + "\n", ""))
    return n_files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="leave the scratch copy in place")
    args = ap.parse_args()

    if sys.platform.startswith("linux"):
        ok("already on Linux — `cargo check` compiles the engine natively, no copy needed")
        return subprocess.run(["cargo", "check", "--lib"], cwd=str(ROOT)).returncode

    dst = scratch_dir()
    info(f"copying the repo to {dst} (target/ excluded; llama.cpp submodule included)")
    copy_repo(dst)

    n = ungate(dst)
    info(f"removed the platform gate in {n} files and unconditionalized the llama deps")
    warn("this builds llama.cpp with the Metal/CPU backend — it checks types, not CUDA")

    info("cargo check --lib")
    rc = subprocess.run(["cargo", "check", "--lib"], cwd=str(dst)).returncode

    if not args.keep:
        info(f"scratch copy left at {dst} (source refreshed each run; target/ reused)")

    if rc != 0:
        die("the engine does NOT typecheck — the errors above are real, and a macOS "
            "`cargo test` would not have shown them")
    ok("the Linux-only engine typechecks; the GPU box remains the run-time gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
