"""Shared helpers for the JADE OS scripts (stdlib only).

The scripts shell out to the same system tools the old bash versions used
(curl, git, cmake, aws, …); this module just carries the logging helpers and a
few small utilities so the flow lives in Python instead of bash.
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


def info(msg):
    print(f"==> {msg}")


def ok(msg):
    print(f"    OK: {msg}")


def warn(msg):
    print(f"    WARN: {msg}")


def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def have(tool):
    """True if `tool` is on PATH (like `command -v`)."""
    return shutil.which(tool) is not None


def run(cmd, check=True, **kwargs):
    """Run a command (list or str+shell); die() on non-zero unless check=False."""
    proc = subprocess.run(cmd, **kwargs)
    if check and proc.returncode != 0:
        shown = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
        die(f"command failed (exit {proc.returncode}): {shown}")
    return proc


def du_sh(path):
    """`du -sh <path>` field 1 (works for files and dirs); '?' on error."""
    try:
        out = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True)
        return out.stdout.split()[0] if out.stdout.split() else "?"
    except Exception:
        return "?"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dotenv(path, override=True):
    """Parse a `KEY=VALUE` / `export KEY=VALUE` file into os.environ.

    Mirrors `set -a; source <file>`: later assignments win (override=True).
    Strips a leading `export `, inline surrounding quotes, and skips comments.
    """
    path = Path(path)
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if override or key not in os.environ:
            os.environ[key] = val
