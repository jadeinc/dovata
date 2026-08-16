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


# ── Environment variable names ─────────────────────────────────────────────
#
# Three naming conventions have to coexist here. The env file uses short,
# unprefixed names (INSTANCE_ID, AWS_REGION). The scripts used to read JADE_-
# prefixed ones. And the AWS CLI, which every one of these scripts shells out
# to, reads its own spellings and knows nothing about either of ours.
#
# So rather than pick a winner and break the other two, each name the code reads
# is listed here with the names accepted for it, in priority order. This is the
# single definition — a copy in the ec2-tests helper is exactly how the last
# mismatch happened.
ENV_ALIASES = {
    "INSTANCE_ID": ("INSTANCE_ID", "JADE_INSTANCE_ID"),
    "AWS_REGION": ("AWS_REGION", "JADE_REGION"),
    "SSH_KEY": ("SSH_KEY", "JADE_SSH_KEY"),
    "SSH_USER": ("SSH_USER", "JADE_SSH_USER"),
    # Not ours to rename: the AWS CLI looks for these exact names, so a file
    # written with AWS_ACCESS_KEY is copied across into the one it reads.
    "AWS_ACCESS_KEY_ID": ("AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY"),
    "AWS_DEFAULT_REGION": ("AWS_DEFAULT_REGION", "AWS_REGION", "JADE_REGION"),
}


def normalize_env_aliases():
    """Fill each canonical name from the first alias that is set.

    Never overwrites a name that already has a value, so anything exported in
    the shell beats anything read out of a file. Returns the (canonical, source)
    pairs it filled, for logging.
    """
    filled = []
    for canonical, names in ENV_ALIASES.items():
        if os.environ.get(canonical):
            continue
        for name in names:
            if os.environ.get(name):
                os.environ[canonical] = os.environ[name]
                filled.append((canonical, name))
                break
    return filled


def env_required(name, source=".ec2.env"):
    """A variable that must be set, or a message naming where to set it."""
    val = os.environ.get(name)
    if not val:
        die(f"{name} not set — add it to {source} "
            f"(see tools/ec2/README.md for the full list)")
    return val
