"""Shared helpers for the EC2 test harness (stdlib only; shells out to aws/ssh).

The old bash scripts each re-derived the repo root, sourced .env/.ec2.env,
located the aws CLI, and resolved the instance hostname. That boilerplate lives
here now so the individual scripts stay small.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

# The variable-name alias map lives in tools/_common.py and is imported rather
# than copied: two copies of it is precisely how the code and the env file
# drifted apart in the first place. It sits one level up, at the root of tools/,
# because both halves of the tooling need it and neither one owns it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import env_required, normalize_env_aliases  # noqa: E402


def info(msg):
    print(f"==> {msg}")


def ok(msg):
    print(f"    OK: {msg}")


def warn(msg):
    print(f"    WARN: {msg}")


def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def repo_root():
    # tools/ec2/_ec2.py -> repo root is two levels up.
    return Path(__file__).resolve().parents[2]


def _load_dotenv(path):
    """Mirror `set -a; source <file>` — export KEY=VALUE, later wins."""
    for raw in Path(path).read_text().splitlines():
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
        os.environ[key] = val


def load_env(require_ec2=True):
    """Load .env (optional) then .ec2.env into os.environ."""
    root = repo_root()
    env_ec2 = root / ".ec2.env"
    if require_ec2 and not env_ec2.is_file():
        die(".ec2.env not found — see tools/ec2/README.md for the variables it needs")
    if (root / ".env").is_file():
        _load_dotenv(root / ".env")
    if env_ec2.is_file():
        _load_dotenv(env_ec2)
    # Reconcile the file's names with the ones the code and the AWS CLI read.
    normalize_env_aliases()


def aws_bin():
    """Prefer the project-local aws from .venv, fall back to a global one."""
    venv_aws = repo_root() / ".venv" / "bin" / "aws"
    if venv_aws.is_file():
        return str(venv_aws)
    found = shutil.which("aws")
    if found:
        return found
    die("aws CLI not found — run: python3 tools/scripts/setup-dev.py")


def resolve_host(aws=None):
    """Return the instance hostname: cached .ec2-host, else describe-instances."""
    root = repo_root()
    host_file = root / ".ec2-host"
    if host_file.is_file():
        host = host_file.read_text().strip()
        if host:
            return host
    aws = aws or aws_bin()
    out = subprocess.run(
        [aws, "ec2", "describe-instances",
         "--instance-ids", env_required("INSTANCE_ID"),
         "--region", env_required("AWS_REGION"),
         "--query", "Reservations[0].Instances[0].PublicDnsName",
         "--output", "text"],
        capture_output=True, text=True,
    )
    host = out.stdout.strip()
    if not host or host == "None":
        die("could not resolve hostname — run start.py first")
    return host


def ssh_key():
    return str(repo_root() / env_required("SSH_KEY"))


def ssh_user():
    return env_required("SSH_USER")


def ssh_cmd(host, remote=None, tty=False, extra_ssh_opts=None):
    """Build an `ssh` argv list matching the old scripts' options."""
    cmd = ["ssh"]
    if tty:
        cmd.append("-t")
    cmd += ["-o", "StrictHostKeyChecking=no"]
    if extra_ssh_opts:
        cmd += list(extra_ssh_opts)
    cmd += ["-i", ssh_key(), f"{ssh_user()}@{host}"]
    if remote is not None:
        cmd.append(remote)
    return cmd
