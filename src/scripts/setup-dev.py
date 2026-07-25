#!/usr/bin/env python3
"""JADE OS — Local dev environment setup.

Creates a project-local Python venv and installs the AWS CLI into it.
Nothing is installed or written globally.

AWS credentials are read from environment variables — add to ~/.zshrc:
  export AWS_ACCESS_KEY_ID=...
  export AWS_SECRET_ACCESS_KEY=...

Usage:
  python3 src/scripts/setup-dev.py
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import info, ok, die, load_dotenv

ROOT = Path(__file__).resolve().parents[2]
VENV = ROOT / ".venv"


def venv_bin(name):
    return VENV / "bin" / name


def aws_version(aws):
    out = subprocess.run([str(aws), "--version"], capture_output=True, text=True)
    return (out.stdout or out.stderr).strip()


def main():
    # ── Create venv ────────────────────────────────────────────────────────
    if not (VENV / "bin" / "activate").is_file():
        info("Creating .venv...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
        ok(".venv created")
    else:
        ok(".venv already exists")

    aws = venv_bin("aws")

    # ── Install AWS CLI ────────────────────────────────────────────────────
    if not aws.is_file():
        info("Installing AWS CLI into .venv...")
        subprocess.run([str(venv_bin("pip")), "install", "--quiet", "awscli"], check=True)
        ok(f"AWS CLI installed: {aws_version(aws)}")
    else:
        ok(f"AWS CLI already installed: {aws_version(aws)}")

    # ── Load credentials and config ────────────────────────────────────────
    env = ROOT / ".env"
    env_ec2 = ROOT / ".env.ec2"
    if env.is_file():
        load_dotenv(env)
    if not env_ec2.is_file():
        die(".env.ec2 not found")
    load_dotenv(env_ec2)

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        die("AWS_ACCESS_KEY_ID not set — add to .env")
    if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        die("AWS_SECRET_ACCESS_KEY not set — add to .env")

    region = os.environ.get("JADE_REGION", "")
    info("Verifying AWS credentials...")
    probe = subprocess.run(
        [str(aws), "sts", "get-caller-identity", "--region", region, "--output", "text"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        die("Credentials invalid — check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
    account = subprocess.run(
        [str(aws), "sts", "get-caller-identity", "--region", region,
         "--query", "Account", "--output", "text"],
        capture_output=True, text=True,
    ).stdout.strip()
    ok(f"Credentials valid (account: {account})")

    print()
    ok("Dev environment ready")
    info("Run: python3 src/ec2-tests/start.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
