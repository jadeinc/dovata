#!/usr/bin/env python3
"""JADE OS — Stop EC2 inference instance.

Usage:
  python3 src/ec2-tests/stop.py
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ec2 import info, ok, load_env, aws_bin, repo_root, env_required


def main():
    load_env()
    aws = aws_bin()
    instance = env_required("INSTANCE_ID")
    region = env_required("AWS_REGION")

    info(f"Stopping {instance}...")
    subprocess.run([aws, "ec2", "stop-instances", "--instance-ids", instance,
                    "--region", region, "--output", "text"],
                   check=True, stdout=subprocess.DEVNULL)

    host_file = repo_root() / ".ec2-host"
    if host_file.exists():
        host_file.unlink()
    ok("Stop initiated — instance will be stopped in ~30s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
