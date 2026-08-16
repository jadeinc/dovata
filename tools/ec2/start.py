#!/usr/bin/env python3
"""JADE OS — Start EC2 inference instance and wait for SSH.

Usage:
  python3 tools/ec2/start.py
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ec2 import (info, ok, die, load_env, aws_bin, repo_root,
                  ssh_key, ssh_user, env_required)


def main():
    load_env()
    aws = aws_bin()
    root = repo_root()
    instance = env_required("INSTANCE_ID")
    region = env_required("AWS_REGION")

    info(f"Starting {instance} in {region}...")
    subprocess.run([aws, "ec2", "start-instances", "--instance-ids", instance,
                    "--region", region, "--output", "text"],
                   check=True, stdout=subprocess.DEVNULL)

    info("Waiting for instance running...")
    subprocess.run([aws, "ec2", "wait", "instance-running",
                    "--instance-ids", instance, "--region", region], check=True)

    public_dns = subprocess.run(
        [aws, "ec2", "describe-instances", "--instance-ids", instance,
         "--region", region,
         "--query", "Reservations[0].Instances[0].PublicDnsName",
         "--output", "text"],
        capture_output=True, text=True,
    ).stdout.strip()
    ok(f"Instance running: {public_dns}")

    info("Waiting for SSH to be ready...")
    key = ssh_key()
    user = ssh_user()
    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                "-o", "BatchMode=yes"]
    for i in range(1, 37):
        probe = subprocess.run(
            ["ssh", *ssh_opts, "-i", key, f"{user}@{public_dns}", "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            ok("SSH ready")
            print()
            print(f"    host: {public_dns}")
            print(f"    ssh:  ssh -i {key} {user}@{public_dns}")
            print()
            (root / ".ec2-host").write_text(public_dns + "\n")
            return 0
        print(f"    waiting... ({i}/36)")
        time.sleep(5)

    die("SSH not ready after 3 minutes — check security group inbound rule for port 22")


if __name__ == "__main__":
    sys.exit(main())
