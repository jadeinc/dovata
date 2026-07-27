#!/usr/bin/env python3
"""JADE OS — Open an interactive SSH session on the EC2 instance.

Usage:
  python3 src/ec2-tests/ssh.py           # interactive shell
  python3 src/ec2-tests/ssh.py <cmd...>  # run a single command and exit
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ec2 import load_env, aws_bin, resolve_host, ssh_cmd


def main(argv):
    load_env()
    aws = aws_bin()
    host = resolve_host(aws)
    remote = " ".join(argv[1:]) if len(argv) > 1 else None
    cmd = ssh_cmd(host, remote=remote)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main(sys.argv)
