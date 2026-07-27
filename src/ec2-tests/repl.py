#!/usr/bin/env python3
"""JADE OS — Interactive jade repl on EC2 with live LLM inference.

jade-tree is started by the deploy step and stays running persistently.
This script just opens an SSH PTY and execs into `jade repl`.

Usage: python3 src/ec2-tests/repl.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ec2 import load_env, aws_bin, resolve_host, ssh_cmd

# Remote (bash) snippet run under the SSH PTY — checks jade + the llm socket,
# then execs the repl.
REMOTE = r'''
set -e

JADE="/usr/bin/jade"
SOCK="$HOME/.jade/llm.sock"

[[ -x "$JADE" ]] || { echo 'ERROR: jade not installed — run the deploy step'; exit 1; }

if [[ ! -S "$SOCK" ]]; then
    echo 'ERROR: jade-tree not running — run the deploy step'
    exit 1
fi

echo ''
echo 'JADE REPL  (Qwen3-Coder-30B-A3B-Instruct-Q4_K_M  •  live inference via Unix socket)'
echo 'Try:  prompt p = "What is Rust?"   then: ?p'
echo 'Exit: Ctrl-D'
echo ''
exec "$JADE" repl
'''


def main():
    load_env()
    aws = aws_bin()
    host = resolve_host(aws)
    print(f"==> Connecting to {host} ...")
    cmd = ssh_cmd(host, remote=REMOTE, tty=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
