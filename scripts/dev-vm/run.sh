#!/usr/bin/env bash
# dev-vm/run.sh — run a command in the dev VM, mounting /jade-os if needed
#
# Usage: bash scripts/dev-vm/run.sh '<shell command>'
#   Exit code mirrors the remote command.
#
# The repo is available at /jade-os inside the VM (9p virtio share).
# Rust is at /home/jade/.cargo/bin/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VM_DIR="$REPO_ROOT/build/dev-vm"
PID_FILE="$VM_DIR/qemu.pid"
SSH_KEY="$VM_DIR/id_ed25519"
SSH_PORT=12222

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 '<command>'"
    exit 1
fi

# Ensure VM is running
if [[ ! -f "$PID_FILE" ]] || ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "→ VM not running — starting..."
    bash "$SCRIPT_DIR/start.sh" --wait
fi

SSH_OPTS="-i $SSH_KEY -p $SSH_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Ensure 9p mount is active
ssh $SSH_OPTS jade@127.0.0.1 '
    if ! mountpoint -q /jade-os 2>/dev/null; then
        sudo mkdir -p /jade-os
        sudo mount -t 9p -o trans=virtio,version=9p2000.L,msize=1048576 jadeos /jade-os
    fi
' 2>/dev/null

# Run the requested command with Rust in PATH.
# CARGO_TARGET_DIR points to the VM's local filesystem — the 9p share's
# target/ directory has macOS-side permissions that deny writes inside the VM.
ssh $SSH_OPTS jade@127.0.0.1 "
    export PATH=/home/jade/.cargo/bin:\$PATH
    export CARGO_TARGET_DIR=/home/jade/.jade-target
    $1
"
