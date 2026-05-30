#!/usr/bin/env bash
# dev-vm/ssh.sh — open an interactive SSH session into the dev VM
#
# Usage: bash scripts/dev-vm/ssh.sh [command]
#   No args    → interactive shell
#   With args  → run the command and return exit code

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/build/dev-vm"
SSH_KEY="$VM_DIR/id_ed25519"
SSH_PORT=12222

if [[ ! -f "$SSH_KEY" ]]; then
    echo "ERROR: SSH key not found. Run: bash scripts/dev-vm/setup.sh"
    exit 1
fi

exec ssh \
    -i "$SSH_KEY" \
    -p "$SSH_PORT" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR \
    jade@127.0.0.1 "$@"
