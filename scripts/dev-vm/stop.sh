#!/usr/bin/env bash
# dev-vm/stop.sh — gracefully shut down the dev VM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VM_DIR="$REPO_ROOT/build/dev-vm"
PID_FILE="$VM_DIR/qemu.pid"
SSH_KEY="$VM_DIR/id_ed25519"
SSH_PORT=12222
SSH_OPTS="-i $SSH_KEY -p $SSH_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=3 -o LogLevel=ERROR"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file found — VM may not be running."
    exit 0
fi

PID="$(cat "$PID_FILE")"

if ! kill -0 "$PID" 2>/dev/null; then
    echo "VM process $PID not found — already stopped."
    rm -f "$PID_FILE"
    exit 0
fi

echo "→ Sending poweroff to VM (PID $PID)..."
ssh $SSH_OPTS jade@127.0.0.1 'sudo poweroff' 2>/dev/null || true

# Wait up to 15s for clean shutdown, then force kill
for i in $(seq 1 15); do
    if ! kill -0 "$PID" 2>/dev/null; then
        break
    fi
    sleep 1
done

if kill -0 "$PID" 2>/dev/null; then
    echo "→ Force-killing QEMU..."
    kill "$PID"
fi

rm -f "$PID_FILE"
echo "VM stopped."
