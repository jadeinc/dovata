#!/usr/bin/env bash
# dev-vm/start.sh — start the JADE OS Linux dev VM in the background
#
# Starts the ARM64 Ubuntu VM with:
#   - HVF hardware acceleration
#   - 9p virtio share: repo root → /jade-os inside the VM
#   - SSH forwarded on localhost:12222
#
# Usage: bash scripts/dev-vm/start.sh [--wait]
#   --wait   block until SSH is reachable before returning

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VM_DIR="$REPO_ROOT/build/dev-vm"
VM_DISK="$VM_DIR/vm.qcow2"
SSH_KEY="$VM_DIR/id_ed25519"
PID_FILE="$VM_DIR/qemu.pid"
SERIAL_LOG="$VM_DIR/serial.log"
SSH_PORT=12222
EFI_FW="/opt/homebrew/share/qemu/edk2-aarch64-code.fd"

WAIT=false
for arg in "$@"; do
    [[ "$arg" == "--wait" ]] && WAIT=true
done

if [[ ! -f "$VM_DISK" ]]; then
    echo "ERROR: VM disk not found. Run: bash scripts/dev-vm/setup.sh"
    exit 1
fi

# Check if already running
if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if kill -0 "$PID" 2>/dev/null; then
        echo "VM already running (PID $PID, SSH on localhost:$SSH_PORT)"
        exit 0
    fi
    rm -f "$PID_FILE"
fi

echo "→ Starting dev VM (SSH will be on localhost:$SSH_PORT)..."

qemu-system-aarch64 \
    -machine virt,accel=hvf \
    -cpu host \
    -m 6G \
    -smp "$(sysctl -n hw.logicalcpu)" \
    -bios "$EFI_FW" \
    -drive if=virtio,file="$VM_DISK",format=qcow2 \
    -netdev user,id=net0,hostfwd="tcp:127.0.0.1:${SSH_PORT}-:22" \
    -device virtio-net-pci,netdev=net0 \
    -virtfs "local,path=$REPO_ROOT,mount_tag=jadeos,security_model=mapped-xattr,id=jadeos" \
    -nographic \
    -serial "file:$SERIAL_LOG" \
    -monitor none \
    &
QEMU_PID=$!
echo $QEMU_PID > "$PID_FILE"

echo "   PID $QEMU_PID  |  log: $SERIAL_LOG"

if $WAIT; then
    echo -n "   Waiting for SSH"
    SSH_OPTS="-i $SSH_KEY -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=3 -o LogLevel=ERROR"
    for i in $(seq 1 60); do
        if ssh $SSH_OPTS jade@127.0.0.1 true 2>/dev/null; then
            echo " ready."
            break
        fi
        echo -n "."
        sleep 3
        if ! kill -0 $QEMU_PID 2>/dev/null; then
            echo ""
            echo "ERROR: QEMU exited. Check $SERIAL_LOG"
            rm -f "$PID_FILE"
            exit 1
        fi
    done

    # Mount 9p share if not already mounted
    ssh $SSH_OPTS jade@127.0.0.1 '
        if ! mountpoint -q /jade-os 2>/dev/null; then
            sudo mkdir -p /jade-os
            sudo mount -t 9p -o trans=virtio,version=9p2000.L,msize=1048576 jadeos /jade-os
        fi
    ' 2>/dev/null || true
fi

echo ""
echo "Connect:  bash scripts/dev-vm/ssh.sh"
echo "Run cmd:  bash scripts/dev-vm/run.sh 'cargo test -p jade-protocol'"
echo "Stop:     bash scripts/dev-vm/stop.sh"
