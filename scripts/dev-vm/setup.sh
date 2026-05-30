#!/usr/bin/env bash
# dev-vm/setup.sh — one-time setup for the JADE OS Linux dev VM
#
# Downloads Ubuntu 24.04 ARM64 cloud image, creates a QEMU VM, and provisions
# it with Rust + build toolchain via cloud-init. Run once; subsequent boots
# use start.sh.
#
# Usage: bash scripts/dev-vm/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VM_DIR="$REPO_ROOT/build/dev-vm"
BASE_IMG="$VM_DIR/base.img"
VM_DISK="$VM_DIR/vm.qcow2"
CIDATA_ISO="$VM_DIR/cidata.iso"
SSH_KEY="$VM_DIR/id_ed25519"
QEMU_LOG="$VM_DIR/setup.log"
SSH_PORT=12222

# Ubuntu 24.04 LTS ARM64 cloud image
CLOUD_IMG_URL="https://cloud-images.ubuntu.com/releases/24.04/release/ubuntu-24.04-server-cloudimg-arm64.img"

EFI_FW="/opt/homebrew/share/qemu/edk2-aarch64-code.fd"

# ── sanity checks ────────────────────────────────────────────────────────────

if [[ ! -f "$EFI_FW" ]]; then
    echo "ERROR: EFI firmware not found at $EFI_FW"
    echo "  Install QEMU via Homebrew: brew install qemu"
    exit 1
fi

if ! command -v qemu-system-aarch64 &>/dev/null; then
    echo "ERROR: qemu-system-aarch64 not found. Install: brew install qemu"
    exit 1
fi

if [[ -f "$VM_DISK" ]]; then
    echo "VM disk already exists at $VM_DISK"
    echo "Resetting overlay for a fresh first boot (base image and SSH key kept)..."
    rm -f "$VM_DISK"
fi

mkdir -p "$VM_DIR"

# ── SSH key ──────────────────────────────────────────────────────────────────

if [[ ! -f "$SSH_KEY" ]]; then
    echo "→ Generating VM SSH key..."
    ssh-keygen -t ed25519 -f "$SSH_KEY" -N "" -C "jade-dev-vm" -q
fi
SSH_PUBKEY="$(cat "$SSH_KEY.pub")"

# ── download cloud image ─────────────────────────────────────────────────────

if [[ ! -f "$BASE_IMG" ]]; then
    echo "→ Downloading Ubuntu 24.04 ARM64 cloud image (~600 MB)..."
    curl -L --progress-bar -o "$BASE_IMG.tmp" "$CLOUD_IMG_URL"
    mv "$BASE_IMG.tmp" "$BASE_IMG"
fi

# ── create overlay disk (20 GB) ──────────────────────────────────────────────

echo "→ Creating VM overlay disk (20 GB over base image)..."
qemu-img create -f qcow2 -b "$BASE_IMG" -F qcow2 "$VM_DISK" 20G -q

# ── cloud-init ISO ───────────────────────────────────────────────────────────

echo "→ Building cloud-init ISO..."
CIDATA_DIR="$VM_DIR/cidata"
mkdir -p "$CIDATA_DIR"

cat > "$CIDATA_DIR/meta-data" << 'EOF'
instance-id: jade-dev-vm
local-hostname: jade-dev
EOF

# Write user-data (expand SSH_PUBKEY here, not inside heredoc single-quote)
cat > "$CIDATA_DIR/user-data" << USERDATA
#cloud-config

users:
  - name: jade
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    lock_passwd: false
    ssh_authorized_keys:
      - ${SSH_PUBKEY}

packages:
  - build-essential
  - cmake
  - clang
  - libssl-dev
  - pkg-config
  - musl-tools
  - git
  - curl

# Install Rust after packages are ready.
# IMPORTANT: each runcmd item that contains a colon must be quoted to avoid
# YAML parsing it as a key:value pair.
runcmd:
  - ["su", "-", "jade", "-c", "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path 2>&1 | tee /tmp/rustup.log"]
  - ["su", "-", "jade", "-c", "/home/jade/.cargo/bin/rustup target add x86_64-unknown-linux-musl"]
  - ["su", "-", "jade", "-c", "/home/jade/.cargo/bin/rustup component add rust-src"]
  - [touch, /var/lib/cloud/jade-setup-done]
  - [sh, -c, "echo jade-dev-vm setup complete"]

final_message: "jade-dev-vm ready — first boot took \$UPTIME seconds"
USERDATA

hdiutil makehybrid -ov -o "$CIDATA_ISO" -iso -joliet \
    -default-volume-name cidata "$CIDATA_DIR" -quiet

# ── first boot ───────────────────────────────────────────────────────────────

echo ""
echo "→ Starting first-boot VM (installing packages + Rust — ~5 min)..."
echo "   Log: $QEMU_LOG"
echo ""

# Start QEMU in background.
# The cloud-init seed is attached as a SCSI CD-ROM so Linux sees it as
# /dev/sr0 with label CIDATA — the format ds-identify expects.
qemu-system-aarch64 \
    -machine virt,accel=hvf \
    -cpu host \
    -m 4G \
    -smp 4 \
    -bios "$EFI_FW" \
    -drive "if=virtio,file=$VM_DISK,format=qcow2" \
    -device virtio-scsi-pci,id=scsi0 \
    -drive "file=$CIDATA_ISO,if=none,id=cidata,media=cdrom,readonly=on" \
    -device "scsi-cd,bus=scsi0.0,drive=cidata" \
    -netdev "user,id=net0,hostfwd=tcp:127.0.0.1:${SSH_PORT}-:22" \
    -device virtio-net-pci,netdev=net0 \
    -nographic \
    -serial "file:$QEMU_LOG" \
    2>>"$QEMU_LOG" &
QEMU_PID=$!

# Wait for SSH to come up (cloud-init sets up user early in boot)
echo -n "   Waiting for SSH"
SSH_OPTS="-i $SSH_KEY -p $SSH_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=3 -o LogLevel=ERROR"
SSH_UP=false
for i in $(seq 1 120); do
    if ssh $SSH_OPTS jade@127.0.0.1 true 2>/dev/null; then
        echo " up!"
        SSH_UP=true
        break
    fi
    echo -n "."
    sleep 5
    if ! kill -0 "$QEMU_PID" 2>/dev/null; then
        echo ""
        echo "ERROR: QEMU exited unexpectedly. Check $QEMU_LOG"
        exit 1
    fi
done

if ! $SSH_UP; then
    echo ""
    echo "ERROR: SSH never came up. cloud-init may have failed to read the seed."
    echo "  Check $QEMU_LOG for boot errors."
    echo "  Then run: bash $0   (will reset overlay and retry)"
    kill "$QEMU_PID" 2>/dev/null || true
    exit 1
fi

# Wait for cloud-init setup to complete (packages + Rust install)
echo -n "   Waiting for Rust install"
SETUP_DONE=false
for i in $(seq 1 60); do
    if ssh $SSH_OPTS jade@127.0.0.1 'test -f /var/lib/cloud/jade-setup-done' 2>/dev/null; then
        echo " done!"
        SETUP_DONE=true
        break
    fi
    echo -n "."
    sleep 10
done

if ! $SETUP_DONE; then
    echo ""
    echo "WARNING: Rust install timed out. VM may still be installing."
    echo "  Connect manually: bash scripts/dev-vm/ssh.sh"
    echo "  Check install log: sudo cat /var/log/cloud-init-output.log"
fi

# Graceful shutdown
echo "→ Shutting down VM..."
ssh $SSH_OPTS jade@127.0.0.1 'sudo poweroff' 2>/dev/null || true
wait $QEMU_PID 2>/dev/null || true

echo ""
echo "Setup complete. Run 'make vm-start' or 'bash scripts/dev-vm/start.sh' to start the dev VM."
