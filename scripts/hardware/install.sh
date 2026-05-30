#!/usr/bin/env bash
# Full JADE OS hardware install — orchestrates partition, rootfs, and bootloader.
#
# Required env vars:
#   MODEL_PATH    — path to GGUF model on this host
#   KERNEL_IMAGE  — path to bzImage on this host
#
# Optional env vars:
#   DEVICE        — target NVMe device (default: /dev/nvme0n1)
#   ROOT_MOUNT    — root partition mount point (default: /mnt/jade-root)
#   ESP_MOUNT     — ESP mount point (default: /mnt/jade-esp)
#
# Example:
#   MODEL_PATH=/path/to/model.gguf \
#   KERNEL_IMAGE=build/bzImage \
#   DEVICE=/dev/nvme0n1 \
#   bash scripts/hardware/install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DEVICE="${DEVICE:-/dev/nvme0n1}"
export ROOT_MOUNT="${ROOT_MOUNT:-/mnt/jade-root}"
export ESP_MOUNT="${ESP_MOUNT:-/mnt/jade-esp}"
export MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
export KERNEL_IMAGE="${KERNEL_IMAGE:?KERNEL_IMAGE is required}"

# ── Helpers ───────────────────────────────────────────────────────────────────

info() { echo ""; echo "══ $* ══"; }
ok()   { echo "  [install] ✓ $*"; }
die()  { echo "  [install] FATAL: $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "must run as root"

# ── Steps ─────────────────────────────────────────────────────────────────────

info "Step 1/3: Partition $DEVICE"
bash "$SCRIPT_DIR/partition.sh"

info "Step 2/3: Build root filesystem"
bash "$SCRIPT_DIR/build-rootfs.sh"

info "Step 3/3: Install bootloader"
bash "$SCRIPT_DIR/install-bootloader.sh"

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════"
ok "JADE OS installed to $DEVICE"
echo ""
echo "  Boot sequence:"
echo "    UEFI → systemd-boot → bzImage → jade-init (PID 1)"
echo "         → jade_core.ko → jade-tree → jade repl"
echo ""
echo "  Remove install media and reboot."
echo "══════════════════════════════════════════════════════"
