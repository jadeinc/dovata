#!/usr/bin/env bash
# Partition and format the JADE OS target NVMe drive.
#
# Creates three partitions:
#   p1  1 GiB   FAT32   ESP (systemd-boot + bzImage)
#   p2  50 GiB  ext4    root (/)
#   p3  rest    ext4    data (/root/.jade/memory)
#
# Usage: DEVICE=/dev/nvme0n1 bash partition.sh
# WARNING: destroys all data on $DEVICE.

set -euo pipefail

DEVICE="${DEVICE:-/dev/nvme0n1}"

# ── Helpers ───────────────────────────────────────────────────────────────────

info() { echo "  [partition] $*"; }
ok()   { echo "  [partition] ✓ $*"; }
die()  { echo "  [partition] FATAL: $*" >&2; exit 1; }

# ── Preflight ─────────────────────────────────────────────────────────────────

command -v sgdisk  >/dev/null 2>&1 || die "sgdisk not found — install gdisk"
command -v mkfs.fat >/dev/null 2>&1 || die "mkfs.fat not found — install dosfstools"
command -v mkfs.ext4 >/dev/null 2>&1 || die "mkfs.ext4 not found — install e2fsprogs"

[[ -b "$DEVICE" ]] || die "$DEVICE is not a block device"
[[ $EUID -eq 0 ]]  || die "must run as root"

info "Target device: $DEVICE"
info "$(lsblk -o NAME,SIZE,MODEL "$DEVICE" 2>/dev/null | head -2 || true)"
echo ""
read -r -p "  This will DESTROY all data on $DEVICE. Continue? [y/N] " confirm
[[ "$confirm" =~ ^[yY]$ ]] || { info "Aborted."; exit 0; }

# ── Partition ─────────────────────────────────────────────────────────────────

info "Wiping existing partition table ..."
sgdisk --zap-all "$DEVICE"

info "Creating partitions ..."
sgdisk -n 1:0:+1G   -t 1:ef00 -c 1:"ESP"         "$DEVICE"
sgdisk -n 2:0:+50G  -t 2:8300 -c 2:"jade-root"   "$DEVICE"
sgdisk -n 3:0:0     -t 3:8300 -c 3:"jade-memory"  "$DEVICE"

info "Notifying kernel of new partition table ..."
partprobe "$DEVICE" || sleep 2

# ── Format ────────────────────────────────────────────────────────────────────

info "Formatting ${DEVICE}p1 as FAT32 (ESP) ..."
mkfs.fat -F 32 -n "ESP" "${DEVICE}p1"

info "Formatting ${DEVICE}p2 as ext4 (root) ..."
mkfs.ext4 -L "jade-root" "${DEVICE}p2"

info "Formatting ${DEVICE}p3 as ext4 (memory store) ..."
mkfs.ext4 -L "jade-memory" "${DEVICE}p3"

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
ok "Partitioning complete:"
lsblk -o NAME,SIZE,FSTYPE,LABEL "$DEVICE"
