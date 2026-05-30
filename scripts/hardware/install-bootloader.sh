#!/usr/bin/env bash
# Install systemd-boot to the ESP and write the JADE OS boot entry.
#
# Required env vars:
#   KERNEL_IMAGE — path to bzImage on the build host
#
# Optional env vars:
#   DEVICE      — NVMe device (default: /dev/nvme0n1)
#   ESP_MOUNT   — where to mount the ESP (default: /mnt/jade-esp)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEVICE="${DEVICE:-/dev/nvme0n1}"
ESP_MOUNT="${ESP_MOUNT:-/mnt/jade-esp}"
KERNEL_IMAGE="${KERNEL_IMAGE:-$REPO_ROOT/build/bzImage}"

# ── Helpers ───────────────────────────────────────────────────────────────────

info() { echo "  [bootloader] $*"; }
ok()   { echo "  [bootloader] ✓ $*"; }
die()  { echo "  [bootloader] FATAL: $*" >&2; exit 1; }

# ── Preflight ─────────────────────────────────────────────────────────────────

[[ $EUID -eq 0 ]]    || die "must run as root"
[[ -b "${DEVICE}p1" ]] || die "${DEVICE}p1 not found — run partition.sh first"
[[ -f "$KERNEL_IMAGE" ]] || die "bzImage not found at $KERNEL_IMAGE — run: make kernel"

command -v bootctl >/dev/null 2>&1 || die "bootctl not found — install systemd-boot (systemd package)"

ROOT_UUID=$(blkid -s UUID -o value "${DEVICE}p2") || die "could not read UUID for ${DEVICE}p2"

# ── Mount ESP ─────────────────────────────────────────────────────────────────

info "Mounting ${DEVICE}p1 at $ESP_MOUNT ..."
mkdir -p "$ESP_MOUNT"
mount "${DEVICE}p1" "$ESP_MOUNT"

cleanup() { umount "$ESP_MOUNT" 2>/dev/null || true; }
trap cleanup EXIT

# ── Install systemd-boot ──────────────────────────────────────────────────────

info "Installing systemd-boot to ESP ..."
bootctl install --esp-path="$ESP_MOUNT"

# ── Boot loader config ────────────────────────────────────────────────────────

info "Writing loader/loader.conf ..."
mkdir -p "$ESP_MOUNT/loader/entries"
cat > "$ESP_MOUNT/loader/loader.conf" <<'EOF'
default jade.conf
timeout 3
console-mode max
editor no
EOF

# ── Copy kernel ───────────────────────────────────────────────────────────────

info "Copying bzImage -> ESP/EFI/jade/vmlinuz ..."
mkdir -p "$ESP_MOUNT/EFI/jade"
install -m 0644 "$KERNEL_IMAGE" "$ESP_MOUNT/EFI/jade/vmlinuz"

# ── Boot entry ────────────────────────────────────────────────────────────────

info "Writing loader/entries/jade.conf ..."
cat > "$ESP_MOUNT/loader/entries/jade.conf" <<EOF
title   JADE OS
linux   /EFI/jade/vmlinuz
options root=UUID=$ROOT_UUID rw init=/sbin/init console=tty0 console=ttyS0,115200
EOF

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
ok "systemd-boot installed:"
echo ""
echo "    ESP/EFI/jade/vmlinuz"
echo "    ESP/loader/entries/jade.conf"
echo "      root=UUID=$ROOT_UUID"
echo "      init=/sbin/init"
echo ""
ok "Unmounting ${DEVICE}p1"
trap - EXIT
umount "$ESP_MOUNT"
