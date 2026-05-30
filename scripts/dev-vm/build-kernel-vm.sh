#!/usr/bin/env bash
# Build the JADE Linux kernel fork inside the dev VM.
#
# Runs scripts/dev-vm/build-kernel.sh inside the ARM64 Ubuntu dev VM via SSH,
# then deposits build/bzImage in the repo for use by QEMU.
#
# Safe to re-run — skips the build if build/bzImage already exists at the
# correct tag. Delete build/bzImage to force a full rebuild.
#
# Usage:
#   bash scripts/dev-vm/build-kernel-vm.sh
#   make kernel                        (routes here automatically on macOS)
#
# Environment:
#   KERNEL_FORK   git URL  (default: https://github.com/joericks1998/linux.git)
#   KERNEL_TAG    tag      (default: v6.12)
#   JOBS          parallel jobs (default: host logical CPUs)

set -euo pipefail

info() { echo "==> $*"; }
ok()   { echo "    OK: $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

KERNEL_FORK="${KERNEL_FORK:-https://github.com/joericks1998/linux.git}"
KERNEL_TAG="${KERNEL_TAG:-master}"
JOBS="${JOBS:-$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
OUT_BZIMAGE="$BUILD_DIR/bzImage"
STAMP="$BUILD_DIR/.kernel-tag"

mkdir -p "$BUILD_DIR"

# ── Skip if already built at this tag ────────────────────────────────────────
if [[ -f "$OUT_BZIMAGE" && -f "$STAMP" && "$(cat "$STAMP")" == "$KERNEL_TAG" ]]; then
    ok "build/bzImage already at $KERNEL_TAG — skipping kernel build"
    ok "  Delete build/bzImage or build/.kernel-tag to force a rebuild."
    exit 0
fi

info "Kernel build: $KERNEL_FORK @ $KERNEL_TAG  ($JOBS jobs)"
info "Output: $OUT_BZIMAGE"
echo ""

# ── Run inner build script inside the dev VM ─────────────────────────────────
bash "$SCRIPT_DIR/run.sh" \
    "KERNEL_FORK='$KERNEL_FORK' KERNEL_TAG='$KERNEL_TAG' JOBS=$JOBS bash /jade-os/scripts/dev-vm/build-kernel.sh"

# ── SCP bzImage from VM to host (avoids 9p permission issues) ─────────────────
info "Fetching bzImage from VM..."
SSH_KEY="$SCRIPT_DIR/../../build/dev-vm/id_ed25519"
SSH_PORT=12222
scp -i "$SSH_KEY" -P "$SSH_PORT" \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    jade@127.0.0.1:/home/jade/linux/arch/x86_64/boot/bzImage \
    "$OUT_BZIMAGE"

# ── Verify output ─────────────────────────────────────────────────────────────
[[ -f "$OUT_BZIMAGE" ]] || die "bzImage not found after scp — check dev VM logs"

echo "$KERNEL_TAG" > "$STAMP"

echo ""
ok "Kernel ready: build/bzImage  ($(du -sh "$OUT_BZIMAGE" | cut -f1))"
ok "Tag stamp:    build/.kernel-tag = $KERNEL_TAG"
