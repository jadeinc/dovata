#!/usr/bin/env bash
# JADE OS — Fetch and build Linux kernel fork
#
# Usage:
#   make kernel
#   make kernel KERNEL_TAG=v6.12.3   # pin to a specific release
set -euo pipefail

info() { echo "==> $*"; }
ok()   { echo "    OK: $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

KERNEL_FORK="${KERNEL_FORK:-}"
KERNEL_DIR="${KERNEL_DIR:-../linux}"
KERNEL_TAG="${KERNEL_TAG:-master}"
JOBS="${JOBS:-$(nproc)}"
ARCH="x86_64"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FRAGMENT="$SCRIPT_DIR/../kernel/config/jade-x86_64.config"

# ── Validate ─────────────────────────────────────────────────────────────────
[[ -n "$KERNEL_FORK" ]] || die "KERNEL_FORK not set.
  Usage: KERNEL_FORK=https://github.com/<you>/linux make kernel"

[[ -f "$CONFIG_FRAGMENT" ]] || die "kernel config fragment not found: $CONFIG_FRAGMENT"

# ── Clone or verify existing source ──────────────────────────────────────────
if [[ ! -d "$KERNEL_DIR" ]]; then
    info "Cloning $KERNEL_FORK at tag $KERNEL_TAG..."
    git clone --depth=1 --branch "$KERNEL_TAG" "$KERNEL_FORK" "$KERNEL_DIR"
    ok "cloned to $KERNEL_DIR"
else
    info "Kernel source already at $KERNEL_DIR — skipping clone."
    info "To update: git -C $KERNEL_DIR fetch && git -C $KERNEL_DIR checkout $KERNEL_TAG"
fi

cd "$KERNEL_DIR"

# ── Verify Rust toolchain meets kernel requirements ───────────────────────────
info "Checking kernel Rust requirements..."
make ARCH="$ARCH" LLVM=1 rustavailable || die "Rust toolchain does not meet kernel requirements.
  Run 'make setup' to install the correct toolchain."
ok "Rust toolchain compatible"

# ── Configure ─────────────────────────────────────────────────────────────────
info "Generating base config (x86_64 defconfig)..."
make ARCH="$ARCH" LLVM=1 defconfig

info "Merging JADE config fragment..."
./scripts/kconfig/merge_config.sh .config "$CONFIG_FRAGMENT"
ok "kernel configured"

# ── Build ─────────────────────────────────────────────────────────────────────
info "Building kernel ($JOBS parallel jobs, LLVM toolchain)..."
make ARCH="$ARCH" LLVM=1 -j"$JOBS"

echo ""
ok "Kernel build complete."
echo "    Image:   $KERNEL_DIR/arch/x86_64/boot/bzImage"
echo "    Modules: $KERNEL_DIR (run 'make modules_install' to install)"
