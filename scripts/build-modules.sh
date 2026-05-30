#!/usr/bin/env bash
# JADE OS — Build kernel modules
# Builds all JADE Rust kernel modules against the built kernel source.
set -euo pipefail

info() { echo "==> $*"; }
ok()   { echo "    OK: $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

KERNEL_DIR="${KERNEL_DIR:-../linux}"
JOBS="${JOBS:-$(nproc)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULES_DIR="$SCRIPT_DIR/../kernel/modules"

# ── Validate ─────────────────────────────────────────────────────────────────
[[ -f "$KERNEL_DIR/Makefile" ]] || die "Kernel source not found at $KERNEL_DIR.
  Run 'make kernel KERNEL_FORK=<url>' first."

KERNEL_ABS="$(realpath "$KERNEL_DIR")"

# ── Build each module ─────────────────────────────────────────────────────────
# Kbuild creates temp files (*.mod, .tmp_*) in the module source directory.
# When the source is on a 9p mount (macOS host), writes fail. Copy to a local
# temp dir first, build there, then copy the .ko back.
for module_dir in "$MODULES_DIR"/*/; do
    name="$(basename "$module_dir")"
    info "Building module: $name"

    TMPBUILD="$(mktemp -d)"
    trap 'rm -rf "$TMPBUILD"' EXIT
    cp -r "$module_dir/." "$TMPBUILD/"

    make -C "$TMPBUILD" KERNEL_DIR="$KERNEL_ABS" ARCH=x86_64 LLVM=1 -j"$JOBS"

    # Copy .ko back to repo so build-initramfs.sh can find it
    cp "$TMPBUILD/$name.ko" "$module_dir/$name.ko"
    ok "$name.ko built and copied to $module_dir"
done

echo ""
ok "All kernel modules built."
echo "    To install: make -C kernel/modules/<name> KERNEL_DIR=$KERNEL_ABS install"
