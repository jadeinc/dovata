#!/usr/bin/env bash
# dev-vm/build-kernel.sh — inner kernel build script, runs INSIDE the dev VM.
#
# Called by scripts/dev-vm/build-kernel-vm.sh via dev-vm/run.sh.
# Never run directly on the host Mac.
#
# Environment (injected by caller):
#   KERNEL_FORK   git URL of the Linux fork
#   KERNEL_TAG    branch or tag to clone
#   JOBS          parallel make jobs

set -euo pipefail

info() { echo "==> $*"; }

KERNEL_FORK="${KERNEL_FORK:-https://github.com/joericks1998/linux.git}"
KERNEL_TAG="${KERNEL_TAG:-master}"
JOBS="${JOBS:-$(nproc)}"

VM_KERNEL_DIR="/home/jade/linux"
CONFIG_FRAGMENT="/jade-os/kernel/config/jade-x86_64.config"
BUILD_OUT="/jade-os/build"

export PATH="/home/jade/.cargo/bin:$PATH"

# ── Build deps (idempotent) ───────────────────────────────────────────────────
info "Installing kernel build dependencies..."
sudo apt-get install -y -q \
    bc bison flex libelf-dev libssl-dev pahole \
    clang lld llvm 2>&1 | grep -E "^(Setting up|Get:|Err:)" || true

# ── Rust target for kernel (x86_64-unknown-linux-gnu) ────────────────────────
rustup target add x86_64-unknown-linux-gnu 2>/dev/null || true

# bindgen is required for CONFIG_RUST kernel modules
if ! command -v bindgen &>/dev/null; then
    info "Installing bindgen-cli (required for CONFIG_RUST)..."
    cargo install --locked bindgen-cli 2>/dev/null || true
fi

# ── Clone kernel if needed ────────────────────────────────────────────────────
if [[ ! -d "$VM_KERNEL_DIR/.git" ]]; then
    info "Cloning $KERNEL_FORK @ $KERNEL_TAG ..."
    git clone --depth=1 --branch "$KERNEL_TAG" "$KERNEL_FORK" "$VM_KERNEL_DIR"
else
    CURRENT=$(git -C "$VM_KERNEL_DIR" describe --tags --exact-match 2>/dev/null \
              || git -C "$VM_KERNEL_DIR" rev-parse --short HEAD)
    if [[ "$CURRENT" != "$KERNEL_TAG" ]]; then
        info "Re-cloning (was at $CURRENT, want $KERNEL_TAG)..."
        rm -rf "$VM_KERNEL_DIR"
        git clone --depth=1 --branch "$KERNEL_TAG" "$KERNEL_FORK" "$VM_KERNEL_DIR"
    else
        info "Kernel source already at $KERNEL_TAG — skipping clone"
    fi
fi

cd "$VM_KERNEL_DIR"

# ── Configure ─────────────────────────────────────────────────────────────────
info "x86_64 defconfig..."
make ARCH=x86_64 LLVM=1 defconfig -s

info "Merging JADE config fragment..."
./scripts/kconfig/merge_config.sh .config "$CONFIG_FRAGMENT" >/dev/null

# Disable CONFIG_RUST if the toolchain doesn't satisfy the kernel's requirements.
if ! make ARCH=x86_64 LLVM=1 rustavailable >/dev/null 2>&1; then
    echo "    note: Rust toolchain not compatible — disabling CONFIG_RUST for this build"
    ./scripts/config --disable CONFIG_RUST
fi

# ── Build ─────────────────────────────────────────────────────────────────────
info "Building kernel ($JOBS jobs)..."
make ARCH=x86_64 LLVM=1 -j"$JOBS"

info "Kernel build complete — bzImage at /home/jade/linux/arch/x86_64/boot/bzImage"
