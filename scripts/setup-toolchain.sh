#!/usr/bin/env bash
# JADE OS — Toolchain Setup
# Installs everything needed to build JADE OS on Ubuntu/Debian.
# Run once on a fresh machine before any other make target.
set -euo pipefail

info()  { echo "==> $*"; }
ok()    { echo "    OK: $*"; }
die()   { echo "ERROR: $*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || die "This script must run on Linux (Ubuntu/Debian)"

# ── Kernel build dependencies ────────────────────────────────────────────────
info "Installing kernel build dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential \
    bc \
    flex \
    bison \
    libssl-dev \
    libelf-dev \
    libncurses-dev \
    clang \
    llvm \
    lld \
    pahole \
    dwarves \
    gcc \
    make \
    git \
    curl \
    pkg-config \
    musl-tools        # required for x86_64-unknown-linux-musl (static init binary)
ok "kernel build dependencies installed"

# ── llama.cpp build dependencies ─────────────────────────────────────────────
info "Installing llama.cpp build dependencies..."
sudo apt-get install -y \
    cmake \
    libclang-dev \
    ninja-build
ok "llama.cpp build dependencies installed"

# ── LLVM 18 (required for JadeLang native codegen — `make jadec-llvm`) ───────
info "Installing LLVM 18 development libraries..."
sudo apt-get install -y \
    llvm-18-dev \
    libclang-18-dev
ok "LLVM 18 dev libraries installed"

# ── CUDA toolkit (optional — only installs if nvcc is not already present) ───
if ! command -v nvcc &>/dev/null; then
    info "nvcc not found — skipping CUDA toolkit."
    info "To build jade-tree with GPU support, install the NVIDIA CUDA toolkit:"
    info "  https://developer.nvidia.com/cuda-downloads"
    info "Then rebuild: make jade-tree CUDA=1"
else
    ok "CUDA toolkit already present: $(nvcc --version | head -1)"
fi

# ── Rust ─────────────────────────────────────────────────────────────────────
if ! command -v rustup &>/dev/null; then
    info "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
    source "$HOME/.cargo/env"
else
    info "Rust already installed, updating..."
    rustup update stable
fi

# Components and targets (mirrors rust-toolchain.toml)
info "Installing Rust components and targets..."
rustup component add rust-src rustfmt clippy llvm-tools
rustup target add x86_64-unknown-linux-musl
ok "Rust toolchain ready"

# ── bindgen CLI (required for kernel Rust bindings) ──────────────────────────
info "Installing bindgen-cli..."
cargo install --locked bindgen-cli
ok "bindgen installed: $(bindgen --version)"

echo ""
info "Toolchain setup complete."
echo "    Next steps:"
echo "      make kernel          — fetch and build the Linux kernel"
echo "      make jade-tree       — build inference daemon (CPU)"
echo "      make jade-tree CUDA=1 — build inference daemon (NVIDIA GPU)"
