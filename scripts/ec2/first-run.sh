#!/usr/bin/env bash
# JADE OS — One-time EC2 instance setup
#
# Run this ONCE after provisioning. Idempotent — safe to re-run.
# Installs: CUDA 12, Rust toolchain, musl-tools,
# downloads Qwen3-Coder-30B-A3B-Instruct-Q4_K_M model to /jade/inference/model.gguf.
#
# Run from your local machine:
#   bash scripts/ec2/first-run.sh
#
# Or directly on the instance:
#   bash ~/jade-os/scripts/ec2/first-run.sh --local
set -euo pipefail

info() { echo "==> $*"; }
ok()   { echo "    OK: $*"; }
warn() { echo "    WARN: $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LOCAL=0
for arg in "$@"; do [[ "$arg" == "--local" ]] && LOCAL=1; done

# ── Remote dispatch ───────────────────────────────────────────────────────────
if [[ "$LOCAL" -eq 0 ]]; then
    ENV_FILE="$ROOT/.env.ec2"
    [[ -f "$ENV_FILE" ]] || die ".env.ec2 not found"
    # shellcheck source=/dev/null
    [[ -f "$ROOT/.env" ]] && set -a && source "$ROOT/.env" && set +a
    set -a; source "$ENV_FILE"; set +a

    AWS="$ROOT/.venv/bin/aws"
    if [[ ! -f "$AWS" ]]; then
        AWS=$(command -v aws 2>/dev/null || true)
    fi
    [[ -n "$AWS" ]] || die "aws CLI not found — run: bash scripts/setup-dev.sh"

    HOST_FILE="$ROOT/.ec2-host"
    if [[ -f "$HOST_FILE" ]]; then
        HOST=$(cat "$HOST_FILE")
    else
        HOST=$("$AWS" ec2 describe-instances \
            --instance-ids "$JADE_INSTANCE_ID" \
            --region "$JADE_REGION" \
            --query 'Reservations[0].Instances[0].PublicDnsName' \
            --output text)
    fi

    KEY="$ROOT/$JADE_SSH_KEY"
    SSH="ssh -o StrictHostKeyChecking=no -i $KEY $JADE_SSH_USER@$HOST"

    info "Uploading setup script to $HOST..."
    scp -o StrictHostKeyChecking=no -i "$KEY" \
        "$ROOT/scripts/ec2/first-run.sh" \
        "$JADE_SSH_USER@$HOST:~/ec2-first-run.sh"

    info "Running first-run setup on instance (this takes ~10 minutes)..."
    ssh -t -o StrictHostKeyChecking=no -i "$KEY" "$JADE_SSH_USER@$HOST" \
        "bash ~/ec2-first-run.sh --local"
    ok "First-run setup complete"
    exit 0
fi

# ── Everything below runs ON the EC2 instance ─────────────────────────────────

REPO_DIR="${REPO_DIR:-~/jade-os}"
MODEL_DEST="${JADE_MODEL_PATH:-/jade/inference/model.gguf}"
MODEL_URL="https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"

# ── System packages ───────────────────────────────────────────────────────────
info "Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential cmake curl git pkg-config \
    musl-tools clang lld llvm \
    libxml2-dev libzstd-dev \
    gcc-13 g++-13 \
    linux-headers-"$(uname -r)" \
    2>/dev/null || true

# CUDA runfile expects libxml2.so.2 but Ubuntu 26.04 ships libxml2.so.16
if [[ ! -f /usr/lib/x86_64-linux-gnu/libxml2.so.2 ]]; then
    LIB=$(find /usr/lib/x86_64-linux-gnu -name "libxml2.so.*" | sort | head -1)
    [[ -n "$LIB" ]] && sudo ln -sf "$LIB" /usr/lib/x86_64-linux-gnu/libxml2.so.2 && sudo ldconfig
fi
ok "System packages installed"

# ── LLVM 18 dev (required for jadec --features llvm / inkwell 0.8) ───────────
if ! dpkg -s llvm-18-dev &>/dev/null 2>&1; then
    info "Installing LLVM 18 dev..."
    sudo apt-get install -y -qq llvm-18-dev
    ok "LLVM 18 dev installed"
else
    ok "LLVM 18 dev already installed"
fi
# Ubuntu ships LLVMPolly.so but not libPolly.a/libPollyISL.a; llvm-sys 181 links
# them statically. Create empty stubs so the linker is satisfied (jadec doesn't
# call Polly APIs — inkwell uses it as an optimizer, not a required dependency).
for stub in /usr/lib/llvm-18/lib/libPolly.a /usr/lib/llvm-18/lib/libPollyISL.a; do
    [[ -f "$stub" ]] || sudo ar crs "$stub"
done
ok "Polly stub archives present"

# ── CUDA 12 ───────────────────────────────────────────────────────────────────
if ! command -v nvcc &>/dev/null; then
    info "Installing CUDA 12 toolkit..."
    ARCH="x86_64"
    # Derive ubuntu version string (e.g. 2604 for 26.04)
    UBUNTU_VER=$(lsb_release -rs | tr -d '.')
    DISTRO="ubuntu${UBUNTU_VER}"
    KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb"

    # Try the native repo first; fall back to the CUDA runfile installer if the
    # keyring doesn't exist or the package isn't in the repo yet (Ubuntu 26.04).
    APT_OK=0
    if wget -q --spider "$KEYRING_URL" 2>/dev/null; then
        wget -q "$KEYRING_URL"
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        rm -f cuda-keyring_1.1-1_all.deb
        sudo apt-get update -qq
        # Try newest available toolkit first for best distro compatibility
        if sudo apt-get install -y -qq cuda-toolkit-13-1 2>/dev/null; then
            APT_OK=1
            ok "CUDA 13.1 installed via apt"
        elif sudo apt-get install -y -qq cuda-toolkit-13 2>/dev/null; then
            APT_OK=1
            ok "CUDA 13 installed via apt"
        else
            warn "No cuda-toolkit in apt repo for $DISTRO — falling back to runfile"
        fi
    fi
    if [[ "$APT_OK" -eq 0 ]]; then
        RUNFILE="cuda_12.6.3_560.35.05_linux.run"
        RUNFILE_URL="https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/${RUNFILE}"
        # /tmp is tmpfs (RAM-backed); download and extract on disk instead
        sudo mkdir -p /var/tmp/cuda-install
        wget -q --show-progress -O "/var/tmp/${RUNFILE}" "$RUNFILE_URL"
        sudo TMPDIR=/var/tmp/cuda-install sh "/var/tmp/${RUNFILE}" --silent --toolkit --override
        sudo rm -rf "/var/tmp/${RUNFILE}" /var/tmp/cuda-install
        ok "CUDA 12.6 installed via runfile"
    fi
else
    ok "CUDA already installed: $(nvcc --version | head -1)"
fi

# Ensure nvcc is in PATH for this session (13.1 preferred, fall back to symlink)
export PATH="/usr/local/cuda-13.1/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# ── Patch CUDA headers for glibc 2.41 noexcept incompatibility ───────────────
# CUDA 13.1 declares rsqrt/rsqrtf without noexcept; glibc 2.41 redeclares them
# with noexcept — nvcc flags the mismatch as an error. Add noexcept to CUDA side.
MATH_H=$(ls /usr/local/cuda-*/targets/x86_64-linux/include/crt/math_functions.h 2>/dev/null | sort | tail -1)
if [[ -n "$MATH_H" ]]; then
    if ! grep -q "rsqrt(double x) noexcept" "$MATH_H" 2>/dev/null; then
        LINE=$(grep -n "rsqrt(double x);" "$MATH_H" | grep -v "//" | head -1 | cut -d: -f1)
        [[ -n "$LINE" ]] && sudo sed -i "${LINE}s/rsqrt(double x);/rsqrt(double x) noexcept(true);/" "$MATH_H"
        LINE=$(grep -n "rsqrtf(float x);" "$MATH_H" | grep -v "//" | head -1 | cut -d: -f1)
        [[ -n "$LINE" ]] && sudo sed -i "${LINE}s/rsqrtf(float x);/rsqrtf(float x) noexcept(true);/" "$MATH_H"
        ok "Patched CUDA math_functions.h for glibc 2.41 noexcept compat"
    else
        ok "CUDA math_functions.h already patched"
    fi
fi

# ── nouveau blacklist (required for NVIDIA driver to bind on Ubuntu 26.04) ───
if ! grep -q "blacklist nouveau" /etc/modprobe.d/blacklist-nouveau.conf 2>/dev/null; then
    info "Blacklisting nouveau driver..."
    echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf > /dev/null
    echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf > /dev/null
    sudo update-initramfs -u -k "$(uname -r)" 2>/dev/null | tail -1
    ok "nouveau blacklisted"
fi

# ── NVIDIA driver + kernel module ─────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    info "Installing NVIDIA driver..."
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-driver
    ok "NVIDIA driver installed"
fi

# Unbind nouveau if still attached, then load nvidia
if lspci -k 2>/dev/null | grep -q "Kernel driver in use: nouveau"; then
    PCIID=$(lspci | grep -i nvidia | awk '{print $1}' | head -1)
    echo "0000:${PCIID}" | sudo tee /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null || true
    sudo rmmod nouveau 2>/dev/null || true
fi
sudo modprobe nvidia 2>/dev/null || true

# ── NVIDIA driver check ───────────────────────────────────────────────────────
if nvidia-smi &>/dev/null; then
    ok "NVIDIA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
else
    warn "nvidia-smi failed — NVIDIA driver may not be installed"
    warn "Run: sudo apt-get install -y nvidia-driver-535"
    warn "Then reboot and re-run this script"
fi

# ── Rust toolchain ────────────────────────────────────────────────────────────
if ! command -v cargo &>/dev/null; then
    info "Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --quiet
    ok "Rust installed"
fi
# shellcheck source=/dev/null
source "$HOME/.cargo/env"
rustup target add x86_64-unknown-linux-musl
ok "Rust: $(rustc --version)"

# ── socket directory ──────────────────────────────────────────────────────────
mkdir -p "$HOME/.jade"
ok "$HOME/.jade directory ready"

# ── Download model ────────────────────────────────────────────────────────────
sudo mkdir -p "$(dirname "$MODEL_DEST")"
if [[ -f "$MODEL_DEST" ]]; then
    ok "Model already present: $(du -sh "$MODEL_DEST" | cut -f1)  $MODEL_DEST"
else
    info "Downloading Qwen3-Coder-30B-A3B-Instruct-Q4_K_M (~18.6GB)..."
    sudo curl -L --progress-bar -o "$MODEL_DEST" "$MODEL_URL"
    ok "Model downloaded: $(du -sh "$MODEL_DEST" | cut -f1)  $MODEL_DEST"
fi

echo ""
info "First-run setup complete"
info "Next: bash scripts/ec2/deploy.sh"
