#!/usr/bin/env bash
# JADE OS — Sync source to EC2, build, and run inference tests
#
# Usage:
#   bash scripts/ec2/deploy.sh              # sync + build all + test + restart jade-tree
#   bash scripts/ec2/deploy.sh --build      # sync + build all (no tests, restart jade-tree)
#   bash scripts/ec2/deploy.sh --programs   # sync programs/ + build jadec+LLVM + compile programs to /usr/bin/ (no restart)
#   bash scripts/ec2/deploy.sh --jadec      # sync + build jadec+LLVM only + install (no restart)
set -euo pipefail

info() { echo "==> $*"; }
ok()   { echo "    OK: $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="full"
for arg in "$@"; do
    case "$arg" in
        --build)    MODE="build"    ;;
        --programs) MODE="programs" ;;
        --jadec)    MODE="jadec"    ;;
    esac
done

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

[[ -n "$HOST" ]] || die "Could not resolve instance hostname — run ec2-start.sh first"

KEY="$ROOT/$JADE_SSH_KEY"
SSH="ssh -o StrictHostKeyChecking=no -i $KEY $JADE_SSH_USER@$HOST"

# ── Sync source ───────────────────────────────────────────────────────────────
RSYNC_BASE=(rsync -az --progress
    --exclude '.git/'
    --exclude 'target/'
    --exclude 'keys/'
    --exclude '.env.ec2'
    --exclude '.ec2-host'
    --exclude 'build/'
    --exclude '*.ko' --exclude '*.o' --exclude '*.mod'
    -e "ssh -o StrictHostKeyChecking=no -i $KEY"
)

if [[ "$MODE" == "programs" ]]; then
    info "Syncing programs/, jadec/, and runtime/ to $HOST ..."
    "${RSYNC_BASE[@]}" "$ROOT/programs/"       "$JADE_SSH_USER@$HOST:~/jade-os/programs/"
    "${RSYNC_BASE[@]}" "$ROOT/jadec/"          "$JADE_SSH_USER@$HOST:~/jade-os/jadec/"
    "${RSYNC_BASE[@]}" "$ROOT/runtime/"        "$JADE_SSH_USER@$HOST:~/jade-os/runtime/"
    "${RSYNC_BASE[@]}" "$ROOT/Makefile"        "$JADE_SSH_USER@$HOST:~/jade-os/Makefile"
else
    info "Syncing source to $HOST:~/jade-os ..."
    "${RSYNC_BASE[@]}" "$ROOT/" "$JADE_SSH_USER@$HOST:~/jade-os/"
fi
ok "Source synced"

# ── Build on EC2 ──────────────────────────────────────────────────────────────
if [[ "$MODE" == "programs" ]]; then
    info "Building jadec+LLVM and installing programs..."
    $SSH bash <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true
cd ~/jade-os
echo "  cargo build -p jade --features llvm..."
LLVM_SYS_181_PREFIX=/usr/lib/llvm-18 cargo build --release -p jade --features llvm 2>&1 | grep -E 'error\[|ld returned|undefined|cannot find|note:|error:' | head -30
echo "  make install-jadec..."
make install-jadec
echo "  make install-programs..."
make install-programs
echo "  make compile-programs..."
make compile-programs
echo "Done"
REMOTE
    ok "Programs built and installed"

elif [[ "$MODE" == "jadec" ]]; then
    info "Building jadec+LLVM..."
    $SSH bash <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true
cd ~/jade-os
echo "  cargo build -p jade --features llvm (jadec+LLVM)..."
LLVM_SYS_181_PREFIX=/usr/lib/llvm-18 cargo build --release -p jade --features llvm 2>&1 | grep -E 'error\[|ld returned|undefined|cannot find|note:|error:' | head -30 2>&1 | tail -3
echo "  make install-jadec..."
make install-jadec
echo "Done"
REMOTE
    ok "jadec built and installed"

else
    # full or build
    info "Building on instance..."
    $SSH bash <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
source "$HOME/.cargo/env" 2>/dev/null || true

cd ~/jade-os

echo "  cargo build -p jade --features llvm (jadec+LLVM)..."
LLVM_SYS_181_PREFIX=/usr/lib/llvm-18 cargo build --release -p jade --features llvm 2>&1 | grep -E 'error\[|ld returned|undefined|cannot find|note:|error:' | head -30 2>&1 | tail -3

echo "  cargo build -p init (jade-init, musl)..."
cargo build --release -p init --target x86_64-unknown-linux-musl 2>&1 | tail -3

# jade-tree: rebuild only if source changed since last build
TREE_STAMP=~/jade-tree.sha
TREE_SHA=$(find ~/jade-os/jade-tree/src -type f | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1 || echo "none")
CACHED_SHA=$(cat "$TREE_STAMP" 2>/dev/null || echo "")

if [[ "$TREE_SHA" != "$CACHED_SHA" ]]; then
    echo "  cargo build -p jade-tree --features cuda..."
    cargo build --release -p jade-tree --features cuda 2>&1 | tail -5
    echo "$TREE_SHA" > "$TREE_STAMP"
else
    echo "  jade-tree: up to date (SHA unchanged)"
fi

echo "  make install (jade → /usr/bin, programs → /usr/share/jade/programs)..."
make install

# jade-gpu-mon: rebuild only if source changed
GPU_MON_STAMP=~/jade-gpu-mon.sha
GPU_MON_SHA=$(find ~/jade-os/tools/jade-gpu-mon/src -type f | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1 || echo "none")
GPU_MON_CACHED=$(cat "$GPU_MON_STAMP" 2>/dev/null || echo "")

if [[ "$GPU_MON_SHA" != "$GPU_MON_CACHED" ]]; then
    echo "  cargo build -p jade-gpu-mon..."
    cargo build --release --manifest-path ~/jade-os/tools/jade-gpu-mon/Cargo.toml 2>&1 | tail -3
    sudo install -m 0755 ~/jade-os/tools/jade-gpu-mon/target/release/jade-gpu-mon /usr/local/bin/jade-gpu-mon
    echo "$GPU_MON_SHA" > "$GPU_MON_STAMP"
else
    echo "  jade-gpu-mon: up to date (SHA unchanged)"
fi

echo "Build complete"
REMOTE
    ok "Build done"
fi

# ── Run tests ─────────────────────────────────────────────────────────────────
if [[ "$MODE" == "full" ]]; then
    info "Running inference test suite..."
    $SSH "bash ~/jade-os/scripts/ec2/run-tests.sh 2>&1"
fi

# ── Start jade-tree daemon ────────────────────────────────────────────────────
# Skipped for --programs and --jadec — jade-tree doesn't need to restart when
# only programs or the compiler change.
if [[ "$MODE" == "full" || "$MODE" == "build" ]]; then
    info "Starting jade-tree daemon..."
    $SSH bash <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.1/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

MODEL="${JADE_MODEL_PATH:-/jade/inference/model.gguf}"
JADE_TREE="$HOME/jade-os/target/release/jade-tree"

if pgrep -x jade-tree > /dev/null 2>&1; then
    echo "    stopping existing jade-tree..."
    pkill -x jade-tree 2>/dev/null || true
    sleep 2
fi
rm -f "$HOME/.jade/llm.sock"
mkdir -p "$HOME/.jade"

echo "    launching jade-tree (model load ~3min for 18GB model)..."
JADE_MODEL_PATH="$MODEL" nohup "$JADE_TREE" > /tmp/jade-tree.log 2>&1 &

for i in $(seq 1 150); do
    if grep -q 'jade-tree: listening on' /tmp/jade-tree.log 2>/dev/null; then
        echo "    jade-tree ready (pid $(pgrep -x jade-tree))"
        exit 0
    fi
    if ! pgrep -x jade-tree > /dev/null 2>&1; then
        echo "ERROR: jade-tree exited unexpectedly:"
        cat /tmp/jade-tree.log
        exit 1
    fi
    sleep 2
done
echo "ERROR: jade-tree did not become ready within 300s"
cat /tmp/jade-tree.log
exit 1
REMOTE
    ok "jade-tree running"
fi
