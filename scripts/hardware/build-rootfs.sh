#!/usr/bin/env bash
# Populate the JADE OS root ext4 partition with all binaries and config.
#
# Required env vars:
#   MODEL_PATH  — path to the GGUF model on the build host
#
# Optional env vars:
#   DEVICE      — NVMe device (default: /dev/nvme0n1)
#   ROOT_MOUNT  — where to mount the root partition (default: /mnt/jade-root)
#   KO_PATH     — path to jade_core.ko (default: kernel/modules/jade_core/jade_core.ko)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"

DEVICE="${DEVICE:-/dev/nvme0n1}"
ROOT_MOUNT="${ROOT_MOUNT:-/mnt/jade-root}"
KO_PATH="${KO_PATH:-$REPO_ROOT/kernel/modules/jade_core/jade_core.ko}"

JADE_INIT_BIN="$TARGET_DIR/x86_64-unknown-linux-musl/release/jade-init"
JADE_BIN="$TARGET_DIR/release/jade"
JADE_TREE_BIN="$TARGET_DIR/release/jade-tree"

# ── Helpers ───────────────────────────────────────────────────────────────────

info() { echo "  [rootfs] $*"; }
ok()   { echo "  [rootfs] ✓ $*"; }
die()  { echo "  [rootfs] FATAL: $*" >&2; exit 1; }

# ── Preflight ─────────────────────────────────────────────────────────────────

[[ $EUID -eq 0 ]]  || die "must run as root"
[[ -b "${DEVICE}p2" ]] || die "${DEVICE}p2 not found — run partition.sh first"

[[ -n "${MODEL_PATH:-}" ]] || die "MODEL_PATH is required (path to GGUF on this host)"
[[ -f "$MODEL_PATH" ]]     || die "model not found at $MODEL_PATH"

[[ -f "$JADE_INIT_BIN" ]] || die "jade-init not built — run: cargo build -p init --release --target x86_64-unknown-linux-musl"
[[ -f "$JADE_BIN" ]]      || die "jade not built — run: cargo build -p jadec --release"
[[ -f "$JADE_TREE_BIN" ]] || die "jade-tree not built — run: cargo build -p jade-tree --release --features inference"
[[ -f "$KO_PATH" ]]       || die "jade_core.ko not found at $KO_PATH — run: make modules"

# ── Mount ─────────────────────────────────────────────────────────────────────

info "Mounting ${DEVICE}p2 at $ROOT_MOUNT ..."
mkdir -p "$ROOT_MOUNT"
mount "${DEVICE}p2" "$ROOT_MOUNT"

cleanup() { umount "$ROOT_MOUNT" 2>/dev/null || true; }
trap cleanup EXIT

# ── Directory tree ────────────────────────────────────────────────────────────

info "Creating directory tree ..."
mkdir -p \
    "$ROOT_MOUNT/sbin" \
    "$ROOT_MOUNT/usr/bin" \
    "$ROOT_MOUNT/jade/inference" \
    "$ROOT_MOUNT/jade/programs" \
    "$ROOT_MOUNT/jade/modules" \
    "$ROOT_MOUNT/etc/jade" \
    "$ROOT_MOUNT/proc" \
    "$ROOT_MOUNT/sys" \
    "$ROOT_MOUNT/dev/pts" \
    "$ROOT_MOUNT/run/jade" \
    "$ROOT_MOUNT/tmp" \
    "$ROOT_MOUNT/root/.jade/memory/facts" \
    "$ROOT_MOUNT/root/.jade/memory/sessions" \
    "$ROOT_MOUNT/root/projects"

# ── Binaries ──────────────────────────────────────────────────────────────────

info "Installing jade-init -> /sbin/init ..."
install -m 0755 "$JADE_INIT_BIN" "$ROOT_MOUNT/sbin/init"

info "Installing jade -> /usr/bin/jade ..."
install -m 0755 "$JADE_BIN" "$ROOT_MOUNT/usr/bin/jade"

info "Installing jade-tree -> /jade/inference/jade-tree ..."
install -m 0755 "$JADE_TREE_BIN" "$ROOT_MOUNT/jade/inference/jade-tree"

# ── Model ─────────────────────────────────────────────────────────────────────

info "Copying model -> /jade/inference/model.gguf (this may take a moment) ..."
install -m 0644 "$MODEL_PATH" "$ROOT_MOUNT/jade/inference/model.gguf"

# ── Programs ──────────────────────────────────────────────────────────────────

info "Installing .jde programs -> /jade/programs/ ..."
install -m 0644 "$REPO_ROOT"/programs/*.jde "$ROOT_MOUNT/jade/programs/"

# ── Kernel module ─────────────────────────────────────────────────────────────

info "Installing jade_core.ko -> /jade/modules/ ..."
install -m 0644 "$KO_PATH" "$ROOT_MOUNT/jade/modules/jade_core.ko"

# ── Config ────────────────────────────────────────────────────────────────────

info "Writing /etc/jade/config.toml ..."
cat > "$ROOT_MOUNT/etc/jade/config.toml" <<'EOF'
# JADE OS machine configuration
# jade-tree reads JADE_MODEL_PATH (env override) or uses /jade/inference/model.gguf
EOF

info "Writing /etc/fstab ..."
ROOT_UUID=$(blkid -s UUID -o value "${DEVICE}p2")
DATA_UUID=$(blkid -s UUID -o value "${DEVICE}p3")
cat > "$ROOT_MOUNT/etc/fstab" <<EOF
# JADE OS fstab
UUID=$ROOT_UUID  /                    ext4  defaults,noatime  0  1
UUID=$DATA_UUID  /root/.jade/memory   ext4  defaults,noatime  0  2
EOF

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
ok "Root filesystem populated:"
echo ""
echo "    /sbin/init                       $(du -sh "$ROOT_MOUNT/sbin/init" | cut -f1)"
echo "    /usr/bin/jade                    $(du -sh "$ROOT_MOUNT/usr/bin/jade" | cut -f1)"
echo "    /jade/inference/jade-tree        $(du -sh "$ROOT_MOUNT/jade/inference/jade-tree" | cut -f1)"
echo "    /jade/inference/model.gguf       $(du -sh "$ROOT_MOUNT/jade/inference/model.gguf" | cut -f1)"
echo "    /jade/modules/jade_core.ko       $(du -sh "$ROOT_MOUNT/jade/modules/jade_core.ko" | cut -f1)"
echo "    /jade/programs/                  $(ls "$ROOT_MOUNT/jade/programs/" | wc -l) files"
echo ""
ok "Unmounting ${DEVICE}p2"
trap - EXIT
umount "$ROOT_MOUNT"
