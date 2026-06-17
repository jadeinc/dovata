#!/usr/bin/env bash
# JADE OS — Run the model quantization pipeline on the EC2 GPU box, by hand.
#
# This is the manual driver for scripts/quantize-model.sh (download Qwen3-Coder,
# convert -> f16 GGUF, code-calibrated imatrix, quantize Q4_K_M) and publish to
# the jade-os S3 contract:
#   s3://jade-os-release-bucket/models/model.gguf (+ .gguf.sha256)
#
# It: starts the box (scripts/ec2/start.sh), rsyncs THIS repo to
# ~/jade-inference-daemon, then runs scripts/quantize-model.sh over SSH with the
# CUDA toolchain on PATH and your AWS creds (from .env) forwarded so the upload
# happens in-region from the box (no 18 GB round-trip through your laptop).
#
# The job is long (60 GB download + convert + imatrix + quantize — hours) and the
# box needs ~150 GB free disk; the underlying script preflights and fails fast.
#
# Usage:
#   bash scripts/ec2/quantize.sh                 # start, sync, build, upload
#   bash scripts/ec2/quantize.sh --no-upload     # build + verify only (no S3)
#   bash scripts/ec2/quantize.sh --stop          # also stop the box when done
#   bash scripts/ec2/quantize.sh --no-start      # assume the box is already up
#   bash scripts/ec2/quantize.sh -- --local-dir ~/merged   # pass-through to the
#                                                # pipeline (e.g. merged LoRA weights
#                                                # already present on the box)
set -euo pipefail

info() { echo "==> $*"; }
ok()   { echo "    OK: $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DO_START=1
DO_STOP=0
DO_UPLOAD=1
DETACHED=0
FOLLOW=0
PASS_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-start)  DO_START=0; shift ;;
        --stop)      DO_STOP=1; shift ;;
        --no-upload) DO_UPLOAD=0; shift ;;
        --detached)  DETACHED=1; shift ;;   # run on the box under setsid; survives SSH drop
        --follow)    FOLLOW=1; shift ;;      # after a detached launch, tail the log
        --)          shift; PASS_ARGS=("$@"); break ;;
        -h|--help)   sed -n '2,32p' "$0"; exit 0 ;;
        *)           die "unknown arg: $1 (use -- to pass args through to quantize-model.sh)" ;;
    esac
done
# --stop is meaningless with --detached (the job outlives this script).
[[ "$DETACHED" -eq 1 && "$DO_STOP" -eq 1 ]] && die "--stop can't be combined with --detached"

ENV_FILE="$ROOT/.env.ec2"
[[ -f "$ENV_FILE" ]] || die ".env.ec2 not found"
# shellcheck source=/dev/null
[[ -f "$ROOT/.env" ]] && set -a && source "$ROOT/.env" && set +a
set -a; source "$ENV_FILE"; set +a

AWS_LOCAL="$ROOT/.venv/bin/aws"
[[ -f "$AWS_LOCAL" ]] || AWS_LOCAL=$(command -v aws 2>/dev/null || true)

# --- bring the box up + resolve host ----------------------------------------
if [[ "$DO_START" -eq 1 ]]; then
    bash "$SCRIPT_DIR/start.sh"
fi
HOST_FILE="$ROOT/.ec2-host"
if [[ -f "$HOST_FILE" ]]; then
    HOST=$(cat "$HOST_FILE")
elif [[ -n "$AWS_LOCAL" ]]; then
    HOST=$("$AWS_LOCAL" ec2 describe-instances --instance-ids "$JADE_INSTANCE_ID" \
        --region "$JADE_REGION" --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
fi
[[ -n "${HOST:-}" ]] || die "could not resolve host — run scripts/ec2/start.sh first (or drop --no-start)"
KEY="$ROOT/$JADE_SSH_KEY"
SSH=(ssh -o StrictHostKeyChecking=no -i "$KEY" "$JADE_SSH_USER@$HOST")

# --- sync this repo to the box ----------------------------------------------
REMOTE_DIR="jade-inference-daemon"
info "syncing repo to $HOST:~/$REMOTE_DIR"
rsync -az --delete --progress \
    --exclude '.git/' --exclude 'target/' --exclude 'keys/' \
    --exclude '.env' --exclude '.env.ec2' --exclude '.ec2-host' \
    --exclude 'dist/' --exclude 'build/' --exclude '*.gguf' \
    -e "ssh -o StrictHostKeyChecking=no -i $KEY" \
    "$ROOT/" "$JADE_SSH_USER@$HOST:~/$REMOTE_DIR/"
ok "synced"

# --- credentials forwarded for the in-region S3 upload ----------------------
if [[ "$DO_UPLOAD" -eq 1 ]]; then
    [[ -n "${AWS_ACCESS_KEY_ID:-}" && -n "${AWS_SECRET_ACCESS_KEY:-}" ]] \
        || die "AWS creds not in .env — add AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or use --no-upload"
    UPLOAD_FLAG="--upload"
else
    UPLOAD_FLAG=""
fi

# Assemble pass-through args (quoted) for the remote command line.
EXTRA=""
for a in "${PASS_ARGS[@]:-}"; do [[ -n "$a" ]] && EXTRA+=" $(printf '%q' "$a")"; done

# --- build the remote runner script -----------------------------------------
# We write a self-contained runner to ~/quantize-run.sh on the box, then either
# run it synchronously or launch it under setsid (detached, survives SSH drop).
# Unquoted heredoc: local $AWS_*/$UPLOAD_FLAG/$EXTRA/$JADE_REGION are injected
# here; remote-evaluated vars are escaped (\$HOME, \$PATH, ...).
REMOTE_RUNNER="quantize-run.sh"
REMOTE_LOG="quantize.log"
info "staging remote runner ~/$REMOTE_RUNNER on $HOST"
"${SSH[@]}" "cat > ~/$REMOTE_RUNNER" <<REMOTE
#!/usr/bin/env bash
set -uo pipefail
export AWS_ACCESS_KEY_ID='${AWS_ACCESS_KEY_ID:-}'
export AWS_SECRET_ACCESS_KEY='${AWS_SECRET_ACCESS_KEY:-}'
export AWS_DEFAULT_REGION='${JADE_REGION:-us-east-1}'
# CUDA toolchain (box has 13.1) on PATH. (We do NOT add ~/jade-os/.venv/bin: the
# monorepo venv on this box is broken — its python shebang points at a Mac path —
# and would shadow a working aws. quantize-model.sh provisions its own aws.)
export PATH="/usr/local/cuda/bin:/usr/local/cuda-13.1/bin:\$HOME/.cargo/bin:\$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-13.1/lib64:\${LD_LIBRARY_PATH:-}"

# --- scratch on the fast instance-store NVMe --------------------------------
# This box has only ~16 GB RAM, so the imatrix pass mmap-pages the ~55 GB of f16
# expert weights from disk. That MUST be the local NVMe instance store (GB/s),
# not the EBS root (gp3, ~125-250 MB/s) or the pass takes ~10x longer. We mount
# it at /mnt/scratch and run the whole pipeline there. The instance store is
# EPHEMERAL (wiped on stop/start) — fine, it's pure scratch; we reformat each run.
# Only a disk whose hardware MODEL is "Instance Storage" is ever formatted.
SCRATCH=/mnt/scratch
if ! mountpoint -q "\$SCRATCH"; then
    DEV=""
    for d in /dev/nvme*n1; do
        [[ -b "\$d" ]] || continue
        MODEL=\$(lsblk -dno MODEL "\$d" 2>/dev/null || true)
        MNT=\$(lsblk -no MOUNTPOINT "\$d" 2>/dev/null | tr -d ' \n')
        if [[ "\$MODEL" == *"Instance Storage"* && -z "\$MNT" ]]; then DEV="\$d"; break; fi
    done
    if [[ -n "\$DEV" ]]; then
        echo "==> formatting + mounting instance store \$DEV at \$SCRATCH (ephemeral scratch)"
        sudo mkfs.ext4 -F -q "\$DEV"
        sudo mkdir -p "\$SCRATCH"
        sudo mount "\$DEV" "\$SCRATCH"
        sudo chown "\$(id -u):\$(id -g)" "\$SCRATCH"
    else
        echo "==> WARNING: no instance-store NVMe found — falling back to \$HOME (EBS)."
        echo "    The imatrix mmap pass will be slow on EBS; grow the root volume and"
        echo "    ensure ~150 GB free, or attach instance storage."
        SCRATCH="\$HOME"
    fi
fi
export JADE_QUANTIZE_WORKDIR="\$SCRATCH/jade-quantize"
echo "==> workdir (scratch): \$JADE_QUANTIZE_WORKDIR"
df -h "\$SCRATCH"

cd "\$HOME/$REMOTE_DIR"
bash scripts/quantize-model.sh $UPLOAD_FLAG$EXTRA
rc=\$?
echo "JADE_QUANTIZE_DONE rc=\$rc"
exit \$rc
REMOTE
"${SSH[@]}" "chmod +x ~/$REMOTE_RUNNER"

if [[ "$DETACHED" -eq 1 ]]; then
    info "launching detached on $HOST (logs to ~/$REMOTE_LOG)"
    # setsid + nohup so the job outlives this SSH session. </dev/null detaches stdin.
    "${SSH[@]}" "setsid nohup bash ~/$REMOTE_RUNNER > ~/$REMOTE_LOG 2>&1 </dev/null & echo \"started pid \$!\""
    echo ""
    ok "detached build started on $HOST"
    echo "    follow:  bash scripts/ec2/quantize.sh --no-start --follow"
    echo "    or:      bash scripts/ec2/ssh.sh 'tail -f ~/$REMOTE_LOG'"
    echo "    done when the log shows: JADE_QUANTIZE_DONE rc=0"
    exit 0
fi

if [[ "$FOLLOW" -eq 1 ]]; then
    info "following ~/$REMOTE_LOG on $HOST (Ctrl-C to stop tailing; job keeps running)"
    exec "${SSH[@]}" "tail -n +1 -f ~/$REMOTE_LOG"
fi

# --- synchronous run --------------------------------------------------------
info "running quantize-model.sh on $HOST (this is the long part)"
"${SSH[@]}" "bash ~/$REMOTE_RUNNER"
ok "pipeline finished"

if [[ "$DO_UPLOAD" -eq 1 ]]; then
    echo ""
    ok "published s3://${S3_BUCKET:-jade-os-release-bucket}/${S3_PREFIX:-models}/model.gguf (+ .sha256)"
    echo "    jade-os CI will now bake this model into the next image build."
fi

if [[ "$DO_STOP" -eq 1 ]]; then
    info "stopping the box"
    bash "$SCRIPT_DIR/stop.sh"
fi
