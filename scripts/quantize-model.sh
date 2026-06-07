#!/usr/bin/env bash
# Produce the JADE OS production inference model: download (or take locally-merged)
# weights, convert to GGUF f16, build a code-calibrated importance matrix, and
# quantize to Q4_K_M — using the EXACT llama.cpp commit that this repo's jaded
# links, so the GGUF can never be rejected for quant-type / GGUF-version skew.
#
# Outputs (into --out-dir, default ./dist/model):
#   model.gguf            the Q4_K_M quantized model
#   model.gguf.sha256     `sha256sum model.gguf` output (jade-os reads field 1)
#   model.gguf.buildinfo  provenance (source rev, llama.cpp commit, sizes) — NOT
#                         uploaded to S3; for release notes / debugging only.
#
# This is the permanent self-quantization path: post-LoRA-fine-tune there is no
# community GGUF, so it accepts locally-merged weights too (--local-dir), not
# just an HF download.
#
# Heavy host required: peak ~150 GB disk (safetensors ~60 + f16 GGUF ~60 +
# Q4_K_M ~18) and a CUDA GPU for the imatrix pass — it can't run on a laptop or a
# GitHub-hosted runner. Run it on the EC2 GPU box; scripts/ec2/quantize.sh is the
# one-command driver (start box -> sync repo -> run this -> upload in-region).
#
# Usage:
#   scripts/quantize-model.sh                         # pins from ci/model-version.lock
#   scripts/quantize-model.sh --local-dir ./merged    # use locally-merged weights
#   scripts/quantize-model.sh --upload                # also push to the S3 contract
#   scripts/quantize-model.sh --out-dir /tmp/out
#
# Key env overrides (defaults come from ci/model-version.lock):
#   HF_REPO HF_REVISION QUANT IMATRIX_CHUNKS LLAMA_CPP_COMMIT CUDAARCHS
#   HF_TOKEN                 # for gated/private repos (Qwen3-Coder is ungated)
#   JADE_QUANTIZE_WORKDIR    # persistent build/download cache (default ~/jade-quantize)
#   JADE_IMATRIX_OFFLOAD     # llama-imatrix offload args (default: -ngl 99 --cpu-moe)
#   MIN_DISK_GB              # preflight free-space floor (default 150)
#   S3_BUCKET S3_PREFIX      # upload target (default jade-os-release-bucket / models)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

info() { echo "==> $*"; }
ok()   { echo "    OK: $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# --- args -------------------------------------------------------------------
LOCAL_DIR=""           # if set, skip HF download and convert these weights
OUT_DIR="$REPO_ROOT/dist/model"
DO_UPLOAD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-dir) LOCAL_DIR="${2:?--local-dir needs a path}"; shift 2 ;;
        --out-dir)   OUT_DIR="${2:?--out-dir needs a path}"; shift 2 ;;
        --upload)    DO_UPLOAD=1; shift ;;
        -h|--help)   sed -n '2,40p' "$0"; exit 0 ;;
        *)           die "unknown arg: $1 (try --help)" ;;
    esac
done

# --- pinned inputs (ci/model-version.lock is the source of truth) ------------
LOCK="$REPO_ROOT/ci/model-version.lock"
[[ -f "$LOCK" ]] || die "missing $LOCK"
# Load pins as DEFAULTS: a value already set (and non-empty) in the environment
# wins, so callers/CI can override any pin; the lock fills the rest. (Plain
# `source` would clobber overrides, so we assign only unset/empty vars.)
while IFS='=' read -r k v; do
    [[ "$k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -n "${!k:-}" ]] || printf -v "$k" '%s' "$v"
done < <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$LOCK")
export HF_REPO HF_REVISION QUANT IMATRIX_CHUNKS LLAMA_CPP_REPO LLAMA_CPP_COMMIT CUDAARCHS

CALIB="${CALIB:-$REPO_ROOT/ci/calib-code.txt}"
[[ -f "$CALIB" ]] || die "missing calibration corpus $CALIB"

WORKDIR="${JADE_QUANTIZE_WORKDIR:-$HOME/jade-quantize}"
IMATRIX_OFFLOAD="${JADE_IMATRIX_OFFLOAD:--ngl 99 --cpu-moe}"
MIN_DISK_GB="${MIN_DISK_GB:-150}"
S3_BUCKET="${S3_BUCKET:-jade-os-release-bucket}"
S3_PREFIX="${S3_PREFIX:-models}"

info "JADE OS model quantization"
echo "    HF_REPO          : $HF_REPO"
echo "    HF_REVISION      : $HF_REVISION"
echo "    QUANT            : $QUANT  (imatrix chunks cap: $IMATRIX_CHUNKS)"
echo "    LLAMA_CPP_COMMIT : $LLAMA_CPP_COMMIT"
echo "    CUDAARCHS        : ${CUDAARCHS:-86}"
echo "    workdir          : $WORKDIR"
echo "    out-dir          : $OUT_DIR"
[[ -n "$LOCAL_DIR" ]] && echo "    local weights    : $LOCAL_DIR (skipping HF download)"
echo ""

mkdir -p "$WORKDIR" "$OUT_DIR"

# --- preflight: disk --------------------------------------------------------
avail_gb=$(df -BG "$WORKDIR" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
info "free disk on $WORKDIR filesystem: ${avail_gb} GB (need >= ${MIN_DISK_GB})"
[[ "${avail_gb:-0}" -ge "$MIN_DISK_GB" ]] || \
    die "insufficient disk: ${avail_gb} GB < ${MIN_DISK_GB} GB. Attach a larger volume or set MIN_DISK_GB."

# --- preflight: tools -------------------------------------------------------
command -v git    >/dev/null || die "git not found"
command -v cmake  >/dev/null || die "cmake not found"
command -v python3>/dev/null || die "python3 not found"
command -v nvcc   >/dev/null || info "WARNING: nvcc not on PATH — CUDA build of imatrix may fail (imatrix wants a GPU)"

# --- 1. build llama.cpp tools at the pinned commit --------------------------
# We build llama-imatrix + llama-quantize (and use the bundled convert script)
# from the SAME commit jaded links. A marker file skips the rebuild on reruns.
LLAMA_SRC="$WORKDIR/llama.cpp"
BUILT_MARKER="$LLAMA_SRC/.jade-built-$LLAMA_CPP_COMMIT"
LLAMA_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp}"

if [[ -f "$BUILT_MARKER" ]]; then
    ok "llama.cpp tools already built at $LLAMA_CPP_COMMIT (cached)"
else
    info "fetching llama.cpp @ $LLAMA_CPP_COMMIT"
    if [[ ! -d "$LLAMA_SRC/.git" ]]; then
        git init -q "$LLAMA_SRC"
        git -C "$LLAMA_SRC" remote add origin "$LLAMA_REPO"
    fi
    git -C "$LLAMA_SRC" fetch --depth 1 origin "$LLAMA_CPP_COMMIT"
    git -C "$LLAMA_SRC" checkout -q -f "$LLAMA_CPP_COMMIT"

    info "building llama-imatrix + llama-quantize (CUDA arch ${CUDAARCHS:-86})"
    cmake -S "$LLAMA_SRC" -B "$LLAMA_SRC/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDAARCHS:-86}" \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=OFF
    cmake --build "$LLAMA_SRC/build" --target llama-imatrix llama-quantize -j "$(nproc)"
    touch "$BUILT_MARKER"
    ok "llama.cpp tools built"
fi

IMATRIX_BIN="$LLAMA_SRC/build/bin/llama-imatrix"
QUANTIZE_BIN="$LLAMA_SRC/build/bin/llama-quantize"
CONVERT_PY="$LLAMA_SRC/convert_hf_to_gguf.py"
for f in "$IMATRIX_BIN" "$QUANTIZE_BIN" "$CONVERT_PY"; do
    [[ -e "$f" ]] || die "expected tool missing after build: $f"
done

# --- 2. python env for the converter ----------------------------------------
VENV="$WORKDIR/venv"
if [[ ! -x "$VENV/bin/python" ]]; then
    info "creating python venv for convert script"
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# Some distros ship python without a bundled pip/ensurepip (so `python -m venv`
# yields a pip-less venv). Bootstrap pip robustly before installing anything.
if ! python -m pip --version >/dev/null 2>&1; then
    info "venv has no pip — bootstrapping"
    python -m ensurepip --upgrade 2>/dev/null || \
        curl -fsSL https://bootstrap.pypa.io/get-pip.py | python
fi
info "installing convert + huggingface deps (pinned via llama.cpp's requirements)"
pip install -q --upgrade pip
pip install -q -r "$LLAMA_SRC/requirements/requirements-convert_hf_to_gguf.txt"
pip install -q "huggingface_hub[cli,hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- 3. obtain weights ------------------------------------------------------
if [[ -n "$LOCAL_DIR" ]]; then
    SRC_MODEL="$LOCAL_DIR"
    [[ -d "$SRC_MODEL" ]] || die "--local-dir not a directory: $SRC_MODEL"
    ok "using local weights: $SRC_MODEL"
else
    SRC_MODEL="$WORKDIR/src-model"
    info "downloading $HF_REPO @ $HF_REVISION -> $SRC_MODEL"
    # hf cli is idempotent: re-runs resume / skip already-present files.
    hf_args=(download "$HF_REPO" --revision "$HF_REVISION" --local-dir "$SRC_MODEL")
    [[ -n "${HF_TOKEN:-}" ]] && hf_args+=(--token "$HF_TOKEN")
    if command -v hf >/dev/null; then hf "${hf_args[@]}"; else huggingface-cli "${hf_args[@]}"; fi
    ok "weights present ($(du -sh "$SRC_MODEL" | cut -f1))"
fi

# --- 4. convert -> f16 GGUF -------------------------------------------------
F16="$WORKDIR/model-f16.gguf"
info "convert_hf_to_gguf.py -> $F16 (f16)"
python "$CONVERT_PY" "$SRC_MODEL" --outfile "$F16" --outtype f16
ok "f16 GGUF ($(du -sh "$F16" | cut -f1))"

# --- 5. importance matrix (code-calibrated) ---------------------------------
# The committed seed (ci/calib-code.txt) carries the deployment-specific signal
# — JadeLang, the messages protocol, the tool-call wire format. It is only a few
# chunks on its own, so unless disabled we deterministically augment it with real
# source from the PINNED llama.cpp checkout (diverse C/C++/Python/CUDA/CMake,
# fixed by LLAMA_CPP_COMMIT → reproducible, no repo bloat) to reach a solid chunk
# count. Seed goes first so its tokens are always covered. Disable with
# JADE_CALIB_AUGMENT=0 to calibrate on the committed file alone.
EFF_CALIB="$WORKDIR/calib-effective.txt"
cp -f "$CALIB" "$EFF_CALIB"
if [[ "${JADE_CALIB_AUGMENT:-1}" != "0" ]]; then
    # ~100 chunks * 512 tok * ~4 bytes/tok ≈ 200 KB; budget a bit over.
    CALIB_BUDGET_BYTES="${JADE_CALIB_BUDGET_BYTES:-300000}"
    info "augmenting calibration from pinned llama.cpp source (budget ${CALIB_BUDGET_BYTES} B)"
    # Deterministic: sorted file list, real code, stop at the byte budget.
    while IFS= read -r f; do
        [[ -f "$f" ]] || continue
        cat "$f" >> "$EFF_CALIB"
        printf '\n' >> "$EFF_CALIB"
        [[ "$(stat -c %s "$EFF_CALIB" 2>/dev/null || stat -f %z "$EFF_CALIB")" -ge "$CALIB_BUDGET_BYTES" ]] && break
    done < <(find "$LLAMA_SRC/src" "$LLAMA_SRC/common" "$LLAMA_SRC/ggml/src" \
                  -maxdepth 1 -type f \( -name '*.cpp' -o -name '*.c' -o -name '*.cu' -o -name '*.py' \) 2>/dev/null | sort)
fi
ok "effective calibration corpus: $(du -h "$EFF_CALIB" | cut -f1)"

IMAT="$WORKDIR/model.imatrix"
info "llama-imatrix on code corpus ($IMATRIX_OFFLOAD, chunks<=$IMATRIX_CHUNKS)"
# shellcheck disable=SC2086
"$IMATRIX_BIN" -m "$F16" -f "$EFF_CALIB" -o "$IMAT" --chunks "$IMATRIX_CHUNKS" $IMATRIX_OFFLOAD
[[ -s "$IMAT" ]] || die "imatrix not produced"
ok "imatrix ($(du -sh "$IMAT" | cut -f1))"

# --- 6. quantize -> Q4_K_M --------------------------------------------------
QOUT="$WORKDIR/model-$QUANT.gguf"
info "llama-quantize --imatrix -> $QUANT"
"$QUANTIZE_BIN" --imatrix "$IMAT" "$F16" "$QOUT" "$QUANT"
[[ -s "$QOUT" ]] || die "quantized model not produced"
ok "quantized ($(du -sh "$QOUT" | cut -f1))"

# --- 6b. GGUF sanity: valid v3 header + expected MoE arch -------------------
# Cheap structural gate before we ship. The real serve-time compatibility is
# already guaranteed by quantizing with jaded's exact llama.cpp commit; this
# just catches gross corruption / a wrong-arch convert.
info "verifying GGUF header"
python - "$QOUT" <<'PY'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
ver = int(r.fields["GGUF.version"].parts[-1][0]) if "GGUF.version" in r.fields else r.version
arch_f = r.fields.get("general.architecture")
arch = bytes(arch_f.parts[arch_f.data[0]]).decode() if arch_f else "?"
assert ver == 3, f"unexpected GGUF version {ver} (jaded expects v3)"
assert "qwen3" in arch, f"unexpected architecture {arch!r}"
print(f"    OK: GGUF v{ver}, architecture={arch}, tensors={len(r.tensors)}")
PY

# --- 7. finalize: canonical name + sidecar + buildinfo ----------------------
# Name the artifact model.gguf so the sha256 sidecar reads `<hash>  model.gguf`
# and `sha256sum -c model.gguf.sha256` works locally too. jade-os only needs
# field 1 of the sidecar regardless.
info "staging release artifacts in $OUT_DIR"
cp -f "$QOUT" "$OUT_DIR/model.gguf"
( cd "$OUT_DIR" && sha256sum model.gguf > model.gguf.sha256 )
HASH="$(awk '{print $1}' "$OUT_DIR/model.gguf.sha256")"

cat > "$OUT_DIR/model.gguf.buildinfo" <<EOF
model_sha256=$HASH
hf_repo=$HF_REPO
hf_revision=$HF_REVISION
quant=$QUANT
imatrix=code-calibrated (ci/calib-code.txt seed + pinned llama.cpp src, chunks<=$IMATRIX_CHUNKS, augment=${JADE_CALIB_AUGMENT:-1})
llama_cpp_repo=$LLAMA_REPO
llama_cpp_commit=$LLAMA_CPP_COMMIT
cuda_archs=${CUDAARCHS:-86}
size_bytes=$(stat -c %s "$OUT_DIR/model.gguf" 2>/dev/null || stat -f %z "$OUT_DIR/model.gguf")
EOF

ok "model.gguf      $(du -sh "$OUT_DIR/model.gguf" | cut -f1)  sha256=$HASH"
ok "model.gguf.sha256 + model.gguf.buildinfo written"
echo ""
cat "$OUT_DIR/model.gguf.buildinfo"

# --- 8. optional upload to the cross-repo S3 contract -----------------------
if [[ "$DO_UPLOAD" -eq 1 ]]; then
    # We run inside the build venv (activated above), so if no aws is on PATH we
    # install it there rather than depending on a system/monorepo aws.
    if ! command -v aws >/dev/null; then
        info "aws CLI not found — installing awscli into the build venv"
        pip install -q awscli || die "could not install awscli; install aws CLI or rerun without --upload"
    fi
    DEST="s3://${S3_BUCKET}/${S3_PREFIX}"
    info "uploading to $DEST/ (the jade-os contract)"
    aws s3 cp "$OUT_DIR/model.gguf"        "$DEST/model.gguf"        --no-progress
    aws s3 cp "$OUT_DIR/model.gguf.sha256" "$DEST/model.gguf.sha256" --no-progress
    ok "published $DEST/model.gguf (+ .sha256)"
fi

echo ""
ok "done. quantized with llama.cpp $LLAMA_CPP_COMMIT (matches jaded's linked llama-cpp-sys-2)."
