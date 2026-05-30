#!/usr/bin/env bash
# Download the JADE OS inference model.
#
# Default: Llama-3.2-3B-Instruct-Q4_K_M (~2 GB, CPU-capable, Llama 3.x family)
# Downloads to models/model.gguf in the repo root (gitignored).
#
# Override at runtime:
#   MODEL_URL=<url>   MODEL_DEST=<path>   bash scripts/download-model.sh
#
# Usage:
#   bash scripts/download-model.sh    # → models/model.gguf
#   make model
#
# To run jade-tree against this local model:
#   JADE_MODEL_PATH=models/model.gguf cargo run -p jade-tree --features inference

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

info() { echo "==> $*"; }
ok()   { echo "    OK: $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

MODEL_DEST="${MODEL_DEST:-$REPO_ROOT/models/model.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf}"

info "JADE model download"
echo "    dest: $MODEL_DEST"
echo "    url:  $MODEL_URL"
echo ""

mkdir -p "$(dirname "$MODEL_DEST")"

if [[ -f "$MODEL_DEST" ]]; then
    ok "already exists ($(du -sh "$MODEL_DEST" | cut -f1)) — skipping"
    echo "    delete $MODEL_DEST to re-download"
    exit 0
fi

if command -v curl &>/dev/null; then
    curl -L --progress-bar -o "$MODEL_DEST" "$MODEL_URL"
elif command -v wget &>/dev/null; then
    wget -O "$MODEL_DEST" "$MODEL_URL"
else
    die "neither curl nor wget found — install one and retry"
fi

ok "$(du -sh "$MODEL_DEST" | cut -f1)  $MODEL_DEST"
echo ""
echo "    Dev:  JADE_MODEL_PATH=$MODEL_DEST cargo run -p jade-tree --features inference"
echo "    HW:   copy to /jade/inference/model.gguf on the target disk"
