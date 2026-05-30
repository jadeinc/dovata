#!/usr/bin/env bash
# JADE OS — Run inference integration tests
#
# Designed to run on the EC2 instance, either directly or via deploy.sh over SSH.
# REPO is derived from the script's own location so it works regardless of $HOME.
#
# Usage:
#   bash scripts/ec2/run-tests.sh           # binaries must already be built
#   bash scripts/ec2/run-tests.sh --build   # cargo-build jadec + jaded first
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

info() { echo "==> $*"; }
ok()   { echo "    OK: $*"; }
fail() { echo "    FAIL: $*"; FAILURES=$((FAILURES + 1)); }
die()  { echo "ERROR: $*" >&2; cleanup; exit 1; }

BUILD=0
for arg in "$@"; do
    case "$arg" in
        --build) BUILD=1 ;;
    esac
done

JADE="/usr/bin/jade"
JADED="$REPO/target/release/jaded"
MODEL_PATH="${JADE_MODEL_PATH:-/jade/inference/model.gguf}"
TESTS_DIR="$REPO/tests/ec2-inference"
JADED_PID=""
FAILURES=0

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

cleanup() {
    if [[ -n "$JADED_PID" ]] && kill -0 "$JADED_PID" 2>/dev/null; then
        info "Stopping jaded (pid $JADED_PID)..."
        kill "$JADED_PID"
        wait "$JADED_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Optional build ────────────────────────────────────────────────────────────
if [[ "$BUILD" -eq 1 ]]; then
    info "Building jadec..."
    cargo build --release -p jade 2>&1 | tail -3
    info "Building jaded..."
    cargo build --release -p jaded --features inference 2>&1 | tail -3
    ok "Build done"
fi

# ── Preflight ─────────────────────────────────────────────────────────────────
[[ -x "$JADE" ]]        || die "jade not found at $JADE — run: bash scripts/ec2/deploy.sh"
[[ -f "$JADED" ]]   || die "jaded not found: $JADED"
[[ -f "$MODEL_PATH" ]]  || die "model not found: $MODEL_PATH (run ec2-first-run.sh first)"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | \
    while IFS= read -r line; do ok "GPU: $line"; done || true

# ── Start jaded ───────────────────────────────────────────────────────────
# Kill any existing jaded first — a running repl instance holds the socket
# path; starting a second jaded clobbers it, leaving the original with an
# unlinked socket fd (ECONNREFUSED on next connect).
if pgrep -x jaded > /dev/null 2>&1; then
    info "Stopping existing jaded before test run..."
    pkill -x jaded 2>/dev/null || true
    sleep 2
fi
mkdir -p "$HOME/.jade"
rm -f "$HOME/.jade/llm.sock"
info "Starting jaded (model: $MODEL_PATH)..."
JADE_MODEL_PATH="$MODEL_PATH" "$JADED" > /tmp/jaded.log 2>&1 &
JADED_PID=$!

info "Waiting for jaded to load model..."
for i in $(seq 1 60); do
    if grep -q "jaded: listening on" /tmp/jaded.log 2>/dev/null; then
        ok "jaded ready"
        break
    fi
    if ! kill -0 "$JADED_PID" 2>/dev/null; then
        echo "jaded log:"
        cat /tmp/jaded.log
        die "jaded exited unexpectedly"
    fi
    sleep 2
done

# ── Run test suite ────────────────────────────────────────────────────────────
info "Running test suite: $TESTS_DIR"
echo ""

PASS=0
for test_file in "$TESTS_DIR"/*.jde; do
    [[ -f "$test_file" ]] || continue
    name=$(basename "$test_file" .jde)
    output=$("$JADE" run "$test_file" 2>&1) || true

    pass_line=$(echo "$output" | grep '^PASS:' | head -1)
    if [[ -n "$pass_line" ]]; then
        ok "${pass_line#PASS: }"
        PASS=$((PASS + 1))
    else
        # Show the most informative line of failure output (last non-empty line)
        diag=$(echo "$output" | grep -v '^$' | tail -1)
        fail "$name — ${diag:-no output}"
    fi
done

echo ""
info "Results: $PASS passed, $FAILURES failed"
[[ "$FAILURES" -eq 0 ]] || { echo ""; die "$FAILURES test(s) failed"; }
ok "All tests passed"
