#!/usr/bin/env bash
# JADE OS — Local dev environment setup
#
# Creates a project-local Python venv and installs the AWS CLI into it.
# Nothing is installed or written globally.
#
# AWS credentials are read from environment variables — add to ~/.zshrc:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#
# Usage:
#   bash scripts/setup-dev.sh
set -euo pipefail

info() { echo "==> $*"; }
ok()   { echo "    OK: $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$ROOT/.venv"

command -v python3 &>/dev/null || die "python3 not found"

# ── Create venv ───────────────────────────────────────────────────────────────
if [[ ! -f "$VENV/bin/activate" ]]; then
    info "Creating .venv..."
    python3 -m venv "$VENV"
    ok ".venv created"
else
    ok ".venv already exists"
fi

# ── Install AWS CLI ───────────────────────────────────────────────────────────
if [[ ! -f "$VENV/bin/aws" ]]; then
    info "Installing AWS CLI into .venv..."
    "$VENV/bin/pip" install --quiet awscli
    ok "AWS CLI installed: $("$VENV/bin/aws" --version)"
else
    ok "AWS CLI already installed: $("$VENV/bin/aws" --version)"
fi

# ── Load credentials and config ───────────────────────────────────────────────
[[ -f "$ROOT/.env" ]] && set -a && source "$ROOT/.env" && set +a
source "$ROOT/.env.ec2"

[[ -n "${AWS_ACCESS_KEY_ID:-}"     ]] || die "AWS_ACCESS_KEY_ID not set — add to .env"
[[ -n "${AWS_SECRET_ACCESS_KEY:-}" ]] || die "AWS_SECRET_ACCESS_KEY not set — add to .env"

info "Verifying AWS credentials..."
if "$VENV/bin/aws" sts get-caller-identity --region "$JADE_REGION" --output text &>/dev/null; then
    ACCOUNT=$("$VENV/bin/aws" sts get-caller-identity --region "$JADE_REGION" --query Account --output text)
    ok "Credentials valid (account: $ACCOUNT)"
else
    die "Credentials invalid — check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
fi

echo ""
ok "Dev environment ready"
info "Run: bash tests/ec2/start.sh"
