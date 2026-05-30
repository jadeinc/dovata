#!/usr/bin/env bash
# JADE OS — Stop EC2 inference instance
#
# Usage:
#   bash scripts/ec2/stop.sh
set -euo pipefail

info() { echo "==> $*"; }
ok()   { echo "    OK: $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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

info "Stopping $JADE_INSTANCE_ID..."
"$AWS" ec2 stop-instances \
    --instance-ids "$JADE_INSTANCE_ID" \
    --region "$JADE_REGION" \
    --output text > /dev/null

rm -f "$ROOT/.ec2-host"
ok "Stop initiated — instance will be stopped in ~30s"
