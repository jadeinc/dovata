#!/usr/bin/env bash
# JADE OS — Start EC2 inference instance and wait for SSH
#
# Usage:
#   bash scripts/ec2/start.sh
set -euo pipefail

info() { echo "==> $*"; }
ok()   { echo "    OK: $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_FILE="$ROOT/.env.ec2"
[[ -f "$ENV_FILE" ]] || die ".env.ec2 not found — copy from .env.ec2.example"
# shellcheck source=/dev/null
[[ -f "$ROOT/.env" ]] && set -a && source "$ROOT/.env" && set +a
set -a; source "$ENV_FILE"; set +a

# Prefer project-local aws from .venv, fall back to global
AWS="$ROOT/.venv/bin/aws"
if [[ ! -f "$AWS" ]]; then
    AWS=$(command -v aws 2>/dev/null || true)
fi
[[ -n "$AWS" ]] || die "aws CLI not found — run: bash scripts/setup-dev.sh"

info "Starting $JADE_INSTANCE_ID in $JADE_REGION..."
"$AWS" ec2 start-instances \
    --instance-ids "$JADE_INSTANCE_ID" \
    --region "$JADE_REGION" \
    --output text > /dev/null

info "Waiting for instance running..."
"$AWS" ec2 wait instance-running \
    --instance-ids "$JADE_INSTANCE_ID" \
    --region "$JADE_REGION"

PUBLIC_DNS=$("$AWS" ec2 describe-instances \
    --instance-ids "$JADE_INSTANCE_ID" \
    --region "$JADE_REGION" \
    --query 'Reservations[0].Instances[0].PublicDnsName' \
    --output text)

ok "Instance running: $PUBLIC_DNS"

info "Waiting for SSH to be ready..."
KEY="$ROOT/$JADE_SSH_KEY"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes"
for i in $(seq 1 36); do
    # shellcheck disable=SC2086
    if ssh $SSH_OPTS -i "$KEY" "$JADE_SSH_USER@$PUBLIC_DNS" true 2>/dev/null; then
        ok "SSH ready"
        echo ""
        echo "    host: $PUBLIC_DNS"
        echo "    ssh:  ssh -i $KEY $JADE_SSH_USER@$PUBLIC_DNS"
        echo ""
        echo "$PUBLIC_DNS" > "$ROOT/.ec2-host"
        exit 0
    fi
    echo "    waiting... ($i/36)"
    sleep 5
done

die "SSH not ready after 3 minutes — check security group inbound rule for port 22"
