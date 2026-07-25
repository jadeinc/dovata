#!/usr/bin/env bash
# JADE OS — Open an interactive SSH session on the EC2 instance
#
# Usage:
#   bash tests/ec2/ssh.sh           # interactive shell
#   bash tests/ec2/ssh.sh <cmd>     # run a single command and exit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_FILE="$ROOT/.env.ec2"
[[ -f "$ENV_FILE" ]] || { echo "ERROR: .env.ec2 not found" >&2; exit 1; }
[[ -f "$ROOT/.env" ]] && set -a && source "$ROOT/.env" && set +a
set -a; source "$ENV_FILE"; set +a

AWS="$ROOT/.venv/bin/aws"
[[ -f "$AWS" ]] || AWS=$(command -v aws 2>/dev/null || true)
[[ -n "$AWS" ]] || { echo "ERROR: aws CLI not found — run: bash scripts/setup-dev.sh" >&2; exit 1; }

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
[[ -n "$HOST" ]] || { echo "ERROR: could not resolve hostname — run ec2-start.sh first" >&2; exit 1; }

KEY="$ROOT/$JADE_SSH_KEY"
exec ssh -o StrictHostKeyChecking=no -i "$KEY" "$JADE_SSH_USER@$HOST" "$@"
