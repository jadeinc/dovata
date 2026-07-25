#!/usr/bin/env bash
# JADE OS — Interactive jade repl on EC2 with live LLM inference
#
# jade-tree is started by deploy.sh and stays running persistently.
# This script just opens an SSH PTY and execs into `jade repl`.
#
# Usage: bash tests/ec2/repl.sh
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

echo "==> Connecting to $HOST ..."

ssh -t -o StrictHostKeyChecking=no -i "$KEY" "$JADE_SSH_USER@$HOST" "
set -e

JADE=\"/usr/bin/jade\"
SOCK=\"\$HOME/.jade/llm.sock\"

[[ -x \"\$JADE\" ]] || { echo 'ERROR: jade not installed — run: bash tests/ec2/deploy.sh'; exit 1; }

if [[ ! -S \"\$SOCK\" ]]; then
    echo 'ERROR: jade-tree not running — run: bash tests/ec2/deploy.sh'
    exit 1
fi

echo ''
echo 'JADE REPL  (Qwen3-Coder-30B-A3B-Instruct-Q4_K_M  •  live inference via Unix socket)'
echo 'Try:  prompt p = \"What is Rust?\"   then: ?p'
echo 'Exit: Ctrl-D'
echo ''
exec \"\$JADE\" repl
"
