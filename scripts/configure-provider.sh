#!/usr/bin/env bash
# Configure a remote provider's API key.
#
# The anthropic/openai provider libraries resolve their API key from
# ~/.jade/providers/<name>.json (unless <PROVIDER>_API_KEY is set in the env). This
# script writes that file — run it once when you install the provider .so package.
#
# Usage:
#   scripts/configure-provider.sh anthropic sk-ant-...     # key as an argument
#   scripts/configure-provider.sh openai                   # prompt for the key (hidden)
set -euo pipefail

NAME="${1:-}"
case "$NAME" in
    anthropic|openai) ;;
    *) echo "usage: $0 <anthropic|openai> [api_key]" >&2; exit 2 ;;
esac

KEY="${2:-}"
if [[ -z "$KEY" ]]; then
    read -r -s -p "Enter the ${NAME} API key: " KEY
    echo
fi
[[ -n "$KEY" ]] || { echo "error: no API key provided" >&2; exit 1; }

DIR="$HOME/.jade/providers"
FILE="$DIR/$NAME.json"
mkdir -p "$DIR"
# The file holds a secret — create it 0600 before writing the key into it.
umask 077
printf '{"api_key":"%s"}\n' "$KEY" > "$FILE"
chmod 600 "$FILE"
echo "wrote $FILE (mode 600)"
