#!/usr/bin/env python3
"""Configure a remote provider's API key.

The anthropic/openai provider libraries resolve their API key from
~/.jade/providers/<name>.json (unless <PROVIDER>_API_KEY is set in the env). This
script writes that file — run it once when you install the provider .so package.

Usage:
  python3 src/scripts/configure-provider.py anthropic sk-ant-...   # key as an argument
  python3 src/scripts/configure-provider.py openai                 # prompt (hidden)
"""
import getpass
import json
import os
import sys
from pathlib import Path


def main(argv):
    name = argv[1] if len(argv) > 1 else ""
    if name not in ("anthropic", "openai"):
        print(f"usage: {argv[0]} <anthropic|openai> [api_key]", file=sys.stderr)
        return 2

    key = argv[2] if len(argv) > 2 else ""
    if not key:
        key = getpass.getpass(f"Enter the {name} API key: ")
    if not key:
        print("error: no API key provided", file=sys.stderr)
        return 1

    directory = Path.home() / ".jade" / "providers"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"

    # The file holds a secret — create it 0600 before writing the key into it.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"api_key": key}) + "\n")
    os.chmod(path, 0o600)
    print(f"wrote {path} (mode 600)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
