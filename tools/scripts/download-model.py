#!/usr/bin/env python3
"""Download the JADE OS inference model.

Default: Llama-3.2-3B-Instruct-Q4_K_M (~2 GB, CPU-capable, Llama 3.x family)
Downloads to models/model.gguf in the repo root (gitignored).

Override at runtime:
  MODEL_URL=<url>   MODEL_DEST=<path>   python3 tools/scripts/download-model.py

Usage:
  python3 tools/scripts/download-model.py    # -> models/model.gguf

To run jade-tree against this local model:
  JADE_MODEL_PATH=models/model.gguf cargo run -p jade-tree --features inference
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import info, ok, die, have, du_sh, run

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_URL = (
    "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/"
    "resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)


def main():
    dest = Path(os.environ.get("MODEL_DEST", REPO_ROOT / "models" / "model.gguf"))
    url = os.environ.get("MODEL_URL", DEFAULT_URL)

    info("JADE model download")
    print(f"    dest: {dest}")
    print(f"    url:  {url}")
    print()

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file():
        ok(f"already exists ({du_sh(dest)}) — skipping")
        print(f"    delete {dest} to re-download")
        return 0

    if have("curl"):
        run(["curl", "-L", "--progress-bar", "-o", str(dest), url])
    elif have("wget"):
        run(["wget", "-O", str(dest), url])
    else:
        die("neither curl nor wget found — install one and retry")

    ok(f"{du_sh(dest)}  {dest}")
    print()
    print(f"    Dev:  JADE_MODEL_PATH={dest} cargo run -p jade-tree --features inference")
    print("    HW:   copy to /jade/inference/model.gguf on the target disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
