#!/usr/bin/env sh
# Author: Joe Ricks
# File: install.sh
# Description: Installs the prebuilt dovata and gguf-inspect programs for this machine
# from the latest GitHub release.
#
# Usage: curl -fsSL https://raw.githubusercontent.com/jadeinc/dovata/main/install.sh | sh
#
# DOVATA_INSTALL_DIR picks the directory, /usr/local/bin by default. The script uses
# sudo only when it cannot write there itself.

set -e

REPO="jadeinc/dovata"
INSTALL_DIR="${DOVATA_INSTALL_DIR:-/usr/local/bin}"
BINARIES="dovata gguf-inspect"

OS=$(uname -s)
ARCH=$(uname -m)

case "$OS" in
  Darwin)
    case "$ARCH" in
      arm64) ARCHIVE="dovata-macos-arm64.tar.gz" ;;
      *)
        echo "Unsupported macOS architecture: $ARCH"
        echo "dovata ships an Apple Silicon (arm64) build only; Intel macOS is not supported."
        echo "On an Intel Mac, run the arm64 build under Rosetta 2, or build from source:"
        echo "  https://github.com/$REPO"
        exit 1
        ;;
    esac
    ;;
  Linux)
    case "$ARCH" in
      x86_64|amd64) ARCHIVE="dovata-linux-x86_64.tar.gz" ;;
      *)
        echo "Unsupported Linux architecture: $ARCH"
        echo "Build from source: https://github.com/$REPO"
        exit 1
        ;;
    esac
    ;;
  *)
    echo "Unsupported OS: $OS"
    echo "dovata supports macOS and Linux only. On Windows, install inside WSL2."
    exit 1
    ;;
esac

# The latest release's tag, from the GitHub API.
TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
  | grep '"tag_name"' \
  | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')

if [ -z "$TAG" ]; then
  echo "Could not find the latest release. Check https://github.com/$REPO/releases"
  exit 1
fi

URL="https://github.com/$REPO/releases/download/$TAG/$ARCHIVE"
CHECKSUMS_URL="https://github.com/$REPO/releases/download/$TAG/checksums.txt"

echo "Installing dovata $TAG for $OS/$ARCH..."
echo "  From: $URL"
echo "  To:   $INSTALL_DIR"
echo ""

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Downloading checksums..."
curl -fsSL "$CHECKSUMS_URL" -o "$TMP_DIR/checksums.txt"
echo "Downloading archive..."
curl -fsSL --progress-bar "$URL" -o "$TMP_DIR/$ARCHIVE"

echo "Verifying checksum..."
case "$OS" in
  Darwin) SHASUM_CMD="shasum -a 256" ;;
  *)      SHASUM_CMD="sha256sum" ;;
esac
(cd "$TMP_DIR" && grep " $ARCHIVE\$" checksums.txt | $SHASUM_CMD -c -)

mkdir "$TMP_DIR/unpacked"
tar -xzf "$TMP_DIR/$ARCHIVE" -C "$TMP_DIR/unpacked"

# Purpose: This function moves one file into place, with sudo only when needed.
#
# Inputs:
# - $1: the file to move.
# - $2: where it goes.
#
# Outputs: none.
install_file() {
  src="$1"; dest="$2"; dest_dir="$(dirname "$dest")"
  if mkdir -p "$dest_dir" 2>/dev/null && [ -w "$dest_dir" ]; then
    mv "$src" "$dest"
  else
    sudo mkdir -p "$dest_dir"
    sudo mv "$src" "$dest"
  fi
}

for bin in $BINARIES; do
  if [ ! -f "$TMP_DIR/unpacked/$bin" ]; then
    echo "The archive has no $bin; check https://github.com/$REPO/releases/tag/$TAG"
    exit 1
  fi
  chmod +x "$TMP_DIR/unpacked/$bin"
  install_file "$TMP_DIR/unpacked/$bin" "$INSTALL_DIR/$bin"
done

echo ""
echo "$("$INSTALL_DIR/dovata" --version) installed to $INSTALL_DIR/dovata"
echo "gguf-inspect installed to $INSTALL_DIR/gguf-inspect"

# A different copy earlier in PATH would run instead of the one just installed, and
# the install would look like it did nothing.
for bin in $BINARIES; do
  RESOLVED=$(command -v "$bin" 2>/dev/null || true)
  if [ -n "$RESOLVED" ] && [ "$RESOLVED" != "$INSTALL_DIR/$bin" ]; then
    echo ""
    echo "Warning: another '$bin' is earlier in your PATH and will run instead of this one:"
    echo "           $RESOLVED"
    echo "         Remove it (rm '$RESOLVED') or re-run with DOVATA_INSTALL_DIR=$(dirname "$RESOLVED")."
  fi
done

echo ""
echo "dovata plan --quant runs llama.cpp's llama-quantize, which is a separate install;"
echo "on a Mac: brew install llama.cpp"
echo ""
echo "Run 'dovata --help' to get started."
