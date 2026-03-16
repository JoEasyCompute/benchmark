#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[BLENDER][ERROR] install_blender.sh currently supports Linux hosts only." >&2
  exit 1
fi

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) BLENDER_ARCH="linux-x64" ;;
  *)
    echo "[BLENDER][ERROR] Unsupported architecture: $ARCH" >&2
    echo "[BLENDER][ERROR] Update install_blender.sh if you need a different Blender build." >&2
    exit 1
    ;;
esac

BLENDER_VERSION="${BLENDER_VERSION:-4.2.18}"
BLENDER_SERIES="${BLENDER_SERIES:-${BLENDER_VERSION%.*}}"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/.local/opt}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
CACHE_DIR="${CACHE_DIR:-${TMPDIR:-/tmp}}"
ARCHIVE_NAME="blender-${BLENDER_VERSION}-${BLENDER_ARCH}.tar.xz"
ARCHIVE_URL="https://download.blender.org/release/Blender${BLENDER_SERIES}/${ARCHIVE_NAME}"
ARCHIVE_PATH="${CACHE_DIR%/}/${ARCHIVE_NAME}"
TARGET_DIR="${INSTALL_ROOT%/}/blender-${BLENDER_VERSION}"

mkdir -p "$INSTALL_ROOT" "$BIN_DIR"

if command -v curl >/dev/null 2>&1; then
  DOWNLOADER=(curl -fL --retry 3 -o "$ARCHIVE_PATH" "$ARCHIVE_URL")
elif command -v wget >/dev/null 2>&1; then
  DOWNLOADER=(wget -O "$ARCHIVE_PATH" "$ARCHIVE_URL")
else
  echo "[BLENDER][ERROR] Need curl or wget to download Blender." >&2
  exit 1
fi

echo "[BLENDER] Downloading $ARCHIVE_URL"
"${DOWNLOADER[@]}"

rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"
tar -xJf "$ARCHIVE_PATH" -C "$INSTALL_ROOT"

EXTRACTED_DIR="${INSTALL_ROOT%/}/blender-${BLENDER_VERSION}-${BLENDER_ARCH}"
if [[ ! -d "$EXTRACTED_DIR" ]]; then
  echo "[BLENDER][ERROR] Expected extracted directory not found: $EXTRACTED_DIR" >&2
  exit 1
fi

rm -rf "$TARGET_DIR"
mv "$EXTRACTED_DIR" "$TARGET_DIR"
ln -sfn "$TARGET_DIR/blender" "$BIN_DIR/blender"

echo "[BLENDER] Installed Blender $BLENDER_VERSION"
echo "[BLENDER] Binary: $BIN_DIR/blender"
echo "[BLENDER] Version check:"
"$BIN_DIR/blender" --version | head -n1
