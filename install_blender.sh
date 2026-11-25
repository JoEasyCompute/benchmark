#!/usr/bin/env bash
set -euo pipefail

BLENDER_VERSION_DEFAULT=5.3.0
ENSURE=0
INSTALL_SYSTEM_DEPS=0
for arg in "$@"; do
  case "$arg" in
    --ensure) ENSURE=1 ;;
    --install-system-deps) INSTALL_SYSTEM_DEPS=1 ;;
    *) echo "Usage: bash install_blender.sh [--ensure] [--install-system-deps]" >&2; exit 2 ;;
  esac
done

if [[ "$ENSURE" == "1" ]]; then
  BLENDER_VERSION="$BLENDER_VERSION_DEFAULT"
  BLENDER_SERIES="${BLENDER_VERSION%.*}"
  # Reserve stdout for the verified executable path.
  exec 3>&1 1>&2
else
  BLENDER_VERSION="${BLENDER_VERSION:-$BLENDER_VERSION_DEFAULT}"
  BLENDER_SERIES="${BLENDER_SERIES:-${BLENDER_VERSION%.*}}"
fi
if [[ ! "$BLENDER_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ || ! "$BLENDER_SERIES" =~ ^[0-9]+\.[0-9]+$ ]]; then
  echo "[BLENDER][ERROR] Use a three-part Blender version and a major.minor release series." >&2
  exit 2
fi
version_matches() {
  local output
  output="$("$1" --version 2>&1)" || { printf '%s\n' "$output" >&2; return 1; }
  local title="${output%%$'\n'*}"
  if [[ "$title" != "Blender $BLENDER_VERSION" ]]; then
    echo "[BLENDER][ERROR] Expected Blender $BLENDER_VERSION; found $title at $1" >&2
    return 1
  fi
  printf '%s\n' "$title"
}

if [[ "$INSTALL_SYSTEM_DEPS" == "1" ]]; then
  [[ "$(uname -s)" == "Linux" ]] && command -v apt-get >/dev/null && command -v sudo >/dev/null ||
    { echo "[BLENDER][ERROR] --install-system-deps requires Linux, apt-get, and sudo." >&2; exit 1; }
  sudo apt-get update
  sudo apt-get install -y libsm6 libxrender1 libxi6 libxfixes3 libxkbcommon0 libgl1
fi

if [[ "$ENSURE" == "1" ]]; then
  existing="$(command -v blender || true)"
  if [[ -n "$existing" ]] && version_matches "$existing"; then
    existing="$(cd -- "$(dirname -- "$existing")" && pwd)/$(basename -- "$existing")"
    printf '%s\n' "$existing" >&3
    exit 0
  fi
fi

if [[ "$(uname -s)" != "Linux" || ! "$(uname -m)" =~ ^(x86_64|amd64)$ ]]; then
  echo "[BLENDER][ERROR] Automatic installation requires Linux x86-64." >&2
  exit 1
fi

INSTALL_ROOT="${INSTALL_ROOT:-$HOME/.local/opt}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
CACHE_DIR="${CACHE_DIR:-${TMPDIR:-/tmp}}"
mkdir -p "$INSTALL_ROOT" "$BIN_DIR" "$CACHE_DIR"
INSTALL_ROOT="$(cd -- "$INSTALL_ROOT" && pwd)"
BIN_DIR="$(cd -- "$BIN_DIR" && pwd)"
TARGET_DIR="$INSTALL_ROOT/blender-$BLENDER_VERSION"
LAUNCHER="$BIN_DIR/blender"

if [[ -d "$LAUNCHER" ]]; then
  echo "[BLENDER][ERROR] Launcher path is a directory: $LAUNCHER" >&2
  exit 1
fi

stage="$(mktemp -d "$INSTALL_ROOT/.blender-stage.XXXXXX")"
downloads=""
wrapper=""
rollback=0
cleanup() {
  if [[ "$rollback" == "1" && -e "$stage/previous" ]]; then
    mv "$stage/previous" "$TARGET_DIR"
  fi
  [[ -z "$wrapper" ]] || rm -f -- "$wrapper"
  [[ -z "$downloads" ]] || rm -rf -- "$downloads"
  rm -rf -- "$stage"
}
trap cleanup EXIT

# Validate the staged executable with its own bundled libraries before switching.
write_wrapper() {
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$1/lib"
    printf 'exec %q "$@"\n' "$1/blender"
  } > "$2"
  chmod 0755 "$2"
}

write_wrapper "$TARGET_DIR" "$stage/current"
if [[ ! -x "$TARGET_DIR/blender" ]] || ! version_matches "$stage/current"; then
  downloads="$(mktemp -d "$CACHE_DIR/blender-download.XXXXXX")"
  archive="blender-$BLENDER_VERSION-linux-x64.tar.xz"
  manifest="blender-$BLENDER_VERSION.sha256"
  release_url="https://download.blender.org/release/Blender$BLENDER_SERIES"
  if command -v curl >/dev/null; then
    download=(curl -fsSL --retry 3 -o)
  elif command -v wget >/dev/null; then
    download=(wget -O)
  else
    echo "[BLENDER][ERROR] Need curl or wget to download Blender." >&2
    exit 1
  fi
  echo "[BLENDER] Downloading $release_url/$archive"
  "${download[@]}" "$downloads/$manifest" "$release_url/$manifest"
  digest="$(awk -v file="$archive" '$2 == file || $2 == "*" file {print $1}' "$downloads/$manifest")"
  if [[ ! "$digest" =~ ^[[:xdigit:]]{64}$ ]]; then
    echo "[BLENDER][ERROR] Official checksum for $archive is missing or invalid." >&2
    exit 1
  fi
  "${download[@]}" "$downloads/$archive" "$release_url/$archive"
  (cd "$downloads" && printf '%s  %s\n' "$digest" "$archive" | sha256sum -c -)
  tar -xJf "$downloads/$archive" -C "$stage"
  extracted="$stage/${archive%.tar.xz}"
  write_wrapper "$extracted" "$stage/candidate"
  echo "[BLENDER] Version check:"
  if ! version_matches "$stage/candidate"; then
    echo "[BLENDER][ERROR] Blender failed verification. For missing Ubuntu/Debian libraries, run:" >&2
    echo "  bash install_blender.sh --install-system-deps" >&2
    exit 1
  fi

  if [[ -e "$TARGET_DIR" || -L "$TARGET_DIR" ]]; then
    mv "$TARGET_DIR" "$stage/previous"
    rollback=1
  fi
  mv "$extracted" "$TARGET_DIR"
  rollback=0
fi

wrapper="$(mktemp "$BIN_DIR/.blender-wrapper.XXXXXX")"
write_wrapper "$TARGET_DIR" "$wrapper"
mv -f -- "$wrapper" "$LAUNCHER"
echo "[BLENDER] Installed Blender $BLENDER_VERSION"
echo "[BLENDER] Binary: $LAUNCHER"
if [[ "$ENSURE" == "1" ]]; then
  printf '%s\n' "$LAUNCHER" >&3
fi
