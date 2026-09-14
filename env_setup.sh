#!/usr/bin/env bash
# Install the curated GPU profile and validate it before benchmarking.
# Provider-specific vLLM environments are managed separately.

set -euo pipefail

BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$BASE_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
GPU_BACKEND="${GPU_BACKEND:-auto}"
CONFIG_ONLY=0
if [[ "${1:-}" == "--config-only" && $# -eq 1 ]]; then
  CONFIG_ONLY=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: bash env_setup.sh [--config-only] (optional GPU_BACKEND=auto|nvidia|amd)" >&2
  exit 2
fi

pick_python_bin() {
  local candidate
  if [[ -n "$PYTHON_BIN" ]]; then
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      command -v "$PYTHON_BIN"
      return 0
    fi
    echo "[ENV][ERROR] Requested PYTHON_BIN '$PYTHON_BIN' is not on PATH" >&2
    exit 1
  fi

  for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done

  echo "[ENV][ERROR] No supported Python interpreter found. Install python3.10, python3.11, or python3.12." >&2
  exit 1
}

ensure_supported_host() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[ENV][ERROR] This benchmark environment is only supported on Linux GPU hosts." >&2
    exit 1
  fi
}

ensure_supported_venv() {
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    return 0
  fi

  local version
  version="$("$VENV_DIR/bin/python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

  case "$version" in
    3.10|3.11|3.12) return 0 ;;
  esac

  echo "[ENV][ERROR] Existing $VENV_DIR uses unsupported Python $version." >&2
  echo "[ENV][ERROR] Remove or replace $VENV_DIR, then rerun env_setup.sh with Python 3.10-3.12." >&2
  exit 1
}

ensure_supported_host
PYTHON_BIN="$(pick_python_bin)"
ensure_supported_venv
if [[ "$CONFIG_ONLY" != "1" ]]; then
  GPU_BACKEND="$("$PYTHON_BIN" "$BASE_DIR/gpu_platform.py" detect-backend --backend "$GPU_BACKEND")"
  GPU_IDS="$("$PYTHON_BIN" "$BASE_DIR/gpu_platform.py" gpu-ids --backend "$GPU_BACKEND")"
  if [[ -z "$GPU_IDS" ]]; then
    echo "[ENV][ERROR] No responding $GPU_BACKEND GPU found; GPU stack installation cancelled" >&2
    exit 1
  fi
  echo "[ENV] Selected backend: $GPU_BACKEND"
fi

# Create venv if missing
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

if [[ "$CONFIG_ONLY" == "1" ]]; then
  python -m pip install 'PyYAML>=6.0'
  exit 0
fi

# Resolve and install the same concrete package profile used by run_all.sh.
# Existing experimental stacks are retained and validated by the resolver.
resolver_flags=(--backend "$GPU_BACKEND" --install --venv "$VENV_DIR")
if [[ "${ALLOW_UNVERIFIED_HOST:-0}" == "1" ]]; then
  resolver_flags+=(--allow-unverified-host)
fi
python "$BASE_DIR/runtime_resolver.py" "${resolver_flags[@]}"
echo "Environment ready."
