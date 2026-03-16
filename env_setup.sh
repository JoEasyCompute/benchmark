#!/usr/bin/env bash
# Idempotent env setup for vLLM + Stable Diffusion benches (CUDA 12.8 wheels)
# - vLLM 0.11.0 stack: torch 2.8.0 + xformers 0.0.32.post1 + setuptools <80
# - SD stack pinned to avoid offload_state_dict issues: diffusers 0.29.2
# - Transformers kept compatible with vLLM + tokenizers 0.22.x

set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
GPU_BACKEND="${GPU_BACKEND:-auto}"

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
if [[ "$GPU_BACKEND" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_BACKEND="nvidia"
  elif command -v rocm-smi >/dev/null 2>&1; then
    GPU_BACKEND="amd"
  else
    echo "[ENV][ERROR] Could not detect a supported GPU backend. Expected nvidia-smi or rocm-smi." >&2
    exit 1
  fi
fi

# Create venv if missing
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -c 'import sys; print("Python:", sys.version)'

# Base tooling
pip install --upgrade pip wheel

# setuptools pinned for vLLM compatibility
pip install "setuptools<80,>=77.0.3"

# Torch stack
if [[ "$GPU_BACKEND" == "amd" ]]; then
  pip install --extra-index-url https://download.pytorch.org/whl/rocm6.3 \
    "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0"
else
  pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0"
fi

# Optional acceleration stack
if [[ "$GPU_BACKEND" == "nvidia" ]]; then
  pip install \
    "vllm==0.11.0" \
    "xformers==0.0.32.post1"
else
  pip uninstall -y vllm xformers >/dev/null 2>&1 || true
  echo "[ENV][WARN] Skipping default vLLM install on AMD; llm_infer_vllm requires a separately validated ROCm-compatible vLLM build."
  echo "[ENV][WARN] Skipping xformers install on AMD; Stable Diffusion will run without it unless you install a compatible build manually."
fi

# Stable Diffusion trio (pins that avoid CLIP offload kw issues)
# --no-deps prevents pulling mismatched transitive deps, so install
# the shared requirements explicitly afterward.
pip install --upgrade --no-deps \
  "diffusers==0.29.2" \
  "transformers==4.57.0" \
  "accelerate==1.10.1"
pip install --upgrade \
  "huggingface-hub>=0.34.0,<1.0" \
  "importlib-metadata>=6.0" \
  "tokenizers>=0.22.0,<=0.23.0" \
  "regex!=2019.12.17" \
  "psutil>=5.9.8"

# Bench deps
if [[ "$GPU_BACKEND" == "nvidia" ]]; then
  pip install "safetensors>=0.4.3" "pandas>=2.2.0" "tqdm>=4.66" "pyyaml>=6.0" "nvidia-ml-py>=12.560.30"
else
  pip install "safetensors>=0.4.3" "pandas>=2.2.0" "tqdm>=4.66" "pyyaml>=6.0"
fi

echo "---- versions ----"
python - <<'PY'
import os
import torch, torchvision, torchaudio, setuptools
import transformers, diffusers, accelerate
import tokenizers
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torchaudio", torchaudio.__version__)
print("setuptools", setuptools.__version__)
print("transformers", transformers.__version__)
print("tokenizers", tokenizers.__version__)
print("diffusers", diffusers.__version__)
print("accelerate", accelerate.__version__)
try:
    import xformers
    print("xformers", xformers.__version__)
except Exception:
    print("xformers", None)
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception:
    print("vllm", None)
try:
    import pynvml
    print("pynvml", pynvml.__version__)
except Exception:
    print("pynvml", None)
PY

# This will warn if anything is still mismatched (ok to continue if warnings appear for optional extras)
pip check || true

echo "Environment ready."
