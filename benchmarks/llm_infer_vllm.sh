#!/usr/bin/env bash
set -euo pipefail
# activate venv if not already active
if [ -f ".venv/bin/activate" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
  source .venv/bin/activate
fi
python benchmarks/llm_infer_vllm.py --config config.yaml --warmup 5 --duration 30
