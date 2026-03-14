#!/usr/bin/env bash
set -Eeuo pipefail

# --- Setup ---
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$BASE_DIR/benchmarks"
VENV_DIR="$BASE_DIR/.venv"
ENV_SETUP_SCRIPT="$BASE_DIR/env_setup.sh"
BASE_CONFIG_PATH="$BASE_DIR/config.yaml"
CONFIG_UTILS="$BASE_DIR/config_utils.py"
SMOKE_MODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      SMOKE_MODE=1
      shift
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

ensure_python_env() {
  local py_bin="$VENV_DIR/bin/python"
  if [[ ! -x "$py_bin" ]]; then
    echo "[SETUP] Python environment missing; running env_setup.sh"
    bash "$ENV_SETUP_SCRIPT"
    return
  fi

  if ! "$py_bin" - <<'PY' >/dev/null 2>&1
import accelerate
import diffusers
import pandas
import pynvml
import safetensors
import torch
import transformers
import tqdm
import vllm
import xformers
import yaml
PY
  then
    echo "[SETUP] Python environment incomplete; running env_setup.sh"
    bash "$ENV_SETUP_SCRIPT"
  fi
}

ensure_python_env

# Activate repo venv
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python3 "$BASE_DIR/validate_config.py" --config "$BASE_CONFIG_PATH"
MACHINE_STATE_STRICT="$(python3 "$CONFIG_UTILS" get --config "$BASE_CONFIG_PATH" --path preflight.machine_state_strict --default 'false' --format bool-int)"

# Determine results root from config.yaml (fallback: results)
RESULTS_ROOT="$(python3 "$CONFIG_UTILS" get --config "$BASE_CONFIG_PATH" --path results_dir --default '"results"' --format text)"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
REPEAT_COUNT="$(python3 "$CONFIG_UTILS" get --config "$BASE_CONFIG_PATH" --path repeat --default '1' --format text)"
REPEAT_COUNT="${REPEAT_COUNT:-1}"
readarray -t GPU_INCLUDE_VALUES < <(python3 "$CONFIG_UTILS" get --config "$BASE_CONFIG_PATH" --path gpu_include --default '[]' --format lines)

detect_all_gpu_ids() {
  python3 - <<'PY'
import subprocess

try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except Exception:
    out = ""

for line in out.splitlines():
    line = line.strip()
    if line:
        print(line)
PY
}

readarray -t ALL_GPU_IDS < <(detect_all_gpu_ids)
SELECTED_GPU_IDS=("${ALL_GPU_IDS[@]}")
if [[ "${#GPU_INCLUDE_VALUES[@]}" -gt 0 ]]; then
  SELECTED_GPU_IDS=("${GPU_INCLUDE_VALUES[@]}")
fi
if [[ "${#SELECTED_GPU_IDS[@]}" -gt 0 ]]; then
  CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${SELECTED_GPU_IDS[*]}")"
  export CUDA_VISIBLE_DEVICES
  echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
VISIBLE_GPU_COUNT="${#SELECTED_GPU_IDS[@]}"
if [[ "$VISIBLE_GPU_COUNT" -eq 0 ]]; then
  VISIBLE_GPU_COUNT=0
fi

# Build a unique run directory
# --- Build a unique, length-safe run directory ---
HOST="$(hostname -s)"
DATE="$(date +%Y%m%d_%H%M%S)"

# GPU count + first model name
GPU_COUNT="${VISIBLE_GPU_COUNT}"
GPU_MODEL="$(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo '')"

# Compact the model name aggressively:
# - strip "NVIDIA", "GeForce", "RTX", "Ada Generation", "Graphics"
# - keep only [A-Za-z0-9] and spaces
# - remove spaces, e.g. "RTX 6000 Ada Generation" -> "6000Ada"
SHORT_MODEL="$(
  echo "$GPU_MODEL" \
  | sed -E 's/\bNVIDIA\b//g;s/\bGeForce\b//g;s/\bRTX\b//g;s/\bAda Generation\b//g;s/\bGraphics\b//g' \
  | tr -cd '[:alnum:] ' \
  | awk '{$1=$1; gsub(/ /,""); print}'
)"

# Fallback if empty
if [[ -z "$SHORT_MODEL" ]]; then
  SHORT_MODEL="GPU"
fi

# Compose "<N>x<Model>", then truncate to 30 chars to avoid long paths
GPU_TAG="${GPU_COUNT}x${SHORT_MODEL}"
GPU_TAG="${GPU_TAG:0:30}"

# Final run id
RUN_ID="${DATE}_${HOST}_${GPU_TAG}"
RUN_DIR="$BASE_DIR/${RESULTS_ROOT}/${RUN_ID}"
mkdir -p "$RUN_DIR"/{logs,results}
echo "[INFO] Run folder: $RUN_DIR"

RUN_CONFIG_PATH="$RUN_DIR/effective_config.yaml"
if [[ "$SMOKE_MODE" == "1" ]]; then
  python3 "$CONFIG_UTILS" write-effective --config "$BASE_CONFIG_PATH" --output "$RUN_CONFIG_PATH" --smoke
else
  python3 "$CONFIG_UTILS" write-effective --config "$BASE_CONFIG_PATH" --output "$RUN_CONFIG_PATH"
fi
echo "[INFO] Effective config: $RUN_CONFIG_PATH"
if [[ "$SMOKE_MODE" == "1" ]]; then
  echo "[INFO] Smoke mode enabled"
fi
REPEAT_COUNT="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path repeat --default '1' --format text)"
LLM_INFER_WARMUP_S=5
LLM_INFER_DURATION_S=30
SD_ITERATIONS=5
if [[ "$SMOKE_MODE" == "1" ]]; then
  LLM_INFER_WARMUP_S=1
  LLM_INFER_DURATION_S=2
  SD_ITERATIONS=1
fi

python3 "$BASE_DIR/estimate_runtime.py" --config "$RUN_CONFIG_PATH" --json-out "$RUN_DIR/runtime_estimate.json"
python3 "$BASE_DIR/check_system_requirements.py" --config "$RUN_CONFIG_PATH" --json-out "$RUN_DIR/system_requirements.json"
if [[ "$MACHINE_STATE_STRICT" == "1" ]]; then
  python3 "$BASE_DIR/check_machine_state.py" --strict --json-out "$RUN_DIR/machine_state.json"
else
  python3 "$BASE_DIR/check_machine_state.py" --json-out "$RUN_DIR/machine_state.json"
fi

# --- System metadata snapshot ---
meta_file="$RUN_DIR/meta.json"
python3 - <<'PY' >"$meta_file"
import json, os, subprocess, platform, time
from importlib.metadata import PackageNotFoundError, version

def cmd(x):
    try:
        return subprocess.check_output(x, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""

def pkg(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None

gpu = cmd("nvidia-smi --query-gpu=index,name,driver_version,pstate,temperature.gpu,power.draw --format=csv,noheader")
cpu = cmd("lscpu")
mem = cmd("free -h")
osrel = cmd("cat /etc/os-release")
blender_version = cmd("blender --version | head -n1")
data = {
    "ts": int(time.time()),
    "platform": platform.platform(),
    "kernel": platform.release(),
    "python": platform.python_version(),
    "gpu_smi": gpu,
    "cpu_lscpu": cpu,
    "mem_free": mem,
    "os_release": osrel,
    "software_versions": {
        "torch": pkg("torch"),
        "torchvision": pkg("torchvision"),
        "torchaudio": pkg("torchaudio"),
        "vllm": pkg("vllm"),
        "xformers": pkg("xformers"),
        "diffusers": pkg("diffusers"),
        "transformers": pkg("transformers"),
        "accelerate": pkg("accelerate"),
        "tokenizers": pkg("tokenizers"),
        "pyyaml": pkg("PyYAML"),
        "pandas": pkg("pandas"),
        "safetensors": pkg("safetensors"),
        "blender": blender_version or None,
        "nvidia_smi": cmd("nvidia-smi --version | head -n1"),
    },
}
print(json.dumps(data, indent=2))
PY
echo "[INFO] Wrote meta → $meta_file"

# Helper to run a command in RUN_DIR and tee log
run_and_log () {
  local name="$1"; shift
  local logfile="$RUN_DIR/logs/${name}.log"
  echo "[RUN] $name → $logfile"
  ( set -o pipefail; (cd "$RUN_DIR" && stdbuf -oL -eL "$@") 2>&1 | tee "$logfile" )
}

jsonl_line_count () {
  local fp="$1"
  if [[ -f "$fp" ]]; then
    wc -l < "$fp" | tr -d ' '
  else
    echo 0
  fi
}

annotate_jsonl_rows () {
  local fp="$1" start_line="$2" suite="$3" repeat_index="$4" repeat_count="$5"
  python3 - "$fp" "$start_line" "$suite" "$repeat_index" "$repeat_count" <<'PY'
import json, pathlib, sys

fp = pathlib.Path(sys.argv[1])
start_line = int(sys.argv[2])
suite = sys.argv[3]
repeat_index = int(sys.argv[4])
repeat_count = int(sys.argv[5])

if not fp.exists():
    raise SystemExit(0)

lines = fp.read_text().splitlines()
for idx in range(start_line, len(lines)):
    if not lines[idx].strip():
        continue
    row = json.loads(lines[idx])
    row.setdefault("suite", suite)
    row["repeat_index"] = repeat_index
    row["repeat_count"] = repeat_count
    lines[idx] = json.dumps(row)

fp.write_text("".join(line + "\n" for line in lines))
PY
}

append_jsonl_row () {
  local fp="$1" payload="$2"
  mkdir -p "$(dirname "$fp")"
  printf '%s\n' "$payload" >> "$fp"
}

# --- 1) LLM Training ---
readarray -t TRAIN_WORLD_SIZES < <(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path llm_train.world_sizes --default '[1]' --format lines)
for rep in $(seq 1 "$REPEAT_COUNT"); do
  for world_size in "${TRAIN_WORLD_SIZES[@]}"; do
    if [[ "$world_size" -gt "$VISIBLE_GPU_COUNT" ]]; then
      echo "[SKIP] llm_train world_size=$world_size exceeds visible GPUs ($VISIBLE_GPU_COUNT)"
      append_jsonl_row "$RUN_DIR/results/metrics.jsonl" "$(python3 - <<'PY' "$world_size" "$VISIBLE_GPU_COUNT" "$rep" "$REPEAT_COUNT"
import json
import sys

world_size = int(sys.argv[1])
visible_gpu_count = int(sys.argv[2])
repeat_index = int(sys.argv[3])
repeat_count = int(sys.argv[4])

print(json.dumps({
    "benchmark_schema_version": 2,
    "suite": "llm_train",
    "status": "skipped",
    "skip_reason": "insufficient_visible_gpus",
    "requested_world_size": world_size,
    "visible_gpu_count": visible_gpu_count,
    "repeat_index": repeat_index,
    "repeat_count": repeat_count,
}))
PY
)"
      continue
    fi

    train_gpu_ids=("${SELECTED_GPU_IDS[@]:0:$world_size}")
    train_gpu_csv="$(IFS=,; echo "${train_gpu_ids[*]}")"
    start_line="$(jsonl_line_count "$RUN_DIR/results/metrics.jsonl")"
    if [[ "$world_size" -gt 1 ]]; then
      run_and_log "llm_train_ws${world_size}_r${rep}" env CUDA_VISIBLE_DEVICES="$train_gpu_csv" \
        python3 -m torch.distributed.run --standalone --nproc_per_node "$world_size" \
        "$BENCH_DIR/llm_train.py" --config "$RUN_CONFIG_PATH"
    else
      run_and_log "llm_train_ws${world_size}_r${rep}" env CUDA_VISIBLE_DEVICES="$train_gpu_csv" \
        python3 "$BENCH_DIR/llm_train.py" --config "$RUN_CONFIG_PATH"
    fi
    annotate_jsonl_rows "$RUN_DIR/results/metrics.jsonl" "$start_line" "llm_train" "$rep" "$REPEAT_COUNT"
  done
done

LLM_TRAIN_REAL_ENABLED="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path llm_train_real.enabled --default 'false' --format bool-int)"
if [[ "$LLM_TRAIN_REAL_ENABLED" == "1" ]]; then
  for rep in $(seq 1 "$REPEAT_COUNT"); do
    start_line="$(jsonl_line_count "$RUN_DIR/results/metrics.jsonl")"
    run_and_log "llm_train_real_r${rep}" python3 "$BENCH_DIR/llm_train_real.py" --config "$RUN_CONFIG_PATH"
    annotate_jsonl_rows "$RUN_DIR/results/metrics.jsonl" "$start_line" "llm_train_real" "$rep" "$REPEAT_COUNT"
  done
fi

# --- 2) LLM Inference (vLLM) ---
for rep in $(seq 1 "$REPEAT_COUNT"); do
  start_line="$(jsonl_line_count "$RUN_DIR/results/metrics.jsonl")"
  run_and_log "llm_infer_vllm_r${rep}" python3 "$BENCH_DIR/llm_infer_vllm.py" --config "$RUN_CONFIG_PATH" --warmup "$LLM_INFER_WARMUP_S" --duration "$LLM_INFER_DURATION_S"
  annotate_jsonl_rows "$RUN_DIR/results/metrics.jsonl" "$start_line" "llm_infer" "$rep" "$REPEAT_COUNT"
done

# --- 3) Stable Diffusion Inference ---
readarray -t SD_SIZES < <(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path sd_infer.sizes --default '[512]' --format lines)
SD_MODEL="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path sd_infer.model --default '"stabilityai/stable-diffusion-2-1"' --format text)"
SD_STEPS="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path sd_infer.steps --default '20' --format text)"
SD_BS="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path sd_infer.per_gpu_batch --default '1' --format text)"
SD_MULTI_GPU_MODE="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path sd_infer.multi_gpu_mode --default '"single"' --format text)"
SD_EMIT_WORKER_ROWS="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path sd_infer.emit_worker_rows --default 'false' --format bool-int)"
for rep in $(seq 1 "$REPEAT_COUNT"); do
  for sz in "${SD_SIZES[@]}"; do
    start_line="$(jsonl_line_count "$RUN_DIR/results/metrics.jsonl")"
    if [[ "$SD_EMIT_WORKER_ROWS" == "1" ]]; then
      run_and_log "sd_infer_${sz}_r${rep}" python3 "$BENCH_DIR/sd_infer.py" \
        --model "$SD_MODEL" --width "$sz" --height "$sz" \
        --steps "$SD_STEPS" --batch-size "$SD_BS" --iterations "$SD_ITERATIONS" \
        --metrics-path "$RUN_DIR/results/metrics.jsonl" \
        --repeat-index "$rep" --repeat-count "$REPEAT_COUNT" \
        --multi-gpu-mode "$SD_MULTI_GPU_MODE" \
        --emit-worker-rows
    else
      run_and_log "sd_infer_${sz}_r${rep}" python3 "$BENCH_DIR/sd_infer.py" \
        --model "$SD_MODEL" --width "$sz" --height "$sz" \
        --steps "$SD_STEPS" --batch-size "$SD_BS" --iterations "$SD_ITERATIONS" \
        --metrics-path "$RUN_DIR/results/metrics.jsonl" \
        --repeat-index "$rep" --repeat-count "$REPEAT_COUNT" \
        --multi-gpu-mode "$SD_MULTI_GPU_MODE"
    fi
    annotate_jsonl_rows "$RUN_DIR/results/metrics.jsonl" "$start_line" "sd_infer" "$rep" "$REPEAT_COUNT"
  done
done

# --- 4) Blender CUDA Bench (if Blender available) ---
if command -v blender >/dev/null 2>&1; then
  export SCENES_DIR="$BASE_DIR/assets/blender"
  export RESULTS_DIR="$RUN_DIR/results"
  export METRICS_JSONL="$RUN_DIR/results/metrics.jsonl"
  export BLENDER_ENABLED="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path blender.enabled --default 'true' --format bool-int)"
  export BLENDER_SCENES_JSON="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path blender.scenes --default '[]' --format json)"
  for rep in $(seq 1 "$REPEAT_COUNT"); do
    start_line="$(jsonl_line_count "$RUN_DIR/results/metrics.jsonl")"
    export RESULTS_JSON="$RUN_DIR/results/blender_bench_cuda_r${rep}.json"
    export REPEAT_INDEX="$rep"
    export REPEAT_COUNT
    run_and_log "blender_bench_cuda_r${rep}" bash "$BENCH_DIR/blender_bench_cuda.sh"
    annotate_jsonl_rows "$RUN_DIR/results/metrics.jsonl" "$start_line" "blender" "$rep" "$REPEAT_COUNT"
  done
else
  echo "[SKIP] Blender not found in PATH"
fi

# --- 5) Consolidate → CSV ---
run_and_log "consolidate" python3 "$BASE_DIR/harness.py"
if [[ -f "$RUN_DIR/results/metrics.csv" ]]; then
  cp "$RUN_DIR/results/metrics.csv" "$RUN_DIR/metrics.csv"
fi
if [[ -f "$RUN_DIR/results/metrics_summary.csv" ]]; then
  cp "$RUN_DIR/results/metrics_summary.csv" "$RUN_DIR/metrics_summary.csv"
fi
if [[ -f "$RUN_DIR/results/metrics_summary.json" ]]; then
  cp "$RUN_DIR/results/metrics_summary.json" "$RUN_DIR/metrics_summary.json"
fi

echo "[DONE] All artifacts are under: $RUN_DIR"
echo "        - Unified JSONL: $RUN_DIR/results/metrics.jsonl"
echo "        - CSV:           $RUN_DIR/metrics.csv"
echo "        - Summary CSV:   $RUN_DIR/metrics_summary.csv"
echo "        - Summary JSON:  $RUN_DIR/metrics_summary.json"
echo "        - Runtime Est.:  $RUN_DIR/runtime_estimate.json"
echo "        - System Reqs:   $RUN_DIR/system_requirements.json"
echo "        - Machine State: $RUN_DIR/machine_state.json"
echo "        - Blender JSON:  $RUN_DIR/results/blender_bench_cuda_r*.json (if ran)"
echo "        - Logs:          $RUN_DIR/logs/*.log"
