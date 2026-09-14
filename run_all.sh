#!/usr/bin/env bash
set -Eeuo pipefail

# --- Setup ---
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$BASE_DIR/benchmarks"
VENV_DIR="$BASE_DIR/.venv"
ENV_SETUP_SCRIPT="$BASE_DIR/env_setup.sh"
BASE_CONFIG_PATH="$BASE_DIR/config.yaml"
CONFIG_UTILS="$BASE_DIR/config_utils.py"
GPU_PLATFORM="$BASE_DIR/gpu_platform.py"
SMOKE_MODE=0
BASELINE_MODE=0
BACKEND_OVERRIDE=""
GPU_OVERRIDE=""
GPU_OVERRIDE_SET=0
DRY_RUN=0
ALLOW_UNVERIFIED_HOST=0

# Include common user-local bin directories so host-level tools installed by
# helper scripts are discoverable even in non-login shells.
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      cat <<'HELP'
Usage: bash run_all.sh [--config PATH] [--backend auto|nvidia|amd]
                      [--gpus 0,1] [--baseline] [--smoke] [--dry-run] [--allow-unverified-host]

Defaults to config.yaml and automatic backend selection when gpu_backend is auto.
--config PATH  Read a sample/custom config without copying it over config.yaml.
--backend NAME Override the config backend (required to disambiguate mixed hosts).
--gpus IDS     Select physical GPU indices within any inherited visibility mask.
--baseline     Use one detected/selected GPU and five repeats.
--smoke        Reduce enabled workloads and use one repeat.
--dry-run      Print resolved config; do not install the GPU stack or run workloads.
--allow-unverified-host  Continue when OS/kernel is outside the published runtime matrix.
HELP
      exit 0
      ;;
    --config|--backend|--gpus)
      if [[ $# -lt 2 || "$2" == --* || -z "$2" ]]; then
        echo "[ERROR] $1 requires a value" >&2
        exit 2
      fi
      case "$1" in
        --config) BASE_CONFIG_PATH="$2" ;;
        --backend) BACKEND_OVERRIDE="$2" ;;
        --gpus) GPU_OVERRIDE="$2"; GPU_OVERRIDE_SET=1 ;;
      esac
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --allow-unverified-host)
      ALLOW_UNVERIFIED_HOST=1
      shift
      ;;
    --baseline)
      BASELINE_MODE=1
      shift
      ;;
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

case "$BACKEND_OVERRIDE" in
  ""|auto|nvidia|amd) ;;
  *) echo "[ERROR] --backend must be auto, nvidia, or amd" >&2; exit 2 ;;
esac
if [[ ! -f "$BASE_CONFIG_PATH" ]]; then
  echo "[ERROR] Config file not found: $BASE_CONFIG_PATH" >&2
  exit 2
fi
BASE_CONFIG_PATH="$(cd -- "$(dirname -- "$BASE_CONFIG_PATH")" && pwd)/$(basename -- "$BASE_CONFIG_PATH")"

python_env_ready() {
  local py_bin="$1" backend="$2"
  [[ -x "$py_bin" ]] || return 1
  "$py_bin" - "$backend" <<'PY' >/dev/null 2>&1
import sys
import accelerate, diffusers, pandas, safetensors, torch, transformers, tqdm, yaml
actual = 'amd' if torch.version.hip else 'nvidia' if torch.version.cuda else 'cpu'
raise SystemExit(0 if actual == sys.argv[1] else 1)
PY
}

ensure_python_env() {
  local py_bin="$VENV_DIR/bin/python"
  local backend="$1"
  if ! python_env_ready "$py_bin" "$backend"; then
    echo "[SETUP] Installing/repairing the $backend Python stack"
    VENV_DIR="$VENV_DIR" GPU_BACKEND="$backend" ALLOW_UNVERIFIED_HOST="$ALLOW_UNVERIFIED_HOST" bash "$ENV_SETUP_SCRIPT"
  fi
  if ! python_env_ready "$py_bin" "$backend"; then
    echo "[ERROR] The Python environment does not provide the selected $backend stack after setup" >&2
    exit 1
  fi
}

# Reading YAML must work before choosing which GPU stack to install.
CONFIG_PYTHON="python3"
if [[ -x "$VENV_DIR/bin/python" ]] && "$VENV_DIR/bin/python" -c 'import yaml' >/dev/null 2>&1; then
  CONFIG_PYTHON="$VENV_DIR/bin/python"
elif ! python3 -c 'import yaml' >/dev/null 2>&1; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[ERROR] Dry run requires Python with PyYAML; activate the project environment first" >&2
    exit 2
  fi
  VENV_DIR="$VENV_DIR" bash "$ENV_SETUP_SCRIPT" --config-only
  CONFIG_PYTHON="$VENV_DIR/bin/python"
fi
"$CONFIG_PYTHON" "$BASE_DIR/validate_config.py" --config "$BASE_CONFIG_PATH" >&2
# Resolve the profile before selecting devices or determining run metadata.
EFFECTIVE_CONFIG_TMP="$(mktemp)"
trap 'rm -f "$EFFECTIVE_CONFIG_TMP"' EXIT
EFFECTIVE_CONFIG_ARGS=(--resolve-hardware)
if [[ -n "$BACKEND_OVERRIDE" ]]; then EFFECTIVE_CONFIG_ARGS+=(--backend "$BACKEND_OVERRIDE"); fi
if [[ "$GPU_OVERRIDE_SET" == "1" ]]; then EFFECTIVE_CONFIG_ARGS+=(--gpus "$GPU_OVERRIDE"); fi
if [[ "$SMOKE_MODE" == "1" ]]; then EFFECTIVE_CONFIG_ARGS+=(--smoke); fi
if [[ "$BASELINE_MODE" == "1" ]]; then EFFECTIVE_CONFIG_ARGS+=(--baseline); fi
"$CONFIG_PYTHON" "$CONFIG_UTILS" write-effective --config "$BASE_CONFIG_PATH" --output "$EFFECTIVE_CONFIG_TMP" "${EFFECTIVE_CONFIG_ARGS[@]}"
"$CONFIG_PYTHON" "$BASE_DIR/validate_config.py" --config "$EFFECTIVE_CONFIG_TMP" >&2
if [[ "$DRY_RUN" == "1" ]]; then
  cat "$EFFECTIVE_CONFIG_TMP"
  exit 0
fi
GPU_BACKEND="$("$CONFIG_PYTHON" "$CONFIG_UTILS" get --config "$EFFECTIVE_CONFIG_TMP" --path gpu_backend --format text)"
echo "[GPU] Selected backend: $GPU_BACKEND (config: $BASE_CONFIG_PATH)"
if [[ "$GPU_BACKEND" == "amd" ]]; then
  unset CUDA_VISIBLE_DEVICES
else
  unset HIP_VISIBLE_DEVICES
fi
ensure_python_env "$GPU_BACKEND"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
MACHINE_STATE_STRICT="$(python3 "$CONFIG_UTILS" get --config "$EFFECTIVE_CONFIG_TMP" --path preflight.machine_state_strict --default 'false' --format bool-int)"

# Determine results root from config.yaml (fallback: results)
RESULTS_ROOT="$(python3 "$CONFIG_UTILS" get --config "$EFFECTIVE_CONFIG_TMP" --path results_dir --default '"results"' --format text)"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
REPEAT_COUNT="$(python3 "$CONFIG_UTILS" get --config "$EFFECTIVE_CONFIG_TMP" --path repeat --default '1' --format text)"
REPEAT_COUNT="${REPEAT_COUNT:-1}"
readarray -t GPU_INCLUDE_VALUES < <(python3 "$CONFIG_UTILS" get --config "$EFFECTIVE_CONFIG_TMP" --path gpu_include --default '[]' --format lines)
VISIBLE_ENV_VAR="$(python3 "$GPU_PLATFORM" visible-env-var --backend "$GPU_BACKEND")"
GPU_SYSTEM_TOOL="$(python3 "$GPU_PLATFORM" system-tool --backend "$GPU_BACKEND")"
BLENDER_BACKEND_CFG="$(python3 "$CONFIG_UTILS" get --config "$EFFECTIVE_CONFIG_TMP" --path blender.backend --default '"auto"' --format text)"
if [[ "$BLENDER_BACKEND_CFG" == "auto" ]]; then
  BLENDER_GPU_BACKEND="$(python3 "$GPU_PLATFORM" blender-backend --backend "$GPU_BACKEND")"
else
  BLENDER_GPU_BACKEND="${BLENDER_BACKEND_CFG^^}"
fi

detect_all_gpu_ids() {
  python3 "$GPU_PLATFORM" gpu-ids --backend "$GPU_BACKEND"
}

readarray -t ALL_GPU_IDS < <(detect_all_gpu_ids)
SELECTED_GPU_IDS=("${ALL_GPU_IDS[@]}")
if [[ "${#GPU_INCLUDE_VALUES[@]}" -gt 0 ]]; then
  SELECTED_GPU_IDS=("${GPU_INCLUDE_VALUES[@]}")
fi
if [[ "$BASELINE_MODE" == "1" ]]; then
  BASELINE_GPU_FOUND=0
  for gpu_id in "${ALL_GPU_IDS[@]}"; do
    if [[ "$gpu_id" == "${SELECTED_GPU_IDS[0]}" ]]; then BASELINE_GPU_FOUND=1; fi
  done
  if [[ "$BASELINE_GPU_FOUND" != "1" ]]; then
    echo "[ERROR] Baseline GPU ${SELECTED_GPU_IDS[0]} is not present on this host" >&2
    exit 1
  fi
fi
if [[ "${#SELECTED_GPU_IDS[@]}" -gt 0 ]]; then
  SELECTED_GPU_CSV="$(IFS=,; echo "${SELECTED_GPU_IDS[*]}")"
  VISIBLE_DEVICE_MASK="$(python3 "$GPU_PLATFORM" visibility-mask --backend "$GPU_BACKEND" --visible-devices "$SELECTED_GPU_CSV")"
  export "$VISIBLE_ENV_VAR=$VISIBLE_DEVICE_MASK"
  echo "[INFO] Physical GPU indices: $SELECTED_GPU_CSV; $VISIBLE_ENV_VAR=$VISIBLE_DEVICE_MASK"
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
GPU_MODEL="$(python3 "$GPU_PLATFORM" gpu-names --backend "$GPU_BACKEND" --visible-devices "${SELECTED_GPU_CSV:-}" | head -n1 || echo '')"

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
cp "$EFFECTIVE_CONFIG_TMP" "$RUN_CONFIG_PATH"
echo "[INFO] Effective config: $RUN_CONFIG_PATH"
if [[ "$SMOKE_MODE" == "1" ]]; then
  echo "[INFO] Smoke mode enabled"
fi
REPEAT_COUNT="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path repeat --default '1' --format text)"

# Capture the inspected host and resolve a concrete, vendor-compatible runtime profile.
HOST_INVENTORY="$RUN_DIR/host_inventory.json"
python3 "$BASE_DIR/inspect_runtime.py" --backend "$GPU_BACKEND" --gpus "$SELECTED_GPU_CSV" --json-out "$HOST_INVENTORY" || true
RUNTIME_PLAN="$RUN_DIR/runtime_plan.json"
resolver_flags=(--host-json "$HOST_INVENTORY" --json-out "$RUNTIME_PLAN")
if [[ "$ALLOW_UNVERIFIED_HOST" == "1" ]]; then resolver_flags+=(--allow-unverified-host); fi
if ! python3 "$BASE_DIR/runtime_resolver.py" "${resolver_flags[@]}" >/dev/null; then
  echo "[ERROR] Runtime compatibility blocked; inspect $RUNTIME_PLAN" >&2
  exit 1
fi
# Host acknowledgement never bypasses the package, device, or arithmetic checks.
RUNTIME_VALIDATION="$RUN_DIR/runtime_validation.json"
echo "[RUNTIME] Checking packages and arithmetic on all $VISIBLE_GPU_COUNT selected GPU(s)"
if ! python3 "$BASE_DIR/verify_runtime.py" --lock "$RUNTIME_PLAN" --distributed \
    --json-out "$RUNTIME_VALIDATION" >"$RUN_DIR/logs/runtime_validation.log" 2>&1; then
  echo "[ERROR] Runtime validation failed; inspect $RUNTIME_VALIDATION and $RUN_DIR/logs/runtime_validation.log" >&2
  exit 1
fi
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
python3 "$BASE_DIR/lock_model_revisions.py" --config "$RUN_CONFIG_PATH" --manifest "$RUN_DIR/model_revisions.json"
if [[ "$MACHINE_STATE_STRICT" == "1" ]]; then
  python3 "$BASE_DIR/check_machine_state.py" --config "$RUN_CONFIG_PATH" --strict --json-out "$RUN_DIR/machine_state.json"
else
  python3 "$BASE_DIR/check_machine_state.py" --config "$RUN_CONFIG_PATH" --json-out "$RUN_DIR/machine_state.json"
fi

# --- System metadata snapshot ---
meta_file="$RUN_DIR/meta.json"
python3 - "$GPU_BACKEND" "$GPU_SYSTEM_TOOL" "$VISIBLE_ENV_VAR" "${SELECTED_GPU_CSV:-}" <<'PY' >"$meta_file"
import json, os, subprocess, platform, time
from importlib.metadata import PackageNotFoundError, version
import sys

backend = sys.argv[1]
gpu_tool = sys.argv[2]
visible_env_var = sys.argv[3]
visible_csv = sys.argv[4]

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

env = os.environ.copy()
if visible_csv:
    env[visible_env_var] = visible_csv
try:
    if backend == "amd":
        gpu = subprocess.check_output([gpu_tool, "--showproductname", "--showtemp", "--showuse", "--json"], text=True, stderr=subprocess.DEVNULL, env=env).strip()
    else:
        gpu = subprocess.check_output([gpu_tool, "--query-gpu=index,name,driver_version,pstate,temperature.gpu,power.draw", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL, env=env).strip()
except Exception:
    gpu = ""
cpu = cmd("lscpu")
mem = cmd("free -h")
osrel = cmd("cat /etc/os-release")
blender_version = cmd("blender --version | head -n1")
data = {
    "ts": int(time.time()),
    "platform": platform.platform(),
    "kernel": platform.release(),
    "python": platform.python_version(),
    "gpu_backend": backend,
    "selected_gpu_indices": visible_csv.split(',') if visible_csv else [],
    "visible_device_env": visible_env_var,
    "visible_devices": os.environ.get(visible_env_var),
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
        gpu_tool: cmd(f"{gpu_tool} --version | head -n1"),
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

RUN_FAILURE_COUNT=0

run_and_log_allow_fail () {
  local name="$1"; shift
  local had_errexit=0
  case $- in
    *e*) had_errexit=1 ;;
  esac

  set +e
  run_and_log "$name" "$@"
  local rc=$?
  if [[ "$had_errexit" -eq 1 ]]; then
    set -e
  fi

  if [[ "$rc" -eq 0 ]]; then
    return 0
  fi

  RUN_FAILURE_COUNT=$((RUN_FAILURE_COUNT + 1))
  echo "[WARN] $name exited with status $rc; continuing"
  printf '[WARN] suite command exited with status %s\n' "$rc" >> "$RUN_DIR/logs/${name}.log"
  return 0
}

jsonl_line_count () {
  local fp="$1"
  if [[ -f "$fp" ]]; then
    wc -l < "$fp" | tr -d ' '
  else
    echo 0
  fi
}

last_suite_status () {
  local fp="$1" suite="$2"
  if [[ ! -f "$fp" ]]; then echo missing; return; fi
  python3 - "$fp" "$suite" <<'PY'
import json, sys
rows=[]
for line in open(sys.argv[1]):
    if line.strip():
        row=json.loads(line)
        if row.get('suite') == sys.argv[2]: rows.append(row)
print(rows[-1].get('status', 'unknown') if rows else 'missing')
PY
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
if [[ "$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path llm_train.enabled --default 'true' --format bool-int)" == "1" ]]; then
readarray -t TRAIN_WORLD_SIZES < <(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path llm_train.world_sizes --default '[1]' --format lines)
TRAIN_WS2_GPU_CSV=""
TRAIN_PAIR_SELECTION_ENABLED="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path llm_train.pair_selection.enabled --default 'false' --format bool-int)"
TRAIN_WORLD_SIZE_2_REQUESTED=0
for requested_world_size in "${TRAIN_WORLD_SIZES[@]}"; do
  if [[ "$requested_world_size" == "2" ]]; then
    TRAIN_WORLD_SIZE_2_REQUESTED=1
  fi
done
if [[ "$TRAIN_PAIR_SELECTION_ENABLED" == "1" && "$TRAIN_WORLD_SIZE_2_REQUESTED" == "1" && "$VISIBLE_GPU_COUNT" -ge 2 ]]; then
  if [[ -n "${SELECTED_GPU_CSV:-}" ]]; then
    echo "[PAIR] Selecting best llm_train world_size=2 GPU pair from: $SELECTED_GPU_CSV"
    PAIR_SELECTION_JSON="$RUN_DIR/llm_train_ws2_pair_selection.json"
    PAIR_SELECTION_LOG_DIR="$RUN_DIR/logs/gpu_pair_selection"
    python3 "$BASE_DIR/select_gpu_pair.py" \
      --config "$RUN_CONFIG_PATH" \
      --gpu-ids "$SELECTED_GPU_CSV" \
      --backend "$GPU_BACKEND" \
      --visible-env-var "$VISIBLE_ENV_VAR" \
      --bench-script "$BENCH_DIR/llm_train.py" \
      --output "$PAIR_SELECTION_JSON" \
      --log-dir "$PAIR_SELECTION_LOG_DIR" \
      --python-bin "$(command -v python3)"
    TRAIN_WS2_GPU_CSV="$(python3 - "$PAIR_SELECTION_JSON" <<'PYJSON'
import json, sys
with open(sys.argv[1]) as f:
    payload = json.load(f)
print(payload.get("selected_pair_csv", ""))
PYJSON
)"
    if [[ -n "$TRAIN_WS2_GPU_CSV" ]]; then
      echo "[PAIR] Selected llm_train world_size=2 GPU pair: $TRAIN_WS2_GPU_CSV"
    else
      echo "[PAIR][WARN] Pair selector did not return a pair; using first two visible GPUs"
    fi
  fi
fi

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

    if [[ "$world_size" == "2" && -n "$TRAIN_WS2_GPU_CSV" ]]; then
      train_gpu_csv="$TRAIN_WS2_GPU_CSV"
    else
      train_gpu_ids=("${SELECTED_GPU_IDS[@]:0:$world_size}")
      train_gpu_csv="$(IFS=,; echo "${train_gpu_ids[*]}")"
    fi
    train_visibility="$(python3 "$GPU_PLATFORM" visibility-mask --backend "$GPU_BACKEND" --visible-devices "$train_gpu_csv")"
    start_line="$(jsonl_line_count "$RUN_DIR/results/metrics.jsonl")"
    if [[ "$world_size" -gt 1 ]]; then
      run_and_log_allow_fail "llm_train_ws${world_size}_r${rep}" env "$VISIBLE_ENV_VAR=$train_visibility" \
        python3 -m torch.distributed.run --standalone --nproc_per_node "$world_size" \
        "$BENCH_DIR/llm_train.py" --config "$RUN_CONFIG_PATH"
    else
      run_and_log_allow_fail "llm_train_ws${world_size}_r${rep}" env "$VISIBLE_ENV_VAR=$train_visibility" \
        python3 "$BENCH_DIR/llm_train.py" --config "$RUN_CONFIG_PATH"
    fi
    annotate_jsonl_rows "$RUN_DIR/results/metrics.jsonl" "$start_line" "llm_train" "$rep" "$REPEAT_COUNT"
  done
done

fi

LLM_TRAIN_REAL_ENABLED="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path llm_train_real.enabled --default 'false' --format bool-int)"
if [[ "$LLM_TRAIN_REAL_ENABLED" == "1" ]]; then
  for rep in $(seq 1 "$REPEAT_COUNT"); do
    start_line="$(jsonl_line_count "$RUN_DIR/results/metrics.jsonl")"
    run_and_log_allow_fail "llm_train_real_r${rep}" python3 "$BENCH_DIR/llm_train_real.py" --config "$RUN_CONFIG_PATH"
    annotate_jsonl_rows "$RUN_DIR/results/metrics.jsonl" "$start_line" "llm_train_real" "$rep" "$REPEAT_COUNT"
    if [[ "$(last_suite_status "$RUN_DIR/results/metrics.jsonl" llm_train_real)" == "failed" ]]; then
      echo "[WARN] llm_train_real failed; stopping remaining repeats for this deterministic workload"
      break
    fi
  done
fi

# --- 2) LLM Inference ---
if [[ "$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path llm_infer.enabled --default 'true' --format bool-int)" == "1" ]]; then
LLM_INFER_BACKEND="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path llm_infer.backend --default '"transformers"' --format text)"
LLM_INFER_SCRIPT="$BENCH_DIR/llm_infer_hf.py"
LLM_INFER_LOG_NAME="llm_infer_transformers"
if [[ "$LLM_INFER_BACKEND" == "vllm" ]]; then
  LLM_INFER_SCRIPT="$BENCH_DIR/llm_infer_vllm.py"
  LLM_INFER_LOG_NAME="llm_infer_vllm"
fi
for rep in $(seq 1 "$REPEAT_COUNT"); do
  start_line="$(jsonl_line_count "$RUN_DIR/results/metrics.jsonl")"
  run_and_log_allow_fail "${LLM_INFER_LOG_NAME}_r${rep}" python3 "$LLM_INFER_SCRIPT" --config "$RUN_CONFIG_PATH" --warmup "$LLM_INFER_WARMUP_S" --duration "$LLM_INFER_DURATION_S"
  annotate_jsonl_rows "$RUN_DIR/results/metrics.jsonl" "$start_line" "llm_infer" "$rep" "$REPEAT_COUNT"
done

# --- 3) Stable Diffusion Inference ---
fi
if [[ "$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path sd_infer.enabled --default 'true' --format bool-int)" == "1" ]]; then
readarray -t SD_SIZES < <(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path sd_infer.sizes --default '[512]' --format lines)
SD_MODEL="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path sd_infer.model --default '"stabilityai/stable-diffusion-2-1"' --format text)"
SD_REVISION="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path sd_infer.revision --default '"main"' --format text)"
SD_STEPS="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path sd_infer.steps --default '20' --format text)"
SD_BS="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path sd_infer.per_gpu_batch --default '1' --format text)"
SD_MULTI_GPU_MODE="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path sd_infer.multi_gpu_mode --default '"single"' --format text)"
SD_EMIT_WORKER_ROWS="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path sd_infer.emit_worker_rows --default 'false' --format bool-int)"
for rep in $(seq 1 "$REPEAT_COUNT"); do
  for sz in "${SD_SIZES[@]}"; do
    start_line="$(jsonl_line_count "$RUN_DIR/results/metrics.jsonl")"
    if [[ "$SD_EMIT_WORKER_ROWS" == "1" ]]; then
      run_and_log_allow_fail "sd_infer_${sz}_r${rep}" python3 "$BENCH_DIR/sd_infer.py" \
        --model "$SD_MODEL" --revision "$SD_REVISION" --width "$sz" --height "$sz" \
        --steps "$SD_STEPS" --batch-size "$SD_BS" --iterations "$SD_ITERATIONS" \
        --metrics-path "$RUN_DIR/results/metrics.jsonl" \
        --repeat-index "$rep" --repeat-count "$REPEAT_COUNT" \
        --multi-gpu-mode "$SD_MULTI_GPU_MODE" \
        --emit-worker-rows
    else
      run_and_log_allow_fail "sd_infer_${sz}_r${rep}" python3 "$BENCH_DIR/sd_infer.py" \
        --model "$SD_MODEL" --revision "$SD_REVISION" --width "$sz" --height "$sz" \
        --steps "$SD_STEPS" --batch-size "$SD_BS" --iterations "$SD_ITERATIONS" \
        --metrics-path "$RUN_DIR/results/metrics.jsonl" \
        --repeat-index "$rep" --repeat-count "$REPEAT_COUNT" \
        --multi-gpu-mode "$SD_MULTI_GPU_MODE"
    fi
    annotate_jsonl_rows "$RUN_DIR/results/metrics.jsonl" "$start_line" "sd_infer" "$rep" "$REPEAT_COUNT"
  done
done

fi

run_and_log_allow_fail "optional_suites" python3 "$BASE_DIR/run_optional_suites.py" --config "$RUN_CONFIG_PATH" --run-dir "$RUN_DIR"

# --- 4) Blender CUDA Bench (if Blender available) ---
if command -v blender >/dev/null 2>&1; then
  export BENCHMARK_PROFILE="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path benchmark_profile --default '"general"' --format text)"
  export SCENES_DIR="$BASE_DIR/assets/blender"
  export RESULTS_DIR="$RUN_DIR/results"
  export METRICS_JSONL="$RUN_DIR/results/metrics.jsonl"
  export BLENDER_GPU_BACKEND
  export BLENDER_ENABLED="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path blender.enabled --default 'true' --format bool-int)"
  export BLENDER_SCENES_JSON="$(python3 "$CONFIG_UTILS" get --config "$RUN_CONFIG_PATH" --path blender.scenes --default '[]' --format json)"
  for rep in $(seq 1 "$REPEAT_COUNT"); do
    start_line="$(jsonl_line_count "$RUN_DIR/results/metrics.jsonl")"
    export RESULTS_JSON="$RUN_DIR/results/blender_bench_${BLENDER_GPU_BACKEND,,}_r${rep}.json"
    export REPEAT_INDEX="$rep"
    export REPEAT_COUNT
    run_and_log_allow_fail "blender_bench_${BLENDER_GPU_BACKEND,,}_r${rep}" bash "$BENCH_DIR/blender_bench_cuda.sh"
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
echo "        - Blender JSON:  $RUN_DIR/results/blender_bench_*.json (if ran)"
echo "        - Logs:          $RUN_DIR/logs/*.log"

if [[ "$RUN_FAILURE_COUNT" -gt 0 ]]; then
  echo "[WARN] $RUN_FAILURE_COUNT benchmark command(s) exited non-zero during the run" >&2
  exit 1
fi
