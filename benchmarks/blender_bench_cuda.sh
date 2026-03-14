#!/usr/bin/env bash
set -Eeuo pipefail

# ===============================
# Config (env overridable)
# ===============================
BLENDER_BIN="${BLENDER_BIN:-$(command -v blender || true)}"
SCENES_DIR="${SCENES_DIR:-assets/blender}"
RESULTS_DIR="${RESULTS_DIR:-$SCENES_DIR/results}"
RESULTS_JSON="${RESULTS_JSON:-$RESULTS_DIR/bench_results_cuda.json}"
METRICS_JSONL="${METRICS_JSONL:-$RESULTS_DIR/metrics.jsonl}"
REPEAT_INDEX="${REPEAT_INDEX:-1}"
REPEAT_COUNT="${REPEAT_COUNT:-1}"
FRAME="${FRAME:-1}"
SKIP_COLD="${SKIP_COLD:-0}"
DEBUG="${DEBUG:-0}"
BLENDER_ENABLED="${BLENDER_ENABLED:-1}"
BLENDER_SCENES_JSON="${BLENDER_SCENES_JSON:-[]}"

mkdir -p "$RESULTS_DIR"

# ===============================
# Helpers
# ===============================
die() { echo "[ERR] $*" >&2; exit 1; }

require_bin() {
  [[ -n "${BLENDER_BIN:-}" && -x "$BLENDER_BIN" ]] || die "Blender not found. Set BLENDER_BIN to a valid blender binary."
  command -v /usr/bin/time >/dev/null 2>&1 || die "GNU time not found at /usr/bin/time"
}

load_scenes() {
  python3 - "$BLENDER_SCENES_JSON" <<'PY'
import json, sys
payload = sys.argv[1]
try:
    data = json.loads(payload)
except Exception:
    data = []
if not isinstance(data, list):
    data = []
for item in data:
    if isinstance(item, str) and item.strip():
        print(item.strip())
PY
}

# Find a scene path in SCENES_DIR or immediate subfolders
resolve_scene_path() {
  local scene="$1" scpath altpath
  if [[ -f "$scene" ]]; then
    echo "$scene"; return 0
  fi
  scpath="$SCENES_DIR/$scene"
  if [[ -f "$scpath" ]]; then
    echo "$scpath"; return 0
  fi
  altpath="$(find "$SCENES_DIR" -maxdepth 2 -type f -name "$(basename "$scene")" | head -n1 || true)"
  if [[ -n "$altpath" ]]; then
    echo "[INFO] Found scene in subfolder → $altpath" >&2
    echo "$altpath"; return 0
  fi
  return 1
}

# Build the python expr to select CUDA devices (single/all) and set GPU rendering
py_expr_cuda() { # $1=mode single|all  (mode is referenced via env CUDA_BENCH_MODE)
  cat <<'PY'
import bpy, os
p=bpy.context.preferences.addons['cycles'].preferences
p.compute_device_type='CUDA'
p.get_devices()
# disable all first
for d in p.devices:
    if hasattr(d,'use'):
        d.use=False
use_all = os.environ.get('CUDA_BENCH_MODE','single') == 'all'
count = 0
for d in p.devices:
    if d.type=='CUDA':
        if use_all:
            d.use=True
        else:
            if count==0:
                d.use=True
            count+=1
bpy.context.scene.cycles.device='GPU'
print("devices=", [(d.type,d.name,getattr(d,'use',None)) for d in p.devices])
PY
}

# Run blender and emit: "<rc> <seconds>" on STDOUT
# All human-readable logs go to STDERR so callers can safely parse.
run_and_time() { # $1=scene_path  $2=label  $3=mode single|all
  local scpath="$1" label="$2" mode="$3"
  local tmpout cmd rc tm
  tmpout="$(mktemp)"
  trap 'rm -f "$tmpout"' RETURN

  cmd=( "$BLENDER_BIN" -b "$scpath" -E CYCLES --python-expr "$(py_expr_cuda "$mode")" -f "$FRAME" )
  echo "[RUN] $label ($mode)" >&2

  if [[ "$DEBUG" == "1" ]]; then
    echo "  CMD: CUDA_BENCH_MODE=$mode /usr/bin/time -f %e -o $tmpout \\" >&2
    printf '       %q ' "${cmd[@]}" >&2; echo >&2
  fi

  rc=0
  CUDA_BENCH_MODE="$mode" /usr/bin/time -f '%e' -o "$tmpout" \
    "${cmd[@]}" >/dev/null 2>&1 || rc=$?

  tm="$(tr -d ' \t\r\n' < "$tmpout" 2>/dev/null || true)"

  if [[ -z "$tm" ]]; then
    echo "  [TIME] $label ($mode): n/a (rc=$rc)" >&2
    printf '%s %s\n' "$rc" ""
    return 0
  fi

  if ! [[ "$tm" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "  [TIME] $label ($mode): unparsable '$tm' (rc=$rc)" >&2
    printf '%s %s\n' "$rc" ""
    return 0
  fi

  echo "  [TIME] $label ($mode): ${tm}s" >&2
  printf '%s %s\n' "$rc" "$tm"
}

pretty_or_raw_json() { # prints pretty if possible, else raw
  local payload="$1"
  if printf '%s' "$payload" | python3 -c 'import json, sys; json.loads(sys.stdin.read())' >/dev/null 2>&1
  then
    echo "$payload" | python3 -m json.tool
  else
    echo "[WARN] JSON pretty-print failed; dumping raw:" >&2
    echo "$payload"
  fi
}

# ===============================
# Main
# ===============================
if [[ "$BLENDER_ENABLED" != "1" ]]; then
  echo "[SKIP] Blender disabled via config" >&2
  exit 0
fi

require_bin

readarray -t SCENES < <(load_scenes)
if [[ "${#SCENES[@]}" -eq 0 ]]; then
  SCENES=("BMW27.blend" "classroom.blend")
fi

results="["; first=1
metrics_tmp="$(mktemp)"
trap 'rm -f "$metrics_tmp"' EXIT

write_metric_row() {
  local payload="$1"
  printf '%s\n' "$payload" >> "$metrics_tmp"
}

for scene in "${SCENES[@]}"; do
  if ! scpath="$(resolve_scene_path "$scene")"; then
    echo "[WARN] missing scene $SCENES_DIR/$scene — skipping" >&2
    write_metric_row "{\"suite\":\"blender\",\"status\":\"failed\",\"scene\":\"$(basename "$scene")\",\"backend\":\"CUDA\",\"error\":\"missing scene\",\"repeat_index\":$REPEAT_INDEX,\"repeat_count\":$REPEAT_COUNT}"
    continue
  fi
  base="$(basename "$scene")"

  # CUDA single
  [[ "${SKIP_COLD}" == "1" ]] && echo "[SKIP] ${base} CUDA (single): cold run skipped." >&2
  read -r rc t <<<"$(run_and_time "$scpath" "${base} CUDA (single)" "single")"
  if [[ -n "${t:-}" ]]; then
    [[ $first -eq 0 ]] && results+=","
    first=0
    results+="{\"scene\":\"${base}\",\"backend\":\"CUDA\",\"mode\":\"single\",\"cold\":0.0,\"warm\":$t,\"compile\":0.0,\"rc\":$rc}"
    write_metric_row "{\"suite\":\"blender\",\"status\":\"$([[ "$rc" -eq 0 ]] && echo ok || echo failed)\",\"scene\":\"${base}\",\"backend\":\"CUDA\",\"mode\":\"single\",\"cold\":0.0,\"warm\":$t,\"compile\":0.0,\"rc\":$rc,\"time_s\":$t,\"repeat_index\":$REPEAT_INDEX,\"repeat_count\":$REPEAT_COUNT}"
  else
    write_metric_row "{\"suite\":\"blender\",\"status\":\"failed\",\"scene\":\"${base}\",\"backend\":\"CUDA\",\"mode\":\"single\",\"rc\":$rc,\"error\":\"missing timing output\",\"repeat_index\":$REPEAT_INDEX,\"repeat_count\":$REPEAT_COUNT}"
  fi

  # CUDA all
  [[ "${SKIP_COLD}" == "1" ]] && echo "[SKIP] ${base} CUDA (all): cold run skipped." >&2
  read -r rc t <<<"$(run_and_time "$scpath" "${base} CUDA (all)" "all")"
  if [[ -n "${t:-}" ]]; then
    [[ $first -eq 0 ]] && results+=","
    first=0
    results+="{\"scene\":\"${base}\",\"backend\":\"CUDA\",\"mode\":\"all\",\"cold\":0.0,\"warm\":$t,\"compile\":0.0,\"rc\":$rc}"
    write_metric_row "{\"suite\":\"blender\",\"status\":\"$([[ "$rc" -eq 0 ]] && echo ok || echo failed)\",\"scene\":\"${base}\",\"backend\":\"CUDA\",\"mode\":\"all\",\"cold\":0.0,\"warm\":$t,\"compile\":0.0,\"rc\":$rc,\"time_s\":$t,\"repeat_index\":$REPEAT_INDEX,\"repeat_count\":$REPEAT_COUNT}"
  else
    write_metric_row "{\"suite\":\"blender\",\"status\":\"failed\",\"scene\":\"${base}\",\"backend\":\"CUDA\",\"mode\":\"all\",\"rc\":$rc,\"error\":\"missing timing output\",\"repeat_index\":$REPEAT_INDEX,\"repeat_count\":$REPEAT_COUNT}"
  fi
done

results+="]"
pretty_or_raw_json "$results" || true
printf '%s\n' "$results" > "$RESULTS_JSON"
cat "$metrics_tmp" >> "$METRICS_JSONL"
echo "[OK] Results → $RESULTS_JSON" >&2
