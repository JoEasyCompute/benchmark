#!/usr/bin/env bash
# Cycles scale comparison: CUDA vs OPTIX, single GPU vs all GPUs
# Outputs JSON to assets/blender/results/bench_results_scale.json

set -uo pipefail

# ------------------- Config -------------------
SCENES=("BMW27.blend" "classroom.blend")
ASSET_ROOT="assets/blender"
RESULT_DIR="$ASSET_ROOT/results"
RESULT_FILE="$RESULT_DIR/bench_results_scale.json"

BLENDER_BIN="${BLENDER_BIN:-}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
SKIP_COLD="${SKIP_COLD:-0}"
DEBUG="${DEBUG:-0}"

# ------------------- Guards -------------------
if [[ -z "$BLENDER_BIN" ]]; then
  echo "[ERR] BLENDER_BIN is not set. Example:" 1>&2
  echo "      export BLENDER_BIN=\"/opt/blender/blender-4.2.5-linux-x64/blender\"" 1>&2
  exit 1
fi
if ! command -v "$BLENDER_BIN" >/dev/null 2>&1; then
  echo "[ERR] BLENDER_BIN not found at: $BLENDER_BIN" 1>&2
  exit 1
fi
if ! command -v "$TIME_BIN" >/dev/null 2>&1; then
  echo "[ERR] time(1) not found at $TIME_BIN" 1>&2
  exit 1
fi

mkdir -p "$RESULT_DIR"

# ------------------- Helpers -------------------
# Find a scene by filename either directly under ASSET_ROOT or one level deeper
find_scene() {
  local name="$1"
  local direct="$ASSET_ROOT/$name"
  if [[ -f "$direct" ]]; then
    printf '%s\n' "$direct"
    return 0
  fi
  local found
  found="$(find "$ASSET_ROOT" -mindepth 2 -maxdepth 2 -type f -name "$name" | head -n1 || true)"
  if [[ -n "$found" ]]; then
    echo "[INFO] Found scene in subfolder → $found" 1>&2
    printf '%s\n' "$found"
    return 0
  fi
  return 1
}

# Check if backend has devices (CUDA or OPTIX)
check_backend_available() {
  local backend="$1"
  local out
  out="$("$BLENDER_BIN" -b --factory-startup -E CYCLES \
    --python-expr "import bpy, sys
p=bpy.context.preferences.addons['cycles'].preferences
try: p.compute_device_type='$backend'
except Exception: pass
if hasattr(p,'refresh_devices'):
    try: p.refresh_devices()
    except Exception: pass
devs = list(getattr(p,'devices',[]) or [])
cnt = sum(1 for d in devs if getattr(d,'type',None) == '$backend')
print('COUNT', cnt)" -f 0 2>/dev/null || true)"
  local cnt
  cnt="$(printf '%s\n' "$out" | awk '/^COUNT /{print $2; exit}')"
  if [[ -n "$cnt" && "$cnt" =~ ^[0-9]+$ && "$cnt" -ge 1 ]]; then
    return 0
  fi
  return 1
}

# Parse a "Time: HH:MM:SS.xx" OR "Time: MM:SS.xx" fragment from Blender log to seconds
parse_time_from_log() {
  local log="$1"
  # grab the last occurrence of "Time: ..." from the log
  local line
  line="$(grep -a 'Time:' "$log" | tail -n1 || true)"
  [[ -z "$line" ]] && { printf '%s' ""; return 0; }
  # extract the clock group after "Time:"
  local clock
  clock="$(printf '%s' "$line" | sed -n 's/.*Time:[[:space:]]\([0-9:]\+\)\(\.[0-9]\+\)\?.*/\1\2/p')"
  [[ -z "$clock" ]] && { printf '%s' ""; return 0; }

  # Use python for robust parsing (handles H:M:S, M:S, with/without fractional)
  python3 - <<PY
import sys
clk = sys.argv[1]
parts = clk.split(':')
try:
    if len(parts) == 3:
        h = float(parts[0]); m = float(parts[1]); s = float(parts[2])
        print(h*3600 + m*60 + s)
    elif len(parts) == 2:
        m = float(parts[0]); s = float(parts[1])
        print(m*60 + s)
    elif len(parts) == 1:
        print(float(parts[0]))
    else:
        print("")
except:
    print("")
PY
"$clock"
}

# Enable devices for backend; mode=single|all
# Render frame 1 of scene
# Prints two lines to stdout: <elapsed_seconds_or_blank>\n<return_code>
run_one() {
  local scene_path="$1" backend="$2" mode="$3"
  local phase="${4:-warm}"

  local tmp_t tmp_log
  tmp_t="$(mktemp)"
  tmp_log="$(mktemp)"

  local title="$(basename "$scene_path") $backend ($mode)"
  if [[ "$phase" == "warm" ]]; then
    echo "[RUN] $title (warm)" 1>&2
  else
    echo "[RUN] $title (cold)" 1>&2
  fi

  local rc=0
  "$TIME_BIN" -f '%e' -o "$tmp_t" \
    "$BLENDER_BIN" -b "$scene_path" -E CYCLES \
    --python-expr "import bpy, sys
p=bpy.context.preferences.addons['cycles'].preferences
def activate(kind, mode):
    try: p.compute_device_type=kind
    except Exception: pass
    if hasattr(p,'refresh_devices'):
        try: p.refresh_devices()
        except Exception: pass
    devs = list(getattr(p,'devices',[]) or [])
    for d in devs:
        try:
            if hasattr(d,'use'): d.use=False
        except Exception: pass
    sel = [d for d in devs if getattr(d,'type',None)==kind]
    enabled=0
    if sel:
        if mode=='single':
            try:
                if hasattr(sel[0],'use'): sel[0].use=True
                enabled=1
            except Exception: pass
        else:
            for d in sel:
                try:
                    if hasattr(d,'use'): d.use=True
                    enabled+=1
                except Exception: pass
    bpy.context.scene.render.engine='CYCLES'
    bpy.context.scene.cycles.device='GPU'
    print('enabled_count=', enabled)
    return enabled
enabled = activate('$backend', '$mode')
if enabled == 0 and '$backend' == 'OPTIX':
    enabled = activate('CUDA', '$mode')" \
    -f 1 >"$tmp_log" 2>&1 || rc=$?

  # Primary timing from /usr/bin/time
  local t
  t="$(tr -d '\r' <"$tmp_t" | tail -n1 | sed 's/^[ \t]*//;s/[ \t]*$//')"

  # Fallback: parse "Time: ..." from Blender log
  if [[ -z "$t" ]]; then
    t="$(parse_time_from_log "$tmp_log")"
    if [[ -n "$t" && "$DEBUG" == "1" ]]; then
      echo "[DBG] fallback parsed log time: $t s" 1>&2
    fi
  fi

  if [[ "$DEBUG" == "1" ]]; then
    echo "[DBG] rc=$rc, time='$t'" 1>&2
    echo "[DBG] last lines of log:" 1>&2
    tail -n 5 "$tmp_log" 1>&2 || true
  fi

  rm -f "$tmp_t" "$tmp_log"

  # Pretty print to stderr for the operator
  if [[ "$rc" -eq 0 && "$t" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    printf '  [TIME] %s: %ss\n' "$title" "$t" 1>&2
  else
    printf '  [TIME] %s: (rc=%s) %s\n' "$title" "$rc" "${t:-N/A}" 1>&2
  fi

  printf '%s\n%s\n' "${t:-}" "$rc"
}

# ------------------- Main -------------------
results=()
BACKENDS=(CUDA OPTIX)

for scene in "${SCENES[@]}"; do
  scene_path="$(find_scene "$scene" || true)"
  if [[ -z "${scene_path:-}" ]]; then
    echo "[WARN] missing scene $ASSET_ROOT/$scene — skipping" 1>&2
    continue
  fi

  # Probe once per backend
  declare -A BACKEND_OK=()
  for b in "${BACKENDS[@]}"; do
    if check_backend_available "$b"; then BACKEND_OK[$b]=1; else BACKEND_OK[$b]=0; fi
  done

  do_cold=$(( SKIP_COLD == 0 ? 1 : 0 ))

  for backend in "${BACKENDS[@]}"; do
    if [[ "${BACKEND_OK[$backend]}" -eq 0 ]]; then
      echo "[SKIP] $(basename "$scene_path")/$backend: backend not available." 1>&2
      continue
    fi

    for mode in single all; do
      # Cold
      if [[ "$do_cold" -eq 1 ]]; then
        cold_lines="$(run_one "$scene_path" "$backend" "$mode" cold)"
        cold_t="$(printf '%s\n' "$cold_lines" | sed -n '1p')"
        cold_rc="$(printf '%s\n' "$cold_lines" | sed -n '2p')"
      else
        echo "[SKIP] $(basename "$scene_path") $backend ($mode): cold run skipped." 1>&2
        cold_t="0.0"; cold_rc="0"
      fi

      # Warm
      warm_lines="$(run_one "$scene_path" "$backend" "$mode" warm)"
      warm_t="$(printf '%s\n' "$warm_lines" | sed -n '1p')"
      warm_rc="$(printf '%s\n' "$warm_lines" | sed -n '2p')"

      # Build JSON via env → python (robust quoting)
      obj="$(
        SCENE_PATH="$scene_path" BACKEND="$backend" MODE="$mode" \
        COLD_T="$cold_t" WARM_T="$warm_t" COLD_RC="$cold_rc" WARM_RC="$warm_rc" \
        python3 - <<'PY'
import json, os, pathlib
def f(x):
  try: return float(x)
  except: return 0.0
def i(x):
  try: return int(x)
  except: return 0
scene_name = pathlib.Path(os.environ.get("SCENE_PATH","")).name or "unknown"
backend = os.environ.get("BACKEND","")
mode = os.environ.get("MODE","")
cold_t = f(os.environ.get("COLD_T","0"))
warm_t = f(os.environ.get("WARM_T","0"))
cold_rc = i(os.environ.get("COLD_RC","0"))
warm_rc = i(os.environ.get("WARM_RC","0"))
out = {
  "scene": scene_name,
  "backend": backend,
  "mode": mode,
  "cold": cold_t,
  "warm": warm_t,
  "compile": 0.0,
  "rc_cold": cold_rc,
  "rc_warm": warm_rc
}
print(json.dumps(out))
PY
      )"
      results+=("$obj")
    done
  done
done

# Write JSON
{
  printf '[\n'
  for i in "${!results[@]}"; do
    printf '  %s' "${results[$i]}"
    if [[ "$i" -lt $((${#results[@]}-1)) ]]; then printf ','; fi
    printf '\n'
  done
  printf ']\n'
} > "$RESULT_FILE"

# Pretty print (best effort)
if command -v jq >/dev/null 2>&1; then
  jq . "$RESULT_FILE" || echo "[WARN] jq pretty-print failed; raw JSON saved." 1>&2
else
  echo "[WARN] jq not found; raw JSON saved." 1>&2
fi

echo "[OK] Results → $RESULT_FILE"

