#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    raise SystemExit("[MACHINE][ERROR] Missing dependency: PyYAML. Run env_setup.sh or activate the project venv.")

from gpu_platform import detect_backend


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def query_nvidia_gpu(fields):
    if not shutil.which("nvidia-smi"):
        return []
    out = run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == len(fields):
            rows.append(dict(zip(fields, parts)))
    return rows


def query_nvidia_compute_apps():
    if not shutil.which("nvidia-smi"):
        return []
    out = run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    apps = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            apps.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": parts[1],
                    "process_name": parts[2],
                    "used_memory_mb": parts[3],
                }
            )
    return apps


def query_amd_machine_state():
    if not shutil.which("rocm-smi"):
        return [], []
    out = run(["rocm-smi", "--showproductname", "--showtemp", "--showuse", "--showmemuse", "--json"])
    if not out:
        return [], []
    try:
        payload = json.loads(out)
    except Exception:
        return [], []

    card = payload.get("card") or payload
    rows = []
    if isinstance(card, dict):
        for key, value in card.items():
            if not isinstance(value, dict):
                continue
            idx = str(key).replace("card", "") if str(key).startswith("card") else str(key)
            rows.append(
                {
                    "index": idx,
                    "name": value.get("Card series") or value.get("Card model") or value.get("Product Name") or "unknown",
                    "temperature.gpu": value.get("Temperature (Sensor edge) (C)") or value.get("Temperature (edge)"),
                    "utilization.gpu": value.get("GPU use (%)"),
                    "memory.used": value.get("VRAM Total Used Memory (B)") or value.get("VRAM Total Used Memory (MiB)"),
                    "memory.total": value.get("VRAM Total Memory (B)") or value.get("VRAM Total Memory (MiB)"),
                    "power.draw": value.get("Average Graphics Package Power (W)"),
                }
            )
    return rows, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    backend = detect_backend(cfg.get("gpu_backend", "auto"))

    payload = {"status": "ok", "gpu_backend": backend, "checks": [], "warnings": [], "errors": []}

    if backend == "amd":
        gpus, apps = query_amd_machine_state()
        if not gpus:
            payload["status"] = "warn"
            payload["warnings"].append("rocm-smi not found or returned no data; AMD machine-state checks skipped")
        else:
            payload["checks"].append({"gpu_summary": gpus})
            payload["checks"].append({"active_compute_processes": apps})
            for gpu in gpus:
                idx = gpu.get("index", "?")
                try:
                    temp = float(str(gpu.get("temperature.gpu", "0")).split()[0] or 0)
                    if temp >= 80:
                        payload["warnings"].append(f"GPU {idx}: temperature is high at {temp}C")
                except ValueError:
                    pass
    else:
        if not shutil.which("nvidia-smi"):
            payload["status"] = "warn"
            payload["warnings"].append("nvidia-smi not found; GPU machine-state checks skipped")
        else:
            gpus = query_nvidia_gpu(
                [
                    "index",
                    "name",
                    "persistence_mode",
                    "pstate",
                    "temperature.gpu",
                    "utilization.gpu",
                    "memory.used",
                    "memory.total",
                    "clocks_throttle_reasons.active",
                    "power.draw",
                ]
            )
            apps = query_nvidia_compute_apps()
            payload["checks"].append({"gpu_summary": gpus})
            payload["checks"].append({"active_compute_processes": apps})

            for gpu in gpus:
                idx = gpu.get("index", "?")
                persistence = gpu.get("persistence_mode", "").lower()
                if persistence not in {"enabled", "1"}:
                    payload["warnings"].append(f"GPU {idx}: persistence mode is not enabled")

                try:
                    temp = float(gpu.get("temperature.gpu", "0") or 0)
                    if temp >= 80:
                        payload["warnings"].append(f"GPU {idx}: temperature is high at {temp}C")
                except ValueError:
                    pass

                pstate = gpu.get("pstate", "")
                if pstate and pstate not in {"P0", "P2"}:
                    payload["warnings"].append(f"GPU {idx}: current pstate is {pstate}")

                throttle = gpu.get("clocks_throttle_reasons.active", "").lower()
                if throttle and throttle not in {"not active", "0x0000000000000000"}:
                    payload["warnings"].append(f"GPU {idx}: active clock throttling reported ({gpu.get('clocks_throttle_reasons.active')})")

                try:
                    mem_used = float(gpu.get("memory.used", "0") or 0)
                    if mem_used > 1024:
                        payload["warnings"].append(f"GPU {idx}: {int(mem_used)} MiB already allocated before benchmark start")
                except ValueError:
                    pass

            if apps:
                payload["warnings"].append(f"{len(apps)} active GPU compute process(es) detected before benchmark start")

    if payload["warnings"] and payload["status"] == "ok":
        payload["status"] = "warn"

    text_lines = [f"[MACHINE] status={payload['status']} backend={backend}"]
    for warning in payload["warnings"]:
        text_lines.append(f"[MACHINE][WARN] {warning}")
    for error in payload["errors"]:
        text_lines.append(f"[MACHINE][ERROR] {error}")
    print("\n".join(text_lines))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")

    if args.strict and payload["warnings"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
