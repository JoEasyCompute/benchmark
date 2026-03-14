#!/usr/bin/env python3
import argparse
import json
import platform
import shutil
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    raise SystemExit("[SYSTEM][ERROR] Missing dependency: PyYAML. Run env_setup.sh or activate the project venv.")

from gpu_platform import blender_backend, detect_backend, system_tool
REQUIRED_BINS = ("stdbuf", "tee", "hostname")
OPTIONAL_BINS = ("lscpu", "free")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    backend = detect_backend(cfg.get("gpu_backend", "auto"))

    payload = {"status": "ok", "checks": [], "warnings": [], "errors": []}
    payload["checks"].append({"platform": platform.platform()})
    payload["checks"].append({"gpu_backend": backend})

    if platform.system() != "Linux":
        payload["errors"].append("Linux host required for the benchmark environment")

    gpu_tool = system_tool(backend)
    required_bins = (gpu_tool, *REQUIRED_BINS)

    for name in required_bins:
        path = shutil.which(name)
        payload["checks"].append({"binary": name, "path": path})
        if not path:
            payload["errors"].append(f"required binary not found on PATH: {name}")

    for name in OPTIONAL_BINS:
        path = shutil.which(name)
        payload["checks"].append({"binary": name, "path": path})
        if not path:
            payload["warnings"].append(f"optional binary not found on PATH: {name}")

    blender_enabled = (cfg.get("blender", {}) or {}).get("enabled", True)
    if blender_enabled:
        blender_path = shutil.which("blender")
        payload["checks"].append({"binary": "blender", "path": blender_path})
        if not blender_path:
            payload["warnings"].append("blender not found on PATH; Blender benchmark will be skipped")
        payload["checks"].append({"blender_backend": blender_backend(backend)})

        time_path = Path("/usr/bin/time")
        payload["checks"].append({"binary": "/usr/bin/time", "path": str(time_path) if time_path.exists() else None})
        if not time_path.exists():
            payload["warnings"].append("/usr/bin/time not found; Blender timing script will fail if Blender is enabled")

    if payload["errors"]:
        payload["status"] = "error"
    elif payload["warnings"]:
        payload["status"] = "warn"

    print(f"[SYSTEM] status={payload['status']}")
    for warning in payload["warnings"]:
        print(f"[SYSTEM][WARN] {warning}")
    for error in payload["errors"]:
        print(f"[SYSTEM][ERROR] {error}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")

    if payload["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
