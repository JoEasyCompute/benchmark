#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess


BACKENDS = ("nvidia", "amd")


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def detect_backend(configured: str = "auto") -> str:
    configured = (configured or "auto").lower()
    if configured in BACKENDS:
        return configured
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("rocm-smi"):
        return "amd"
    return "nvidia"


def visible_env_var(backend: str) -> str:
    return "HIP_VISIBLE_DEVICES" if backend == "amd" else "CUDA_VISIBLE_DEVICES"


def system_tool(backend: str) -> str:
    return "rocm-smi" if backend == "amd" else "nvidia-smi"


def power_package(backend: str) -> str | None:
    return None if backend == "amd" else "pynvml"


def blender_backend(backend: str) -> str:
    return "HIP" if backend == "amd" else "CUDA"


def query_gpu_ids(backend: str) -> list[str]:
    tool = system_tool(backend)
    if backend == "nvidia":
        out = run([tool, "--query-gpu=index", "--format=csv,noheader,nounits"])
        return [line.strip() for line in out.splitlines() if line.strip()]

    if not shutil.which(tool):
        return []
    out = run([tool, "--showid", "--json"])
    if out:
        try:
            payload = json.loads(out)
            card = payload.get("card") or payload
            if isinstance(card, dict):
                return [str(k).replace("card", "") if str(k).startswith("card") else str(k) for k in card.keys()]
        except Exception:
            pass
    out = run([tool, "-i"])
    gpu_ids = []
    for line in out.splitlines():
        line = line.strip()
        if line.lower().startswith("gpu[") and "]" in line:
            gpu_ids.append(line.split("[", 1)[1].split("]", 1)[0])
    return gpu_ids


def query_gpu_names(backend: str, visible_csv: str = "") -> list[str]:
    tool = system_tool(backend)
    env = os.environ.copy()
    if visible_csv:
        env[visible_env_var(backend)] = visible_csv

    if backend == "nvidia":
        try:
            out = subprocess.check_output(
                [tool, "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
                env=env,
            ).strip()
        except Exception:
            out = ""
        return [line.strip() for line in out.splitlines() if line.strip()]

    if not shutil.which(tool):
        return []
    out = run([tool, "--showproductname", "--json"])
    if out:
        try:
            payload = json.loads(out)
            card = payload.get("card") or payload
            names = []
            if isinstance(card, dict):
                for value in card.values():
                    if isinstance(value, dict):
                        product = value.get("Card series") or value.get("Card model") or value.get("Product Name")
                        if product:
                            names.append(str(product))
            return names
        except Exception:
            pass
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["detect-backend", "visible-env-var", "system-tool", "gpu-ids", "gpu-names", "blender-backend"])
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--visible-devices", default="")
    args = ap.parse_args()

    backend = detect_backend(args.backend)
    if args.command == "detect-backend":
        print(backend)
    elif args.command == "visible-env-var":
        print(visible_env_var(backend))
    elif args.command == "system-tool":
        print(system_tool(backend))
    elif args.command == "gpu-ids":
        for item in query_gpu_ids(backend):
            print(item)
    elif args.command == "gpu-names":
        for item in query_gpu_names(backend, args.visible_devices):
            print(item)
    elif args.command == "blender-backend":
        print(blender_backend(backend))


if __name__ == "__main__":
    main()
