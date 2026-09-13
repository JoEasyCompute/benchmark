#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    raise SystemExit("[SYSTEM][ERROR] Missing dependency: PyYAML. Run env_setup.sh or activate the project venv.")

from gpu_platform import blender_backend, detect_backend, system_tool
REQUIRED_BINS = ("stdbuf", "tee", "hostname")
OPTIONAL_BINS = ("lscpu", "free")


def parse_missing_shared_libs(output):
    missing = []
    for line in output.splitlines():
        if ' => not found' in line:
            name = line.split(' => ', 1)[0].strip()
            if name and name not in missing:
                missing.append(name)
    return missing


def check_shared_libs(binary, runner=subprocess.run):
    try:
        result = runner(['ldd', binary], capture_output=True, text=True, timeout=15, check=False,
                        env=dict(os.environ, LD_LIBRARY_PATH=str(Path(binary).resolve().parent / 'lib')))
        return parse_missing_shared_libs(result.stdout + '\n' + result.stderr)
    except (OSError, subprocess.SubprocessError):
        return []


def optional_capabilities(cfg, backend, importer=importlib.import_module):
    checks, warnings = [], []
    modules = set()
    if (cfg.get('vision_infer') or {}).get('enabled'):
        modules.update(('torch', 'torchvision'))
    kernel = cfg.get('kernel_bench') or {}
    if kernel.get('enabled'):
        modules.add('torch')
    serving = cfg.get('llm_serve') or {}
    if serving.get('enabled'):
        if serving.get('endpoint'):
            checks.append({'serving': 'external streaming endpoint; server hardware/revision must be verified separately'})
        else:
            modules.update(('torch', 'transformers'))
    if backend == 'nvidia':
        modules.add('pynvml')
    else:
        checks.append({'power_tool': 'rocm-smi --showpower --json',
                       'available': bool(find_binary('rocm-smi'))})
    loaded = {}
    for name in sorted(modules):
        try:
            loaded[name] = importer(name)
            checks.append({'module': name, 'available': True})
        except Exception as exc:
            checks.append({'module': name, 'available': False})
            warnings.append(f'optional capability {name} unavailable; suite/telemetry may skip ({type(exc).__name__})')
    if kernel.get('enabled') and 'attention' in kernel.get('cases', ['gemm', 'attention', 'memory']):
        functional = getattr(getattr(loaded.get('torch'), 'nn', None), 'functional', None)
        available = hasattr(functional, 'scaled_dot_product_attention')
        checks.append({'kernel_api': 'scaled_dot_product_attention', 'available': available})
        if not available:
            warnings.append('scaled_dot_product_attention is unavailable for the requested attention benchmark')
    return checks, warnings


def find_binary(name: str) -> str | None:
    path = shutil.which(name)
    if path:
        return path

    candidates = (
        Path.home() / ".local" / "bin" / name,
        Path.home() / "bin" / name,
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    backend = detect_backend(cfg.get("gpu_backend", "auto"))

    payload = {"status": "ok", "checks": [], "warnings": [], "errors": []}
    checks, warnings = optional_capabilities(cfg, backend)
    payload['checks'].extend(checks)
    payload['warnings'].extend(warnings)
    payload["checks"].append({"platform": platform.platform()})
    payload["checks"].append({"gpu_backend": backend})

    if platform.system() != "Linux":
        payload["errors"].append("Linux host required for the benchmark environment")

    gpu_tool = system_tool(backend)
    required_bins = (gpu_tool, *REQUIRED_BINS)

    for name in required_bins:
        path = find_binary(name)
        payload["checks"].append({"binary": name, "path": path})
        if not path:
            payload["errors"].append(f"required binary not found on PATH: {name}")

    for name in OPTIONAL_BINS:
        path = find_binary(name)
        payload["checks"].append({"binary": name, "path": path})
        if not path:
            payload["warnings"].append(f"optional binary not found on PATH: {name}")

    llm_infer_backend = ((cfg.get("llm_infer", {}) or {}).get("backend", "transformers") or "transformers").lower()
    if backend == "amd" and llm_infer_backend == "vllm":
        try:
            importlib.import_module("vllm")
            importlib.import_module("vllm._C")
            payload["checks"].append({"amd_vllm_runtime": "available"})
        except Exception as exc:
            payload["checks"].append({"amd_vllm_runtime": "unavailable"})
            payload["warnings"].append(
                f"AMD llm_infer_vllm is not validated in this environment; benchmark will likely be skipped ({type(exc).__name__}: {exc})"
            )

    blender_cfg = cfg.get("blender", {}) or {}
    blender_enabled = blender_cfg.get("enabled", True)
    blender_require_installed = bool(blender_cfg.get("require_installed", False))
    blender_strict = bool((cfg.get("preflight", {}) or {}).get("blender_strict", False))
    if blender_enabled:
        blender_path = find_binary("blender")
        payload["checks"].append({"binary": "blender", "path": blender_path})
        if not blender_path:
            message = "blender not found on PATH"
            if blender_require_installed or blender_strict:
                payload["errors"].append(f"{message}; Blender benchmark is enabled and required")
            else:
                payload["warnings"].append(f"{message}; Blender benchmark will be skipped")
        payload["checks"].append({"blender_backend": blender_backend(backend)})

        if blender_path:
            missing_libs = check_shared_libs(blender_path)
            payload['checks'].append({'blender_missing_shared_libs': missing_libs})
            if missing_libs:
                message = 'Blender missing shared libraries: ' + ', '.join(missing_libs)
                if blender_require_installed or blender_strict:
                    payload['errors'].append(message)
                else:
                    payload['warnings'].append(message + '; Blender benchmark will be skipped')
            else:
                try:
                    probe = subprocess.run([blender_path, '--background', '--factory-startup', '--python-expr',
                        "import bpy; print('BENCH_CYCLES_AVAILABLE=' + str('cycles' in bpy.context.preferences.addons))"],
                        capture_output=True, text=True, timeout=30, check=False)
                    cycles = probe.returncode == 0 and 'BENCH_CYCLES_AVAILABLE=True' in probe.stdout
                    payload['checks'].append({'blender_cycles_available': cycles})
                    if not cycles:
                        payload['warnings'].append('Blender Cycles capability probe failed; render suite may fail')
                except (OSError, subprocess.SubprocessError) as exc:
                    payload['warnings'].append(f'Blender capability probe unavailable: {type(exc).__name__}')

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
