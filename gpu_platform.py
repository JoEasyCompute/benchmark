#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from copy import deepcopy


BACKENDS = ("nvidia", "amd")


class NoGpuError(ValueError):
    """No responding supported device is available."""


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except Exception:
        return ""


def detect_backend(configured: str = "auto") -> str:
    configured = (configured or "auto").lower()
    if configured in BACKENDS:
        return configured
    if configured != 'auto':
        raise ValueError('GPU backend must be auto, nvidia, or amd')
    detected = [backend for backend in BACKENDS if query_gpu_ids(backend)]
    if len(detected) == 1:
        return detected[0]
    if len(detected) > 1:
        raise ValueError('Both NVIDIA and AMD GPUs are available. Select --backend nvidia or --backend amd.')
    raise NoGpuError('No responding NVIDIA/AMD GPUs found. Check drivers and nvidia-smi/rocm-smi availability.')


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
        return [line.strip() for line in out.splitlines() if line.strip().isdigit()]

    if not shutil.which(tool):
        return []
    out = run([tool, "--showid", "--json"])
    if out:
        try:
            payload = json.loads(out)
            card = payload.get("card") or payload
            if isinstance(card, dict):
                return [str(k)[4:] if str(k).startswith('card') else str(k) for k in card
                        if (str(k).startswith('card') and str(k)[4:].isdigit()) or str(k).isdigit()]
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
    if backend == "nvidia":
        command = [tool, '--query-gpu=name', '--format=csv,noheader']
        if visible_csv:
            command.append('--id=' + visible_csv)
        out = run(command)
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
                selected = visible_csv.split(',') if visible_csv else [str(k).removeprefix('card') for k in card]
                for identifier in selected:
                    value = card.get('card' + identifier, card.get(identifier))
                    if isinstance(value, dict):
                        product = value.get("Card series") or value.get("Card model") or value.get("Product Name")
                        if product:
                            names.append(str(product))
            return names
        except Exception:
            pass
    return []


def visible_physical_ids(backend, available, environ):
    """Respect an inherited numeric/UUID CUDA or numeric HIP visibility mask."""
    raw = environ.get(visible_env_var(backend))
    if raw is None:
        return list(available)
    if not raw.strip() or raw.strip() == '-1':
        return []
    requested = [value.strip() for value in raw.split(',')]
    if backend == 'nvidia' and any(not value.isdigit() for value in requested):
        mapping = {}
        for line in run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits']).splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 2 and parts[0] in available:
                mapping[parts[1]] = parts[0]
        translated = []
        for value in requested:
            matches = [identifier for uuid, identifier in mapping.items() if uuid.startswith(value)]
            if value.isdigit():
                translated.append(value)
            elif len(matches) == 1:
                translated.append(matches[0])
            else:
                raise ValueError(f'Cannot resolve visible GPU identifier {value!r}')
        requested = translated
    if any(value not in available for value in requested) or len(requested) != len(set(requested)):
        raise ValueError(f'Invalid {visible_env_var(backend)} mask for detected devices: {raw!r}')
    return requested


def visibility_mask(backend, identifiers):
    """Use stable NVIDIA UUIDs so CUDA enumeration cannot change the selected card."""
    if backend != 'nvidia':
        return ','.join(map(str, identifiers))
    mapping = {}
    for line in run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits']).splitlines():
        parts = [part.strip() for part in line.split(',')]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].startswith('GPU-'):
            mapping[parts[0]] = parts[1]
    if any(str(identifier) not in mapping for identifier in identifiers):
        raise ValueError('Unable to map selected NVIDIA indices to GPU UUIDs')
    return ','.join(mapping[str(identifier)] for identifier in identifiers)


def resolve_config(cfg, backend_override=None, gpu_override=None, environ=None):
    """Resolve host-specific settings without changing workload parameters."""
    env = os.environ if environ is None else environ
    resolved = deepcopy(cfg)
    backend = detect_backend(backend_override or cfg.get('gpu_backend', 'auto'))
    available = query_gpu_ids(backend)
    if not available:
        raise ValueError(f'No responding {backend} GPUs found for the requested backend')
    visible = visible_physical_ids(backend, available, env)
    requested = cfg.get('gpu_include') or visible
    if gpu_override is not None:
        requested = [value.strip() for value in gpu_override.split(',')]
        if not requested or any(not value.isdigit() for value in requested):
            raise ValueError('--gpus must be comma-separated physical GPU indices, e.g. 0 or 0,1')
    requested = [str(value) for value in requested]
    if not requested or any(value not in visible for value in requested) or len(requested) != len(set(requested)):
        raise ValueError(f'Requested GPUs {requested} must be distinct members of the visible {backend} GPUs {visible}')
    resolved['gpu_backend'] = backend
    resolved['gpu_include'] = [int(value) for value in requested]
    render = resolved.setdefault('blender', {})
    selected_render = render.get('backend', 'auto')
    expected = blender_backend(backend).lower()
    if selected_render not in ('auto', expected):
        raise ValueError(f'Blender backend {selected_render} conflicts with selected {backend}; use auto or {expected}')
    render['backend'] = expected
    return resolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["detect-backend", "visible-env-var", "system-tool", "gpu-ids", "gpu-names", "blender-backend", "visibility-mask"])
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--visible-devices", default="")
    args = ap.parse_args()

    try:
        backend = detect_backend(args.backend)
    except ValueError as exc:
        ap.exit(2, f'[GPU][ERROR] {exc}\n')
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
    elif args.command == 'visibility-mask':
        try:
            print(visibility_mask(backend, args.visible_devices.split(',')))
        except ValueError as exc:
            ap.exit(2, f'[GPU][ERROR] {exc}\n')


if __name__ == "__main__":
    main()
