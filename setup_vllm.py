#!/usr/bin/env python3
"""Prepare the pinned vLLM environment without changing the main benchmark stack."""
import argparse
from contextlib import nullcontext
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from runtime_resolver import version


def check_host(host, release):
    if release != '0.34.0':
        return
    if host.get('system', 'Linux') != 'Linux':
        raise ValueError('The pinned vLLM 0.34.0 profiles require Linux')
    backend = host.get('backend', 'nvidia')
    gpus = host.get('gpus') or []
    if not gpus:
        raise ValueError('No selected GPUs were found for vLLM')
    if backend == 'amd':
        if host.get('machine', 'x86_64') != 'x86_64':
            raise ValueError('The official vLLM ROCm wheel requires x86-64')
        if (version(host.get('glibc_version')) or (2, 35, 0)) < (2, 35, 0):
            raise ValueError('The official vLLM ROCm stack requires glibc 2.35 or newer')
        return
    for gpu in gpus:
        if (version(gpu.get('driver_version')) or ()) < (580, 65, 6):
            raise ValueError(f"GPU {gpu['index']} requires NVIDIA driver 580.65.06 or newer "
                             f"for CUDA 13.0; found {gpu.get('driver_version')}")


def packages_ready(python, release, backend):
    code = '''
from importlib.metadata import version
import sys
installed_vllm = version('vllm')
release, backend = sys.argv[1:]
expected = release + '+rocm100' if backend == 'amd' and release == '0.34.0' else release
if installed_vllm != expected:
    raise RuntimeError(f'Expected vLLM {expected}, found {installed_vllm}')
import torch, transformers, yaml
from vllm import LLM, SamplingParams
if backend == 'amd':
    if not torch.version.hip:
        raise RuntimeError('Expected ROCm Torch')
    if release == '0.34.0':
        import vllm._rocm_C
elif release == '0.34.0':
    if torch.__version__.split('+')[0] != '2.14.0':
        raise RuntimeError(f'Expected Torch 2.14.0, found {torch.__version__}')
    if not torch.version.cuda or not torch.version.cuda.startswith('13.'):
        raise RuntimeError(f'Expected CUDA 13.x Torch, found {torch.version.cuda}')
    import vllm._C_stable_libtorch
'''
    try:
        for command in ([python, '-c', code, release, backend], [python, '-m', 'pip', 'check']):
            if subprocess.run(command, timeout=90).returncode:
                return False
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f'[vLLM] Package check failed: {exc}', file=sys.stderr)
        return False
    return True


def ensure_packages(target, release, python_override=None, backend='nvidia'):
    if python_override:
        if not packages_ready(python_override, release, backend):
            raise ValueError(f'VLLM_PYTHON must provide a working vLLM {release} environment')
        return python_override

    python = str(target / 'bin/python')
    if not target.exists():
        creator = sys.executable
        if backend == 'amd' and release == '0.34.0':
            creator = sys.executable if sys.version_info[:2] == (3, 12) else shutil.which('python3.12')
            if not creator:
                raise ValueError('Install Python 3.12 to create the pinned vLLM ROCm environment')
        elif release == '0.34.0' and not (3, 11) <= sys.version_info[:2] < (3, 15):
            creator = shutil.which('python3.12') or shutil.which('python3.11')
            if not creator:
                raise ValueError('Install Python 3.11 or 3.12 to create the vLLM 0.34.0 environment')
        subprocess.run([creator, '-m', 'venv', str(target)], check=True)
    if not (target / 'pyvenv.cfg').is_file():
        raise ValueError(f'{target} is not a virtual environment')
    if not packages_ready(python, release, backend):
        print(f'[SETUP] Installing/repairing vLLM {release} in {target}', flush=True)
        pin = release
        index = []
        if backend == 'amd':
            if release != '0.34.0':
                raise ValueError('Automatic ROCm setup is available for vLLM 0.34.0')
            pin += '+rocm100'
            index = ['--extra-index-url', f'https://wheels.vllm.ai/rocm/{release}/rocm100']
        subprocess.run([python, '-m', 'pip', 'install', '--upgrade', '--force-reinstall',
                        '--only-binary=vllm', *index, f'vllm=={pin}', 'PyYAML'], check=True)
        if not packages_ready(python, release, backend):
            raise ValueError(f'vLLM {release} package checks still fail after repair')
    return python


def verify_devices(python, host):
    code = '''
import sys
import torch
expected = int(sys.argv[1])
if torch.cuda.device_count() != expected:
    raise RuntimeError(f'Expected {expected} GPUs, found {torch.cuda.device_count()}')
for index in range(expected):
    gpu = torch.device(f'cuda:{index}')
    value = torch.ones((2, 2), dtype=torch.bfloat16, device=gpu)
    if not torch.equal(value @ value, torch.full_like(value, 2)):
        raise RuntimeError(f'BF16 arithmetic failed on GPU {index}')
    print(f'[vLLM] GPU {index}: {torch.cuda.get_device_name(index)} passed BF16 arithmetic')
if expected > 1:
    from verify_runtime import run_collective
    print(run_collective(expected))
'''
    subprocess.run([python, '-c', code, str(len(host['gpus']))], check=True, timeout=120,
                   cwd=Path(__file__).resolve().parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--venv', type=Path, required=True)
    parser.add_argument('--host-json', type=Path, required=True)
    parser.add_argument('--python')
    args = parser.parse_args()
    if not re.fullmatch(r'\d+\.\d+\.\d+', args.version):
        parser.error('--version must be an exact release (e.g. 0.34.0)')
    try:
        host = json.loads(args.host_json.read_text())
        check_host(host, args.version)
        lock = (nullcontext() if args.python else
                args.venv.with_name(args.venv.name + '.lock').open('a'))
        with lock as handle:
            if handle is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            python = ensure_packages(args.venv, args.version, args.python, host.get('backend', 'nvidia'))
            verify_devices(python, host)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'[vLLM][ERROR] {exc}\n')


if __name__ == '__main__':
    main()
