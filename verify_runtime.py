#!/usr/bin/env python3
"""Check the locked framework and tiny GPU operations before benchmarking."""
import argparse
from datetime import timedelta
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


def check_packages(lock):
    installed = {}
    errors = []
    for package, expected in lock['profile']['expected_packages'].items():
        try:
            installed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed[package] = None
        if installed[package] != expected:
            errors.append(f'{package}: expected {expected}, installed {installed[package]}')
    return installed, errors


def numerical_check(torch, index):
    # Keep allocations local: each device is tested and released separately.
    matrix = torch.tensor([[1., 2.], [3., 4.]], device=f'cuda:{index}', dtype=torch.float32)
    expected = torch.tensor([[7., 10.], [15., 22.]], device=f'cuda:{index}', dtype=torch.float32)
    actual = matrix @ matrix
    torch.cuda.synchronize(index)
    if not bool(torch.isfinite(actual).all()):
        raise RuntimeError(f'GPU {index} produced nonfinite values')
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def version_parts(value):
    match = re.match(r'^(\d+(?:\.\d+)*)', str(value))
    return tuple(int(part) for part in match.group(1).split('.')) if match else ()


def run_collective(count):
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node', str(count), str(Path(__file__).resolve()), '--collective-worker']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise RuntimeError('Distributed GPU check timed out after 90 seconds')
    if process.returncode:
        raise RuntimeError(f'Distributed GPU check failed ({process.returncode}): {stderr[-4000:]}')
    return {'status': 'pass', 'world_size': count, 'stdout': stdout[-4000:]}


def run_checks(torch_module, lock, distributed=False):
    report = {'status': 'error', 'errors': [], 'installed_packages': {}, 'devices': [],
              'compile_architectures': [], 'collective': {'status': 'not_requested'}}
    torch = torch_module
    try:
        profile = lock['profile']
        if lock.get('status') == 'blocked':
            raise ValueError('Cannot validate a blocked runtime selection; resolve its compatibility errors first')
        if lock.get('schema_version') != 1:
            raise ValueError('Unsupported lock schema_version')
        report['installed_packages'], errors = check_packages(lock)
        if errors:
            raise RuntimeError('; '.join(errors))
        expected_torch = profile['expected_packages'].get('torch')
        if str(torch.__version__) != expected_torch:
            raise RuntimeError(f'Imported torch version {torch.__version__} != {expected_torch}')
        if str(torch.__version__).split('+')[0] != profile['torch_version']:
            raise RuntimeError('Imported torch does not match profile torch_version')
        backend = 'amd' if torch.version.hip else ('nvidia' if torch.version.cuda else 'cpu')
        if backend != profile['backend']:
            raise RuntimeError(f'Backend mismatch: expected {profile["backend"]}, imported {backend}')
        actual_runtime = torch.version.hip if backend == 'amd' else torch.version.cuda
        expected = version_parts(profile['runtime_version'])
        actual = version_parts(actual_runtime)
        if not expected or actual[:len(expected)] != expected:
            raise RuntimeError(f'Runtime mismatch: expected {profile["runtime_version"]}, imported {actual_runtime}')
        report['backend'] = backend
        report['runtime_version'] = actual_runtime
        if not torch.cuda.is_available():
            raise RuntimeError('GPU runtime is unavailable')
        count = torch.cuda.device_count()
        if count < 1:
            raise RuntimeError('No visible GPU devices')
        if count != len(lock['host']['gpus']):
            raise RuntimeError(f'Visible GPU count {count} differs from locked count {len(lock["host"]["gpus"])}')
        report['compile_architectures'] = list(torch.cuda.get_arch_list())
        for index in range(count):
            properties = torch.cuda.get_device_properties(index)
            architecture = (getattr(properties, 'gcnArchName', None) if backend == 'amd'
                            else f'sm_{properties.major}{properties.minor}')
            expected_arch = lock['host']['gpus'][index].get('architecture')
            if expected_arch and str(architecture).split(':')[0] != expected_arch:
                raise RuntimeError(f'GPU {index} architecture {architecture} differs from locked {expected_arch}')
            numerical_check(torch, index)
            report['devices'].append({'index': index, 'name': torch.cuda.get_device_name(index),
                                      'architecture': architecture, 'numerical_check': 'pass'})
        if distributed:
            report['collective'] = run_collective(count) if count > 1 else {'status': 'skipped_single_gpu'}
        report['status'] = 'pass'
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    return report


def collective_worker():
    torch = importlib.import_module('torch')
    dist = importlib.import_module('torch.distributed')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=45))
    try:
        numerical_check(torch, local_rank)
        rank = dist.get_rank()
        size = dist.get_world_size()
        value = torch.tensor([float(rank + 1)], device=f'cuda:{local_rank}')
        dist.all_reduce(value)
        expected = torch.tensor([float(size * (size + 1) // 2)], device=f'cuda:{local_rank}')
        torch.cuda.synchronize(local_rank)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--json-out', type=Path)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--collective-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.collective_worker:
        collective_worker()
        return 0
    if not args.lock:
        parser.error('--lock is required')
    try:
        lock = json.loads(args.lock.read_text())
        report = run_checks(importlib.import_module('torch'), lock, args.distributed)
    except Exception as exc:
        report = {'status': 'error', 'errors': [f'{type(exc).__name__}: {exc}']}
    payload = json.dumps(report, indent=2) + '\n'
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload)
    print(payload, end='')
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
