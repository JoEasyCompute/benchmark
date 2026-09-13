#!/usr/bin/env python3
"""Read-only, dependency-free host inspection (also usable through python3 -)."""
import argparse
import csv
import importlib.metadata
import io
import json
import platform
import os
import re
import subprocess
from pathlib import Path


PACKAGES = ('torch', 'torchvision', 'torchaudio', 'triton', 'transformers',
            'diffusers', 'accelerate', 'numpy', 'vllm', 'xformers')


def parse_os_release(text):
    values = {}
    for line in text.splitlines():
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            values[key] = value.strip().strip('\"\'')
    pretty_version = re.search(r'\b\d+\.\d+(?:\.\d+)?\b', values.get('PRETTY_NAME', ''))
    return {'os_id': values.get('ID'), 'os_version': values.get('VERSION_ID'),
            'os_release': pretty_version.group() if pretty_version else values.get('VERSION_ID')}


def parse_nvidia(text):
    gpus = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) not in (4, 5) or not row[0].strip().isdigit():
            continue
        index, name, driver, capability = (value.strip() for value in row[:4])
        match = re.fullmatch(r'(\d+)\.(\d+)', capability)
        gpus.append({'index': index, 'name': name, 'driver_version': driver,
                     'architecture': 'sm_' + ''.join(match.groups()) if match else None})
        if len(row) == 5:
            gpus[-1]['uuid'] = row[4].strip()
    return gpus


def parse_amd(text):
    data = json.loads(text)
    gpus = []
    for key, value in data.items():
        match = re.fullmatch(r'card(\d+)', key)
        if not match or not isinstance(value, dict):
            continue
        arch = re.search(r'gfx[0-9a-f]+', str(value.get('GFX Version', '')), re.I)
        gpus.append({'index': match.group(1),
                     'name': value.get('Card Series') or value.get('Card Model'),
                     'architecture': arch.group().lower() if arch else None,
                     'driver_version': None})
    return sorted(gpus, key=lambda gpu: int(gpu['index']))


def collect_host(backend='auto', gpu_ids=None):
    if backend not in ('auto', 'amd', 'nvidia'):
        raise ValueError('backend must be auto, amd, or nvidia')
    warnings = []

    def run(args):
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=15,
                                    check=False)
            if result.returncode:
                warnings.append('{}: {}'.format(args[0], result.stderr.strip() or
                                               'exit {}'.format(result.returncode)))
                return ''
            return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            warnings.append('{}: {}'.format(args[0], exc))
            return ''

    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError as exc:
            warnings.append('{}: {}'.format(path, exc))
            return ''

    result = {'system': platform.system(), 'machine': platform.machine(),
              'kernel': platform.release(), 'python_version': platform.python_version(),
              'glibc_version': None, 'inspection_warnings': warnings}
    libc, version = platform.libc_ver()
    if libc == 'glibc':
        result['glibc_version'] = version
    result.update(parse_os_release(read('/etc/os-release')))
    detected = {}
    if backend in ('auto', 'nvidia'):
        detected['nvidia'] = parse_nvidia(run([
            'nvidia-smi', '--query-gpu=index,name,driver_version,compute_cap,uuid',
            '--format=csv,noheader,nounits']))
    if backend in ('auto', 'amd'):
        raw = run(['rocm-smi', '--showproductname', '--json'])
        try:
            detected['amd'] = parse_amd(raw) if raw else []
        except (ValueError, AttributeError) as exc:
            warnings.append('rocm-smi product data: {}'.format(exc))
            detected['amd'] = []
    if backend == 'auto':
        vendors = [vendor for vendor, devices in detected.items() if devices]
        if len(vendors) > 1:
            raise ValueError('Both AMD and NVIDIA GPUs detected; select --backend explicitly')
        backend = vendors[0] if vendors else None
        if backend is None:
            warnings.append('No supported GPU detected; select --backend explicitly if tools are unavailable')
    result['backend'] = backend
    gpus = detected.get(backend, [])
    mask_name = 'HIP_VISIBLE_DEVICES' if backend == 'amd' else 'CUDA_VISIBLE_DEVICES'
    mask = os.environ.get(mask_name)
    if gpu_ids is None and mask is not None:
        gpu_ids = [] if mask.strip() in ('', '-1') else mask.split(',')
        translated = []
        for identifier in gpu_ids:
            identifier = identifier.strip()
            matches = [gpu['index'] for gpu in gpus if gpu.get('uuid', '').startswith(identifier)]
            if identifier.isdigit():
                translated.append(identifier)
            elif len(matches) == 1:
                translated.append(matches[0])
            else:
                raise ValueError('Cannot map inherited GPU visibility to a physical device: ' + identifier)
        gpu_ids = translated
    if gpu_ids is not None:
        requested = [str(index) for index in gpu_ids]
        if any(not index.isdigit() for index in requested) or len(requested) != len(set(requested)):
            raise ValueError('GPU selection requires distinct nonnegative physical indices')
        selected = set(requested)
        missing = selected - {gpu['index'] for gpu in gpus}
        if missing:
            raise ValueError('Requested GPU indices not detected: ' + ', '.join(sorted(missing)))
        by_index = {gpu['index']: gpu for gpu in gpus}
        gpus = [by_index[index] for index in requested]
    result['gpus'] = gpus
    if backend == 'amd':
        result['rocm_version'] = (read('/opt/rocm/.info/version')
                                  or read('/opt/rocm/core-10.0/.info/version')
                                  or read('/opt/rocm-7.2.0/core-10.0/.info/version')
                                  or None)
        if not result['rocm_version']:
            result['rocm_version'] = run(['dpkg-query', '-W', '-f=${Version}', 'rocm-core']) or None
        result['amdgpu_module_version'] = run(['modinfo', '-F', 'version', 'amdgpu']) or None
        smi_version = run(['rocm-smi', '--version'])
        match = re.search(r'ROCM-SMI version:\s*([^\s]+)', smi_version)
        lib_match = re.search(r'ROCM-SMI-LIB version:\s*([^\s]+)', smi_version)
        result['rocm_smi_version'] = match.group(1) if match else None
        result['rocm_smi_lib_version'] = lib_match.group(1) if lib_match else None
        result['amdgpu_driver_version'] = None
        raw = run(['rocm-smi', '--showdriverversion', '--json'])
        try:
            drivers = json.loads(raw) if raw else {}
            system_driver = drivers.get('system', {}).get('Driver version')
            result['amdgpu_driver_version'] = system_driver
            for gpu in gpus:
                gpu['driver_version'] = drivers.get('card' + gpu['index'], {}).get('Driver version') or system_driver
        except (ValueError, AttributeError) as exc:
            warnings.append('rocm-smi driver data: {}'.format(exc))
        if any(not gpu['architecture'] for gpu in gpus):
            info = run(['rocminfo'])
            agent_architectures = re.findall(r'^\s*Name:\s*(gfx[0-9a-f]+)\s*$', info, re.M)
            architectures = set(agent_architectures)
            # A homogeneous inventory needs no ordinal mapping. Mixed agents
            # cannot safely be assigned to rocm-smi indices without topology.
            if len(architectures) == 1 and len(agent_architectures) == len(detected.get('amd', [])):
                architecture = next(iter(architectures))
                if all(not gpu['architecture'] or gpu['architecture'] == architecture
                       for gpu in detected.get('amd', [])):
                    for gpu in gpus:
                        if not gpu['architecture']:
                            gpu['architecture'] = architecture
            if any(not gpu['architecture'] for gpu in gpus):
                warnings.append('AMD architecture unavailable: rocminfo agent order cannot safely identify GPU indices')
    result['installed_packages'] = {}
    for package in PACKAGES:
        try:
            result['installed_packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['installed_packages'][package] = None
    for gpu in gpus:
        if not gpu['architecture']:
            warnings.append('GPU {} architecture unavailable'.format(gpu['index']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('auto', 'amd', 'nvidia'), default='auto')
    parser.add_argument('--gpus', help='Comma-separated numeric GPU indices')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args()
    ids = None
    if args.gpus is not None:
        ids = args.gpus.split(',')
        if not all(re.fullmatch(r'\d+', index) for index in ids):
            parser.error('--gpus requires comma-separated numeric indices')
        ids = [str(int(index)) for index in ids]
    try:
        report = collect_host(args.backend, ids)
    except ValueError as exc:
        parser.error(str(exc))
    output = json.dumps(report, indent=2) + '\n'
    if args.json_out:
        args.json_out.write_text(output)
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
