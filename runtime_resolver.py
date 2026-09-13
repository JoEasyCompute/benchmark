#!/usr/bin/env python3
"""Select a curated GPU package set; never infer ROCm compatibility from one driver number."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys

from inspect_runtime import collect_host

CATALOG_VERSION = '2026-09-13.1'
COMMON = {'transformers': '4.57.0', 'diffusers': '0.29.2', 'accelerate': '1.10.1', 'numpy': '1.26.4'}
PROFILE_IDS = ('torch291-cu128', 'torch291-cu126', 'torch291-rocm72', 'existing-torch28-rocm64-compat')
SOURCES = {
    'torch': 'https://pytorch.org/get-started/previous-versions/#v291',
    'cuda128': 'https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html',
    'cuda126': 'https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html',
    'amd': 'https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/native_linux/install-pytorch.html',
    'amd_matrix': 'https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html',
}


def version(value):
    match = re.match(r'^(\d+)\.(\d+)(?:\.(\d+))?', str(value or ''))
    return tuple(int(part or 0) for part in match.groups()) if match else None


def make_profile(identifier, python_version):
    expected = dict(COMMON)
    if identifier in ('torch291-cu126', 'torch291-cu128'):
        tag = identifier.split('-')[1]
        expected.update(torch='2.9.1+' + tag, torchvision='0.24.1+' + tag,
                        torchaudio='2.9.1+' + tag, triton='3.5.1')
        install = dict(index_url='https://download.pytorch.org/whl/' + tag,
                       packages=[f'{name}=={expected[name]}' for name in ('torch', 'torchvision', 'torchaudio')])
        return dict(id=identifier, backend='nvidia', torch_version='2.9.1',
                    runtime_version='12.8' if tag == 'cu128' else '12.6',
                    minimum_driver='570.26.0' if tag == 'cu128' else '560.28.3',
                    expected_packages=expected, core_install=install,
                    sources=[SOURCES['torch'], SOURCES['cuda128' if tag == 'cu128' else 'cuda126']])
    if identifier == 'torch291-rocm72':
        abi = 'cp310' if version(python_version) and version(python_version)[:2] == (3, 10) else 'cp312'
        core = dict(torch='2.9.1+rocm7.2.0.lw.git7e1940d4',
                    torchvision='0.24.0+rocm7.2.0.gitb919bd0c',
                    torchaudio='2.9.0+rocm7.2.0.gite3c6ee2b',
                    triton='3.5.1+rocm7.2.0.gita272dfa8')
        expected.update(core)
        urls = [f'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/{name}-{value.replace("+", "%2B")}-{abi}-{abi}-linux_x86_64.whl'
                for name, value in core.items()]
        return dict(id=identifier, backend='amd', torch_version='2.9.1', runtime_version='7.2', rocm_release='7.2.0',
                    expected_packages=expected, core_install=dict(packages=urls),
                    sources=[SOURCES['amd'], SOURCES['amd_matrix']])
    if identifier == 'existing-torch28-rocm64-compat':
        return dict(id=identifier, backend='amd', torch_version='2.8.0', runtime_version='10.0',
                    expected_packages={}, core_install={'packages': []},
                    sources=[SOURCES['amd_matrix']], experimental=True)
    raise ValueError(f'Unknown runtime profile {identifier!r}; choose auto or one of {PROFILE_IDS}')


def resolve_runtime(host, profile='auto', allow_unverified_host=False):
    host = deepcopy(host)
    errors, warnings, host_exceptions = [], list(host.get('inspection_warnings', [])), []
    backend = host.get('backend')
    gpus = host.get('gpus') or []
    candidate = None
    if backend not in ('nvidia', 'amd') or not gpus:
        errors.append('No supported GPU inventory; inspect drivers and select an explicit backend on mixed hosts.')
    if host.get('system') != 'Linux' or host.get('machine') != 'x86_64':
        errors.append('This catalogue requires Linux x86_64 wheels.')
    py = version(host.get('python_version'))
    if not py or py[:2] not in ((3, 10), (3, 11), (3, 12)):
        errors.append('Select Python 3.10, 3.11 or 3.12; current interpreter is outside this catalogue.')
    libc = version(host.get('glibc_version'))
    if not libc or libc < (2, 28, 0):
        errors.append('The wheel catalogue requires detected glibc >= 2.28.')
    if profile == 'auto' and backend == 'nvidia':
        drivers = [version(gpu.get('driver_version')) for gpu in gpus]
        newer_arch = any(gpu.get('architecture') in ('sm_100', 'sm_101', 'sm_120', 'sm_121') for gpu in gpus)
        profile = 'torch291-cu128' if newer_arch or all(d and d >= (570, 26, 0) for d in drivers) else 'torch291-cu126'
    elif profile == 'auto' and backend == 'amd':
        installed_torch = str((host.get('installed_packages') or {}).get('torch') or '')
        if version(host.get('rocm_version')) and version(host.get('rocm_version'))[0] >= 10 and installed_torch.startswith('2.8.0+rocm6.4'):
            profile = 'existing-torch28-rocm64-compat'
        else:
            profile = 'torch291-rocm72'
    if profile != 'auto':
        try:
            candidate = make_profile(profile, host.get('python_version'))
        except ValueError as exc:
            errors.append(str(exc))
    if candidate and candidate['backend'] != backend:
        errors.append('Requested profile backend does not match the detected GPU vendor.')
    elif candidate and backend == 'nvidia':
        supported = {'sm_75', 'sm_80', 'sm_86', 'sm_89', 'sm_90'}
        if candidate['runtime_version'] == '12.8':
            supported.update(('sm_100', 'sm_120'))
        for gpu in gpus:
            driver = version(gpu.get('driver_version'))
            if not driver or driver < version(candidate['minimum_driver']):
                errors.append(f"GPU {gpu.get('index')}: driver {gpu.get('driver_version')} is below the catalogue's conservative {candidate['minimum_driver']} requirement.")
            if gpu.get('architecture') not in supported:
                errors.append(f"GPU {gpu.get('index')}: architecture {gpu.get('architecture')} is not covered by {candidate['id']}.")
    elif candidate and backend == 'amd':
        rocm = version(host.get('rocm_version'))
        if candidate.get('experimental'):
            host_exceptions.append('Existing Torch 2.8/ROCm 6.4 wheel build is running against a ROCm 10 user-space stack; this compatibility profile is experimental and requires numerical validation.')
        elif rocm != (7, 2, 0):
            errors.append('This Radeon bundle is qualified against ROCm 7.2.0; no automatic fallback from another ROCm release.')
        if not host.get('amdgpu_module_version') and not any(gpu.get('driver_version') for gpu in gpus):
            errors.append('Cannot identify the loaded AMDGPU driver/module.')
        for gpu in gpus:
            if gpu.get('architecture') not in ('gfx1100', 'gfx1101', 'gfx1200', 'gfx1201'):
                errors.append(f"GPU {gpu.get('index')}: unsupported Radeon architecture {gpu.get('architecture')}.")
        os_version = host.get('os_version')
        if host.get('os_id') != 'ubuntu' or os_version not in ('22.04', '24.04'):
            errors.append('The Radeon bundle catalogue currently supports Ubuntu 22.04 and 24.04 only.')
        else:
            expected_py = (3, 12) if os_version == '24.04' else (3, 10)
            if not py or py[:2] != expected_py:
                errors.append(f'Ubuntu {os_version} Radeon wheels require Python {expected_py[0]}.{expected_py[1]}.')
            expected_kernel = (6, 14) if os_version == '24.04' else (6, 8)
            actual_kernel = version(host.get('kernel'))
            if not actual_kernel or actual_kernel[:2] != expected_kernel:
                host_exceptions.append(f"Ubuntu {os_version} kernel {host.get('kernel')} is outside the published ROCm 7.2 kernel {expected_kernel[0]}.{expected_kernel[1]} combination.")
    installed = host.get('installed_packages') or {}
    if installed.get('vllm') == '0.11.0' or installed.get('xformers') == '0.0.32.post1':
        errors.append('Legacy vLLM/xFormers pins target Torch 2.8. Use a new VENV_DIR for this Torch 2.9.1 profile; provider environments are not modified automatically.')
    warnings.extend(host_exceptions)
    if host_exceptions and not allow_unverified_host:
        errors.extend(host_exceptions)
        errors.append('Review the host mismatch, then use --allow-unverified-host only to qualify it experimentally; drivers/kernels will not be changed.')
    return dict(schema_version=1, catalogue_version=CATALOG_VERSION,
                resolved_at=datetime.now(timezone.utc).isoformat(),
                status='blocked' if errors else 'compatible_with_warnings' if warnings else 'compatible',
                host_qualified=not errors and not host_exceptions,
                allow_unverified_host=allow_unverified_host, host=host, profile=candidate,
                warnings=warnings, errors=errors)


def installed_matches(plan, packages):
    profile = plan.get('profile')
    return bool(profile) and plan.get('status') != 'blocked' and all(
        packages.get(name) == wanted for name, wanted in profile['expected_packages'].items())


def package_inventory():
    return {dist.metadata['Name'].lower().replace('-', '_'): dist.version
            for dist in importlib.metadata.distributions() if dist.metadata.get('Name')}


def install_commands(profile, python):
    core = profile['core_install']
    commands = [[python, '-m', 'pip', 'install', '--upgrade', 'pip', 'wheel']]
    core_args = ['--index-url', core['index_url']] if core.get('index_url') else []
    commands.append([python, '-m', 'pip', 'install', '--force-reinstall', *core_args, *core['packages']])
    commands.append([python, '-m', 'pip', 'install',
                     *[f'{name}=={value}' for name, value in COMMON.items()],
                     'huggingface-hub>=0.34.0,<1.0', 'tokenizers>=0.22.0,<=0.23.0',
                     'safetensors>=0.4.3', 'pandas>=2.2.0,<3', 'PyYAML>=6.0', 'tqdm>=4.66', 'psutil>=5.9.8'])
    if profile['backend'] == 'nvidia':
        commands.append([python, '-m', 'pip', 'install', 'nvidia-ml-py>=12.560.30'])
    commands.append([python, '-m', 'pip', 'check'])
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('auto', 'nvidia', 'amd'), default='auto')
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--profile', default='auto')
    parser.add_argument('--allow-unverified-host', action='store_true')
    parser.add_argument('--host-json', help='Resolve a captured read-only inventory without probing this host')
    parser.add_argument('--json-out')
    parser.add_argument('--check-installed', action='store_true')
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--venv', default=str(Path(__file__).resolve().parent / '.venv'))
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()
    if args.install and args.host_json:
        parser.error('--host-json is planning-only and cannot authorize installation on another host')
    try:
        host = json.loads(Path(args.host_json).read_text()) if args.host_json else collect_host(args.backend, args.gpus.split(',') if args.gpus else None)
        plan = resolve_runtime(host, args.profile, args.allow_unverified_host)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'[RUNTIME][ERROR] {exc}\n')
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(plan, indent=2) + '\n')
    if args.check_installed:
        raise SystemExit(0 if installed_matches(plan, package_inventory()) else 1)
    if not args.install:
        print(json.dumps(plan, indent=2))
        raise SystemExit(2 if plan['status'] == 'blocked' else 0)
    for warning in plan['warnings']:
        print('[RUNTIME][WARN] ' + warning, flush=True)
    if plan['errors']:
        for error in plan['errors']:
            print('[RUNTIME][ERROR] ' + error, file=sys.stderr)
        raise SystemExit(2)
    target = Path(args.venv).resolve()
    python = str(target / 'bin/python')
    if not target.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(target)], check=True)
    if not Path(python).is_file():
        parser.exit(2, '[RUNTIME][ERROR] VENV_DIR exists but is not a Python virtual environment. Choose a new directory.\n')
    # Refuse a target interpreter mismatch rather than putting incompatible wheels in it.
    actual = subprocess.check_output([python, '-c', 'import platform; print(platform.python_version())'], text=True).strip()
    if version(actual)[:2] != version(host['python_version'])[:2]:
        parser.exit(2, '[RUNTIME][ERROR] Target venv Python differs from the resolved wheel ABI. Run setup using the target interpreter.\n')
    lock = target / 'runtime-lock.json'
    lock.write_text(json.dumps(plan, indent=2) + '\n')
    print(f"[RUNTIME] Installing {plan['profile']['id']} into {target}", flush=True)
    for command in install_commands(plan['profile'], python):
        subprocess.run(command, check=True)
    verification = [python, str(Path(__file__).with_name('verify_runtime.py')), '--lock', str(lock),
                    '--json-out', str(target / 'runtime-validation.json')]
    if args.distributed:
        verification.append('--distributed')
    subprocess.run(verification, check=True)
    print('[RUNTIME] Package and numerical validation passed.', flush=True)


if __name__ == '__main__':
    main()
