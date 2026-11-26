#!/usr/bin/env python3
"""Sample GPU-only PearlHash throughput from a bounded SRBMiner session."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import tarfile
import tempfile
import time
from urllib.request import ProxyHandler, build_opener, urlopen
from uuid import uuid4

if __package__:
    from .energy import EnergySampler
else:
    from energy import EnergySampler


DEFAULT_WALLET = 'prl1p7zz6lr8ydjdsznwagr5xf08ry4z23rpzmqmalzrg4pkjrcm8xl2s5jegp4'
RELEASE_REPO = 'doktor83/SRBMiner-Multi'
ANSI_ESCAPE = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
MINER_DEVICE = re.compile(
    r'^\s*GPU(\d+)\s+(?:\[[^\]]+\]\s*)*?\[([\da-f]+:[\da-f]{2}:[\da-f]{2}\.[0-7])\]',
    re.IGNORECASE | re.MULTILINE,
)


def sha256_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resolve_miner(metrics_path):
    override = os.environ.get('SRBMINER_BIN')
    if override:
        return Path(override).expanduser().resolve(strict=True), None

    # Optional jobs run sequentially. Keep one release for both modes and all
    # repeats of this benchmark run, even if upstream publishes a new release.
    directory = Path(metrics_path).resolve().parent
    cache = directory / '.srbminer'
    if not cache.exists():
        with urlopen(f'https://api.github.com/repos/{RELEASE_REPO}/releases/latest', timeout=20) as response:
            release = json.load(response)
        version = release['tag_name']
        if not re.fullmatch(r'\d+(?:\.\d+)+', version):
            raise ValueError(f'Unexpected SRBMiner release tag: {version}')
        archive_name = f'SRBMiner-Multi-{version.replace(".", "-")}-Linux.tar.gz'
        url = f'https://github.com/{RELEASE_REPO}/releases/download/{version}/{archive_name}'
        asset = next((asset for asset in release['assets'] if asset['name'] == archive_name), None)
        digest = asset.get('digest') if asset and asset['browser_download_url'] == url else None
        if not digest or not re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
            raise ValueError('Official SRBMiner Linux release has no verified SHA-256 metadata')

        with tempfile.TemporaryDirectory(prefix='.srbminer-', dir=directory) as temporary:
            stage = Path(temporary) / 'ready'
            stage.mkdir()
            archive_path = Path(temporary) / archive_name
            with urlopen(url, timeout=60) as response, archive_path.open('wb') as stream:
                shutil.copyfileobj(response, stream)
            if sha256_file(archive_path) != digest.removeprefix('sha256:'):
                raise ValueError('SRBMiner release archive SHA-256 mismatch')
            binary = stage / 'SRBMiner-MULTI'
            member_name = f'SRBMiner-Multi-{version.replace(".", "-")}/SRBMiner-MULTI'
            with tarfile.open(archive_path, 'r:gz') as archive:
                member = archive.getmember(member_name)
                if not member.isfile():
                    raise ValueError('SRBMiner archive executable is not a regular file')
                with archive.extractfile(member) as source, binary.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
            binary.chmod(0o700)
            (stage / 'release').write_text(version)
            stage.rename(cache)

    return (cache / 'SRBMiner-MULTI').resolve(strict=True), (cache / 'release').read_text()


def pci_key(address):
    return tuple(int(field, 16) for field in re.split(r'[:.]', address.strip()))


def map_pci_devices(listing, inventory, physical_ids):
    by_pci = {}
    for miner_id, address in MINER_DEVICE.findall(ANSI_ESCAPE.sub('', listing)):
        by_pci.setdefault(pci_key(address), set()).add(int(miner_id))
    result = []
    for physical_id in physical_ids:
        address = inventory.get(physical_id)
        matches = by_pci.get(pci_key(address), set()) if address else set()
        if len(matches) != 1 or next(iter(matches)) in result:
            raise ValueError(f'Cannot uniquely map physical GPU {physical_id} by PCI address; check miner visibility or set PEARL_GPU_MAP')
        result.append(next(iter(matches)))
    return result


def physical_pci_inventory(backend):
    if backend == 'nvidia':
        command = ['nvidia-smi', '--query-gpu=index,pci.bus_id', '--format=csv,noheader,nounits']
        output = subprocess.check_output(command, text=True, timeout=10)
        return {int(index): address.strip() for line in output.splitlines()
                if line.count(',') == 1 for index, address in [line.split(',')]}
    if backend == 'amd':
        output = subprocess.check_output(['rocm-smi', '--showbus', '--json'], text=True, timeout=10)
        return {int(card[4:]): fields['PCI Bus'] for card, fields in json.loads(output).items()
                if re.fullmatch(r'card\d+', card) and isinstance(fields, dict) and 'PCI Bus' in fields}
    raise ValueError(f'Unsupported PearlHash GPU backend: {backend}')


def miner_gpu_flags(backend):
    if backend == 'nvidia':
        return ['--disable-gpu-amd', '--disable-gpu-intel']
    if backend == 'amd':
        return ['--disable-gpu-nvidia', '--disable-gpu-intel']
    return []


def resolve_miner_ids(binary, backend, physical_ids):
    mapping = os.environ.get('PEARL_GPU_MAP')
    if not mapping:
        # SRBMiner emits its listing only on a terminal. Use a PTY so redirected
        # benchmark logs do not suppress the GPU/PCI records.
        command = [str(binary), '--list-devices', '--disable-cpu', *miner_gpu_flags(backend)]
        with tempfile.TemporaryDirectory(prefix='pearlhash-devices-') as directory:
            listing = subprocess.check_output(['script', '-qec', shlex.join(command), '/dev/null'],
                                              cwd=directory, stdin=subprocess.DEVNULL,
                                              text=True, timeout=20)
        return map_pci_devices(listing, physical_pci_inventory(backend), physical_ids)
    if not re.fullmatch(r'[0-9]+:[0-9]+(?:,[0-9]+:[0-9]+)*', mapping):
        raise ValueError('PEARL_GPU_MAP must contain physical:miner ID pairs')
    pairs = [tuple(map(int, pair.split(':'))) for pair in mapping.split(',')]
    gpu_map = dict(pairs)
    if len(gpu_map) != len(pairs) or len(set(gpu_map.values())) != len(pairs):
        raise ValueError('PEARL_GPU_MAP must map distinct physical and miner IDs')
    if any(i not in gpu_map for i in physical_ids):
        raise ValueError('PEARL_GPU_MAP is missing a selected GPU')
    return [gpu_map[i] for i in physical_ids]


def read_sample(payload, miner_ids):
    algorithm = next((a for a in payload['algorithms'] if a['name'] == 'pearlhash'), None)
    if algorithm is None:
        raise ValueError('PearlHash stats are missing')
    rates = algorithm['hashrate']['gpu']
    devices = [f'gpu{i}' for i in miner_ids]
    if any(key.startswith('gpu') and key not in devices and value
           for key, value in rates.items()):
        raise ValueError('Unexpected GPU is actively mining')
    values = [rates[key] for key in devices]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in values):
        raise ValueError('Invalid GPU hashrate')
    shares = algorithm['shares']
    counters = (shares['accepted'], shares['rejected'])
    if any(type(value) is not int or value < 0 for value in counters):
        raise ValueError('Invalid share counters')
    return values, *counters


def summarize(samples):
    for previous, current in zip(samples, samples[1:]):
        if current[1] < previous[1] or current[2] < previous[2]:
            raise ValueError('Miner share counters reset during measurement')
    rates = [statistics.mean(values) for values in zip(*(sample[0] for sample in samples))]
    if not all(rates):
        raise ValueError('At least one requested GPU produced no hashrate')
    return {
        'hashrate_hs': sum(rates),
        'per_gpu_hashrate_hs': rates,
        'accepted_shares': samples[-1][1] - samples[0][1],
        'rejected_shares': samples[-1][2] - samples[0][2],
    }


def measure(process, url, token, args, physical_ids, miner_ids):
    opener = build_opener(ProxyHandler({}))

    def sample():
        if process.poll() is not None:
            raise RuntimeError(f'SRBMiner exited with code {process.returncode}')
        with opener.open(url, timeout=3) as response:
            payload = json.load(response)
        if payload.get('rig_name') != token:
            raise ValueError('Statistics endpoint belongs to another miner session')
        return read_sample(payload, miner_ids), payload['miner_version']

    deadline = time.perf_counter() + args.startup_timeout
    while True:
        try:
            first, version = sample()
            if all(first[0]):
                break
            detail = 'at least one GPU has zero hashrate'
        except (OSError, KeyError, ValueError, TypeError) as exc:
            detail = str(exc)
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise RuntimeError(f'Timed out waiting for active PearlHash statistics ({detail})')
        time.sleep(min(1, remaining))

    try:
        exit_code = process.wait(timeout=args.warmup)
    except subprocess.TimeoutExpired:
        pass
    else:
        raise RuntimeError(f'SRBMiner exited during warm-up with code {exit_code}')
    power = (EnergySampler(args.backend, physical_device_ids=physical_ids)
             if args.backend in ('nvidia', 'amd') else None)
    if power:
        power.start()
    try:
        started = time.perf_counter()
        samples = [sample()[0]]
        for offset in range(args.interval, args.duration, args.interval):
            time.sleep(max(0, started + offset - time.perf_counter()))
            samples.append(sample()[0])
        time.sleep(max(0, started + args.duration - time.perf_counter()))
        samples.append(sample()[0])
        result = summarize(samples)
    finally:
        if power:
            telemetry = power.stop(started_s=started)
    if power:
        result.update(telemetry)
        watts = telemetry['mean_power_w']
        result['hashes_per_joule'] = result['hashrate_hs'] / watts if watts else None
    return dict(result, miner_version=version)


def run(args, row):
    if not args.physical_gpus:
        row.update(status='skipped', reason='No resolved physical GPUs in the run config')
        return
    wallet = os.environ.get('PEARL_WALLET') or DEFAULT_WALLET
    physical_ids = [int(value) for value in args.physical_gpus.split(',')]
    binary, release = resolve_miner(args.metrics_path)
    row['miner_sha256'] = sha256_file(binary)
    if release:
        row['miner_release'] = release
    miner_ids = resolve_miner_ids(binary, args.backend, physical_ids)
    worker = re.sub(r'[^a-zA-Z0-9_-]', '_', socket.gethostname())
    if not re.fullmatch(r'prl1[a-z0-9]+', wallet):
        raise ValueError('PEARL_WALLET must be a Pearl address without a worker suffix')
    token = f'pearlhash-{uuid4().hex}'
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    command = [
        str(binary), '--disable-cpu', '--algorithm', 'pearlhash', '--pool', args.pool,
        '--wallet', f'{wallet}.{worker}', '--tls', 'true', '--gpu-id',
        ','.join(map(str, miner_ids)), '--api-enable', '--api-port', str(port),
        '--api-rig-name', token, '--autotune-no-load',
    ]
    command.extend(miner_gpu_flags(args.backend))
    with tempfile.TemporaryDirectory(prefix='pearlhash-') as directory:
        with subprocess.Popen(command, cwd=directory, stdin=subprocess.DEVNULL) as process:
            try:
                result = measure(process, f'http://127.0.0.1:{port}/', token, args, physical_ids, miner_ids)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    row.update(result, status='ok', gpu_count=len(miner_ids),
               physical_gpu_ids=physical_ids, miner_gpu_ids=miner_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('warmup', 'duration', 'interval', 'startup-timeout'):
        parser.add_argument(f'--{key}', required=True, type=int)
    for key in ('pool', 'mode', 'backend', 'metrics-path'):
        parser.add_argument(f'--{key}', required=True)
    parser.add_argument('--physical-gpus', default='')
    args = parser.parse_args()
    if args.warmup < 0 or min(args.duration, args.interval, args.startup_timeout) < 1:
        parser.error('Invalid timing settings')

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    row = dict(suite='pearlhash', algorithm='pearlhash', mode=args.mode, pool=args.pool,
               warmup_s=args.warmup, duration_s=args.duration, sample_interval_s=args.interval)
    try:
        run(args, row)
    except (OSError, RuntimeError, KeyError, ValueError, TypeError, tarfile.TarError, subprocess.SubprocessError) as exc:
        row.update(status='failed', error=str(exc))
    with Path(args.metrics_path).open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
    return int(row['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
