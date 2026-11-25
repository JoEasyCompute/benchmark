#!/usr/bin/env python3
"""Sample GPU-only PearlHash throughput from a bounded SRBMiner session."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import tempfile
import time
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

if __package__:
    from .energy import EnergySampler
else:
    from energy import EnergySampler


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
    executable = os.environ.get('SRBMINER_BIN')
    wallet = os.environ.get('PEARL_WALLET')
    mapping = os.environ.get('PEARL_GPU_MAP')
    if not all((executable, wallet, mapping)):
        row.update(status='skipped', reason='Set SRBMINER_BIN, PEARL_WALLET and PEARL_GPU_MAP')
        return
    if not args.physical_gpus:
        row.update(status='skipped', reason='No resolved physical GPUs in the run config')
        return
    if not re.fullmatch(r'[0-9]+:[0-9]+(?:,[0-9]+:[0-9]+)*', mapping):
        raise ValueError('PEARL_GPU_MAP must contain physical:miner ID pairs')
    pairs = [tuple(map(int, pair.split(':'))) for pair in mapping.split(',')]
    gpu_map = dict(pairs)
    if len(gpu_map) != len(pairs) or len(set(gpu_map.values())) != len(pairs):
        raise ValueError('PEARL_GPU_MAP must map distinct physical and miner IDs')
    physical_ids = [int(value) for value in args.physical_gpus.split(',')]
    if any(i not in gpu_map for i in physical_ids):
        raise ValueError('PEARL_GPU_MAP is missing a selected GPU')
    miner_ids = [gpu_map[i] for i in physical_ids]
    binary = Path(executable).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with binary.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    checksum = digest.hexdigest()
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
    if args.backend in ('nvidia', 'amd'):
        command.append('--disable-gpu-amd' if args.backend == 'nvidia' else '--disable-gpu-nvidia')
        command.append('--disable-gpu-intel')
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
    row.update(result, status='ok', miner_sha256=checksum, gpu_count=len(miner_ids),
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
    except (OSError, RuntimeError, KeyError, ValueError, TypeError) as exc:
        row.update(status='failed', error=str(exc))
    with Path(args.metrics_path).open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
    return int(row['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
