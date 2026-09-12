#!/usr/bin/env python3
"""Execute every configured optional workload and repetition into one run folder."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from config_utils import load_config
from gpu_platform import NoGpuError, detect_backend
from suite_config import optional_jobs, validate_optional_suites

ROOT = Path(__file__).resolve().parent


def execute_jobs(cfg, run_dir, runner=subprocess.run):
    errors = validate_optional_suites(cfg)
    if errors:
        raise ValueError('; '.join(errors))
    (run_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (run_dir / 'results').mkdir(parents=True, exist_ok=True)
    destination = run_dir / 'results/metrics.jsonl'
    repeats = int(cfg.get('repeat', 1))
    failures = 0
    try:
        backend = detect_backend(cfg.get('gpu_backend', 'auto'))
    except NoGpuError:
        # Standalone CPU dependency checks may intentionally emit only skipped rows.
        backend = 'unknown'
    for repeat in range(1, repeats + 1):
        for index, job in enumerate(optional_jobs(cfg)):
            name = f"{job['suite']}_{index + 1}_r{repeat}"
            print(f'[RUN] {name}', flush=True)
            with tempfile.TemporaryDirectory(prefix='benchmark-metrics-') as directory:
                metrics = Path(directory) / 'metrics.jsonl'
                command = [sys.executable, str(ROOT / 'benchmarks' / job['script']),
                           *job['args'], '--metrics-path', str(metrics)]
                rows = []
                with (run_dir / 'logs' / f'{name}.log').open('w') as log:
                    result = runner(command, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT, check=False)
                if metrics.exists():
                    try:
                        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
                        if any(not isinstance(row, dict) for row in rows):
                            rows = []
                    except (ValueError, OSError):
                        rows = []
                if not rows:
                    rows = [dict(suite=job['suite'], status='failed',
                                 error='benchmark exited without valid metric rows', exit_code=result.returncode)]
                if result.returncode or any(row.get('status') == 'failed' for row in rows):
                    failures += 1
                    if result.returncode and all(row.get('status') == 'ok' for row in rows):
                        for row in rows:
                            row.update(status='failed', error='benchmark exited unsuccessfully', exit_code=result.returncode)
                with destination.open('a') as stream:
                    for row in rows:
                        row.setdefault('gpu_backend', backend)
                        row.update(repeat_index=repeat, repeat_count=repeats,
                                   benchmark_profile=cfg.get('benchmark_profile', 'general'))
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    raise SystemExit(execute_jobs(load_config(args.config), Path(args.run_dir).resolve()))


if __name__ == '__main__':
    main()
