#!/usr/bin/env python3
"""Schedule the additional standard inference cases and capture their results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

from benchmarks.benchmark_protocol import resolve_revision
from config_utils import load_config

ROOT = Path(__file__).resolve().parent
VLLM_VERSION = '0.34.0'


def inference_cases(cfg, gpu_count):
    if not (cfg.get('llm_infer') or {}).get('enabled', True):
        return []
    smoke = cfg.get('smoke_mode', False)
    baseline = cfg.get('benchmark_profile') == 'single_gpu_baseline'
    cases = []
    for entry in cfg.get('llm_infer_matrix', []):
        common = dict(model=entry['model'], revision=entry.get('revision', 'main'),
                      prompt_len=64 if smoke else 512, output_len=32 if smoke else 128,
                      batch_sizes=[1] if smoke else [1, 4])
        if 'tensor_parallel_sizes' in entry:
            if baseline:
                continue
            sizes = entry['tensor_parallel_sizes']
            for size in sizes[:1] if smoke else sizes:
                cases.append(dict(common, backend='vllm', dtype='bfloat16',
                                  multi_gpu_mode='tensor_parallel', tensor_parallel_sizes=[size],
                                  vllm_version=VLLM_VERSION, max_model_len=1024,
                                  enable_prefix_caching=False, sample_vram=True))
        else:
            modes = ['single', 'replicated'] if gpu_count > 1 and not (smoke or baseline) else ['single']
            for image_size in entry.get('image_sizes', [0])[:1 if smoke else None]:
                for mode in modes:
                    case = dict(common, backend='transformers', dtype='float16',
                                tensor_parallel_sizes=[1], multi_gpu_mode=mode)
                    if image_size:
                        case['image_size'] = image_size
                    cases.append(case)
    return cases


def matrix_preflight(host_path, log_path):
    command = [sys.executable, str(ROOT / 'setup_vllm.py'), '--version', VLLM_VERSION,
               '--venv', str(ROOT / f'.venv-vllm-{VLLM_VERSION}'), '--host-json', str(host_path)]
    override = os.environ.get('VLLM_PYTHON')
    python = str(Path(override).absolute()) if override else str(ROOT / f'.venv-vllm-{VLLM_VERSION}/bin/python')
    if override:
        command.extend(['--python', python])
    with log_path.open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        return None
    return python


def prepare_runtime(cfg, run_dir):
    gpu_count = len(cfg.get('gpu_include') or [])
    cases = inference_cases(cfg, gpu_count)
    python = None
    if cfg.get('gpu_backend') == 'nvidia' and any(
        case['backend'] == 'vllm' and case['tensor_parallel_sizes'][0] <= gpu_count for case in cases
    ):
        python = matrix_preflight(run_dir / 'host_inventory.json',
                                  run_dir / 'logs/inference_matrix_vllm_preflight.log')
        if not python:
            print('[WARN] Large-model vLLM cases will be skipped; see logs/inference_matrix_vllm_preflight.log')
    (run_dir / 'inference_matrix_runtime.json').write_text(json.dumps({'vllm_python': python}) + '\n')


def execute(cfg, run_dir, warmup, duration):
    selected = cfg.get('gpu_include') or []
    cases = inference_cases(cfg, len(selected))
    if not cases:
        return 0
    backend = cfg.get('gpu_backend')
    runtime_path = run_dir / 'inference_matrix_runtime.json'
    vllm_python = json.loads(runtime_path.read_text())['vllm_python']
    revisions = {}
    failures = False
    repeat_count = cfg.get('repeat', 1)
    with (run_dir / 'results/metrics.jsonl').open('a') as destination:
        for index, case in enumerate(cases, start=1):
            tp = case['tensor_parallel_sizes'][0]
            name = f'llm_infer_matrix_{index}'
            common = dict(suite='llm_infer', gpu_backend=backend, model=case['model'],
                          dtype=case['dtype'], backend=case['backend'], tensor_parallel=tp,
                          multi_gpu_mode=case['multi_gpu_mode'])
            if case.get('image_size'):
                common['image_size'] = case['image_size']
            reason = None
            if case['backend'] == 'vllm':
                if backend != 'nvidia':
                    reason = 'The pinned vLLM matrix deployment uses CUDA'
                elif tp > len(selected):
                    reason = f'Tensor parallel size {tp} requires at least {tp} selected GPUs'
                elif not vllm_python:
                    reason = 'vLLM preflight failed; see logs/inference_matrix_vllm_preflight.log'
            key = (case['model'], case['revision'])
            if reason is None and key not in revisions:
                try:
                    revisions[key] = (resolve_revision(*key), None)
                except Exception as exc:
                    response = getattr(exc, 'response', None)
                    gated = (type(exc).__name__ == 'GatedRepoError'
                             or getattr(response, 'status_code', None) in (401, 403))
                    status = 'skipped' if gated else 'failed'
                    detail = ('Checkpoint access requires Hugging Face authorization'
                              if gated else f'Checkpoint revision could not be resolved ({type(exc).__name__})')
                    revisions[key] = (None, dict(status=status, detail=detail))
            if reason is not None:
                setup_row = dict(status='skipped', detail=reason)
            else:
                revision, setup_row = revisions[key]
                if revision:
                    case = dict(case, revision=revision)
            for repeat in range(1, repeat_count + 1):
                log_path = run_dir / 'logs' / f'{name}_r{repeat}.log'
                if setup_row:
                    rows = [setup_row]
                    log_path.write_text(setup_row['detail'] + '\n')
                else:
                    script = 'llm_infer_vllm.py' if case['backend'] == 'vllm' else 'llm_infer_hf.py'
                    python = vllm_python if case['backend'] == 'vllm' else sys.executable
                    print(f"[RUN] {case['model']} {case['multi_gpu_mode']} TP={tp} → {log_path}", flush=True)
                    with tempfile.TemporaryDirectory(prefix='inference-matrix-') as directory:
                        temp = Path(directory)
                        config_path = temp / 'config.yaml'
                        config_path.write_text(yaml.safe_dump({'llm_infer': case}))
                        with log_path.open('w') as log:
                            result = subprocess.run(
                                [python, str(ROOT / 'benchmarks' / script), '--config', str(config_path),
                                 '--warmup', str(warmup), '--duration', str(duration)],
                                cwd=temp, stdout=log, stderr=subprocess.STDOUT, check=False)
                        metrics = temp / 'results/metrics.jsonl'
                        try:
                            rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
                            if any(not isinstance(row, dict) or row.get('status') not in ('ok', 'failed', 'skipped')
                                   for row in rows):
                                rows = []
                        except (OSError, ValueError):
                            rows = []
                        if not rows or (result.returncode and all(row['status'] == 'ok' for row in rows)):
                            if result.returncode:
                                rows = []
                            rows.append(dict(status='failed', detail=f'Runner exited with code {result.returncode}; see {log_path.name}'))
                for row in rows:
                    failures |= row['status'] == 'failed'
                    destination.write(json.dumps({**common, **row, 'repeat_index': repeat,
                                                 'repeat_count': repeat_count,
                                                 'benchmark_profile': cfg.get('benchmark_profile', 'general')}) + '\n')
                destination.flush()
    return int(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--prepare-runtime', action='store_true')
    parser.add_argument('--warmup', default=5, type=int)
    parser.add_argument('--duration', default=30, type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = args.run_dir.resolve()
    if args.prepare_runtime:
        prepare_runtime(cfg, run_dir)
        return 0
    return execute(cfg, run_dir, args.warmup, args.duration)


if __name__ == '__main__':
    raise SystemExit(main())
