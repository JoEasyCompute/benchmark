"""Configuration contracts and ordered jobs for the optional benchmark suites."""
from copy import deepcopy
from itertools import product
import math
from urllib.parse import urlparse


OPTIONAL_DEFAULTS = {
    'vision_infer': {'enabled': False, 'model': 'resnet18',
                     'weights': 'ResNet18_Weights.IMAGENET1K_V1',
                     'batch_sizes': [1, 32], 'sizes': [224], 'dtype': 'float32',
                     'modes': ['throughput'], 'iterations': 50, 'warmup': 5, 'min_duration_s': 5.0},
    'kernel_bench': {'enabled': False, 'cases': ['gemm', 'attention', 'memory'],
                     'size': 2048, 'dtype': 'float32', 'iterations': 20, 'warmup': 3,
                     'heads': 8, 'head_dim': 64, 'min_duration_s': 5.0},
    'llm_serve': {'enabled': False, 'provider': 'transformers', 'endpoint': '',
                  'model': '', 'revision': '', 'dtype': 'float16', 'concurrency': [1, 4],
                  'prompt': 'A deterministic benchmark prompt.', 'output_len': 32,
                  'duration': 30, 'warmup': 1, 'timeout': 120},
}


def suite_values(cfg, suite):
    values = deepcopy(OPTIONAL_DEFAULTS[suite])
    values.update(cfg.get(suite) or {})
    if suite == 'llm_serve' and not values['model']:
        values['model'] = (cfg.get('llm_infer') or {}).get('model', '')
    return values


def validate_optional_suites(cfg):
    errors = []
    for suite, defaults in OPTIONAL_DEFAULTS.items():
        raw = cfg.get(suite, {})
        if not isinstance(raw, dict):
            errors.append(f'{suite} must be a mapping')
            continue
        for key in set(raw) - set(defaults):
            errors.append(f'{suite}: unsupported key {key!r}')
        values = suite_values(cfg, suite)
        if not isinstance(values['enabled'], bool):
            errors.append(f'{suite}.enabled must be boolean')
        for key in ('iterations', 'size', 'heads', 'head_dim', 'output_len'):
            if key in values and (type(values[key]) is not int or values[key] <= 0):
                errors.append(f'{suite}.{key} must be a positive integer')
        if 'min_duration_s' in values:
            duration = values['min_duration_s']
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 <= duration <= 3600:
                errors.append(f'{suite}.min_duration_s must be finite and between 0 and 3600')
        if type(values['warmup']) is not int or values['warmup'] < 0:
            errors.append(f'{suite}.warmup must be a non-negative integer')
        for key in ('sizes', 'batch_sizes', 'concurrency'):
            if key in values:
                items = values[key]
                if not isinstance(items, list) or not items or any(type(i) is not int or i <= 0 for i in items):
                    errors.append(f'{suite}.{key} must contain positive integers')
                elif len(items) != len(set(items)):
                    errors.append(f'{suite}.{key} must not contain duplicates')
        if values['dtype'] not in ('float32', 'float16', 'bfloat16'):
            errors.append(f'{suite}.dtype must be float32, float16, or bfloat16')
        for key, choices in (('cases', {'gemm', 'attention', 'memory'}),
                             ('modes', {'throughput', 'latency'})):
            if key in values:
                items = values[key]
                if not isinstance(items, list) or not items or any(not isinstance(i, str) or i not in choices for i in items):
                    errors.append(f'{suite}.{key} must contain supported values {sorted(choices)}')
                elif len(items) != len(set(items)):
                    errors.append(f'{suite}.{key} must not contain duplicates')
        if suite == 'vision_infer':
            if not isinstance(values['model'], str) or not values['model']:
                errors.append('vision_infer.model must be non-empty')
            weights = values['weights']
            if not isinstance(weights, str) or '.' not in weights or weights.endswith('.DEFAULT') or weights == 'DEFAULT':
                errors.append('vision_infer.weights must name an explicit pretrained weights version')
        if suite == 'llm_serve':
            if values['provider'] not in ('transformers', 'vllm'):
                errors.append('llm_serve.provider must be transformers or vllm')
            for key in ('endpoint', 'model', 'revision', 'prompt'):
                if not isinstance(values[key], str):
                    errors.append(f'llm_serve.{key} must be a string')
            endpoint = values['endpoint']
            if isinstance(endpoint, str) and endpoint:
                parsed = urlparse(endpoint)
                if parsed.scheme not in ('http', 'https') or not parsed.hostname:
                    errors.append('llm_serve.endpoint must be an HTTP(S) URL')
            if values['enabled'] and values['provider'] == 'vllm' and not endpoint:
                errors.append('llm_serve vllm requires a configured endpoint')
            if values['enabled'] and not values['model']:
                errors.append('llm_serve.model or llm_infer.model is required')
            for key in ('duration', 'timeout'):
                number = values[key]
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
                    errors.append(f'llm_serve.{key} must be a finite positive number')
    return errors


def smoke_optional_suites(cfg):
    for suite in OPTIONAL_DEFAULTS:
        if not (cfg.get(suite) or {}).get('enabled', False):
            continue
        values = suite_values(cfg, suite)
        values['warmup'] = min(values['warmup'], 1)
        if suite == 'vision_infer':
            values.update(batch_sizes=[1], sizes=[values['sizes'][0]], modes=[values['modes'][0]], iterations=2, min_duration_s=0)
        elif suite == 'kernel_bench':
            values.update(size=min(values['size'], 64), iterations=2, min_duration_s=0)
        else:
            values.update(duration=1, concurrency=[1], output_len=min(values['output_len'], 8))
        cfg[suite] = values


def optional_jobs(cfg):
    """Return all configured combinations once; the runner owns repetitions."""
    jobs = []
    for suite in OPTIONAL_DEFAULTS:
        values = suite_values(cfg, suite)
        if not values['enabled']:
            continue
        if suite == 'vision_infer':
            combinations = [dict(size=size, batch_size=batch, mode=mode)
                            for mode in values['modes'] for size, batch in
                            product(values['sizes'], [1] if mode == 'latency' else values['batch_sizes'])]
            keys = ('model', 'weights', 'dtype', 'iterations', 'warmup', 'min_duration_s')
        elif suite == 'kernel_bench':
            combinations = [dict(case=case) for case in values['cases']]
            keys = ('size', 'dtype', 'iterations', 'warmup', 'heads', 'head_dim', 'min_duration_s')
        else:
            combinations = [dict(concurrency=c) for c in values['concurrency']]
            keys = ('provider', 'model', 'revision', 'dtype', 'endpoint', 'prompt',
                    'output_len', 'duration', 'warmup', 'timeout')
        for combination in combinations:
            args = []
            for key, value in {**{k: values[k] for k in keys}, **combination}.items():
                if value != '':
                    args.extend(['--' + key.replace('_', '-'), str(value)])
            jobs.append({'suite': suite, 'script': suite + '.py', 'args': args})
    return jobs
