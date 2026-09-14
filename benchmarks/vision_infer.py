#!/usr/bin/env python3
"""Pinned pretrained torchvision inference on deterministic synthetic RGB images."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse
from benchmark_protocol import SEED
from kernel_bench import summarize, measure_iterations


def weight_member(model, requested):
    defaults = {'resnet18': 'ResNet18_Weights.IMAGENET1K_V1',
                'resnet50': 'ResNet50_Weights.IMAGENET1K_V2',
                'vit_b_16': 'ViT_B_16_Weights.IMAGENET1K_V1'}
    selected = requested or defaults.get(model)
    if not selected or '.' not in selected or selected.split('.')[-1] == 'DEFAULT':
        raise ValueError('An explicit versioned torchvision weights enum is required')
    return selected


def run(args, row):
    selected = weight_member(args.model, args.weights)
    row['weights'] = selected
    try:
        import torch
    except ImportError:
        row.update(status='skipped', skip_reason='torch_unavailable')
        return
    if not torch.cuda.is_available():
        row.update(status='skipped', skip_reason='gpu_runtime_unavailable')
        return
    import torchvision
    from torchvision import models
    from energy import EnergySampler
    weights = models.get_weight(selected)
    if weights.__class__ is not models.get_model_weights(args.model):
        raise ValueError('Weights enum does not match selected model')
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    backend = 'amd' if torch.version.hip else 'nvidia'
    transform = weights.transforms(crop_size=args.size,
                                   resize_size=round(args.size * 256 / 224))
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    rgb = torch.randint(0, 256, (args.batch_size, 3, args.size + 32, args.size + 32),
                        generator=generator, dtype=torch.uint8)
    inputs_cpu = transform(rgb)
    preprocessing = repr(transform)
    row.update(model_revision=selected + ':' + weights.url, weights_url=weights.url,
               torchvision_version=torchvision.__version__,
               input_sha256=hashlib.sha256(inputs_cpu.numpy().tobytes()).hexdigest(),
               preprocessing=preprocessing,
               preprocessing_sha256=hashlib.sha256(preprocessing.encode()).hexdigest(),
               input_source='seeded_synthetic_uint8_rgb', gpu_count=1, gpu_backend=backend,
               effective_batch_size=args.batch_size)
    model = models.get_model(args.model, weights=weights).eval().to(device)
    cache = Path(torch.hub.get_dir()) / 'checkpoints' / Path(urlparse(weights.url).path).name
    row['weights_sha256'] = None
    if cache.is_file():
        with cache.open('rb') as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
            row['weights_sha256'] = digest.hexdigest()
    inputs = inputs_cpu.to(device)
    dtype = getattr(torch, args.dtype)
    precision = nullcontext() if args.dtype == 'float32' else torch.autocast('cuda', dtype=dtype)
    with torch.inference_mode(), precision:
        output = model(inputs)
        if not torch.isfinite(output).all().item():
            raise ValueError('Model output contains nonfinite values')
        del output
        for _ in range(args.warmup):
            model(inputs)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        sampler = EnergySampler(backend, device_indices=[0])
        sampler.start()
        try:
            samples = measure_iterations(lambda: model(inputs), torch.cuda.synchronize,
                                         args.iterations, args.min_duration_s)
        finally:
            row.update(sampler.stop())
    row.update(summarize(samples, args.batch_size))
    row['images_per_sec'] = row.pop('throughput')
    row['images_per_joule'] = args.batch_size * len(samples) / row['energy_j'] if row.get('energy_j') else None
    row.update(status='ok', measured_iterations=len(samples), images_total=args.batch_size * len(samples),
               precision_method='float32' if args.dtype == 'float32' else 'autocast',
               correctness_check='finite_pretrained_output',
               memory_peak_bytes=torch.cuda.max_memory_allocated(device))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--weights')
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='float32')
    parser.add_argument('--mode', choices=('throughput', 'latency'), default='throughput')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--min-duration-s', type=float, default=5.0)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--metrics-path', default='results/metrics.jsonl')
    args = parser.parse_args(argv)
    row = dict(benchmark_schema_version=2, suite='vision_infer', status='failed',
               model=args.model, batch_size=args.batch_size, input_size=args.size,
               iterations=args.iterations, warmup=args.warmup, dtype=args.dtype,
               mode=args.mode, min_duration_s=args.min_duration_s, seed=SEED, timing_method='synchronized_iteration_min_duration_v3')
    try:
        if min(args.size, args.batch_size, args.iterations) <= 0 or args.warmup < 0:
            raise ValueError('Sizes and iterations must be positive; warmup nonnegative')
        if args.mode == 'latency' and args.batch_size != 1:
            raise ValueError('Latency mode requires batch_size=1')
        if not math.isfinite(args.min_duration_s) or not 0 <= args.min_duration_s <= 3600:
            raise ValueError('min_duration_s must be finite and between 0 and 3600')
        run(args, row)
    except Exception as exc:
        row.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    path = Path(args.metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row, indent=2, allow_nan=False))
    return 1 if row['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
