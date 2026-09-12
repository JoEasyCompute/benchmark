#!/usr/bin/env python3
"""Synchronized GPU kernel diagnostics with bounded numerical reference checks."""
import argparse
import json
import math
from pathlib import Path
from benchmark_protocol import SEED, timed_call


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(samples, work):
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError('Measured durations must be finite and positive')
    elapsed = sum(samples)
    return dict(time_s=elapsed, throughput=work * len(samples) / elapsed,
                latency_ms_mean=elapsed / len(samples) * 1000,
                latency_ms_p50=percentile(samples, .5) * 1000,
                latency_ms_p95=percentile(samples, .95) * 1000)


def operation_work(case, size, heads=8, head_dim=64, itemsize=4):
    if case == 'gemm':
        return 2 * size ** 3
    if case == 'attention':
        return 4 * heads * size ** 2 * head_dim
    if case == 'memory':
        return 2 * size ** 2 * itemsize
    raise ValueError(f'Unknown kernel {case}')


def check_output(torch, case, inputs, output, dtype):
    if not torch.isfinite(output).all().item():
        raise ValueError('Kernel output contains nonfinite values')
    if case == 'gemm':
        a, b = inputs
        expected = a[:8].double().cpu() @ b[:, :8].double().cpu()
        actual = output[:8, :8].double().cpu()
    elif case == 'attention':
        q, k, v = inputs
        scores = q[:, :, :4].double().cpu() @ k.double().cpu().transpose(-2, -1)
        expected = (scores / math.sqrt(q.shape[-1])).softmax(-1) @ v.double().cpu()
        actual = output[:, :, :4].double().cpu()
    else:
        expected, actual = inputs[0].cpu(), output.cpu()
    tol = {'float32': 2e-4, 'float16': 5e-3, 'bfloat16': 5e-2}[dtype]
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)


def run(args, row):
    try:
        import torch
    except ImportError:
        row.update(status='skipped', skip_reason='torch_unavailable')
        return
    if not torch.cuda.is_available():
        row.update(status='skipped', skip_reason='gpu_runtime_unavailable')
        return
    from energy import EnergySampler
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    device, dtype, n = torch.device('cuda:0'), getattr(torch, args.dtype), args.size
    backend = 'amd' if torch.version.hip else 'nvidia'
    row.update(gpu_count=1, gpu_backend=backend)
    if args.case == 'gemm':
        inputs = [torch.randn(n, n, device=device, dtype=dtype) for _ in range(2)]
        operation = lambda: inputs[0] @ inputs[1]
    elif args.case == 'attention':
        inputs = [torch.randn(1, args.heads, n, args.head_dim, device=device,
                              dtype=dtype) for _ in range(3)]
        operation = lambda: torch.nn.functional.scaled_dot_product_attention(*inputs)
    else:
        inputs = [torch.randn(n * n, device=device, dtype=dtype)]
        destination = torch.empty_like(inputs[0])
        operation = lambda: destination.copy_(inputs[0])
    work = operation_work(args.case, n, args.heads, args.head_dim, inputs[0].element_size())
    with torch.inference_mode():
        output = operation()
        check_output(torch, args.case, inputs, output, args.dtype)
        del output
        for _ in range(args.warmup):
            operation()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        sampler = EnergySampler(backend, device_indices=[0])
        sampler.start()
        try:
            samples = []
            for _ in range(args.iterations):
                output, elapsed = timed_call(operation, torch.cuda.synchronize)
                samples.append(elapsed)
                del output
        finally:
            row.update(sampler.stop())
    row.update(summarize(samples, work))
    rate = row['throughput']
    row.update(status='ok', work_per_iteration=work,
               work_unit='bytes' if args.case == 'memory' else 'flops',
               work_convention='read_plus_write' if args.case == 'memory' else
               'matmul_multiply_add_2flops_softmax_excluded',
               tflops=rate / 1e12 if args.case != 'memory' else None,
               bandwidth_gbps=rate / 1e9 if args.case == 'memory' else None,
               memory_peak_bytes=torch.cuda.max_memory_allocated(device),
               correctness_check='finite_and_sampled_cpu_reference')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('gemm', 'attention', 'memory'), default='gemm')
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--head-dim', type=int, default=64)
    parser.add_argument('--dtype', choices=('float32', 'float16', 'bfloat16'), default='float32')
    parser.add_argument('--metrics-path', default='results/metrics.jsonl')
    args = parser.parse_args(argv)
    row = dict(benchmark_schema_version=2, suite='kernel_bench', status='failed',
               seed=SEED, timing_method='synchronized_iteration_v2',
               **{k: v for k, v in vars(args).items() if k != 'metrics_path'})
    try:
        if min(args.size, args.iterations, args.heads, args.head_dim) <= 0 or args.warmup < 0:
            raise ValueError('Sizes and iterations must be positive; warmup nonnegative')
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
