#!/usr/bin/env python3
"""Closed-loop serving measurements with real streamed completion events.

HTTP providers report content-chunk arrival gaps, never token-level latency.
The local Transformers provider serializes one model and observes generated IDs.
Loading and warmup are excluded; in-flight requests drain after the submission
deadline and their full elapsed time is included in the throughput denominator.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import threading
import time
import urllib.request

from benchmark_protocol import SEED, fixed_generation_kwargs, resolve_revision, text_hash


class ProviderUnavailable(RuntimeError):
    """A supported provider cannot run with the installed host capabilities."""


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'Invalid {name} in server usage')
    return value


class HttpProvider:
    timing_method = 'streaming_http_closed_loop_v1'
    ttft_method = 'first_nonempty_content_chunk_arrival'

    def __init__(self, endpoint, model, prompt, output_len, timeout=120, provider='openai'):
        self.endpoint = endpoint
        self.timeout = timeout
        self.fixed_length = provider == 'vllm'
        self.payload = dict(model=model, prompt=prompt, max_tokens=output_len,
                            temperature=0, seed=SEED, stream=True, n=1,
                            stream_options={'include_usage': True})
        if self.fixed_length:
            self.payload.update(min_tokens=output_len, ignore_eos=True)

    def request(self):
        started = time.perf_counter()
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(self.payload).encode(),
            headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
        content_times = []
        usage = None
        finished = False
        done = False
        data_lines = []

        def consume(data):
            nonlocal usage, finished, done
            if data == '[DONE]':
                done = True
                return
            parsed = json.loads(data)
            if not isinstance(parsed, dict) or 'error' in parsed:
                raise ValueError('Server returned a malformed or error stream event')
            if parsed.get('usage') is not None:
                usage = parsed['usage']
                if not isinstance(usage, dict):
                    raise ValueError('Malformed server usage')
            choices = parsed.get('choices', [])
            if not isinstance(choices, list):
                raise ValueError('Malformed completion choices')
            for choice in choices:
                if not isinstance(choice, dict) or choice.get('index', 0) != 0:
                    raise ValueError('Unexpected completion choice')
                text = choice.get('text', '')
                if not isinstance(text, str):
                    raise ValueError('Malformed completion content')
                if text:
                    content_times.append(time.perf_counter())
                if choice.get('finish_reason') is not None:
                    if choice['finish_reason'] not in ('stop', 'length'):
                        raise ValueError('Completion did not finish normally')
                    finished = True

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                raise ValueError('Provider must return an SSE streaming response')
            for raw in response:
                if time.perf_counter() - started > self.timeout:
                    raise TimeoutError('Serving request exceeded its deadline')
                line = raw.decode('utf-8').rstrip('\r\n')
                if not line:
                    if data_lines:
                        consume('\n'.join(data_lines))
                        data_lines.clear()
                        if done:
                            break
                elif line.startswith('data:'):
                    data_lines.append(line[5:].lstrip(' '))
        if not done or not finished or not content_times:
            raise ValueError('Incomplete stream: require content, finish_reason and [DONE]')
        tokens = None
        prompt_tokens = None
        if usage is not None:
            if 'completion_tokens' in usage:
                tokens = _count(usage['completion_tokens'], 'completion_tokens')
            if 'prompt_tokens' in usage:
                prompt_tokens = _count(usage['prompt_tokens'], 'prompt_tokens')
        if tokens == 0:
            raise ValueError('Nonempty completion has zero reported tokens')
        if self.fixed_length and tokens is not None and tokens != self.payload['max_tokens']:
            raise ValueError('vLLM did not complete the requested fixed token count')
        return dict(latency_s=time.perf_counter() - started,
                    ttft_s=content_times[0] - started,
                    chunk_gaps_s=[b - a for a, b in zip(content_times, content_times[1:])],
                    token_gaps_s=None, queue_s=None, generated_tokens=tokens,
                    prompt_tokens=prompt_tokens, token_count_source='server_usage' if tokens else None)


class TokenEventStreamer:
    """Observe single generated token IDs, excluding the initial prompt callback."""

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.prompt_seen = False
        self.token_times = []
        self.ended = False

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        if value.numel() != 1:
            raise ValueError('Expected one token per greedy generation event')
        self.token_times.append(self.clock())

    def end(self):
        self.ended = True


class TransformersProvider:
    timing_method = 'local_transformers_serial_queue_closed_loop_v1'
    ttft_method = 'first_generated_token_id_cpu_arrival_including_queue'

    def __init__(self, model, revision, dtype, prompt, output_len, gpu_index=0):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ProviderUnavailable(f'Local Transformers dependency unavailable: {exc}') from exc
        if not torch.cuda.is_available():
            raise ProviderUnavailable('Local Transformers serving requires a CUDA/ROCm GPU')
        if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
            raise ValueError('Selected GPU index is not visible')
        self.torch = torch
        self.device = f'cuda:{gpu_index}'
        self.gpu_index = gpu_index
        self.backend = 'amd' if torch.version.hip else 'nvidia'
        torch.cuda.set_device(gpu_index)
        torch.manual_seed(SEED)
        self.revision = resolve_revision(model, revision)
        if self.revision is None:
            raise ValueError('Local serving requires an immutable model revision')
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=self.revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=self.revision,
            torch_dtype={'float16': torch.float16, 'bfloat16': torch.bfloat16,
                         'float32': torch.float32, 'fp16': torch.float16,
                         'bf16': torch.bfloat16, 'fp32': torch.float32}[dtype]).to(self.device).eval()
        if getattr(self.model.config, 'is_encoder_decoder', False):
            raise ValueError('Serving workload requires a decoder-only causal LM')
        self.inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        self.prompt_tokens = int(self.inputs['input_ids'].shape[-1])
        self.output_len = output_len
        self.lock = threading.Lock()
        self.options = fixed_generation_kwargs(
            output_len, self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id)

    def request(self):
        requested = time.perf_counter()
        with self.lock:
            admitted = time.perf_counter()
            self.torch.cuda.set_device(self.gpu_index)
            self.torch.cuda.synchronize(self.device)
            streamer = TokenEventStreamer()
            with self.torch.inference_mode():
                output = self.model.generate(**self.inputs, **self.options, streamer=streamer)
            self.torch.cuda.synchronize(self.device)
            finished = time.perf_counter()
            count = int(output.shape[-1]) - self.prompt_tokens
            if not streamer.ended or count != self.output_len or len(streamer.token_times) != count:
                raise ValueError('Local generation did not produce the fixed token sequence')
        ticks = streamer.token_times
        return dict(latency_s=finished - requested, ttft_s=ticks[0] - requested,
                    token_gaps_s=[b - a for a, b in zip(ticks, ticks[1:])],
                    chunk_gaps_s=[], queue_s=admitted - requested,
                    generated_tokens=count, prompt_tokens=self.prompt_tokens,
                    token_count_source='generated_token_ids')


def measure(provider, duration, concurrency, warmup, sampler=None):
    """Release concurrent clients together; resubmit until a shared deadline."""
    if duration <= 0 or concurrency <= 0 or warmup < 0:
        raise ValueError('duration/concurrency must be positive and warmup nonnegative')
    for _ in range(warmup):
        provider.request()
    ready = threading.Barrier(concurrency + 1)
    release = threading.Event()
    abort = threading.Event()
    started = 0.0

    def client():
        successes, failures = [], []
        ready.wait()
        release.wait()
        while not abort.is_set() and time.perf_counter() < started + duration:
            try:
                successes.append(provider.request())
            except Exception as exc:
                failures.append(f'{type(exc).__name__}: {exc}')
                abort.set()
        return successes, failures

    samples, errors = [], []
    energy = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(client) for _ in range(concurrency)]
        ready.wait()
        try:
            if sampler:
                sampler.start()
            started = time.perf_counter()
            release.set()
            for future in futures:
                successes, failures = future.result()
                samples.extend(successes)
                errors.extend(failures)
            elapsed = time.perf_counter() - started
        finally:
            release.set()
            if sampler:
                energy = sampler.stop(started_s=started, ended_s=time.perf_counter())
    row = dict(status='failed' if errors or not samples else 'ok',
               requests=len(samples), errors=len(errors), error_details=errors[:10],
               time_s=elapsed, submission_duration_s=duration,
               drain_s=max(0., elapsed - duration), warmup_requests=warmup,
               reqs_per_s=len(samples) / elapsed if samples and not errors else None,
               generated_tokens=None, generated_tokens_per_s=None,
               requests_without_token_usage=sum(s['generated_tokens'] is None for s in samples),
               queue_s=None, inter_token_latency_ms=None,
               power_sampler_available=False, energy_j=None, mean_power_w=None,
               tokens_per_joule=None, energy_scope='unavailable_external_server')
    if samples and not errors and all(s['generated_tokens'] is not None for s in samples):
        row['generated_tokens'] = sum(s['generated_tokens'] for s in samples)
        row['generated_tokens_per_s'] = row['generated_tokens'] / elapsed
    prompt_counts = {s['prompt_tokens'] for s in samples}
    row['prompt_len'] = next(iter(prompt_counts)) if len(prompt_counts) == 1 else None
    row['observed_output_tokens_min'] = min(
        (s['generated_tokens'] for s in samples if s['generated_tokens'] is not None), default=None)
    row['observed_output_tokens_max'] = max(
        (s['generated_tokens'] for s in samples if s['generated_tokens'] is not None), default=None)
    for prefix, values in (
        ('latency_ms', [s['latency_s'] * 1000 for s in samples]),
        ('ttft_ms', [s['ttft_s'] * 1000 for s in samples]),
        ('stream_chunk_gap_ms', [v * 1000 for s in samples for v in s['chunk_gaps_s']]),
        ('inter_token_latency_ms', [v * 1000 for s in samples for v in (s['token_gaps_s'] or [])]),
        ('queue_ms', [s['queue_s'] * 1000 for s in samples if s['queue_s'] is not None]),
    ):
        row[prefix + '_mean'] = sum(values) / len(values) if values else None
        for suffix, quantile in (('p50', .5), ('p95', .95), ('p99', .99)):
            row[prefix + '_' + suffix] = percentile(values, quantile)
    if row['queue_ms_mean'] is not None:
        row['queue_s'] = row['queue_ms_mean'] / 1000
    row['inter_token_latency_ms'] = row['inter_token_latency_ms_mean']
    row.update(energy)
    if sampler:
        row['energy_scope'] = 'local_gpu_steady_state_with_drain'
    if row.get('energy_j') and row['generated_tokens'] is not None:
        row['tokens_per_joule'] = row['generated_tokens'] / row['energy_j']
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('transformers', 'vllm', 'openai'), default='transformers')
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', default=None)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16', 'float32', 'fp16', 'bf16', 'fp32'), default='float16')
    parser.add_argument('--endpoint', default='')
    parser.add_argument('--prompt', default='A deterministic benchmark prompt.')
    parser.add_argument('--output-len', type=int, default=32)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--duration', type=float, default=10)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--backend', choices=('amd', 'nvidia'), default=None)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--metrics-path', default='results/metrics.jsonl')
    args = parser.parse_args(argv)
    row = dict(benchmark_schema_version=2, suite='llm_serve', status='failed',
               provider=args.provider, model=args.model, requested_revision=args.revision,
               model_revision=None, tokenizer_revision=None, dtype=args.dtype,
               concurrency=args.concurrency, seed=SEED, output_len=args.output_len,
               prompt_sha256=text_hash(args.prompt), prompt_len=None,
               generation_protocol='greedy_fixed_tokens_v1' if not args.endpoint or args.provider == 'vllm'
               else 'greedy_max_tokens_streaming_v1',
               scheduling_protocol='closed_loop_shared_deadline_drain_v1',
               timing_method=HttpProvider.timing_method if args.endpoint else TransformersProvider.timing_method,
               power_sampler_available=False, energy_j=None, mean_power_w=None,
               tokens_per_joule=None, gpu_count=None if args.endpoint else 1,
               num_gpus=None if args.endpoint else 1, backend=args.backend, gpu_backend=None,
               identity_verified=False)
    try:
        if (args.output_len <= 0 or args.concurrency <= 0 or not math.isfinite(args.duration) or args.duration <= 0
                or args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0 or not args.prompt):
            raise ValueError('Invalid serving workload: positive lengths/duration/concurrency required')
        sampler = None
        if args.endpoint:
            provider = HttpProvider(args.endpoint, args.model, args.prompt, args.output_len,
                                    args.timeout, args.provider)
            row['revision_source'] = 'external_server_unverified'
        elif args.provider == 'transformers':
            provider = TransformersProvider(args.model, args.revision, args.dtype,
                                            args.prompt, args.output_len, args.gpu_index)
            from energy import EnergySampler
            sampler = EnergySampler(provider.backend, device_indices=[args.gpu_index])
            row.update(model_revision=provider.revision, tokenizer_revision=provider.revision,
                       backend=provider.backend, gpu_backend=provider.backend, identity_verified=True,
                       revision_source='resolved_immutable_hf_commit')
        else:
            raise ProviderUnavailable('Configure --endpoint with the provider /v1/completions URL')
        row['ttft_method'] = provider.ttft_method
        row.update(measure(provider, args.duration, args.concurrency, args.warmup, sampler))
    except ProviderUnavailable as exc:
        row.update(status='skipped', skip_reason='serving_provider_unavailable', detail=str(exc))
    except Exception as exc:
        row.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    path = Path(args.metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row, indent=2, allow_nan=False))
    return 1 if row['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
