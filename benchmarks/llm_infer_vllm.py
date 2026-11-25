#!/usr/bin/env python3
import argparse
from contextlib import nullcontext
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import shutil
import statistics
import subprocess
import time
import threading
import traceback
from typing import List

import yaml
from energy import EnergySampler, physical_devices
from benchmark_protocol import SEED, resolve_revision, text_hash

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_IMPORT_ERROR = None
except Exception as exc:
    AutoTokenizer = None
    TRANSFORMERS_IMPORT_ERROR = exc

try:
    from vllm import LLM, SamplingParams
    VLLM_IMPORT_ERROR = None
except Exception:
    LLM = None
    SamplingParams = None
    VLLM_IMPORT_ERROR = traceback.format_exc(limit=3)


def detect_backend() -> str:
    if os.environ.get("HIP_VISIBLE_DEVICES"):
        return "amd"
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "nvidia"
    if shutil.which("rocm-smi") and not shutil.which("nvidia-smi"):
        return "amd"
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("rocm-smi"):
        return "amd"
    return "nvidia"


def make_prompt(tokenizer, target_tokens: int) -> tuple[str, int]:
    base = "The quick brown fox jumps over the lazy dog. "
    target_tokens = max(1, int(target_tokens))
    text = base
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += base

    lo, hi = 1, len(text)
    best = text
    best_count = len(tokenizer.encode(text, add_special_tokens=False))
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        count = len(tokenizer.encode(candidate, add_special_tokens=False))
        if count >= target_tokens:
            best = candidate
            best_count = count
            hi = mid - 1
        else:
            lo = mid + 1

    return best, best_count

def detect_gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    if detect_backend() == "amd":
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showproductname", "--json"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            payload = json.loads(out)
            card = payload.get("card") or payload
            if isinstance(card, dict):
                for value in card.values():
                    if isinstance(value, dict):
                        product = value.get("Card series") or value.get("Card model") or value.get("Product Name")
                        if product:
                            return str(product)
        except Exception:
            pass
    try:
        import pynvml as N
        N.nvmlInit()
        name = N.nvmlDeviceGetName(N.nvmlDeviceGetHandleByIndex(0)).decode()
        N.nvmlShutdown()
        return name
    except Exception:
        return "unknown"


def write_metric(row):
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def classify_failure(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or "cuda error: out of memory" in text or "hip out of memory" in text:
        return "oom"
    if "tensor_parallel_size" in text or "tensor parallel" in text:
        return "tensor_parallel_invalid"
    if "no such file" in text or "404" in text or "repositorynotfounderror" in text:
        return "model_unavailable"
    if "trust_remote_code" in text:
        return "remote_code_requirement"
    if "cuda" in text or "hip" in text or "rocm" in text:
        return "gpu_runtime_error"
    return "unknown"


def dependency_error_message() -> str | None:
    if TRANSFORMERS_IMPORT_ERROR is not None:
        exc = TRANSFORMERS_IMPORT_ERROR
        return f"transformers import failed: {type(exc).__name__}: {exc}"
    if LLM is None or SamplingParams is None:
        return f"vllm import failed: {VLLM_IMPORT_ERROR or 'unknown import error'}"
    return None


def probe_vllm_runtime(backend: str, release=None) -> str | None:
    if backend != "amd":
        return None
    try:
        importlib.import_module("vllm._rocm_C" if release == "0.34.0" else "vllm._C")
        return None
    except Exception as exc:
        return (
            "AMD vLLM runtime is unavailable in the current environment: "
            f"{type(exc).__name__}: {exc}"
        )


def write_skip_row(cfg, reason: str, detail: str, **case):
    row = {
        "benchmark_schema_version": 2,
        "suite": "llm_infer",
        "status": "skipped",
        "skip_reason": reason,
        "gpu_backend": detect_backend(),
        "model": cfg.get("model"),
        "dtype": cfg.get("dtype", "float16"),
        "gpu_name": detect_gpu_name(),
        "detail": detail,
        **case,
    }
    write_metric(row)
    print(f"[SKIP] {detail}")


class VramSampler:
    """Observe device-wide memory without querying GPUs in the timed generation loop."""

    def __init__(self, device_ids):
        self.device_ids = device_ids
        self.peaks = {}
        self.sample_count = 0
        self.done = threading.Event()
        self.worker = threading.Thread(target=self.poll, daemon=True)

    def sample(self):
        try:
            raw = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits',
                 '--id=' + ','.join(self.device_ids)], text=True, stderr=subprocess.DEVNULL, timeout=2)
            values = {}
            for line in raw.splitlines():
                uuid, used = (part.strip() for part in line.split(','))
                if uuid in values or int(used) < 0:
                    return
                values[uuid] = int(used)
        except (OSError, ValueError, subprocess.SubprocessError):
            return
        if (len(values) != len(self.device_ids)
                or (self.peaks and values.keys() != self.peaks.keys())
                or (all(i.startswith('GPU-') for i in self.device_ids)
                    and set(self.device_ids) != values.keys())):
            return
        for uuid, used in values.items():
            self.peaks[uuid] = max(self.peaks.get(uuid, 0), used)
        self.sample_count += 1

    def poll(self):
        while not self.done.is_set():
            self.sample()
            self.done.wait(0.5)

    def __enter__(self):
        self.worker.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.worker.join()


def installed_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def run_combo(model: str, dtype: str, tp: int, bs: int, prompt: str, prompt_tokens: int, requested_prompt_len: int,
              out_len: int, warmup_s: int, duration_s: int, gpu_mem_util: float | None, revision=None,
              max_model_len=8192, enable_prefix_caching=None):
    cache_options = {} if enable_prefix_caching is None else {'enable_prefix_caching': enable_prefix_caching}
    if gpu_mem_util is not None:
        cache_options['gpu_memory_utilization'] = gpu_mem_util
    llm = LLM(enforce_eager=True, disable_custom_all_reduce=True, max_model_len=max_model_len, model=model, dtype=dtype, # 'auto' | 'half' | 'float16' | 'bfloat16' | 'float' | 'float32'
        tensor_parallel_size=tp, trust_remote_code=True, disable_log_stats=True,
        revision=revision, tokenizer_revision=revision, seed=SEED, **cache_options)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=out_len,
        min_tokens=out_len,
        ignore_eos=True,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    prompts = [prompt] * bs

    # Warmup
    t_warm = time.time() + warmup_s
    while time.time() < t_warm:
        _ = llm.generate(prompts, sp)

    # Timed loop + power sampling
    ps = EnergySampler(detect_backend(), device_indices=list(range(tp)))
    ps.start()
    gen_tokens = 0
    reqs = 0
    batch_latencies_ms = []
    t0 = time.perf_counter()
    t_end = t0 + duration_s
    try:
        while time.perf_counter() < t_end:
            t_batch = time.perf_counter()
            outputs = llm.generate(prompts, sp)
            batch_latencies_ms.append((time.perf_counter() - t_batch) * 1000.0)
            reqs += len(outputs)
            for out in outputs:
                if len(out.outputs[0].token_ids) != out_len:
                    raise ValueError('vLLM returned an unexpected output token count')
                gen_tokens += len(out.outputs[0].token_ids)
        finished = time.perf_counter()
        elapsed = finished - t0
    finally:
        energy = ps.stop(started_s=t0, ended_s=time.perf_counter())

    gpu_name = detect_gpu_name()
    batch_latency_mean = round(statistics.fmean(batch_latencies_ms), 3) if batch_latencies_ms else 0.0
    batch_latency_p50 = round(percentile(batch_latencies_ms, 0.50), 3)
    batch_latency_p95 = round(percentile(batch_latencies_ms, 0.95), 3)
    row = {
        "benchmark_schema_version": 2,
        "suite": "llm_infer",
        "backend": "vllm",
        "model_revision": revision,
        "tokenizer_revision": revision,
        "prompt_sha256": text_hash(prompt),
        "seed": SEED,
        "generation_mode": "greedy_fixed_length",
        "timing_method": "vllm_blocking_generate_v1",
        "multi_gpu_mode": "tensor_parallel" if tp > 1 else "single",
        "gpu_count": tp,
        "global_batch_size": bs,
        "status": "ok",
        "gpu_backend": detect_backend(),
        "model": model,
        "dtype": dtype,
        "tensor_parallel": tp,
        "batch_size": bs,
        "prompt_len": prompt_tokens,
        "requested_prompt_len": requested_prompt_len,
        "output_len": out_len,
        "warmup_s": warmup_s,
        "duration_s": duration_s,
        "requests": reqs,
        "reqs_per_s": reqs / elapsed if elapsed > 0 else 0.0,
        "generated_tokens": gen_tokens,
        "gen_tokens_per_s": gen_tokens / elapsed if elapsed > 0 else 0.0,
        "batch_latency_ms_mean": batch_latency_mean,
        "batch_latency_ms_p50": batch_latency_p50,
        "batch_latency_ms_p95": batch_latency_p95,
        "batch_latency_per_item_proxy_ms_mean": round(batch_latency_mean / bs, 3) if bs > 0 else 0.0,
        "batch_latency_per_item_proxy_ms_p50": round(batch_latency_p50 / bs, 3) if bs > 0 else 0.0,
        "batch_latency_per_item_proxy_ms_p95": round(batch_latency_p95 / bs, 3) if bs > 0 else 0.0,
        "latency_samples": len(batch_latencies_ms),
        **energy,
        "gen_tokens_per_watt": gen_tokens / energy['energy_j'] if energy.get('energy_j') else None,
        "gpu_name": gpu_name,
        "time_s": elapsed,
    }

    # Clean up
    del llm
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--gpu-mem", type=float, help="Override GPU memory utilization for vLLM")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)["llm_infer"]
    backend = detect_backend()
    dep_error = dependency_error_message()
    if dep_error is not None:
        write_skip_row(cfg, "dependency_unavailable", dep_error)
        return
    build_version = installed_version('vllm')
    actual_version = build_version.split('+', 1)[0] if build_version else None
    runtime_error = probe_vllm_runtime(backend, actual_version)
    if runtime_error is not None:
        write_skip_row(cfg, "unsupported_runtime", runtime_error)
        return

    model = cfg["model"]
    dtype = cfg.get("dtype", "float16")
    prompt_len = int(cfg.get("prompt_len", 512))
    out_len = int(cfg.get("output_len", 128))
    batch_sizes: List[int] = list(map(int, cfg.get("batch_sizes", [1,4,16,64])))
    tp_sizes: List[int] = list(map(int, cfg.get("tensor_parallel_sizes", [1])))
    gpu_mem = args.gpu_mem
    if gpu_mem is None and actual_version != '0.34.0':
        gpu_mem = 0.95
    expected_version = cfg.get('vllm_version')
    if expected_version and expected_version != actual_version:
        write_skip_row(cfg, 'vllm_version_mismatch',
                       f'Profile requires vLLM {expected_version}; found {actual_version}')
        return

    revision = resolve_revision(model, cfg.get('revision'))
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, revision=revision)
    prompt, actual_prompt_tokens = make_prompt(tokenizer, prompt_len)

    software = {f'inference_{package}_version': installed_version(package)
                for package in ('torch', 'transformers')}
    failures = 0
    visible = os.environ.get('HIP_VISIBLE_DEVICES' if backend == 'amd' else 'CUDA_VISIBLE_DEVICES')
    for tp in tp_sizes:
        if visible and tp > len(visible.split(',')):
            write_skip_row(cfg, 'insufficient_gpus', f'Tensor parallel size {tp} exceeds visible GPU count',
                           tensor_parallel=tp)
            continue
        for bs in batch_sizes:
            try:
                max_model_len = cfg.get('max_model_len', 8192)
                prefix_caching = cfg.get('enable_prefix_caching')
                sampler = None
                if cfg.get('sample_vram') and backend == 'nvidia':
                    sampler = VramSampler(physical_devices(backend, list(range(tp))))
                with sampler if sampler else nullcontext():
                    row = run_combo(model, dtype, tp, bs, prompt, actual_prompt_tokens, prompt_len,
                                    out_len, args.warmup, args.duration, gpu_mem,
                                    revision=revision, max_model_len=max_model_len,
                                    enable_prefix_caching=prefix_caching)
                row.update(vllm_version=actual_version, vllm_build_version=build_version, max_model_len=max_model_len,
                           gpu_memory_utilization=gpu_mem, enable_prefix_caching=prefix_caching, **software)
                if sampler:
                    row.update(vram_peak_mib_by_gpu=sampler.peaks, vram_sample_count=sampler.sample_count,
                               vram_max_gpu_peak_mib=max(sampler.peaks.values(), default=None))
                write_metric(row)
                print(json.dumps(row, indent=2))
            except Exception as e:
                failures += 1
                row = {
                    "benchmark_schema_version": 2,
                    "suite": "llm_infer",
                    "status": "failed",
                    "failure_kind": classify_failure(e),
                    "gpu_backend": detect_backend(),
                    "model": model,
                    "dtype": dtype,
                    "tensor_parallel": tp,
                    "batch_size": bs,
                    "prompt_len": actual_prompt_tokens,
                    "requested_prompt_len": prompt_len,
                    "output_len": out_len,
                    "warmup_s": args.warmup,
                    "duration_s": args.duration,
                    "gpu_name": detect_gpu_name(),
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "error_traceback_tail": traceback.format_exc(limit=3).strip().splitlines()[-1],
                }
                write_metric(row)
                print(f"[ERROR] TP={tp} BS={bs}: {e}")
    if failures:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
