#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import shutil
import statistics
import subprocess
import time
import traceback
from typing import List

import yaml
from energy import EnergySampler
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


def probe_vllm_runtime(backend: str) -> str | None:
    if backend != "amd":
        return None
    try:
        importlib.import_module("vllm._C")
        return None
    except Exception as exc:
        return (
            "AMD vLLM runtime is unavailable in the current environment: "
            f"{type(exc).__name__}: {exc}"
        )


def write_skip_row(cfg, reason: str, detail: str):
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
    }
    write_metric(row)
    print(f"[SKIP] {detail}")


def run_combo(model: str, dtype: str, tp: int, bs: int, prompt: str, prompt_tokens: int, requested_prompt_len: int,
              out_len: int, warmup_s: int, duration_s: int, gpu_mem_util: float, revision=None):
    llm = LLM(enforce_eager=True, disable_custom_all_reduce=True, max_model_len=8192, model=model, dtype=dtype, # 'auto' | 'half' | 'float16' | 'bfloat16' | 'float' | 'float32'
        tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem_util, trust_remote_code=True, disable_log_stats=True,
        revision=revision, tokenizer_revision=revision, seed=SEED)
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
        "per_gpu_batch_size": bs,
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
    ap.add_argument("--gpu-mem", type=float, default=0.95, help="gpu_memory_utilization for vLLM")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)["llm_infer"]
    backend = detect_backend()
    dep_error = dependency_error_message()
    if dep_error is not None:
        write_skip_row(cfg, "dependency_unavailable", dep_error)
        return
    runtime_error = probe_vllm_runtime(backend)
    if runtime_error is not None:
        write_skip_row(cfg, "unsupported_runtime", runtime_error)
        return

    model = cfg["model"]
    dtype = cfg.get("dtype", "float16")
    prompt_len = int(cfg.get("prompt_len", 512))
    out_len = int(cfg.get("output_len", 128))
    batch_sizes: List[int] = list(map(int, cfg.get("batch_sizes", [1,4,16,64])))
    tp_sizes: List[int] = list(map(int, cfg.get("tensor_parallel_sizes", [1])))
    revision = resolve_revision(model, cfg.get('revision'))
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, revision=revision)
    prompt, actual_prompt_tokens = make_prompt(tokenizer, prompt_len)

    failures = 0
    for tp in tp_sizes:
        for bs in batch_sizes:
            try:
                row = run_combo(model, dtype, tp, bs, prompt, actual_prompt_tokens, prompt_len, out_len,
                                args.warmup, args.duration, args.gpu_mem, revision=revision)
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
