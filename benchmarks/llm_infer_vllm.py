#!/usr/bin/env python3
import argparse
import json
import os
import statistics
import subprocess
import threading
import time
import traceback
from typing import List

import yaml
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None


def detect_backend() -> str:
    return "amd" if os.environ.get("HIP_VISIBLE_DEVICES") else "nvidia"


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

# ---------- NVML power sampler ----------
class PowerSampler:
    def __init__(self, gpu_limit: int, interval_s: float = 0.5):
        self.interval = interval_s
        self.gpu_limit = max(1, int(gpu_limit))
        self.samples = []  # watts (total across visible GPUs)
        self._stop = threading.Event()
        self._thr = None
        self._ok = False
        self.backend = detect_backend()
        try:
            if self.backend == "nvidia":
                import pynvml as N
                self.N = N
                N.nvmlInit()
                self.handles = self._resolve_handles()
                self._ok = True
        except Exception:
            self._ok = False

    def _resolve_handles(self):
        visible_env = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible = [x.strip() for x in visible_env.split(",") if x.strip()]
        selected = visible[: self.gpu_limit] if visible else []
        handles = []

        if selected:
            for item in selected:
                try:
                    if item.isdigit():
                        handles.append(self.N.nvmlDeviceGetHandleByIndex(int(item)))
                    else:
                        handles.append(self.N.nvmlDeviceGetHandleByUUID(item.encode()))
                except Exception:
                    continue
            if handles:
                return handles

        count = self.N.nvmlDeviceGetCount()
        for i in range(min(self.gpu_limit, count)):
            handles.append(self.N.nvmlDeviceGetHandleByIndex(i))
        return handles

    def _tick(self):
        while not self._stop.is_set():
            total_w = 0.0
            if self._ok and self.backend == "nvidia":
                for h in self.handles:
                    try:
                        # powerUsage is in milliwatts
                        mw = self.N.nvmlDeviceGetPowerUsage(h)
                        total_w += (mw or 0.0) / 1000.0
                    except Exception:
                        pass
            self.samples.append(total_w if total_w > 0 else 0.0)
            time.sleep(self.interval)

    def start(self):
        self._thr = threading.Thread(target=self._tick, daemon=True)
        self._thr.start()

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join()
        if self._ok:
            try:
                self.N.nvmlShutdown()
            except Exception:
                pass

    def mean_watts(self) -> float:
        return sum(self.samples)/len(self.samples) if self.samples else 0.0

    def available(self) -> bool:
        return self._ok and bool(getattr(self, "handles", []))

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


def run_combo(model: str, dtype: str, tp: int, bs: int, prompt: str, prompt_tokens: int, requested_prompt_len: int,
              out_len: int, warmup_s: int, duration_s: int, gpu_mem_util: float):
    llm = LLM(enforce_eager=True, disable_custom_all_reduce=True, max_model_len=8192, model=model, dtype=dtype, # 'auto' | 'half' | 'float16' | 'bfloat16' | 'float' | 'float32'
        tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem_util, trust_remote_code=True, disable_log_stats=True)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=out_len,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    prompts = [prompt] * bs

    # Warmup
    t_warm = time.time() + warmup_s
    while time.time() < t_warm:
        _ = llm.generate(prompts, sp)

    # Timed loop + power sampling
    ps = PowerSampler(gpu_limit=tp, interval_s=0.5); ps.start()
    gen_tokens = 0
    reqs = 0
    batch_latencies_ms = []
    t0 = time.time()
    t_end = t0 + duration_s
    while time.time() < t_end:
        t_batch = time.perf_counter()
        outputs = llm.generate(prompts, sp)
        batch_latencies_ms.append((time.perf_counter() - t_batch) * 1000.0)
        reqs += len(outputs)
        for out in outputs:
            gen_tokens += len(out.outputs[0].token_ids)
    elapsed = time.time() - t0
    ps.stop()

    gpu_name = detect_gpu_name()
    mean_w = ps.mean_watts()
    batch_latency_mean = round(statistics.fmean(batch_latencies_ms), 3) if batch_latencies_ms else 0.0
    batch_latency_p50 = round(percentile(batch_latencies_ms, 0.50), 3)
    batch_latency_p95 = round(percentile(batch_latencies_ms, 0.95), 3)
    row = {
        "benchmark_schema_version": 2,
        "suite": "llm_infer",
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
        "power_sampler_available": ps.available(),
        "mean_power_w": round(mean_w, 2),
        "gen_tokens_per_watt": (gen_tokens / elapsed / mean_w) if mean_w > 1e-6 else 0.0,
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
    if LLM is None or SamplingParams is None:
        row = {
            "benchmark_schema_version": 2,
            "suite": "llm_infer",
            "status": "failed",
            "failure_kind": "dependency_unavailable",
            "gpu_backend": detect_backend(),
            "model": cfg.get("model"),
            "dtype": cfg.get("dtype", "float16"),
            "error_type": "ImportError",
            "error": "vllm is not importable in the current environment",
            "gpu_name": detect_gpu_name(),
        }
        write_metric(row)
        print("[ERROR] vllm is not importable in the current environment")
        raise SystemExit(1)

    model = cfg["model"]
    dtype = cfg.get("dtype", "float16")
    prompt_len = int(cfg.get("prompt_len", 512))
    out_len = int(cfg.get("output_len", 128))
    batch_sizes: List[int] = list(map(int, cfg.get("batch_sizes", [1,4,16,64])))
    tp_sizes: List[int] = list(map(int, cfg.get("tensor_parallel_sizes", [1])))
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompt, actual_prompt_tokens = make_prompt(tokenizer, prompt_len)

    for tp in tp_sizes:
        for bs in batch_sizes:
            try:
                row = run_combo(model, dtype, tp, bs, prompt, actual_prompt_tokens, prompt_len, out_len,
                                args.warmup, args.duration, args.gpu_mem)
                write_metric(row)
                print(json.dumps(row, indent=2))
            except Exception as e:
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

if __name__ == "__main__":
    main()
