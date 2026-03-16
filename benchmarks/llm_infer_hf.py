#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import signal
import shutil
import statistics
import subprocess
import threading
import time
import traceback
from queue import Empty
from typing import List

import yaml

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    IMPORT_ERROR = None
except Exception as exc:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    IMPORT_ERROR = exc


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


def detect_gpu_name() -> str:
    try:
        if torch is not None and torch.cuda.is_available():
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


def visible_gpu_count() -> int:
    visible_env = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible = [x.strip() for x in visible_env.split(",") if x.strip()]
    if visible:
        return len(visible)
    try:
        if torch is not None:
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0


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


def render_chat_prompt(tokenizer, user_prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        return user_prompt

    messages = [{"role": "user", "content": user_prompt}]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    model_name = (getattr(tokenizer, "name_or_path", "") or "").lower()
    if "qwen3" in model_name:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


class PowerSampler:
    def __init__(self, gpu_limit: int, interval_s: float = 0.5):
        self.interval = interval_s
        self.gpu_limit = max(1, int(gpu_limit))
        self.samples = []
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
                        total_w += (self.N.nvmlDeviceGetPowerUsage(h) or 0.0) / 1000.0
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
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    def available(self) -> bool:
        return self._ok and bool(getattr(self, "handles", []))


def classify_failure(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or "cuda error: out of memory" in text or "hip out of memory" in text:
        return "oom"
    if "no such file" in text or "404" in text or "repositorynotfounderror" in text:
        return "model_unavailable"
    if "trust_remote_code" in text:
        return "remote_code_requirement"
    if "cuda" in text or "hip" in text or "rocm" in text:
        return "gpu_runtime_error"
    return "unknown"


def dtype_for_config(name: str):
    key = (name or "auto").lower()
    if key == "auto":
        return None
    if key in {"float16", "half"}:
        return torch.float16
    if key == "bfloat16":
        return torch.bfloat16
    if key in {"float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def dependency_error_message() -> str | None:
    if IMPORT_ERROR is None:
        return None
    return f"transformers import failed: {type(IMPORT_ERROR).__name__}: {IMPORT_ERROR}"


def write_skip_row(cfg, batch_size: int, tensor_parallel: int, reason: str, detail: str):
    row = {
        "benchmark_schema_version": 2,
        "suite": "llm_infer",
        "status": "skipped",
        "skip_reason": reason,
        "backend": "transformers",
        "gpu_backend": detect_backend(),
        "model": cfg.get("model"),
        "dtype": cfg.get("dtype", "float16"),
        "batch_size": batch_size,
        "tensor_parallel": tensor_parallel,
        "prompt_len": int(cfg.get("prompt_len", 512)),
        "output_len": int(cfg.get("output_len", 128)),
        "gpu_name": detect_gpu_name(),
        "detail": detail,
    }
    write_metric(row)
    print(f"[SKIP] {detail}")


def load_model(model_name: str, dtype_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    dtype = dtype_for_config(dtype_name)
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        # Some instruct models ship sampling-oriented defaults that conflict with
        # our deterministic throughput path and trigger noisy warnings.
        model.generation_config.do_sample = False
        for attr in ("temperature", "top_p", "top_k", "min_p"):
            if hasattr(model.generation_config, attr):
                setattr(model.generation_config, attr, None)
    return tokenizer, model


def run_combo(
    model_name: str,
    dtype_name: str,
    batch_size: int,
    prompt: str,
    prompt_tokens: int,
    requested_prompt_len: int,
    out_len: int,
    warmup_s: int,
    duration_s: int,
    device_index: int = 0,
):
    if torch is None:
        raise RuntimeError("PyTorch is unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")

    torch.cuda.set_device(device_index)
    device = f"cuda:{device_index}"
    load_started = time.perf_counter()
    tokenizer, model = load_model(model_name, dtype_name, device)
    load_seconds = time.perf_counter() - load_started

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].repeat(batch_size, 1).to(device)
    attention_mask = encoded["attention_mask"].repeat(batch_size, 1).to(device)
    generate_kwargs = {
        "max_new_tokens": out_len,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    def generate_once():
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        torch.cuda.synchronize()
        return outputs

    warmup_deadline = time.time() + warmup_s
    while time.time() < warmup_deadline:
        _ = generate_once()

    ps = PowerSampler(gpu_limit=1, interval_s=0.5)
    ps.start()
    generated_tokens = 0
    requests = 0
    batch_latencies_ms = []
    started = time.time()
    ended = started + duration_s
    while time.time() < ended:
        batch_started = time.perf_counter()
        outputs = generate_once()
        batch_latencies_ms.append((time.perf_counter() - batch_started) * 1000.0)
        requests += batch_size
        generated_tokens += int((outputs.shape[1] - input_ids.shape[1]) * batch_size)
    elapsed = time.time() - started
    ps.stop()

    mean_w = ps.mean_watts()
    batch_latency_mean = round(statistics.fmean(batch_latencies_ms), 3) if batch_latencies_ms else 0.0
    batch_latency_p50 = round(percentile(batch_latencies_ms, 0.50), 3)
    batch_latency_p95 = round(percentile(batch_latencies_ms, 0.95), 3)
    row = {
        "benchmark_schema_version": 2,
        "suite": "llm_infer",
        "status": "ok",
        "backend": "transformers",
        "gpu_backend": detect_backend(),
        "model": model_name,
        "dtype": dtype_name,
        "tensor_parallel": 1,
        "batch_size": batch_size,
        "prompt_len": prompt_tokens,
        "requested_prompt_len": requested_prompt_len,
        "output_len": out_len,
        "warmup_s": warmup_s,
        "duration_s": duration_s,
        "requests": requests,
        "reqs_per_s": requests / elapsed if elapsed > 0 else 0.0,
        "generated_tokens": generated_tokens,
        "gen_tokens_per_s": generated_tokens / elapsed if elapsed > 0 else 0.0,
        "batch_latency_ms_mean": batch_latency_mean,
        "batch_latency_ms_p50": batch_latency_p50,
        "batch_latency_ms_p95": batch_latency_p95,
        "batch_latency_per_item_proxy_ms_mean": round(batch_latency_mean / batch_size, 3) if batch_size > 0 else 0.0,
        "batch_latency_per_item_proxy_ms_p50": round(batch_latency_p50 / batch_size, 3) if batch_size > 0 else 0.0,
        "batch_latency_per_item_proxy_ms_p95": round(batch_latency_p95 / batch_size, 3) if batch_size > 0 else 0.0,
        "latency_samples": len(batch_latencies_ms),
        "power_sampler_available": ps.available(),
        "mean_power_w": round(mean_w, 2),
        "gen_tokens_per_watt": (generated_tokens / elapsed / mean_w) if mean_w > 1e-6 else 0.0,
        "gpu_name": detect_gpu_name(),
        "gpu_index": device_index,
        "gpu_count": 1,
        "multi_gpu_mode": "single",
        "per_gpu_batch_size": batch_size,
        "load_seconds": round(load_seconds, 3),
        "time_s": elapsed,
    }

    del model
    del input_ids
    del attention_mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def run_worker(
    q: mp.Queue,
    model_name: str,
    dtype_name: str,
    batch_size: int,
    prompt: str,
    prompt_tokens: int,
    requested_prompt_len: int,
    out_len: int,
    warmup_s: int,
    duration_s: int,
    device_index: int,
):
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        row = run_combo(
            model_name=model_name,
            dtype_name=dtype_name,
            batch_size=batch_size,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            requested_prompt_len=requested_prompt_len,
            out_len=out_len,
            warmup_s=warmup_s,
            duration_s=duration_s,
            device_index=device_index,
        )
        q.put(row)
    except Exception as exc:
        q.put(
            {
                "benchmark_schema_version": 2,
                "suite": "llm_infer",
                "status": "failed",
                "backend": "transformers",
                "failure_kind": classify_failure(exc),
                "gpu_backend": detect_backend(),
                "model": model_name,
                "dtype": dtype_name,
                "tensor_parallel": 1,
                "batch_size": batch_size,
                "per_gpu_batch_size": batch_size,
                "prompt_len": prompt_tokens,
                "requested_prompt_len": requested_prompt_len,
                "output_len": out_len,
                "warmup_s": warmup_s,
                "duration_s": duration_s,
                "gpu_index": device_index,
                "gpu_count": 1,
                "multi_gpu_mode": "single",
                "gpu_name": detect_gpu_name(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "error_traceback_tail": traceback.format_exc(limit=3).strip().splitlines()[-1],
            }
        )


def aggregate_rows(rows, worker_count: int, batch_size: int, mode: str):
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]
    ref_row = ok_rows[0] if ok_rows else rows[0]
    timed_s = max(float(row.get("time_s", 0.0)) for row in ok_rows) if ok_rows else 0.0
    requests = sum(int(row.get("requests", 0)) for row in ok_rows)
    generated_tokens = sum(int(row.get("generated_tokens", 0)) for row in ok_rows)
    mean_power_w = sum(float(row.get("mean_power_w", 0.0)) for row in ok_rows)
    aggregate = {
        "benchmark_schema_version": 2,
        "suite": "llm_infer",
        "status": "ok" if len(ok_rows) == worker_count and not failed_rows else "failed",
        "backend": "transformers",
        "gpu_backend": detect_backend(),
        "model": ref_row.get("model"),
        "dtype": ref_row.get("dtype"),
        "tensor_parallel": 1,
        "batch_size": batch_size * worker_count if mode == "replicated" else batch_size,
        "per_gpu_batch_size": batch_size,
        "gpu_count": worker_count,
        "multi_gpu_mode": mode,
        "prompt_len": ref_row.get("prompt_len"),
        "requested_prompt_len": ref_row.get("requested_prompt_len"),
        "output_len": ref_row.get("output_len"),
        "warmup_s": ref_row.get("warmup_s"),
        "duration_s": ref_row.get("duration_s"),
        "requests": requests,
        "reqs_per_s": requests / timed_s if timed_s > 0 else 0.0,
        "generated_tokens": generated_tokens,
        "gen_tokens_per_s": generated_tokens / timed_s if timed_s > 0 else 0.0,
        "batch_latency_ms_mean": round(
            sum(float(row.get("batch_latency_ms_mean", 0.0)) for row in ok_rows) / len(ok_rows), 3
        ) if ok_rows else None,
        "batch_latency_ms_p50": round(
            sum(float(row.get("batch_latency_ms_p50", 0.0)) for row in ok_rows) / len(ok_rows), 3
        ) if ok_rows else None,
        "batch_latency_ms_p95": round(
            sum(float(row.get("batch_latency_ms_p95", 0.0)) for row in ok_rows) / len(ok_rows), 3
        ) if ok_rows else None,
        "batch_latency_per_item_proxy_ms_mean": round(
            sum(float(row.get("batch_latency_per_item_proxy_ms_mean", 0.0)) for row in ok_rows) / len(ok_rows), 3
        ) if ok_rows else None,
        "batch_latency_per_item_proxy_ms_p50": round(
            sum(float(row.get("batch_latency_per_item_proxy_ms_p50", 0.0)) for row in ok_rows) / len(ok_rows), 3
        ) if ok_rows else None,
        "batch_latency_per_item_proxy_ms_p95": round(
            sum(float(row.get("batch_latency_per_item_proxy_ms_p95", 0.0)) for row in ok_rows) / len(ok_rows), 3
        ) if ok_rows else None,
        "latency_samples": sum(int(row.get("latency_samples", 0)) for row in ok_rows),
        "power_sampler_available": all(bool(row.get("power_sampler_available")) for row in ok_rows) if ok_rows else False,
        "mean_power_w": round(mean_power_w, 2),
        "gen_tokens_per_watt": (generated_tokens / timed_s / mean_power_w) if timed_s > 0 and mean_power_w > 1e-6 else 0.0,
        "gpu_name": ref_row.get("gpu_name"),
        "load_seconds": round(max(float(row.get("load_seconds", 0.0)) for row in ok_rows), 3) if ok_rows else None,
        "time_s": timed_s if timed_s > 0 else None,
        "workers_ok": len(ok_rows),
        "workers_failed": len(failed_rows),
    }
    if failed_rows:
        aggregate["error"] = " | ".join(
            f"gpu{row.get('gpu_index', '?')}: {row.get('error', 'unknown error')}" for row in failed_rows
        )
    return aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--duration", type=int, default=30)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["llm_infer"]

    dep_error = dependency_error_message()
    if dep_error is not None:
        write_skip_row(cfg, 0, 1, "dependency_unavailable", dep_error)
        return

    model_name = cfg["model"]
    dtype_name = cfg.get("dtype", "float16")
    prompt_len = int(cfg.get("prompt_len", 512))
    out_len = int(cfg.get("output_len", 128))
    batch_sizes: List[int] = list(map(int, cfg.get("batch_sizes", [1, 4, 16, 64])))
    tp_sizes: List[int] = list(map(int, cfg.get("tensor_parallel_sizes", [1])))
    requested_multi_gpu_mode = (cfg.get("multi_gpu_mode", "single") or "single").lower()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    user_prompt, _ = make_prompt(tokenizer, prompt_len)
    prompt = render_chat_prompt(tokenizer, user_prompt)
    actual_prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

    for tp in tp_sizes:
        for batch_size in batch_sizes:
            if tp != 1:
                write_skip_row(
                    cfg,
                    batch_size,
                    tp,
                    "unsupported_backend_config",
                    "transformers backend only supports tensor_parallel=1",
                )
                continue
            try:
                worker_count = visible_gpu_count() if requested_multi_gpu_mode == "replicated" else 1
                worker_count = max(1, worker_count)
                actual_mode = "replicated" if requested_multi_gpu_mode == "replicated" and worker_count > 1 else "single"
                if actual_mode == "single":
                    row = run_combo(
                        model_name,
                        dtype_name,
                        batch_size,
                        prompt,
                        actual_prompt_tokens,
                        prompt_len,
                        out_len,
                        args.warmup,
                        args.duration,
                    )
                    row["multi_gpu_mode"] = actual_mode
                    row["gpu_count"] = 1
                else:
                    mp.set_start_method("spawn", force=True)
                    q: mp.Queue = mp.Queue(maxsize=worker_count)
                    workers = []
                    for device_index in range(worker_count):
                        proc = mp.Process(
                            target=run_worker,
                            kwargs=dict(
                                q=q,
                                model_name=model_name,
                                dtype_name=dtype_name,
                                batch_size=batch_size,
                                prompt=prompt,
                                prompt_tokens=actual_prompt_tokens,
                                requested_prompt_len=prompt_len,
                                out_len=out_len,
                                warmup_s=args.warmup,
                                duration_s=args.duration,
                                device_index=device_index,
                            ),
                            daemon=True,
                        )
                        workers.append(proc)

                    def _sigint_handler(_signum, _frame):
                        try:
                            for proc in workers:
                                if proc.is_alive():
                                    proc.terminate()
                        finally:
                            raise SystemExit(130)

                    signal.signal(signal.SIGINT, _sigint_handler)
                    for proc in workers:
                        proc.start()

                    worker_rows = []
                    try:
                        for _ in workers:
                            worker_rows.append(q.get(timeout=7200))
                    except Empty:
                        worker_rows = [
                            {
                                "status": "failed",
                                "gpu_index": "?",
                                "error": "worker timed out",
                            }
                        ]
                    finally:
                        for proc in workers:
                            proc.join(timeout=5)
                            if proc.is_alive():
                                proc.terminate()

                    row = aggregate_rows(worker_rows, worker_count, batch_size, actual_mode)
                write_metric(row)
                print(json.dumps(row, indent=2))
                if row.get("status") != "ok":
                    raise SystemExit(1)
            except Exception as exc:
                row = {
                    "benchmark_schema_version": 2,
                    "suite": "llm_infer",
                    "status": "failed",
                    "backend": "transformers",
                    "failure_kind": classify_failure(exc),
                    "gpu_backend": detect_backend(),
                    "model": model_name,
                    "dtype": dtype_name,
                    "tensor_parallel": tp,
                    "batch_size": batch_size,
                    "per_gpu_batch_size": batch_size,
                    "gpu_count": 1,
                    "multi_gpu_mode": requested_multi_gpu_mode,
                    "prompt_len": actual_prompt_tokens,
                    "requested_prompt_len": prompt_len,
                    "output_len": out_len,
                    "warmup_s": args.warmup,
                    "duration_s": args.duration,
                    "gpu_name": detect_gpu_name(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "error_traceback_tail": traceback.format_exc(limit=3).strip().splitlines()[-1],
                }
                write_metric(row)
                print(f"[ERROR] TP={tp} BS={batch_size}: {exc}")
                raise SystemExit(1)


if __name__ == "__main__":
    main()
