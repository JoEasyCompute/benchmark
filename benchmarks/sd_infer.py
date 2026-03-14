#!/usr/bin/env python3
"""
Stable Diffusion inference benchmark (single-GPU).

- Manually assembles the pipeline to avoid transformers' offload_state_dict path.
- Accepts --bf16 / --xformers and common scheduler flags (euler, euler_a).
- Prints images/sec and per-iteration timing.

Example:
  python benchmarks/sd_infer.py \
    --model runwayml/stable-diffusion-v1-5 \
    --prompt "a cyberpunk cat wearing sunglasses" \
    --width 512 --height 512 --steps 25 --it 3 --bf16 --xformers
"""

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from queue import Empty
from typing import Optional

import torch
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor


def write_metric(row, metrics_path: str):
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    with open(metrics_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _select_dtype(prefer_bf16: bool, device_index: int) -> torch.dtype:
    """Use bf16 on >= Ada if requested; else fp16."""
    if prefer_bf16 and torch.cuda.is_available():
        major = torch.cuda.get_device_capability(device_index)[0]
        if major >= 8:  # Ampere+ supports bf16 well; 4090 is Ada (SM89)
            return torch.bfloat16
    return torch.float16


def visible_gpu_count() -> int:
    visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if visible:
        return len(visible)
    return torch.cuda.device_count()


def _maybe_set_scheduler(pipe, scheduler_flag: str):
    """Optionally override scheduler via CLI flag."""
    if not scheduler_flag:
        return pipe
    s = scheduler_flag.lower()
    if "euler_a" in s or "euler-ancestral" in s:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif "euler" in s:
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe


def run_worker(
    q: mp.Queue,
    model: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    batch_size: int,
    iterations: int,
    seed: Optional[int],
    scheduler: Optional[str],
    prefer_bf16: bool,
    enable_xformers: bool,
    device_index: int,
):
    try:
        # child: ignore SIGINT; parent handles it
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        torch.cuda.set_device(device_index)
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        device = f"cuda:{device_index}"

        dtype = _select_dtype(prefer_bf16, device_index)
        t0 = time.perf_counter()

        # --- Manual assembly to avoid offload_state_dict being forwarded ---
        text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer")
        vae = AutoencoderKL.from_pretrained(model, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet")
        feature_extractor = CLIPImageProcessor.from_pretrained(model, subfolder="feature_extractor")

        # Load a scheduler from the repo (or derive sensible defaults)
        try:
            sd_scheduler = EulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler")
        except Exception:
            try:
                sd_scheduler = PNDMScheduler.from_pretrained(model, subfolder="scheduler")
            except Exception:
                # Fallback defaults commonly used for SD1.5
                sd_scheduler = EulerDiscreteScheduler.from_config(
                    {
                        "beta_start": 0.00085,
                        "beta_end": 0.012,
                        "beta_schedule": "scaled_linear",
                        "num_train_timesteps": 1000,
                        "timestep_spacing": "leading",
                        "prediction_type": "epsilon",
                        "use_karras_sigmas": False,
                    }
                )

        pipe = StableDiffusionPipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            unet=unet,
            scheduler=sd_scheduler,
            feature_extractor=feature_extractor,
            safety_checker=None,
        )

        pipe = pipe.to(device, dtype=dtype)

        if enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"[WARN] xFormers not enabled: {e}", file=sys.stderr)

        pipe = _maybe_set_scheduler(pipe, scheduler)

        _ = pipe(
            prompt="warmup image",
            negative_prompt="",
            height=min(height, 512),
            width=min(width, 512),
            num_inference_steps=10,
            guidance_scale=guidance,
            num_images_per_prompt=1,
            generator=torch.Generator(device=device).manual_seed(1234),
        )

        load_s = time.perf_counter() - t0

        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed + device_index)

        times = []
        total_images = iterations * batch_size

        for _ in range(iterations):
            t_iter = time.perf_counter()
            _ = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                num_images_per_prompt=batch_size,
                generator=g,
            )
            times.append(time.perf_counter() - t_iter)

        imgs_per_sec = total_images / sum(times) if times else 0.0

        try:
            pipe._execution_device = None
            del pipe
            torch.cuda.synchronize(device)
        except Exception:
            pass

        q.put(
            {
                "suite": "sd_infer",
                "status": "ok",
                "model": model,
                "dtype": str(dtype),
                "load_seconds": round(load_s, 3),
                "iters": iterations,
                "batch_size": batch_size,
                "steps": steps,
                "hw": f"{width}x{height}",
                "mean_s_per_iter": round(sum(times) / len(times), 4) if times else None,
                "images_total": total_images,
                "images_per_sec": round(imgs_per_sec, 3),
                "time_s": round(sum(times), 4),
                "gpu_index": device_index,
            }
        )
    except Exception as e:
        q.put(
            {
                "suite": "sd_infer",
                "status": "failed",
                "model": model,
                "batch_size": batch_size,
                "steps": steps,
                "hw": f"{width}x{height}",
                "gpu_index": device_index,
                "error": str(e),
            }
        )


def aggregate_results(results, args, worker_count: int, mode: str, wall_time_s: float):
    ok_results = [row for row in results if row.get("status") == "ok"]
    failed_results = [row for row in results if row.get("status") != "ok"]

    row = {
        "suite": "sd_infer",
        "status": "ok" if len(ok_results) == worker_count and not failed_results else "failed",
        "model": args.model,
        "iters": args.iterations,
        "per_gpu_batch_size": args.batch_size,
        "batch_size": args.batch_size * worker_count if mode == "replicated" else args.batch_size,
        "steps": args.steps,
        "hw": f"{args.width}x{args.height}",
        "gpu_count": worker_count,
        "multi_gpu_mode": mode,
        "repeat_index": args.repeat_index,
        "repeat_count": args.repeat_count,
        "time_s": None,
        "end_to_end_time_s": round(wall_time_s, 4),
        "workers_ok": len(ok_results),
        "workers_failed": len(failed_results),
    }

    if ok_results:
        total_images = sum(int(r.get("images_total", 0)) for r in ok_results)
        timed_s = max(float(r.get("time_s", 0.0)) for r in ok_results)
        row.update(
            {
                "dtype": ok_results[0].get("dtype"),
                "load_seconds": round(max(float(r.get("load_seconds", 0.0)) for r in ok_results), 3),
                "mean_s_per_iter": round(
                    sum(float(r.get("mean_s_per_iter") or 0.0) for r in ok_results) / len(ok_results), 4
                ),
                "images_total": total_images,
                "time_s": round(timed_s, 4),
                "images_per_sec": round(total_images / timed_s, 3) if timed_s > 0 else 0.0,
            }
        )

    if failed_results:
        row["error"] = " | ".join(
            f"gpu{r.get('gpu_index', '?')}: {r.get('error', 'unknown error')}" for r in failed_results
        )

    return row


def worker_result_rows(results, args, mode):
    rows = []
    for row in results:
        worker_row = dict(row)
        worker_row["suite"] = "sd_infer_worker"
        worker_row["repeat_index"] = args.repeat_index
        worker_row["repeat_count"] = args.repeat_count
        worker_row["multi_gpu_mode"] = mode
        worker_row["per_gpu_batch_size"] = args.batch_size
        worker_row["gpu_count"] = 1
        rows.append(worker_row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--prompt", default="A photo of a cute corgi wearing sunglasses, cinematic, high detail")
    ap.add_argument("--neg", "--negative-prompt", dest="negative_prompt", default="")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", "--guidance", dest="guidance", type=float, default=7.5)
    ap.add_argument("--bs", "--batch-size", dest="batch_size", type=int, default=1)
    ap.add_argument("--it", "--iterations", dest="iterations", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scheduler", default="", help="euler, euler_a, ...")
    ap.add_argument("--bf16", action="store_true", help="prefer bfloat16 if available")
    ap.add_argument("--xformers", action="store_true", help="enable xFormers attention")
    ap.add_argument("--metrics-path", default="results/metrics.jsonl")
    ap.add_argument("--repeat-index", type=int, default=1)
    ap.add_argument("--repeat-count", type=int, default=1)
    ap.add_argument("--multi-gpu-mode", default="single", help="single or replicated")
    ap.add_argument("--emit-worker-rows", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        write_metric(
            {
                "suite": "sd_infer",
                "status": "failed",
                "model": args.model,
                "iters": args.iterations,
                "per_gpu_batch_size": args.batch_size,
                "batch_size": args.batch_size,
                "steps": args.steps,
                "hw": f"{args.width}x{args.height}",
                "gpu_count": 0,
                "multi_gpu_mode": (args.multi_gpu_mode or "single").lower(),
                "repeat_index": args.repeat_index,
                "repeat_count": args.repeat_count,
                "time_s": None,
                "end_to_end_time_s": 0.0,
                "workers_ok": 0,
                "workers_failed": 1,
                "error": "CUDA not available",
            },
            args.metrics_path,
        )
        sys.exit(1)

    mp.set_start_method("spawn", force=True)
    requested_mode = (args.multi_gpu_mode or "single").lower()
    worker_count = visible_gpu_count() if requested_mode == "replicated" else 1
    worker_count = max(1, worker_count)
    actual_mode = "replicated" if requested_mode == "replicated" and worker_count > 1 else "single"

    q: mp.Queue = mp.Queue(maxsize=worker_count)
    workers = []
    for device_index in range(worker_count):
        p = mp.Process(
            target=run_worker,
            kwargs=dict(
                q=q,
                model=args.model,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                width=args.width,
                height=args.height,
                steps=args.steps,
                guidance=args.guidance,
                batch_size=args.batch_size,
                iterations=args.iterations,
                seed=args.seed,
                scheduler=args.scheduler,
                prefer_bf16=args.bf16,
                enable_xformers=args.xformers,
                device_index=device_index,
            ),
            daemon=True,
        )
        workers.append(p)

    def _sigint_handler(_signum, _frame):
        try:
            for proc in workers:
                if proc.is_alive():
                    proc.terminate()
        finally:
            sys.exit(130)

    signal.signal(signal.SIGINT, _sigint_handler)

    start = time.perf_counter()
    for proc in workers:
        proc.start()

    results = []
    try:
        for _ in workers:
            results.append(q.get(timeout=3600))
    except Empty:
        print("Worker timed out.", file=sys.stderr)
        results = None
    finally:
        for proc in workers:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
    wall_time_s = time.perf_counter() - start

    if results is not None:
        if args.emit_worker_rows and len(results) > 1:
            for row in worker_result_rows(results, args, actual_mode):
                write_metric(row, args.metrics_path)
        result = aggregate_results(results, args, worker_count, actual_mode, wall_time_s)
        write_metric(result, args.metrics_path)
        print("[RESULT]", result)
        if result.get("status") != "ok":
            print(f"[SD][ERROR] {result.get('error', 'unknown error')}", file=sys.stderr)
            sys.exit(2)
        print(
            f"[SD] model={result['model']} dtype={result['dtype']} "
            f"{result['hw']} steps={result['steps']} bs={result['batch_size']} "
            f"gpus={result['gpu_count']} mode={result['multi_gpu_mode']} "
            f"iters={result['iters']} load_s={result['load_seconds']} "
            f"mean_iter_s={result['mean_s_per_iter']} img/s={result['images_per_sec']}"
        )
    else:
        write_metric(
            {
                "suite": "sd_infer",
                "status": "failed",
                "model": args.model,
                "iters": args.iterations,
                "per_gpu_batch_size": args.batch_size,
                "batch_size": args.batch_size,
                "steps": args.steps,
                "hw": f"{args.width}x{args.height}",
                "gpu_count": worker_count,
                "multi_gpu_mode": actual_mode,
                "repeat_index": args.repeat_index,
                "repeat_count": args.repeat_count,
                "time_s": None,
                "end_to_end_time_s": round(wall_time_s, 4),
                "workers_ok": 0 if results is None else len([r for r in results if r.get("status") == "ok"]),
                "workers_failed": worker_count,
                "error": "Worker timed out",
            },
            args.metrics_path,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
