#!/usr/bin/env python3
import argparse
import json
import os
import time

import torch
import yaml
from transformers import AutoModelForCausalLM


DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    cfg = cfg.get("llm_train_real", {})
    if not cfg.get("enabled", False):
        print("[SKIP] llm_train_real disabled in config")
        return

    if not torch.cuda.is_available():
        metric = {
            "suite": "llm_train_real",
            "status": "failed",
            "model": cfg.get("model"),
            "error": "CUDA not available",
        }
        os.makedirs("results", exist_ok=True)
        with open("results/metrics.jsonl", "a") as f:
            f.write(json.dumps(metric) + "\n")
        print(json.dumps(metric, indent=2))
        raise SystemExit(1)

    dtype_name = cfg.get("dtype", "fp16")
    dtype = DTYPE_MAP[dtype_name]
    model_name = cfg["model"]
    seq_len = int(cfg["seq_len"])
    batch_size = int(cfg["batch_size"])
    steps = int(cfg["steps"])
    device = torch.device("cuda:0")

    torch.backends.cuda.matmul.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    vocab_size = getattr(model.config, "vocab_size", 50257)

    def sample_batch():
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
        return x, x.clone()

    optim.zero_grad(set_to_none=True)
    x, y = sample_batch()
    out = model(input_ids=x, labels=y)
    out.loss.float().backward()
    optim.step()
    torch.cuda.synchronize(device)
    optim.zero_grad(set_to_none=True)

    start = time.time()
    for _ in range(steps):
        x, y = sample_batch()
        optim.zero_grad(set_to_none=True)
        out = model(input_ids=x, labels=y)
        out.loss.float().backward()
        optim.step()
    torch.cuda.synchronize(device)
    elapsed = time.time() - start

    tokens_per_step = batch_size * seq_len
    metric = {
        "suite": "llm_train_real",
        "status": "ok",
        "model": model_name,
        "dtype": dtype_name,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "steps": steps,
        "tokens_per_step": tokens_per_step,
        "steps_per_sec": steps / elapsed if elapsed > 0 else 0.0,
        "tokens_per_sec": (steps * tokens_per_step) / elapsed if elapsed > 0 else 0.0,
        "gpu_name": torch.cuda.get_device_name(device),
        "time_s": elapsed,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/metrics.jsonl", "a") as f:
        f.write(json.dumps(metric) + "\n")
    print(json.dumps(metric, indent=2))


if __name__ == "__main__":
    main()
