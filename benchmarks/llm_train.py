#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import time

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Enable TensorFloat32 on Ampere/Lovelace for better speed without extra memory
if hasattr(torch.backends, "cuda"):
    torch.backends.cuda.matmul.allow_tf32 = True

class SyntheticCausal(Dataset):
    def __init__(self, num_tokens=10_000_000, seq_len=2048, vocab=50257):
        self.num_tokens = num_tokens
        self.seq_len = seq_len
        self.vocab = vocab
        self.samples = num_tokens // seq_len
    def __len__(self): return self.samples
    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab, (self.seq_len,), dtype=torch.long)
        y = x.clone()
        return x, y

class TinyGPT(nn.Module):
    def __init__(self, vocab=50257, hidden=3072, n_layers=24, n_heads=24, seq_len=2048, dtype=torch.float32):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, batch_first=True)
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm = nn.Linear(hidden, vocab)
        self.seq_len = seq_len
        self.dtype = dtype
    def forward(self, x):
        x = self.emb(x).to(self.dtype)
        x = self.tr(x)
        return self.lm(x)

def init_dist():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


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


def visible_device_env_var() -> str:
    return "HIP_VISIBLE_DEVICES" if detect_backend() == "amd" else "CUDA_VISIBLE_DEVICES"

@torch.no_grad()
def tokens_per_sec(num_tokens, elapsed):
    return num_tokens / elapsed if elapsed > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="llm_train")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = cfg["llm_train"]

    init_dist()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if world_size > 1 else 0
    rank = int(os.environ.get("RANK", "0")) if world_size > 1 else 0
    backend = detect_backend()
    visible_device_env = visible_device_env_var()
    visible_devices = os.environ.get(visible_device_env, "")

    # ★ FIX 1: Correct dtype mapping
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg["dtype"]]

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = TinyGPT(hidden=cfg["hidden_size"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                    seq_len=cfg["seq_len"], dtype=dtype).to(device)

    # ★ FIX 2: Ensure model weights are the correct dtype
    model = model.to(dtype)
    model.train()

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    ds = SyntheticCausal(seq_len=cfg["seq_len"])
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True) if world_size > 1 else None
    dl = DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler, shuffle=(sampler is None),
                    num_workers=2, pin_memory=True, drop_last=True)

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Warmup
    optim.zero_grad(set_to_none=True)
    it = iter(dl)
    x, y = next(it)
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    logits = model(x)

    # ★ FIX 3: keep loss in float32 for numeric stability
    loss = loss_fn(logits.float().view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    optim.step()
    torch.cuda.synchronize(device)
    optim.zero_grad(set_to_none=True)

    steps = cfg["steps"]
    tokens_per_step = cfg["batch_size"] * cfg["seq_len"]
    if world_size > 1:
        tokens_per_step *= world_size

    # Timed run
    start = time.time()
    n = 0
    for i, (x, y) in enumerate(dl):
        if i >= steps: break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits.float().view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optim.step()
        n += 1
    torch.cuda.synchronize(device)
    elapsed = time.time() - start

    metric = {
        "benchmark_schema_version": 2,
        "suite": "llm_train",
        "status": "ok",
        "dtype": cfg["dtype"],
        "seq_len": cfg["seq_len"],
        "batch_size": cfg["batch_size"],
        "steps": steps,
        "world_size": world_size,
        "gpu_count": world_size,
        "gpu_backend": backend,
        "visible_device_env": visible_device_env,
        "visible_devices": visible_devices,
        "cuda_visible_devices": visible_devices,
        "distributed_backend": "nccl" if world_size > 1 else None,
        "tokens_per_step": tokens_per_step,
        "steps_per_sec": n / elapsed if elapsed > 0 else 0.0,
        "tokens_per_sec": tokens_per_sec(tokens_per_step, elapsed / n) if n else 0.0,
        "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
        "time_s": elapsed,
    }

    if rank == 0:
        os.makedirs("results", exist_ok=True)
        with open("results/metrics.jsonl", "a") as f:
            f.write(json.dumps(metric) + "\n")
        print(json.dumps(metric, indent=2))

if __name__ == "__main__":
    main()
