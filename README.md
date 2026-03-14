# GPU Benchmark Kit

This repo is a script-driven GPU benchmarking harness for comparing datacenter and consumer GPUs across:
- **LLM training** with a synthetic PyTorch transformer
- **LLM inference** with vLLM
- **Image generation** with Stable Diffusion / Diffusers
- **Rendering** with Blender Cycles

The main entrypoint is [run_all.sh](./run_all.sh). It creates a timestamped run folder, captures machine metadata, runs the benchmark suites sequentially, stores logs, and consolidates structured outputs into CSV.

## Repo Shape

This is not a packaged Python project. It is a collection of scripts plus local assets.

- `run_all.sh`: top-level orchestrator
- `config.yaml`: shared benchmark configuration
- `env_setup.sh`: creates `.venv` and installs the pinned Python stack
- `harness.py`: consolidates JSON/JSONL results into CSV
- `benchmarks/`: benchmark implementations
- `assets/`: local inputs, primarily Blender scenes
- `results/`: historical run outputs and reporting artifacts

## Quick Start

```bash
cd /path/to/benchmark
bash env_setup.sh
bash run_all.sh
```

Assumptions:
- NVIDIA drivers are already installed
- CUDA-compatible GPUs are visible to `nvidia-smi`
- Python 3 is available
- Blender is optional; the Blender benchmark is skipped if `blender` is not on `PATH`

## Runtime Flow

`run_all.sh` does the following:

1. Activates `.venv` if present.
2. Reads `results_dir` from `config.yaml`.
3. Creates a unique run directory under `results/` using timestamp, hostname, GPU count, and GPU model.
4. Writes a system snapshot to `meta.json`.
5. Runs the benchmark scripts in sequence.
6. Saves per-benchmark logs under `logs/`.
7. Saves structured outputs under `results/` inside the run directory.
8. Runs `harness.py` to produce consolidated raw and repeat-summary outputs.

The run directory is the main artifact unit for the repo.

If `repeat` is greater than 1 in `config.yaml`, the orchestrator reruns each suite that many times and tags emitted rows with `repeat_index` and `repeat_count`.

Before benchmark execution, the harness also performs config validation, runtime estimation, and machine-state inspection.

## Benchmark Suites

### 1. LLM Training

File: [benchmarks/llm_train.py](./benchmarks/llm_train.py)

What it does:
- Builds a small GPT-like model from standard PyTorch modules
- Uses synthetic token data generated on the fly
- Runs forward, loss, backward, and optimizer steps
- Reports step throughput and token throughput

What it is:
- A synthetic training microbenchmark

What it is not:
- A benchmark of a real pretrained model, real data pipeline, or production training stack

Multi-GPU:
- The script supports DDP if launched with `torchrun`
- The current `run_all.sh` invocation runs it as a normal single-process Python script

### 1b. Real-Model LLM Training

File: [benchmarks/llm_train_real.py](./benchmarks/llm_train_real.py)

What it does:
- Loads a real causal LM from Hugging Face
- Runs synthetic token batches through actual model weights
- Measures forward/backward/update throughput

Notes:
- Controlled by `llm_train_real.enabled` in `config.yaml`
- Disabled by default because it increases runtime and model-download requirements
- Currently single-GPU in the active orchestration flow

### 2. LLM Inference

File: [benchmarks/llm_infer_vllm.py](./benchmarks/llm_infer_vllm.py)

What it does:
- Loads a model through vLLM
- Sweeps configured batch sizes and tensor-parallel sizes
- Warms up, then runs repeated synchronous `generate()` calls for a fixed duration
- Records requests/sec, generated tokens/sec, batch latency stats, average power, and tokens/watt

Notes:
- This is an offline throughput-style benchmark, not an interactive latency benchmark
- Prompt length is now tokenizer-verified; results include both requested and actual prompt token counts
- Failures for oversized TP or batch settings are recorded as structured failed rows in `metrics.jsonl`
- Latency fields are measured per `generate()` batch call; `item_latency_*` is a simple batch-latency-per-item proxy, not a true online per-request latency measurement

### 3. Stable Diffusion

File: [benchmarks/sd_infer.py](./benchmarks/sd_infer.py)

What it does:
- Builds a Diffusers `StableDiffusionPipeline` manually from subcomponents
- Runs a warmup image, then timed inference iterations
- Reports images/sec and per-iteration timing

Notes:
- `run_all.sh` runs it once per configured image size
- `multi_gpu_mode: replicated` now launches one worker per visible GPU and reports aggregate throughput
- If only one GPU is visible, replicated mode falls back to single-GPU execution
- The script writes structured rows directly to `metrics.jsonl`
- Failed SD runs are recorded as structured failed rows
- `emit_worker_rows: true` adds one `sd_infer_worker` row per GPU in replicated mode in addition to the aggregate row

### 4. Blender

File: [benchmarks/blender_bench_cuda.sh](./benchmarks/blender_bench_cuda.sh)

What it does:
- Runs Blender in background mode against bundled scenes
- Measures wall-clock render time with `/usr/bin/time`
- Compares one CUDA GPU vs all CUDA GPUs

Notes:
- Scene selection now comes from `config.yaml` when `blender.scenes` is provided; otherwise it falls back to bundled scenes
- It writes structured rows directly to `metrics.jsonl` and also keeps a standalone JSON file in the run results directory
- It is skipped if `blender` is unavailable

## Configuration

Primary configuration lives in [config.yaml](./config.yaml).

Main sections:
- `preflight`
- `llm_train`
- `llm_train_real`
- `llm_infer`
- `sd_infer`
- `blender`

Important caveat:
- `llm_train_real` is optional and disabled by default because it adds significant runtime and depends on model availability.

## Outputs

Each `run_all.sh` execution creates a run folder like:

```text
results/<timestamp>_<host>_<gpu-tag>/
```

Typical contents:
- `meta.json`: machine snapshot plus captured software versions
- `machine_state.json`: preflight machine-state warnings/checks
- `logs/*.log`: per-suite logs
- `results/metrics.jsonl`: unified structured metrics for all active suites
- `results/*.json`: suite-specific JSON outputs such as Blender repeat files
- `metrics.csv`: consolidated CSV copied to the run root
- `metrics_summary.csv`: repeat-level summary CSV with mean/stdev/min/max for tracked metrics
- `metrics_summary.json`: repeat-level summary JSON
- `runtime_estimate.json`: estimated runtime breakdown for the configured run

`harness.py` builds the CSV by reading:
- `results/metrics.jsonl`

It also builds repeat-level summaries grouped by benchmark configuration and status, with mean/stdev/min/max for tracked numeric metrics such as throughput and timing.

## Environment Setup

[env_setup.sh](./env_setup.sh) creates a local `.venv` and installs a pinned stack intended to work together:
- PyTorch / torchvision / torchaudio
- vLLM
- xFormers
- Diffusers
- Transformers
- Accelerate
- pandas / PyYAML / tqdm / safetensors

The script assumes system-level CUDA and driver setup are already handled outside the repo.

## Metadata

Each run writes `meta.json` with:
- platform, kernel, Python version, and basic OS details
- `nvidia-smi`, CPU, and memory snapshots
- captured software versions for key tools and libraries such as PyTorch, vLLM, Diffusers, Transformers, xFormers, tokenizers, and Blender when available

## Preflight

Config validation:
- `validate_config.py` checks supported keys, required sections, and basic value constraints before the run begins.

Runtime estimation:
- `estimate_runtime.py` writes a rough per-suite and total runtime estimate to `runtime_estimate.json`.

Machine-state inspection:
- `check_machine_state.py` records GPU machine-state warnings to `machine_state.json`.
- `preflight.machine_state_strict: true` turns machine-state warnings into a hard stop before benchmark execution.

## Current Documentation vs Implementation

The following reflects the code as it exists now:
- The active vLLM benchmark entrypoint is `benchmarks/llm_infer_vllm.py`, not a `.sh` wrapper.
- The default orchestrated flow is mostly single-process and sequential.
- Blender benchmarking is integrated through `benchmarks/blender_bench_cuda.sh`.

## Notes

- Some Hugging Face models may require authentication or gated access.
- The repo already contains large assets and prior results under `assets/` and `results/`.
- There is no formal test suite at the moment.
- If a dependency is missing, some sections may skip or fail independently while other suites still run.
