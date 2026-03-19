# GPU Benchmark Kit

This repo is a script-driven GPU benchmarking harness for comparing datacenter and consumer GPUs across:
- **LLM training** with a synthetic PyTorch transformer
- **LLM inference** with a canonical Transformers/PyTorch benchmark
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
bash run_all.sh
```

On the first run, `run_all.sh` bootstraps `.venv` automatically by invoking `env_setup.sh` if the repo environment is missing or incomplete.
That auto-bootstrap path is intended for supported Linux GPU hosts and now supports both NVIDIA/CUDA and AMD/ROCm setups through `gpu_backend`.
You can still run `bash env_setup.sh` manually if you want to preinstall dependencies ahead of time.
Blender is intentionally not installed by `env_setup.sh`; use [install_blender.sh](./install_blender.sh) if you want full-suite host setup.

For a quick validation run, use:

```bash
bash run_all.sh --smoke
```

Smoke mode writes an `effective_config.yaml` into the run folder and reduces repeat counts, per-suite durations, and sweep breadth so you can validate the harness without paying full benchmark cost.

Assumptions:
- Linux benchmark host
- NVIDIA drivers with `nvidia-smi` or AMD ROCm with `rocm-smi` are already installed
- Python 3.10, 3.11, or 3.12 is available
- Blender is a host-level prerequisite for full-suite runs; if it is missing and not required, the Blender benchmark is skipped

## Backend Support

The harness is intended to support both:
- NVIDIA GPUs through CUDA
- AMD GPUs through ROCm / HIP

Backend selection is controlled by `gpu_backend` in [config.yaml](./config.yaml):
- `auto`: detect from `nvidia-smi` or `rocm-smi`
- `nvidia`
- `amd`

What "supported" means in the current codebase:
- `run_all.sh` selects backend-specific visible-device env vars and host tooling
- benchmark result rows now include `gpu_backend` so cross-vendor comparisons remain explicit in `metrics.jsonl`, `metrics.csv`, and summaries
- Blender switches between `CUDA` and `HIP` based on the selected backend

Important caveat:
- practical support still depends on your installed ROCm/CUDA stack, PyTorch build, vLLM build, Diffusers stack, and Blender build on the target machine
- AMD support should be treated as implementation-level support that still requires runtime validation on a real ROCm host
- the default `llm_infer` path is backend-agnostic `transformers`; `vllm` remains optional and opt-in
- the default `env_setup.sh` path no longer installs `vllm` or `xformers` on AMD automatically; `llm_infer_vllm` is skipped unless you provide a separately validated ROCm-compatible `vllm` install
- Blender should be pinned to the same version across benchmark hosts; this repo provides [install_blender.sh](./install_blender.sh) for that purpose

## Runtime Flow

`run_all.sh` does the following:

1. Bootstraps `.venv` via `env_setup.sh` if needed, then activates it.
2. Reads `results_dir` from `config.yaml`.
3. Creates a unique run directory under `results/` using timestamp, hostname, GPU count, and GPU model.
4. Writes an `effective_config.yaml` for the run, applying smoke-mode overrides when requested.
5. Writes a system snapshot to `meta.json`.
6. Runs the benchmark scripts in sequence.
7. Saves per-benchmark logs under `logs/`.
8. Saves structured outputs under `results/` inside the run directory.
9. Runs `harness.py` to produce consolidated raw and repeat-summary outputs.

The run directory is the main artifact unit for the repo.

If `repeat` is greater than 1 in `config.yaml`, the orchestrator reruns each suite that many times and tags emitted rows with `repeat_index` and `repeat_count`.

Before benchmark execution, the harness also performs config validation, runtime estimation, and machine-state inspection.
It now also runs a system-requirements preflight that fails fast on unsupported hosts or missing required binaries.

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
- `run_all.sh` now launches this suite with DDP via `python -m torch.distributed.run` for each configured `llm_train.world_sizes` value
- Training scaling is controlled by `llm_train.world_sizes` in `config.yaml`
- The root `gpu_include` list constrains which local GPU indices are visible to the run
- The root `gpu_backend` selects `nvidia`, `amd`, or `auto`

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
- Intended to run on either NVIDIA or AMD as long as the installed PyTorch build exposes a working GPU runtime

### 2. LLM Inference

File: [benchmarks/llm_infer_hf.py](./benchmarks/llm_infer_hf.py)

What it does:
- Loads a model through Hugging Face Transformers on PyTorch
- Sweeps configured batch sizes with a canonical Transformers path used across GPU vendors
- Warms up, then runs repeated synchronous `generate()` calls for a fixed duration
- Records requests/sec, generated tokens/sec, batch latency stats, average power, and tokens/watt

Notes:
- This is an offline throughput-style benchmark, not an interactive latency benchmark
- Prompt length is now tokenizer-verified; results include both requested and actual prompt token counts
- The canonical `transformers` backend only supports `tensor_parallel=1`; larger configured TP values are emitted as structured skipped rows
- `multi_gpu_mode: replicated` launches one worker per visible GPU and aggregates total requests/sec and tokens/sec into a single `llm_infer` row
- Latency fields are measured per `generate()` batch call; `batch_latency_per_item_proxy_*` is a simple batch-latency-per-item proxy, not a true online per-request latency measurement
- Power sampling is currently NVIDIA-only through NVML; on AMD the benchmark still runs, but power-related fields may be zero or unavailable
- Optional vLLM benchmark: [benchmarks/llm_infer_vllm.py](./benchmarks/llm_infer_vllm.py)
- Set `llm_infer.backend: vllm` if you want to run the vLLM-specific benchmark instead of the canonical Transformers path

### 3. Stable Diffusion

File: [benchmarks/sd_infer.py](./benchmarks/sd_infer.py)

What it does:
- Builds a Diffusers `StableDiffusionPipeline` manually from subcomponents
- Runs a warmup image, then timed inference iterations
- Reports images/sec and per-iteration timing

Notes:
- Default model is now `stable-diffusion-v1-5/stable-diffusion-v1-5` because previously configured SD model identifiers are no longer reliably accessible on Hugging Face.
- `run_all.sh` runs it once per configured image size
- `multi_gpu_mode: replicated` now launches one worker per visible GPU and reports aggregate throughput
- If only one GPU is visible, replicated mode falls back to single-GPU execution
- The script writes structured rows directly to `metrics.jsonl`
- Failed SD runs are recorded as structured failed rows
- `emit_worker_rows: true` adds one `sd_infer_worker` row per GPU in replicated mode in addition to the aggregate row
- BF16 selection on AMD depends on the installed runtime and device support; when support cannot be inferred cleanly, the benchmark still emits the chosen dtype and reason fields

### 4. Blender

File: [benchmarks/blender_bench_cuda.sh](./benchmarks/blender_bench_cuda.sh)

What it does:
- Runs Blender in background mode against bundled scenes
- Measures wall-clock render time with `/usr/bin/time`
- Compares one GPU vs all GPUs using the configured Blender backend (`CUDA` for NVIDIA, `HIP` for AMD)

Notes:
- Scene selection now comes from `config.yaml` when `blender.scenes` is provided; otherwise it falls back to bundled scenes
- It writes structured rows directly to `metrics.jsonl` and also keeps a standalone JSON file in the run results directory
- It is skipped if `blender` is unavailable unless Blender is marked required in config/preflight

## Configuration

Primary configuration lives in [config.yaml](./config.yaml).

Main sections:
- `gpu_backend`
- `preflight`
- `llm_train`
- `llm_train_real`
- `llm_infer`
- `sd_infer`
- `blender`

Important caveat:
- `llm_train_real` is optional and disabled by default because it adds significant runtime and depends on model availability.
- `llm_infer.backend` defaults to `transformers` so inference results are comparable across mixed GPU vendors. The default model is currently `Qwen/Qwen3-8B`.
- `llm_infer.multi_gpu_mode` supports `single` and `replicated`. `replicated` is the default and runs one worker per visible GPU for comparable multi-GPU aggregate throughput.
- Set the backend to `vllm` only when you intentionally want the vLLM-specific benchmark.
- `blender.require_installed: true` and/or `preflight.blender_strict: true` turns missing Blender from a warning into a hard preflight error

## Outputs

Each `run_all.sh` execution creates a run folder like:

```text
results/<timestamp>_<host>_<gpu-tag>/
```

Typical contents:
- `meta.json`: machine snapshot plus captured software versions
- `effective_config.yaml`: the exact config used for that run, including smoke-mode overrides
- `machine_state.json`: preflight machine-state warnings/checks
- `logs/*.log`: per-suite logs
- `results/metrics.jsonl`: unified structured metrics for all active suites
- `results/*.json`: suite-specific JSON outputs such as Blender repeat files
- `metrics.csv`: consolidated CSV copied to the run root
- `metrics_summary.csv`: repeat-level summary CSV with mean/stdev/min/max for tracked metrics
- `metrics_summary.json`: repeat-level summary JSON
- `runtime_estimate.json`: estimated runtime breakdown for the configured run
- `system_requirements.json`: required/optional host-tool checks for the configured run

`harness.py` builds the CSV by reading:
- `results/metrics.jsonl`

It also builds repeat-level summaries grouped by benchmark configuration and status, with mean/stdev/min/max for tracked numeric metrics such as throughput and timing.

## Comparing Runs

Use [compare_runs.py](./compare_runs.py) to compare two or more completed run folders and generate:
- a Markdown report
- a machine-readable JSON payload

The comparison tool reads:
- `meta.json`
- `effective_config.yaml`
- `metrics_summary.json`

It groups rows by suite-specific comparison keys, labels groups as `strict`, `directional`, or `partial`, and adds caveats when runs differ in GPU count, backend, or benchmark coverage.
It also surfaces repeat variability from summary artifacts and now calls out caveats such as differing software versions and repeat counts.

Example:

```bash
.venv/bin/python compare_runs.py \
  --label AMD=results/20260317_081332_ezc-test-20260316_2xGPU \
  --label RTX4090=results/20260318_143637_ezc-benchmark-17c_8x4090 \
  --baseline RTX4090 \
  --out-dir results/comparison_report
```

This writes:
- `results/comparison_report/comparison.md`
- `results/comparison_report/comparison.json`

What the report includes:
- executive summary split into:
  - decision view for top-line workload picks
  - benchmark view for baseline-aware metric diagnostics
- run overview table
- comparability summary table
- grouped per-suite metric comparisons
- best-run annotations per metric
- tie-aware best-run reporting for near-equal results
- per-GPU throughput normalization when total GPU counts differ
- repeat variability using summary min/max/stdev/CV when available
- explicit "no decision-grade pick yet" messaging when a suite only has partial coverage

Baseline behavior:
- if `--baseline` is omitted, the first provided run is used
- `--baseline` may match a label, run folder name, or full run path

Label behavior:
- use `--label NAME=PATH` when you want readable report labels
- unlabeled run paths are still supported and fall back to `<run_id> [backend]`

Suite filtering:
- use `--suites llm_train,llm_infer` to restrict the report to selected suites

Current caveats the report can surface:
- differing `gpu_count` values
- backend or Blender render-backend differences
- differing `torch` / `transformers` versions
- differing repeat counts from summary rows

## Environment Setup

[env_setup.sh](./env_setup.sh) creates a local `.venv` and installs a pinned stack intended to work together:
- PyTorch / torchvision / torchaudio
- vLLM
- xFormers
- Diffusers
- Transformers
- Accelerate
- pandas / PyYAML / tqdm / safetensors

The script assumes system-level GPU driver setup is already handled outside the repo. It chooses CUDA or ROCm wheels based on `gpu_backend`.
On AMD, the default setup intentionally skips `vllm` and `xformers` because the common wheels are often CUDA-oriented or otherwise not validated for the target ROCm runtime.

## Metadata

Each run writes `meta.json` with:
- platform, kernel, Python version, and basic OS details
- backend-specific GPU tool snapshots (`nvidia-smi` or `rocm-smi`), CPU, and memory snapshots
- captured software versions for key tools and libraries such as PyTorch, vLLM, Diffusers, Transformers, xFormers, tokenizers, and Blender when available

## Preflight

Config validation:
- `validate_config.py` checks supported keys, required sections, and basic value constraints before the run begins.

Runtime estimation:
- `estimate_runtime.py` writes a rough per-suite and total runtime estimate to `runtime_estimate.json`.

Machine-state inspection:
- `check_machine_state.py` records GPU machine-state warnings to `machine_state.json`.
- `preflight.machine_state_strict: true` turns machine-state warnings into a hard stop before benchmark execution.

System requirements:
- `check_system_requirements.py` verifies the Linux GPU host assumptions and required binaries such as `nvidia-smi` or `rocm-smi` and `stdbuf`.
- Missing required host tools are a hard stop before benchmark execution begins.
- If Blender is enabled and required, missing `blender` or `/usr/bin/time` is also a hard stop before benchmark execution begins.

## Blender Install

For hosts that should run the full suite, install a pinned Blender build separately from the Python environment:

```bash
bash install_blender.sh
```

Defaults:
- Blender version: `4.2.18`
- install root: `~/.local/opt/blender-4.2.18`
- symlink: `~/.local/bin/blender`

`run_all.sh` prepends `~/.local/bin` and `~/bin` to `PATH`, and `check_system_requirements.py` also checks those locations directly, so the installed Blender binary is discoverable even in non-login shell sessions.

Example with explicit version:

```bash
BLENDER_VERSION=4.2.18 bash install_blender.sh
```

The installer downloads Blender from the official archive:
- `https://download.blender.org/release/Blender4.2/blender-4.2.18-linux-x64.tar.xz`

Recommended practice:
- use the same Blender version on every benchmark host
- set `blender.require_installed: true` once Blender is part of your required comparison suite

Post-run validation:
- `validate_run_artifacts.py` checks a completed run folder for missing artifacts, backend mismatches, missing suite rows, failed/skipped rows, and AMD power-metric caveats before you compare runs across vendors.

## Smoke-Test Checklist

Use this before trusting cross-vendor comparisons:

1. On the target machine, set `gpu_backend` in [config.yaml](./config.yaml) to the intended backend instead of relying on `auto` during initial validation.
2. Run `bash run_all.sh --smoke`.
3. Confirm the run folder contains `machine_state.json`, `runtime_estimate.json`, `meta.json`, and `results/metrics.jsonl`.
4. Check `meta.json` and verify `gpu_backend` and GPU model match the host you intended to benchmark.
5. Check `results/metrics.jsonl` and confirm each active suite writes rows with the expected `gpu_backend`, `status`, and `gpu_count`.
6. Review `logs/llm_train*.log`, `logs/llm_infer_vllm*.log`, `logs/sd_infer*.log`, and `logs/blender*.log` for backend-specific runtime errors.
7. On AMD hosts, verify whether `llm_infer` power fields are unavailable rather than silently assuming they are comparable to NVIDIA.
8. Only after smoke passes on both vendors should you run the full benchmark configuration for comparison.

After the smoke run, validate the produced run folder:

```bash
python3 validate_run_artifacts.py results/<run_id>
```

If you want to assert the intended backend explicitly:

```bash
python3 validate_run_artifacts.py results/<run_id> --expected-backend amd
python3 validate_run_artifacts.py results/<run_id> --expected-backend nvidia
```

Exit codes:
- `0`: no issues detected
- `1`: warnings detected; the run may still be usable, but comparison caveats need review
- `2`: errors detected; do not trust the run for comparison until fixed

For a fair NVIDIA vs AMD comparison:
- keep model ids, prompt lengths, output lengths, image sizes, steps, repeat count, and visible GPU count aligned
- compare rows by `suite`, `status`, `gpu_backend`, `gpu_count`, and the workload-defining config fields rather than by run folder name alone
- treat missing power metrics as a reporting limitation, not as zero-power performance

## Current Documentation vs Implementation

The following reflects the code as it exists now:
- The active vLLM benchmark entrypoint is `benchmarks/llm_infer_vllm.py`, not a `.sh` wrapper.
- The default orchestrated flow is mostly single-process and sequential.
- Blender benchmarking is integrated through `benchmarks/blender_bench_cuda.sh`.
- Post-run artifact validation is available through `validate_run_artifacts.py`.

## Notes

- Some Hugging Face models may require authentication or gated access.
- The repo already contains large assets and prior results under `assets/` and `results/`.
- There are lightweight repository tests under `tests/`, but they do not replace runtime validation on a real GPU host.
- If a dependency is missing, some sections may skip or fail independently while other suites still run.
- `validate_run_artifacts.py` is intended to catch obvious comparison hazards quickly, not to prove that two runs are methodologically identical.
