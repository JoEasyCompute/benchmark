# GPU Benchmark Kit

This repo is a script-driven GPU benchmarking harness for comparing datacenter and consumer GPUs across:
- **LLM training** with a synthetic PyTorch transformer
- **LLM inference** with a canonical Transformers/PyTorch benchmark
- **Image generation** with Stable Diffusion / Diffusers
- **Rendering** with Blender Cycles
- **Vision inference** with pinned pretrained torchvision models
- **LLM serving** with a local Transformers worker or streaming vLLM endpoint
- **Kernel diagnostics** for GEMM, attention, and device-copy bandwidth
- **Board energy** where selected-device telemetry is complete

The main entrypoint is [run_all.sh](./run_all.sh). It creates a timestamped run folder, captures machine metadata, runs the benchmark suites sequentially, stores logs, and consolidates structured outputs into CSV.

## Repo Shape

This is not a packaged Python project. It is a collection of scripts plus local assets.

- `run_all.sh`: top-level orchestrator
- `config.yaml`: shared benchmark configuration
- `configs/auto.yaml`, `configs/nvidia.yaml`, `configs/amd.yaml`: matched complete examples
- `env_setup.sh`: creates `.venv` and installs the pinned Python stack
- `run_optional_suites.py`: executes all configured new-suite combinations and repeats
- `lock_model_revisions.py`: resolves model revisions once per run
- `harness.py`: consolidates JSON/JSONL results into CSV
- `benchmarks/`: benchmark implementations
- `assets/`: local inputs, primarily Blender scenes
- `results/`: historical run outputs and reporting artifacts

## Quick Start

Run these commands from the repository root on a Linux GPU host. For a new host,
follow the [recommended procedure](#recommended-run-procedure) below before the full run.

```bash
cd /path/to/benchmark
bash run_all.sh --baseline --smoke
```

On the first run, `run_all.sh` bootstraps `.venv` automatically by invoking `env_setup.sh` if the repo environment is missing or incomplete.
It first makes YAML configuration readable when necessary, probes responding GPU
devices, and selects the CUDA or ROCm installation path. You do not need to change
`gpu_backend: auto` when moving between NVIDIA-only and AMD-only hosts.
For the stricter driver/architecture-aware package resolver and post-install checks,
see [Runtime compatibility](docs/runtime-compatibility.md). The normal runner records
the inspection and plan; use the resolver's `--install` mode in a new virtual
environment when changing framework versions.

For driver/architecture-aware environment setup, inspect and resolve the host first:

```bash
python3 inspect_runtime.py --backend auto --json-out host_inventory.json
python3 runtime_resolver.py --host-json host_inventory.json --json-out runtime_plan.json
```

The resolver selects a complete pinned profile from the detected GPU architecture,
driver/runtime, OS, kernel and Python ABI. It stops when the host is outside the
published compatibility matrix. To run an experimental installation after reviewing
the warning, use `--allow-unverified-host`; it never changes drivers or kernels:

```bash
python3 runtime_resolver.py --host-json host_inventory.json \
  --allow-unverified-host --install --venv .venv-rocm72
```

The installation writes `runtime-lock.json` and runs `verify_runtime.py`, which
checks package identity, a known 2×2 GPU matmul on every visible card, and optional
multi-GPU all-reduce. Keep the existing `.venv` until this new environment passes
those checks. `run_all.sh` records an inventory and runtime plan in every normal run;
the explicit resolver command is the safe path for changing an environment.
You can still run `bash env_setup.sh` manually if you want to preinstall dependencies ahead of time.
For Blender-enabled runs, `run_all.sh` uses [install_blender.sh](./install_blender.sh) to verify or install Blender 5.3.0 before workloads begin.
If the downloaded Blender binary reports missing desktop/X11 libraries, install
them through the repository installer with `bash install_blender.sh --install-system-deps`.
This is explicit because it uses `sudo apt-get` and changes host packages.
The installer also creates a wrapper in `~/.local/bin/blender` that sets Blender's
bundled `lib/` directory in `LD_LIBRARY_PATH`; this is required because a plain
symlink breaks the binary's `$ORIGIN/lib` lookup.

For a quick validation run, use:

```bash
bash run_all.sh --smoke
```

For a controlled card-level baseline, use `bash run_all.sh --baseline`.
This profile selects one configured GPU, runs five repeats, forces single-GPU
training/inference/SD execution, and records `benchmark_profile: single_gpu_baseline`
in the effective configuration. Combine it with
`--smoke` to validate wiring without a full benchmark run.

### Benchmark storage

Mount a writable data volume at `/mnt/benchmark-scratch` before running the
suite. For a different mounted volume, set `BENCHMARK_STORAGE_ROOT` to its
mount point. The runner checks the mount and available space before setting
up the GPU environment. It stores temporary files, Hugging Face and pip
caches, and other application caches on that volume. Existing Hugging Face
login tokens remain readable from their original location.

```bash
BENCHMARK_STORAGE_ROOT=/mnt/gpu-data bash run_all.sh
```

Without an override, new runs go under `/mnt/benchmark-scratch/results/`.
A relative `results_dir` is resolved under the storage mount. An absolute
`results_dir` explicitly selects a different output location and gets its own
space check. Historical results in the checkout remain available for reports.

The default free-space floor is 20 GiB. Set `BENCHMARK_MIN_FREE_GIB` to raise
it for a cold model cache. Allow several hundred GiB for a full eight-GPU
sweep with the large checkpoints. The floor is a preflight check and does
not reserve space. `--dry-run` can inspect the config without mounting storage.
Python virtual environments still live in the checkout.

On a host with the data volume mounted at `/var/lib/docker`, bind-mount a
dedicated benchmark directory from that volume at `/mnt/benchmark-scratch`.
Keep it separate from Docker's managed directories and configure the bind
mount to come up after the underlying volume at boot.

### Standard LLM inference sweep

`run_all.sh` runs the original Qwen3-8B baseline plus the checkpoints in
`llm_infer_matrix` in `config.yaml`.

| Workload | Standard cases |
|---|---|
| Text | Qwen2.5 7B Instruct, Llama 3.1 8B Instruct, Mistral 7B Instruct v0.3, Gemma 2 9B Instruct, and Qwen2.5 Coder 7B Instruct. Each runs on one GPU and with one replica per selected GPU. |
| Image and text | Qwen2-VL 7B Instruct with one deterministic 224px or 448px image per request, on one GPU and with one replica per selected GPU. |
| vLLM text | Qwen3-8B on 1, 2, 4, and 8 GPUs, Qwen3-32B on 4 and 8 GPUs, and Qwen2.5-72B-Instruct on 8 GPUs. The same matrix applies to CUDA and ROCm. |

Additional cases use batches of 1 and 4 and request 512 text prompt tokens
and 128 generated tokens. Chat formatting and image tokens are recorded in
the measured prompt length. Text replica batches are per GPU. vLLM batches are global. Smoke runs take the first case per workload, with short prompts
and batch size 1. The single-GPU baseline keeps Qwen3-8B vLLM at TP=1.

Accept the checkpoint terms on Hugging Face and provide `HF_TOKEN` or log in
on the benchmark host for gated checkpoints such as Llama and Gemma. A gated
model without access, a missing GPU count, or a failed vLLM preflight yields a
skip row and lets the remaining workloads continue. Each successful row carries
the resolved checkpoint revision. The large-model rows use the same pinned
vLLM version and tensor-parallel sizes for cross-host comparisons. An
eight-GPU host schedules 21 additional configurations across the matrix
before counting batches and repeats. Inspect the planned workload with
`bash run_all.sh --dry-run` and find measured or skipped cases in
`results/metrics.jsonl` inside the run directory. Set `llm_infer_matrix: []`
to run the original inference baseline by itself.

### Optional Qwen inference profiles

These profiles run specific cases from the standard sweep on their own. They use pinned
checkpoint revisions, 512 input tokens, 128 output tokens, and batch sizes 1 and 4.

| Config | Model | GPU configuration |
|---|---|---|
| `configs/infer-qwen25-7b.yaml` | Qwen2.5-7B-Instruct, FP16 Transformers | One replica per selected GPU. Use `--gpus 0` for the single-card reference. |
| `configs/infer-qwen3-32b.yaml` | Qwen3-32B, BF16 vLLM | Tensor parallelism at 4 and 8 GPUs |
| `configs/infer-qwen25-72b.yaml` | Qwen2.5-72B-Instruct, BF16 vLLM | Tensor parallelism at 8 GPUs |

```bash
bash run_all.sh --config configs/infer-qwen25-7b.yaml --gpus 0
bash run_all.sh --config configs/infer-qwen25-7b.yaml
```

Use an 8× RTX 4090 rig as the reference for the large profiles and repeat the
same cases on faster rigs. Start with `--smoke` to check model loading at the first
configured tensor-parallel size. The full profiles use physical GPU indices 0–7
by default. Pass `--gpus` to select a different set of eight cards. The 4-GPU
case uses the first four selected cards.

Both large profiles pin vLLM 0.34.0. For the NVIDIA wheel, use a Linux host
with a CUDA 13.0-compatible NVIDIA driver (580.65.06 or newer) and Python
3.11–3.14. If the main suite runs under Python 3.10, put `python3.12` or
`python3.11` on PATH for the vLLM environment. See the
[vLLM 0.34.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.34.0)
and [NVIDIA's CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html).

On AMD, vLLM 0.34.0 uses the official `+rocm100` wheel on Linux x86-64 with
Python 3.12, glibc 2.35 or newer, and a compatible AMD host driver. Its bundled
ROCm user-space dependencies are installed in `.venv-vllm-0.34.0-rocm`.
[ROCm installation guide](https://github.com/vllm-project/vllm/blob/v0.34.0/docs/getting_started/installation/gpu.rocm.inc.md)

Before workloads start, `run_all.sh` checks the main Python stack against the
resolved runtime and repairs version drift. For the large profiles, it creates
or repairs the vendor-specific vLLM environment, checks the GPU build and package
dependencies, and verifies BF16 arithmetic and NCCL or RCCL communication on the
selected devices. Pass `VLLM_PYTHON` to use a separately managed interpreter.
That interpreter is verified in place. Setup leaves system drivers unchanged.
Use `--dry-run` to inspect the effective config without preparing environments.
Preflight diagnostics are in `logs/vllm_preflight.log`.

```bash
bash run_all.sh --config configs/infer-qwen3-32b.yaml --smoke
bash run_all.sh --config configs/infer-qwen3-32b.yaml
bash run_all.sh --config configs/infer-qwen25-72b.yaml --smoke
bash run_all.sh --config configs/infer-qwen25-72b.yaml
```

The large-model batch size is global across the shards. The profiles disable
prefix caching, cap the context length at 1,024 tokens, and use vLLM 0.34.0’s automatic KV-cache sizing. Inspect throughput
and p95 batch latency in the comparison report, sampled peak device-wide VRAM
by GPU UUID in `results/metrics.jsonl`, and PCIe topology in `meta.json`.
VRAM samples include other allocations on each device, so use idle GPUs and
allow headroom for model initialization and KV cache. The smoke profile reduces
the token lengths and batch size while retaining the first tensor-parallel size.

### PearlHash mining benchmark

The standard suite includes PearlHash and records a skip until the local miner,
wallet, and GPU mapping are provided. Extract a pinned Linux release of
[SRBMiner-MULTI](https://github.com/doktor83/SRBMiner-Multi/releases) into a
versioned directory and use the same release on every machine. The suite records
the miner version and binary SHA-256.

Use SRBMiner's `--list-devices` to match its IDs to the benchmark's physical GPU
indices by PCI bus address. Use the same GPU visibility as the benchmark run
when checking the miner's IDs. This example assumes all eight indices match.

```bash
export SRBMINER_BIN="/absolute/path/to/SRBMiner-MULTI"
export PEARL_WALLET="prl1..."
export PEARL_GPU_MAP="0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7"
bash run_all.sh --smoke
bash run_all.sh
```

Use the wallet address without a worker suffix. The benchmark appends the
hostname and connects to Kryptex over TLS. Full runs measure one selected GPU
and, on multi-GPU configurations, all selected GPUs for 10 minutes each after
a 5-minute warm-up. With three repeats that adds about 90 minutes plus miner
startup. Smoke runs measure one GPU for 20 seconds.

Find total and per-GPU hashrate and measured-window share counts in
`results/metrics.jsonl` under the run directory. Board telemetry adds
power and hashes per joule when available. Use `pearlhash.enabled: false` in
your config to omit the workload.

### Optional diagnostic suites

Optional diagnostic suites are configured but disabled by default: `vision_infer`,
`kernel_bench`, and `llm_serve`. Enable them explicitly after validating their
runtime dependencies. Serving supports a local Transformers worker or a configured
streaming vLLM endpoint. See [the expanded suite guide](docs/expanded-benchmarks.md)
for configuration, metric definitions, energy coverage and hardware rollout steps.

Smoke mode writes an `effective_config.yaml` into the run folder and reduces repeat counts, per-suite durations, and sweep breadth so you can validate the harness without paying full benchmark cost.

Assumptions:
- Linux benchmark host
- NVIDIA drivers with `nvidia-smi` or AMD ROCm with `rocm-smi` are already installed
- Python 3.10, 3.11, or 3.12 is available
- Blender-enabled runs require Blender 5.3.0 and working Cycles devices. Use `blender.enabled: false` for a run without Blender.

## Automatic Detection and Sample Configs

Use the complete [auto example](configs/auto.yaml) on either vendor without editing
the root config:

```bash
bash run_all.sh --config configs/auto.yaml --baseline --dry-run
bash run_all.sh --config configs/auto.yaml --baseline --smoke
bash run_all.sh --config configs/auto.yaml --baseline
```

`--dry-run` prints resolved settings without installing the GPU stack, downloading
models or executing benchmarks. It requires Python with PyYAML already available.
The normal run can bootstrap that dependency automatically. Use `bash run_all.sh --help`
to see all supported switches.

| Example | Behavior |
| --- | --- |
| [configs/auto.yaml](configs/auto.yaml) | Detect NVIDIA or AMD; all suite families enabled |
| [configs/nvidia.yaml](configs/nvidia.yaml) | Require NVIDIA, with the same workloads |
| [configs/amd.yaml](configs/amd.yaml) | Require AMD, with the same workloads |
| [configs/auto-8gpu.yaml](configs/auto-8gpu.yaml) | Detect either vendor; eight-GPU scaling profile |
| [configs/nvidia-8gpu.yaml](configs/nvidia-8gpu.yaml) | NVIDIA eight-GPU scaling profile |
| [configs/amd-8gpu.yaml](configs/amd-8gpu.yaml) | AMD eight-GPU scaling profile |

The examples differ only in their backend selector. Detection selects devices,
the visibility environment variable, Blender CUDA/HIP and the compatible Python
stack. It does not silently change model, precision or batch settings between
vendors. Run folders record the resolved choices; source YAML files stay unchanged.
Environment setup uses the runtime resolver's package profile so installation
and validation agree. An acknowledged experimental existing stack is preserved
for validation; it is not an installation recipe. Optional vLLM/xformers stacks
must match the chosen runtime and are not installed automatically.

For a fresh Ubuntu 24.04 / Python 3.12 / gfx1201 host with ROCm 10.0.0,
the resolver provides an experimental installation profile using the official
PyTorch 2.8.0, torchvision 0.23.0 and torchaudio 2.8.0 ROCm 6.4 wheels.
This matches the framework family used in the earlier AMD runs; it is not native
ROCm 10 support. Common dependencies follow the repository pins, so record and
compare the complete software inventory. Existing Torch 2.8 environments remain
unchanged. Installation requires acknowledgement and must pass numerical checks:

```bash
ALLOW_UNVERIFIED_HOST=1 GPU_BACKEND=amd bash env_setup.sh
./run_all.sh --config configs/amd.yaml --gpus 0 --baseline --smoke --allow-unverified-host
```

The same `run_all.sh` command installs the stack automatically when missing.
After smoke validation succeeds, omit `--smoke` for the full baseline.

For mixed-vendor machines or a specific card, override from the command line:

```bash
bash run_all.sh --config configs/auto.yaml --backend amd --gpus 1 --baseline
bash run_all.sh --config configs/nvidia.yaml --gpus 0 --baseline --smoke
```

`--backend` overrides `gpu_backend`; `--gpus` overrides `gpu_include`. Selection
respects inherited CUDA/HIP visibility masks. With no explicit device list,
baseline mode takes the first detected/visible card. Auto mode checks responding
devices rather than just installed tools; if both vendors respond, it requires an
explicit `--backend`, and if neither responds it reports a driver/tooling error.
NVIDIA indices are translated to UUIDs for execution so CUDA enumeration order
cannot silently select another card. AMD selection uses HIP device indices.
See [sample configuration details](configs/README.md) for precedence and limitations.

The `*-8gpu.yaml` profiles set synthetic training world sizes to `[1, 2, 4, 8]`
and enable replicated LLM inference and Stable Diffusion workers. Select the
profile directly at runtime; no YAML editing is required:

```bash
./run_all.sh --config configs/amd-8gpu.yaml --gpus 0,1,2,3,4,5,6,7 --smoke --allow-unverified-host
./run_all.sh --config configs/amd-8gpu.yaml --gpus 0,1,2,3,4,5,6,7 --allow-unverified-host
```

The base `auto.yaml`, `amd.yaml`, and `nvidia.yaml` files remain single-GPU
workload profiles. `--baseline` additionally forces one GPU and five repeats.

## Recommended Run Procedure

Use this sequence on each card/host. A complete comparison includes successful
application measurements, serving latency, kernel diagnostics and any available
energy data, with failures and skipped coverage retained in the report.
The implementation has local automated-test coverage; physical NVIDIA/AMD runs
still need qualification on the hosts where you will benchmark.

### 1. Prepare the host and environment

Install the GPU drivers/runtime first. Inspect the available cards using
`nvidia-smi` on NVIDIA or `rocm-smi` on AMD. Use the same repository revision,
workload settings and Blender version on both hosts. Record any differences in
CPU, GPU count, power limits and software versions.

The runner can set up the environment itself. To prepare it manually, automatic
selection uses the same device probes:

```bash
bash env_setup.sh
```

On a mixed-vendor host, use `GPU_BACKEND=nvidia bash env_setup.sh` or
`GPU_BACKEND=amd bash env_setup.sh` to select the stack explicitly.

Then activate it before manual Python commands so `python3` uses the installed
dependencies:

```bash
source .venv/bin/activate
python -m pip check
bash install_blender.sh
```

Blender installation is needed only when testing rendering. Its installer does not
supply scenes: place `.blend` files under `assets/blender/` or configure their paths.
Resolve relevant dependency conflicts before benchmarking. AMD setup deliberately
omits vLLM/xFormers and removes them from this environment if already present;
keep a separately validated ROCm vLLM server in its own environment.

### 2. Enable the complete application and diagnostic suite

For ready-to-run examples, choose `--config configs/auto.yaml`,
`--config configs/nvidia.yaml` or `--config configs/amd.yaml`. Those examples enable
all suite families with identical workload settings. If customizing instead,
edit [config.yaml](config.yaml) or a separate file selected with `--config PATH`.
Merge the following values into its existing sections, preserving model/shape
settings; do not append duplicate YAML section names. Keep `gpu_backend: auto`
for portable selection and use `--gpus` when you want a specific physical card.

```yaml
gpu_backend: auto
gpu_include: []
repeat: 5

llm_train:
  enabled: true
llm_train_real:
  enabled: true
llm_infer:
  enabled: true
  backend: transformers
sd_infer:
  enabled: true
blender:
  enabled: true
  require_installed: true

vision_infer:
  enabled: true
  model: resnet18
  weights: ResNet18_Weights.IMAGENET1K_V1
  dtype: float32
  modes: [throughput, latency]
  batch_sizes: [1, 32]
  sizes: [224]
  iterations: 50
  warmup: 5

kernel_bench:
  enabled: true
  cases: [gemm, attention, memory]
  size: 2048
  dtype: float32
  iterations: 20
  warmup: 3
  heads: 8
  head_dim: 64

llm_serve:
  enabled: true
  provider: transformers
  endpoint: ""
  dtype: float16
  concurrency: [1, 4]
  duration: 30
  warmup: 1
  timeout: 120
  output_len: 32
```

This includes all suite families using **local Transformers serving**; no server
startup is needed. Unless `llm_serve.model` is set, serving uses `llm_infer.model`.
The four opt-in sections (`llm_train_real`, `vision_infer`, `kernel_bench`,
`llm_serve`) are disabled in the shipped configuration. `--baseline` does not enable them.

| Suite / config key | What it measures | Execution behavior |
| --- | --- | --- |
| `llm_train` | Synthetic transformer training throughput | Sweeps `world_sizes`; baseline forces `[1]` |
| `llm_train_real` | Real causal-LM training on fixed synthetic tokens | Single GPU; includes finite loss/gradient checks |
| `llm_infer` | Offline batched generation | Transformers by default; optional offline vLLM |
| `sd_infer` | Image-generation throughput | Sweeps image sizes; single or replicated workers |
| `vision_infer` | Pretrained vision inference throughput and latency | Sweeps sizes/batches; latency mode uses batch one |
| `kernel_bench` | GEMM, attention and device-copy rates | Sweeps configured cases on one GPU; diagnostic only |
| `llm_serve` | Concurrent requests, first-token/chunk timing and latency | Local serialized model worker or external streaming endpoint |
| `blender` | Render-operation and process timing | Single/all modes; baseline runs single only |

There is no energy enable flag. Supported workloads attempt sampling automatically;
missing or incomplete telemetry yields `null`, not an efficiency result.

### 3. Validate configuration, then run smoke

```bash
python validate_config.py --config configs/auto.yaml
bash run_all.sh --config configs/auto.yaml --baseline --smoke
```

Substitute your selected config path in this and subsequent commands. Omitting
`--config` uses the root `config.yaml` with its own enable flags and workloads.

The runner performs requirements checks, estimates runtime, records machine state
and locks model revisions. Smoke reduces repeats, shapes/iterations and serving
duration. Model downloads and loading still occur and can dominate the first run.
Blender smoke limits configured scenes but does not reduce scene render samples.
The runtime estimate is approximate and does not bound downloads or server drains.

Copy the exact run folder printed by the runner into `RUN_DIR` below; replace the
example path rather than choosing a folder solely because it is newest:

```bash
RUN_DIR="/mnt/benchmark-scratch/results/your-smoke-run-id"
python validate_run_artifacts.py "$RUN_DIR" --expected-backend nvidia
```

Use `--expected-backend amd` on AMD. The artifact validator returns `0` for no
issues, `1` for warnings and `2` for errors. Inspect `system_requirements.json`,
`machine_state.json`, `results/metrics.jsonl` and `logs/`. An enabled suite that
only skipped is not hardware-qualified, even if the runner exited successfully.
The runner's nonzero benchmark exit also leaves useful logs/artifacts to inspect.

### 4. Match revisions and workloads across hosts

The smoke folder's `model_revisions.json` lists resolved commits; its
`effective_config.yaml` contains those commits in each enabled local model section.
Copy the `revision` values for `llm_train_real`, `llm_infer`, `sd_infer` and local
`llm_serve` into the selected source config on **both** hosts. Copy only the revision
values, not the reduced smoke configuration. Set the serving model explicitly if
needed to associate its revision with the correct model.

Also match precision, batches, prompt/output lengths, image sizes, pretrained vision
weights and Blender scene bytes/settings. Let the second host run its own smoke
check with these values. If a workload exceeds memory, retain its failure as a
capacity result; use the same reduced workload on both hosts for a successful
performance comparison. Do not silently change only one vendor's settings.

### 5. Run and validate the full single-GPU baseline

```bash
bash run_all.sh --config configs/auto.yaml --baseline
```

Validate the newly printed folder as in step 3. This runs every enabled suite with
five repeats. Stop unrelated GPU jobs and external serving processes on the tested
card first; their memory/power usage can affect the results. Keep the complete run
folders from both hosts, including logs and raw metrics.

### 6. Run separate scaling and offline-vLLM experiments

For **system scaling**, edit the existing sections as follows (use valid physical
GPU indices and the same count on both systems), then omit `--baseline`:

```yaml
gpu_include: [0, 1]
repeat: 5
llm_train:
  world_sizes: [1, 2]
llm_infer:
  backend: transformers
  multi_gpu_mode: replicated
sd_infer:
  multi_gpu_mode: replicated
```

```bash
bash run_all.sh --config configs/auto.yaml --smoke
bash run_all.sh --config configs/auto.yaml
```

General smoke intentionally reduces training to world size one; it does not
qualify DDP scaling. Inspect world-size-two results in the full run. Real-model
training, vision, kernels and local serving still use one GPU even when more are
visible. Blender runs single and all modes. Pair selection applies only to synthetic
world-size-two training; preserve its selection artifact when comparing scaling.

For **offline vLLM**, set `llm_infer.backend: vllm` and its
`tensor_parallel_sizes` to the intended values. Use `gpu_include: [0]`, `repeat: 5`
and `llm_train.world_sizes: [1]` for a one-card experiment, then run without
`--baseline`: that flag forces offline inference back to Transformers. Offline
vLLM is distinct from `llm_serve.provider: vllm`. Use a runtime validated for that
vendor; AMD's default environment does not install it.

### 7. Run an external vLLM serving experiment

This is a separate run from the complete local suite. The runner does not start,
stop or select GPUs for an already running server. In a terminal with a validated
vLLM environment, replace the revision below with the resolved commit for this model:

```bash
MODEL_REVISION="replace-with-the-resolved-model-commit"
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B \
  --revision "$MODEL_REVISION" --dtype half --tensor-parallel-size 1 \
  --host 127.0.0.1 --port 8000
```

On AMD, use `HIP_VISIBLE_DEVICES=0` instead of the NVIDIA prefix. Wait for the server
to report readiness. See the [vLLM 0.11 serving documentation](https://docs.vllm.ai/en/v0.11.0/serving/openai_compatible_server.html)
for its runtime requirements and options. Record the server command, model commit,
precision, GPU allocation and version alongside the run artifacts.

In your selected config (for example, `configs/auto.yaml`), disable the other workloads so they do not compete with the
server on its GPU, and configure the full completion URL:

```yaml
llm_train:
  enabled: false
llm_train_real:
  enabled: false
llm_infer:
  enabled: false
sd_infer:
  enabled: false
vision_infer:
  enabled: false
kernel_bench:
  enabled: false
blender:
  enabled: false
llm_serve:
  enabled: true
  provider: vllm
  endpoint: http://127.0.0.1:8000/v1/completions
  model: Qwen/Qwen3-8B
  concurrency: [1, 4]
  duration: 30
  output_len: 32
  warmup: 1
  timeout: 120
```

Activate the benchmark environment in a second terminal, validate this config,
then run `bash run_all.sh --config configs/auto.yaml --smoke` and
`bash run_all.sh --config configs/auto.yaml`. Substitute your config path and validate both folders.
Keep `repeat: 5` if that is the comparison protocol. Stop the server before returning
to other workloads and restore their enable flags.

The client measures first nonempty streamed content and chunk gaps over HTTP.
It cannot infer token-level gaps, server queue time, GPU count, model revision or
energy from that connection. Missing usage means null token throughput. Local
Transformers serving measures token-ID callbacks and queue delay, so its timings
belong in a different comparison group from HTTP serving.

### 8. Generate and review reports

Copy the run folders to one machine with the benchmark Python environment. Replace
the following paths with the validated full runs, not smoke folders:

```bash
NVIDIA_RUN="results/your-nvidia-full-run-id"
AMD_RUN="results/your-amd-full-run-id"
python compare_runs.py \
  --label "NVIDIA=$NVIDIA_RUN" --label "AMD=$AMD_RUN" \
  --baseline NVIDIA --out-dir results/comparison_full
```

Here `compare_runs.py --baseline NVIDIA` selects a report reference; it does not
apply the `run_all.sh --baseline` execution profile. Use separate reports for
single-card, scaling and HTTP-serving runs. To focus on the new suites:

```bash
python compare_runs.py \
  --label "NVIDIA=$NVIDIA_RUN" --label "AMD=$AMD_RUN" \
  --suites llm_train_real,vision_infer,kernel_bench,llm_serve \
  --baseline NVIDIA --out-dir results/comparison_expanded
```

Review `comparison.md` and `comparison.json` for failed/skipped coverage, matching
workloads and repeat variability before using their recommendations. Cross-vendor
groups are normally `directional`; `strict` is an identity/metadata classification,
not proof of hardware fairness. Kernels are diagnostics, per-GPU normalization is
not an isolated-card measurement, and incomplete energy is not zero consumption.
Energy tables require complete telemetry across repeats and infer no energy winner.

## Backend Support

The harness is intended to support both:
- NVIDIA GPUs through CUDA
- AMD GPUs through ROCm / HIP

Backend selection comes from `--backend` or `gpu_backend` in the selected config:
- `auto`: probe responding GPUs through `nvidia-smi` and `rocm-smi`; require an override if both respond
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

1. Reads the selected config, bootstrapping PyYAML when needed, and resolves GPU/backend choices.
2. Installs/repairs the chosen CUDA/ROCm stack if needed, activates it, and reads `results_dir`.
3. Creates a unique run directory under the configured storage results directory using timestamp, hostname, GPU count, and GPU model.
4. Writes an `effective_config.yaml` for the run, applying smoke-mode overrides when requested.
5. Writes a system snapshot to `meta.json`.
6. Runs the benchmark scripts in sequence.
7. Saves per-benchmark logs under `logs/`.
8. Saves structured outputs under `results/` inside the run directory.
9. Runs `harness.py` to produce consolidated raw and repeat-summary outputs.

The run directory is the main artifact unit for the repo.

If `repeat` is greater than 1 in `config.yaml`, the orchestrator reruns each suite that many times and tags emitted rows with `repeat_index` and `repeat_count`.

For `llm_train` with `world_size: 2`, the orchestrator can run a short pre-benchmark pair probe before the main repeats. Configure `llm_train.pair_selection` in `config.yaml`; the default `benchmark` strategy tries candidate two-GPU pairs with a small step count, records `llm_train_ws2_pair_selection.json`, and uses the fastest pair for every ws=2 repeat. Use `strategy: topology` for a faster topology-only choice or `strategy: first` to preserve first-two-GPU behavior.

Before benchmark execution, the harness also performs config validation, runtime estimation, and machine-state inspection.
It now also runs a system-requirements preflight that fails fast on unsupported hosts or missing required binaries.

## Benchmark Suites

The catalogue below covers every active suite and the shared energy measurements:

- [Synthetic LLM training](#1-llm-training)
- [Real-model causal-LM training](#1b-real-model-llm-training)
- [Offline Transformers inference](#2-llm-inference)
- [Offline vLLM inference](#2b-offline-vllm-inference)
- [Stable Diffusion](#3-stable-diffusion)
- [Blender rendering](#4-blender)
- [Pretrained vision inference](#5-pretrained-vision-inference)
- [GEMM, attention and memory diagnostics](#6-kernel-diagnostics-gemm-attention-and-memory)
- [Concurrent LLM serving](#7-concurrent-llm-serving-and-latency)
- [Board power and energy](#8-board-power-and-energy)

For complete comparison artifacts, enable the suite in `config.yaml` and use the
[recommended run procedure](#recommended-run-procedure). The standalone commands
below are for inspecting one combination: they do not perform the runner's
preflight, repeat annotation, metadata capture or report consolidation. Activate
`.venv` first. Commands with `CUDA_VISIBLE_DEVICES=0` select NVIDIA GPU 0; replace
that prefix with `HIP_VISIBLE_DEVICES=0` on AMD. Merge YAML examples into existing
sections rather than appending duplicate keys.

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
- Training scaling is controlled by `llm_train.world_sizes`. The standard config
  also adds the selected GPU count via `llm_train.include_all_gpus`.
- The root `gpu_include` list constrains which local GPU indices are visible to the run
- The root `gpu_backend` selects `nvidia`, `amd`, or `auto`

### 1b. Real-Model LLM Training

File: [benchmarks/llm_train_real.py](./benchmarks/llm_train_real.py)

What it does:
- Loads a real causal LM from Hugging Face
- Runs synthetic token batches through actual model weights
- Measures forward/backward/update throughput
- Keeps parameters and AdamW state in FP32; `fp16`/`bf16` select autocast
  compute precision. FP16 uses gradient scaling. Failed runs record the phase
  and step so warmup failures can be distinguished from measured-step failures.

Notes:
- Controlled by `llm_train_real.enabled` in `config.yaml`
- Disabled by default because it increases runtime and model-download requirements
- Currently single-GPU in the active orchestration flow
- Intended to run on either NVIDIA or AMD as long as the installed PyTorch build exposes a working GPU runtime

Example configuration:

```yaml
llm_train_real:
  enabled: true
  model: Qwen/Qwen2.5-0.5B
  revision: main
  dtype: fp16
  seq_len: 512
  batch_size: 2
  steps: 20
  warmup_steps: 2
```

Run with `bash run_all.sh --baseline --smoke`, then `bash run_all.sh --baseline`.
The runner resolves `main` once per run; replace it with the same resolved commit
on both hosts for a matched comparison.

This suite uses a fixed seeded batch of integer tokens, not a dataset or tokenizer.
The causal model shifts labels internally exactly once. Throughput counts
`batch_size * (seq_len - 1)` supervised tokens per step. Results include
`tokens_per_sec`, `steps_per_sec`, `final_loss`, `peak_memory_bytes`, optimizer
settings and model revision. Finite-loss and gradient checks are included in timing.
It measures real model computation, but does not measure dataset loading or
convergence on a learning task. Distributed execution is currently skipped.

### 2. LLM Inference

File: [benchmarks/llm_infer_hf.py](./benchmarks/llm_infer_hf.py)

What it does:
- Loads a model through Hugging Face Transformers on PyTorch
- Sweeps configured batch sizes with a canonical Transformers path used across GPU vendors
- Warms up, then runs repeated synchronous `generate()` calls for a fixed duration
- Records requests/sec, generated tokens/sec, batch latency stats, average power and tokens/J (the compatibility field is named `gen_tokens_per_watt`)

Notes:
- This is an offline throughput-style benchmark, not an interactive latency benchmark
- Prompt length is now tokenizer-verified; results include both requested and actual prompt token counts
- The canonical `transformers` backend only supports `tensor_parallel=1`; larger configured TP values are emitted as structured skipped rows
- `multi_gpu_mode: replicated` launches one worker per visible GPU and aggregates total requests/sec and tokens/sec into a single `llm_infer` row
- Latency fields are measured per `generate()` batch call; `batch_latency_per_item_proxy_*` is a simple batch-latency-per-item proxy, not a true online per-request latency measurement
- Board-energy sampling uses NVML on NVIDIA and selected-device ROCm telemetry on AMD; missing or incomplete telemetry is null
- Optional vLLM benchmark: [benchmarks/llm_infer_vllm.py](./benchmarks/llm_infer_vllm.py)
- Set `llm_infer.backend: vllm` if you want to run the vLLM-specific benchmark instead of the canonical Transformers path

### 2b. Offline vLLM Inference

File: [benchmarks/llm_infer_vllm.py](benchmarks/llm_infer_vllm.py)

This alternative `llm_infer` implementation runs blocking vLLM generation calls
with fixed output-token counts. It sweeps batch sizes and tensor-parallel sizes
and reports request/token throughput, batch latency and available board energy.
Its backend and timing identity keep it distinct from Transformers inference.

```yaml
llm_infer:
  enabled: true
  backend: vllm
  model: Qwen/Qwen3-8B
  dtype: float16
  prompt_len: 512
  output_len: 128
  batch_sizes: [1, 4, 16]
  tensor_parallel_sizes: [1]
```

For a single-card run, set `gpu_include: [0]`, `repeat: 5` and training
`world_sizes: [1]`, then run `bash run_all.sh --smoke` followed by `bash run_all.sh`.
Do not add `--baseline`, which overrides this backend to Transformers. This suite
requires a working vLLM install for the selected vendor. The standard matrix
configures its pinned install automatically. It measures offline batching, while the serving suite below measures
concurrent clients and streaming arrivals.

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
- Runs Blender in background mode against the configured scenes
- Measures the blocking Blender render operation separately from process and startup/loading time
- Compares one GPU vs all GPUs using the configured Blender backend (`CUDA` for NVIDIA, `HIP` for AMD)

Notes:
- The standard runner matches the selected physical GPUs to Cycles devices by PCI
  address. Each scene renders once on the first selected GPU and once on all
  selected GPUs per repeat. A missing Cycles device fails the all-GPU result.
- The standard config runs BMW27, Classroom, and the Agent 327 Barbershop Interior.
  The runner downloads and verifies the [Blender Open Data Barbershop archive](https://download.blender.org/release/BlenderBenchmark2.0/scenes/barbershop_interior.tar.bz2)
  on the first run that selects it, then reuses the files in
  `assets/blender/barbershop_interior/`. Download errors are recorded in
  `logs/blender_preflight.log`. Smoke mode uses the first configured scene. An
  empty or null `blender.scenes` value uses the two bundled scenes.
- Barbershop uses the archive's scene settings and frame 1. The single-card
  qualification target is an RTX 4090 with 24 GB VRAM.
- It writes structured rows directly to `metrics.jsonl` and also keeps a standalone JSON file in the run results directory
- Preflight verifies Blender 5.3.0 and checks every selected GPU with Cycles before the benchmark begins.

New rendering outputs include `render_time_s` (also stored as `time_s`),
`end_to_end_time_s` and `startup_load_time_s`. The render timer surrounds Blender's
blocking render operation, including internal setup/compilation. Scene SHA-256,
Blender version, render-settings hash and actual selected devices are recorded.
Baseline mode runs only the single-device case; general mode runs single and all.

### 5. Pretrained Vision Inference

File: [benchmarks/vision_infer.py](benchmarks/vision_infer.py)

Measures pretrained image-model inference on deterministic synthetic RGB images.
The images pass through the selected weights' preprocessing transforms. The script
records the weights-file hash, input hash, preprocessing hash and torchvision
version so different model/preprocessing configurations are not combined.

```yaml
vision_infer:
  enabled: true
  model: resnet18
  weights: ResNet18_Weights.IMAGENET1K_V1
  dtype: float32
  modes: [throughput, latency]
  batch_sizes: [1, 32]
  sizes: [224]
  iterations: 50
  warmup: 5
```

Throughput mode sweeps the configured batch sizes and image sizes. Latency mode
uses batch size one, independently of the throughput batch list. Precision accepts
`float32`, `float16` or `bfloat16`; reduced precision uses autocast. Other built-in
weight choices include `resnet50` with `ResNet50_Weights.IMAGENET1K_V2` and
`vit_b_16` with `ViT_B_16_Weights.IMAGENET1K_V1`. Use shapes supported by the model.

Standalone example:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/vision_infer.py \
  --model resnet18 --weights ResNet18_Weights.IMAGENET1K_V1 \
  --dtype float32 --mode throughput --batch-size 32 --size 224 \
  --warmup 5 --iterations 50 --metrics-path results/manual-vision/metrics.jsonl
```

Outputs: `images_per_sec`, mean/median/p95 batch latency, `memory_peak_bytes` and
available `images_per_joule`. Loading and warmup are excluded. It uses pretrained
weights and synthetic images: no dataset download is needed, but weights may
download on the first run. This is a performance benchmark, not an accuracy test.

### 6. Kernel Diagnostics: GEMM, Attention and Memory

File: [benchmarks/kernel_bench.py](benchmarks/kernel_bench.py)

These three cases help explain low-level performance differences independently
of full applications. Each checks finite output and a sampled CPU reference
before synchronized timing.

| Case | Operation | Work counted | Main rate |
| --- | --- | --- | --- |
| `gemm` | Two `size × size` matrices multiplied | `2 × size³` FLOPs per iteration | `tflops` |
| `attention` | Scaled dot-product attention | `4 × heads × size² × head_dim` FLOPs, excluding softmax | `tflops` |
| `memory` | Copy `size²` elements between device buffers | Read plus write bytes | `bandwidth_gbps` |

```yaml
kernel_bench:
  enabled: true
  cases: [gemm, attention, memory]
  size: 2048
  dtype: float32
  iterations: 20
  warmup: 3
  heads: 8
  head_dim: 64
```

Standalone examples for all three cases:

```bash
for benchmark_case in gemm attention memory; do
  CUDA_VISIBLE_DEVICES=0 python benchmarks/kernel_bench.py \
    --case "$benchmark_case" --size 2048 --dtype float32 \
    --heads 8 --head-dim 64 --warmup 3 --iterations 20 \
    --metrics-path "results/manual-kernels/${benchmark_case}.jsonl"
done
```

Outputs include the work convention, iteration-scaled `throughput`, mean/median/p95
latency and peak allocated memory. Precision accepts `float32`, `float16` or
`bfloat16`. Device-copy bandwidth uses reused buffers and may include cache effects;
it is not necessarily uncached memory bandwidth. These results appear as kernel
diagnostics and cannot become application recommendations in the report.

### 7. Concurrent LLM Serving and Latency

File: [benchmarks/llm_serve.py](benchmarks/llm_serve.py)

Measures concurrent clients against either a local Transformers model worker or
an already running streaming endpoint. Clients share a submission deadline;
in-flight requests drain afterwards, and drain time is included in throughput.
Model loading and warmup requests are excluded.

Local Transformers configuration:

```yaml
llm_serve:
  enabled: true
  provider: transformers
  endpoint: ""
  model: Qwen/Qwen3-8B
  dtype: float16
  prompt: A deterministic benchmark prompt.
  output_len: 32
  concurrency: [1, 4]
  duration: 30
  warmup: 1
  timeout: 120
```

No HTTP server is needed for this mode. One model worker serializes GPU access;
concurrent clients can queue, and the benchmark measures that queue delay. For
vLLM, set `provider: vllm` and a `/v1/completions` endpoint and follow the
[server setup procedure](#7-run-an-external-vllm-serving-experiment). Keep other
workloads disabled while benchmarking a server on the same card.

Standalone local example:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/llm_serve.py \
  --provider transformers --model Qwen/Qwen3-8B --dtype float16 \
  --prompt "A deterministic benchmark prompt." --output-len 32 \
  --concurrency 4 --duration 30 --warmup 1 --timeout 120 \
  --metrics-path results/manual-serving/metrics.jsonl
```

| Measurement | Local Transformers | HTTP endpoint |
| --- | --- | --- |
| Requests/s and end-to-end latency | Measured | Measured |
| Generated tokens/s | Counted from generated IDs | Server usage required; otherwise null |
| First-token timing | First generated ID, including queue delay | First nonempty content chunk |
| Inter-token latency | Measured from token callbacks | Null; chunk gaps are reported separately |
| Queue delay | Measured | Unknown/null |
| Board energy | Attempted on the local GPU | Unknown/null |

Latency summaries include mean/p50/p95/p99 where samples exist. Failed or truncated
streams produce failures, not invented token counts. External server model
revision, precision and GPU allocation cannot be verified from this client, so
those results remain distinct from a verified local single-GPU baseline.

### 8. Board Power and Energy

File: [benchmarks/energy.py](benchmarks/energy.py)

Energy is a shared measurement, not another workload. Sampling is attempted by
training, offline inference, SD, vision, kernels, local serving and Blender.
NVIDIA uses NVML with a selected-device `nvidia-smi` fallback. AMD reads explicit
watt fields from `rocm-smi`. Blender matches PCI identities to telemetry devices.

| Field | Meaning |
| --- | --- |
| `energy_j` | Timestamp-integrated board energy over the measurement window |
| `mean_power_w` | Integrated energy divided by measured duration |
| `power_coverage` | Fraction of the interval covered by valid sample pairs |
| `power_sample_count`, `power_device_ids` | Sampling evidence and selected devices |
| `tokens_per_joule`, `images_per_joule` | Work per joule, where the suite emits it |
| `gen_tokens_per_watt` | Historical inference field name for tokens per joule |

There is no separate command or enable flag: run the workload and inspect its raw
JSONL and the report's energy table. Missing/invalid samples or ambiguous devices
produce null energy. Replicated-worker energy is combined only for disjoint
devices with aligned timing windows. Reports require complete telemetry across
repeats and do not select an energy winner. These measurements cover GPU boards,
not wall-plug electricity for the whole system.

## Configuration

Default configuration lives in [config.yaml](./config.yaml). Select a complete
[sample](configs/README.md) or custom file with `--config PATH`; use `--backend` and
`--gpus` to override device selection without modifying that file.

Main sections:
- `gpu_backend`
- `preflight`
- `llm_train`
- `llm_train_real`
- `llm_infer`
- `sd_infer`
- `vision_infer`
- `kernel_bench`
- `llm_serve`
- `blender`

Important caveat:
- `llm_train_real` is optional and disabled by default because it adds significant runtime and depends on model availability.
- `llm_infer.backend` defaults to `transformers` so inference results are comparable across mixed GPU vendors. The reference model is `Qwen/Qwen3-8B`. The standard config also runs the [inference matrix](#standard-llm-inference-sweep).
- `llm_infer.multi_gpu_mode` supports `single` and `replicated`. `replicated` is the default and runs one worker per visible GPU for comparable multi-GPU aggregate throughput.
- Set the backend to `vllm` only when you intentionally want the vLLM-specific benchmark.
- The stand-alone requirements report treats missing Blender as an error when `blender.require_installed` or `preflight.blender_strict` is enabled.

## Outputs

Each `run_all.sh` execution creates a run folder like:

```text
/mnt/benchmark-scratch/results/<timestamp>_<host>_<gpu-tag>/
```

Typical contents:
- `meta.json`: machine snapshot plus captured software versions
- `effective_config.yaml`: the exact config used for that run, including smoke-mode overrides
- `model_revisions.json`: requested refs and resolved model commits for enabled local workloads
- `machine_state.json`: preflight machine-state warnings/checks
- `logs/*.log`: per-suite logs
- `results/metrics.jsonl`: unified structured metrics for all active suites
- `results/*.json`: suite-specific JSON outputs such as Blender repeat files
- `metrics.csv`: consolidated CSV copied to the run root
- `metrics_summary.csv`: repeat-level summary CSV with mean/median/stdev/min/max and measurement counts
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
  - benchmark view for baseline-aware metric diagnostics with explicit "better/worse than baseline" wording
  - decision confidence by suite
  - suite takeaways
  - risk flags
- single-GPU comparison view where the run data supports it
- run overview table
- comparability summary table
- grouped per-suite metric comparisons
- best-run annotations per metric
- tie-aware best-run reporting for near-equal results
- per-GPU throughput normalization when total GPU counts differ
- repeat variability using summary min/max/stdev/CV when available
- explicit "no decision-grade pick yet" messaging when a suite only has partial coverage

Reporting integrity:
- Repeat summaries aggregate request counts and latency measurements as well as throughput; latency percentile summaries describe variation between repeat-level percentiles, not pooled request percentiles.
- Stable Diffusion comparison keys accept both emitted `hw` / `per_gpu_batch_size` fields and older `width` / `height` / `per_gpu_batch` fields.
- Duplicate successful workload rows stop comparison instead of silently replacing a row. Older inference summaries must be regenerated from raw metrics with the current `harness.py`.
- These repairs do not turn unequal-GPU-count runs into isolated card comparisons or validate cross-vendor power efficiency.

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
- explicit failed comparable rows when a run produced non-`ok` summary rows

Single-GPU behavior:
- `llm_train` single-GPU rows are summarized when grouped rows use `gpu_count == 1`
- `llm_infer` and `sd_infer` appear in the single-GPU view only when their comparable rows actually use one GPU
- `blender` uses `mode=single` for the single-GPU view

## Environment Setup

[env_setup.sh](./env_setup.sh) creates a local `.venv` and installs a pinned stack intended to work together:
- PyTorch / torchvision / torchaudio
- vLLM
- xFormers
- Diffusers
- Transformers
- Accelerate
- pandas / PyYAML / tqdm / safetensors

The script assumes system-level GPU driver setup is already handled outside the repo.
It detects the responding vendor, or uses `GPU_BACKEND` when explicitly set. The
runner supplies the backend resolved from its selected config and CLI overrides.
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
- For Blender-enabled runs, an installation, version, or Cycles device check failure stops the run. See `logs/blender_preflight.log`.
- Preflight runs `ldd` on Blender and reports missing shared objects before attempting a render.

## Blender Install

The installer defaults to **Blender 5.3.0** on Linux x86-64. Use it to
install or upgrade ahead of a benchmark:

```bash
bash install_blender.sh
```

The installer verifies the [official release checksum](https://mirror.blender.org/release/Blender5.3/)
and checks the staged executable before switching the launcher in
`~/.local/bin/blender`. It installs to `~/.local/opt/blender-5.3.0`
alongside older versions. With `--ensure`, it reuses a working 5.3.0
executable already on PATH and returns its absolute path.

On minimal Ubuntu/Debian hosts, install the runtime libraries explicitly:

```bash
bash install_blender.sh --install-system-deps
```

The benchmark calls the installer with `--ensure` and validates the selected
CUDA or HIP GPUs in Cycles. On NVIDIA, Blender 5.3's CUDA backend requires
a driver from the 580 series or newer. A failed check stops the run before
workloads start. Use `blender.enabled: false` for profiles without Blender.

Post-run validation:
- `validate_run_artifacts.py` checks a completed run folder for missing artifacts, backend mismatches, missing suite rows, failed/skipped rows, and AMD power-metric caveats before you compare runs across vendors.

## Smoke-Test Checklist

The runner now writes `runtime_validation.json` before launching suites. It checks
the selected GPUs numerically and checks collectives when multiple GPUs are
visible. `--allow-unverified-host` accepts an experimental host combination; it
does not bypass these checks or qualify the host as vendor-supported.

Before moving from one GPU to eight, pass a smoke run on each intended GPU,
then test two GPUs before the full eight-GPU run. Omit `--baseline` for multi-GPU
runs and explicitly configure training world sizes and inference replication;
selecting eight visible GPUs alone does not make every suite distributed.
Real-model training and local Transformers serving remain single-GPU workloads.

Vision and kernel benchmarks use `min_duration_s: 5` by default in addition to
their minimum iteration count. Smoke mode disables that duration minimum.
Use the same duration and iteration settings on comparison hosts. Local
Transformers serving uses a FIFO serial queue; it measures that queue's behaviour,
not continuous batching as offered by a dedicated serving engine.

Treat energy as unavailable when device identity or sufficient power samples
cannot be verified. Do not compare the historical 11–12 W readings from
`20260913_newGen4-1x9700` as board-power efficiency. Rerun the baseline after these
measurement changes; protocol identifiers distinguish the new results.

Use this before trusting cross-vendor comparisons:

1. Preview selection with `bash run_all.sh --config configs/auto.yaml --baseline --dry-run`; use `--backend` or `--gpus` if needed.
2. Run `bash run_all.sh --config configs/auto.yaml --baseline --smoke` with the same overrides.
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
