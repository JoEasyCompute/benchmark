# Expanded Benchmark Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the GPU comparison harness with representative real-model, serving, vision, low-level kernel, rendering, and energy measurements while preserving controlled cross-vendor comparisons.

**Architecture:** Keep each workload as an independent benchmark script that emits the existing JSONL schema. Share deterministic configuration, revision, timing, synchronization, power, and identity helpers through `benchmarks/benchmark_protocol.py`. Extend validation and comparison keys only when a workload has complete identity metadata; incomplete or unsupported measurements remain explicitly directional or skipped.

**Tech Stack:** Python 3.10–3.12, PyTorch, Transformers, Diffusers, optional vLLM, torchvision, NumPy, existing Bash orchestration, unittest.

**Spec:** User-approved workload expansion: real causal-LM training, LLM serving/latency, vision inference, GEMM/attention/memory microbenchmarks, improved Blender timing, and cross-vendor energy accounting.

## Implementation status

Code for Tasks 1–7 and Task 8's preflight checks is implemented. The behavior and
rollout instructions are documented in `docs/expanded-benchmarks.md`. Hardware
smoke/full runs in Task 8 remain unverified: this session has no NVIDIA/AMD host.
Local validation includes loopback HTTP streams and dependency-free CLI-to-report
smoke tests. Old completion messages were superseded by this audit.

Plan corrections made during implementation:
- Preserve the existing `llm_train_real` suite name instead of renaming stored artifacts.
- AutoModelForCausalLM shifts labels internally; pass unshifted labels exactly once.
- Do not label HTTP chunk spacing as token spacing. Local token callbacks measure
  token-level latency; external SSE measures content-chunk arrivals.
- External server identity/energy cannot be attested by this client's local GPU.
- Blender's render-operator time includes internal setup and compilation; separate
  process/loading time rather than fabricating kernel-only timing.

## Global Constraints

- All workloads use the existing `TIMING_METHOD`, deterministic seed, repeat, identity, and artifact conventions.
- Default runs remain reproducible and do not silently download unpinned model revisions.
- Baseline runs use one physical GPU; multi-GPU results are reported separately.
- Missing power, revision, or timing data is `null`/skipped and never treated as zero or a valid win.
- No new dependency is added without an optional-install path and a CPU-only unit-test path.
- GPU execution is validated on supported Linux hosts; macOS unit tests must not require CUDA, ROCm, model downloads, Blender, or a serving daemon.

---

### Task 1: Shared workload identity and energy protocol

**Files:**
- Modify: `benchmarks/benchmark_protocol.py`
- Modify: `validate_run_artifacts.py`
- Modify: `compare_runs.py`
- Test: `tests/test_benchmark_protocol.py`, `tests/test_validate_run_artifacts.py`, `tests/test_compare_runs.py`

- [ ] Add a typed identity builder covering model revision, tokenizer revision, input hashes, seed, precision, scheduler, resolution, batch size, and timing method.
- [ ] Add an energy sampler interface with NVIDIA NVML and AMD ROCm implementations, returning `None` when unavailable and recording sampling interval, device IDs, and sample count.
- [ ] Add tests proving unavailable power cannot create efficiency metrics and incomplete identity lowers comparison quality.
- [ ] Add artifact validation checks for null power, mixed power availability, sampler coverage, and malformed identity fields.
- [ ] Run `unittest`, Python compilation, and artifact fixtures before proceeding.

### Task 2: Real causal-LM training

**Files:**
- Create: `benchmarks/llm_train_real.py` (replace the current optional prototype with the controlled implementation)
- Modify: `config.yaml`, `validate_config.py`, `run_all.sh`, `estimate_runtime.py`
- Test: `tests/test_llm_train_real.py`

- [ ] Pin model and tokenizer revisions in configuration and resolve the revision once in the parent process.
- [ ] Implement causal labels shifted by one token, fixed synthetic token generation, finite-loss checks, warmup, synchronized timing, and deterministic seed.
- [ ] Emit tokens/sec, steps/sec, loss, memory peak, model revision, tokenizer revision, parameter count, and identity metadata.
- [ ] Keep this workload single-GPU in the first implementation; emit a structured skip for unsupported distributed mode.
- [ ] Add a `real_model_training` suite key and report it independently from the synthetic microbenchmark.

### Task 3: LLM serving and latency

**Files:**
- Create: `benchmarks/llm_serve.py`
- Modify: `config.yaml`, `run_all.sh`, `validate_config.py`, `estimate_runtime.py`, `compare_runs.py`
- Test: `tests/test_llm_serve.py`

- [ ] Define a provider-neutral serving client interface with a local Transformers fallback and optional vLLM provider.
- [ ] Measure time-to-first-token, inter-token latency, end-to-end latency, completed requests/sec, generated tokens/sec, concurrency, queue time, and errors over a common duration.
- [ ] Use fixed prompts, output length, revision, seed, and concurrency schedule; synchronize request waves and exclude model startup from steady-state metrics.
- [ ] Emit structured skipped rows when the provider is unavailable rather than falling back silently.
- [ ] Add comparison groups that separate offline generation from serving results.

### Task 4: Vision inference

**Files:**
- Create: `benchmarks/vision_infer.py`
- Modify: `config.yaml`, `run_all.sh`, `validate_config.py`, `estimate_runtime.py`, `compare_runs.py`
- Test: `tests/test_vision_infer.py`

- [ ] Use a pinned torchvision model and a generated deterministic image batch so no dataset download is required.
- [ ] Implement warmup, synchronized batch throughput, latency mode, fixed preprocessing, precision selection, and optional autocast.
- [ ] Emit images/sec, batch latency percentiles, memory peak, model revision/hash, input shape, preprocessing hash, and timing identity.
- [ ] Add comparison keys for model, input shape, batch size, precision, preprocessing, and mode.

### Task 5: GEMM, attention, and memory microbenchmarks

**Files:**
- Create: `benchmarks/kernel_bench.py`
- Modify: `config.yaml`, `run_all.sh`, `validate_config.py`, `estimate_runtime.py`, `compare_runs.py`
- Test: `tests/test_kernel_bench.py`

- [ ] Implement separate GEMM, scaled-dot-product attention, and streaming memory-bandwidth cases with explicit dimensions and dtype.
- [ ] Use device synchronization, warmup, fixed iteration counts, correctness checks against CPU/reference results, and outlier-resistant summaries.
- [ ] Emit TFLOP/s, effective bandwidth, latency, occupancy-independent metadata, and memory allocation details.
- [ ] Keep kernel results as diagnostic microbenchmarks; do not combine them into an application winner.

### Task 6: Blender render-only timing

**Files:**
- Modify: `benchmarks/blender_bench_cuda.sh`
- Modify: `config.yaml`, `validate_run_artifacts.py`, `compare_runs.py`
- Test: `tests/test_blender_bench.py`

- [ ] Record scene SHA-256, Blender version, render engine, samples, resolution, denoiser, device mode, and backend.
- [ ] Separate process startup/scene-load time from render time using Blender-side timestamps and retain end-to-end time separately.
- [ ] Synchronize completion before stopping the render timer and emit structured rows for missing scenes or unsupported backends.
- [ ] Add scene identity and render-setting fields to comparison keys.

### Task 7: Controlled orchestration and reporting

**Files:**
- Modify: `run_all.sh`, `harness.py`, `compare_runs.py`, `validate_run_artifacts.py`
- Modify: `README.md`, `docs/run-comparison-report-plan.md`
- Test: `tests/test_harness.py`, `tests/test_compare_runs.py`, `tests/test_validate_run_artifacts.py`

- [ ] Add explicit suite ordering and per-suite enable/disable configuration without changing the existing default workloads.
- [ ] Ensure each suite receives the selected baseline GPU, effective config path, repeat index, and common output path.
- [ ] Add report sections distinguishing application benchmarks, serving/latency, kernel diagnostics, and energy results.
- [ ] Reject mixed identity rows, failed winners, null-efficiency winners, and unequal-GPU comparisons marked as strict.
- [ ] Add a synthetic two-vendor report fixture covering successful, failed, skipped, legacy, and unavailable-power rows.

### Task 8: Host validation and rollout

**Files:**
- Modify: `check_system_requirements.py`, `README.md`
- Test: `tests/test_check_system_requirements.py`

- [ ] Add optional checks for ROCm power tooling, NVML, Blender render support, serving-provider availability, and required kernel APIs.
- [ ] Run the full CPU-only test suite and shell/static checks.
- [ ] On one NVIDIA and one AMD host, run `--baseline --smoke`, then full baseline profiles for each suite.
- [ ] Validate every run folder with `validate_run_artifacts.py` and compare only matched baseline profiles.
- [ ] Record known host limitations and exclude unsupported suites explicitly from decision summaries.

## Review gates

- After Task 1, review the identity and energy schema before adding new workloads.
- After Tasks 2–5, review each workload’s reproducibility and whether its metric answers a distinct question.
- After Task 6, review Blender timing against a known scene before integrating it into winner summaries.
- After Task 7, review a generated two-vendor report and confirm no incomplete or unavailable metric is selected as a winner.

## Stop conditions

The expansion is ready for hardware validation when all unit tests pass, every suite has deterministic CPU-only validation, shell syntax and compilation pass, and a smoke run emits schema-valid rows or explicit structured skips for every configured suite.
