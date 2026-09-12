# Expanded benchmark suite

The expansion is implemented behind optional configuration sections. Existing
synthetic training, offline inference, SD and Blender remain enabled by default.
Enable `llm_train_real`, `vision_infer`, `kernel_bench` and `llm_serve` individually
with `enabled: true` in your selected config. The complete samples under `configs/`
enable all families; select one with `--config` without editing the root config.
Every configured combination runs for the
effective repeat count. A crash without JSON output becomes a failed metric row.

## Running a controlled comparison

On each supported Linux GPU host:

```bash
bash run_all.sh --config configs/auto.yaml --baseline --smoke
bash run_all.sh --config configs/auto.yaml --baseline
python3 validate_run_artifacts.py results/<run-id> --expected-backend nvidia
# Use --expected-backend amd on the AMD host.
```

Auto detection selects NVIDIA/CUDA or AMD/ROCm and the matching Blender backend.
Use `--backend nvidia|amd` on mixed-vendor hosts, and select a card with
`--gpus physical_index` (or config `gpu_include`). Baseline mode uses the first
configured card, five repeats, single-GPU workloads and a single Blender mode.
Smoke mode reduces all enabled new suites, including iterations and serving
duration, and uses one repeat. The current host's GPU metadata still matters:
different CPUs, power limits, interconnects and software builds are not eliminated
by selecting one card.

`model_revisions.json` records requested and resolved model refs. Remote model refs
are resolved once before repeats, and the resolved commit is written into the
run's `effective_config.yaml`. Copy those commit values to the other host's config
to reproduce exactly the same revisions. Local model directories have no verified
commit identity. External servers are not attested by a local model download.

## Real-model training

The existing `llm_train_real` suite name is retained for artifact compatibility.
It trains a real pretrained causal LM on a fixed seeded synthetic token batch
resident on the GPU. Hugging Face shifts labels internally: the benchmark passes
an unshifted clone and counts `batch_size * (seq_len - 1)` supervised tokens per
step. It checks finite loss and gradients before optimizer updates, including in
the measured loop; the timings include these checks. Warmup is outside timing.
It reports the objective, optimizer settings, resolved model revision, final loss,
memory peak and throughput. No tokenizer is used for synthetic integer tokens;
`tokenizer_used: false` and `tokenizer_revision: null` state that explicitly.
Distributed real-model training is a structured skip in this version.

## Serving and latency

`provider: transformers` with an empty endpoint loads a local model. A serialized
model worker services concurrent clients; queue delay is measured separately.
Generated token-ID callbacks provide token counts, first-token arrival and
inter-token latency. This is a simple reference serving implementation, not a
continuous-batching server.

For vLLM, start and configure the server separately, then set:

```yaml
llm_serve:
  enabled: true
  provider: vllm
  endpoint: http://127.0.0.1:8000/v1/completions
  model: your-served-model-name
  concurrency: [1, 4]
  duration: 30
  output_len: 32
  warmup: 1
  timeout: 120
```

The client uses SSE streaming with usage reporting and vLLM's `min_tokens` and
`ignore_eos` controls. It verifies fixed output length when token usage is present.
It never invents usage: missing token counts yield null token throughput. HTTP
TTFT means first nonempty content-chunk arrival. Chunk gaps are named chunk gaps;
HTTP token-level latency and server queue time stay null. Local and HTTP timing
protocols are separate comparison identities. External server precision, GPU
count and revisions are unverified, so its results cannot receive strict identity
confidence. External energy is null; the client does not measure its own GPU and
attribute that energy to a remote server.

Concurrent clients start together and submit until a common deadline. In-flight
requests drain afterwards; all work and drain time enter the throughput
denominator. Warmup requests and model loading are excluded. Malformed, truncated,
error or inconsistent streams produce failed rows and a nonzero exit.

## Vision and kernels

Vision defaults to `ResNet18_Weights.IMAGENET1K_V1`, not random weights or a mutable
`DEFAULT` alias. The pretrained weights enum, URL, cached file SHA-256,
torchvision version, transformed-input hash and preprocessing hash are recorded.
Inputs are deterministic synthetic uint8 RGB tensors, transformed using the
weights' prescribed normalization and the selected crop size. Precision is
explicit; timed work runs under inference mode. Throughput mode sweeps configured
batches and resolutions. Latency mode always uses batch size one. Each run reports
mean, median and p95 batch latency, peak allocated memory and image throughput.

Kernel diagnostics include GEMM, scaled-dot-product attention and device copy.
They check finite output and compare sampled output against a float64 CPU
reference before timing. GEMM counts `2*n^3` FLOPs, attention counts both matrix
products (`4*heads*n*n*head_dim`, excluding softmax), and copy counts read plus
write bytes. Rates count every measured iteration. Mean/median/p95 latency and
the work convention are emitted. Device-copy bandwidth is effective bandwidth
over reused buffers and can include cache effects; it is not a claim of uncached
HBM bandwidth. Kernel results appear in diagnostic tables, not application picks.

## Blender

`blender_bench_cuda.sh` delegates to `blender_render.py`. Blender executes a blocking
`bpy.ops.render.render(write_still=False)` call. `time_s` and `render_time_s` cover
that call, including scene setup/compilation done internally by the render
operator; they are not pure GPU-kernel time. `end_to_end_time_s` covers the process,
and `startup_load_time_s` records time until the Python script begins. Scene bytes,
Blender version, render settings hash, samples, resolution, denoiser, seed and
actual enabled devices are recorded. Missing devices fail instead of using CPU.
No artificial cold/warm/compile measurements are emitted.

## Energy

`energy.py` maps selected logical devices to telemetry identifiers and integrates
timestamped board-power samples using the trapezoidal rule. NVIDIA uses NVML with
a selected-device `nvidia-smi` fallback for embedded Python environments;
AMD uses explicit watt fields from `rocm-smi --showpower --json`. Power caps and
unselected GPUs are excluded. Sampling has bounded subprocess timeouts and
cleanup. Missing, invalid, out-of-order or excessively separated samples produce
null energy with coverage information. Workload boundary times, sample count,
device IDs, interval and method are recorded.

The shared sampler is used by synthetic and real-model training, HF/vLLM offline
inference, SD, vision, kernels, local serving and Blender. Blender maps Cycles PCI
IDs to NVML/ROCm identities; opaque or ambiguous device IDs make telemetry null.
Replicated HF/SD workers report combined energy only when device sets are disjoint
and their sampling windows align. Different worker end times intentionally yield
null aggregate energy. This is board energy, not whole-system electricity.

The energy report shows measurements and efficiency only for complete telemetry
across all repeats. It does not infer an energy winner. Historical
`gen_tokens_per_watt` is retained as a compatibility name for tokens per joule.

## Tests and rollout status

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
bash -n run_all.sh benchmarks/blender_bench_cuda.sh
```

The test suite includes local streaming HTTP fixtures, dependency-free CLI-to-report
smoke tests, numerical energy/formula fixtures, training objective checks, Blender
process stubs, configuration sweeps and comparison regressions. The streaming
tests need permission to bind a loopback socket. CPU-only tests do not replace
CUDA/ROCm, pretrained-model or Blender runtime validation. No NVIDIA/AMD hardware
run has been completed in this macOS session.

Upstream contracts checked during implementation:

- [Hugging Face causal language modeling](https://huggingface.co/docs/transformers/main/tasks/language_modeling)
- [vLLM completion protocol](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/openai/completion/protocol/)
- [ROCm SMI CLI source](https://github.com/ROCm/rocm_smi_lib/blob/rocm-6.3.x/python_smi_tools/rocm_smi.py)
