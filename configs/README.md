# Sample configurations

These are complete configuration files; no copy/merge into the root config is
needed. Run commands from the repository root.

| File | Selection | Workloads |
| --- | --- | --- |
| [auto.yaml](auto.yaml) | Detect one responding vendor automatically | All suite families enabled |
| [nvidia.yaml](nvidia.yaml) | Require NVIDIA | Same workloads as auto/AMD |
| [amd.yaml](amd.yaml) | Require AMD | Same workloads as auto/NVIDIA |

The three files differ only in `gpu_backend`. They use the same model, precision,
batch and shape settings so the examples do not silently benchmark different work
on different vendors. They use modest inference batches and a single SD size;
they are starting points, not per-card tuning or a guarantee of fitting every GPU.
The launcher can skip Blender when it is not installed, but validation will flag
that missing enabled suite. Install it and supply scenes for complete rendering
coverage, or explicitly disable the suite when omitting rendering. Models and
pretrained vision weights may download on first use.

```bash
# Preview device selection and resolved options without installing the GPU stack.
bash run_all.sh --config configs/auto.yaml --baseline --dry-run

# Then validate and run every enabled workload on one detected card.
bash run_all.sh --config configs/auto.yaml --baseline --smoke
bash run_all.sh --config configs/auto.yaml --baseline
```

The dry run requires Python with PyYAML; it does not bootstrap dependencies. Normal
runs bootstrap configuration support first when needed, detect the responding GPU
vendor, then install/repair the appropriate CUDA/ROCm stack.
For the pinned PyTorch 2.8 stack, the installer uses CUDA 12.8 or ROCm 6.4 wheel
indexes, following the [published PyTorch combinations](https://pytorch.org/get-started/previous-versions/#v280).
The host still needs a compatible driver/runtime for its GPU model.

Use the vendor-specific examples when you want an explicit requirement:

```bash
bash run_all.sh --config configs/nvidia.yaml --baseline --smoke
bash run_all.sh --config configs/amd.yaml --baseline --smoke
```

On a host with both vendors, choose one per run. Select physical card indices with
`--gpus`, without editing any YAML:

```bash
bash run_all.sh --config configs/auto.yaml --backend amd --gpus 1 --baseline
bash run_all.sh --config configs/auto.yaml --backend nvidia --gpus 0 --baseline
```

Backend precedence is `--backend` → config `gpu_backend` → device detection for
`auto`. Device precedence is `--gpus` → nonempty config `gpu_include` → inherited
CUDA/HIP visibility mask → detected devices. Requested IDs must stay within an
inherited visibility mask. Baseline mode takes the first selected device. When
both vendors respond, auto mode fails with an override instruction rather than
guessing. A missing/broken driver also produces an explicit error.
NVIDIA runtime visibility uses UUIDs translated from the selected physical indices;
the config and metadata retain those physical indices for auditability.

Automatic options are the GPU backend, selected devices, CUDA/HIP visibility,
Blender CUDA/HIP backend and environment installation path. Models, precision,
batches, serving provider and repeat settings are kept as configured, except for
the documented `--baseline` and `--smoke` transformations. External serving
processes still need their own device selection.

The runner writes the resolved choices into each run's `effective_config.yaml`.
It never rewrites these samples or the root `config.yaml`. Pin the same resolved
model commits on both hosts as described in the [main procedure](../README.md#4-match-revisions-and-workloads-across-hosts)
before interpreting a cross-vendor comparison.
