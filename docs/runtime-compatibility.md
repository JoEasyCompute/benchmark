# Runtime compatibility policy

The benchmark does not choose PyTorch from driver version alone. It inspects the
responding GPU inventory, architecture, driver/runtime, operating system, kernel,
Python ABI, glibc and existing package builds. A compatibility profile contains
the exact Torch/vision/audio/Triton builds, package sources, minimum driver/runtime
and evidence links.

Current curated profiles:

| Profile | Intended host | Core Torch packages | Status |
| --- | --- | --- | --- |
| `torch291-cu128` | NVIDIA Blackwell or modern NVIDIA with driver ≥570.26 | PyTorch 2.9.1 CUDA 12.8, torchvision 0.24.1, torchaudio 2.9.1 | Curated from PyTorch’s 2.9.1 matrix |
| `torch291-cu126` | NVIDIA Ada/older supported NVIDIA with driver ≥560.28 | PyTorch 2.9.1 CUDA 12.6, torchvision 0.24.1, torchaudio 2.9.1 | Curated from PyTorch’s 2.9.1 matrix |
| `torch291-rocm72` | AMD Radeon/Radeon AI Pro on ROCm 7.2, Ubuntu 22.04/24.04 and matching Python ABI | AMD’s versioned ROCm 7.2.0 Torch 2.9.1, torchvision 0.24.0, torchaudio 2.9.0, Triton 3.5.1 wheels | Curated from AMD’s published Radeon wheels |

The resolver selects `cu128` for Blackwell architectures such as `sm_120` when
the driver meets its conservative floor, and `cu126` for Ada `sm_89` when a
12.8 driver is unavailable. NVIDIA documents CUDA 12.x minor-version compatibility
from driver 525 upward, but the resolver uses stricter profile floors to keep
framework support explicit.

For AMD, the R9700/R9700S `gfx1201` architecture is covered by the ROCm 7.2 matrix.
The inspected server `ezc-amdtest-7v13-47` reports Ubuntu 24.04.3, kernel
6.8.0-137-generic, Python 3.12.3, ROCm 7.2.0, AMDGPU module 7.1.3.31500000,
driver component 6.16.13, and eight Radeon AI Pro R9700S GPUs. It is a candidate
for the `torch291-rocm72` profile, but its kernel differs from the published
Ubuntu 24.04 ROCm 7.2 combination (kernel 6.14 in AMD’s matrix). The resolver
therefore blocks by default; `--allow-unverified-host` produces an experimental
plan with `host_qualified: false` and does not alter the host.

## Procedure

1. Run `inspect_runtime.py` on the target host. Use `--backend amd` or `--backend nvidia` if both vendors are present.
2. Review `host_inventory.json`, especially GPU architecture, driver, runtime, OS/kernel and Python version.
3. Run `runtime_resolver.py --host-json ...`. Fix an incompatibility or explicitly accept an experimental host with `--allow-unverified-host`.
4. Install into a new venv with `--install --venv <new-path>`; no existing venv is overwritten by default.
5. Let `verify_runtime.py` complete. It must pass package versions, backend/runtime, per-card numerical checks and any requested collective check.
6. Run `bash run_all.sh --config configs/auto.yaml --baseline --smoke` with the validated environment.
7. Keep `runtime_plan.json`, `runtime-lock.json`, `runtime_validation.json`, `host_inventory.json`, logs and raw metrics with the run.

The resolver has no authority to install kernel modules, alter drivers, reboot the
host or select a vendor-specific vLLM image. vLLM and xFormers remain separate
provider/runtime decisions; qualify them after the base PyTorch environment passes.

References: [PyTorch 2.9.1 package matrix](https://pytorch.org/get-started/previous-versions/#v291), [AMD ROCm 7.2 PyTorch installation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/native_linux/install-pytorch.html), [AMD ROCm 7.2 compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html), and [NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
