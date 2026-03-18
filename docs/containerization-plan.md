# Containerization Plan

Status: exploratory
Owner: unassigned
Last updated: 2026-03-18

## Goal

Investigate how to package this benchmark repo into container images so benchmark runs require less manual environment setup, while preserving GPU access and cross-vendor comparability.

This is a planning document only. It does not assume containerization is already the default or recommended execution path.

## Why This Needs Investigation

The repo is straightforward to package at the Python/application layer, but benchmark execution depends on host GPU drivers, vendor runtime integration, device visibility, and Blender behavior.

Important repo-specific constraints:

- Linux is required.
- NVIDIA support depends on `nvidia-smi` and NVIDIA container runtime integration.
- AMD support depends on `rocm-smi`, ROCm compatibility, and correct device exposure.
- Blender is more operationally fragile than the Python benchmark stack.
- `vllm` should remain optional, especially on AMD.
- The canonical inference path should stay `transformers` unless there is a strong reason to change it.

## Desired Outcome

Produce a container strategy that:

- reduces or eliminates per-host Python/venv setup
- keeps benchmark behavior reproducible
- works on real GPU hosts
- preserves current benchmark outputs and workflow
- does not overpromise full host independence

## Non-Goals

- Do not assume one image will work reliably for both NVIDIA and AMD.
- Do not assume host GPU drivers can be eliminated.
- Do not make `vllm` part of the mandatory default container path.
- Do not require Blender containerization in the first milestone if it slows down adoption.

## Proposed Direction

Investigate a split-image approach:

- `Dockerfile.nvidia` for CUDA/NVIDIA hosts
- `Dockerfile.amd` for ROCm/AMD hosts
- optionally a shared base image or shared build logic

Default benchmark path inside containers should remain:

- `llm_infer.backend: transformers`
- model/backend behavior aligned with current repo defaults

Blender should be treated as:

- optional in the first container milestone
- either baked into image later, or mounted/provided from host if that proves more robust

## Investigation Checklist

### 1. Inventory current host requirements

- [ ] Enumerate all required host binaries and runtime assumptions from:
  - `run_all.sh`
  - `check_system_requirements.py`
  - benchmark scripts
- [ ] Separate requirements into:
  - must stay on host
  - can move into image
  - optional / benchmark-specific
- [ ] Verify whether `/usr/bin/time`, `stdbuf`, `tee`, `hostname`, `nvidia-smi`, and `rocm-smi` are expected inside the container, on the host, or both

### 2. Define target execution model

- [ ] Decide whether the container should:
  - run `run_all.sh` as the main entrypoint
  - expose a shell for interactive benchmarking
  - support both patterns
- [ ] Decide how results will be persisted:
  - bind mount repo working tree
  - bind mount only `results/`
  - named volume
- [ ] Decide how config overrides should be passed:
  - mounted `config.yaml`
  - env vars
  - alternative config path

### 3. NVIDIA container path

- [ ] Choose a CUDA-compatible base image
- [ ] Reproduce current `env_setup.sh` package set in image build form
- [ ] Validate `torch`, `transformers`, `diffusers`, and optional `vllm`
- [ ] Confirm GPU visibility using `CUDA_VISIBLE_DEVICES`
- [ ] Confirm `nvidia-smi` is visible inside container where needed
- [ ] Run a smoke benchmark in-container on an NVIDIA host
- [ ] Compare outputs with a native-host run

### 4. AMD container path

- [ ] Choose a ROCm-compatible base image
- [ ] Reproduce current AMD package set in image build form
- [ ] Confirm `torch` and canonical HF inference work in-container
- [ ] Confirm GPU visibility using `HIP_VISIBLE_DEVICES`
- [ ] Confirm `rocm-smi` visibility and behavior inside container
- [ ] Validate Stable Diffusion on AMD container path
- [ ] Treat `vllm` as optional and likely disabled by default
- [ ] Run a smoke benchmark in-container on a real AMD host
- [ ] Compare outputs with the known-good native AMD path

### 5. Blender strategy

- [ ] Decide whether Blender belongs in the initial container milestone
- [ ] Test whether headless Blender GPU rendering works reliably inside vendor-specific containers
- [ ] Identify extra runtime libraries needed for Blender in-container
- [ ] Decide whether Blender should be:
  - installed in the image
  - mounted from host
  - skipped by default in container mode
- [ ] Ensure benchmark behavior remains consistent with current `blender_bench_cuda.sh`

### 6. Repo changes needed for container mode

- [ ] Determine whether `run_all.sh` needs explicit container-awareness
- [ ] Determine whether `check_system_requirements.py` needs different logic in containerized runs
- [ ] Decide whether to add a `container_mode` flag or auto-detection
- [ ] Decide whether host-tool detection should be relaxed or redirected when running inside a container
- [ ] Avoid making the native-host workflow worse

### 7. Image build and run UX

- [ ] Define image naming and tagging conventions
- [ ] Add reproducible build commands
- [ ] Add vendor-specific run commands
- [ ] Decide whether to use:
  - plain `docker run`
  - `docker compose`
  - helper shell scripts
- [ ] Document required host setup clearly for each vendor

### 8. Validation and parity checks

- [ ] Run `--smoke` in each supported container path
- [ ] Verify `metrics.jsonl`, `metrics.csv`, `metrics_summary.csv`, and `meta.json` are still produced
- [ ] Run `validate_run_artifacts.py` on container-produced runs
- [ ] Compare key throughput fields versus native runs
- [ ] Confirm repeat handling, logs, and failure semantics remain unchanged

## Suggested Delivery Phases

### Phase 1: Minimal container support

- NVIDIA-first
- no Blender requirement
- canonical `transformers` inference only
- bind-mounted results
- smoke-tested end to end

### Phase 2: AMD container support

- ROCm-specific image
- canonical HF path validated
- Stable Diffusion validated
- `vllm` still optional

### Phase 3: Blender support

- either in-container Blender or a documented host-assisted approach
- verified on both vendors where practical

### Phase 4: Polish

- final docs
- helper scripts
- reduced duplication between native setup and image build logic

## Open Questions

- Should container runs share the repo working tree, or should the image contain a copy of the repo plus only mounted outputs?
- Do we want benchmark artifacts to record that the run occurred inside a container?
- Should Blender be excluded from container mode by default until explicitly enabled?
- Is parity with native-host power telemetry possible on both vendors from inside containers?
- Should we generate images from `env_setup.sh`, or maintain separate Docker-specific dependency manifests?

## Exit Criteria For Calling This "Feasible"

Containerization should only be considered practically feasible when all of the following are true:

- at least one NVIDIA container path completes `bash run_all.sh --smoke`
- at least one AMD container path completes `bash run_all.sh --smoke`
- canonical HF inference works in both paths
- artifact generation matches current repo expectations
- operational host prerequisites are documented clearly
- the container path does not create more maintenance burden than the native setup it replaces
