# Expanded suite verification — 2026-09-11

Software implementation and integration are complete for the approved expanded
suite. This record supersedes earlier completion messages that relied on the
original 35 tests without exercising the new workloads.

| Area | Implemented behavior | Evidence |
| --- | --- | --- |
| Training | Exactly one causal-label shift, supervised-token accounting, finite loss/gradient checks, fixed resident inputs | `test_llm_train_real.py` |
| Serving | Local token callbacks and queue timing; HTTP SSE first-content/chunk timing, actual usage, concurrent deadline/drain, vLLM controls | `test_llm_serve.py` with real loopback HTTP server |
| Vision | Versioned pretrained weights, weight/input/preprocessing identities, inference mode, size/batch/mode sweeps | `test_vision_infer.py`, `test_suite_config.py`, CLI smoke |
| Kernels | Correct iteration-scaled work, attention matrix-product FLOPs, copy bandwidth, numerical reference checks, latency summaries | `test_kernel_bench.py`; GPU reference execution pending |
| Blender | Render-operator time distinct from process/startup, actual settings and device identity, PCI-based telemetry mapping | `test_blender_bench.py` with process stubs |
| Energy | Selected-device NVML/CLI probes, trapezoidal integration, window/coverage/null safeguards, cleanup | `test_energy.py` and inference tests |
| Integration | All repetitions and combinations, model locks, smoke limits, optional checks, structured failures | `test_optional_runner.py`, `test_model_locks.py`, `test_check_system_requirements.py`, `test_expanded_smoke.py` |
| Reporting | Repeat summaries, identity checks, failed-suite visibility, scoped single-GPU picks, diagnostic categories, complete-only energy table | `test_expanded_reporting.py`, comparison/artifact tests |

Final local checks:

- `unittest discover`: **98 tests passed** (including loopback-server fixtures).
- Python compilation: passed.
- Bash syntax for orchestration and Blender wrapper: passed.
- Configuration validation: passed.
- `git diff --check`: passed.

The host has no installed torch/torchvision runtime and no NVIDIA/AMD GPU. The
dependency-free CLI smoke test verifies valid skipped output through the report
pipeline; it does not produce hardware performance measurements. No model-weight
download, CUDA/ROCm numerical run, physical power probe or actual Blender render
has been validated here. Run the steps in `expanded-benchmarks.md` on both GPU
hosts before treating performance or energy numbers as qualified results.
