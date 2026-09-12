# Repository Guidelines

## Project Structure & Module Organization
Root-level scripts drive orchestration and validation: `run_all.sh`, `harness.py`, `validate_config.py`, `estimate_runtime.py`, `check_*`, and `compare_runs.py`. Benchmark suite implementations live in `benchmarks/` (`llm_train.py`, `llm_infer_*.py`, `sd_infer.py`, Blender helpers). Tests live in `tests/` and mirror the main entry points (`test_compare_runs.py`, `test_harness.py`, etc.). Static inputs belong in `assets/`, planning notes in `docs/`, and generated benchmark outputs in `results/`; treat run folders as artifacts, not hand-edited source.

## Build, Test, and Development Commands
- `bash env_setup.sh` — create or refresh the local `.venv` with the pinned benchmark stack.
- `bash run_all.sh` — execute the full benchmark pipeline from `config.yaml`.
- `bash run_all.sh --smoke` — run a reduced-cost validation pass before longer GPU jobs.
- `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'` — run the unit test suite.
- `.venv/bin/python compare_runs.py --out-dir results/comparison_report results/<run_a> results/<run_b>` — build Markdown and JSON comparisons.
- `python3 validate_run_artifacts.py results/<run_id> --expected-backend amd` — verify a completed run folder.

## Coding Style & Naming Conventions
Use 4-space indentation in Python, `snake_case` for modules/functions, `PascalCase` for `unittest` classes, and `UPPER_CASE` for constants. Prefer small helper functions, stdlib-first solutions, and `pathlib.Path` for filesystem work. Bash scripts should start with `#!/usr/bin/env bash` and strict mode (`set -euo pipefail` or `set -Eeuo pipefail`). No formatter config is checked in, so keep diffs PEP 8-aligned and consistent with neighboring files.

## Testing Guidelines
This repo uses `unittest`. Name tests `test_<target>.py`, keep GPU-independent logic covered with temp directories and synthetic JSON/YAML fixtures, and update tests whenever orchestration, artifact validation, or comparison semantics change. For shell or pipeline changes, pair unit coverage with `bash run_all.sh --smoke` on a supported host.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects such as `Add ...`, `Fix ...`, and `Refine ...`. Follow the repo’s Lore commit protocol: lead with intent, add a short rationale, and include trailers like `Constraint:`, `Confidence:`, `Tested:`, and `Not-tested:`. Pull requests should summarize affected suites/backends, call out `config.yaml` or artifact-schema changes, link related issues, and include verification evidence such as test output, a smoke-run path, or a `comparison.md` excerpt when report formatting changes.
