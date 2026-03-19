# Run Comparison And Reporting Plan

Status: in progress
Owner: unassigned
Last updated: 2026-03-19

## Current Status

Implemented in repo today:

- `compare_runs.py` exists and is the active multi-run comparison tool
- compares completed run folders using:
  - `meta.json`
  - `effective_config.yaml`
  - `metrics_summary.json`
- emits:
  - `comparison.json`
  - `comparison.md`
- supports:
  - positional run paths
  - `--label NAME=PATH`
  - `--baseline <label|run_name|path>`
  - `--out-dir <dir>`
  - `--json-out`
  - `--md-out`
  - `--suites <comma-separated>`
- groups rows by suite-specific keys
- labels groups as `strict`, `directional`, or `partial`
- adds comparability notes for:
  - differing `gpu_count`
  - differing `gpu_backend`
  - differing Blender render backends
  - differing `torch` versions
  - differing `transformers` versions
  - differing repeat counts
  - partial suite/group coverage
- reports:
  - executive summary
  - executive-summary decision confidence
  - executive-summary suite takeaways
  - executive-summary risk flags
  - run overview
  - comparability summary
  - per-suite metric tables
  - baseline-relative deltas with explicit better/worse wording in summary sections
  - repeat variability from summary rows
  - tie-aware best-run annotations
  - failed comparable-row visibility in notes/summary when a run has non-`ok` status

Still not implemented:

- raw-row comparison mode using `metrics.jsonl`
- `--strict-comparability`
- optional `comparison.csv`
- charts / HTML / dashboard output
- broader real-run validation beyond the current synthetic regression coverage

Recent refinement:

- benchmark-view summary text now says `better than baseline by X%` / `worse than baseline by X%`
  instead of relying on signed percentages alone

## Goal

Add a repo feature that compares benchmark results across multiple run folders and produces a proper analysis report, rather than only per-run raw outputs.

The intended use case is to answer questions such as:

- how does GPU A compare to GPU B on the same benchmark configuration
- how does one host compare to another across the same suite
- how much scaling benefit came from using more GPUs
- which benchmark dimensions changed enough to invalidate a fair comparison

## Why This Is Valuable

The repo already does a good job of producing structured artifacts for a single run:

- `meta.json`
- `effective_config.yaml`
- `results/metrics.jsonl`
- `metrics.csv`
- `metrics_summary.csv`
- `metrics_summary.json`

What is missing is a first-class way to compare runs against each other and explain the result quality.

Without this feature, users must manually inspect multiple output folders and reason about comparability themselves.

## Desired Outcome

Produce a comparison tool that:

- accepts 2 or more run folders
- extracts key benchmark and machine metadata
- determines which rows are comparable
- computes percent deltas and rankings
- emits a readable report for humans
- emits structured comparison output for downstream processing

## Non-Goals

- Do not mix multi-run comparison logic into `harness.py` unless there is a strong reason.
- Do not silently compare non-equivalent runs as if they were fully fair.
- Do not make HTML or dashboard work the first milestone unless Markdown output proves insufficient.
- Do not assume every suite has the same comparison semantics.

## Proposed Direction

Add a dedicated comparison/reporting tool, likely:

- `compare_runs.py`

That tool should read existing run artifacts and produce:

- a machine-readable comparison file such as JSON
- a human-readable report such as Markdown

The tool should treat comparability as a first-class concept, not an afterthought.

## Key Principles

### 1. Compare like with like

Rows should only be compared directly when the benchmark setup is materially equivalent.

Important dimensions include:

- suite name
- benchmark backend
- model
- dtype
- batch size or per-GPU batch
- prompt length / output length
- world size
- visible GPU count
- multi-GPU mode
- scene name for Blender
- image size and step count for Stable Diffusion

### 2. Separate fair comparisons from directional comparisons

The report should distinguish:

- directly comparable rows
- partially comparable rows with warnings
- non-comparable rows that should not be ranked together

### 3. Explain the result, not just the number

The output should include:

- winner / loser
- percent delta
- repeat variability where available
- notable config mismatches
- benchmark/backend caveats

## Candidate Inputs

Primary run artifacts to inspect:

- `meta.json`
- `effective_config.yaml`
- `results/metrics.jsonl`
- `metrics.csv`
- `metrics_summary.csv`
- `metrics_summary.json`

Preferred comparison source should likely be:

- `metrics_summary.json` for repeat-aggregated comparisons
- `metrics.jsonl` when row-level detail is needed

## Candidate Outputs

### Machine-readable

- [x] `comparison.json`
- [ ] optional `comparison.csv`

### Human-readable

- [x] `comparison.md`

Potential later outputs:

- [ ] HTML report
- [ ] charts
- [ ] dashboard ingestion format

## Suggested CLI Shape

Example possibilities:

```bash
python compare_runs.py results/run_a results/run_b
python compare_runs.py results/run_a results/run_b results/run_c --out report/
python compare_runs.py --label 4090=results/run_a --label R9700=results/run_b
```

Possible flags to investigate:

- [ ] `--out`
- [ ] `--format markdown|json|both`
- [ ] `--use-summary`
- [ ] `--strict-comparability`
- [x] `--label NAME=PATH`
- [x] `--suites llm_train,llm_infer`

## Report Structure Idea

### 1. Run Overview

Include:

- run label
- host name
- GPU model
- GPU backend
- visible GPU count
- software versions when relevant

### 2. Comparability Summary

Include:

- which runs are safely comparable
- which dimensions differ
- which suites are excluded from strict comparison

### 3. Per-suite Comparison

Potential sections:

- `llm_train`
  - compare `steps_per_sec`, `tokens_per_sec`
- `llm_infer`
  - compare `reqs_per_s`, `gen_tokens_per_s`
- `sd_infer`
  - compare `images_per_sec`
- `blender`
  - compare `time_s` with lower-is-better semantics

### 4. Repeat Stability

Where repeat data exists, include:

- mean
- min / max
- spread or coefficient of variation if useful

### 5. Caveats

Explicitly report:

- model mismatch
- backend mismatch
- tensor parallel mismatch
- multi-GPU mode mismatch
- AMD power telemetry limitations
- optional suite missing from one run

## Comparison Semantics To Define

### LLM Training

Likely primary metrics:

- `tokens_per_sec`
- `steps_per_sec`

Comparison keys likely include:

- `dtype`
- `seq_len`
- `batch_size`
- `hidden_size`
- `n_layers`
- `n_heads`
- `requested_world_size` or effective world size

### LLM Inference

Likely primary metrics:

- `reqs_per_s`
- `gen_tokens_per_s`
- latency proxy fields where valid

Comparison keys likely include:

- `backend`
- `model`
- `dtype`
- `batch_size`
- `per_gpu_batch_size`
- `tensor_parallel`
- `multi_gpu_mode`
- `prompt_len`
- `actual_prompt_tokens`
- `output_len`
- `gpu_count`

### Stable Diffusion

Likely primary metrics:

- `images_per_sec`
- `wall_time_s` or iteration timing

Comparison keys likely include:

- `model`
- `steps`
- `width`
- `height`
- `batch_size`
- `per_gpu_batch`
- `multi_gpu_mode`
- `dtype`

### Blender

Likely primary metrics:

- `time_s`

Comparison keys likely include:

- `scene`
- `backend`
- `mode` (`single` vs `all`)

Special note:

- Blender is lower-is-better, unlike the throughput suites

## Investigation Checklist

### 1. Inspect current artifact quality

- [ ] Verify which fields are consistently present in `metrics.jsonl`
- [ ] Verify which fields are consistently present in `metrics_summary.json`
- [ ] Identify any suite-specific field naming inconsistencies
- [ ] Decide whether comparison should be based on raw rows, summary rows, or both

### 2. Define run identity and labels

- [x] Decide how runs should be named in reports
- [x] Prefer human-friendly labels without losing the original run path
- [x] Decide whether labels come from CLI, metadata, or both

### 3. Define comparability rules

- [x] Create per-suite comparison keys
- [ ] Define which mismatches are hard blockers
- [x] Define which mismatches become warnings only
- [x] Define how missing suites should be reported

### 4. Define metric ranking behavior

- [x] Mark each metric as higher-is-better or lower-is-better
- [x] Decide how percent delta is computed
- [x] Decide how to report ties and near-ties
- [ ] Decide how to represent skipped or failed rows

### 5. Implement reporting output

- [x] Produce Markdown first
- [x] Keep report easy to read in GitHub and terminal
- [x] Include compact tables where they help
- [x] Avoid hiding caveats behind raw numbers

### 6. Validate with real repo data

- [ ] Compare at least two real run folders from different GPUs
- [ ] Compare repeated runs from the same GPU
- [x] Compare runs with intentionally mismatched configs to verify warning behavior
- [x] Ensure the report stays useful when optional suites are missing

## Suggested Delivery Phases

### Phase 1: Core comparison engine

- [x] compare two runs only
- [x] use existing summary artifacts
- [x] emit JSON plus Markdown
- [x] support strict like-for-like comparison

### Phase 2: Multi-run ranking

- [x] compare 3 or more runs
- [x] produce ranked suite summaries
- [x] include percent deltas against a chosen baseline

### Phase 3: Richer analysis

- [x] better repeat variability analysis
- [x] stronger mismatch diagnostics
- optional chart generation

### Phase 4: UX polish

- helper scripts
- [x] README documentation
- examples using real run folders

## Open Questions

- Should the tool compare run roots or arbitrary metrics files?
- Should baseline selection be automatic or explicit?
- Should we compare raw repeats directly, or only summary rows by default?
- How should we handle cases where runs used different repeat counts?
- Should the report include software version differences from `meta.json` as possible explanatory factors?
- Should the tool generate one combined report or one report per suite?

## Exit Criteria For Calling This "Done"

This feature should only be considered complete when:

- at least two real run folders can be compared successfully
- the report clearly distinguishes fair comparisons from misleading ones
- per-suite winners and deltas are easy to read
- missing or incompatible data is explained explicitly
- the output is useful without manual post-processing
