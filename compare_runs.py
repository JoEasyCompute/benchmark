#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    raise SystemExit("[COMPARE][ERROR] Missing dependency: PyYAML. Run env_setup.sh or activate the project venv.")


SUITE_KEY_FIELDS = {
    "llm_train": (
        "dtype",
        "seq_len",
        "batch_size",
        "hidden_size",
        "n_layers",
        "n_heads",
        "world_size",
    ),
    "llm_infer": (
        "backend",
        "model",
        "dtype",
        "multi_gpu_mode",
        "per_gpu_batch_size",
        "tensor_parallel",
        "requested_prompt_len",
        "output_len",
    ),
    "sd_infer": (
        "model",
        "steps",
        "width",
        "height",
        "per_gpu_batch",
        "multi_gpu_mode",
        "dtype",
    ),
    "blender": (
        "scene",
        "mode",
    ),
}

SUITE_METRICS = {
    "llm_train": ("tokens_per_sec_mean", "steps_per_sec_mean"),
    "llm_infer": ("gen_tokens_per_s_mean", "reqs_per_s_mean"),
    "sd_infer": ("images_per_sec_mean",),
    "blender": ("time_s_mean",),
}

LOWER_IS_BETTER_METRICS = {"time_s_mean"}
THROUGHPUT_METRICS = {
    "tokens_per_sec_mean",
    "steps_per_sec_mean",
    "gen_tokens_per_s_mean",
    "reqs_per_s_mean",
    "images_per_sec_mean",
}
REQUIRED_FILES = ("meta.json", "effective_config.yaml", "metrics_summary.json")


def load_json(path: Path):
    return json.loads(path.read_text())


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text()) or {}


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def default_label(run_dir: Path, meta: dict) -> str:
    gpu_backend = meta.get("gpu_backend") or "unknown"
    gpu_name = (((meta.get("gpu_smi") or "").splitlines() or ["unknown"])[0]).strip()
    if not gpu_name:
        gpu_name = "unknown"
    return f"{run_dir.name} [{gpu_backend}]"


def infer_gpu_name(meta: dict, summary_rows: list[dict]) -> str:
    for row in summary_rows:
        name = row.get("gpu_name")
        if name:
            return str(name)
    gpu_smi = meta.get("gpu_smi")
    if isinstance(gpu_smi, str) and gpu_smi:
        return gpu_smi.splitlines()[0][:80]
    return "unknown"


def infer_max_gpu_count(summary_rows: list[dict]) -> int | None:
    counts = [int(row["gpu_count"]) for row in summary_rows if str(row.get("gpu_count", "")).isdigit()]
    return max(counts) if counts else None


def load_run(run_dir: Path, label: str | None = None) -> dict:
    missing = [rel for rel in REQUIRED_FILES if not (run_dir / rel).exists()]
    if missing:
        raise SystemExit(f"[COMPARE][ERROR] Missing required files in {run_dir}: {', '.join(missing)}")

    meta = load_json(run_dir / "meta.json")
    effective_config = load_yaml(run_dir / "effective_config.yaml")
    summary_rows = load_json(run_dir / "metrics_summary.json")
    if not isinstance(summary_rows, list):
        raise SystemExit(f"[COMPARE][ERROR] metrics_summary.json is not a list: {run_dir}")

    label = label or default_label(run_dir, meta)
    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "label": label,
        "meta": meta,
        "effective_config": effective_config,
        "summary_rows": summary_rows,
        "gpu_backend": meta.get("gpu_backend", "unknown"),
        "gpu_name": infer_gpu_name(meta, summary_rows),
        "max_gpu_count": infer_max_gpu_count(summary_rows),
        "python": meta.get("python"),
        "platform": meta.get("platform"),
        "software_versions": meta.get("software_versions", {}) or {},
    }


def suite_key_fields(suite: str) -> tuple[str, ...]:
    return SUITE_KEY_FIELDS.get(suite, ("status",))


def row_key(row: dict) -> tuple[tuple[str, object], ...]:
    suite = row.get("suite", "unknown")
    fields = suite_key_fields(suite)
    return tuple((field, row.get(field)) for field in fields if field in row)


def key_to_dict(key: tuple[tuple[str, object], ...]) -> dict:
    return {k: v for k, v in key}


def format_value(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def format_key_summary(key: tuple[tuple[str, object], ...]) -> str:
    payload = key_to_dict(key)
    if not payload:
        return "default"
    return ", ".join(f"{field}={format_value(value)}" for field, value in payload.items())


def build_groups(runs: list[dict]) -> dict[str, dict[tuple[tuple[str, object], ...], dict[str, dict]]]:
    groups = {}
    for run in runs:
        for row in run["summary_rows"]:
            suite = row.get("suite", "unknown")
            if row.get("status") != "ok":
                continue
            key = row_key(row)
            groups.setdefault(suite, {}).setdefault(key, {})[run["label"]] = row
    return groups


def metric_delta(metric: str, baseline, current):
    if not is_number(baseline) or not is_number(current) or baseline == 0:
        return None
    return ((current - baseline) / baseline) * 100.0


def metric_per_gpu(metric: str, value, gpu_count):
    if metric not in THROUGHPUT_METRICS:
        return None
    if not is_number(value) or not is_number(gpu_count) or gpu_count <= 0:
        return None
    return value / gpu_count


def compare_quality(suite: str, rows_present: list[dict], run_count: int) -> tuple[str, list[str]]:
    notes = []
    if len(rows_present) != run_count:
        notes.append("Not all runs contain this comparable row.")

    gpu_counts_present = sorted(
        {int(row["gpu_count"]) for row in rows_present if str(row.get("gpu_count", "")).isdigit()}
    )
    backends_present = sorted({str(row.get("gpu_backend")) for row in rows_present if row.get("gpu_backend")})
    blender_backends_present = sorted({str(row.get("backend")) for row in rows_present if row.get("backend")})

    if len(gpu_counts_present) > 1 and any(metric in THROUGHPUT_METRICS for metric in SUITE_METRICS.get(suite, ())):
        notes.append(
            "Runs in this group use different gpu_count values; total throughput is not normalized. "
            "Per-GPU values are shown for throughput metrics."
        )
    if suite == "blender" and len(blender_backends_present) > 1:
        notes.append(
            "Blender rows use different render backends across runs; timings are shown together, "
            "but backend differences may affect fairness."
        )
    elif len(backends_present) > 1:
        notes.append("Runs in this group span different gpu_backend values.")

    if len(rows_present) != run_count:
        quality = "partial"
    elif suite == "blender" and len(blender_backends_present) > 1:
        quality = "directional"
    elif len(backends_present) > 1 and len(gpu_counts_present) > 1:
        quality = "directional"
    else:
        quality = "strict"

    return quality, notes


def best_row(metric: str, rows: list[dict]):
    candidates = [row for row in rows if is_number(row.get("value"))]
    if not candidates:
        return None
    reverse = metric not in LOWER_IS_BETTER_METRICS
    return sorted(candidates, key=lambda row: row["value"], reverse=reverse)[0]


def baseline_key(run: dict) -> tuple[str, str, str]:
    return run["label"], run["run_name"], run["run_dir"]


def resolve_baseline_label(runs: list[dict], baseline: str | None) -> str:
    if baseline is None:
        return runs[0]["label"]
    for run in runs:
        if baseline in baseline_key(run):
            return run["label"]
    raise SystemExit(f"[COMPARE][ERROR] Unknown baseline run: {baseline}")


def compute_executive_summary(payload: dict) -> dict:
    counts = {"strict": 0, "directional": 0, "partial": 0}
    winners = {}
    suite_highlights = []
    baseline_label = payload["baseline_label"]

    for suite, suite_payload in payload["suites"].items():
        best_highlight = None
        for group in suite_payload["groups"]:
            counts[group["quality"]] = counts.get(group["quality"], 0) + 1
            for metric, metric_payload in group["metrics"].items():
                winner = metric_payload.get("winner")
                if winner:
                    winners[winner] = winners.get(winner, 0) + 1
                if group["quality"] != "strict":
                    continue
                if baseline_label not in group["runs_present"]:
                    continue
                baseline_row = next((row for row in metric_payload["rows"] if row["label"] == baseline_label), None)
                if not baseline_row:
                    continue
                competitor_rows = [
                    row for row in metric_payload["rows"]
                    if row["label"] != baseline_label and row.get("delta_vs_first_run_pct") is not None
                ]
                if not competitor_rows:
                    continue
                competitor = max(competitor_rows, key=lambda row: abs(row["delta_vs_first_run_pct"]))
                delta = competitor["delta_vs_first_run_pct"]
                score = abs(delta) if delta is not None else None
                if score is None:
                    continue
                candidate = {
                    "suite": suite,
                    "metric": metric,
                    "winner": competitor["label"],
                    "delta_vs_baseline_pct": round(delta, 3),
                    "group_key_text": group["key_text"],
                }
                if best_highlight is None or score > abs(best_highlight["delta_vs_baseline_pct"]):
                    best_highlight = candidate
        if best_highlight:
            suite_highlights.append(best_highlight)

    winner_counts = [{"label": label, "metric_wins": count} for label, count in sorted(winners.items(), key=lambda item: (-item[1], item[0]))]
    return {
        "group_counts": counts,
        "winner_counts": winner_counts,
        "suite_highlights": suite_highlights,
    }


def build_payload(runs: list[dict], baseline: str | None = None) -> dict:
    groups = build_groups(runs)
    baseline_label = resolve_baseline_label(runs, baseline)
    payload = {
        "run_count": len(runs),
        "baseline_label": baseline_label,
        "runs": [],
        "suites": {},
    }

    for run in runs:
        payload["runs"].append(
            {
                "label": run["label"],
                "run_dir": run["run_dir"],
                "gpu_backend": run["gpu_backend"],
                "gpu_name": run["gpu_name"],
                "max_gpu_count": run["max_gpu_count"],
                "python": run["python"],
                "transformers": run["software_versions"].get("transformers"),
                "torch": run["software_versions"].get("torch"),
            }
        )

    for suite, suite_groups in sorted(groups.items()):
        suite_payload = {"groups": []}
        for key, entries in sorted(suite_groups.items(), key=lambda item: format_key_summary(item[0])):
            rows_present = list(entries.values())
            gpu_counts_present = sorted(
                {int(row["gpu_count"]) for row in rows_present if str(row.get("gpu_count", "")).isdigit()}
            )
            quality, notes = compare_quality(suite, rows_present, len(runs))
            group_payload = {
                "key": key_to_dict(key),
                "key_text": format_key_summary(key),
                "run_count": len(entries),
                "fully_comparable": len(entries) == len(runs),
                "quality": quality,
                "runs_present": sorted(entries.keys()),
                "notes": notes,
                "metrics": {},
            }
            for metric in SUITE_METRICS.get(suite, ()):
                metric_rows = []
                baseline_value = None
                show_per_gpu = len(gpu_counts_present) > 1 and metric in THROUGHPUT_METRICS
                if baseline_label in entries:
                    baseline_value = entries[baseline_label].get(metric)
                for run in runs:
                    row = entries.get(run["label"])
                    value = row.get(metric) if row else None
                    gpu_count = row.get("gpu_count") if row else None
                    per_gpu_value = metric_per_gpu(metric, value, gpu_count)
                    delta_pct = metric_delta(metric, baseline_value, value) if baseline_value is not None else None
                    metric_rows.append(
                        {
                            "label": run["label"],
                            "value": value,
                            "per_gpu_value": None if per_gpu_value is None else round(per_gpu_value, 6),
                            "delta_vs_first_run_pct": None if delta_pct is None else round(delta_pct, 3),
                            "gpu_backend": run["gpu_backend"],
                            "gpu_name": run["gpu_name"],
                            "max_gpu_count": run["max_gpu_count"],
                        }
                    )
                winner = best_row(metric, metric_rows)
                group_payload["metrics"][metric] = {
                    "lower_is_better": metric in LOWER_IS_BETTER_METRICS,
                    "show_per_gpu": show_per_gpu,
                    "winner": None if winner is None else winner["label"],
                    "rows": metric_rows,
                }
            suite_payload["groups"].append(group_payload)
        payload["suites"][suite] = suite_payload

    payload["executive_summary"] = compute_executive_summary(payload)
    return payload


def render_markdown(payload: dict) -> str:
    lines = []
    lines.append("# Run Comparison Report")
    lines.append("")
    lines.append(f"Compared runs: {payload['run_count']}")
    lines.append(f"Baseline run: `{payload['baseline_label']}`")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    summary = payload["executive_summary"]
    group_counts = summary["group_counts"]
    lines.append(
        f"- Groups: strict={group_counts.get('strict', 0)}, "
        f"directional={group_counts.get('directional', 0)}, partial={group_counts.get('partial', 0)}"
    )
    if summary["winner_counts"]:
        top_winner = summary["winner_counts"][0]
        lines.append(f"- Most metric wins: `{top_winner['label']}` ({top_winner['metric_wins']} metrics)")
    else:
        lines.append("- Most metric wins: n/a")
    if summary["suite_highlights"]:
        for item in summary["suite_highlights"]:
            delta = item["delta_vs_baseline_pct"]
            delta_text = f"{delta:+.3f}%"
            lines.append(
                f"- {item['suite']}: `{item['winner']}` is strongest on `{item['metric']}` "
                f"vs baseline ({delta_text})"
            )
    else:
        lines.append("- No strict-comparison highlights available yet.")
    lines.append("")
    lines.append("## Run Overview")
    lines.append("")
    lines.append("| Label | Backend | GPU | Max GPU Count | Torch | Transformers |")
    lines.append("| --- | --- | --- | ---: | --- | --- |")
    for run in payload["runs"]:
        lines.append(
            f"| {run['label']} | {run['gpu_backend']} | {run['gpu_name']} | "
            f"{format_value(run['max_gpu_count'])} | {format_value(run['torch'])} | {format_value(run['transformers'])} |"
        )
    lines.append("")

    for suite, suite_payload in sorted(payload["suites"].items()):
        lines.append(f"## Suite: {suite}")
        lines.append("")
        if not suite_payload["groups"]:
            lines.append("No successful comparable rows found.")
            lines.append("")
            continue

        for index, group in enumerate(suite_payload["groups"], start=1):
            lines.append(f"### Group {index} ({group['quality']})")
            lines.append("")
            lines.append(f"Key: `{group['key_text']}`")
            lines.append("")
            lines.append(f"Runs present: {', '.join(group['runs_present'])}")
            lines.append("")
            for note in group.get("notes", []):
                lines.append(f"Note: {note}")
                lines.append("")
            for metric, metric_payload in group["metrics"].items():
                direction = "lower is better" if metric_payload["lower_is_better"] else "higher is better"
                lines.append(f"#### Metric: {metric} ({direction})")
                lines.append("")
                if metric_payload.get("winner"):
                    lines.append(f"Best run: `{metric_payload['winner']}`")
                    lines.append("")
                if metric_payload.get("show_per_gpu"):
                    lines.append("| Run | Value | Per-GPU Value | Delta vs First Run | GPU Count |")
                    lines.append("| --- | ---: | ---: | ---: | ---: |")
                else:
                    lines.append("| Run | Value | Delta vs First Run | GPU Count |")
                    lines.append("| --- | ---: | ---: | ---: |")
                for row in metric_payload["rows"]:
                    delta = row["delta_vs_first_run_pct"]
                    delta_text = "n/a" if delta is None else f"{delta:+.3f}%"
                    value_text = "n/a" if row["value"] is None else format_value(row["value"])
                    if metric_payload.get("show_per_gpu"):
                        per_gpu_text = "n/a" if row["per_gpu_value"] is None else format_value(row["per_gpu_value"])
                        lines.append(
                            f"| {row['label']} | {value_text} | {per_gpu_text} | {delta_text} | {format_value(row['max_gpu_count'])} |"
                        )
                    else:
                        lines.append(
                            f"| {row['label']} | {value_text} | {delta_text} | {format_value(row['max_gpu_count'])} |"
                        )
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_labeled_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit("[COMPARE][ERROR] --label must be in NAME=PATH format")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise SystemExit("[COMPARE][ERROR] --label must be in NAME=PATH format")
    return name, Path(raw_path).resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="*", help="Path to completed results/<run_id> directories")
    ap.add_argument("--label", action="append", default=[], help="Label a run as NAME=PATH")
    ap.add_argument("--baseline", default="", help="Baseline run label, run name, or run path")
    ap.add_argument("--out-dir", default="", help="Directory to write comparison.json and comparison.md")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()

    labeled_runs = [parse_labeled_run(spec) for spec in args.label]
    unlabeled_runs = [Path(item).resolve() for item in args.run_dirs]
    all_paths = [path for _, path in labeled_runs] + unlabeled_runs
    if len(all_paths) < 2:
        raise SystemExit("[COMPARE][ERROR] Provide at least two run directories")
    if len({str(path) for path in all_paths}) != len(all_paths):
        raise SystemExit("[COMPARE][ERROR] Duplicate run directories are not allowed")

    runs = [load_run(run_dir, label=name) for name, run_dir in labeled_runs]
    runs.extend(load_run(run_dir) for run_dir in unlabeled_runs)
    payload = build_payload(runs, baseline=args.baseline or None)
    markdown = render_markdown(payload)

    print(markdown, end="")

    json_target = Path(args.json_out) if args.json_out else None
    md_target = Path(args.md_out) if args.md_out else None
    if args.out_dir:
        out_dir = Path(args.out_dir)
        json_target = out_dir / "comparison.json"
        md_target = out_dir / "comparison.md"

    if json_target:
        json_path = json_target
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2) + "\n")

    if md_target:
        md_path = md_target
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown)


if __name__ == "__main__":
    main()
