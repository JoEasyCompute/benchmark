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
TIE_EPSILON_PCT = 1.0


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


def parse_suite_filter(raw: str) -> set[str] | None:
    if not raw.strip():
        return None
    suites = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = sorted(suites - set(SUITE_METRICS))
    if unknown:
        raise SystemExit(f"[COMPARE][ERROR] Unknown suites in --suites: {', '.join(unknown)}")
    return suites


def build_status_index(runs: list[dict], allowed_suites: set[str] | None = None) -> dict[str, dict[tuple[tuple[str, object], ...], dict[str, dict]]]:
    statuses = {}
    for run in runs:
        for row in run["summary_rows"]:
            suite = row.get("suite", "unknown")
            if allowed_suites is not None and suite not in allowed_suites:
                continue
            row = dict(row)
            row["_run_label"] = run["label"]
            key = row_key(row)
            statuses.setdefault(suite, {}).setdefault(key, {})[run["label"]] = row
    return statuses


def build_groups(runs: list[dict], allowed_suites: set[str] | None = None) -> dict[str, dict[tuple[tuple[str, object], ...], dict[str, dict]]]:
    groups = {}
    for run in runs:
        for row in run["summary_rows"]:
            suite = row.get("suite", "unknown")
            if row.get("status") != "ok":
                continue
            if allowed_suites is not None and suite not in allowed_suites:
                continue
            row = dict(row)
            row["_run_label"] = run["label"]
            row["_torch_version"] = run["software_versions"].get("torch")
            row["_transformers_version"] = run["software_versions"].get("transformers")
            key = row_key(row)
            groups.setdefault(suite, {}).setdefault(key, {})[run["label"]] = row
    return groups


def metric_delta(metric: str, baseline, current):
    if not is_number(baseline) or not is_number(current) or baseline == 0:
        return None
    return ((current - baseline) / baseline) * 100.0


def competitive_delta_pct(metric: str, winner_value, runner_up_value):
    if not is_number(winner_value) or not is_number(runner_up_value):
        return None
    if metric in LOWER_IS_BETTER_METRICS:
        if runner_up_value == 0:
            return None
        return ((runner_up_value - winner_value) / runner_up_value) * 100.0
    if runner_up_value == 0:
        return None
    return ((winner_value - runner_up_value) / runner_up_value) * 100.0


def metric_per_gpu(metric: str, value, gpu_count):
    if metric not in THROUGHPUT_METRICS:
        return None
    if not is_number(value) or not is_number(gpu_count) or gpu_count <= 0:
        return None
    return value / gpu_count


def metric_base_name(metric: str) -> str:
    return metric[:-5] if metric.endswith("_mean") else metric


def metric_variability(row: dict | None, metric: str) -> dict | None:
    if not row:
        return None
    base = metric_base_name(metric)
    minimum = row.get(f"{base}_min")
    maximum = row.get(f"{base}_max")
    stdev = row.get(f"{base}_stdev")
    mean = row.get(metric)
    summary_count = row.get("summary_count")
    cv_pct = None
    if is_number(mean) and mean != 0 and is_number(stdev):
        cv_pct = abs(float(stdev) / float(mean)) * 100.0
    if not any(is_number(value) for value in (minimum, maximum, stdev, cv_pct)):
        return None
    return {
        "min": minimum,
        "max": maximum,
        "stdev": stdev,
        "cv_pct": cv_pct,
        "summary_count": summary_count,
    }


def format_variability(variability: dict | None) -> str:
    if not variability:
        return "n/a"
    parts = []
    minimum = variability.get("min")
    maximum = variability.get("max")
    stdev = variability.get("stdev")
    cv_pct = variability.get("cv_pct")
    summary_count = variability.get("summary_count")
    if is_number(minimum) and is_number(maximum):
        parts.append(f"range {format_value(minimum)}-{format_value(maximum)}")
    if is_number(stdev):
        parts.append(f"sd {format_value(stdev)}")
    if is_number(cv_pct):
        parts.append(f"cv {cv_pct:.2f}%")
    if is_number(summary_count):
        parts.append(f"n={int(summary_count)}")
    return ", ".join(parts) if parts else "n/a"


def compare_quality(
    suite: str,
    rows_present: list[dict],
    run_count: int,
    missing_statuses: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    notes = []
    if len(rows_present) != run_count:
        present_labels = [str(row.get("_run_label", "unknown")) for row in rows_present]
        notes.append(f"Not all runs contain this comparable row. Present in: {', '.join(sorted(present_labels))}.")
    if missing_statuses:
        for label, status in sorted(missing_statuses.items()):
            notes.append(f"Run `{label}` has status `{status}` for this comparable row.")

    gpu_counts_present = sorted(
        {int(row["gpu_count"]) for row in rows_present if str(row.get("gpu_count", "")).isdigit()}
    )
    backends_present = sorted({str(row.get("gpu_backend")) for row in rows_present if row.get("gpu_backend")})
    blender_backends_present = sorted({str(row.get("backend")) for row in rows_present if row.get("backend")})
    torch_versions = sorted({str(row.get("_torch_version")) for row in rows_present if row.get("_torch_version")})
    transformers_versions = sorted(
        {str(row.get("_transformers_version")) for row in rows_present if row.get("_transformers_version")}
    )
    repeat_counts = sorted(
        {int(row["summary_count"]) for row in rows_present if str(row.get("summary_count", "")).isdigit()}
    )

    if len(gpu_counts_present) > 1 and any(metric in THROUGHPUT_METRICS for metric in SUITE_METRICS.get(suite, ())):
        notes.append(
            f"Runs in this group use different gpu_count values ({', '.join(map(str, gpu_counts_present))}); total throughput is not normalized. "
            "Per-GPU values are shown for throughput metrics."
        )
    if suite == "blender" and len(blender_backends_present) > 1:
        notes.append(
            f"Blender rows use different render backends across runs ({', '.join(blender_backends_present)}); timings are shown together, "
            "but backend differences may affect fairness."
        )
    if len(backends_present) > 1:
        notes.append(f"Runs in this group span different gpu_backend values ({', '.join(backends_present)}).")
    if len(torch_versions) > 1:
        notes.append(f"Runs in this group use different torch versions ({', '.join(torch_versions)}).")
    if len(transformers_versions) > 1:
        notes.append(f"Runs in this group use different transformers versions ({', '.join(transformers_versions)}).")
    if len(repeat_counts) > 1:
        notes.append(f"Runs in this group use different repeat counts ({', '.join(map(str, repeat_counts))}).")

    if len(rows_present) != run_count:
        quality = "partial"
    elif suite == "blender" and len(blender_backends_present) > 1:
        quality = "directional"
    elif (
        len(backends_present) > 1
        or len(gpu_counts_present) > 1
        or len(torch_versions) > 1
        or len(transformers_versions) > 1
        or len(repeat_counts) > 1
    ):
        quality = "directional"
    else:
        quality = "strict"

    return quality, notes


def best_rows(metric: str, rows: list[dict]):
    candidates = [row for row in rows if is_number(row.get("value"))]
    if not candidates:
        return []
    reverse = metric not in LOWER_IS_BETTER_METRICS
    ranked = sorted(candidates, key=lambda row: row["value"], reverse=reverse)
    winner = ranked[0]
    ties = [winner]
    for candidate in ranked[1:]:
        gap = competitive_delta_pct(metric, winner["value"], candidate["value"])
        if gap is None or abs(gap) <= TIE_EPSILON_PCT:
            ties.append(candidate)
        else:
            break
    return ties


def best_row(metric: str, rows: list[dict]):
    winners = best_rows(metric, rows)
    return winners[0] if winners else None


def comparability_summary(payload: dict) -> list[dict]:
    summary = []
    for suite, suite_payload in sorted(payload["suites"].items()):
        groups = suite_payload["groups"]
        qualities = [group["quality"] for group in groups]
        if not groups:
            summary.append({"suite": suite, "group_count": 0, "best_quality": "none", "issues": []})
            continue
        best_quality = "strict" if "strict" in qualities else "directional" if "directional" in qualities else "partial"
        issues = []
        for group in groups:
            for note in group.get("notes", []):
                if note not in issues:
                    issues.append(note)
        summary.append(
            {
                "suite": suite,
                "group_count": len(groups),
                "best_quality": best_quality,
                "issues": issues,
            }
        )
    return summary


def is_single_gpu_group(suite: str, group: dict) -> bool:
    if suite == "blender":
        return group.get("key", {}).get("mode") == "single"
    gpu_counts = {
        int(row["gpu_count"])
        for metric_payload in group.get("metrics", {}).values()
        for row in metric_payload.get("rows", [])
        if str(row.get("gpu_count", "")).isdigit() and row.get("value") is not None
    }
    return bool(gpu_counts) and gpu_counts == {1}


def single_gpu_summary(payload: dict) -> list[dict]:
    summary = []
    for suite, suite_payload in sorted(payload["suites"].items()):
        groups = [group for group in suite_payload.get("groups", []) if is_single_gpu_group(suite, group)]
        if not groups:
            continue
        qualities = [group["quality"] for group in groups]
        best_quality = "strict" if "strict" in qualities else "directional" if "directional" in qualities else "partial"
        top_pick = None
        for decision in payload.get("executive_summary", {}).get("suite_decisions", []):
            if decision.get("suite") == suite:
                top_pick = decision
                break
        issues = []
        for group in groups:
            for note in group.get("notes", []):
                if note not in issues:
                    issues.append(note)
        summary.append(
            {
                "suite": suite,
                "group_count": len(groups),
                "best_quality": best_quality,
                "top_pick": None if not top_pick or not top_pick.get("winner") else top_pick["winner"],
                "metric": None if not top_pick or not top_pick.get("metric") else top_pick["metric"],
                "issues": issues,
            }
        )
    return summary


def metric_competitive_score(metric: str, rows: list[dict]) -> float | None:
    candidates = [row for row in rows if is_number(row.get("value"))]
    if len(candidates) < 2:
        return None
    reverse = metric not in LOWER_IS_BETTER_METRICS
    ranked = sorted(candidates, key=lambda row: row["value"], reverse=reverse)
    score = competitive_delta_pct(metric, ranked[0]["value"], ranked[1]["value"])
    if score is None:
        return None
    return abs(score)


def delta_sign_for_preference(metric: str, delta_pct: float | None) -> float | None:
    if delta_pct is None:
        return None
    return -delta_pct if metric in LOWER_IS_BETTER_METRICS else delta_pct


def describe_relative_to_baseline(metric: str, delta_pct: float | None) -> str:
    if delta_pct is None:
        return "n/a"
    preferred_delta = delta_sign_for_preference(metric, delta_pct)
    magnitude = abs(delta_pct)
    if preferred_delta is None:
        return "n/a"
    if preferred_delta > 0:
        return f"better than baseline by {magnitude:.3f}%"
    if preferred_delta < 0:
        return f"worse than baseline by {magnitude:.3f}%"
    return "matched baseline"


def summarize_suite_takeaway(suite: str, suite_payload: dict, decision_item: dict | None) -> str:
    groups = suite_payload.get("groups", [])
    if not groups:
        return "No successful comparable rows."
    strict_groups = [group for group in groups if group["quality"] == "strict"]
    directional_groups = [group for group in groups if group["quality"] == "directional"]
    partial_groups = [group for group in groups if group["quality"] == "partial"]
    if decision_item is None or decision_item.get("winner") is None:
        if partial_groups and not strict_groups and not directional_groups:
            return "No decision-grade result because coverage is only partial."
        return "No decision-grade result yet."
    winner = decision_item["winner"]
    metric = decision_item["metric"]
    quality = decision_item["quality"]
    tied_group_count = 0
    winner_group_count = 0
    for group in groups:
        metric_payload = group["metrics"].get(metric)
        if not metric_payload:
            continue
        tied_winners = metric_payload.get("tied_winners", [])
        if winner in tied_winners:
            if len(tied_winners) > 1:
                tied_group_count += 1
            else:
                winner_group_count += 1
    confidence = f"{quality} evidence"
    if strict_groups:
        confidence += f", {len(strict_groups)} strict group(s)"
    elif directional_groups:
        confidence += f", {len(directional_groups)} directional group(s)"
    if partial_groups:
        confidence += f", {len(partial_groups)} partial group(s)"
    if winner_group_count and tied_group_count:
        return f"`{winner}` leads on `{metric}` with {confidence}; some other comparable groups are ties."
    if tied_group_count:
        return f"`{winner}` is still the current pick on `{metric}` with {confidence}, but comparable groups include ties."
    return f"`{winner}` leads on `{metric}` with {confidence}."


def collect_risk_flags(payload: dict) -> list[str]:
    flags = []
    for suite_item in payload.get("comparability_summary", []):
        suite = suite_item["suite"]
        issues = suite_item.get("issues", [])
        if suite_item.get("best_quality") == "partial":
            flags.append(f"{suite}: comparison coverage is only partial.")
        for issue in issues:
            if "status `failed`" in issue:
                flags.append(f"{suite}: at least one run failed a comparable row.")
            elif "different torch versions" in issue:
                flags.append(f"{suite}: torch versions differ across runs.")
            elif "different transformers versions" in issue:
                flags.append(f"{suite}: transformers versions differ across runs.")
            elif "different repeat counts" in issue:
                flags.append(f"{suite}: repeat counts differ across runs.")
            elif "different gpu_count values" in issue:
                flags.append(f"{suite}: gpu_count differs across runs.")
            elif "render backends" in issue:
                flags.append(f"{suite}: Blender backend differs across runs.")
            elif "different gpu_backend values" in issue:
                flags.append(f"{suite}: gpu backend differs across runs.")
    deduped = []
    for flag in flags:
        if flag not in deduped:
            deduped.append(flag)
    return deduped[:8]


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
    suite_decisions = []
    baseline_label = payload["baseline_label"]
    strongest_gain = None
    strongest_loss = None

    for suite, suite_payload in payload["suites"].items():
        best_highlight = None
        decision_candidate = None
        saw_partial_only = False
        for group in suite_payload["groups"]:
            counts[group["quality"]] = counts.get(group["quality"], 0) + 1
            for metric, metric_payload in group["metrics"].items():
                winner = metric_payload.get("winner")
                tied_winners = metric_payload.get("tied_winners", [])
                if len(tied_winners) == 1 and winner:
                    winners[winner] = winners.get(winner, 0) + 1
                winner_row = next((row for row in metric_payload["rows"] if row["label"] == winner), None) if winner else None
                if winner and winner_row and winner_row.get("value") is not None:
                    decision_score = metric_competitive_score(metric, metric_payload["rows"])
                    candidate = {
                        "suite": suite,
                        "quality": group["quality"],
                        "winner": winner,
                        "metric": metric,
                        "group_key_text": group["key_text"],
                    }
                    if group["quality"] == "partial":
                        saw_partial_only = True
                    else:
                        if decision_candidate is None:
                            decision_candidate = (candidate, metric_payload["lower_is_better"], decision_score)
                        else:
                            current_candidate, current_lower_is_better, current_score = decision_candidate
                            if group["quality"] == "strict" and current_candidate["quality"] != "strict":
                                decision_candidate = (candidate, metric_payload["lower_is_better"], decision_score)
                            elif (
                                group["quality"] == current_candidate["quality"]
                                and metric_payload["lower_is_better"] == current_lower_is_better
                                and decision_score is not None
                                and (current_score is None or decision_score > current_score)
                            ):
                                decision_candidate = (candidate, metric_payload["lower_is_better"], decision_score)
                if group["quality"] != "strict":
                    continue
                if baseline_label not in group["runs_present"]:
                    continue
                baseline_row = next((row for row in metric_payload["rows"] if row["label"] == baseline_label), None)
                if not baseline_row:
                    continue
                competitor_rows = [
                    row for row in metric_payload["rows"]
                    if row["label"] != baseline_label and row.get("delta_vs_baseline_pct") is not None
                ]
                if not competitor_rows:
                    continue
                competitor = max(competitor_rows, key=lambda row: abs(row["delta_vs_baseline_pct"]))
                delta = competitor["delta_vs_baseline_pct"]
                score = abs(delta) if delta is not None else None
                signed_delta = delta_sign_for_preference(metric, delta)
                if score is None:
                    continue
                candidate = {
                    "suite": suite,
                    "metric": metric,
                    "winner": competitor["label"],
                    "delta_vs_baseline_pct": round(delta, 3),
                    "preferred_delta_vs_baseline_pct": None if signed_delta is None else round(signed_delta, 3),
                    "relative_to_baseline_text": describe_relative_to_baseline(metric, delta),
                    "group_key_text": group["key_text"],
                }
                if best_highlight is None or score > abs(best_highlight["delta_vs_baseline_pct"]):
                    best_highlight = candidate
                if signed_delta is not None and signed_delta > 0 and (
                    strongest_gain is None or signed_delta > strongest_gain["preferred_delta_vs_baseline_pct"]
                ):
                    strongest_gain = candidate
                if signed_delta is not None and signed_delta < 0 and (
                    strongest_loss is None or signed_delta < strongest_loss["preferred_delta_vs_baseline_pct"]
                ):
                    strongest_loss = candidate
        if best_highlight:
            suite_highlights.append(best_highlight)
        if decision_candidate:
            suite_decisions.append(decision_candidate[0])
        elif saw_partial_only:
            suite_decisions.append(
                {
                    "suite": suite,
                    "quality": "partial",
                    "winner": None,
                    "metric": None,
                    "group_key_text": None,
                }
            )

    winner_counts = [{"label": label, "metric_wins": count} for label, count in sorted(winners.items(), key=lambda item: (-item[1], item[0]))]
    suite_confidence = []
    suite_takeaways = []
    decision_map = {item["suite"]: item for item in suite_decisions}
    for suite, suite_payload in payload["suites"].items():
        groups = suite_payload["groups"]
        quality_counts = {"strict": 0, "directional": 0, "partial": 0}
        for group in groups:
            quality_counts[group["quality"]] += 1
        best_quality = "strict" if quality_counts["strict"] else "directional" if quality_counts["directional"] else "partial"
        has_failures = any("status `failed`" in note for group in groups for note in group.get("notes", []))
        suite_confidence.append(
            {
                "suite": suite,
                "best_quality": best_quality,
                "group_count": len(groups),
                "strict_groups": quality_counts["strict"],
                "directional_groups": quality_counts["directional"],
                "partial_groups": quality_counts["partial"],
                "has_failures": has_failures,
            }
        )
        suite_takeaways.append(
            {
                "suite": suite,
                "text": summarize_suite_takeaway(suite, suite_payload, decision_map.get(suite)),
            }
        )
    return {
        "group_counts": counts,
        "winner_counts": winner_counts,
        "suite_decisions": suite_decisions,
        "suite_confidence": suite_confidence,
        "suite_takeaways": suite_takeaways,
        "suite_highlights": suite_highlights,
        "risk_flags": collect_risk_flags(payload),
        "strongest_gain": strongest_gain,
        "strongest_loss": strongest_loss,
    }


def build_payload(runs: list[dict], baseline: str | None = None, allowed_suites: set[str] | None = None) -> dict:
    groups = build_groups(runs, allowed_suites=allowed_suites)
    statuses = build_status_index(runs, allowed_suites=allowed_suites)
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
            status_entries = ((statuses.get(suite) or {}).get(key) or {})
            missing_statuses = {
                run["label"]: status_entries[run["label"]].get("status", "unknown")
                for run in runs
                if run["label"] not in entries and run["label"] in status_entries
            }
            quality, notes = compare_quality(suite, rows_present, len(runs), missing_statuses=missing_statuses)
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
                            "delta_vs_baseline_pct": None if delta_pct is None else round(delta_pct, 3),
                            "gpu_count": gpu_count,
                            "gpu_backend": run["gpu_backend"],
                            "gpu_name": run["gpu_name"],
                            "max_gpu_count": run["max_gpu_count"],
                            "variability": metric_variability(row, metric),
                        }
                    )
                winners = best_rows(metric, metric_rows)
                winner = winners[0] if winners else None
                group_payload["metrics"][metric] = {
                    "lower_is_better": metric in LOWER_IS_BETTER_METRICS,
                    "show_per_gpu": show_per_gpu,
                    "winner": None if winner is None else winner["label"],
                    "tied_winners": [row["label"] for row in winners],
                    "rows": metric_rows,
                }
            suite_payload["groups"].append(group_payload)
        payload["suites"][suite] = suite_payload

    payload["comparability_summary"] = comparability_summary(payload)
    payload["executive_summary"] = compute_executive_summary(payload)
    payload["single_gpu_summary"] = single_gpu_summary(payload)
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
    lines.append("### Decision View")
    lines.append("")
    if summary["suite_decisions"]:
        for item in summary["suite_decisions"]:
            if item["winner"] is None:
                lines.append(f"- No decision-grade pick for `{item['suite']}` yet (partial coverage only)")
            else:
                lines.append(
                    f"- Best current pick for `{item['suite']}`: `{item['winner']}` "
                    f"based on `{item['metric']}` ({item['quality']})"
                )
    else:
        lines.append("- No decision-oriented suite picks available yet.")
    lines.append("")
    lines.append("### Benchmark View")
    lines.append("")
    lines.append(
        f"- Groups: strict={group_counts.get('strict', 0)}, "
        f"directional={group_counts.get('directional', 0)}, partial={group_counts.get('partial', 0)}"
    )
    if summary["winner_counts"]:
        top_winner = summary["winner_counts"][0]
        lines.append(f"- Most metric wins: `{top_winner['label']}` ({top_winner['metric_wins']} metrics)")
    else:
        lines.append("- Most metric wins: n/a")
    strongest_gain = summary.get("strongest_gain")
    strongest_loss = summary.get("strongest_loss")
    if strongest_gain:
        lines.append(
            f"- Strongest gain vs baseline: `{strongest_gain['winner']}` on "
            f"`{strongest_gain['suite']}` / `{strongest_gain['metric']}` "
            f"({strongest_gain['relative_to_baseline_text']})"
        )
    else:
        lines.append("- Strongest gain vs baseline: n/a")
    if strongest_loss:
        lines.append(
            f"- Largest baseline lead: baseline stays ahead on "
            f"`{strongest_loss['suite']}` / `{strongest_loss['metric']}` "
            f"({strongest_loss['relative_to_baseline_text']})"
        )
    else:
        lines.append("- Largest baseline lead: n/a")
    if summary["suite_highlights"]:
        lines.append("- Per-suite highlights:")
        for item in summary["suite_highlights"]:
            lines.append(
                f"  - {item['suite']}: `{item['winner']}` on `{item['metric']}` "
                f"({item['relative_to_baseline_text']})"
            )
    else:
        lines.append("- Per-suite highlights: n/a")
    lines.append("")
    lines.append("### Decision Confidence")
    lines.append("")
    for item in summary.get("suite_confidence", []):
        flags = []
        if item["strict_groups"]:
            flags.append(f"strict={item['strict_groups']}")
        if item["directional_groups"]:
            flags.append(f"directional={item['directional_groups']}")
        if item["partial_groups"]:
            flags.append(f"partial={item['partial_groups']}")
        if item["has_failures"]:
            flags.append("failed rows present")
        flags_text = ", ".join(flags) if flags else "no comparable groups"
        lines.append(f"- `{item['suite']}`: best quality `{item['best_quality']}`, groups={item['group_count']}, {flags_text}")
    lines.append("")
    lines.append("### Suite Takeaways")
    lines.append("")
    for item in summary.get("suite_takeaways", []):
        lines.append(f"- `{item['suite']}`: {item['text']}")
    lines.append("")
    lines.append("### Risk Flags")
    lines.append("")
    risk_flags = summary.get("risk_flags", [])
    if risk_flags:
        for flag in risk_flags:
            lines.append(f"- {flag}")
    else:
        lines.append("- No major report-level risk flags.")
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
    lines.append("## Single-GPU View")
    lines.append("")
    single_gpu_items = payload.get("single_gpu_summary", [])
    if single_gpu_items:
        lines.append("| Suite | Best Quality | Groups | Top Pick | Metric | Issues |")
        lines.append("| --- | --- | ---: | --- | --- | --- |")
        for item in single_gpu_items:
            issues_text = "; ".join(item["issues"]) if item["issues"] else "n/a"
            lines.append(
                f"| {item['suite']} | {item['best_quality']} | {item['group_count']} | "
                f"{format_value(item['top_pick'])} | {format_value(item['metric'])} | {issues_text} |"
            )
    else:
        lines.append("No single-GPU comparable groups found.")
    lines.append("")
    lines.append("## Comparability Summary")
    lines.append("")
    lines.append("| Suite | Best Quality | Groups | Issues |")
    lines.append("| --- | --- | ---: | --- |")
    for item in payload.get("comparability_summary", []):
        issues_text = "; ".join(item["issues"]) if item["issues"] else "n/a"
        lines.append(f"| {item['suite']} | {item['best_quality']} | {item['group_count']} | {issues_text} |")
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
                tied_winners = metric_payload.get("tied_winners", [])
                if len(tied_winners) > 1:
                    lines.append(f"Best run: tie between `{', '.join(tied_winners)}`")
                    lines.append("")
                elif metric_payload.get("winner"):
                    lines.append(f"Best run: `{metric_payload['winner']}`")
                    lines.append("")
                if metric_payload.get("show_per_gpu"):
                    lines.append("| Run | Value | Per-GPU Value | Delta vs Baseline | GPU Count | Repeat Variability |")
                    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
                else:
                    lines.append("| Run | Value | Delta vs Baseline | GPU Count | Repeat Variability |")
                    lines.append("| --- | ---: | ---: | ---: | --- |")
                for row in metric_payload["rows"]:
                    delta = row["delta_vs_baseline_pct"]
                    delta_text = "n/a" if delta is None else f"{delta:+.3f}%"
                    value_text = "n/a" if row["value"] is None else format_value(row["value"])
                    variability_text = format_variability(row.get("variability"))
                    if metric_payload.get("show_per_gpu"):
                        per_gpu_text = "n/a" if row["per_gpu_value"] is None else format_value(row["per_gpu_value"])
                        lines.append(
                            f"| {row['label']} | {value_text} | {per_gpu_text} | {delta_text} | {format_value(row['gpu_count'])} | {variability_text} |"
                        )
                    else:
                        lines.append(
                            f"| {row['label']} | {value_text} | {delta_text} | {format_value(row['gpu_count'])} | {variability_text} |"
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
    ap.add_argument("--suites", default="", help="Comma-separated suites to include")
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
    allowed_suites = parse_suite_filter(args.suites)
    payload = build_payload(runs, baseline=args.baseline or None, allowed_suites=allowed_suites)
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
