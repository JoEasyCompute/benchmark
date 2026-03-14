#!/usr/bin/env python3
import csv
import json
import math
import os
import statistics

os.makedirs("results", exist_ok=True)
jsonl_path = "results/metrics.jsonl"
csv_path = "results/metrics.csv"
summary_json_path = "results/metrics_summary.json"
summary_csv_path = "results/metrics_summary.csv"

SUMMARY_METRIC_KEYS = {
    "steps_per_sec",
    "tokens_per_sec",
    "time_s",
    "end_to_end_time_s",
    "reqs_per_s",
    "generated_tokens",
    "gen_tokens_per_s",
    "mean_power_w",
    "gen_tokens_per_watt",
    "images_total",
    "images_per_sec",
    "load_seconds",
    "mean_s_per_iter",
    "warm",
    "cold",
    "compile",
}


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def summarize_rows(rows):
    groups = {}
    for row in rows:
        group_key = tuple(
            sorted(
                (k, json.dumps(v, sort_keys=True))
                for k, v in row.items()
                if k not in SUMMARY_METRIC_KEYS and k not in {"repeat_index", "repeat_count"}
            )
        )
        groups.setdefault(group_key, []).append(row)

    summaries = []
    for key, group_rows in groups.items():
        summary = {k: json.loads(v) for k, v in key}
        repeat_values = sorted(
            {int(r["repeat_index"]) for r in group_rows if is_number(r.get("repeat_index"))}
        )
        summary["summary_count"] = len(group_rows)
        if repeat_values:
            summary["repeat_indices"] = ",".join(map(str, repeat_values))

        for metric_key in sorted(SUMMARY_METRIC_KEYS):
            values = [float(r[metric_key]) for r in group_rows if is_number(r.get(metric_key))]
            if not values:
                continue
            summary[f"{metric_key}_mean"] = round(statistics.fmean(values), 6)
            summary[f"{metric_key}_min"] = round(min(values), 6)
            summary[f"{metric_key}_max"] = round(max(values), 6)
            summary[f"{metric_key}_stdev"] = round(statistics.stdev(values), 6) if len(values) > 1 else 0.0

        summaries.append(summary)

    return summaries


rows = []
if os.path.exists(jsonl_path):
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

if rows:
    write_csv(csv_path, rows)
    summaries = summarize_rows(rows)
    with open(summary_json_path, "w") as f:
        json.dump(summaries, f, indent=2)
        f.write("\n")
    write_csv(summary_csv_path, summaries)
    print(f"Wrote {csv_path} with {len(rows)} rows")
    print(f"Wrote {summary_csv_path} with {len(summaries)} summary rows")
else:
    print("No metrics found to consolidate.")
