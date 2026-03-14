#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


REQUIRED_FILES = (
    "meta.json",
    "machine_state.json",
    "runtime_estimate.json",
    "results/metrics.jsonl",
)

REQUIRED_SUITES = ("llm_train", "llm_infer", "sd_infer")
OPTIONAL_SUITES = ("llm_train_real", "blender")


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def add_issue(payload, level: str, message: str):
    payload[f"{level}s"].append(message)


def suite_status_summary(rows):
    summary = {}
    for row in rows:
        suite = row.get("suite", "unknown")
        status = row.get("status", "unknown")
        suite_entry = summary.setdefault(suite, {"total_rows": 0, "statuses": {}})
        suite_entry["total_rows"] += 1
        suite_entry["statuses"][status] = suite_entry["statuses"].get(status, 0) + 1
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Path to a completed results/<run_id> directory")
    ap.add_argument("--expected-backend", choices=("nvidia", "amd"))
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    payload = {
        "status": "ok",
        "run_dir": str(run_dir),
        "checks": [],
        "warnings": [],
        "errors": [],
    }

    if not run_dir.exists():
        add_issue(payload, "error", f"run directory does not exist: {run_dir}")
    elif not run_dir.is_dir():
        add_issue(payload, "error", f"run path is not a directory: {run_dir}")

    if payload["errors"]:
        payload["status"] = "error"
    else:
        for rel_path in REQUIRED_FILES:
            path = run_dir / rel_path
            payload["checks"].append({"file": rel_path, "exists": path.exists()})
            if not path.exists():
                add_issue(payload, "error", f"missing required artifact: {rel_path}")

    metrics_rows = []
    meta = {}
    machine_state = {}
    if not payload["errors"]:
        try:
            meta = load_json(run_dir / "meta.json")
        except Exception as exc:
            add_issue(payload, "error", f"failed to parse meta.json: {exc}")
        try:
            machine_state = load_json(run_dir / "machine_state.json")
        except Exception as exc:
            add_issue(payload, "error", f"failed to parse machine_state.json: {exc}")
        try:
            metrics_rows = load_jsonl(run_dir / "results/metrics.jsonl")
        except Exception as exc:
            add_issue(payload, "error", f"failed to parse results/metrics.jsonl: {exc}")

    if payload["errors"]:
        payload["status"] = "error"
    else:
        backend = meta.get("gpu_backend")
        payload["checks"].append({"gpu_backend": backend})
        if backend not in {"nvidia", "amd"}:
            add_issue(payload, "error", f"meta.json has unexpected gpu_backend: {backend!r}")
        if args.expected_backend and backend != args.expected_backend:
            add_issue(payload, "error", f"expected backend {args.expected_backend} but run reports {backend}")

        if not metrics_rows:
            add_issue(payload, "error", "results/metrics.jsonl has no rows")
        else:
            payload["checks"].append({"metrics_row_count": len(metrics_rows)})

        machine_status = machine_state.get("status")
        payload["checks"].append({"machine_state_status": machine_status})
        if machine_status == "warn":
            add_issue(payload, "warning", "machine_state.json recorded warnings before benchmark start")
        elif machine_status == "error":
            add_issue(payload, "error", "machine_state.json recorded an error state")

        summary = suite_status_summary(metrics_rows)
        payload["checks"].append({"suite_summary": summary})

        for suite in REQUIRED_SUITES:
            if suite not in summary:
                add_issue(payload, "error", f"required suite missing from metrics.jsonl: {suite}")
                continue
            if "ok" not in summary[suite]["statuses"]:
                add_issue(payload, "warning", f"suite has no successful rows: {suite}")

        for suite in OPTIONAL_SUITES:
            if suite not in summary:
                add_issue(payload, "warning", f"optional suite missing from metrics.jsonl: {suite}")

        row_backends = sorted({row.get("gpu_backend") for row in metrics_rows if row.get("gpu_backend")})
        payload["checks"].append({"row_gpu_backends": row_backends})
        if not row_backends:
            add_issue(payload, "warning", "metrics rows do not include gpu_backend fields")
        elif len(row_backends) > 1:
            add_issue(payload, "error", f"metrics rows contain mixed gpu_backend values: {row_backends}")
        elif backend and row_backends[0] != backend:
            add_issue(payload, "error", f"metrics row backend {row_backends[0]} does not match meta.json backend {backend}")

        failed_rows = [row for row in metrics_rows if row.get("status") == "failed"]
        skipped_rows = [row for row in metrics_rows if row.get("status") == "skipped"]
        if failed_rows:
            add_issue(payload, "warning", f"{len(failed_rows)} failed metric row(s) present")
        if skipped_rows:
            add_issue(payload, "warning", f"{len(skipped_rows)} skipped metric row(s) present")

        if backend == "amd":
            infer_rows = [row for row in metrics_rows if row.get("suite") == "llm_infer" and row.get("status") == "ok"]
            if infer_rows:
                missing_power = [
                    row for row in infer_rows
                    if not row.get("power_sampler_available") and float(row.get("mean_power_w", 0.0) or 0.0) == 0.0
                ]
                if missing_power:
                    add_issue(
                        payload,
                        "warning",
                        "AMD llm_infer rows do not include comparable power metrics; treat power fields as unavailable",
                    )

    if payload["errors"]:
        payload["status"] = "error"
    elif payload["warnings"]:
        payload["status"] = "warn"

    print(f"[RUN-VALIDATE] status={payload['status']} run_dir={run_dir}")
    for warning in payload["warnings"]:
        print(f"[RUN-VALIDATE][WARN] {warning}")
    for error in payload["errors"]:
        print(f"[RUN-VALIDATE][ERROR] {error}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")

    if payload["status"] == "error":
        raise SystemExit(2)
    if payload["status"] == "warn":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
