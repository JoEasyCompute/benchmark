#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    raise SystemExit("[ESTIMATE][ERROR] Missing dependency: PyYAML. Run env_setup.sh or activate the project venv.")

from gpu_platform import detect_backend, query_gpu_ids
from suite_config import optional_jobs


def visible_gpu_count():
    visible_env = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible = [x.strip() for x in visible_env.split(",") if x.strip()]
    if visible:
        return len(visible)
    try:
        return len(query_gpu_ids(detect_backend("auto")))
    except ValueError:
        return 0


def add_range(acc, low_s, likely_s, high_s):
    acc["min_s"] += low_s
    acc["likely_s"] += likely_s
    acc["max_s"] += high_s


def fmt_minutes(seconds):
    return round(seconds / 60.0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    repeat = int(cfg.get("repeat", 1) or 1)
    gpu_count = visible_gpu_count()
    sections = []
    totals = {"min_s": 0.0, "likely_s": 0.0, "max_s": 0.0}

    llm_train = cfg.get("llm_train", {})
    steps = int(llm_train.get("steps", 0) or 0)
    requested_world_sizes = [int(v) for v in (llm_train.get("world_sizes") or [1])]
    train_world_sizes = [ws for ws in requested_world_sizes if ws <= max(1, gpu_count)] or [1]
    train_per_repeat = {
        "min_s": max(30.0, steps * 0.8),
        "likely_s": max(60.0, steps * 1.5),
        "max_s": max(150.0, steps * 3.0),
    }
    train_total = {k: v * repeat * len(train_world_sizes) for k, v in train_per_repeat.items()}
    sections.append({"suite": "llm_train", "world_sizes": train_world_sizes, **train_total})
    add_range(totals, train_total["min_s"], train_total["likely_s"], train_total["max_s"])

    pair_selection = llm_train.get("pair_selection") or {}
    if pair_selection.get("enabled", False) and 2 in train_world_sizes and gpu_count >= 2:
        candidate_count = gpu_count * (gpu_count - 1) // 2
        candidate_limit = int(pair_selection.get("candidate_limit", 0) or 0)
        if candidate_limit > 0:
            candidate_count = min(candidate_count, candidate_limit)
        strategy = (pair_selection.get("strategy", "benchmark") or "benchmark").lower()
        probe_steps = int(pair_selection.get("probe_steps", 10) or 10)
        if strategy == "benchmark":
            pair_total = {
                "min_s": candidate_count * max(10.0, probe_steps * 0.8),
                "likely_s": candidate_count * max(20.0, probe_steps * 1.5),
                "max_s": candidate_count * max(45.0, probe_steps * 3.0),
            }
        else:
            pair_total = {"min_s": 1.0, "likely_s": 3.0, "max_s": 10.0}
        sections.append({
            "suite": "llm_train_pair_selection",
            "strategy": strategy,
            "candidate_pairs": candidate_count,
            "probe_steps": probe_steps if strategy == "benchmark" else 0,
            **pair_total,
        })
        add_range(totals, pair_total["min_s"], pair_total["likely_s"], pair_total["max_s"])

    llm_train_real = cfg.get("llm_train_real", {})
    if llm_train_real.get("enabled", False):
        real_steps = int(llm_train_real.get("steps", 0) or 0)
        real_per_repeat = {
            "min_s": max(60.0, real_steps * 3.0),
            "likely_s": max(180.0, real_steps * 8.0),
            "max_s": max(600.0, real_steps * 20.0),
        }
        real_total = {k: v * repeat for k, v in real_per_repeat.items()}
        sections.append({"suite": "llm_train_real", **real_total})
        add_range(totals, real_total["min_s"], real_total["likely_s"], real_total["max_s"])

    llm_infer = cfg.get("llm_infer", {})
    infer_backend = (llm_infer.get("backend", "transformers") or "transformers").lower()
    infer_multi_gpu_mode = (llm_infer.get("multi_gpu_mode", "single") or "single").lower()
    batch_sizes = list(llm_infer.get("batch_sizes", []))
    tp_sizes = list(llm_infer.get("tensor_parallel_sizes", []))
    if infer_backend == "vllm":
        combos = len(batch_sizes) * len(tp_sizes)
    else:
        combos = len(batch_sizes) * sum(1 for tp in tp_sizes if int(tp) == 1)
    infer_per_combo = {"min_s": 35.0, "likely_s": 55.0, "max_s": 95.0}
    infer_total = {k: combos * repeat * v for k, v in infer_per_combo.items()}
    sections.append({"suite": "llm_infer", "backend": infer_backend, "multi_gpu_mode": infer_multi_gpu_mode, "combos": combos, **infer_total})
    add_range(totals, infer_total["min_s"], infer_total["likely_s"], infer_total["max_s"])

    sd_infer = cfg.get("sd_infer", {})
    sd_mode = (sd_infer.get("multi_gpu_mode", "single") or "single").lower()
    sizes = list(sd_infer.get("sizes", []))
    sd_runs = len(sizes) * repeat
    sd_min = sd_likely = sd_max = 0.0
    for size in sizes:
        size = int(size)
        if size <= 512:
            low, likely, high = 35.0, 70.0, 140.0
        else:
            low, likely, high = 70.0, 140.0, 260.0
        sd_min += low * repeat
        sd_likely += likely * repeat
        sd_max += high * repeat
    sections.append(
        {
            "suite": "sd_infer",
            "runs": sd_runs,
            "visible_gpus": gpu_count,
            "multi_gpu_mode": "replicated" if sd_mode == "replicated" and gpu_count > 1 else "single",
            "min_s": sd_min,
            "likely_s": sd_likely,
            "max_s": sd_max,
        }
    )
    add_range(totals, sd_min, sd_likely, sd_max)

    blender = cfg.get("blender", {})
    blender_enabled = blender.get("enabled", True)
    scenes = blender.get("scenes") or ["BMW27.blend", "classroom.blend"]
    if blender_enabled:
        blend_min = blend_likely = blend_max = 0.0
        for scene in scenes:
            name = Path(scene).name.lower()
            if "classroom" in name:
                low, likely, high = 70.0, 150.0, 300.0
            else:
                low, likely, high = 25.0, 60.0, 150.0
            modes = 1 if cfg.get('benchmark_profile') == 'single_gpu_baseline' else 2
            blend_min += low * modes * repeat
            blend_likely += likely * modes * repeat
            blend_max += high * modes * repeat
        sections.append({"suite": "blender", "scenes": len(scenes), "min_s": blend_min, "likely_s": blend_likely, "max_s": blend_max})
        add_range(totals, blend_min, blend_likely, blend_max)

    for job in optional_jobs(cfg):
        suite = job['suite']
        if suite == 'llm_serve':
            duration = float(job['args'][job['args'].index('--duration') + 1])
            likely = duration + 30
        else:
            likely = 30 if suite == 'kernel_bench' else 60
        sections.append(dict(suite=suite, min_s=likely*.5*repeat, likely_s=likely*repeat, max_s=likely*4*repeat))
    sections = [s for s in sections if (cfg.get('llm_train') or {}).get('enabled', True)
                or s['suite'] not in ('llm_train', 'llm_train_pair_selection')]
    sections = [s for s in sections if (cfg.get(s['suite']) or {}).get('enabled', True)]
    totals = {key: sum(section[key] for section in sections) for key in ('min_s', 'likely_s', 'max_s')}
    payload = {
        'estimate_scope': 'heuristic; model downloads and serving request drains can exceed these ranges',
        "repeat": repeat,
        "visible_gpus": gpu_count,
        "sections": sections,
        "total": totals,
        "total_minutes": {
            "min": fmt_minutes(totals["min_s"]),
            "likely": fmt_minutes(totals["likely_s"]),
            "max": fmt_minutes(totals["max_s"]),
        },
    }

    print(
        f"[ESTIMATE] total runtime: {fmt_minutes(totals['min_s'])}-{fmt_minutes(totals['max_s'])} min "
        f"(likely ~{fmt_minutes(totals['likely_s'])} min)"
    )
    for section in sections:
        print(
            f"[ESTIMATE] {section['suite']}: {fmt_minutes(section['min_s'])}-"
            f"{fmt_minutes(section['max_s'])} min (likely ~{fmt_minutes(section['likely_s'])} min)"
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
