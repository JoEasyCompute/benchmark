#!/usr/bin/env python3
"""Select the best two-GPU pair for llm_train world_size=2.

The selector can either rank pairs from `nvidia-smi topo -m` or run a short
actual llm_train probe for every candidate pair and choose the highest
`tokens_per_sec` result. It is intentionally stdlib-first except for PyYAML,
which the benchmark stack already requires.
"""

import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable
from gpu_platform import visibility_mask

try:
    import yaml
except ModuleNotFoundError:
    raise SystemExit("[PAIR][ERROR] Missing dependency: PyYAML. Run env_setup.sh or activate the project venv.")


TOPOLOGY_COSTS = {
    "NV": 0,
    "PIX": 10,
    "PXB": 20,
    "PHB": 30,
    "NODE": 40,
    "SYS": 50,
    "X": 999,
}


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_gpu_csv(raw: str) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def candidate_pairs(gpu_ids: Iterable[str]) -> list[tuple[str, str]]:
    return list(itertools.combinations([str(g) for g in gpu_ids], 2))


def normalize_topology_token(token: str) -> str:
    token = (token or "").strip().upper()
    if token.startswith("NV"):
        return "NV"
    return token


def parse_nvidia_topology(raw: str) -> dict[tuple[str, str], str]:
    """Parse `nvidia-smi topo -m` into pair -> link token.

    Expected rows look like:
      GPU0    X      PIX    SYS
      GPU1    PIX    X      SYS
    """
    lines = [line.rstrip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return {}

    header_tokens = lines[0].split()
    gpu_headers = [tok for tok in header_tokens if tok.startswith("GPU")]
    topology = {}

    for line in lines[1:]:
        parts = line.split()
        if not parts or not parts[0].startswith("GPU"):
            continue
        row_gpu = parts[0].replace("GPU", "")
        links = parts[1 : 1 + len(gpu_headers)]
        for col_header, link in zip(gpu_headers, links):
            col_gpu = col_header.replace("GPU", "")
            if row_gpu == col_gpu:
                continue
            topology[tuple(sorted((row_gpu, col_gpu), key=int))] = normalize_topology_token(link)
    return topology


def topology_link_cost(link: str | None) -> int:
    token = normalize_topology_token(link or "SYS")
    return TOPOLOGY_COSTS.get(token, 100)


def query_nvidia_topology() -> dict[tuple[str, str], str]:
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        raw = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {}
    return parse_nvidia_topology(raw)


def rank_pairs_by_topology(pairs: list[tuple[str, str]], topology: dict[tuple[str, str], str]) -> list[tuple[str, str]]:
    def pair_key(pair: tuple[str, str]):
        sorted_pair = tuple(sorted(pair, key=int))
        return (topology_link_cost(topology.get(sorted_pair)), int(pair[0]), int(pair[1]))

    return sorted(pairs, key=pair_key)


def write_probe_config(src_config: Path, dst_config: Path, probe_steps: int) -> None:
    cfg = load_config(src_config)
    llm_train = cfg.get("llm_train") or {}
    llm_train["world_sizes"] = [2]
    llm_train["steps"] = int(probe_steps)
    cfg["llm_train"] = llm_train
    dst_config.write_text(yaml.safe_dump(cfg, sort_keys=False))


def parse_probe_metric(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    for line in reversed(metrics_path.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("suite") == "llm_train" and row.get("status") == "ok":
            return row
    return None


def run_probe(
    pair: tuple[str, str],
    config_path: Path,
    bench_script: Path,
    visible_env_var: str,
    log_dir: Path,
    python_bin: str,
    probe_steps: int,
) -> dict:
    pair_csv = ",".join(pair)
    pair_name = f"gpu_{pair[0]}_{pair[1]}"
    probe_dir = log_dir / pair_name
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_config = probe_dir / "probe_config.yaml"
    write_probe_config(config_path, probe_config, probe_steps)

    cmd = [
        python_bin,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        str(bench_script),
        "--config",
        str(probe_config),
    ]
    env = os.environ.copy()
    env[visible_env_var] = visibility_mask('amd' if visible_env_var == 'HIP_VISIBLE_DEVICES' else 'nvidia', pair)
    log_path = log_dir / f"{pair_name}.log"

    with log_path.open("w") as log_file:
        proc = subprocess.run(cmd, cwd=probe_dir, env=env, text=True, stdout=log_file, stderr=subprocess.STDOUT, check=False)

    metric = parse_probe_metric(probe_dir / "results" / "metrics.jsonl")
    payload = {
        "pair": list(pair),
        "pair_csv": pair_csv,
        "returncode": proc.returncode,
        "log_path": str(log_path),
        "probe_dir": str(probe_dir),
        "status": "ok" if proc.returncode == 0 and metric else "failed",
    }
    if metric:
        payload.update(
            {
                "steps_per_sec": metric.get("steps_per_sec"),
                "tokens_per_sec": metric.get("tokens_per_sec"),
                "time_s": metric.get("time_s"),
            }
        )
    return payload


def best_probe_result(results: list[dict]) -> dict | None:
    ok = [r for r in results if r.get("status") == "ok" and isinstance(r.get("tokens_per_sec"), (int, float))]
    if not ok:
        return None
    return max(ok, key=lambda r: (float(r["tokens_per_sec"]), -int(r["pair"][0]), -int(r["pair"][1])))


def select_pair(args) -> dict:
    cfg = load_config(args.config)
    pair_cfg = (cfg.get("llm_train") or {}).get("pair_selection") or {}
    strategy = args.strategy or pair_cfg.get("strategy", "benchmark")
    probe_steps = int(args.probe_steps or pair_cfg.get("probe_steps", 10))
    candidate_limit = int(args.candidate_limit if args.candidate_limit is not None else pair_cfg.get("candidate_limit", 0) or 0)

    gpu_ids = parse_gpu_csv(args.gpu_ids)
    pairs = candidate_pairs(gpu_ids)
    if not pairs:
        raise SystemExit("[PAIR][ERROR] Need at least two GPU ids to select a pair")

    topology = query_nvidia_topology() if args.backend == "nvidia" else {}
    ranked_pairs = rank_pairs_by_topology(pairs, topology) if topology else pairs
    if candidate_limit > 0:
        ranked_pairs = ranked_pairs[:candidate_limit]

    payload = {
        "schema_version": 1,
        "strategy": strategy,
        "backend": args.backend,
        "visible_env_var": args.visible_env_var,
        "gpu_ids": gpu_ids,
        "probe_steps": probe_steps,
        "candidate_limit": candidate_limit,
        "topology": {"-".join(k): v for k, v in topology.items()},
        "candidates": [{"pair": list(pair), "pair_csv": ",".join(pair), "topology_link": topology.get(tuple(sorted(pair, key=int)))} for pair in ranked_pairs],
    }

    if strategy == "first":
        selected = ranked_pairs[0]
        payload.update({"selected_pair": list(selected), "selected_pair_csv": ",".join(selected), "selection_reason": "first_candidate"})
        return payload

    if strategy == "topology":
        selected = ranked_pairs[0]
        payload.update({"selected_pair": list(selected), "selected_pair_csv": ",".join(selected), "selection_reason": "best_topology"})
        return payload

    if strategy != "benchmark":
        payload["warning"] = f"unknown strategy {strategy!r}; falling back to benchmark"

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for pair in ranked_pairs:
        print(f"[PAIR] probing {','.join(pair)}", flush=True)
        results.append(run_probe(pair, Path(args.config), Path(args.bench_script), args.visible_env_var, log_dir, args.python_bin, probe_steps))

    payload["probe_results"] = results
    best = best_probe_result(results)
    if best:
        payload.update(
            {
                "selected_pair": best["pair"],
                "selected_pair_csv": best["pair_csv"],
                "selection_reason": "highest_probe_tokens_per_sec",
                "selected_tokens_per_sec": best.get("tokens_per_sec"),
                "selected_steps_per_sec": best.get("steps_per_sec"),
            }
        )
    else:
        fallback = ranked_pairs[0]
        payload.update(
            {
                "selected_pair": list(fallback),
                "selected_pair_csv": ",".join(fallback),
                "selection_reason": "fallback_first_candidate_no_successful_probe",
                "warning": "no successful benchmark probes; using first candidate",
            }
        )
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu-ids", required=True, help="Comma-separated physical GPU ids available to the run")
    ap.add_argument("--backend", default="nvidia")
    ap.add_argument("--visible-env-var", default="CUDA_VISIBLE_DEVICES")
    ap.add_argument("--bench-script", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--strategy", choices=("benchmark", "topology", "first"))
    ap.add_argument("--probe-steps", type=int)
    ap.add_argument("--candidate-limit", type=int)
    args = ap.parse_args()

    try:
        payload = select_pair(args)
    except Exception as exc:
        # Be conservative: write a fallback selection so run_all can continue.
        gpu_ids = parse_gpu_csv(args.gpu_ids)
        fallback = gpu_ids[:2]
        payload = {
            "schema_version": 1,
            "strategy": "fallback_after_selector_error",
            "backend": args.backend,
            "gpu_ids": gpu_ids,
            "selected_pair": fallback,
            "selected_pair_csv": ",".join(fallback),
            "selection_reason": "selector_error_fallback_first_two",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"[PAIR][WARN] selector failed: {exc}; using {payload['selected_pair_csv']}", file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(payload.get("selected_pair_csv", ""))


if __name__ == "__main__":
    main()
