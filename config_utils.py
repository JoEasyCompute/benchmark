#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from suite_config import smoke_optional_suites

try:
    import yaml
except ModuleNotFoundError:
    raise SystemExit("[CONFIG][ERROR] Missing dependency: PyYAML. Run env_setup.sh or activate the project venv.")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_path_value(data, dotted_path, default=None):
    current = data
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def write_effective_config(src_path, dst_path, smoke_mode, baseline=False,
                           resolve_hardware=False, backend_override=None, gpu_override=None):
    cfg = load_config(src_path)
    if resolve_hardware:
        from gpu_platform import resolve_config
        cfg = resolve_config(cfg, backend_override, gpu_override)
    cfg["smoke_mode"] = smoke_mode
    cfg["benchmark_profile"] = "single_gpu_baseline" if baseline else "general"

    if baseline:
        cfg["gpu_include"] = (cfg.get("gpu_include") or [0])[:1]
        cfg["repeat"] = 5
        llm_train = cfg.setdefault("llm_train", {})
        llm_train["world_sizes"] = [1]
        pair_selection = llm_train.get("pair_selection") or {}
        pair_selection["enabled"] = False
        llm_train["pair_selection"] = pair_selection
        llm_infer = cfg.setdefault("llm_infer", {})
        llm_infer["backend"] = "transformers"
        llm_infer["multi_gpu_mode"] = "single"
        llm_infer["tensor_parallel_sizes"] = [1]
        cfg.setdefault("sd_infer", {})["multi_gpu_mode"] = "single"

    if smoke_mode:
        cfg["repeat"] = 1

        llm_train = cfg.get("llm_train") or {}
        llm_train["world_sizes"] = [1]
        llm_train["steps"] = min(int(llm_train.get("steps", 1) or 1), 2)
        llm_train["batch_size"] = min(int(llm_train.get("batch_size", 1) or 1), 1)
        llm_train["seq_len"] = min(int(llm_train.get("seq_len", 128) or 128), 128)
        cfg["llm_train"] = llm_train

        llm_train_real = cfg.get("llm_train_real") or {}
        if llm_train_real.get("enabled", False):
            llm_train_real["steps"] = min(int(llm_train_real.get("steps", 1) or 1), 1)
            llm_train_real["batch_size"] = min(int(llm_train_real.get("batch_size", 1) or 1), 1)
            llm_train_real["seq_len"] = min(int(llm_train_real.get("seq_len", 128) or 128), 128)
            cfg["llm_train_real"] = llm_train_real

        llm_infer = cfg.get("llm_infer") or {}
        batch_sizes = llm_infer.get("batch_sizes") or [1]
        tp_sizes = llm_infer.get("tensor_parallel_sizes") or [1]
        llm_infer["batch_sizes"] = [int(batch_sizes[0])]
        llm_infer["tensor_parallel_sizes"] = [int(tp_sizes[0])]
        llm_infer["prompt_len"] = min(int(llm_infer.get("prompt_len", 64) or 64), 64)
        llm_infer["output_len"] = min(int(llm_infer.get("output_len", 32) or 32), 32)
        cfg["llm_infer"] = llm_infer

        sd_infer = cfg.get("sd_infer") or {}
        sizes = sd_infer.get("sizes") or [512]
        sd_infer["sizes"] = [int(sizes[0])]
        sd_infer["steps"] = min(int(sd_infer.get("steps", 4) or 4), 4)
        sd_infer["per_gpu_batch"] = 1
        cfg["sd_infer"] = sd_infer

        blender = cfg.get("blender") or {}
        scenes = blender.get("scenes") or []
        if scenes:
            blender["scenes"] = [scenes[0]]
        cfg["blender"] = blender

        smoke_optional_suites(cfg)
        if (cfg.get('llm_train_real') or {}).get('enabled'):
            cfg['llm_train_real']['warmup_steps'] = 1

    output_path = Path(dst_path)
    output_path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def parse_default(raw_default):
    if raw_default is None:
        return None
    return json.loads(raw_default)


def cmd_get(args):
    cfg = load_config(args.config)
    value = get_path_value(cfg, args.path, parse_default(args.default))

    if args.format == "bool-int":
        print(1 if value else 0)
    elif args.format == "json":
        print(json.dumps(value))
    elif args.format == "lines":
        for item in value or []:
            print(item)
    else:
        print("" if value is None else value)


def cmd_write_effective(args):
    write_effective_config(args.config, args.output, args.smoke, args.baseline,
                           args.resolve_hardware, args.backend, args.gpus)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    get_ap = sub.add_parser("get")
    get_ap.add_argument("--config", required=True)
    get_ap.add_argument("--path", required=True)
    get_ap.add_argument("--default")
    get_ap.add_argument("--format", choices=("text", "json", "lines", "bool-int"), default="text")
    get_ap.set_defaults(func=cmd_get)

    write_ap = sub.add_parser("write-effective")
    write_ap.add_argument("--config", required=True)
    write_ap.add_argument("--output", required=True)
    write_ap.add_argument("--smoke", action="store_true")
    write_ap.add_argument("--baseline", action="store_true")
    write_ap.add_argument('--resolve-hardware', action='store_true')
    write_ap.add_argument('--backend', choices=('auto', 'nvidia', 'amd'))
    write_ap.add_argument('--gpus', help='Comma-separated physical GPU indices')
    write_ap.set_defaults(func=cmd_write_effective)

    args = ap.parse_args()
    try:
        args.func(args)
    except ValueError as exc:
        ap.exit(2, f'[CONFIG][ERROR] {exc}\n')


if __name__ == "__main__":
    main()
