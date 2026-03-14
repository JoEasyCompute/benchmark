#!/usr/bin/env python3
import argparse
import sys

try:
    import yaml
except ModuleNotFoundError:
    print("[CONFIG][ERROR] Missing dependency: PyYAML. Run env_setup.sh or activate the project venv.", file=sys.stderr)
    sys.exit(1)


def is_pos_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def expect_keys(section_name, data, allowed, errors):
    unknown = sorted(set(data) - set(allowed))
    for key in unknown:
        errors.append(f"{section_name}: unsupported key '{key}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    errors = []
    expect_keys(
        "root",
        cfg,
        {"gpu_include", "repeat", "results_dir", "preflight", "llm_train", "llm_train_real", "llm_infer", "sd_infer", "blender"},
        errors,
    )

    if not is_pos_int(cfg.get("repeat", 1)):
        errors.append("repeat must be a positive integer")
    if not isinstance(cfg.get("results_dir", "results"), str) or not cfg.get("results_dir", "results").strip():
        errors.append("results_dir must be a non-empty string")

    preflight = cfg.get("preflight", {})
    if preflight is not None:
        if not isinstance(preflight, dict):
            errors.append("preflight must be a mapping if provided")
        else:
            expect_keys("preflight", preflight, {"machine_state_strict"}, errors)
            if "machine_state_strict" in preflight and not isinstance(preflight["machine_state_strict"], bool):
                errors.append("preflight.machine_state_strict must be true or false")

    llm_train = cfg.get("llm_train")
    if not isinstance(llm_train, dict):
        errors.append("llm_train section is required")
    else:
        expect_keys(
            "llm_train",
            llm_train,
            {"dtype", "hidden_size", "n_layers", "n_heads", "seq_len", "batch_size", "steps"},
            errors,
        )
        if llm_train.get("dtype") not in {"bf16", "fp16", "fp32"}:
            errors.append("llm_train.dtype must be one of bf16, fp16, fp32")
        for key in ("hidden_size", "n_layers", "n_heads", "seq_len", "batch_size", "steps"):
            if not is_pos_int(llm_train.get(key)):
                errors.append(f"llm_train.{key} must be a positive integer")
        if is_pos_int(llm_train.get("hidden_size")) and is_pos_int(llm_train.get("n_heads")):
            if llm_train["hidden_size"] % llm_train["n_heads"] != 0:
                errors.append("llm_train.hidden_size must be divisible by llm_train.n_heads")

    llm_train_real = cfg.get("llm_train_real", {})
    if llm_train_real is not None:
        if not isinstance(llm_train_real, dict):
            errors.append("llm_train_real must be a mapping if provided")
        else:
            expect_keys(
                "llm_train_real",
                llm_train_real,
                {"enabled", "model", "dtype", "seq_len", "batch_size", "steps"},
                errors,
            )
            if "enabled" in llm_train_real and not isinstance(llm_train_real["enabled"], bool):
                errors.append("llm_train_real.enabled must be true or false")
            if llm_train_real.get("enabled", False):
                if not isinstance(llm_train_real.get("model"), str) or not llm_train_real["model"].strip():
                    errors.append("llm_train_real.model must be a non-empty string when enabled")
                if llm_train_real.get("dtype") not in {"bf16", "fp16", "fp32"}:
                    errors.append("llm_train_real.dtype must be one of bf16, fp16, fp32")
                for key in ("seq_len", "batch_size", "steps"):
                    if not is_pos_int(llm_train_real.get(key)):
                        errors.append(f"llm_train_real.{key} must be a positive integer when enabled")

    llm_infer = cfg.get("llm_infer")
    if not isinstance(llm_infer, dict):
        errors.append("llm_infer section is required")
    else:
        expect_keys(
            "llm_infer",
            llm_infer,
            {"model", "dtype", "prompt_len", "output_len", "batch_sizes", "tensor_parallel_sizes"},
            errors,
        )
        if not isinstance(llm_infer.get("model"), str) or not llm_infer["model"].strip():
            errors.append("llm_infer.model must be a non-empty string")
        if llm_infer.get("dtype") not in {"float16", "half", "bfloat16", "float32", "float", "auto"}:
            errors.append("llm_infer.dtype must be one of auto, float16, half, bfloat16, float32, float")
        for key in ("prompt_len", "output_len"):
            if not is_pos_int(llm_infer.get(key)):
                errors.append(f"llm_infer.{key} must be a positive integer")
        for key in ("batch_sizes", "tensor_parallel_sizes"):
            values = llm_infer.get(key)
            if not isinstance(values, list) or not values or not all(is_pos_int(v) for v in values):
                errors.append(f"llm_infer.{key} must be a non-empty list of positive integers")

    sd_infer = cfg.get("sd_infer")
    if not isinstance(sd_infer, dict):
        errors.append("sd_infer section is required")
    else:
        expect_keys(
            "sd_infer",
            sd_infer,
            {"model", "steps", "sizes", "per_gpu_batch", "multi_gpu_mode", "emit_worker_rows"},
            errors,
        )
        if not isinstance(sd_infer.get("model"), str) or not sd_infer["model"].strip():
            errors.append("sd_infer.model must be a non-empty string")
        if not is_pos_int(sd_infer.get("steps")):
            errors.append("sd_infer.steps must be a positive integer")
        sizes = sd_infer.get("sizes")
        if not isinstance(sizes, list) or not sizes or not all(is_pos_int(v) for v in sizes):
            errors.append("sd_infer.sizes must be a non-empty list of positive integers")
        if not is_pos_int(sd_infer.get("per_gpu_batch")):
            errors.append("sd_infer.per_gpu_batch must be a positive integer")
        if sd_infer.get("multi_gpu_mode", "single") not in {"single", "replicated"}:
            errors.append("sd_infer.multi_gpu_mode must be 'single' or 'replicated'")
        if "emit_worker_rows" in sd_infer and not isinstance(sd_infer["emit_worker_rows"], bool):
            errors.append("sd_infer.emit_worker_rows must be true or false")

    blender = cfg.get("blender", {})
    if not isinstance(blender, dict):
        errors.append("blender must be a mapping if provided")
    else:
        expect_keys("blender", blender, {"enabled", "scenes"}, errors)
        if "enabled" in blender and not isinstance(blender["enabled"], bool):
            errors.append("blender.enabled must be true or false")
        scenes = blender.get("scenes", [])
        if scenes is not None and (not isinstance(scenes, list) or not all(isinstance(x, str) for x in scenes)):
            errors.append("blender.scenes must be a list of strings")

    if errors:
        for err in errors:
            print(f"[CONFIG][ERROR] {err}", file=sys.stderr)
        sys.exit(1)

    print("[CONFIG] config.yaml validation passed")


if __name__ == "__main__":
    main()
