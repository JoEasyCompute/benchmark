import json
import importlib.util
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
SCRIPT = ROOT / "compare_runs.py"


def write_run(root: Path, name: str, meta: dict, config_text: str, summary_rows: list[dict]) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps(meta) + "\n")
    (run_dir / "effective_config.yaml").write_text(textwrap.dedent(config_text).strip() + "\n")
    (run_dir / "metrics_summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")
    return run_dir


class WorkloadIdentityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("compare_runs", SCRIPT)
        cls.compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.compare)

    def test_incompatible_identity_never_matches(self):
        suites = {
            "llm_train": {"hidden_size": 256, "n_layers": 4, "n_heads": 8,
                          "seed": 1234, "timing_method": "synchronized_v1"},
            "llm_infer": {"model_revision": "abc", "tokenizer_revision": "abc",
                          "prompt_len": 512, "output_len": 128,
                          "prompt_sha256": "abc", "seed": 1234,
                          "generation_mode": "greedy_fixed_length",
                          "timing_method": "synchronized_v1"},
            "sd_infer": {"model_revision": "abc", "prompt_sha256": "abc",
                         "negative_prompt_sha256": "def", "scheduler_name": "DDIM",
                         "guidance_scale": 7.5, "xformers_enabled": False,
                         "seed": 1234, "timing_method": "synchronized_v1"},
        }
        for suite, identity in suites.items():
            row = dict(suite=suite, **identity)
            for field in identity:
                with self.subTest(suite=suite, field=field):
                    changed = dict(row, **{field: "different"})
                    self.assertNotEqual(self.compare.row_key(row), self.compare.row_key(changed))
                    missing = dict(row)
                    del missing[field]
                    self.assertNotEqual(self.compare.row_key(row), self.compare.row_key(missing))

    def test_legacy_and_unresolved_identity_are_directional(self):
        for suite in ("llm_train", "llm_infer", "sd_infer", "blender"):
            with self.subTest(suite=suite):
                quality, notes = self.compare.compare_quality(suite, [{"suite": suite}], 1)
                self.assertEqual(quality, "directional")
                self.assertTrue(any("Missing workload identity" in note for note in notes))

    def test_null_revision_prevents_strict_comparison(self):
        row = dict(suite="llm_infer", backend="transformers", model="qwen",
                   dtype="float16", multi_gpu_mode="single", per_gpu_batch_size=1,
                   tensor_parallel=1, requested_prompt_len=512, prompt_len=512,
                   output_len=128, model_revision=None, tokenizer_revision="abc",
                   prompt_sha256="abc", generation_mode="greedy_fixed_length",
                   seed=1234, timing_method="synchronized_v1", gpu_count=1)
        quality, notes = self.compare.compare_quality("llm_infer", [row], 1)
        self.assertEqual(quality, "directional")
        self.assertIn("model_revision", notes[-1])
        row["model_revision"] = "abc"
        self.assertEqual(self.compare.compare_quality("llm_infer", [row], 1)[0], "strict")

    def test_complete_training_identity_can_be_strict(self):
        row = dict(suite="llm_train", dtype="bf16", seq_len=512, batch_size=4,
                   hidden_size=256, n_layers=4, n_heads=8, world_size=1,
                   seed=1234, timing_method="synchronized_v1", gpu_count=1)
        self.assertEqual(self.compare.compare_quality("llm_train", [row], 1)[0], "strict")


class CompareRunsTest(unittest.TestCase):
    def run_comparison(self, rows):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runs = [write_run(root, name, {"gpu_backend": name}, "{}", rows)
                    for name in ("amd", "nvidia")]
            result = subprocess.run(
                [str(PYTHON), str(SCRIPT), *map(str, runs),
                 "--out-dir", str(root / "report")], capture_output=True, text=True)
            payload = None
            if result.returncode == 0:
                payload = json.loads((root / "report/comparison.json").read_text())
            return result, payload

    def test_sd_emitted_dimensions_and_batch_sizes_stay_separate(self):
        rows = [dict(suite="sd_infer", status="ok", model="sd", steps=20,
                     hw=f"{size}x{size}", per_gpu_batch_size=batch,
                     dtype="float16", images_per_sec_mean=1, gpu_count=1)
                for size in (512, 1024) for batch in (1, 4)]
        result, payload = self.run_comparison(rows)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(payload["suites"]["sd_infer"]["groups"]), 4)

    def test_duplicate_workload_rows_are_rejected(self):
        row = dict(suite="llm_infer", status="ok", model="qwen",
                   batch_size=1, gen_tokens_per_s_mean=10, summary_count=1)
        result, _ = self.run_comparison([row, dict(row, gen_tokens_per_s_mean=20)])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Duplicate", result.stderr)

    def test_generates_markdown_and_json_for_matching_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            run_a = write_run(
                tmpdir_path,
                "run_a",
                {
                    "gpu_backend": "amd",
                    "python": "3.10.12",
                    "platform": "Linux",
                    "software_versions": {"torch": "2.8.0+rocm6.3", "transformers": "4.57.0"},
                },
                """
                gpu_backend: amd
                """,
                [
                    {
                        "suite": "llm_train",
                        "status": "ok",
                        "dtype": "bf16",
                        "seq_len": 512,
                        "batch_size": 4,
                        "world_size": 1,
                        "gpu_name": "AMD Radeon AI PRO R9700",
                        "gpu_count": 1,
                        "tokens_per_sec_mean": 20000.0,
                        "tokens_per_sec_min": 19000.0,
                        "tokens_per_sec_max": 21000.0,
                        "tokens_per_sec_stdev": 1000.0,
                        "steps_per_sec_mean": 10.0,
                        "steps_per_sec_min": 9.5,
                        "steps_per_sec_max": 10.5,
                        "steps_per_sec_stdev": 0.5,
                        "summary_count": 3,
                    },
                    {
                        "suite": "llm_infer",
                        "status": "ok",
                        "backend": "transformers",
                        "gpu_backend": "amd",
                        "model": "Qwen/Qwen3-8B",
                        "dtype": "float16",
                        "multi_gpu_mode": "replicated",
                        "per_gpu_batch_size": 1,
                        "tensor_parallel": 1,
                        "requested_prompt_len": 512,
                        "output_len": 128,
                        "gpu_name": "AMD Radeon AI PRO R9700",
                        "gpu_count": 2,
                        "gen_tokens_per_s_mean": 40.0,
                        "gen_tokens_per_s_min": 36.0,
                        "gen_tokens_per_s_max": 44.0,
                        "gen_tokens_per_s_stdev": 4.0,
                        "reqs_per_s_mean": 0.3,
                        "reqs_per_s_min": 0.27,
                        "reqs_per_s_max": 0.33,
                        "reqs_per_s_stdev": 0.03,
                        "summary_count": 3,
                    },
                    {
                        "suite": "blender",
                        "status": "ok",
                        "gpu_backend": "amd",
                        "scene": "BMW27.blend",
                        "mode": "single",
                        "backend": "HIP",
                        "gpu_name": "AMD Radeon AI PRO R9700",
                        "gpu_count": 2,
                        "time_s_mean": 11.8,
                    },
                ],
            )
            run_b = write_run(
                tmpdir_path,
                "run_b",
                {
                    "gpu_backend": "nvidia",
                    "python": "3.10.12",
                    "platform": "Linux",
                    "software_versions": {"torch": "2.8.0+cu128", "transformers": "4.57.0"},
                },
                """
                gpu_backend: nvidia
                """,
                [
                    {
                        "suite": "llm_train",
                        "status": "ok",
                        "dtype": "bf16",
                        "seq_len": 512,
                        "batch_size": 4,
                        "world_size": 1,
                        "gpu_name": "NVIDIA GeForce RTX 4090",
                        "gpu_count": 1,
                        "tokens_per_sec_mean": 38000.0,
                        "tokens_per_sec_min": 37000.0,
                        "tokens_per_sec_max": 39000.0,
                        "tokens_per_sec_stdev": 1000.0,
                        "steps_per_sec_mean": 19.0,
                        "steps_per_sec_min": 18.0,
                        "steps_per_sec_max": 20.0,
                        "steps_per_sec_stdev": 1.0,
                        "summary_count": 3,
                    },
                    {
                        "suite": "llm_infer",
                        "status": "ok",
                        "backend": "transformers",
                        "gpu_backend": "nvidia",
                        "model": "Qwen/Qwen3-8B",
                        "dtype": "float16",
                        "multi_gpu_mode": "replicated",
                        "per_gpu_batch_size": 1,
                        "tensor_parallel": 1,
                        "requested_prompt_len": 512,
                        "output_len": 128,
                        "gpu_name": "NVIDIA GeForce RTX 4090",
                        "gpu_count": 8,
                        "gen_tokens_per_s_mean": 150.0,
                        "gen_tokens_per_s_min": 145.0,
                        "gen_tokens_per_s_max": 155.0,
                        "gen_tokens_per_s_stdev": 5.0,
                        "reqs_per_s_mean": 1.17,
                        "reqs_per_s_min": 1.1,
                        "reqs_per_s_max": 1.24,
                        "reqs_per_s_stdev": 0.07,
                        "summary_count": 3,
                    },
                    {
                        "suite": "blender",
                        "status": "ok",
                        "gpu_backend": "nvidia",
                        "scene": "BMW27.blend",
                        "mode": "single",
                        "backend": "CUDA",
                        "gpu_name": "NVIDIA GeForce RTX 4090",
                        "gpu_count": 8,
                        "time_s_mean": 5.1,
                    },
                ],
            )

            out_dir = tmpdir_path / "report"
            result = subprocess.run(
                [
                    str(PYTHON),
                    str(SCRIPT),
                    "--label",
                    f"AMD={run_a}",
                    "--label",
                    f"NVIDIA={run_b}",
                    "--baseline",
                    "NVIDIA",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            json_out = out_dir / "comparison.json"
            md_out = out_dir / "comparison.md"
            self.assertTrue(json_out.exists())
            self.assertTrue(md_out.exists())

            payload = json.loads(json_out.read_text())
            self.assertEqual(payload["run_count"], 2)
            self.assertEqual(payload["baseline_label"], "NVIDIA")
            self.assertIn("llm_train", payload["suites"])
            self.assertIn("llm_infer", payload["suites"])
            self.assertIn("executive_summary", payload)
            self.assertEqual(payload["executive_summary"]["group_counts"]["strict"], 0)
            self.assertEqual(payload["executive_summary"]["group_counts"]["directional"], 3)

            llm_train_groups = payload["suites"]["llm_train"]["groups"]
            self.assertEqual(len(llm_train_groups), 1)
            self.assertTrue(llm_train_groups[0]["fully_comparable"])
            self.assertEqual(llm_train_groups[0]["quality"], "directional")
            self.assertTrue(any("different torch versions" in note for note in llm_train_groups[0]["notes"]))

            llm_train_metrics = llm_train_groups[0]["metrics"]["tokens_per_sec_mean"]["rows"]
            self.assertEqual(llm_train_metrics[0]["value"], 20000.0)
            self.assertEqual(llm_train_metrics[1]["value"], 38000.0)
            self.assertEqual(
                llm_train_groups[0]["metrics"]["tokens_per_sec_mean"]["winner"],
                "NVIDIA",
            )
            self.assertEqual(llm_train_metrics[0]["delta_vs_baseline_pct"], -47.368)
            self.assertEqual(llm_train_metrics[1]["delta_vs_baseline_pct"], 0.0)
            self.assertEqual(llm_train_metrics[0]["variability"]["summary_count"], 3)
            self.assertEqual(llm_train_metrics[0]["variability"]["min"], 19000.0)
            self.assertEqual(llm_train_metrics[0]["variability"]["max"], 21000.0)
            self.assertEqual(llm_train_metrics[0]["variability"]["stdev"], 1000.0)

            llm_infer_groups = payload["suites"]["llm_infer"]["groups"]
            self.assertEqual(len(llm_infer_groups), 1)
            self.assertEqual(llm_infer_groups[0]["quality"], "directional")
            self.assertIn("Per-GPU values are shown", llm_infer_groups[0]["notes"][0])
            infer_metric = llm_infer_groups[0]["metrics"]["gen_tokens_per_s_mean"]
            self.assertTrue(infer_metric["show_per_gpu"])
            self.assertEqual(infer_metric["winner"], "NVIDIA")
            self.assertEqual(infer_metric["rows"][0]["per_gpu_value"], 20.0)
            self.assertEqual(infer_metric["rows"][1]["per_gpu_value"], 18.75)

            blender_groups = payload["suites"]["blender"]["groups"]
            self.assertEqual(len(blender_groups), 1)
            self.assertTrue(blender_groups[0]["fully_comparable"])
            self.assertEqual(blender_groups[0]["quality"], "directional")
            self.assertEqual(blender_groups[0]["key"], {"scene": "BMW27.blend", "mode": "single"})
            self.assertIn("different render backends", blender_groups[0]["notes"][0])
            self.assertEqual(blender_groups[0]["metrics"]["time_s_mean"]["winner"], "NVIDIA")

            markdown = md_out.read_text()
            self.assertIn("# Run Comparison Report", markdown)
            self.assertIn("Baseline run: `NVIDIA`", markdown)
            self.assertIn("## Executive Summary", markdown)
            self.assertIn("### Decision View", markdown)
            self.assertIn("Best current pick for `llm_train`: `NVIDIA` based on `tokens_per_sec_mean` (directional)", markdown)
            self.assertIn("Best current pick for `llm_infer`: `NVIDIA` based on `reqs_per_s_mean` (directional)", markdown)
            self.assertIn("Best current pick for `blender`: `NVIDIA` based on `time_s_mean` (directional)", markdown)
            self.assertIn("### Benchmark View", markdown)
            self.assertIn("Most metric wins: `NVIDIA`", markdown)
            self.assertIn("Strongest gain vs baseline: n/a", markdown)
            self.assertIn("Largest baseline lead: n/a", markdown)
            self.assertIn("Per-suite highlights: n/a", markdown)
            self.assertIn("### Decision Confidence", markdown)
            self.assertIn("### Suite Takeaways", markdown)
            self.assertIn("### Risk Flags", markdown)
            self.assertIn("`llm_train`: best quality `directional`, groups=1, directional=1", markdown)
            self.assertIn("`llm_train`: `NVIDIA` leads on `tokens_per_sec_mean` with directional evidence, 1 directional group(s).", markdown)
            self.assertIn("llm_train: torch versions differ across runs.", markdown)
            self.assertIn("## Suite: llm_train", markdown)
            self.assertIn("## Suite: llm_infer", markdown)
            self.assertIn("## Suite: blender", markdown)
            self.assertIn("## Single-GPU View", markdown)
            self.assertIn("| llm_train | directional | 1 | NVIDIA | tokens_per_sec_mean |", markdown)
            self.assertIn("| blender | directional | 1 | NVIDIA | time_s_mean |", markdown)
            self.assertIn("## Comparability Summary", markdown)
            self.assertIn("Delta vs Baseline", markdown)
            self.assertIn("Repeat Variability", markdown)
            self.assertNotIn("Delta vs First Run", markdown)
            self.assertIn("different torch versions", markdown)
            self.assertIn("range 19000-21000, sd 1000, cv 5.00%, n=3", markdown)
            self.assertEqual({item["suite"] for item in payload["single_gpu_summary"]}, {"blender", "llm_train"})

    def test_uses_row_level_gpu_count_in_group_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            run_a = write_run(
                tmpdir_path,
                "run_a",
                {
                    "gpu_backend": "nvidia",
                    "python": "3.10.12",
                    "platform": "Linux",
                    "software_versions": {"torch": "2.8.0+cu128", "transformers": "4.57.0"},
                },
                """
                gpu_backend: nvidia
                """,
                [
                    {
                        "suite": "llm_train",
                        "status": "ok",
                        "dtype": "bf16",
                        "seq_len": 512,
                        "batch_size": 4,
                        "world_size": 1,
                        "gpu_name": "GPU A",
                        "gpu_count": 1,
                        "tokens_per_sec_mean": 100.0,
                        "tokens_per_sec_min": 95.0,
                        "tokens_per_sec_max": 105.0,
                        "tokens_per_sec_stdev": 5.0,
                        "steps_per_sec_mean": 10.0,
                        "steps_per_sec_min": 9.0,
                        "steps_per_sec_max": 11.0,
                        "steps_per_sec_stdev": 1.0,
                        "summary_count": 2,
                    },
                    {
                        "suite": "llm_infer",
                        "status": "ok",
                        "backend": "transformers",
                        "gpu_backend": "nvidia",
                        "model": "Qwen/Qwen3-8B",
                        "dtype": "float16",
                        "multi_gpu_mode": "replicated",
                        "per_gpu_batch_size": 1,
                        "tensor_parallel": 1,
                        "requested_prompt_len": 512,
                        "output_len": 128,
                        "gpu_name": "GPU A",
                        "gpu_count": 4,
                        "gen_tokens_per_s_mean": 40.0,
                        "gen_tokens_per_s_min": 39.0,
                        "gen_tokens_per_s_max": 41.0,
                        "gen_tokens_per_s_stdev": 1.0,
                        "reqs_per_s_mean": 1.0,
                        "reqs_per_s_min": 0.9,
                        "reqs_per_s_max": 1.1,
                        "reqs_per_s_stdev": 0.1,
                        "summary_count": 2,
                    },
                ],
            )
            run_b = write_run(
                tmpdir_path,
                "run_b",
                {
                    "gpu_backend": "nvidia",
                    "python": "3.10.12",
                    "platform": "Linux",
                    "software_versions": {"torch": "2.8.0+cu128", "transformers": "4.57.0"},
                },
                """
                gpu_backend: nvidia
                """,
                [
                    {
                        "suite": "llm_train",
                        "status": "ok",
                        "dtype": "bf16",
                        "seq_len": 512,
                        "batch_size": 4,
                        "world_size": 1,
                        "gpu_name": "GPU B",
                        "gpu_count": 2,
                        "tokens_per_sec_mean": 200.0,
                        "tokens_per_sec_min": 190.0,
                        "tokens_per_sec_max": 210.0,
                        "tokens_per_sec_stdev": 10.0,
                        "steps_per_sec_mean": 20.0,
                        "steps_per_sec_min": 19.0,
                        "steps_per_sec_max": 21.0,
                        "steps_per_sec_stdev": 1.0,
                        "summary_count": 2,
                    },
                    {
                        "suite": "llm_infer",
                        "status": "ok",
                        "backend": "transformers",
                        "gpu_backend": "nvidia",
                        "model": "Qwen/Qwen3-8B",
                        "dtype": "float16",
                        "multi_gpu_mode": "replicated",
                        "per_gpu_batch_size": 1,
                        "tensor_parallel": 1,
                        "requested_prompt_len": 512,
                        "output_len": 128,
                        "gpu_name": "GPU B",
                        "gpu_count": 8,
                        "gen_tokens_per_s_mean": 80.0,
                        "gen_tokens_per_s_min": 79.0,
                        "gen_tokens_per_s_max": 81.0,
                        "gen_tokens_per_s_stdev": 1.0,
                        "reqs_per_s_mean": 2.0,
                        "reqs_per_s_min": 1.9,
                        "reqs_per_s_max": 2.1,
                        "reqs_per_s_stdev": 0.1,
                        "summary_count": 2,
                    },
                ],
            )

            out_dir = tmpdir_path / "report"
            result = subprocess.run(
                [
                    str(PYTHON),
                    str(SCRIPT),
                    "--label",
                    f"A={run_a}",
                    "--label",
                    f"B={run_b}",
                    "--baseline",
                    "B",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((out_dir / "comparison.json").read_text())
            llm_train_rows = payload["suites"]["llm_train"]["groups"][0]["metrics"]["tokens_per_sec_mean"]["rows"]
            self.assertEqual(llm_train_rows[0]["gpu_count"], 1)
            self.assertEqual(llm_train_rows[1]["gpu_count"], 2)
            self.assertEqual(llm_train_rows[0]["max_gpu_count"], 4)
            self.assertEqual(llm_train_rows[1]["max_gpu_count"], 8)

            markdown = (out_dir / "comparison.md").read_text()
            self.assertIn("| A | 100 | 100 | -50.000% | 1 |", markdown)
            self.assertIn("| B | 200 | 100 | +0.000% | 2 |", markdown)
            self.assertIn("Per-GPU Value", markdown)
            self.assertIn("Repeat Variability", markdown)
            self.assertIn("range 95-105, sd 5, cv 5.00%, n=2", markdown)
            self.assertIn("Best run: `B`", markdown)
            self.assertIn("### Group 1 (directional)", markdown)
            self.assertIn("different gpu_count values (1, 2)", markdown)

    def test_suite_filter_tie_reporting_and_repeat_count_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            run_a = write_run(
                tmpdir_path,
                "run_a",
                {
                    "gpu_backend": "nvidia",
                    "python": "3.10.12",
                    "platform": "Linux",
                    "software_versions": {"torch": "2.8.0+cu128", "transformers": "4.57.0"},
                },
                """
                gpu_backend: nvidia
                """,
                [
                    {
                        "suite": "llm_train",
                        "status": "ok",
                        "dtype": "bf16",
                        "seq_len": 512,
                        "batch_size": 4,
                        "world_size": 1,
                        "gpu_name": "GPU A",
                        "gpu_count": 1,
                        "tokens_per_sec_mean": 100.0,
                        "tokens_per_sec_min": 99.0,
                        "tokens_per_sec_max": 101.0,
                        "tokens_per_sec_stdev": 1.0,
                        "steps_per_sec_mean": 10.0,
                        "steps_per_sec_min": 9.5,
                        "steps_per_sec_max": 10.5,
                        "steps_per_sec_stdev": 0.5,
                        "summary_count": 2,
                    },
                    {
                        "suite": "sd_infer",
                        "status": "ok",
                        "model": "sd",
                        "steps": 20,
                        "width": 512,
                        "height": 512,
                        "per_gpu_batch": 1,
                        "multi_gpu_mode": "single",
                        "dtype": "float16",
                        "gpu_name": "GPU A",
                        "gpu_backend": "nvidia",
                        "gpu_count": 1,
                        "images_per_sec_mean": 5.0,
                    },
                ],
            )
            run_b = write_run(
                tmpdir_path,
                "run_b",
                {
                    "gpu_backend": "nvidia",
                    "python": "3.10.12",
                    "platform": "Linux",
                    "software_versions": {"torch": "2.8.0+cu128", "transformers": "4.57.0"},
                },
                """
                gpu_backend: nvidia
                """,
                [
                    {
                        "suite": "llm_train",
                        "status": "ok",
                        "dtype": "bf16",
                        "seq_len": 512,
                        "batch_size": 4,
                        "world_size": 1,
                        "gpu_name": "GPU B",
                        "gpu_count": 1,
                        "tokens_per_sec_mean": 100.5,
                        "tokens_per_sec_min": 100.0,
                        "tokens_per_sec_max": 101.0,
                        "tokens_per_sec_stdev": 0.5,
                        "steps_per_sec_mean": 10.0,
                        "steps_per_sec_min": 9.8,
                        "steps_per_sec_max": 10.2,
                        "steps_per_sec_stdev": 0.2,
                        "summary_count": 3,
                    },
                    {
                        "suite": "sd_infer",
                        "status": "ok",
                        "model": "sd",
                        "steps": 20,
                        "width": 512,
                        "height": 512,
                        "per_gpu_batch": 1,
                        "multi_gpu_mode": "single",
                        "dtype": "float16",
                        "gpu_name": "GPU B",
                        "gpu_backend": "nvidia",
                        "gpu_count": 1,
                        "images_per_sec_mean": 6.0,
                    },
                ],
            )

            out_dir = tmpdir_path / "report"
            result = subprocess.run(
                [
                    str(PYTHON),
                    str(SCRIPT),
                    "--label",
                    f"A={run_a}",
                    "--label",
                    f"B={run_b}",
                    "--suites",
                    "llm_train",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((out_dir / "comparison.json").read_text())
            self.assertEqual(set(payload["suites"].keys()), {"llm_train"})
            self.assertEqual(payload["comparability_summary"][0]["suite"], "llm_train")
            self.assertIn("different repeat counts (2, 3)", payload["comparability_summary"][0]["issues"][0])
            metric = payload["suites"]["llm_train"]["groups"][0]["metrics"]["tokens_per_sec_mean"]
            self.assertEqual(metric["tied_winners"], ["B", "A"])

            markdown = (out_dir / "comparison.md").read_text()
            self.assertIn("## Comparability Summary", markdown)
            self.assertIn("| llm_train | directional | 1 |", markdown)
            self.assertIn("different repeat counts (2, 3)", markdown)
            self.assertIn("Best run: tie between `B, A`", markdown)
            self.assertIn("## Suite: llm_train", markdown)
            self.assertNotIn("## Suite: sd_infer", markdown)

    def test_surfaces_failed_rows_and_handles_lower_is_better_gains(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            run_a = write_run(
                tmpdir_path,
                "run_a",
                {
                    "gpu_backend": "nvidia",
                    "python": "3.10.12",
                    "platform": "Linux",
                    "software_versions": {"torch": "2.8.0+cu128", "transformers": "4.57.0"},
                },
                """
                gpu_backend: nvidia
                """,
                [
                    {
                        "suite": "blender",
                        "status": "ok",
                        "scene": "classroom.blend",
                        "scene_sha256": "fixed-scene-hash",
                        "blender_version": "4.2.18",
                        "render_settings_sha256": "fixed-settings-hash",
                        "timing_method": "process_elapsed_v1",
                        "mode": "single",
                        "backend": "CUDA",
                        "time_s_mean": 10.0,
                        "time_s_min": 9.9,
                        "time_s_max": 10.1,
                        "time_s_stdev": 0.1,
                        "summary_count": 3,
                    },
                    {
                        "suite": "llm_infer",
                        "status": "ok",
                        "backend": "transformers",
                        "model": "Qwen/Qwen3-8B",
                        "dtype": "float16",
                        "multi_gpu_mode": "replicated",
                        "per_gpu_batch_size": 1,
                        "tensor_parallel": 1,
                        "requested_prompt_len": 512,
                        "output_len": 128,
                        "gpu_count": 8,
                        "gen_tokens_per_s_mean": 100.0,
                        "reqs_per_s_mean": 1.0,
                        "summary_count": 1,
                    },
                ],
            )
            run_b = write_run(
                tmpdir_path,
                "run_b",
                {
                    "gpu_backend": "nvidia",
                    "python": "3.10.12",
                    "platform": "Linux",
                    "software_versions": {"torch": "2.8.0+cu128", "transformers": "4.57.0"},
                },
                """
                gpu_backend: nvidia
                """,
                [
                    {
                        "suite": "blender",
                        "status": "ok",
                        "scene": "classroom.blend",
                        "scene_sha256": "fixed-scene-hash",
                        "blender_version": "4.2.18",
                        "render_settings_sha256": "fixed-settings-hash",
                        "timing_method": "process_elapsed_v1",
                        "mode": "single",
                        "backend": "CUDA",
                        "time_s_mean": 20.0,
                        "time_s_min": 19.8,
                        "time_s_max": 20.2,
                        "time_s_stdev": 0.2,
                        "summary_count": 3,
                    },
                    {
                        "suite": "llm_infer",
                        "status": "failed",
                        "backend": "transformers",
                        "model": "Qwen/Qwen3-8B",
                        "dtype": "float16",
                        "multi_gpu_mode": "replicated",
                        "per_gpu_batch_size": 1,
                        "tensor_parallel": 1,
                        "requested_prompt_len": 512,
                        "output_len": 128,
                        "gpu_count": 8,
                        "gen_tokens_per_s_mean": 0.0,
                        "reqs_per_s_mean": 0.0,
                        "summary_count": 1,
                    },
                ],
            )

            out_dir = tmpdir_path / "report"
            result = subprocess.run(
                [
                    str(PYTHON),
                    str(SCRIPT),
                    "--label",
                    f"A={run_a}",
                    "--label",
                    f"B={run_b}",
                    "--baseline",
                    "A",
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((out_dir / "comparison.json").read_text())
            self.assertEqual(payload["executive_summary"]["strongest_gain"], None)
            self.assertEqual(payload["executive_summary"]["strongest_loss"]["suite"], "blender")
            self.assertEqual(
                payload["executive_summary"]["strongest_loss"]["preferred_delta_vs_baseline_pct"],
                -100.0,
            )
            self.assertTrue(any(item["suite"] == "llm_infer" and item["has_failures"] for item in payload["executive_summary"]["suite_confidence"]))
            self.assertTrue(any(item["suite"] == "llm_infer" and "partial" in item["text"] for item in payload["executive_summary"]["suite_takeaways"]))
            self.assertIn("llm_infer: at least one run failed a comparable row.", payload["executive_summary"]["risk_flags"])

            infer_notes = payload["suites"]["llm_infer"]["groups"][0]["notes"]
            self.assertIn("Run `B` has status `failed` for this comparable row.", infer_notes)

            markdown = (out_dir / "comparison.md").read_text()
            self.assertIn("Largest baseline lead: baseline stays ahead on `blender` / `time_s_mean` (worse than baseline by 100.000%)", markdown)
            self.assertIn("Strongest gain vs baseline: n/a", markdown)
            self.assertIn("`llm_infer`: best quality `partial`, groups=1, partial=1, failed rows present", markdown)
            self.assertIn("`llm_infer`: No decision-grade result because coverage is only partial.", markdown)
            self.assertIn("llm_infer: at least one run failed a comparable row.", markdown)
            self.assertIn("Run `B` has status `failed` for this comparable row.", markdown)


if __name__ == "__main__":
    unittest.main()
