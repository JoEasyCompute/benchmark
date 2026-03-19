import json
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


class CompareRunsTest(unittest.TestCase):
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
            self.assertIn("## Suite: llm_train", markdown)
            self.assertIn("## Suite: llm_infer", markdown)
            self.assertIn("## Suite: blender", markdown)
            self.assertIn("## Comparability Summary", markdown)
            self.assertIn("Delta vs Baseline", markdown)
            self.assertIn("Repeat Variability", markdown)
            self.assertNotIn("Delta vs First Run", markdown)
            self.assertIn("different torch versions", markdown)
            self.assertIn("range 19000-21000, sd 1000, cv 5.00%, n=3", markdown)

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


if __name__ == "__main__":
    unittest.main()
