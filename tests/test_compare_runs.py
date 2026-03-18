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
                        "steps_per_sec_mean": 10.0,
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
                        "reqs_per_s_mean": 0.3,
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
                        "steps_per_sec_mean": 19.0,
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
                        "reqs_per_s_mean": 1.17,
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
            self.assertEqual(payload["executive_summary"]["group_counts"]["strict"], 1)

            llm_train_groups = payload["suites"]["llm_train"]["groups"]
            self.assertEqual(len(llm_train_groups), 1)
            self.assertTrue(llm_train_groups[0]["fully_comparable"])
            self.assertEqual(llm_train_groups[0]["quality"], "strict")

            llm_train_metrics = llm_train_groups[0]["metrics"]["tokens_per_sec_mean"]["rows"]
            self.assertEqual(llm_train_metrics[0]["value"], 20000.0)
            self.assertEqual(llm_train_metrics[1]["value"], 38000.0)
            self.assertEqual(
                llm_train_groups[0]["metrics"]["tokens_per_sec_mean"]["winner"],
                "NVIDIA",
            )
            self.assertEqual(llm_train_metrics[0]["delta_vs_first_run_pct"], -47.368)
            self.assertEqual(llm_train_metrics[1]["delta_vs_first_run_pct"], 0.0)

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
            self.assertIn("Best current pick for `llm_train`: `NVIDIA` based on `tokens_per_sec_mean` (strict)", markdown)
            self.assertIn("Best current pick for `llm_infer`: `NVIDIA` based on `gen_tokens_per_s_mean` (directional)", markdown)
            self.assertIn("Best current pick for `blender`: `NVIDIA` based on `time_s_mean` (directional)", markdown)
            self.assertIn("### Benchmark View", markdown)
            self.assertIn("Most metric wins: `NVIDIA`", markdown)
            self.assertIn("Strongest gain vs baseline: n/a", markdown)
            self.assertIn("Largest baseline lead: baseline stays ahead on `llm_train` / `tokens_per_sec_mean` (-47.368%)", markdown)
            self.assertIn("Per-suite highlights:", markdown)
            self.assertIn("llm_train: `AMD` on `tokens_per_sec_mean` (-47.368%)", markdown)
            self.assertIn("## Suite: llm_train", markdown)
            self.assertIn("## Suite: llm_infer", markdown)
            self.assertIn("## Suite: blender", markdown)
            self.assertIn("AMD", markdown)
            self.assertIn("NVIDIA", markdown)
            self.assertIn("Per-GPU Value", markdown)
            self.assertIn("backend differences may affect fairness", markdown)
            self.assertIn("Best run: `NVIDIA`", markdown)
            self.assertIn("### Group 1 (strict)", markdown)


if __name__ == "__main__":
    unittest.main()
