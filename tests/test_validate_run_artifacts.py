import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validate_run_artifacts.py"


class ValidateRunArtifactsTest(unittest.TestCase):
    def test_warns_for_missing_optional_suite_and_amd_power_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "results" / "sample_run"
            results_dir = run_dir / "results"
            results_dir.mkdir(parents=True)

            (run_dir / "meta.json").write_text(json.dumps({"gpu_backend": "amd"}) + "\n")
            (run_dir / "machine_state.json").write_text(json.dumps({"status": "ok"}) + "\n")
            (run_dir / "runtime_estimate.json").write_text(json.dumps({"status": "ok"}) + "\n")

            rows = [
                {"suite": "llm_train", "status": "ok", "gpu_backend": "amd", "gpu_count": 1},
                {
                    "suite": "llm_infer",
                    "status": "ok",
                    "gpu_backend": "amd",
                    "gpu_count": 1,
                    "power_sampler_available": False,
                    "mean_power_w": 0.0,
                },
                {"suite": "sd_infer", "status": "ok", "gpu_backend": "amd", "gpu_count": 1},
            ]
            (results_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(run_dir)],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("optional suite missing from metrics.jsonl: blender", result.stdout)
            self.assertIn("AMD llm_infer rows do not include comparable power metrics", result.stdout)

    def test_errors_when_required_suite_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "results" / "sample_run"
            results_dir = run_dir / "results"
            results_dir.mkdir(parents=True)

            (run_dir / "meta.json").write_text(json.dumps({"gpu_backend": "nvidia"}) + "\n")
            (run_dir / "machine_state.json").write_text(json.dumps({"status": "ok"}) + "\n")
            (run_dir / "runtime_estimate.json").write_text(json.dumps({"status": "ok"}) + "\n")
            (results_dir / "metrics.jsonl").write_text(
                json.dumps({"suite": "llm_train", "status": "ok", "gpu_backend": "nvidia"}) + "\n"
            )

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(run_dir), "--expected-backend", "nvidia"],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("required suite missing from metrics.jsonl: llm_infer", result.stdout)
            self.assertIn("required suite missing from metrics.jsonl: sd_infer", result.stdout)


if __name__ == "__main__":
    unittest.main()
