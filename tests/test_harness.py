import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
SCRIPT = ROOT / "harness.py"


class HarnessTest(unittest.TestCase):
    def test_consolidates_metrics_and_writes_repeat_summary(self):
        rows = [
            {
                "suite": "llm_train",
                "status": "ok",
                "dtype": "bf16",
                "seq_len": 512,
                "batch_size": 4,
                "repeat_index": 1,
                "repeat_count": 2,
                "steps_per_sec": 10.0,
                "tokens_per_sec": 100.0,
                "time_s": 5.0,
            },
            {
                "suite": "llm_train",
                "status": "ok",
                "dtype": "bf16",
                "seq_len": 512,
                "batch_size": 4,
                "repeat_index": 2,
                "repeat_count": 2,
                "steps_per_sec": 12.0,
                "tokens_per_sec": 120.0,
                "time_s": 4.0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            results_dir = tmpdir_path / "results"
            results_dir.mkdir()
            metrics_path = results_dir / "metrics.jsonl"
            metrics_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

            result = subprocess.run(
                [str(PYTHON), str(SCRIPT)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((results_dir / "metrics.csv").exists())
            self.assertTrue((results_dir / "metrics_summary.csv").exists())
            summary = json.loads((results_dir / "metrics_summary.json").read_text())
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary[0]["summary_count"], 2)
            self.assertEqual(summary[0]["repeat_indices"], "1,2")
            self.assertEqual(summary[0]["steps_per_sec_mean"], 11.0)
            self.assertEqual(summary[0]["tokens_per_sec_max"], 120.0)
            self.assertIn("Wrote results/metrics.csv with 2 rows", result.stdout)


if __name__ == "__main__":
    unittest.main()
