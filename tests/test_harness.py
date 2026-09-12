import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
SCRIPT = ROOT / "harness.py"


class HarnessTest(unittest.TestCase):
    def test_inference_repeats_combine_varying_measurements(self):
        rows = [dict(suite="llm_infer", status="ok", batch_size=1,
                     repeat_index=i, repeat_count=3, requests=10*i,
                     latency_samples=i, batch_latency_ms_mean=100/i,
                     batch_latency_ms_p50=90/i, batch_latency_ms_p95=120/i,
                     batch_latency_per_item_proxy_ms_mean=100/i,
                     batch_latency_per_item_proxy_ms_p50=90/i,
                     batch_latency_per_item_proxy_ms_p95=120/i,
                     gen_tokens_per_s=20*i) for i in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            results = Path(tmpdir) / "results"
            results.mkdir()
            (results / "metrics.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
            subprocess.run([str(PYTHON), str(SCRIPT)], cwd=tmpdir, check=True,
                           capture_output=True)
            summaries = json.loads((results / "metrics_summary.json").read_text())
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["summary_count"], 3)
            self.assertEqual(summaries[0]["gen_tokens_per_s_mean"], 40)
            self.assertEqual(summaries[0]["gen_tokens_per_s_stdev"], 20)
            self.assertEqual(summaries[0]["requests_mean"], 20)

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
