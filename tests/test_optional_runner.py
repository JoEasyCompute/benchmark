import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from run_optional_suites import execute_jobs


class OptionalRunnerTest(unittest.TestCase):
    def test_repeats_every_combination_and_annotations_preserve_backend(self):
        calls = []
        def run(command, **kwargs):
            calls.append(command)
            path = Path(command[command.index('--metrics-path') + 1])
            path.write_text(json.dumps({'suite': 'vision_infer', 'status': 'ok',
                                        'gpu_backend': 'amd', 'images_per_sec': 10}) + '\n')
            return subprocess.CompletedProcess(command, 0)
        with tempfile.TemporaryDirectory() as directory:
            cfg = {'repeat': 3, 'gpu_backend': 'amd', 'vision_infer': {
                'enabled': True, 'sizes': [224, 256], 'batch_sizes': [1, 2]}}
            self.assertEqual(execute_jobs(cfg, Path(directory), runner=run), 0)
            rows = [json.loads(line) for line in (Path(directory) / 'results/metrics.jsonl').read_text().splitlines()]
            self.assertEqual(len(calls), 12)
            self.assertEqual(len(rows), 12)
            self.assertEqual({r['repeat_index'] for r in rows}, {1, 2, 3})
            self.assertTrue(all(r['repeat_count'] == 3 and r['gpu_backend'] == 'amd' for r in rows))

    def test_crash_without_metrics_is_structured_and_failure_exit(self):
        def run(command, **kwargs):
            return subprocess.CompletedProcess(command, 1)
        with tempfile.TemporaryDirectory() as directory:
            cfg = {'kernel_bench': {'enabled': True, 'cases': ['gemm']}}
            self.assertEqual(execute_jobs(cfg, Path(directory), runner=run), 1)
            row = json.loads((Path(directory) / 'results/metrics.jsonl').read_text())
            self.assertEqual(row['status'], 'failed')
            self.assertEqual(row['suite'], 'kernel_bench')
