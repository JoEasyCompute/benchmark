"""Exercise actual optional CLIs and the reporting pipeline on a dependency-free host."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml
from suite_config import smoke_optional_suites

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(importlib.util.find_spec('torch') is not None, 'dependency-free CLI smoke only')
class ExpandedSmokeTest(unittest.TestCase):
    def test_config_to_clis_to_summary_to_report(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
            cfg['repeat'] = 1
            for suite in ('vision_infer', 'kernel_bench', 'llm_serve'):
                cfg[suite]['enabled'] = True
            smoke_optional_suites(cfg)
            config = base / 'effective_config.yaml'
            config.write_text(yaml.safe_dump(cfg))
            subprocess.run([sys.executable, str(ROOT / 'validate_config.py'), '--config', str(config)],
                           check=True, capture_output=True)
            runs = []
            for label in ('A', 'B'):
                run = base / label
                result = subprocess.run([sys.executable, str(ROOT / 'run_optional_suites.py'),
                    '--config', str(config), '--run-dir', str(run)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                rows = [json.loads(line) for line in (run / 'results/metrics.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 5)
                self.assertTrue(all(row['status'] == 'skipped' and row['repeat_index'] == 1 for row in rows))
                subprocess.run([sys.executable, str(ROOT / 'harness.py')], cwd=run,
                               check=True, capture_output=True)
                shutil.copyfile(run / 'results/metrics_summary.json', run / 'metrics_summary.json')
                shutil.copyfile(config, run / 'effective_config.yaml')
                (run / 'meta.json').write_text(json.dumps({'gpu_backend': 'unknown'}))
                runs.append(run)
            result = subprocess.run([sys.executable, str(ROOT / 'compare_runs.py'), *map(str, runs),
                                     '--out-dir', str(base / 'report')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((base / 'report/comparison.json').read_text())
            self.assertEqual(set(report['suites']), {'vision_infer', 'kernel_bench', 'llm_serve'})
            self.assertEqual(report['executive_summary']['suite_decisions'], [])
