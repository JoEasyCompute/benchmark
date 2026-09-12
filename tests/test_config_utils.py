import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import yaml

from config_utils import load_config, write_effective_config

ROOT = Path(__file__).resolve().parents[1]


class EffectiveConfigTests(unittest.TestCase):
    def effective(self, baseline=False, smoke=False, include=None):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.yaml'
            output = Path(directory) / 'effective.yaml'
            cfg = load_config(ROOT / 'config.yaml')
            cfg['gpu_include'] = include or []
            source.write_text(yaml.safe_dump(cfg))
            write_effective_config(source, output, smoke, baseline)
            self.assertEqual(load_config(source), cfg)
            return load_config(output)

    def test_baseline_selects_first_configured_gpu_and_single_gpu_workloads(self):
        cfg = self.effective(baseline=True, include=[3, 1])
        self.assertEqual(cfg['benchmark_profile'], 'single_gpu_baseline')
        self.assertEqual(cfg['gpu_include'], [3])
        self.assertEqual(cfg['repeat'], 5)
        self.assertEqual(cfg['llm_train']['world_sizes'], [1])
        self.assertFalse(cfg['llm_train']['pair_selection']['enabled'])
        self.assertEqual(cfg['llm_infer']['backend'], 'transformers')
        self.assertEqual(cfg['llm_infer']['multi_gpu_mode'], 'single')
        self.assertEqual(cfg['llm_infer']['tensor_parallel_sizes'], [1])
        self.assertEqual(cfg['sd_infer']['multi_gpu_mode'], 'single')

    def test_baseline_smoke_defaults_to_gpu_zero_and_one_repeat(self):
        cfg = self.effective(baseline=True, smoke=True)
        self.assertEqual(cfg['gpu_include'], [0])
        self.assertEqual(cfg['repeat'], 1)
        self.assertEqual(cfg['llm_infer']['tensor_parallel_sizes'], [1])
        self.assertTrue(cfg['smoke_mode'])

    def test_general_preserves_gpu_selection(self):
        cfg = self.effective(include=[3, 1])
        self.assertEqual(cfg['benchmark_profile'], 'general')
        self.assertEqual(cfg['gpu_include'], [3, 1])

    def test_hardware_resolution_selects_first_detected_card_for_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'effective.yaml'
            with patch('gpu_platform.query_gpu_ids', side_effect=lambda b: ['2', '4'] if b == 'amd' else []), \
                    patch.dict('os.environ', {}, clear=True):
                write_effective_config(ROOT / 'config.yaml', output, False, True, resolve_hardware=True)
            cfg = load_config(output)
            self.assertEqual(cfg['gpu_backend'], 'amd')
            self.assertEqual(cfg['gpu_include'], [2])
            self.assertEqual(cfg['blender']['backend'], 'hip')

    def test_cli_baseline_effective_config_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'effective.yaml'
            subprocess.run([sys.executable, str(ROOT / 'config_utils.py'),
                            'write-effective', '--config', str(ROOT / 'config.yaml'),
                            '--output', str(output), '--baseline', '--smoke'], check=True)
            subprocess.run([sys.executable, str(ROOT / 'validate_config.py'),
                            '--config', str(output)], check=True)
            self.assertEqual(load_config(output)['repeat'], 1)

    def test_shell_baseline_rejects_absent_gpu(self):
        script = (ROOT / 'run_all.sh').read_text()
        start = script.index('if [[ "$BASELINE_MODE" == "1" ]]; then\n  BASELINE_GPU_FOUND=0')
        end = script.index('if [[ "${#SELECTED_GPU_IDS[@]}" -gt 0 ]]; then', start)
        check = script[start:end]
        for gpu, expected in (("3", 0), ("2", 1)):
            with self.subTest(gpu=gpu):
                result = subprocess.run(
                    ['bash', '-c', 'BASELINE_MODE=1; ALL_GPU_IDS=(0 3); '
                     + 'SELECTED_GPU_IDS=(' + gpu + '); ' + check],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, expected)
                if expected:
                    self.assertIn('is not present', result.stderr)
