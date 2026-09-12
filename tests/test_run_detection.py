import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]


class RunDetectionTest(unittest.TestCase):
    def run_dry(self, nvidia='', amd='', args=()):
        with tempfile.TemporaryDirectory(prefix='gpu-detection-') as directory:
            tools = Path(directory)
            nvidia_script = tools / 'nvidia-smi'
            nvidia_script.write_text('#!/usr/bin/env python3\nimport os\n'
                                    'print(os.environ.get("FAKE_NVIDIA_IDS", "").replace(",", "\\n"))\n')
            amd_script = tools / 'rocm-smi'
            amd_script.write_text('#!/usr/bin/env python3\nimport os, json\n'
                                 'print(json.dumps({"card"+i: {"Device ID": "0xab"} '
                                 'for i in os.environ.get("FAKE_AMD_IDS", "").split(",") if i}))\n')
            nvidia_script.chmod(0o755)
            amd_script.chmod(0o755)
            env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'],
                       FAKE_NVIDIA_IDS=nvidia, FAKE_AMD_IDS=amd)
            env.pop('CUDA_VISIBLE_DEVICES', None)
            env.pop('HIP_VISIBLE_DEVICES', None)
            source = ROOT / 'configs/auto.yaml'
            before = source.read_bytes()
            result = subprocess.run(['bash', str(ROOT / 'run_all.sh'), '--config', str(source),
                                     '--dry-run', *args], env=env, capture_output=True, text=True)
            self.assertEqual(source.read_bytes(), before)
            return result

    def test_amd_detection_resolves_runtime_without_installing_stack(self):
        result = self.run_dry(amd='2,3', args=('--baseline', '--smoke'))
        self.assertEqual(result.returncode, 0, result.stderr)
        config = yaml.safe_load(result.stdout)
        self.assertEqual(config['gpu_backend'], 'amd')
        self.assertEqual(config['gpu_include'], [2])
        self.assertEqual(config['blender']['backend'], 'hip')
        self.assertEqual(config['repeat'], 1)
        self.assertNotIn('[SETUP]', result.stdout)

    def test_nvidia_detection_and_cli_gpu_selection(self):
        result = self.run_dry(nvidia='0,2', args=('--gpus', '2'))
        self.assertEqual(result.returncode, 0, result.stderr)
        config = yaml.safe_load(result.stdout)
        self.assertEqual(config['gpu_backend'], 'nvidia')
        self.assertEqual(config['gpu_include'], [2])
        self.assertEqual(config['blender']['backend'], 'cuda')

    def test_mixed_vendor_needs_override_and_override_works(self):
        result = self.run_dry(nvidia='0', amd='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--backend', result.stderr)
        result = self.run_dry(nvidia='0', amd='1', args=('--backend', 'amd'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(yaml.safe_load(result.stdout)['gpu_backend'], 'amd')

    def test_missing_gpu_and_bad_cli_arguments_fail_early(self):
        result = self.run_dry()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('No responding', result.stderr)
        for args in (['--config'], ['--gpus'], ['--config', '/no/such/config.yaml']):
            result = subprocess.run(['bash', str(ROOT / 'run_all.sh'), *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)

    def test_samples_validate_and_only_backend_differs(self):
        configs = []
        for backend in ('auto', 'nvidia', 'amd'):
            path = ROOT / 'configs' / f'{backend}.yaml'
            result = subprocess.run([sys.executable, str(ROOT / 'validate_config.py'), '--config', str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = yaml.safe_load(path.read_text())
            self.assertEqual(config.pop('gpu_backend'), backend)
            configs.append(config)
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[1], configs[2])
