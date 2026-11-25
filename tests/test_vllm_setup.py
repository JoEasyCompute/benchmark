from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import setup_vllm


class VllmSetupTest(unittest.TestCase):
    def test_valid_environment_is_not_reinstalled(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / 'pyvenv.cfg').touch()
            with patch.object(setup_vllm, 'packages_ready', return_value=True), \
                    patch.object(setup_vllm.subprocess, 'run') as run:
                setup_vllm.ensure_packages(target, '0.34.0')
            run.assert_not_called()

    def test_managed_environment_repairs_and_rechecks(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / 'pyvenv.cfg').touch()
            ready = Mock(side_effect=[False, True])
            with patch.object(setup_vllm, 'packages_ready', ready), \
                    patch.object(setup_vllm.subprocess, 'run') as run:
                setup_vllm.ensure_packages(target, '0.34.0')
            self.assertEqual(ready.call_count, 2)
            self.assertIn('vllm==0.34.0', run.call_args.args[0])

    def test_external_environment_is_never_modified(self):
        with patch.object(setup_vllm, 'packages_ready', return_value=False), \
                patch.object(setup_vllm.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'VLLM_PYTHON'):
                setup_vllm.ensure_packages(None, '0.34.0', python_override='/external/bin/python')
        run.assert_not_called()

    def test_cuda_13_rejects_incompatible_driver(self):
        with self.assertRaisesRegex(ValueError, '580'):
            setup_vllm.check_host({'gpus': [{'index': '0', 'driver_version': '575.64.03'}]}, '0.34.0')

    def test_rocm_installs_the_exact_vendor_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / 'pyvenv.cfg').touch()
            backend = {'backend': 'amd', 'gpus': [{'index': '0'}]}
            setup_vllm.check_host(backend, '0.34.0')
            with patch.object(setup_vllm, 'packages_ready', side_effect=[False, True]), \
                    patch.object(setup_vllm.subprocess, 'run') as run:
                setup_vllm.ensure_packages(target, '0.34.0', backend='amd')
            command = run.call_args.args[0]
            self.assertIn('vllm==0.34.0+rocm100', command)
            self.assertIn('https://wheels.vllm.ai/rocm/0.34.0/rocm100', command)
