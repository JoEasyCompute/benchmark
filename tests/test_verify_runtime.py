import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import verify_runtime as runtime


class Tensor:
    def __init__(self, values, broken=False):
        self.values = values
        self.broken = broken

    def __matmul__(self, other):
        return Tensor([[0, 0], [0, 0]] if self.broken else [[7, 10], [15, 22]])

    def all(self):
        return True


def fake_torch(count=1, broken=False):
    def assert_close(actual, expected, **kwargs):
        if actual.values != expected.values:
            raise AssertionError('incorrect numerical result')
    return SimpleNamespace(
        __version__='2.9.1+cu128', version=SimpleNamespace(cuda='12.8', hip=None),
        float32='float32', tensor=lambda values, **kw: Tensor(values, broken),
        isfinite=lambda value: Tensor(True), testing=SimpleNamespace(assert_close=assert_close),
        cuda=SimpleNamespace(is_available=lambda: count > 0, device_count=lambda: count,
                             synchronize=lambda index: None, get_device_name=lambda index: 'Test GPU',
                             get_device_properties=lambda index: SimpleNamespace(major=8, minor=9),
                             get_arch_list=lambda: ['sm_89']))


def lock():
    return {'schema_version': 1, 'profile': {'id': 'test', 'backend': 'nvidia',
            'torch_version': '2.9.1', 'runtime_version': '12.8',
            'expected_packages': {'torch': '2.9.1+cu128'}}, 'host': {'gpus': [{}]}}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.packages = patch.object(runtime.importlib.metadata, 'version', return_value='2.9.1+cu128')
        self.packages.start()
        self.env = patch.dict(runtime.os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.packages.stop)
        self.addCleanup(self.env.stop)

    def test_numerical_success(self):
        result = runtime.run_checks(fake_torch(), lock())
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(len(result['devices']), 1)

    def test_arithmetic_failure(self):
        self.assertEqual(runtime.run_checks(fake_torch(broken=True), lock())['status'], 'error')

    def test_backend_and_runtime_mismatch(self):
        for field, value in [('backend', 'amd'), ('runtime_version', '12.9')]:
            fixture = lock()
            fixture['profile'][field] = value
            self.assertEqual(runtime.run_checks(fake_torch(), fixture)['status'], 'error')

    def test_package_mismatch(self):
        fixture = lock()
        fixture['profile']['expected_packages']['torch'] = '2.9.0'
        self.assertEqual(runtime.run_checks(fake_torch(), fixture)['status'], 'error')

    def test_no_gpu_and_count_mismatch(self):
        for count in (0, 2):
            self.assertEqual(runtime.run_checks(fake_torch(count), lock())['status'], 'error')

    def test_explicit_visibility_mask_respected(self):
        fixture = lock()
        fixture['host']['gpus'] = [{}, {}]
        with patch.dict(runtime.os.environ, {'CUDA_VISIBLE_DEVICES': '1'}):
            self.assertEqual(runtime.run_checks(fake_torch(), fixture)['status'], 'error')

    def test_collective_failure_propagates(self):
        fixture = lock()
        fixture['host']['gpus'] = [{}, {}]
        with patch.object(runtime, 'run_collective', side_effect=RuntimeError('collective failed')):
            result = runtime.run_checks(fake_torch(2), fixture, distributed=True)
        self.assertEqual(result['status'], 'error')
        self.assertIn('collective failed', result['errors'][0])

    def test_cli_failure_writes_json_and_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'lock.json'
            output = Path(directory) / 'report.json'
            source.write_text(json.dumps(lock()))
            with patch.object(runtime.importlib, 'import_module', return_value=fake_torch(0)):
                code = runtime.main(['--lock', str(source), '--json-out', str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.read_text())['status'], 'error')

    def test_collective_subprocess_nonzero(self):
        process = Mock(returncode=2)
        process.communicate.return_value = ('', 'NCCL failure')
        with patch.object(runtime.subprocess, 'Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'NCCL failure'):
                runtime.run_collective(2)

    def test_collective_timeout_kills_process_group(self):
        process = Mock(pid=123)
        process.communicate.side_effect = [runtime.subprocess.TimeoutExpired('torchrun', 90), ('', '')]
        with patch.object(runtime.subprocess, 'Popen', return_value=process), patch.object(runtime.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                runtime.run_collective(2)
            kill.assert_called_once_with(123, runtime.signal.SIGKILL)

    def test_amd_runtime_suffix_and_architecture(self):
        fixture = lock()
        fixture['profile'].update(backend='amd', runtime_version='7.2.0')
        torch = fake_torch()
        torch.version = SimpleNamespace(cuda=None, hip='7.2.0-abcdef')
        torch.cuda.get_device_properties = lambda index: SimpleNamespace(gcnArchName='gfx1201')
        torch.cuda.get_arch_list = lambda: ['gfx11-generic']
        result = runtime.run_checks(torch, fixture)
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['devices'][0]['architecture'], 'gfx1201')

    def test_validation_records_host_qualification_and_physical_selection(self):
        fixture = lock()
        fixture.update(host_qualified=False, allow_unverified_host=True)
        fixture['host']['gpus'] = [{'index': '3'}]
        result = runtime.run_checks(fake_torch(), fixture)
        self.assertEqual(result['status'], 'pass')
        self.assertFalse(result['host_qualified'])
        self.assertEqual(result['devices'][0]['physical_index'], '3')

    def test_unverified_host_flag_cannot_bypass_arithmetic(self):
        fixture = lock()
        fixture.update(host_qualified=False, allow_unverified_host=True)
        result = runtime.run_checks(fake_torch(broken=True), fixture)
        self.assertEqual(result['status'], 'error')
