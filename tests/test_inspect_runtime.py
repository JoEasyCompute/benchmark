import json
import subprocess
import unittest
from unittest.mock import patch

import inspect_runtime


class InspectRuntimeTests(unittest.TestCase):
    def test_os_release_patch_version(self):
        self.assertEqual(inspect_runtime.parse_os_release(
            'ID=ubuntu\nVERSION_ID="24.04"\nPRETTY_NAME="Ubuntu 24.04.3 LTS"'),
            {'os_id': 'ubuntu', 'os_version': '24.04', 'os_release': '24.04.3'})

    def test_nvidia_capability(self):
        gpu = inspect_runtime.parse_nvidia('0, NVIDIA RTX 4090, 580.1, 8.9')[0]
        self.assertEqual(gpu['architecture'], 'sm_89')
        self.assertEqual(gpu['driver_version'], '580.1')
        self.assertIsNone(inspect_runtime.parse_nvidia('0, GPU, 580, N/A')[0]['architecture'])

    def probe(self, backend='amd', gpu_ids=None, nvidia=''):
        products = {'card' + str(i): {'Card Series': 'AMD Radeon AI Pro R9700S',
                                     'GFX Version': 'gfx1201'} for i in range(8)}

        def command(args, **kwargs):
            self.assertEqual(kwargs['timeout'], 15)
            output = ''
            if args[0] == 'nvidia-smi':
                output = nvidia
            elif '--showproductname' in args:
                output = json.dumps(products)
            elif '--showdriverversion' in args:
                output = json.dumps({'system': {'Driver version': '6.16.13'}})
            elif args[0] == 'modinfo':
                output = '7.1.3.31500000'
            return subprocess.CompletedProcess(args, 0, output, '')

        def read(path):
            if str(path) == '/etc/os-release':
                return 'ID=ubuntu\nVERSION_ID="24.04"\nPRETTY_NAME="Ubuntu 24.04.3 LTS"'
            return '7.2.0'

        with patch('inspect_runtime.subprocess.run', side_effect=command), \
                patch('inspect_runtime.Path.read_text', read), \
                patch('inspect_runtime.platform.system', return_value='Linux'), \
                patch('inspect_runtime.platform.machine', return_value='x86_64'), \
                patch('inspect_runtime.platform.release', return_value='6.8.0-137-generic'), \
                patch('inspect_runtime.platform.python_version', return_value='3.12.3'), \
                patch('inspect_runtime.platform.libc_ver', return_value=('glibc', '2.39')), \
                patch('inspect_runtime.importlib.metadata.version', return_value='1.0'):
            return inspect_runtime.collect_host(backend, gpu_ids)

    def test_remote_fixture_keeps_version_dimensions_separate(self):
        result = self.probe()
        self.assertEqual(result['rocm_version'], '7.2.0')
        self.assertEqual(result['amdgpu_module_version'], '7.1.3.31500000')
        self.assertEqual(result['amdgpu_driver_version'], '6.16.13')
        self.assertEqual(result['os_release'], '24.04.3')
        self.assertEqual(result['python_version'], '3.12.3')
        self.assertEqual(result['glibc_version'], '2.39')
        self.assertEqual(len(result['gpus']), 8)
        self.assertTrue(all(gpu['architecture'] == 'gfx1201' for gpu in result['gpus']))
        self.assertFalse(result['inspection_warnings'])

    def test_selected_gpu_indices_are_preserved(self):
        result = self.probe(gpu_ids=['3', '7'])
        self.assertEqual([gpu['index'] for gpu in result['gpus']], ['3', '7'])

    def test_mixed_vendors_require_selection(self):
        with self.assertRaisesRegex(ValueError, 'Both AMD and NVIDIA'):
            self.probe('auto', nvidia='0, NVIDIA RTX 4090, 580.1, 8.9')

    def test_timeout_returns_warning(self):
        with patch('inspect_runtime.subprocess.run', side_effect=subprocess.TimeoutExpired('probe', 15)), \
                patch('inspect_runtime.Path.read_text', return_value='ID=ubuntu'), \
                patch('inspect_runtime.importlib.metadata.version', return_value='1.0'):
            result = inspect_runtime.collect_host('nvidia')
        self.assertEqual(result['gpus'], [])
        self.assertTrue(any('timed out' in warning for warning in result['inspection_warnings']))

    def test_amd_does_not_invent_architecture(self):
        gpus = inspect_runtime.parse_amd('{"card3":{"Card Series":"AMD GPU"}}')
        self.assertEqual(gpus[0]['index'], '3')
        self.assertIsNone(gpus[0]['architecture'])


if __name__ == '__main__':
    unittest.main()
