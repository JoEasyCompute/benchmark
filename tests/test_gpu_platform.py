import unittest
from unittest.mock import patch

import gpu_platform as gpu


class GpuDetectionTest(unittest.TestCase):
    def test_auto_uses_responding_devices_not_just_installed_commands(self):
        with patch.object(gpu, 'query_gpu_ids', side_effect=lambda backend: ['2'] if backend == 'amd' else []):
            self.assertEqual(gpu.detect_backend(), 'amd')

    def test_no_devices_and_mixed_devices_require_explicit_resolution(self):
        for inventories in ({'nvidia': [], 'amd': []}, {'nvidia': ['0'], 'amd': ['1']}):
            with self.subTest(inventories=inventories), patch.object(gpu, 'query_gpu_ids', side_effect=inventories.get):
                with self.assertRaises(ValueError):
                    gpu.detect_backend()

    def test_resolved_config_preserves_workload_and_source(self):
        original = {'gpu_backend': 'auto', 'gpu_include': [], 'blender': {'backend': 'auto'},
                    'llm_infer': {'backend': 'transformers', 'dtype': 'float16', 'batch_sizes': [1, 4]}}
        with patch.object(gpu, 'query_gpu_ids', side_effect=lambda backend: ['2', '3'] if backend == 'amd' else []):
            resolved = gpu.resolve_config(original, environ={})
        self.assertEqual(resolved['gpu_backend'], 'amd')
        self.assertEqual(resolved['gpu_include'], [2, 3])
        self.assertEqual(resolved['blender']['backend'], 'hip')
        self.assertEqual(resolved['llm_infer'], original['llm_infer'])
        self.assertEqual(original['gpu_backend'], 'auto')
        self.assertEqual(original['gpu_include'], [])

    def test_cli_overrides_config_but_not_visibility_restrictions(self):
        original = {'gpu_backend': 'amd', 'gpu_include': [2], 'blender': {'backend': 'auto'}}
        with patch.object(gpu, 'query_gpu_ids', return_value=['0', '1', '2']):
            resolved = gpu.resolve_config(original, backend_override='nvidia', gpu_override='1',
                                          environ={'CUDA_VISIBLE_DEVICES': '1,2'})
            self.assertEqual(resolved['gpu_include'], [1])
            self.assertEqual(resolved['blender']['backend'], 'cuda')
            with self.assertRaises(ValueError):
                gpu.resolve_config(original, backend_override='nvidia', gpu_override='0',
                                   environ={'CUDA_VISIBLE_DEVICES': '1,2'})

    def test_invalid_gpu_list_or_missing_explicit_backend_fails(self):
        with patch.object(gpu, 'query_gpu_ids', return_value=['0']):
            for value in ('1', '-1', '0,0', 'x', ''):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    gpu.resolve_config({'gpu_backend': 'nvidia'}, gpu_override=value, environ={})
        with patch.object(gpu, 'query_gpu_ids', return_value=[]):
            with self.assertRaises(ValueError):
                gpu.resolve_config({'gpu_backend': 'amd'}, environ={})

    def test_nvidia_names_are_queried_for_selected_physical_ids(self):
        with patch.object(gpu, 'run', return_value='NVIDIA GPU 2') as query:
            self.assertEqual(gpu.query_gpu_names('nvidia', '2'), ['NVIDIA GPU 2'])
            self.assertIn('--id=2', query.call_args.args[0])

    def test_amd_ignores_non_device_json_keys(self):
        with patch.object(gpu.shutil, 'which', return_value='/bin/rocm-smi'), \
                patch.object(gpu, 'run', return_value='{"card3": {"Device ID": "0xab"}, "system": {}}'):
            self.assertEqual(gpu.query_gpu_ids('amd'), ['3'])

    def test_nvidia_visibility_uses_uuid_to_avoid_cuda_index_reordering(self):
        with patch.object(gpu, 'run', return_value='0, GPU-aaa\n2, GPU-bbb'):
            self.assertEqual(gpu.visibility_mask('nvidia', ['2', '0']), 'GPU-bbb,GPU-aaa')
        self.assertEqual(gpu.visibility_mask('amd', ['2', '0']), '2,0')
