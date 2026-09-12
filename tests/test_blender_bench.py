import importlib.util
import pathlib
import unittest
import json
import os
import tempfile
from unittest.mock import patch
from types import SimpleNamespace


class BlenderBenchmarkTests(unittest.TestCase):
    def module(self):
        path = pathlib.Path(__file__).parents[1] / 'benchmarks/blender_render.py'
        self.assertTrue(path.exists(), 'render-only measurement implementation missing')
        spec = importlib.util.spec_from_file_location('blender_render', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_select_devices_rejects_cpu_fallback(self):
        module = self.module()
        devices = [SimpleNamespace(type='CPU', id='cpu', name='cpu', use=True)]
        with self.assertRaises(ValueError):
            module.select_devices(devices, 'HIP', 'single')

    def test_select_devices_single_and_all(self):
        module = self.module()
        devices = [SimpleNamespace(type='CUDA', id=str(i), name=str(i), use=False)
                   for i in range(2)]
        self.assertEqual(len(module.select_devices(devices, 'CUDA', 'single')), 1)
        self.assertEqual([d.use for d in devices], [True, False])
        self.assertEqual(len(module.select_devices(devices, 'CUDA', 'all')), 2)

    def test_settings_hash_changes_with_render_work(self):
        module = self.module()
        self.assertNotEqual(module.settings_hash({'samples': 32}),
                            module.settings_hash({'samples': 64}))

    def test_telemetry_mapping_uses_pci_not_order(self):
        module = self.module()
        selected = [SimpleNamespace(id='CUDA_NVIDIA GPU_0000:65:00.0')]
        self.assertEqual(module.telemetry_devices(selected, [('GPU-A', '00000000:01:00.0'),
                                                             ('GPU-B', '00000000:65:00.0')]), ['GPU-B'])
        self.assertIsNone(module.telemetry_devices([SimpleNamespace(id='opaque')], []))

    def test_amd_power_mapping_uses_selected_pci_device(self):
        module = self.module()
        selected = [SimpleNamespace(id='HIP_AMD_0000:65:00.0')]
        raw = json.dumps({'card0': {'PCI Bus': '0000:01:00.0'},
                          'card2': {'PCI Bus': '0000:65:00.0'}})
        with patch.object(module.subprocess, 'check_output', return_value=raw):
            sampler = module.energy_sampler(selected, 'HIP')
        self.assertIsNotNone(sampler)
        self.assertEqual(sampler.device_ids, ['2'])

    def test_driver_preserves_render_time_separate_from_process(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            scene = pathlib.Path(temporary) / 'quoted "scene.blend'
            scene.write_bytes(b'scene')
            def process(command, env, check):
                pathlib.Path(env['BLENDER_TIMING_OUTPUT']).write_text(json.dumps({
                    'status': 'ok', 'render_time_s': 2.5, 'time_s': 2.5,
                    'startup_load_time_s': 1.0, 'num_gpus': 1}))
                return SimpleNamespace(returncode=0)
            with patch.object(module.subprocess, 'run', side_effect=process), \
                    patch.object(module.time, 'monotonic', side_effect=[10.0, 14.0]):
                row = module.run_scene('blender', scene, 'single', {})
            self.assertEqual(row['time_s'], 2.5)
            self.assertEqual(row['end_to_end_time_s'], 4.0)
            self.assertEqual(row['startup_load_time_s'], 1.0)
            self.assertNotIn('warm', row)
            self.assertEqual(json.loads(json.dumps(row))['scene'], scene.name)

    def test_failed_process_never_has_render_time(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            scene = pathlib.Path(temporary) / 'scene.blend'
            scene.write_bytes(b'scene')
            with patch.object(module.subprocess, 'run', return_value=SimpleNamespace(returncode=1)):
                row = module.run_scene('blender', scene, 'single', {})
            self.assertEqual(row['status'], 'failed')
            self.assertNotIn('time_s', row)

    def test_baseline_emits_single_mode_only(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            metrics = pathlib.Path(temporary) / 'metrics.jsonl'
            env = {'BENCHMARK_PROFILE': 'single_gpu_baseline',
                   'RESULTS_JSON': str(pathlib.Path(temporary) / 'results.json'),
                   'METRICS_JSONL': str(metrics), 'SCENES_DIR': temporary,
                   'BLENDER_SCENES_JSON': '["missing.blend"]'}
            with patch.dict(os.environ, env, clear=True):
                module.main()
            rows = [json.loads(line) for line in metrics.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['mode'], 'single')


if __name__ == '__main__':
    unittest.main()
