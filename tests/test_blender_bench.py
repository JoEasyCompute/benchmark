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

    def test_all_mode_matches_exact_selected_pci_devices(self):
        module = self.module()
        devices = [SimpleNamespace(type='CUDA', id=f'CUDA_NVIDIA GPU_0000:{bus}:00', use=True)
                   for bus in ('65', '01', '82')]
        requested = ['00000000:82:00.0', '00000000:01:00.0']
        selected = module.select_devices(devices, 'CUDA', 'all', requested)
        self.assertEqual(selected, [devices[2], devices[1]])
        self.assertEqual([d.use for d in devices], [False, True, True])
        self.assertEqual(module.select_devices(devices, 'CUDA', 'single', requested), [devices[2]])

    def test_all_mode_rejects_missing_requested_gpu(self):
        module = self.module()
        devices = [SimpleNamespace(type='HIP', id='HIP_AMD GPU_0000:01:00', use=True)]
        with self.assertRaisesRegex(ValueError, '65:00'):
            module.select_devices(devices, 'HIP', 'all', ['0000:01:00.0', '0000:65:00.0'])

    def test_preflight_rejects_a_selected_device_with_an_unsupported_driver(self):
        module = self.module()
        device = SimpleNamespace(type='CUDA', id='CUDA_GPU_0000:65:00', name='Test GPU',
                                 use=False, meets_driver_requirement=False)
        prefs = SimpleNamespace(get_devices_for_type=lambda backend: [device])
        bpy = SimpleNamespace(context=SimpleNamespace(preferences=SimpleNamespace(
            addons={'cycles': SimpleNamespace(preferences=prefs)})))
        with patch.dict('sys.modules', {'bpy': bpy}), \
             patch.dict(os.environ, {'BLENDER_GPU_BACKEND': 'CUDA',
                                     'BLENDER_PHYSICAL_GPU_IDS': '2'}), \
             patch.object(module, 'gpu_inventory', return_value=[('2', '0000:65:00.0')]):
            with self.assertRaisesRegex(ValueError, 'driver requirement not met'):
                module.check_cycles()

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

    def test_null_scenes_use_bundled_defaults(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            env = {'BLENDER_SCENES_JSON': 'null', 'SCENES_DIR': temporary}
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(module, 'run_scene', return_value={'status': 'ok'}) as render:
                self.assertEqual(module.main(), 0)
            self.assertEqual([(call.args[1].name, call.args[2]) for call in render.call_args_list],
                             [('BMW27.blend', 'single'), ('BMW27.blend', 'all'),
                              ('classroom.blend', 'single'), ('classroom.blend', 'all')])

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
