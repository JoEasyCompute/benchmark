import unittest
from unittest.mock import patch
from benchmarks.energy import EnergySampler, integrate_samples, parse_rocm_power, physical_devices, aggregate_energy


class EnergyTests(unittest.TestCase):
    def test_trapezoid(self):
        self.assertEqual(integrate_samples([(0, 100), (2, 200)], 0, 2, 3), (300, 1.0))

    def test_missing_and_gap(self):
        for samples in ([(0, 100), (2, None)], [(0, 100), (2, float('nan'))], [(0, 100), (2, -1)]):
            self.assertIsNone(integrate_samples(samples, 0, 2, 3)[0])
        self.assertIsNone(integrate_samples([(0, 100), (2, 200)], 0, 2, 1)[0])

    def test_out_of_order_or_nonfinite_timestamps_cannot_double_count(self):
        for samples in ([(0, 100), (2, 100), (1, 100), (3, 100)],
                        [(0, 100), (float('nan'), 100), (3, 100)]):
            self.assertIsNone(integrate_samples(samples, 0, 3, 5)[0])

    def test_selected_rocm_power_only(self):
        payload = {'card0': {'Average Graphics Package Power (W)': '100.0', 'Power Cap (W)': '300'}, 'card1': {'Current Socket Graphics Package Power (W)': '200'}}
        self.assertEqual(parse_rocm_power(payload, ['1']), 200)
        self.assertIsNone(parse_rocm_power(payload, ['2']))

    def test_device_mapping(self):
        self.assertEqual(physical_devices('amd', [1], {'HIP_VISIBLE_DEVICES': '4,7'}), ['7'])
        self.assertEqual(physical_devices('nvidia', [0], {'CUDA_VISIBLE_DEVICES': 'GPU-abc'}), ['GPU-abc'])

    def test_nvidia_cli_fallback_works_in_embedded_blender_python(self):
        sampler = EnergySampler('nvidia')
        sampler.device_ids = ['GPU-test']
        with patch('benchmarks.energy.subprocess.check_output', return_value='125.5\n') as query:
            self.assertEqual(sampler._read(), 125.5)
            self.assertIn('--id=GPU-test', query.call_args.args[0])

    def test_aggregation_rejects_overlapping_devices(self):
        row = dict(energy_j=100, power_sampler_available=True, energy_method='board_power_trapezoid_v1', power_device_ids=['0'], power_started_s=0, power_ended_s=2)
        self.assertIsNone(aggregate_energy([row, row])['energy_j'])

    def test_lifecycle_samples_both_boundaries(self):
        sampler = EnergySampler('unsupported', interval_s=1)
        with patch.object(sampler, '_read', side_effect=[100, 200]), \
             patch('benchmarks.energy.time.perf_counter', side_effect=[0, 2, 2]), \
             patch('benchmarks.energy.threading.Thread'):
            sampler.start()
            result = sampler.stop()
        self.assertIsNone(result['energy_j'])
        self.assertIsNone(result['mean_power_w'])
        self.assertEqual(result['power_unavailable_reason'], 'insufficient_in_window_samples')
        self.assertEqual(result['power_coverage'], 1)
        self.assertEqual(result['power_sample_count'], 2)

    def test_clips_to_exact_workload_interval(self):
        self.assertEqual(integrate_samples([(0, 100), (2, 200)], .5, 1.5, 3), (150, 1))

    def test_context_cleans_up_on_error(self):
        sampler = EnergySampler('unsupported')
        with patch.object(sampler, 'start'), patch.object(sampler, 'stop') as stop:
            with self.assertRaises(ValueError):
                with sampler:
                    raise ValueError('operation failed')
            stop.assert_called_once()


class VerifiedMappingTests(unittest.TestCase):
    def test_hip_ordinal_is_matched_by_pci_not_smi_card_number(self):
        from benchmarks.energy import match_rocm_devices
        payload = {'card0': {'PCI Bus': '0000:03:00.0'},
                   'card7': {'PCI Bus': '0000:E3:00.0'}}
        self.assertEqual(match_rocm_devices(['0000:e3:00.0'], payload), ['7'])
        self.assertIsNone(match_rocm_devices(['0000:ff:00.0'], payload))

    def test_ambiguous_bus_is_rejected(self):
        from benchmarks.energy import match_rocm_devices
        self.assertIsNone(match_rocm_devices(['0000:03:00.0'], {
            'card0': {'PCI Bus': '0000:03:00.0'},
            'card1': {'PCI Bus': '0000:03:00.0'}}))

    def test_unverified_amd_mapping_does_not_query_power(self):
        sampler = EnergySampler('amd')
        with patch('benchmarks.energy.subprocess.check_output') as query:
            self.assertIsNone(sampler._read())
            query.assert_not_called()

    def test_three_in_window_samples_allow_energy(self):
        sampler = EnergySampler('unsupported', interval_s=1)
        sampler.started = 0
        sampler.samples = [(0, 100), (1, 100), (2, 100)]
        with patch.object(sampler, '_sample'):
            result = sampler.stop(started_s=0, ended_s=2)
        self.assertEqual(result['energy_j'], 200)

    def test_start_queries_bus_and_uses_verified_smi_device(self):
        sampler = EnergySampler('amd', [0])
        with patch('benchmarks.energy.hip_pci_devices', return_value=['0000:e3:00.0']), \
             patch('benchmarks.energy.subprocess.check_output', side_effect=[
                 '{"card7": {"PCI Bus": "0000:E3:00.0"}}',
                 '{"card7": {"Average Graphics Package Power (W)": "250"}}']), \
             patch('benchmarks.energy.threading.Thread'):
            sampler.start()
        self.assertTrue(sampler.mapping_verified)
        self.assertEqual(sampler.device_ids, ['7'])
        self.assertEqual(sampler.samples[0][1], 250)

    def test_no_loaded_hip_runtime_stays_unavailable(self):
        from benchmarks.energy import hip_pci_devices
        with patch('benchmarks.energy.Path.read_text', return_value=''):
            self.assertIsNone(hip_pci_devices([0]))
