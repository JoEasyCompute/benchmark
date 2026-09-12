import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import compare_runs as compare

ROOT = Path(__file__).resolve().parents[1]


def run_data(label, rows):
    return dict(label=label, summary_rows=rows, software_versions={},
                run_dir=label, gpu_backend='nvidia', gpu_name=label,
                max_gpu_count=1, python='3.11')


class ExpandedReportingTest(unittest.TestCase):
    def test_measurement_telemetry_does_not_split_repeats(self):
        rows = [dict(suite='kernel_bench', status='ok', case='gemm', size=64,
                     repeat_index=i, repeat_count=3, throughput=i*10,
                     tflops=i, memory_peak_bytes=i*100, power_sample_count=i+1,
                     power_started_s=i*1000, power_ended_s=i*1000+2,
                     power_coverage=1, energy_j=i*20,
                     power_sampler_available=True, latency_ms_p99=i,
                     energy_method='board_power_trapezoid_v1') for i in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory) / 'results'
            result_dir.mkdir()
            (result_dir / 'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            subprocess.run([sys.executable, str(ROOT / 'harness.py')], cwd=directory,
                           check=True, capture_output=True)
            summary = json.loads((result_dir / 'metrics_summary.json').read_text())
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]['throughput_mean'], 20)
        self.assertEqual(summary[0]['throughput_median'], 20)
        self.assertEqual(summary[0]['energy_j_count'], 3)

    def test_diagnostics_cannot_become_application_picks(self):
        base = dict(suite='kernel_bench', status='ok', case='gemm', size=64,
                    seed=1234, timing_method='synchronized_iteration_v2', throughput_mean=10)
        payload = compare.build_payload([run_data('A', [base]), run_data('B', [dict(base, throughput_mean=20)])])
        self.assertFalse(any(d['suite']=='kernel_bench' for d in payload['executive_summary']['suite_decisions']))
        self.assertIn('Kernel diagnostics', compare.render_markdown(payload))

    def test_incompatible_new_identities_never_group(self):
        for suite, field in [('vision_infer', 'weights_sha256'), ('kernel_bench', 'head_dim'),
                             ('llm_train_real', 'objective'), ('blender', 'render_settings_sha256'),
                             ('llm_serve', 'generation_protocol')]:
            with self.subTest(suite=suite):
                self.assertNotEqual(compare.row_key(dict(suite=suite, **{field: 'a'})),
                                    compare.row_key(dict(suite=suite, **{field: 'b'})))

    def test_all_failed_suite_remains_visible(self):
        row = dict(suite='vision_infer', status='failed', model='resnet18')
        payload = compare.build_payload([run_data('A', [row]), run_data('B', [row])])
        self.assertIn('vision_infer', payload['suites'])
        self.assertIn('failed', compare.render_markdown(payload))

    def test_single_gpu_pick_is_not_borrowed_from_multi_gpu_group(self):
        base = dict(suite='llm_train', status='ok', dtype='fp16', batch_size=1,
                    seq_len=32, hidden_size=64, n_layers=1, n_heads=2, seed=1234,
                    timing_method='test', summary_count=3)
        a = [dict(base, world_size=1, gpu_count=1, tokens_per_sec_mean=10),
             dict(base, world_size=2, gpu_count=2, tokens_per_sec_mean=10)]
        b = [dict(base, world_size=1, gpu_count=1, tokens_per_sec_mean=5),
             dict(base, world_size=2, gpu_count=2, tokens_per_sec_mean=100)]
        payload = compare.build_payload([run_data('A', a), run_data('B', b)])
        self.assertEqual(payload['single_gpu_summary'][0]['top_pick'], 'A')

    def test_energy_table_requires_every_repeat_and_no_energy_winner(self):
        row = dict(suite='vision_infer', status='ok', model='resnet18', gpu_count=1,
                   images_per_sec_mean=10, summary_count=3, energy_j_count=2,
                   energy_j_mean=100, images_per_joule_mean=5,
                   power_sampler_available=True, energy_method='board_power_trapezoid_v1')
        payload = compare.build_payload([run_data('A', [row])])
        item = payload['energy_summary'][0]
        self.assertIsNone(item['energy_j_mean'])
        self.assertIsNone(item['efficiency'])
        self.assertNotIn('winner', item)
        self.assertIn('No energy winner', compare.render_markdown(payload))
