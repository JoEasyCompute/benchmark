import unittest

from suite_config import optional_jobs, validate_optional_suites, smoke_optional_suites


class SuiteConfigTest(unittest.TestCase):
    def test_full_sweeps_are_config_driven(self):
        cfg = {
            'vision_infer': {'enabled': True, 'sizes': [224, 256], 'batch_sizes': [1, 4]},
            'kernel_bench': {'enabled': True, 'cases': ['memory'], 'size': 128},
            'llm_serve': {'enabled': True, 'provider': 'vllm', 'endpoint': 'http://localhost:8000/v1/completions',
                          'concurrency': [1, 3], 'model': 'test'},
        }
        jobs = optional_jobs(cfg)
        self.assertEqual([j['suite'] for j in jobs].count('vision_infer'), 4)
        kernel = next(j for j in jobs if j['suite'] == 'kernel_bench')
        self.assertIn('memory', kernel['args'])
        self.assertIn('128', kernel['args'])
        self.assertEqual([j['suite'] for j in jobs].count('llm_serve'), 2)

    def test_bad_values_fail_before_gpu_or_network_work(self):
        for suite, values in (
            ('vision_infer', {'enabled': True, 'batch_sizes': [0]}),
            ('kernel_bench', {'enabled': True, 'cases': ['unknown']}),
            ('llm_serve', {'enabled': True, 'concurrency': [True]}),
            ('vision_infer', {'enabled': True, 'weights': 'DEFAULT'}),
            ('vision_infer', {'enabled': 'yes'}),
            ('kernel_bench', {'enabled': True, 'iterations': 0}),
            ('llm_serve', {'enabled': True, 'duration': float('nan')}),
        ):
            with self.subTest(suite=suite, values=values):
                self.assertTrue(validate_optional_suites({suite: values}))

    def test_smoke_reduces_each_enabled_workload(self):
        cfg = {name: {'enabled': True} for name in ('vision_infer', 'kernel_bench', 'llm_serve')}
        smoke_optional_suites(cfg)
        self.assertEqual(cfg['vision_infer']['batch_sizes'], [1])
        self.assertEqual(cfg['vision_infer']['iterations'], 2)
        self.assertEqual(cfg['kernel_bench']['size'], 64)
        self.assertEqual(cfg['llm_serve']['duration'], 1)
        self.assertEqual(cfg['llm_serve']['concurrency'], [1])

    def test_disabled_suites_produce_no_jobs(self):
        self.assertEqual(optional_jobs({}), [])
