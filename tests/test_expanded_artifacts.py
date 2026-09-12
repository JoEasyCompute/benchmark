import unittest

from validate_run_artifacts import metric_issues


class ExpandedArtifactTest(unittest.TestCase):
    def test_incomplete_telemetry_must_not_claim_energy(self):
        errors, warnings = metric_issues([dict(suite='vision_infer', status='ok',
            energy_j=100, power_sampler_available=False)], {})
        self.assertTrue(errors)

    def test_unavailable_energy_is_valid_null(self):
        errors, warnings = metric_issues([dict(suite='vision_infer', status='ok',
            energy_j=None, power_sampler_available=False)], {})
        self.assertEqual(errors, [])

    def test_enabled_optional_suite_cannot_disappear(self):
        errors, warnings = metric_issues([], {'vision_infer': {'enabled': True}})
        self.assertTrue(any('vision_infer' in error for error in errors))

    def test_baseline_rejects_multi_gpu_result(self):
        errors, warnings = metric_issues([dict(suite='vision_infer', status='ok', gpu_count=2)],
                                        {'benchmark_profile': 'single_gpu_baseline'})
        self.assertTrue(errors)

    def test_nonfinite_measurements_and_malformed_rows_rejected(self):
        for row in ([], dict(suite='llm_serve', status='ok', time_s=float('nan'))):
            self.assertTrue(metric_issues([row], {})[0])
