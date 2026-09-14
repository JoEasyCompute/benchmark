import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import kernel_bench


class KernelMathTests(unittest.TestCase):
    def test_work_counts_include_both_attention_products(self):
        self.assertEqual(kernel_bench.operation_work('gemm', 10), 2000)
        self.assertEqual(kernel_bench.operation_work('attention', 10, 2, 4), 3200)
        self.assertEqual(kernel_bench.operation_work('memory', 10, itemsize=2), 400)

    def test_throughput_counts_every_measured_iteration(self):
        result = kernel_bench.summarize([1, 2, 3], 120)
        self.assertEqual(result['throughput'], 60)
        self.assertEqual(result['latency_ms_p50'], 2000)
        self.assertAlmostEqual(result['latency_ms_p95'], 2900)

    def test_rejects_invalid_measurements(self):
        for samples in ([], [0], [float('nan')]):
            with self.assertRaises(ValueError):
                kernel_bench.summarize(samples, 100)

class MeasurementWindowTests(unittest.TestCase):
    def test_extends_fast_operations_to_minimum_duration(self):
        from unittest.mock import patch
        with patch.object(kernel_bench, 'timed_call', return_value=(None, .25)):
            samples = kernel_bench.measure_iterations(lambda: None, lambda: None, 2, 1.0)
        self.assertEqual(samples, [.25] * 4)

    def test_preserves_iteration_minimum_and_zero_duration_smoke(self):
        from unittest.mock import patch
        with patch.object(kernel_bench, 'timed_call', return_value=(None, 2.0)):
            self.assertEqual(len(kernel_bench.measure_iterations(lambda: None, lambda: None, 3, 1)), 3)
            self.assertEqual(len(kernel_bench.measure_iterations(lambda: None, lambda: None, 2, 0)), 2)

    def test_rejects_nonprogressing_clock(self):
        from unittest.mock import patch
        with patch.object(kernel_bench, 'timed_call', return_value=(None, 0)):
            with self.assertRaises(ValueError):
                kernel_bench.measure_iterations(lambda: None, lambda: None, 2, 1)
