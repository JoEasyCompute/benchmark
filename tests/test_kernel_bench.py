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
