import itertools
import subprocess
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import llm_infer_vllm as benchmark


class VllmProtocolTest(unittest.TestCase):
    def run_combo(self, token_count):
        ticks = itertools.count(step=.001)
        model = SimpleNamespace(generate=lambda prompts, options: [
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=list(range(token_count)))])])
        sampler = SimpleNamespace(start=lambda: None, stop=lambda **kw: {
            'energy_j': None, 'mean_power_w': None, 'power_sampler_available': False})
        with patch.object(benchmark, 'LLM', return_value=model) as engine, \
             patch.object(benchmark, 'SamplingParams', side_effect=lambda **kw: kw) as params, \
             patch.object(benchmark, 'EnergySampler', return_value=sampler), \
             patch.object(benchmark, 'detect_gpu_name', return_value='test'), \
             patch.object(benchmark.time, 'perf_counter', side_effect=lambda: next(ticks)):
            row = benchmark.run_combo('model', 'float16', 1, 1, 'prompt', 3, 3,
                                      2, 0, .01, .8, revision='a' * 40)
            self.assertEqual(engine.call_args.kwargs['revision'], 'a' * 40)
            self.assertEqual(params.call_args.kwargs['min_tokens'], 2)
            self.assertTrue(params.call_args.kwargs['ignore_eos'])
            return row

    def test_fixed_work_revision_and_unavailable_energy(self):
        row = self.run_combo(2)
        self.assertEqual(row['backend'], 'vllm')
        self.assertEqual(row['generated_tokens'], row['requests'] * 2)
        self.assertIsNone(row['gen_tokens_per_watt'])

    def test_vram_samples_match_ids_and_preserve_maximum(self):
        output = ['GPU-B, 70\nGPU-A, 50', 'GPU-A, 65\nGPU-B, 60', 'GPU-A, 99']
        sampler = benchmark.VramSampler(['GPU-A', 'GPU-B'])
        with patch.object(subprocess, 'check_output', side_effect=output):
            for _ in output:
                sampler.sample()
        self.assertEqual(sampler.peaks, {'GPU-A': 65, 'GPU-B': 70})
        self.assertEqual(sampler.sample_count, 2)

    def test_short_generation_fails(self):
        with self.assertRaises(ValueError):
            self.run_combo(1)
