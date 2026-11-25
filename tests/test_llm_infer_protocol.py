"""Exercise the real HF timing loop with GPU/model boundaries replaced."""
import contextlib
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import llm_infer_hf as bench


class Tensor:
    shape = (1, 8)

    def repeat(self, batch, _):
        return self

    def to(self, device):
        return self


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, *args, **kwargs):
        return dict(input_ids=Tensor(), attention_mask=Tensor())


class InferenceProtocolTest(unittest.TestCase):
    def test_real_loop_records_fixed_work_and_revision(self):
        calls = []
        events = []
        def generate(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(shape=(1, 12))
        gpu = SimpleNamespace(is_available=lambda: True, set_device=lambda _: None,
                              synchronize=lambda: events.append('sync'),
                              empty_cache=lambda: None)
        torch = SimpleNamespace(cuda=gpu, manual_seed=lambda _: None,
                                no_grad=contextlib.nullcontext)
        sampler = SimpleNamespace(start=lambda: None, stop=lambda **kwargs: dict(energy_j=None),
                                  mean_watts=lambda: 0, available=lambda: False)
        ticks = itertools.count(step=0.01)
        with patch.object(bench, 'torch', torch), \
             patch.object(bench, 'load_model', return_value=(Tokenizer(), SimpleNamespace(generate=generate))), \
             patch.object(bench, 'PowerSampler', return_value=sampler), \
             patch.object(bench, 'detect_gpu_name', return_value='test'), \
             patch.object(bench.time, 'perf_counter', side_effect=lambda: next(ticks)):
            row = bench.run_combo('model', 'float16', 1, 'prompt', 8, 8,
                                  4, 0.02, 0.1, revision='a'*40)
        self.assertTrue(calls)
        self.assertTrue(events)
        self.assertEqual(row['model_revision'], 'a'*40)
        self.assertEqual(row['tokenizer_revision'], 'a'*40)
        self.assertEqual(row['generation_mode'], 'greedy_fixed_length')
        self.assertEqual(row['generated_tokens'], row['requests'] * 4)
        self.assertGreater(row['time_s'], 0)
        self.assertEqual(calls[-1]['min_new_tokens'], 4)
        self.assertEqual(calls[-1]['pad_token_id'], 0)

    def test_multimodal_batch_passes_one_fixed_image_per_prompt(self):
        try:
            import PIL
        except ImportError:
            self.skipTest('Pillow is unavailable')
        processor = Mock()
        processor.apply_chat_template.return_value = 'formatted'
        batch = processor.return_value.to.return_value
        inputs, prompt = bench.encode_image_batch(processor, 'Describe the image', 224, 2,
                                                  'cuda:0', 'float16')
        self.assertIs(inputs, batch)
        self.assertEqual(prompt, 'formatted')
        self.assertEqual(processor.call_args.kwargs['text'], ['formatted', 'formatted'])
        images = processor.call_args.kwargs['images']
        self.assertEqual([image.size for image in images], [(224, 224), (224, 224)])
        self.assertEqual(images[0].tobytes(), images[1].tobytes())
        self.assertEqual(processor.apply_chat_template.call_args.args[0][0]['content'][0],
                         {'type': 'image'})

    def test_aggregate_uses_common_window_latest_completion(self):
        rows = [dict(status='ok', time_s=seconds, requests=count,
                     generated_tokens=count*4, timing_method='synchronized_v1')
                for seconds, count in ((2, 10), (4, 20))]
        row = bench.aggregate_rows(rows, 2, 1, 'replicated')
        self.assertEqual(row['reqs_per_s'], 7.5)
        self.assertEqual(row['gen_tokens_per_s'], 30)
        self.assertEqual(row['timing_method'], 'synchronized_v1')

    def test_unavailable_power_is_null_and_not_efficiency_zero(self):
        rows = [dict(status='ok', time_s=2, requests=10, generated_tokens=40,
                     mean_power_w=None, gen_tokens_per_watt=None,
                     power_sampler_available=False)] * 2
        row = bench.aggregate_rows(rows, 2, 1, 'replicated')
        self.assertIsNone(row['mean_power_w'])
        self.assertIsNone(row['gen_tokens_per_watt'])
