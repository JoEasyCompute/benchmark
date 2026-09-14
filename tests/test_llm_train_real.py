import math
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import llm_train_real as training


class Tokens(list):
    def clone(self):
        return Tokens(self)


class Scalar:
    def __init__(self, value):
        self.value = value
    def float(self):
        return self
    def backward(self):
        pass


class TorchStub:
    @staticmethod
    def isfinite(value):
        return SimpleNamespace(all=lambda: math.isfinite(value.value))


class TrainingTests(unittest.TestCase):
    def test_low_precision_run_loads_fp32_master_parameters(self):
        loaded_dtypes = []
        def load_model(*args, **kwargs):
            loaded_dtypes.append(kwargs['torch_dtype'])
            raise RuntimeError('deliberate model load stop')
        torch = SimpleNamespace(
            float16='fp16', bfloat16='bf16', float32='fp32',
            cuda=SimpleNamespace(is_available=lambda: True),
            version=SimpleNamespace(hip='test'), manual_seed=lambda seed: None,
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace())),
            device=lambda name: name)
        transformers = SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model))
        with patch.dict(sys.modules, {'torch': torch, 'transformers': transformers}), \
             patch.object(training, 'resolve_revision', return_value='fixed'):
            for dtype in ('fp16', 'bf16', 'fp32'):
                result = training.run({'enabled': True, 'model': 'tiny', 'dtype': dtype,
                                       'batch_size': 1, 'seq_len': 4, 'steps': 1})
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['failure_stage'], 'model_load')
                self.assertEqual(result['parameter_dtype'], 'fp32')
                self.assertEqual(result['gradient_scaling'], dtype == 'fp16')
        self.assertEqual(loaded_dtypes, ['fp32', 'fp32', 'fp32'])

    def test_amp_unscales_before_checking_and_updating(self):
        events = []
        class Model:
            def __call__(self, **kw):
                events.append('forward')
                return SimpleNamespace(loss=Scalar(1.0))
            def parameters(self):
                return []
        scaler = SimpleNamespace(
            scale=lambda loss: (events.append('scale') or loss),
            unscale_=lambda opt: events.append('unscale'),
            step=lambda opt: events.append('step'),
            update=lambda: events.append('update'))
        optimizer = SimpleNamespace(zero_grad=lambda **kw: None)
        training.training_step(Model(), optimizer, Tokens([1, 2]), TorchStub,
                               autocast_context=nullcontext, scaler=scaler)
        self.assertEqual(events, ['forward', 'scale', 'unscale', 'step', 'update'])

    def test_failure_diagnostics_identify_forward_loss(self):
        diagnostics = {}
        model = lambda **kw: SimpleNamespace(loss=Scalar(float('nan')))
        optimizer = SimpleNamespace(zero_grad=lambda **kw: None)
        with self.assertRaises(FloatingPointError):
            training.training_step(model, optimizer, Tokens([1, 2]), TorchStub,
                                   diagnostics=diagnostics)
        self.assertEqual(diagnostics['stage'], 'forward_loss')

    def test_scaled_overflow_retries_without_counting_skipped_update(self):
        updates, attempts = [], []
        parameter = SimpleNamespace(grad=Scalar(float('inf')))
        class Model:
            def __call__(self, **kw):
                attempts.append(1)
                return SimpleNamespace(loss=Scalar(1.0))
            def parameters(self):
                return [parameter]
        def update_scale():
            parameter.grad = Scalar(1.0)
        scaler = SimpleNamespace(
            scale=lambda loss: loss, unscale_=lambda opt: None,
            step=lambda opt: updates.append(1) if math.isfinite(parameter.grad.value) else None,
            update=update_scale)
        diagnostics = {}
        training.training_step(Model(), SimpleNamespace(zero_grad=lambda **kw: None),
                               Tokens([1, 2]), TorchStub, scaler=scaler,
                               diagnostics=diagnostics)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(updates, [1])
        self.assertEqual(diagnostics['overflow_retries'], 1)

    def test_persistent_overflow_stops_at_retry_limit(self):
        attempts = []
        class Model:
            def __call__(self, **kw):
                attempts.append(1)
                return SimpleNamespace(loss=Scalar(1.0))
            def parameters(self):
                return [SimpleNamespace(grad=Scalar(float('inf')))]
        scaler = SimpleNamespace(scale=lambda loss: loss, unscale_=lambda opt: None,
                                 step=lambda opt: None, update=lambda: None)
        with self.assertRaisesRegex(FloatingPointError, 'loss-scale retries'):
            training.training_step(Model(), SimpleNamespace(zero_grad=lambda **kw: None),
                                   Tokens([1, 2]), TorchStub, scaler=scaler,
                                   max_overflow_retries=2)
        self.assertEqual(len(attempts), 3)

    def test_causal_model_receives_unshifted_labels(self):
        class TinyCausal:
            def __call__(self, input_ids, labels):
                # Equivalent to an HF causal loss: shift once inside the model.
                self.targets = labels[1:]
                logits = [[0.0] * 8 for _ in input_ids[:-1]]
                for row, next_token in zip(logits, [3, 5, 7]):
                    row[next_token] = 2.0
                cross_entropy = sum(
                    math.log(sum(math.exp(value) for value in row)) - row[target]
                    for row, target in zip(logits, self.targets)) / len(self.targets)
                return SimpleNamespace(loss=Scalar(cross_entropy))
            def parameters(self):
                return []
        model = TinyCausal()
        optimizer = SimpleNamespace(zero_grad=lambda **kw: None, step=lambda: None)
        loss = training.training_step(model, optimizer, Tokens([2, 3, 5, 7]), TorchStub)
        self.assertEqual(model.targets, [3, 5, 7])
        self.assertAlmostEqual(loss.value, math.log(math.exp(2.0) + 7) - 2)
        self.assertEqual(training.supervised_tokens(2, 4), 6)

    def test_nonfinite_loss_prevents_optimizer_update(self):
        updates = []
        model = lambda **kw: SimpleNamespace(loss=Scalar(float('nan')))
        optimizer = SimpleNamespace(zero_grad=lambda **kw: None, step=lambda: updates.append(1))
        with self.assertRaisesRegex(FloatingPointError, 'loss'):
            training.training_step(model, optimizer, Tokens([1, 2]), TorchStub)
        self.assertEqual(updates, [])

    def test_disabled_and_distributed_skip_without_gpu_import(self):
        self.assertEqual(training.run({'enabled': False})['status'], 'skipped')
        self.assertEqual(training.run({'enabled': True, 'world_size': 2})['status'], 'skipped')

    def test_nonfinite_gradient_prevents_optimizer_update(self):
        class Model:
            def __call__(self, **kw):
                return SimpleNamespace(loss=Scalar(1.0))
            def parameters(self):
                return [SimpleNamespace(grad=Scalar(float('inf')))]
        updates = []
        optimizer = SimpleNamespace(zero_grad=lambda **kw: None, step=lambda: updates.append(1))
        with self.assertRaisesRegex(FloatingPointError, 'gradient'):
            training.training_step(Model(), optimizer, Tokens([1, 2]), TorchStub)
        self.assertEqual(updates, [])

    def test_supervised_tokens_rejects_empty_objective(self):
        with self.assertRaises(ValueError):
            training.supervised_tokens(1, 1)
