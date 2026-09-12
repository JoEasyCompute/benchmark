import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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
