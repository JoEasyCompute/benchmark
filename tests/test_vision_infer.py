from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import vision_infer


class VisionIdentityTests(unittest.TestCase):
    def test_default_weights_are_versioned(self):
        self.assertEqual(vision_infer.weight_member('resnet18', None),
                         'ResNet18_Weights.IMAGENET1K_V1')

    def test_rejects_mutable_or_random_weights(self):
        for name in ('DEFAULT', 'ResNet18_Weights.DEFAULT', 'none'):
            with self.assertRaises(ValueError):
                vision_infer.weight_member('resnet18', name)

    def test_unknown_model_requires_explicit_weights(self):
        with self.assertRaises(ValueError):
            vision_infer.weight_member('unrecognized', None)
