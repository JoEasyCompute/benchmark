import unittest
from types import SimpleNamespace

from check_system_requirements import optional_capabilities, parse_missing_shared_libs


class CapabilityTest(unittest.TestCase):
    def test_external_vllm_does_not_require_local_install(self):
        called = []
        def load(name):
            called.append(name)
            raise ImportError(name)
        checks, warnings = optional_capabilities(
            {'llm_serve': {'enabled': True, 'provider': 'vllm', 'endpoint': 'http://localhost:8000/v1/completions'}},
            'amd', importer=load)
        self.assertNotIn('vllm', called)
        self.assertTrue(any('external' in str(check) for check in checks))

    def test_enabled_missing_vision_dependency_has_explicit_warning(self):
        def load(name):
            raise ImportError('not installed')
        checks, warnings = optional_capabilities({'vision_infer': {'enabled': True}}, 'amd', importer=load)
        self.assertTrue(any('torchvision' in item for item in warnings))

    def test_attention_api_is_checked_when_requested(self):
        def load(name):
            return SimpleNamespace(nn=SimpleNamespace(functional=SimpleNamespace()))
        checks, warnings = optional_capabilities({'kernel_bench': {'enabled': True, 'cases': ['attention']}},
                                                'amd', importer=load)
        self.assertTrue(any('scaled_dot_product_attention' in item for item in warnings))

    def test_blender_missing_shared_objects_are_reported(self):
        missing = parse_missing_shared_libs('libSM.so.6 => not found\nlibXrender.so.1 => /lib/xrender.so\n')
        self.assertEqual(missing, ['libSM.so.6'])
        self.assertEqual(parse_missing_shared_libs(''), [])
