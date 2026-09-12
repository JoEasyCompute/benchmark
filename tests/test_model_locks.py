import unittest

from lock_model_revisions import lock_revisions


class ModelLocksTest(unittest.TestCase):
    def test_enabled_local_workloads_use_one_resolved_commit(self):
        cfg = {'llm_infer': {'model': 'model-a'}, 'sd_infer': {'enabled': False},
               'llm_train_real': {'enabled': True, 'model': 'model-a'},
               'llm_serve': {'enabled': True, 'provider': 'transformers'}}
        calls = []
        def resolve(model, revision=None, filename='config.json'):
            calls.append((model, revision, filename))
            return 'a' * 40
        locked, manifest = lock_revisions(cfg, resolve)
        self.assertEqual(len(calls), 1)
        for name in ('llm_infer', 'llm_train_real', 'llm_serve'):
            self.assertEqual(locked[name]['revision'], 'a' * 40)
        self.assertNotIn('revision', cfg['llm_infer'])

    def test_external_server_revision_is_never_claimed_as_verified(self):
        cfg = {'llm_infer': {'enabled': False}, 'sd_infer': {'enabled': False},
               'llm_serve': {'enabled': True, 'endpoint': 'http://localhost:8000/v1/completions', 'model': 'm'}}
        def resolve(*args, **kwargs):
            self.fail('external endpoint must not resolve an unrelated local model')
        locked, manifest = lock_revisions(cfg, resolve)
        self.assertFalse(manifest['llm_serve']['verified'])
