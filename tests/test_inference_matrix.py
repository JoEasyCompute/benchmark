import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from config_utils import load_config
from inference_matrix import execute, inference_cases, prepare_runtime


class InferenceMatrixTest(unittest.TestCase):
    def test_default_matrix_covers_text_vision_and_tensor_parallel(self):
        cfg = load_config(Path(__file__).parents[1] / 'config.yaml')
        cases = inference_cases(cfg, 8)
        self.assertEqual(len({case['model'] for case in cases}), 8)
        self.assertEqual([case['tensor_parallel_sizes'][0] for case in cases if case['backend'] == 'vllm'],
                         [4, 8, 8])
        self.assertEqual({case.get('image_size') for case in cases if case.get('image_size')}, {224, 448})
        text = [case for case in cases if case['model'] == 'Qwen/Qwen2.5-7B-Instruct']
        self.assertEqual([case['multi_gpu_mode'] for case in text], ['single', 'replicated'])

    def test_smoke_and_baseline_select_bounded_cases(self):
        cfg = load_config(Path(__file__).parents[1] / 'config.yaml')
        cfg.update(smoke_mode=True, benchmark_profile='single_gpu_baseline')
        cases = inference_cases(cfg, 1)
        self.assertTrue(all(case['batch_sizes'] == [1] for case in cases))
        self.assertTrue(all(case['multi_gpu_mode'] == 'single' for case in cases))
        self.assertFalse(any(case['backend'] == 'vllm' for case in cases))

    def test_execution_continues_after_gated_and_unsupported_cases(self):
        cfg = dict(gpu_include=[0, 1], gpu_backend='amd', repeat=1, llm_infer_matrix=[
            {'model': 'gated/model'}, {'model': 'large/model', 'tensor_parallel_sizes': [8]},
            {'model': 'public/model'},
        ])
        class GatedRepoError(Exception):
            pass

        def run_case(command, cwd, **kwargs):
            path = cwd / 'results/metrics.jsonl'
            path.parent.mkdir()
            path.write_text(json.dumps({'status': 'ok', 'gpu_count': 1}) + '\n')
            return type('Result', (), {'returncode': 0})()

        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder)
            (run_dir / 'logs').mkdir()
            (run_dir / 'results').mkdir()
            with patch('inference_matrix.resolve_revision', side_effect=[GatedRepoError(), 'a' * 40]), \
                 patch('inference_matrix.subprocess.run', side_effect=run_case) as runner:
                prepare_runtime(cfg, run_dir)
                self.assertEqual(execute(cfg, run_dir, 0, 1), 0)
            rows = [json.loads(line) for line in (run_dir / 'results/metrics.jsonl').read_text().splitlines()]
        self.assertEqual([row['status'] for row in rows], ['skipped'] * 3 + ['ok'] * 2)
        self.assertEqual(runner.call_count, 2)
        self.assertIn('Hugging Face authorization', rows[0]['detail'])
        self.assertIn('uses CUDA', rows[2]['detail'])
