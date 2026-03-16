import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
SCRIPT = ROOT / "validate_config.py"


class ValidateConfigTest(unittest.TestCase):
    def run_validate(self, config_text):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(textwrap.dedent(config_text).strip() + "\n")
            return subprocess.run(
                [str(PYTHON), str(SCRIPT), "--config", str(config_path)],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_valid_config_passes(self):
        result = self.run_validate(
            """
            gpu_backend: amd
            repeat: 1
            results_dir: results
            preflight:
              machine_state_strict: false
              blender_strict: true
            llm_train:
              dtype: bf16
              hidden_size: 1024
              n_layers: 12
              n_heads: 16
              seq_len: 512
              batch_size: 4
              steps: 2
            llm_train_real:
              enabled: false
              model: "Qwen/Qwen2.5-0.5B"
              dtype: fp16
              seq_len: 128
              batch_size: 1
              steps: 1
            llm_infer:
              backend: transformers
              model: "Qwen/Qwen2.5-0.5B"
              dtype: float16
              prompt_len: 64
              output_len: 32
              batch_sizes: [1]
              tensor_parallel_sizes: [1]
              multi_gpu_mode: replicated
            sd_infer:
              model: "stabilityai/stable-diffusion-2-1"
              steps: 4
              sizes: [512]
              per_gpu_batch: 1
              multi_gpu_mode: "single"
              emit_worker_rows: false
            blender:
              enabled: true
              require_installed: true
              backend: hip
              scenes: []
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validation passed", result.stdout)

    def test_unknown_key_fails(self):
        result = self.run_validate(
            """
            gpu_backend: invalid
            repeat: 1
            results_dir: results
            bogus: true
            llm_train:
              dtype: bf16
              hidden_size: 1024
              n_layers: 12
              n_heads: 16
              seq_len: 512
              batch_size: 4
              steps: 2
            llm_infer:
              backend: invalid
              model: "Qwen/Qwen2.5-0.5B"
              dtype: float16
              prompt_len: 64
              output_len: 32
              batch_sizes: [1]
              tensor_parallel_sizes: [1]
              multi_gpu_mode: invalid
            sd_infer:
              model: "stabilityai/stable-diffusion-2-1"
              steps: 4
              sizes: [512]
              per_gpu_batch: 1
              multi_gpu_mode: "single"
            blender:
              enabled: true
              require_installed: invalid
              scenes: []
            """
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("gpu_backend must be one of auto, nvidia, amd", result.stderr)
        self.assertIn("llm_infer.backend must be one of transformers, vllm", result.stderr)
        self.assertIn("llm_infer.multi_gpu_mode must be 'single' or 'replicated'", result.stderr)
        self.assertIn("blender.require_installed must be true or false", result.stderr)
        self.assertIn("unsupported key 'bogus'", result.stderr)


if __name__ == "__main__":
    unittest.main()
