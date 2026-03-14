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
            repeat: 1
            results_dir: results
            preflight:
              machine_state_strict: false
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
              model: "Qwen/Qwen2.5-0.5B"
              dtype: float16
              prompt_len: 64
              output_len: 32
              batch_sizes: [1]
              tensor_parallel_sizes: [1]
            sd_infer:
              model: "stabilityai/stable-diffusion-2-1"
              steps: 4
              sizes: [512]
              per_gpu_batch: 1
              multi_gpu_mode: "single"
              emit_worker_rows: false
            blender:
              enabled: true
              scenes: []
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validation passed", result.stdout)

    def test_unknown_key_fails(self):
        result = self.run_validate(
            """
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
              model: "Qwen/Qwen2.5-0.5B"
              dtype: float16
              prompt_len: 64
              output_len: 32
              batch_sizes: [1]
              tensor_parallel_sizes: [1]
            sd_infer:
              model: "stabilityai/stable-diffusion-2-1"
              steps: 4
              sizes: [512]
              per_gpu_batch: 1
              multi_gpu_mode: "single"
            blender:
              enabled: true
              scenes: []
            """
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported key 'bogus'", result.stderr)


if __name__ == "__main__":
    unittest.main()
