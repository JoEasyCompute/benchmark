import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
SCRIPT = ROOT / "estimate_runtime.py"


class EstimateRuntimeTest(unittest.TestCase):
    def test_writes_runtime_json_and_uses_visible_gpu_env(self):
        config_text = textwrap.dedent(
            """
            repeat: 2
            llm_train:
              dtype: bf16
              hidden_size: 1024
              n_layers: 12
              n_heads: 16
              seq_len: 512
              batch_size: 4
              steps: 10
            llm_train_real:
              enabled: true
              model: "Qwen/Qwen2.5-0.5B"
              dtype: fp16
              seq_len: 128
              batch_size: 1
              steps: 3
            llm_infer:
              model: "Qwen/Qwen2.5-0.5B"
              dtype: float16
              prompt_len: 64
              output_len: 32
              batch_sizes: [1, 2]
              tensor_parallel_sizes: [1, 2]
            sd_infer:
              model: "stabilityai/stable-diffusion-2-1"
              steps: 20
              sizes: [512, 1024]
              per_gpu_batch: 1
              multi_gpu_mode: replicated
            blender:
              enabled: true
              scenes:
                - BMW27.blend
            """
        ).strip() + "\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            json_path = Path(tmpdir) / "runtime.json"
            config_path.write_text(config_text)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0,1,2"

            result = subprocess.run(
                [str(PYTHON), str(SCRIPT), "--config", str(config_path), "--json-out", str(json_path)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(json_path.read_text())
            self.assertEqual(payload["repeat"], 2)
            self.assertEqual(payload["visible_gpus"], 3)
            self.assertEqual(payload["total_minutes"]["min"], round(payload["total"]["min_s"] / 60.0, 1))
            suites = {section["suite"]: section for section in payload["sections"]}
            self.assertEqual(suites["llm_infer"]["combos"], 4)
            self.assertEqual(suites["sd_infer"]["multi_gpu_mode"], "replicated")
            self.assertIn("[ESTIMATE] total runtime:", result.stdout)


if __name__ == "__main__":
    unittest.main()
