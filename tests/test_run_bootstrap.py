import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BootstrapTest(unittest.TestCase):
    def test_setup_uses_same_resolver_as_runtime_gate(self):
        with tempfile.TemporaryDirectory(prefix='stack-select-') as directory:
            root = Path(directory)
            tools = root / 'tools'
            tools.mkdir()
            venv = root / 'venv'
            bin_dir = venv / 'bin'
            bin_dir.mkdir(parents=True)
            (bin_dir / 'activate').write_text(f'export PATH="{bin_dir}:$PATH"\n')
            python = bin_dir / 'python'
            python.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                              'if sys.argv[1:] == ["-"]: print("3.12")\n'
                              'elif "runtime_resolver.py" in sys.argv[1]:\n'
                              ' with open(os.environ["RESOLVER_LOG"], "w") as f: json.dump(sys.argv[1:], f)\n')
            python.chmod(0o755)
            for name, output in (('uname', 'Linux'), ('rocm-smi', '{"card0":{}}')):
                command = tools / name
                command.write_text(f"#!/bin/sh\nprintf '%s\\n' '{output}'\n")
                command.chmod(0o755)
            log = root / 'resolver.json'
            env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'],
                       VENV_DIR=str(venv), PYTHON_BIN=sys.executable, GPU_BACKEND='amd',
                       ALLOW_UNVERIFIED_HOST='1', RESOLVER_LOG=str(log))
            result = subprocess.run(['bash', str(ROOT / 'env_setup.sh')], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            args = json.loads(log.read_text())
            self.assertEqual(args, [str(ROOT / 'runtime_resolver.py'), '--backend', 'amd',
                                   '--install', '--venv', str(venv), '--allow-unverified-host'])

    def test_runner_passes_detected_backend_and_absolute_venv_to_setup(self):
        with tempfile.TemporaryDirectory(prefix='benchmark-bootstrap-') as directory:
            project = Path(directory) / 'project'
            project.mkdir()
            for name in ('run_all.sh', 'config_utils.py', 'gpu_platform.py', 'suite_config.py',
                         'validate_config.py', 'config.yaml'):
                shutil.copy2(ROOT / name, project / name)
            executable_dir = project / '.venv/bin'
            executable_dir.mkdir(parents=True)
            python = executable_dir / 'python'
            # Simulate an incompatible GPU stack while keeping YAML readable.
            python.write_text('#!/bin/sh\nif [ "$1" = "-" ]; then exit 1; fi\n'
                              f'exec "{sys.executable}" "$@"\n')
            python.chmod(0o755)
            (project / 'env_setup.sh').write_text(
                '#!/bin/sh\nprintf "%s\\n%s\\n" "$GPU_BACKEND" "$VENV_DIR" > "$VENV_DIR/setup-choice"\nexit 7\n')
            tools = Path(directory) / 'tools'
            tools.mkdir()
            for name, text in (
                ('nvidia-smi', '#!/bin/sh\nexit 1\n'),
                ('rocm-smi', '#!/bin/sh\nprintf \'{"card1":{"Device ID":"0xab"}}\\n\'\n'),
            ):
                tool = tools / name
                tool.write_text(text)
                tool.chmod(0o755)
            env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'])
            env.pop('HIP_VISIBLE_DEVICES', None)
            env.pop('CUDA_VISIBLE_DEVICES', None)
            result = subprocess.run(['bash', str(project / 'run_all.sh'), '--baseline'],
                                    cwd=directory, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
            choice = (project / '.venv/setup-choice').read_text().splitlines()
            self.assertEqual(choice, ['amd', str(project / '.venv')])

    def test_manual_setup_no_gpu_fails_before_environment_creation(self):
        with tempfile.TemporaryDirectory(prefix='benchmark-setup-') as directory:
            tools = Path(directory) / 'tools'
            tools.mkdir()
            for name, output in (('uname', 'Linux'), ('nvidia-smi', ''), ('rocm-smi', '{}')):
                tool = tools / name
                tool.write_text(f'#!/bin/sh\nprintf "%s\\n" \'{output}\'\n')
                tool.chmod(0o755)
            target = Path(directory) / 'new-venv'
            env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'],
                       VENV_DIR=str(target), PYTHON_BIN=sys.executable, GPU_BACKEND='auto')
            result = subprocess.run(['bash', str(ROOT / 'env_setup.sh')], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('No responding', result.stderr)
            self.assertFalse(target.exists())
