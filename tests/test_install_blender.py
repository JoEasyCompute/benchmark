import hashlib
import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BlenderInstallerTest(unittest.TestCase):
    def run_installer(self, binary_body, *args, existing='4.5.12', corrupt_checksum=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / 'tools'
            tools.mkdir()
            release = root / 'release'
            release.mkdir()
            archive = release / 'blender-5.3.0-linux-x64.tar.xz'
            binary = ('#!/bin/sh\n' + binary_body).encode()
            with tarfile.open(archive, 'w:xz') as bundle:
                entry = tarfile.TarInfo('blender-5.3.0-linux-x64/blender')
                entry.mode = 0o755
                entry.size = len(binary)
                bundle.addfile(entry, io.BytesIO(binary))
            checksum = '0' * 64 if corrupt_checksum else hashlib.sha256(archive.read_bytes()).hexdigest()
            (release / 'blender-5.3.0.sha256').write_text(f'{checksum}  {archive.name}\n')
            commands = {
                'uname': '#!/bin/sh\ncase "$1" in -s) echo Linux;; -m) echo x86_64;; esac\n',
                'curl': '#!/bin/sh\nwhile [ "$1" != -o ]; do shift; done\n'
                        'cp "$TEST_RELEASE/${3##*/}" "$2"\n',
            }
            for name, body in commands.items():
                tool = tools / name
                tool.write_text(body)
                tool.chmod(0o755)
            launcher = tools / 'blender'
            original = root / 'old-blender'
            original.write_text(f'#!/bin/sh\necho "Blender {existing}"\n')
            original.chmod(0o755)
            launcher.symlink_to(original)
            env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'],
                       INSTALL_ROOT=str(root / 'opt'), BIN_DIR=str(tools),
                       CACHE_DIR=str(root), TEST_RELEASE=str(release))
            for name in ('BLENDER_VERSION', 'BLENDER_SERIES'):
                env.pop(name, None)
            result = subprocess.run(['bash', str(ROOT / 'install_blender.sh'), *args],
                                    env=env, capture_output=True, text=True)
            result.launcher_preserved = launcher.is_symlink() and launcher.readlink() == original
            result.version = subprocess.check_output([launcher, '--version'], text=True).strip()
            return result

    def test_failed_startup_preserves_old_launcher(self):
        result = self.run_installer('echo "error loading libXfixes.so.3" >&2\nexit 127\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.launcher_preserved)
        self.assertIn('libXfixes.so.3', result.stderr)
        self.assertIn('--install-system-deps', result.stderr)

    def test_default_install_upgrades_launcher(self):
        result = self.run_installer('echo "Blender 5.3.0"\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(result.launcher_preserved)
        self.assertEqual(result.version, 'Blender 5.3.0')
        self.assertLess(result.stdout.index('Version check:'), result.stdout.index('Installed Blender'))

    def test_ensure_reuses_matching_blender(self):
        result = self.run_installer('exit 1\n', '--ensure', existing='5.3.0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.launcher_preserved)
        self.assertEqual(Path(result.stdout.strip()).name, 'blender')
        self.assertNotIn('Downloading', result.stderr)

    def test_ensure_upgrades_mismatched_blender(self):
        result = self.run_installer('echo "Blender 5.3.0"\n', '--ensure')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(result.launcher_preserved)
        self.assertEqual(result.version, 'Blender 5.3.0')
        self.assertEqual(Path(result.stdout.strip()).name, 'blender')

    def test_wrong_version_cannot_replace_the_previous_launcher(self):
        result = self.run_installer('echo "Blender 5.2.4"\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.launcher_preserved)
        self.assertIn('Expected Blender 5.3.0', result.stderr)

    def test_bad_checksum_cannot_replace_the_previous_launcher(self):
        result = self.run_installer('echo "Blender 5.3.0"\n', corrupt_checksum=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.launcher_preserved)
