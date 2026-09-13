from pathlib import Path
import unittest


class BlenderInstallerTest(unittest.TestCase):
    def test_wrapper_removes_old_symlink_before_redirection(self):
        script = (Path(__file__).resolve().parents[1] / 'install_blender.sh').read_text()
        remove = script.index('rm -f "$BIN_DIR/blender"')
        wrapper = script.index('cat > "$BIN_DIR/blender"')
        self.assertLess(remove, wrapper)
        self.assertIn('export LD_LIBRARY_PATH="$TARGET_DIR/lib', script)

    def test_default_version_supports_rdnA4_cycles(self):
        script = (Path(__file__).resolve().parents[1] / 'install_blender.sh').read_text()
        self.assertIn('BLENDER_VERSION="${BLENDER_VERSION:-4.5.12}"', script)
        self.assertIn('BLENDER_VERSION=', (Path(__file__).resolve().parents[1] / 'README.md').read_text())
