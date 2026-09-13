from pathlib import Path
import unittest


class BlenderInstallerTest(unittest.TestCase):
    def test_wrapper_removes_old_symlink_before_redirection(self):
        script = (Path(__file__).resolve().parents[1] / 'install_blender.sh').read_text()
        remove = script.index('rm -f "$BIN_DIR/blender"')
        wrapper = script.index('cat > "$BIN_DIR/blender"')
        self.assertLess(remove, wrapper)
        self.assertIn('export LD_LIBRARY_PATH="$TARGET_DIR/lib', script)
