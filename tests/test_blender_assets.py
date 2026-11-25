import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import blender_assets


class BlenderAssetTest(unittest.TestCase):
    def test_verified_archive_installs_scene_and_background_once(self):
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode='w:bz2') as source:
            for name, data in [('main.blend', b'scene'), ('background.png', b'background')]:
                entry = tarfile.TarInfo(f'barbershop_interior/{name}')
                entry.size = len(data)
                source.addfile(entry, io.BytesIO(data))
        data = archive.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(blender_assets, 'ARCHIVE_SHA256', digest), \
                patch.object(blender_assets, 'urlopen', return_value=io.BytesIO(data)) as download:
            root = Path(directory)
            blender_assets.ensure_barbershop(root)
            blender_assets.ensure_barbershop(root)
            self.assertEqual((root / 'barbershop_interior/barbershop_interior.blend').read_bytes(), b'scene')
            self.assertEqual((root / 'barbershop_interior/background.png').read_bytes(), b'background')
        download.assert_called_once()
