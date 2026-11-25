"""Provision the pinned Blender Open Data scene used by the standard suite."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

ARCHIVE_URL = ('https://download.blender.org/release/BlenderBenchmark2.0/scenes/'
               'barbershop_interior.tar.bz2')
ARCHIVE_SHA256 = '099f581574457791e8cc753b2bf350c42fcdedb9ddcf3499909d119f0ab9137f'
SCENE = 'barbershop_interior.blend'


def ensure_barbershop(scenes_dir):
    target = scenes_dir / 'barbershop_interior'
    files = {'background.png': 'background.png', 'main.blend': SCENE}
    if all((target / name).is_file() for name in files.values()):
        return

    with tempfile.TemporaryDirectory(prefix='.barbershop-', dir=scenes_dir) as directory:
        staged = Path(directory)
        archive = staged / 'scene.tar.bz2'
        digest = hashlib.sha256()
        print('[BLENDER] Downloading the Agent 327 Barbershop benchmark scene', flush=True)
        with urlopen(ARCHIVE_URL, timeout=60) as response, archive.open('wb') as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != ARCHIVE_SHA256:
            raise ValueError('Barbershop archive SHA-256 does not match the pinned benchmark asset')
        with tarfile.open(archive, 'r:bz2') as bundle:
            for source_name, target_name in files.items():
                member = bundle.getmember(f'barbershop_interior/{source_name}')
                if not member.isfile():
                    raise ValueError(f'Expected a regular file in Barbershop archive: {source_name}')
                with bundle.extractfile(member) as source, (staged / target_name).open('wb') as output:
                    shutil.copyfileobj(source, output)
        target.mkdir(exist_ok=True)
        for name in files.values():
            (staged / name).replace(target / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes-dir', type=Path, required=True)
    parser.add_argument('--scenes-json', required=True)
    args = parser.parse_args()
    if SCENE in (json.loads(args.scenes_json) or []):
        ensure_barbershop(args.scenes_dir)


if __name__ == '__main__':
    main()
