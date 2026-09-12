"""Blender driver and render-only measurement, with no third-party dependencies."""
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def settings_hash(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def select_devices(devices, backend, mode):
    selected = []
    for device in devices:
        device.use = device.type == backend and (mode == 'all' or not selected)
        if device.use:
            selected.append(device)
    if not selected:
        raise ValueError('No available Blender GPU for backend ' + backend)
    return selected


def telemetry_devices(selected, inventory):
    """Match Cycles PCI identifiers to telemetry IDs, never device-list order."""
    result = []
    for device in selected:
        buses = re.findall(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', device.id)
        if len(buses) != 1:
            return None
        bus = buses[0].lower().lstrip('0')
        matches = [identifier for identifier, pci in inventory
                   if pci.lower().lstrip('0') == bus]
        if len(matches) != 1 or matches[0] in result:
            return None
        result.append(matches[0])
    return result


def energy_sampler(selected, backend):
    if backend not in ('CUDA', 'OPTIX', 'HIP'):
        return None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from energy import EnergySampler
        if backend == 'HIP':
            raw = subprocess.check_output(['rocm-smi', '--showbus', '--json'], text=True, timeout=5)
            payload = json.loads(raw)
            inventory = [(card[4:], values['PCI Bus']) for card, values in payload.items()
                         if card.startswith('card') and card[4:].isdigit()
                         and isinstance(values, dict) and isinstance(values.get('PCI Bus'), str)]
        else:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,pci.bus_id',
                                           '--format=csv,noheader,nounits'], text=True, timeout=5)
            inventory = [tuple(item.strip() for item in line.split(','))
                         for line in raw.splitlines() if line.count(',') == 1]
        ids = telemetry_devices(selected, inventory)
        if not ids:
            return None
        sampler = EnergySampler('amd' if backend == 'HIP' else 'nvidia')
        sampler.device_ids = ids
        return sampler
    except (ImportError, OSError, ValueError, subprocess.SubprocessError):
        return None


def render_in_blender():
    entered = time.monotonic()
    import bpy
    output = Path(os.environ['BLENDER_TIMING_OUTPUT'])
    try:
        backend = os.environ.get('BLENDER_GPU_BACKEND', 'CUDA')
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = backend
        prefs.get_devices()
        selected = select_devices(prefs.devices, backend, os.environ['BLENDER_BENCH_MODE'])
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
        scene.frame_set(int(os.environ.get('FRAME', '1')))
        settings = dict(engine=scene.render.engine, samples=scene.cycles.samples,
                        width=scene.render.resolution_x, height=scene.render.resolution_y,
                        resolution_percentage=scene.render.resolution_percentage,
                        denoising=scene.cycles.use_denoising,
                        denoiser=getattr(scene.cycles, 'denoiser', None),
                        use_adaptive_sampling=scene.cycles.use_adaptive_sampling,
                        adaptive_threshold=scene.cycles.adaptive_threshold,
                        adaptive_min_samples=scene.cycles.adaptive_min_samples,
                        seed=scene.cycles.seed, frame=scene.frame_current,
                        max_bounces=scene.cycles.max_bounces,
                        use_persistent_data=scene.render.use_persistent_data)
        sampler = energy_sampler(selected, backend)
        if sampler:
            sampler.start()
        start = time.perf_counter()
        try:
            result = bpy.ops.render.render(write_still=False)
        finally:
            ended = time.perf_counter()
            energy = sampler.stop(start, ended) if sampler else {
                'energy_j': None, 'power_sampler_available': False,
                'energy_unavailable_reason': 'Blender device IDs lack verified telemetry mapping'}
        elapsed = ended - start
        if 'FINISHED' not in result or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('Blender render did not finish successfully')
        row = dict(status='ok', **settings, render_settings_sha256=settings_hash(settings),
                   blender_version=bpy.app.version_string, num_gpus=len(selected), gpu_count=len(selected),
                   gpu_backend='amd' if backend == 'HIP' else 'nvidia',
                   device_ids=[d.id for d in selected], device_names=[d.name for d in selected],
                   render_time_s=elapsed, time_s=elapsed,
                   startup_load_time_s=entered-float(os.environ['BLENDER_PROCESS_START']),
                   timing_method='blender_render_operator_v1', **energy)
    except Exception as exc:
        row = dict(status='failed', error=str(exc))
    output.write_text(json.dumps(row, allow_nan=False))


def run_scene(binary, scene, mode, env):
    row = dict(suite='blender', scene=scene.name, mode=mode,
               backend=env.get('BLENDER_GPU_BACKEND', 'CUDA'),
               repeat_index=int(env.get('REPEAT_INDEX', '1')),
               repeat_count=int(env.get('REPEAT_COUNT', '1')),
               benchmark_profile=env.get('BENCHMARK_PROFILE', 'general'))
    if not scene.is_file():
        return dict(row, status='failed', error='missing scene')
    with scene.open('rb') as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    row['scene_sha256'] = digest.hexdigest()
    with tempfile.TemporaryDirectory(prefix='blender-timing-') as temporary:
        output = Path(temporary) / 'timing.json'
        process_env = dict(env, BLENDER_TIMING_OUTPUT=str(output), BLENDER_BENCH_MODE=mode,
                           BLENDER_PROCESS_START=str(time.monotonic()))
        start = float(process_env['BLENDER_PROCESS_START'])
        try:
            proc = subprocess.run([binary, '-b', str(scene), '--python', str(Path(__file__).resolve()),
                                   '--', '--inside-blender'], env=process_env, check=False)
            row.update(end_to_end_time_s=time.monotonic()-start, rc=proc.returncode)
            if proc.returncode:
                raise ValueError('Blender process failed with exit code ' + str(proc.returncode))
            measurement = json.loads(output.read_text())
            if measurement.get('status') != 'ok':
                raise ValueError(measurement.get('error', 'Blender render failed'))
            elapsed = measurement.get('render_time_s')
            if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError('Missing or invalid render-only timing')
            if measurement.get('num_gpus', 0) < 1:
                raise ValueError('No GPU rendered the scene')
            row.update(measurement)
        except (OSError, ValueError) as exc:
            row.update(status='failed', error=str(exc))
    return row


def main():
    env = dict(os.environ)
    if env.get('BLENDER_ENABLED', '1') != '1':
        return
    scenes_dir = Path(env.get('SCENES_DIR', 'assets/blender'))
    results_dir = Path(env.get('RESULTS_DIR', str(scenes_dir / 'results')))
    results = Path(env.get('RESULTS_JSON', str(results_dir / 'bench_results_cuda.json')))
    metrics = Path(env.get('METRICS_JSONL', str(results_dir / 'metrics.jsonl')))
    scenes = json.loads(env.get('BLENDER_SCENES_JSON', env.get('SCENES_JSON', '[]')))
    if not isinstance(scenes, list) or any(not isinstance(s, str) for s in scenes):
        raise ValueError('Blender scenes must be a JSON list of paths')
    scenes = scenes or ['BMW27.blend', 'classroom.blend']
    binary = env.get('BLENDER_BIN') or shutil.which('blender') or 'blender'
    rows = []
    for name in scenes:
        scene = Path(name)
        if not scene.is_file():
            scene = scenes_dir / name
        if not scene.is_file() and scenes_dir.is_dir():
            matches = sorted(scenes_dir.glob('*/' + Path(name).name))
            if matches:
                scene = matches[0]
        modes = ['single'] if env.get('BENCHMARK_PROFILE') == 'single_gpu_baseline' else ['single', 'all']
        for mode in modes:
            rows.append(run_scene(binary, scene, mode, env))
    results.parent.mkdir(parents=True, exist_ok=True)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(rows, indent=2, allow_nan=False) + '\n')
    with metrics.open('a') as sink:
        for row in rows:
            sink.write(json.dumps(row, allow_nan=False) + '\n')
    return 1 if any(row['status'] == 'failed' for row in rows) else 0


if __name__ == '__main__':
    if '--inside-blender' in sys.argv:
        render_in_blender()
    else:
        raise SystemExit(main())
