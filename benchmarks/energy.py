"""Board energy from timestamped samples; unavailable telemetry stays missing."""
import ctypes
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time

METHOD = 'board_power_trapezoid_v1'


def physical_devices(backend, indices, environ=None):
    env = os.environ if environ is None else environ
    key = 'HIP_VISIBLE_DEVICES' if backend == 'amd' else 'CUDA_VISIBLE_DEVICES'
    raw = env.get(key, env.get('CUDA_VISIBLE_DEVICES') if backend == 'amd' else None)
    visible = raw.split(',') if raw is not None else None
    return [visible[i].strip() if visible is not None else str(i) for i in indices]


def hip_pci_devices(indices):
    """Resolve logical ordinals using the HIP library already used by the workload.

    Loading a different system HIP runtime could apply a different visibility/order
    policy. If no unique loaded runtime can be identified, telemetry stays missing.
    """
    try:
        paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if '/libamdhip64.so' in line and line.split()[-1].startswith('/')}
        if len(paths) != 1:
            return None
        runtime = ctypes.CDLL(paths.pop())
        query = runtime.hipDeviceGetPCIBusId
        query.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        query.restype = ctypes.c_int
        buses = []
        for index in indices:
            bus = ctypes.create_string_buffer(32)
            if query(bus, len(bus), index) != 0:
                return None
            buses.append(bus.value.decode('ascii').lower())
        return buses
    except (OSError, AttributeError, ValueError, UnicodeError):
        return None


def match_rocm_devices(buses, payload):
    if not buses or not isinstance(payload, dict):
        return None
    devices = []
    for bus in buses:
        matches = [name[4:] for name, card in payload.items()
                   if re.fullmatch(r'card[0-9]+', name) and isinstance(card, dict)
                   and str(card.get('PCI Bus', '')).strip().lower() == bus.lower()]
        if len(matches) != 1:
            return None
        devices.append(matches[0])
    return devices if len(set(devices)) == len(devices) else None


def parse_rocm_power(payload, device_ids):
    if not isinstance(payload, dict):
        return None
    fields = ('Current Socket Graphics Package Power (W)', 'Average Graphics Package Power (W)',
              'Current Graphics Package Power (W)')
    values = []
    for device in device_ids:
        card = payload.get('card' + str(device), {})
        if not isinstance(card, dict):
            return None
        value = next((card[name] for name in fields if name in card), None)
        try:
            value = float(str(value).split()[0])
        except (TypeError, ValueError, IndexError):
            return None
        if not math.isfinite(value) or value < 0:
            return None
        values.append(value)
    return sum(values) if values else None


def integrate_samples(samples, started, ended, max_gap):
    if (not all(math.isfinite(value) for value in (started, ended, max_gap))
            or max_gap <= 0 or ended <= started or len(samples) < 2
            or any(not math.isfinite(t) for t, _ in samples)
            or any(b[0] <= a[0] for a, b in zip(samples, samples[1:]))):
        return None, 0.0
    energy = covered = 0.0
    for (left, a), (right, b) in zip(samples, samples[1:]):
        lo, hi = max(left, started), min(right, ended)
        if hi <= lo:
            continue
        if right <= left or right-left > max_gap or any(v is None or not math.isfinite(v) or v < 0 for v in (a, b)):
            continue
        start_w = a + (b-a)*(lo-left)/(right-left)
        end_w = a + (b-a)*(hi-left)/(right-left)
        energy += (start_w+end_w)*0.5*(hi-lo)
        covered += hi-lo
    coverage = min(1.0, covered/(ended-started))
    return (energy if coverage >= 1-1e-9 else None), coverage


class EnergySampler:
    def __init__(self, backend, device_indices=None, interval_s=0.5):
        if interval_s <= 0:
            raise ValueError('Power interval must be positive')
        self.backend, self.interval = backend, interval_s
        self.device_indices = device_indices if device_indices is not None else [0]
        self.device_ids = ([] if backend == 'amd' else
                           physical_devices(backend, self.device_indices))
        self.mapping_verified = backend != 'amd'
        self.unavailable_reason = None
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None
        self._handles = []
        self.result = None

    def _read(self):
        try:
            if self.backend == 'amd':
                if not self.mapping_verified:
                    return None
                raw = subprocess.check_output(['rocm-smi', '--showpower', '--json'], text=True,
                                              stderr=subprocess.DEVNULL, timeout=2)
                return parse_rocm_power(json.loads(raw), self.device_ids)
            if self._nvml is not None and len(self._handles) == len(self.device_ids):
                return sum(self._nvml.nvmlDeviceGetPowerUsage(h)/1000 for h in self._handles)
            if self.backend == 'nvidia':
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw',
                    '--format=csv,noheader,nounits', '--id=' + ','.join(self.device_ids)],
                    text=True, stderr=subprocess.DEVNULL, timeout=2)
                values = [float(line.strip()) for line in raw.splitlines() if line.strip()]
                if len(values) == len(self.device_ids) and all(math.isfinite(v) and v >= 0 for v in values):
                    return sum(values)
        except Exception:
            pass
        return None

    def _sample(self):
        value = self._read()
        self.samples.append((time.perf_counter(), value))

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self):
        if hasattr(self, 'started'):
            raise RuntimeError('EnergySampler is single-use')
        if self.backend == 'amd':
            try:
                buses = hip_pci_devices(self.device_indices)
                raw = subprocess.check_output(['rocm-smi', '--showbus', '--json'], text=True,
                                              stderr=subprocess.DEVNULL, timeout=2) if buses else '{}'
                devices = match_rocm_devices(buses, json.loads(raw))
                self.mapping_verified = devices is not None
                self.device_ids = devices or []
            except (OSError, ValueError, subprocess.SubprocessError):
                self.mapping_verified = False
                self.device_ids = []
            if not self.mapping_verified:
                self.unavailable_reason = 'unverified_hip_to_smi_pci_mapping'
        if self.backend == 'nvidia':
            try:
                import pynvml
                pynvml.nvmlInit()
                self._nvml = pynvml
                self._handles = [pynvml.nvmlDeviceGetHandleByIndex(int(d)) if d.isdigit()
                                 else pynvml.nvmlDeviceGetHandleByUUID(d.encode()) for d in self.device_ids]
            except Exception:
                self._handles = []
        self._sample()
        self.started = self.samples[-1][0]
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self.started

    def stop(self, started_s=None, ended_s=None):
        if self.result is not None:
            return self.result
        ended = time.perf_counter() if ended_s is None else ended_s
        started = self.started if started_s is None else started_s
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        energy, coverage = integrate_samples(self.samples, started, ended, self.interval*3)
        in_window = sum(started <= t <= ended and watts is not None
                        and math.isfinite(watts) and watts >= 0 for t, watts in self.samples)
        if in_window < 3:
            energy = None
            self.unavailable_reason = self.unavailable_reason or 'insufficient_in_window_samples'
        elif energy is None:
            self.unavailable_reason = self.unavailable_reason or 'incomplete_power_coverage'
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        self.result = dict(energy_j=energy, mean_power_w=energy/(ended-started) if energy is not None else None,
                           power_sampler_available=energy is not None, energy_method=METHOD,
                           power_sample_count=len(self.samples), power_coverage=coverage,
                           power_device_ids=self.device_ids, power_interval_s=self.interval,
                           power_unavailable_reason=self.unavailable_reason,
                           power_started_s=started, power_ended_s=ended)
        self.result['power_source'] = ('rocm-smi' if self.backend == 'amd' else
                                       'nvml' if self._handles else 'nvidia-smi')
        return self.result

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def aggregate_energy(rows):
    result = dict(energy_j=None, mean_power_w=None, power_sampler_available=False, energy_method=METHOD)
    if not rows or any(not r.get('power_sampler_available') or r.get('energy_method') != METHOD or r.get('energy_j') is None or not math.isfinite(r['energy_j']) or r['energy_j'] < 0 for r in rows):
        return result
    ids = [d for row in rows for d in row.get('power_device_ids', [])]
    starts = [r.get('power_started_s') for r in rows]
    ends = [r.get('power_ended_s') for r in rows]
    if not ids or len(set(ids)) != len(ids) or None in starts or None in ends:
        return result
    # Different worker windows cannot support a combined energy efficiency claim.
    if max(starts)-min(starts) > 0.001 or max(ends)-min(ends) > 0.001:
        return result
    seconds = max(ends)-min(starts)
    if seconds <= 0:
        return result
    energy = sum(r['energy_j'] for r in rows)
    return dict(result, energy_j=energy, mean_power_w=energy/seconds, power_sampler_available=True,
                power_device_ids=ids, power_started_s=min(starts), power_ended_s=max(ends),
                power_sample_count=sum(r.get('power_sample_count', 0) for r in rows), power_coverage=1.0)
