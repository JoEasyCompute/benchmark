"""Shared measurement and workload controls, independent of GPU libraries."""
import hashlib
from pathlib import Path
import re
import time
import subprocess

TIMING_METHOD = 'synchronized_v1'
SEED = 1234


class MeasurementWindow:
    """One common monotonic start after all workers finish device warmup."""

    def __init__(self, context, workers, timeout=600):
        self.barrier = context.Barrier(workers, timeout=timeout)
        self.started = context.Value('d', 0.0)

    def start(self, synchronize):
        synchronize()
        self.barrier.wait()
        with self.started.get_lock():
            if self.started.value == 0:
                self.started.value = time.perf_counter()
        self.barrier.wait()
        return self.started.value

    def abort(self):
        self.barrier.abort()


def timed_call(operation, synchronize, clock=time.perf_counter):
    synchronize()
    started = clock()
    result = operation()
    synchronize()
    return result, clock() - started


def fixed_generation_kwargs(output_len, pad_token_id):
    return dict(max_new_tokens=output_len, min_new_tokens=output_len,
                do_sample=False, use_cache=True, num_beams=1,
                num_return_sequences=1, eos_token_id=None,
                forced_eos_token_id=None, max_time=None,
                pad_token_id=pad_token_id)


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def snapshot_revision(filename):
    parts = Path(filename).parts
    for index, part in enumerate(parts[:-1]):
        if part == 'snapshots' and re.fullmatch(r'[0-9a-f]{40}', parts[index + 1]):
            return parts[index + 1]
    return None


def resolve_revision(model, revision=None, filename='config.json'):
    """Resolve a remote ref once so all components load the same commit.

    Local directories have no verified immutable revision and remain explicitly
    unidentified. hf_hub_download uses the existing cache and authentication.
    """
    if Path(model).is_dir():
        return None
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(model, filename, revision=revision or 'main')
    resolved = snapshot_revision(path)
    if resolved is None:
        raise ValueError(f'Cannot resolve immutable model revision for {model}')
    return resolved


def sample_rocm_power_watts(device_indices=None):
    """Compatibility read for explicitly selected logical devices."""
    try:
        from energy import EnergySampler
    except ImportError:
        from .energy import EnergySampler
    return EnergySampler('amd', device_indices or [0])._read()
