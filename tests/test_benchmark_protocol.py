import multiprocessing as mp
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from threading import BrokenBarrierError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import benchmark_protocol as protocol


def start_worker(window, queue):
    queue.put(window.start(lambda: None))


class ProtocolTest(unittest.TestCase):
    def test_spawned_workers_share_one_start(self):
        ctx = mp.get_context('spawn')
        window = protocol.MeasurementWindow(ctx, 2, timeout=5)
        queue = ctx.Queue()
        workers = [ctx.Process(target=start_worker, args=(window, queue)) for _ in range(2)]
        try:
            for worker in workers:
                worker.start()
            starts = [queue.get(timeout=10) for _ in workers]
            self.assertEqual(starts[0], starts[1])
            self.assertGreater(starts[0], 0)
        finally:
            for worker in workers:
                worker.join(5)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
            queue.close()

    def test_aborted_worker_breaks_readiness_wait(self):
        window = protocol.MeasurementWindow(mp.get_context('spawn'), 2, timeout=1)
        window.abort()
        with self.assertRaises(BrokenBarrierError):
            window.start(lambda: None)

    def test_batch_timing_synchronizes_before_and_after_work(self):
        events = []
        ticks = iter([10.0, 12.0])
        result, elapsed = protocol.timed_call(
            lambda: events.append('work') or 42,
            lambda: events.append('sync'), clock=lambda: next(ticks))
        self.assertEqual(events, ['sync', 'work', 'sync'])
        self.assertEqual((result, elapsed), (42, 2.0))

    def test_fixed_generation_disables_early_eos_and_beam_defaults(self):
        options = protocol.fixed_generation_kwargs(128, 0)
        self.assertEqual(options['min_new_tokens'], 128)
        self.assertEqual(options['max_new_tokens'], 128)
        self.assertIsNone(options['eos_token_id'])
        self.assertEqual(options['num_beams'], 1)
        self.assertEqual(options['pad_token_id'], 0)

    def test_model_cache_revision_requires_immutable_snapshot(self):
        self.assertEqual(protocol.snapshot_revision('/cache/models--q/snapshots/' + 'a'*40 + '/config.json'), 'a'*40)
        self.assertIsNone(protocol.snapshot_revision('/models/local/config.json'))

    def test_rocm_power_returns_none_when_tool_unavailable(self):
        with patch.object(protocol.subprocess, 'check_output', side_effect=OSError):
            self.assertIsNone(protocol.sample_rocm_power_watts())
