import contextlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import llm_serve as serving


@contextlib.contextmanager
def endpoint(events):
    payloads = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payloads.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            try:
                for delay, value in events:
                    time.sleep(delay)
                    self.wfile.write(value.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1/completions', payloads
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def event(text='', finish=None, usage=None):
    data = {'choices': [{'index': 0, 'text': text, 'finish_reason': finish}]}
    if usage is not None:
        data['usage'] = {'completion_tokens': usage, 'prompt_tokens': 3}
    return 'data: ' + json.dumps(data) + '\n\n'


class ServingTests(unittest.TestCase):
    def test_fifo_admission_and_release_after_exception(self):
        self.assertTrue(hasattr(serving, 'FifoLock'))
        lock = serving.FifoLock()
        order = []
        threads = []
        with lock:
            for index in range(4):
                def worker(value=index):
                    with lock:
                        order.append(value)
                thread = threading.Thread(target=worker)
                thread.start()
                threads.append(thread)
                deadline = time.monotonic() + 2
                while True:
                    with lock.condition:
                        queued = lock.next_ticket
                    if queued == index + 2:
                        break
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.001)
        for thread in threads:
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(order, list(range(4)))
        with self.assertRaises(ValueError):
            with lock:
                raise ValueError('generation failed')
        with lock:
            pass

    def provider(self, url, **kwargs):
        return serving.HttpProvider(url, 'fixture', 'hello', 2, timeout=2, **kwargs)

    def test_ttft_ignores_empty_preamble_and_counts_reported_tokens(self):
        events = [(0, ': keepalive\n\n'), (0, event()), (.025, event('a')),
                  (.025, event('b', 'length', 2)), (0, 'data: [DONE]\n\n')]
        with endpoint(events) as (url, payloads):
            result = self.provider(url, provider='vllm').request()
        self.assertGreaterEqual(result['ttft_s'], .02)
        self.assertGreaterEqual(result['chunk_gaps_s'][0], .02)
        self.assertEqual(result['generated_tokens'], 2)
        self.assertEqual(result['prompt_tokens'], 3)
        self.assertIsNone(result['token_gaps_s'])
        self.assertTrue(payloads[0]['stream'])
        self.assertTrue(payloads[0]['stream_options']['include_usage'])
        self.assertEqual(payloads[0]['min_tokens'], 2)
        self.assertTrue(payloads[0]['ignore_eos'])

    def test_missing_usage_keeps_latency_without_inventing_tokens(self):
        with endpoint([(0, event('hello', 'stop')), (0, 'data: [DONE]\n\n')]) as (url, _):
            result = self.provider(url).request()
        self.assertIsNone(result['generated_tokens'])
        self.assertGreater(result['latency_s'], 0)

    def test_invalid_truncated_error_or_nonfinished_stream_fails(self):
        fixtures = [
            'data: not-json\n\n',
            event('hello', 'length'),
            'data: {"error":{"message":"bad"}}\n\n',
            event('hello') + 'data: [DONE]\n\n',
            event('', 'length', 2) + 'data: [DONE]\n\n',
            event('hello', 'length', -1) + 'data: [DONE]\n\n',
        ]
        for stream in fixtures:
            with self.subTest(stream=stream), endpoint([(0, stream)]) as (url, _):
                with self.assertRaises(ValueError):
                    self.provider(url).request()

    def test_vllm_short_output_is_failure(self):
        with endpoint([(0, event('a', 'stop', 1)), (0, 'data: [DONE]\n\n')]) as (url, _):
            with self.assertRaises(ValueError):
                self.provider(url, provider='vllm').request()

    def test_concurrent_requests_drain_and_warmup_is_excluded(self):
        with endpoint([(.04, event('ab', 'length', 2)), (0, 'data: [DONE]\n\n')]) as (url, payloads):
            row = serving.measure(self.provider(url), duration=.015, concurrency=3, warmup=2)
        self.assertEqual(row['requests'], 3)
        self.assertEqual(len(payloads), 5)
        self.assertEqual(row['generated_tokens'], 6)
        self.assertGreaterEqual(row['time_s'], .04)
        self.assertAlmostEqual(row['generated_tokens_per_s'], 6 / row['time_s'])
        self.assertGreater(row['drain_s'], 0)

    def test_missing_usage_excludes_token_throughput(self):
        with endpoint([(.02, event('ab', 'stop')), (0, 'data: [DONE]\n\n')]) as (url, _):
            row = serving.measure(self.provider(url), duration=.005, concurrency=1, warmup=0)
        self.assertEqual(row['status'], 'ok')
        self.assertIsNone(row['generated_tokens_per_s'])
        self.assertEqual(row['requests_without_token_usage'], 1)

    def test_failure_excludes_throughput_winner(self):
        with endpoint([(0, 'data: invalid\n\n')]) as (url, _):
            row = serving.measure(self.provider(url), duration=.02, concurrency=1, warmup=0)
        self.assertEqual(row['status'], 'failed')
        self.assertGreater(row['errors'], 0)
        self.assertIsNone(row['generated_tokens_per_s'])
        self.assertIsNone(row['reqs_per_s'])

    def test_token_streamer_excludes_prompt_and_rejects_multitoken_events(self):
        class Tokens:
            def __init__(self, count):
                self.count = count

            def numel(self):
                return self.count

        ticks = iter([1., 2.])
        streamer = serving.TokenEventStreamer(clock=lambda: next(ticks))
        streamer.put(Tokens(6))
        streamer.put(Tokens(1))
        streamer.put(Tokens(1))
        streamer.end()
        self.assertEqual(streamer.token_times, [1., 2.])
        with self.assertRaises(ValueError):
            streamer.put(Tokens(2))

    def test_cli_missing_vllm_endpoint_skips_with_json_row(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metrics.jsonl'
            completed = subprocess.run([
                sys.executable, str(Path(serving.__file__)), '--provider', 'vllm',
                '--model', 'fixture', '--metrics-path', str(path)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            row = json.loads(path.read_text())
            self.assertEqual(row['status'], 'skipped')
            self.assertIsNone(row['energy_j'])


if __name__ == '__main__':
    unittest.main()
