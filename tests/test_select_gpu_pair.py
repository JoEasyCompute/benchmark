import unittest
from select_gpu_pair import (
    best_probe_result,
    candidate_pairs,
    parse_gpu_csv,
    parse_nvidia_topology,
    rank_pairs_by_topology,
)


class SelectGpuPairTest(unittest.TestCase):
    def test_parse_gpu_csv_and_candidate_pairs(self):
        self.assertEqual(parse_gpu_csv("0, 2,4"), ["0", "2", "4"])
        self.assertEqual(candidate_pairs(["0", "2", "4"]), [("0", "2"), ("0", "4"), ("2", "4")])

    def test_parse_and_rank_nvidia_topology(self):
        raw = """
              GPU0    GPU1    GPU2    CPU Affinity
        GPU0     X      PIX     SYS    0-31
        GPU1    PIX      X      PHB    0-31
        GPU2    SYS     PHB      X     32-63
        """
        topology = parse_nvidia_topology(raw)
        self.assertEqual(topology[("0", "1")], "PIX")
        self.assertEqual(topology[("0", "2")], "SYS")
        self.assertEqual(topology[("1", "2")], "PHB")
        ranked = rank_pairs_by_topology([("0", "2"), ("1", "2"), ("0", "1")], topology)
        self.assertEqual(ranked, [("0", "1"), ("1", "2"), ("0", "2")])

    def test_best_probe_result_uses_tokens_per_sec(self):
        result = best_probe_result(
            [
                {"pair": ["0", "1"], "status": "ok", "tokens_per_sec": 100.0},
                {"pair": ["0", "2"], "status": "failed"},
                {"pair": ["1", "2"], "status": "ok", "tokens_per_sec": 150.0},
            ]
        )
        self.assertEqual(result["pair"], ["1", "2"])


if __name__ == "__main__":
    unittest.main()
