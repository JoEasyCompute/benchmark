import unittest

from benchmarks.pearlhash import map_pci_devices, read_sample, summarize
from suite_config import optional_jobs, smoke_optional_suites, validate_optional_suites


class PearlHashTest(unittest.TestCase):
    def test_standard_jobs_use_selected_physical_gpus(self):
        cfg = {'gpu_include': [4, 2], 'pearlhash': {'enabled': True}}
        self.assertFalse(validate_optional_suites(cfg))
        jobs = optional_jobs(cfg)
        self.assertEqual([job['args'][-1] for job in jobs], ['4', '4,2'])
        cfg['smoke_mode'] = True
        smoke_optional_suites(cfg)
        self.assertEqual(len(optional_jobs(cfg)), 1)

    def test_maps_miner_ids_by_pci_not_listing_order(self):
        listing = ("\x1b[32mGPU7\x1b[0m [CUDA][1] [0000:26:00.0] : NVIDIA GPU\n"
                   "GPU3 [CUDA][0] [0000:01:00.0] : NVIDIA GPU\n")
        inventory = {0: "00000000:01:00.0", 1: "00000000:26:00.0"}
        self.assertEqual(map_pci_devices(listing, inventory, [1, 0]), [7, 3])
        with self.assertRaises(ValueError):
            map_pci_devices(listing + "GPU8 [CUDA][2] [0000:26:00.0] : NVIDIA GPU\n", inventory, [1])
        with self.assertRaises(ValueError):
            map_pci_devices(listing, inventory, [2])

    def test_selects_only_pearlhash_and_reports_per_gpu_rates(self):
        payload = {'algorithms': [
            {'name': 'other'},
            {'name': 'pearlhash', 'hashrate': {'gpu': {'gpu3': 10, 'gpu7': 20}},
             'shares': {'accepted': 4, 'rejected': 1}},
        ]}
        sample = read_sample(payload, [3, 7])
        end = ([20, 40], 7, 2)
        result = summarize([sample, end])
        self.assertEqual(result['hashrate_hs'], 45)
        self.assertEqual(result['per_gpu_hashrate_hs'][1], 30)
        self.assertEqual(result['accepted_shares'], 3)
        self.assertEqual(result['rejected_shares'], 1)

    def test_rejects_unexpected_active_gpu(self):
        payload = {'algorithms': [
            {'name': 'pearlhash', 'hashrate': {'gpu': {'gpu0': 10, 'gpu1': 20}},
             'shares': {'accepted': 0, 'rejected': 0}},
        ]}
        with self.assertRaises(ValueError):
            read_sample(payload, [0])
