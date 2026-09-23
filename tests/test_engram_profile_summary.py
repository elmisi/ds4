#!/usr/bin/env python3
"""Check attribution and frontier filtering in the profiling report."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'speed-bench/engram_profile_summary.py'
SPEC = importlib.util.spec_from_file_location('engram_profile_summary', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProfileSummaryTests(unittest.TestCase):
    def test_empty_is_not_success(self):
        result = MODULE.summarize('', 65536)
        self.assertFalse(result['decode_all_ok'])
        self.assertIsNone(result['decode_cache_miss_fraction'])

    def test_frontier_and_attribution(self):
        log = '''ds4: CUDA SSD cache layer=0 unique=6 misses=6 bytes=999
ds4: V4.1 prefill layer=0 map=1 encode=10
ds4: CUDA SSD cache layer=1 unique=6 misses=1 bytes=1024 load_ms=2
ds4: V4.1 decode layer pos=9 layer=1 wall_ms=3
ds4: V4.1 decode token pos=9 total_ms=4 ok=1
ds4: CUDA SSD cache layer=1 unique=6 misses=2 bytes=2048 load_ms=5
ds4: V4.1 decode layer pos=10 layer=1 wall_ms=6
ds4: V4.1 decode token pos=10 total_ms=7 wait0_ms=0.5 ok=1
ds4: V4.1 stage layer=0 rows=8192 attention core/index=10 ms
ds4: CUDA SSD prefetch layer=1 bytes=4096 read=3 wait=1
'''
        result = MODULE.summarize(log, 10)
        self.assertEqual(result['decode_tokens'], 1)
        self.assertTrue(result['decode_all_ok'])
        self.assertEqual(result['decode_layer_records'], 1)
        self.assertEqual(result['decode_cache_records'], 1)
        self.assertEqual(result['decode_cache_miss_fraction'], 2 / 6)
        self.assertEqual(result['decode_cache_ms_per_token']['load_ms'], 5)
        self.assertEqual(result['decode_mean_ms']['wait0_ms'], .5)
        self.assertEqual(result['decode_logical_expert_read_GiB'], 2048 / 2**30)
        self.assertEqual(result['prefill_stage_seconds']['attention core/index'], .01)
        self.assertEqual(result['prefill_inclusive_seconds']['encode'], .01)
        self.assertEqual(result['prefetch_foreground_wait_seconds'], .001)
        self.assertEqual(result['prefetch_background_elapsed_seconds'], .003)


if __name__ == '__main__':
    unittest.main()
