#!/usr/bin/env python3
"""Host-only checks for the full-logit diagnostic (not model-quality tests)."""
from array import array
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'speed-bench/compare_decode_logits.py'
SPEC = importlib.util.spec_from_file_location('compare_decode_logits', MODULE)
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


class LogitComparisonTests(unittest.TestCase):
    def run_compare(self, x, y, vocab=2):
        with tempfile.TemporaryDirectory() as folder:
            left, right = Path(folder) / 'left', Path(folder) / 'right'
            left.write_bytes(array('f', x).tobytes())
            right.write_bytes(array('f', y).tobytes())
            return COMPARE.compare(left, right, vocab)

    def test_exact(self):
        result = self.run_compare([1, 2, -4, 1], [1, 2, -4, 1])
        self.assertEqual(result['exact_vectors'], 2)
        self.assertEqual(result['top1_matches'], 2)
        self.assertEqual(result['max_abs'], 0)
        self.assertEqual(result['mean_kl_left_right'], 0)

    def test_shift_leaves_distribution_unchanged(self):
        result = self.run_compare([1, 2], [1001, 1002])
        self.assertEqual(result['exact_vectors'], 0)
        self.assertEqual(result['top1_matches'], 1)
        self.assertAlmostEqual(result['mean_kl_left_right'], 0, places=12)
        self.assertEqual(result['rms'], 1000)

    def test_known_kl(self):
        result = self.run_compare([0, 0], [0, math.log(3)])
        self.assertAlmostEqual(result['mean_kl_left_right'], math.log(4 / 3) / 2, places=6)
        self.assertEqual(result['top1_matches'], 0)

    def test_invalid(self):
        for x, y in [([], []), ([1, 2], [1]), ([1], [1]),
                     ([math.nan, 0], [0, 0]), ([0, 0], [math.inf, 0])]:
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                self.run_compare(x, y)


if __name__ == '__main__':
    unittest.main()
