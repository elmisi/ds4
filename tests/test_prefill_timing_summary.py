#!/usr/bin/env python3
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'speed-bench'))
from prefill_timing_summary import paired_summary


def row(pair, mode, speed, swap=0, success=True):
    return dict(pair=pair, mode=mode, prefill_tps=speed, swap_kib=swap, success=success)


class SummaryTests(unittest.TestCase):
    def test_entire_swapped_pair_excluded(self):
        result = paired_summary([row(1, 'on', 400), row(1, 'off', 300, swap=80),
                                 row(2, 'off', 350), row(2, 'on', 385)])
        self.assertEqual(len(result['all_samples']), 4)
        self.assertEqual(result['clean_pair_count'], 1)
        self.assertAlmostEqual(result['clean_prefill_gain_percent'], 10)
        self.assertEqual(result['excluded_pairs'][0]['pair'], 1)

    def test_missing_unknown_and_failed_are_not_clean(self):
        for rows in [[row(1, 'off', 350)], [row(1, 'off', 350, None), row(1, 'on', 385)],
                     [row(1, 'off', 350), row(1, 'on', 385, success=False)]]:
            result = paired_summary(rows)
            self.assertEqual(result['clean_pair_count'], 0)
            self.assertIsNone(result['clean_prefill_gain_percent'])

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            paired_summary([row(1, 'off', 350), row(1, 'off', 350)])


if __name__ == '__main__':
    unittest.main()
