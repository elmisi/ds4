from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'speed-bench'))
from nsys_activity_summary import intervals_summary


class IntervalTests(unittest.TestCase):
    def test_overlap_nesting_and_gap(self):
        result = intervals_summary([(0, 10), (1, 2), (5, 20), (30, 40)])
        self.assertEqual(result['busy_union_seconds'], 30 / 1e9)
        self.assertEqual(result['gap_seconds'], 10 / 1e9)
        self.assertEqual(result['events'], 4)

    def test_empty(self):
        self.assertEqual(intervals_summary([])['span_seconds'], 0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            intervals_summary([(2, 1)])
        with self.assertRaises(ValueError):
            intervals_summary([(0, 10), (5, 8), (3, 4)])
