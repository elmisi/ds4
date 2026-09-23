#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'speed-bench'))
SPEC = importlib.util.spec_from_file_location('prefill_stage_suite', ROOT / 'speed-bench/prefill_stage_suite.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StageSuiteTests(unittest.TestCase):
    def test_exact_gates_precede_counterbalanced_timings(self):
        plan = MODULE.jobs(2)
        self.assertEqual(plan[:2], [('gate-off', 0, True), ('gate-on', 1, True)])
        self.assertEqual([job[1] for job in plan[2:]], [1, 0, 0, 1])
        self.assertEqual(len({job[0] for job in plan}), len(plan))
        self.assertTrue(all(not job[2] for job in plan[2:]))
        with self.assertRaises(ValueError):
            MODULE.jobs(0)

    def test_comparator_rejects_partial_or_different_vectors(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / 'a', Path(folder) / 'b'
            a.write_bytes(bytes(129280 * 4))
            b.write_bytes(a.read_bytes())
            self.assertTrue(MODULE.identical(a, b, 1))
            self.assertFalse(MODULE.identical(a, b, 2))
            b.write_bytes(b'\1' + b.read_bytes()[1:])
            self.assertFalse(MODULE.identical(a, b, 1))
            b.write_bytes(b'')
            self.assertFalse(MODULE.identical(a, b, 1))


if __name__ == '__main__':
    unittest.main()
