import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rebin import RemapError, remap


class InvalidInputTestCase(unittest.TestCase):
    def assertRejected(self, sources, targets, T, V):
        with self.assertRaises(RemapError):
            remap(sources, targets, T, V)

    def test_negative_total(self):
        self.assertRejected([(0.0, 1.0)], [(0.0, 0.5)], [-1.0], [1.0])

    def test_negative_variance(self):
        self.assertRejected([(0.0, 1.0)], [(0.0, 0.5)], [1.0], [-0.1])

    def test_non_finite_values(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            self.assertRejected([(0.0, 1.0)], [(0.0, 0.5)], [bad], [1.0])
            self.assertRejected([(0.0, 1.0)], [(0.0, 0.5)], [1.0], [bad])
        self.assertRejected([(0.0, bad)], [(0.0, 0.5)], [1.0], [1.0])
        self.assertRejected([(0.0, 1.0)], [(0.0, bad)], [1.0], [1.0])

    def test_zero_width_interval(self):
        self.assertRejected([(1.0, 1.0)], [(0.0, 0.5)], [1.0], [1.0])
        self.assertRejected([(0.0, 1.0)], [(0.5, 0.5)], [1.0], [1.0])

    def test_reversed_interval(self):
        self.assertRejected([(2.0, 1.0)], [(0.0, 0.5)], [1.0], [1.0])

    def test_overlapping_sources(self):
        self.assertRejected(
            [(0.0, 2.0), (1.0, 3.0)], [(0.0, 1.0)], [1.0, 1.0], [1.0, 1.0]
        )

    def test_overlapping_targets(self):
        self.assertRejected(
            [(0.0, 5.0)], [(0.0, 2.0), (1.0, 3.0)], [1.0], [1.0]
        )

    def test_dimension_mismatch(self):
        self.assertRejected(
            [(0.0, 1.0), (1.0, 2.0)], [(0.0, 1.0)], [1.0], [1.0, 1.0]
        )
        self.assertRejected(
            [(0.0, 1.0), (1.0, 2.0)], [(0.0, 1.0)], [1.0, 2.0], [1.0]
        )
        # Group count mismatch between totals and variances.
        with self.assertRaises(RemapError):
            remap(
                [(0.0, 1.0)],
                [(0.0, 1.0)],
                [[1.0], [2.0]],
                [[1.0]],
            )

    def test_empty_intervals(self):
        self.assertRejected([], [(0.0, 1.0)], [1.0], [1.0])
        self.assertRejected([(0.0, 1.0)], [], [1.0], [1.0])

    def test_bad_pair_shape(self):
        self.assertRejected([(0.0,)], [(0.0, 0.5)], [1.0], [1.0])
        self.assertRejected([(0.0, 1.0)], [0.5], [1.0], [1.0])

    def test_touching_is_accepted(self):
        # Touching (but not overlapping) intervals are legal.
        res = remap(
            [(0.0, 1.0), (1.0, 2.0)],
            [(0.0, 1.0), (1.0, 2.0)],
            [2.0, 4.0],
            [1.0, 1.0],
        )
        self.assertAlmostEqual(sum(res.groups[0].target_totals), 6.0)

    def test_zero_total_and_zero_variance_valid(self):
        res = remap([(0.0, 1.0)], [(0.0, 0.5)], [0.0], [0.0])
        gr = res.groups[0]
        self.assertEqual(gr.target_totals[0], 0.0)
        self.assertEqual(gr.target_cov[0][0], 0.0)
        self.assertEqual(gr.uncovered_total, 0.0)
        self.assertEqual(gr.uncovered_variance, 0.0)


if __name__ == "__main__":
    unittest.main()
