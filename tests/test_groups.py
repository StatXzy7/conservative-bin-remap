import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rebin import remap

from tests.common import exact_remap
from tests.test_core import assert_fraction_close


class MultiGroupTestCase(unittest.TestCase):
    # 5. Multiple measurement groups share the same interval edges.
    def test_multiple_groups_share_edges(self):
        sources = [(0.0, 10.0), (10.0, 30.0)]
        targets = [(0.0, 15.0), (20.0, 30.0)]
        totals = [[30.0, 20.0], [10.0, 8.0], [0.0, 0.0]]
        variances = [[9.0, 16.0], [4.0, 0.0], [0.0, 0.0]]
        res = remap(sources, targets, totals, variances)
        self.assertEqual(len(res.groups), 3)
        self.assertEqual(len(res.diagnostics), 3)
        for g, (T, V) in enumerate(zip(totals, variances)):
            exact = exact_remap(sources, targets, T, V)
            gr = res.groups[g]
            for j in range(2):
                assert_fraction_close(
                    self, gr.target_totals[j], exact["target_totals"][j]
                )
                for k in range(2):
                    assert_fraction_close(
                        self, gr.target_cov[j][k], exact["target_cov"][j][k]
                    )
                assert_fraction_close(
                    self, gr.uncovered_cov[j], exact["uncovered_cov"][j]
                )
            assert_fraction_close(
                self, gr.uncovered_total, exact["uncovered_total"]
            )
            assert_fraction_close(
                self, gr.uncovered_variance, exact["uncovered_variance"]
            )
            self.assertLessEqual(abs(res.diagnostics[g].total_error), 1e-11)
            self.assertLessEqual(abs(res.diagnostics[g].variance_error), 1e-11)

        # Group 2: zero total and zero variance are valid -> all-zero output.
        gr2 = res.groups[2]
        self.assertTrue(all(v == 0.0 for v in gr2.target_totals))
        self.assertTrue(all(v == 0.0 for row in gr2.target_cov for v in row))
        self.assertEqual(gr2.uncovered_total, 0.0)
        self.assertEqual(gr2.uncovered_variance, 0.0)

    def test_1d_input_is_single_group(self):
        sources = [(0.0, 5.0)]
        targets = [(0.0, 2.5)]
        res = remap(sources, targets, [10.0], [4.0])
        self.assertEqual(len(res.groups), 1)
        self.assertAlmostEqual(res.groups[0].target_totals[0], 5.0)
        self.assertAlmostEqual(res.groups[0].target_cov[0][0], 1.0)


if __name__ == "__main__":
    unittest.main()
