import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rebin import remap


class OrderingTestCase(unittest.TestCase):
    # 6. Scrambled inputs: source and target order may be arbitrary;
    #    target output order must follow the caller's target order.
    def test_scrambled_inputs_keep_target_order(self):
        sources_fwd = [(0.0, 10.0), (10.0, 30.0)]
        sources_rev = [(10.0, 30.0), (0.0, 10.0)]
        targets_fwd = [(0.0, 15.0), (20.0, 30.0)]
        targets_rev = [(20.0, 30.0), (0.0, 15.0)]
        T = [30.0, 20.0]
        V = [9.0, 16.0]

        r0 = remap(sources_fwd, targets_fwd, T, V)
        # Sources reversed -> totals/variances reordered correspondingly.
        r1 = remap(sources_rev, targets_fwd, [20.0, 30.0], [16.0, 9.0])
        # Targets reversed: output rows/cols must follow reversed target order.
        r2 = remap(sources_fwd, targets_rev, T, V)

        g0, g1, g2 = r0.groups[0], r1.groups[0], r2.groups[0]
        for a, b in zip(g0.target_totals, g1.target_totals):
            self.assertAlmostEqual(a, b)
        for j in range(2):
            for k in range(2):
                self.assertAlmostEqual(g0.target_cov[j][k], g1.target_cov[j][k])
        # r2 target 0 corresponds to fwd target 1, and target 1 to fwd target 0.
        self.assertAlmostEqual(g2.target_totals[0], g0.target_totals[1])
        self.assertAlmostEqual(g2.target_totals[1], g0.target_totals[0])
        self.assertAlmostEqual(g2.target_cov[0][0], g0.target_cov[1][1])
        self.assertAlmostEqual(g2.target_cov[1][1], g0.target_cov[0][0])
        self.assertAlmostEqual(g2.target_cov[0][1], g0.target_cov[1][0])
        self.assertAlmostEqual(g2.uncovered_total, g0.uncovered_total)
        self.assertAlmostEqual(g2.uncovered_cov[0], g0.uncovered_cov[1])
        # Caller target order is literally preserved in the result metadata.
        self.assertEqual(r2.target_edges, [(20.0, 30.0), (0.0, 15.0)])

    def test_touching_edges_not_double_allocated(self):
        sources = [(0.0, 5.0), (5.0, 10.0)]
        targets = [(0.0, 5.0), (5.0, 10.0)]
        res = remap(sources, targets, [4.0, 8.0], [1.0, 2.0])
        gr = res.groups[0]
        self.assertAlmostEqual(gr.target_totals[0], 4.0)
        self.assertAlmostEqual(gr.target_totals[1], 8.0)
        # No cross-covariance: shared edge carries zero overlap.
        self.assertAlmostEqual(gr.target_cov[0][1], 0.0)
        self.assertAlmostEqual(gr.uncovered_total, 0.0)

    def test_failure_does_not_mutate_inputs(self):
        sources = [(0.0, 5.0), (5.0, 10.0)]
        targets = [(0.0, 6.0), (6.0, 10.0)]
        T = [[3.0, 9.0]]
        V = [[1.0, 2.0]]
        T_snap, V_snap = copy.deepcopy(T), copy.deepcopy(V)
        s_snap, t_snap = copy.deepcopy(sources), copy.deepcopy(targets)
        with self.assertRaises(Exception):
            remap(sources, targets, [[3.0]], [[1.0]])  # dim mismatch
        self.assertEqual(T, T_snap)
        self.assertEqual(V, V_snap)
        self.assertEqual(sources, s_snap)
        self.assertEqual(targets, t_snap)
        # A successful call also leaves inputs untouched.
        remap(sources, targets, T, V)
        self.assertEqual(T, T_snap)
        self.assertEqual(V, V_snap)


if __name__ == "__main__":
    unittest.main()
