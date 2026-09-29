import math
import os
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rebin import RemapError, remap

from tests.common import exact_remap

TOL = 1e-11


def assert_fraction_close(testcase, value, frac, tol=TOL):
    testcase.assertTrue(
        math.isfinite(value), f"non-finite value {value}"
    )
    testcase.assertLessEqual(
        abs(value - float(frac)), tol, f"{value} != exact {frac}"
    )


class RemapTestCase(unittest.TestCase):
    def assertMatchesExact(self, res, sources, targets, totals, variances):
        self.assertEqual(len(res.groups), len(totals))
        for g, (T, V) in enumerate(zip(totals, variances)):
            exact = exact_remap(sources, targets, T, V)
            gr = res.groups[g]
            for j in range(len(targets)):
                assert_fraction_close(
                    self, gr.target_totals[j], exact["target_totals"][j]
                )
                assert_fraction_close(
                    self,
                    gr.uncovered_total,
                    exact["uncovered_total"],
                )
                assert_fraction_close(
                    self,
                    gr.uncovered_variance,
                    exact["uncovered_variance"],
                )
                for k in range(len(targets)):
                    assert_fraction_close(
                        self,
                        gr.target_cov[j][k],
                        exact["target_cov"][j][k],
                    )
                    self.assertEqual(gr.target_cov[j][k], gr.target_cov[k][j])
                assert_fraction_close(
                    self, gr.uncovered_cov[j], exact["uncovered_cov"][j]
                )

    def assertConserved(self, res, totals, variances, tol=TOL):
        for g, (T, V) in enumerate(zip(totals, variances)):
            d = res.diagnostics[g]
            self.assertLessEqual(abs(d.total_error), tol)
            self.assertLessEqual(abs(d.variance_error), tol)
            self.assertAlmostEqual(d.input_total, sum(T), places=10)
            self.assertAlmostEqual(d.input_variance, sum(V), places=10)

    # 1. Main flow: full coverage, several aligned and split bins.
    def test_full_coverage(self):
        sources = [(0.0, 10.0), (10.0, 30.0)]
        targets = [(0.0, 15.0), (15.0, 30.0)]
        totals = [[30.0, 20.0]]
        variances = [[9.0, 16.0]]
        res = remap(sources, targets, totals, variances)
        self.assertMatchesExact(res, sources, targets, totals, variances)
        self.assertConserved(res, totals, variances)
        # Hand calculation.
        gr = res.groups[0]
        self.assertAlmostEqual(gr.target_totals[0], 35.0)
        self.assertAlmostEqual(gr.target_totals[1], 15.0)
        self.assertAlmostEqual(gr.uncovered_total, 0.0)
        self.assertAlmostEqual(gr.target_cov[0][1], 3.0)
        self.assertAlmostEqual(gr.uncovered_cov[0], 0.0)

    # 2. Partial coverage with a gap and source range outside targets.
    def test_partial_coverage_with_gap(self):
        sources = [(0.0, 10.0), (10.0, 30.0)]
        targets = [(0.0, 15.0), (20.0, 30.0)]
        totals = [[30.0, 20.0]]
        variances = [[9.0, 16.0]]
        res = remap(sources, targets, totals, variances)
        self.assertMatchesExact(res, sources, targets, totals, variances)
        self.assertConserved(res, totals, variances)
        gr = res.groups[0]
        self.assertAlmostEqual(gr.target_totals[0], 35.0)
        self.assertAlmostEqual(gr.target_totals[1], 10.0)
        self.assertAlmostEqual(gr.uncovered_total, 5.0)
        self.assertAlmostEqual(gr.uncovered_variance, 1.0)
        self.assertAlmostEqual(gr.uncovered_cov[0], 1.0)
        self.assertAlmostEqual(gr.uncovered_cov[1], 2.0)

    # Targets only cover a middle slice; uncovered on both sides.
    def test_targets_inside_source_with_side_gaps(self):
        sources = [(0.0, 4.0)]
        targets = [(1.0, 2.0), (3.0, 3.5)]
        T, V = [[8.0]], [[4.0]]
        res = remap(sources, targets, T, V)
        self.assertMatchesExact(res, sources, targets, T, V)
        self.assertConserved(res, T, V)
        gr = res.groups[0]
        self.assertAlmostEqual(gr.target_totals[0], 2.0)
        self.assertAlmostEqual(gr.target_totals[1], 1.0)
        self.assertAlmostEqual(gr.uncovered_total, 5.0)

    # 3. Split correlations are non-zero and survive the full covariance.
    def test_split_nonzero_covariance(self):
        sources = [(0.0, 3.0)]
        targets = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
        T, V = [[12.0]], [[9.0]]
        res = remap(sources, targets, T, V)
        gr = res.groups[0]
        for j in range(3):
            self.assertAlmostEqual(gr.target_totals[j], 4.0)
            self.assertAlmostEqual(gr.target_cov[j][j], 1.0)
            for k in range(3):
                if j != k:
                    self.assertAlmostEqual(gr.target_cov[j][k], 1.0)
        # Independent direct check from first principles.
        f = Fraction(1) / 3
        expected_cov = float((f * f) * Fraction(9))
        self.assertAlmostEqual(gr.target_cov[0][2], expected_cov)
        self.assertGreater(gr.target_cov[0][1], 0.0)

    # 4. Merge conservation for a non-trivial multi-split multi-source case.
    def test_merge_conservation_general(self):
        sources = [(0.0, 2.0), (2.0, 7.0), (7.0, 8.0)]
        targets = [(0.5, 3.0), (3.0, 6.5), (6.5, 7.5)]
        T = [[3.0, 50.0, 7.0]]
        V = [[2.0, 11.0, 0.5]]
        res = remap(sources, targets, T, V)
        self.assertMatchesExact(res, sources, targets, T, V)
        self.assertConserved(res, T, V)
        gr = res.groups[0]
        # Merge: total = sum targets + uncovered.
        merged_total = sum(gr.target_totals) + gr.uncovered_total
        self.assertAlmostEqual(merged_total, sum(T[0]), places=10)
        # Variance of the sum with all cross terms.
        m = len(targets)
        merged_var = gr.uncovered_variance
        for j in range(m):
            merged_var += gr.target_cov[j][j] + 2.0 * gr.uncovered_cov[j]
            for k in range(m):
                if j != k:
                    merged_var += gr.target_cov[j][k]
        self.assertAlmostEqual(merged_var, sum(V[0]), places=10)
