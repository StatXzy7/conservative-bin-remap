import unittest

import numpy as np

from rebinning import rebin
from tests.reference import TOL, reference_rebin


class TestSplitCovariance(unittest.TestCase):
    """Splitting one source across several targets must preserve correlation."""

    def test_overdispersed_split_has_positive_covariance(self):
        # Two equal halves of a source with V > M: covariance must be
        # strictly positive and survive into the reported matrix.
        result = rebin([(0.0, 2.0)], [10.0], [16.0],
                       [(0.0, 1.0), (1.0, 2.0)])
        # f = 1/2 each: cov = (1/2)(1/2)(16-10) = 1.5
        self.assertGreater(result.target_cov[0, 0, 1], 0.0)
        self.assertAlmostEqual(result.target_cov[0, 0, 1], 1.5)

    def test_underdispersed_split_has_negative_covariance(self):
        # A very precisely known total (V < M): fixing the total forces
        # pieces to move oppositely -> negative covariance.
        result = rebin([(0.0, 2.0)], [10.0], [2.0],
                       [(0.0, 1.0), (1.0, 2.0)])
        self.assertLess(result.target_cov[0, 0, 1], 0.0)
        self.assertAlmostEqual(result.target_cov[0, 0, 1], -2.0)
        # Diagonal variance must remain non-negative.
        self.assertTrue(np.all(result.target_variance >= 0.0))

    def test_three_way_split_covariances(self):
        sources = [(0.0, 9.0)]
        targets = [(0.0, 1.0), (1.0, 4.0), (4.0, 9.0)]  # f = 1/9, 3/9, 5/9
        m, v = 90.0, 270.0
        result = rebin(sources, [m], [v], targets)
        fracs = [1 / 9, 3 / 9, 5 / 9]
        cov = result.target_cov[0]
        for i in range(3):
            for j in range(3):
                if i == j:
                    expected = v * fracs[i] ** 2 + m * fracs[i] * (
                        1 - fracs[i]
                    )
                else:
                    expected = (v - m) * fracs[i] * fracs[j]
                self.assertAlmostEqual(cov[i, j], expected, places=9)

    def test_covariance_adds_across_sources_but_stays_zero_independent(self):
        # Each target overlaps a different source: cross cov stays 0 even
        # though each source is over-dispersed.
        result = rebin([(0.0, 2.0), (2.0, 4.0)], [4.0, 4.0], [8.0, 8.0],
                       [(0.0, 1.0), (3.0, 4.0)])
        self.assertAlmostEqual(result.target_cov[0, 0, 1], 0.0)

    def test_uncovered_remainder_keeps_covariance_with_pieces(self):
        result = rebin([(0.0, 4.0)], [4.0], [8.0], [(0.0, 1.0)])
        # f_t = 1/4, f_u = 3/4, V-M = 4 -> 0.75
        np.testing.assert_allclose(result.uncovered_cov[0], [0.75],
                                   atol=TOL)
        ref = reference_rebin([(0.0, 4.0)], [[4.0]], [[8.0]],
                              [(0.0, 1.0)])
        np.testing.assert_allclose(result.uncovered_cov,
                                   ref["uncovered_cov"], atol=TOL)


if __name__ == "__main__":
    unittest.main()
