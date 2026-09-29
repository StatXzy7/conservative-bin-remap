import unittest

import numpy as np

from rebinning import RebinError, rebin
from tests.reference import TOL, reference_rebin


def make_result(mass, variance, sources=((0.0, 4.0),),
                targets=((0.0, 1.0), (1.0, 3.0), (3.0, 4.0))):
    sources = [tuple(x) for x in sources]
    targets = [tuple(x) for x in targets]
    result = rebin(sources, np.asarray(mass, dtype=float),
                   np.asarray(variance, dtype=float), targets)
    ref = reference_rebin(sources, np.atleast_2d(mass),
                          np.atleast_2d(variance), targets)
    return result, ref


class TestFullSplit(unittest.TestCase):
    """Example 1 from the CLI demo: one source fully covered by 3 targets."""

    def setUp(self):
        self.result, self.ref = make_result(mass=[4.0], variance=[8.0])

    def test_masses_hand_computed(self):
        np.testing.assert_allclose(self.result.target_mass[0], [1, 2, 1],
                                   atol=TOL)

    def test_variance_and_covariance_hand_computed(self):
        expected = np.array([[1.25, 0.5, 0.25],
                             [0.5, 3.0, 0.5],
                             [0.25, 0.5, 1.25]])
        np.testing.assert_allclose(self.result.target_cov[0], expected,
                                   atol=TOL)
        # Diagonal accessor matches the matrix diagonal.
        np.testing.assert_allclose(self.result.target_variance[0],
                                   np.diag(expected), atol=TOL)

    def test_matches_independent_reference(self):
        for key in ("target_mass", "target_cov", "uncovered_mass",
                    "uncovered_variance", "uncovered_cov"):
            np.testing.assert_allclose(getattr(self.result, key),
                                       self.ref[key], atol=TOL)

    def test_covariance_matrix_is_symmetric(self):
        cov = self.result.target_cov[0]
        np.testing.assert_allclose(cov, cov.T, atol=TOL)

    def test_full_coverage_leaves_nothing_uncovered(self):
        self.assertAlmostEqual(self.result.uncovered_mass[0], 0.0)
        self.assertAlmostEqual(self.result.uncovered_variance[0], 0.0)
        np.testing.assert_allclose(self.result.uncovered_cov[0], 0.0,
                                   atol=TOL)

    def test_poisson_like_splits_are_uncorrelated(self):
        # V == M: shared-total pieces collapse to independent Poisson draws.
        result, _ = make_result(mass=[4.0], variance=[4.0])
        cov = result.target_cov[0]
        off_diag = cov - np.diag(np.diag(cov))
        np.testing.assert_allclose(off_diag, 0.0, atol=TOL)
        np.testing.assert_allclose(np.diag(cov), [1, 2, 1], atol=TOL)


if __name__ == "__main__":
    unittest.main()
