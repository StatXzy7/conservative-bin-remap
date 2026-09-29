import unittest

import numpy as np

from rebinning import rebin
from tests.reference import TOL


class TestOrdering(unittest.TestCase):
    def test_unsorted_sources_match_sorted_input(self):
        sorted_sources = [(0.0, 2.0), (2.0, 6.0), (6.0, 9.0)]
        shuffled = [sorted_sources[2], sorted_sources[0], sorted_sources[1]]
        mass_sorted = [4.0, 8.0, 6.0]
        mass_shuffled = [mass_sorted[2], mass_sorted[0], mass_sorted[1]]
        var_sorted = [5.0, 20.0, 3.0]
        var_shuffled = [var_sorted[2], var_sorted[0], var_sorted[1]]
        targets = [(0.0, 1.0), (2.0, 5.0), (6.0, 9.0)]

        ordered = rebin(sorted_sources, mass_sorted, var_sorted, targets)
        shuffled_result = rebin(shuffled, mass_shuffled, var_shuffled,
                                targets)
        np.testing.assert_allclose(shuffled_result.target_mass,
                                   ordered.target_mass, atol=TOL)
        np.testing.assert_allclose(shuffled_result.target_cov,
                                   ordered.target_cov, atol=TOL)
        np.testing.assert_allclose(shuffled_result.uncovered_mass,
                                   ordered.uncovered_mass, atol=TOL)
        np.testing.assert_allclose(shuffled_result.uncovered_cov,
                                   ordered.uncovered_cov, atol=TOL)

    def test_caller_target_order_is_preserved(self):
        sources = [(0.0, 10.0)]
        mass = [10.0]
        variance = [10.0]
        # Spatially reversed / interleaved caller order must be kept.
        targets = [(8.0, 10.0), (0.0, 3.0), (3.0, 8.0)]
        result = rebin(sources, mass, variance, targets)
        np.testing.assert_allclose(result.target_mass[0], [2, 3, 5],
                                   atol=TOL)
        # Covariance rows/cols follow the same caller order.
        expected_mass_order = [2.0, 3.0, 5.0]
        np.testing.assert_allclose(result.target_variance[0],
                                   expected_mass_order, atol=TOL)

    def test_touching_sources_do_not_double_allocate(self):
        sources = [(0.0, 2.0), (2.0, 4.0)]
        targets = [(0.0, 2.0), (2.0, 4.0)]  # exactly tile, shared boundary
        result = rebin(sources, [2.0, 2.0], [5.0, 7.0], targets)
        np.testing.assert_allclose(result.target_mass[0], [2, 2], atol=TOL)
        np.testing.assert_allclose(result.target_mass.sum(), 4.0,
                                   atol=TOL)
        np.testing.assert_allclose(result.uncovered_mass, 0.0, atol=TOL)

    def test_target_boundary_touching_but_not_overlapping(self):
        sources = [(0.0, 4.0)]
        targets = [(0.0, 2.0), (2.0, 4.0)]
        result = rebin(sources, [4.0], [4.0], targets)
        np.testing.assert_allclose(result.target_mass[0], [2, 2], atol=TOL)
        # Split within one source -> shared covariance reported.
        self.assertAlmostEqual(result.target_cov[0, 0, 1], 0.0)  # V == M


if __name__ == "__main__":
    unittest.main()
