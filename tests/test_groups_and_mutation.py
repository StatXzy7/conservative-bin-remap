import copy
import unittest

import numpy as np

from rebinning import RebinError, rebin
from tests.reference import TOL


class TestInputImmutability(unittest.TestCase):
    def test_success_does_not_mutate_inputs(self):
        source_list = [(0.0, 2.0), (2.0, 5.0)]
        target_list = [(0.0, 1.0), (3.0, 5.0)]
        mass = np.array([2.0, 6.0])
        variance = np.array([3.0, 9.0])
        source_snapshot = copy.deepcopy(source_list)
        target_snapshot = copy.deepcopy(target_list)
        mass_snapshot = mass.copy()
        var_snapshot = variance.copy()

        rebin(source_list, mass, variance, target_list)

        self.assertEqual(source_list, source_snapshot)
        self.assertEqual(target_list, target_snapshot)
        np.testing.assert_array_equal(mass, mass_snapshot)
        np.testing.assert_array_equal(variance, var_snapshot)

    def test_failure_does_not_mutate_inputs(self):
        source_list = [(0.0, 2.0), (2.0, 5.0)]
        target_list = [(0.0, 6.0)]  # overlaps source boundary harmlessly...
        mass = np.array([2.0, -6.0])  # ...negative mass rejected
        variance = np.array([3.0, 9.0])
        source_snapshot = copy.deepcopy(source_list)
        target_snapshot = copy.deepcopy(target_list)
        mass_snapshot = mass.copy()

        with self.assertRaises(RebinError):
            rebin(source_list, mass, variance, target_list)

        self.assertEqual(source_list, source_snapshot)
        self.assertEqual(target_list, target_snapshot)
        np.testing.assert_array_equal(mass, mass_snapshot)

    def test_result_arrays_are_independent_copies(self):
        result = rebin([(0.0, 4.0)], [4.0], [8.0],
                       [(0.0, 2.0), (2.0, 4.0)])
        result.target_mass[0, 0] = 999.0
        result.target_cov[0, 0, 0] = 999.0
        # Re-query: internal recomputation unaffected (dataclass is frozen,
        # arrays themselves are distinct allocations from the input).
        again = rebin([(0.0, 4.0)], [4.0], [8.0],
                      [(0.0, 2.0), (2.0, 4.0)])
        np.testing.assert_allclose(again.target_mass[0], [2, 2], atol=TOL)


class TestMultiGroup(unittest.TestCase):
    def test_1d_input_adds_singleton_group_axis(self):
        result = rebin([(0.0, 2.0)], [2.0], [2.0], [(0.0, 1.0)])
        self.assertEqual(result.target_mass.shape, (1, 1))
        self.assertEqual(result.target_cov.shape, (1, 1, 1))
        self.assertEqual(result.uncovered_cov.shape, (1, 1))
        self.assertEqual(result.uncovered_mass.shape, (1,))

    def test_groups_share_boundaries_but_not_results(self):
        sources = [(0.0, 4.0)]
        targets = [(0.0, 2.0), (2.0, 4.0)]
        mass = [[4.0], [0.0], [8.0]]
        variance = [[8.0], [0.0], [2.0]]
        result = rebin(sources, mass, variance, targets)
        np.testing.assert_allclose(result.target_mass[:, 0], [2, 0, 4],
                                   atol=TOL)
        # Group 0 over-dispersed: positive split covariance.
        self.assertGreater(result.target_cov[0, 0, 1], 0.0)
        # Group 1 trivial.
        self.assertEqual(result.target_cov[1, 0, 1], 0.0)
        # Group 2 under-dispersed: negative split covariance.
        self.assertLess(result.target_cov[2, 0, 1], 0.0)
        # Per-group diagnostics available.
        self.assertEqual(
            result.mass_conservation_error["absolute"].shape, (3,)
        )
        self.assertTrue(
            np.all(result.variance_conservation_error["absolute"] < 1e-10)
        )

    def test_lists_and_numpy_arrays_accepted_equivalently(self):
        sources = [(0.0, 2.0), (2.0, 4.0)]
        targets = [(0.0, 3.0)]
        from_lists = rebin(sources, [2.0, 4.0], [3.0, 5.0], targets)
        from_arrays = rebin(
            np.array(sources), np.array([2.0, 4.0]),
            np.array([3.0, 5.0]), np.array(targets)
        )
        np.testing.assert_array_equal(from_lists.target_mass,
                                      from_arrays.target_mass)
        np.testing.assert_array_equal(from_lists.target_cov,
                                      from_arrays.target_cov)


if __name__ == "__main__":
    unittest.main()
