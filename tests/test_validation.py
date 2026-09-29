import unittest

import numpy as np

from rebinning import RebinError, rebin
from tests.reference import TOL


class TestValidation(unittest.TestCase):
    def _good_kwargs(self, **overrides):
        kwargs = dict(
            source_intervals=[(0.0, 2.0), (2.0, 4.0)],
            mass=[2.0, 2.0],
            variance=[1.0, 3.0],
            target_intervals=[(0.0, 1.0), (3.0, 4.0)],
        )
        kwargs.update(overrides)
        return kwargs

    def assertRejected(self, **overrides):
        with self.assertRaises(RebinError):
            rebin(**self._good_kwargs(**overrides))

    def test_negative_mass(self):
        self.assertRejected(mass=[-0.1, 2.0])

    def test_negative_variance(self):
        self.assertRejected(variance=[1.0, -3.0])

    def test_nan_and_infinity(self):
        self.assertRejected(mass=[np.nan, 2.0])
        self.assertRejected(variance=[np.inf, 3.0])
        self.assertRejected(source_intervals=[(0.0, np.nan), (2.0, 4.0)])
        self.assertRejected(target_intervals=[(0.0, 1.0), (3.0, np.inf)])

    def test_zero_width_intervals(self):
        self.assertRejected(source_intervals=[(2.0, 2.0), (2.0, 4.0)])
        self.assertRejected(source_intervals=[(3.0, 1.0), (2.0, 4.0)])
        self.assertRejected(target_intervals=[(1.0, 1.0)])

    def test_overlapping_source_intervals(self):
        self.assertRejected(source_intervals=[(0.0, 2.5), (2.0, 4.0)])

    def test_overlapping_target_intervals(self):
        with self.assertRaises(RebinError):
            rebin([(0.0, 10.0)], [10.0], [10.0],
                  [(0.0, 6.0), (5.0, 10.0)])

    def test_dimension_mismatch(self):
        self.assertRejected(mass=[2.0])  # only 1 mass for 2 sources
        self.assertRejected(variance=[1.0, 3.0, 9.0])
        with self.assertRaises(RebinError):
            rebin([(0.0, 2.0), (2.0, 4.0)],
                  [[2.0, 2.0]], [[1.0, 3.0, 9.0]],
                  [(0.0, 1.0)])

    def test_malformed_interval_shape(self):
        good = self._good_kwargs()
        bad_cases = [
            [0.0, 2.0, 4.0],            # 1-D
            [(0.0, 2.0, 4.0)],          # triples
            [],                         # empty is fine for targets only
        ]
        # 1-D / triple sources rejected:
        with self.assertRaises(RebinError):
            rebin(bad_cases[0], good["mass"], good["variance"],
                  good["target_intervals"])
        with self.assertRaises(RebinError):
            rebin(bad_cases[1], good["mass"], good["variance"],
                  good["target_intervals"])

    def test_mass_variance_group_shape_mismatch(self):
        with self.assertRaises(RebinError):
            rebin([(0.0, 2.0)], [[1.0]], [[2.0], [3.0]], [(0.0, 1.0)])

    def test_error_messages_are_descriptive(self):
        try:
            rebin([(0.0, 2.0), (2.0, 4.0)], [-1.0, 2.0], [1.0, 1.0],
                  [(0.0, 1.0)])
        except RebinError as exc:
            self.assertIn("negative", str(exc))
        else:
            self.fail("RebinError not raised")


class TestZeroValues(unittest.TestCase):
    def test_zero_mass_and_variance_is_valid(self):
        result = rebin([(0.0, 4.0)], [0.0], [0.0],
                       [(0.0, 2.0), (2.0, 4.0)])
        np.testing.assert_array_equal(result.target_mass[0], [0, 0])
        np.testing.assert_array_equal(result.target_cov[0], 0.0)
        np.testing.assert_array_equal(result.uncovered_mass, 0.0)

    def test_zero_variance_with_positive_mass(self):
        # Perfectly known total: all pieces are deterministic fractions.
        result = rebin([(0.0, 4.0)], [4.0], [0.0],
                       [(0.0, 1.0), (3.0, 4.0)])
        # var(f M) = f^2 V + M f(1-f) = M f (1-f)
        np.testing.assert_allclose(result.target_variance[0], [0.75, 0.75],
                                   atol=TOL)
        # Same source, perfectly known total: pieces move oppositely.
        self.assertAlmostEqual(result.target_cov[0, 0, 1], -0.25)
        # uncovered f = 1/2
        self.assertAlmostEqual(result.uncovered_variance[0], 1.0)
        np.testing.assert_allclose(result.uncovered_cov[0], [-0.5, -0.5],
                                   atol=TOL)
        total_var = (result.target_cov[0].sum()
                     + 2 * result.uncovered_cov[0].sum()
                     + result.uncovered_variance[0])
        self.assertAlmostEqual(total_var, 0.0, places=10)

    def test_zero_mass_positive_variance(self):
        # Zero mass but uncertain measurement is still a valid variance.
        result = rebin([(0.0, 2.0)], [0.0], [8.0],
                       [(0.0, 1.0), (1.0, 2.0)])
        np.testing.assert_array_equal(result.target_mass[0], [0, 0])
        np.testing.assert_allclose(result.target_variance[0], [2, 2],
                                   atol=TOL)
        self.assertAlmostEqual(result.target_cov[0, 0, 1], 2.0)


if __name__ == "__main__":
    unittest.main()
