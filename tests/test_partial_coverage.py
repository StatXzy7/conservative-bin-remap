import unittest

import numpy as np

from rebinning import rebin
from tests.reference import TOL, reference_rebin


SOURCES = [(0.0, 2.0), (2.0, 6.0)]
TARGETS = [(0.0, 1.0), (3.0, 5.0)]  # gap; each source half covered
MASS = [[2.0, 8.0], [2.0, 4.0]]
VARIANCE = [[2.0, 8.0], [6.0, 12.0]]


class TestPartialCoverage(unittest.TestCase):
    def setUp(self):
        self.result = rebin(SOURCES, MASS, VARIANCE, TARGETS)
        self.ref = reference_rebin(SOURCES, MASS, VARIANCE, TARGETS)

    def test_hand_computed_group_a(self):
        r = self.result
        np.testing.assert_allclose(r.target_mass[0], [1, 4], atol=TOL)
        np.testing.assert_allclose(r.target_variance[0], [1, 4], atol=TOL)
        self.assertAlmostEqual(r.target_cov[0, 0, 1], 0.0)
        self.assertAlmostEqual(r.uncovered_mass[0], 5.0)
        self.assertAlmostEqual(r.uncovered_variance[0], 5.0)
        np.testing.assert_allclose(r.uncovered_cov[0], [0, 0], atol=TOL)

    def test_hand_computed_group_b(self):
        r = self.result
        np.testing.assert_allclose(r.target_mass[1], [1, 2], atol=TOL)
        np.testing.assert_allclose(r.target_variance[1], [2.0, 4.0],
                                   atol=TOL)
        self.assertAlmostEqual(r.uncovered_mass[1], 3.0)
        self.assertAlmostEqual(r.uncovered_variance[1], 6.0)
        np.testing.assert_allclose(r.uncovered_cov[1], [1, 2], atol=TOL)

    def test_targets_from_independent_sources_stay_uncorrelated(self):
        # t0 draws only from source 0, t1 only from source 1.
        self.assertAlmostEqual(self.result.target_cov[0, 0, 1], 0.0)
        self.assertAlmostEqual(self.result.target_cov[1, 0, 1], 0.0)

    def test_matches_independent_reference(self):
        for key in ("target_mass", "target_cov", "uncovered_mass",
                    "uncovered_variance", "uncovered_cov"):
            np.testing.assert_allclose(getattr(self.result, key),
                                       self.ref[key], atol=TOL)

    def test_target_outside_source_range_allocates_nothing(self):
        targets = [(10.0, 12.0)]
        result = rebin(SOURCES, MASS, VARIANCE, targets)
        np.testing.assert_allclose(result.target_mass, 0.0)
        np.testing.assert_allclose(result.target_cov, 0.0)
        np.testing.assert_allclose(result.uncovered_cov, 0.0)
        # All source mass remains uncovered.
        np.testing.assert_allclose(result.uncovered_mass,
                                   np.sum(MASS, axis=1))

    def test_target_partly_outside_source_range(self):
        # Target starts inside source 0 but extends past its end; the part
        # outside the source contributes nothing.
        sources = [(0.0, 2.0)]
        result = rebin(sources, [4.0], [4.0], [(-3.0, 1.0)])
        # Half-open overlap is [0,1): half of the source.
        np.testing.assert_allclose(result.target_mass[0], [2.0], atol=TOL)
        np.testing.assert_allclose(result.uncovered_mass[0], 2.0, atol=TOL)

    def test_empty_targets_reports_everything_as_uncovered(self):
        result = rebin(SOURCES, MASS, VARIANCE, [])
        self.assertEqual(result.target_mass.shape, (2, 0))
        self.assertEqual(result.target_cov.shape, (2, 0, 0))
        np.testing.assert_allclose(result.uncovered_mass,
                                   np.sum(MASS, axis=1))
        np.testing.assert_allclose(result.uncovered_variance,
                                   np.sum(VARIANCE, axis=1))


if __name__ == "__main__":
    unittest.main()
