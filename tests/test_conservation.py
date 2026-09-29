import unittest

import numpy as np

from rebinning import rebin
from tests.reference import TOL, reference_rebin


def merged_totals(result):
    """Re-merge every output piece and the uncovered remainder."""
    total_mass = result.target_mass.sum(axis=1) + result.uncovered_mass
    total_variance = (
        result.target_cov.sum(axis=(1, 2))
        + 2.0 * result.uncovered_cov.sum(axis=1)
        + result.uncovered_variance
    )
    return total_mass, total_variance


class TestConservation(unittest.TestCase):
    RNG_SEED = 20260929

    def _random_case(self, n_sources, n_targets, n_groups):
        rng = np.random.default_rng(self.RNG_SEED)
        # Random partition of [0, L) into source bins.
        edges = np.sort(rng.uniform(0, 20, size=n_sources + 1))
        edges[0] = 0.0
        edges[-1] = 20.0
        edges = np.unique(edges)
        if edges.size < 2:
            self.skipTest("degenerate random partition")
        sources = list(zip(edges[:-1], edges[1:]))
        n_sources = len(sources)
        # Non-overlapping targets with gaps: build a strictly increasing
        # edge chain, alternate gaps vs. covered windows.
        picks = np.sort(rng.uniform(0, 20, size=2 * n_targets + 1))
        if picks[-1] - picks[0] < 1e-9:
            self.skipTest("degenerate random targets")
        targets = [(float(picks[2 * i + 1]), float(picks[2 * i + 2]))
                   for i in range(n_targets)]
        mass = rng.uniform(0, 10, size=(n_groups, n_sources))
        variance = mass * rng.uniform(0.25, 4.0, size=(n_groups, n_sources))
        return sources, mass, variance, targets

    def test_full_coverage_conserves_mass_and_variance(self):
        # Targets exactly tile the sources (fractional cuts) -> no uncovered.
        sources = [(0.0, 3.0), (3.0, 10.0), (10.0, 11.0)]
        # Cut across source boundaries and through the middle source.
        targets = [(0.0, 2.0), (2.0, 3.0), (3.0, 7.0), (7.0, 10.0),
                   (10.0, 11.0)]
        mass = np.array([[3.0, 14.0, 2.0]])
        variance = np.array([[5.0, 21.0, 0.5]])
        result = rebin(sources, mass, variance, targets)
        np.testing.assert_allclose(result.uncovered_mass, 0.0, atol=TOL)
        np.testing.assert_allclose(result.uncovered_variance, 0.0, atol=TOL)
        tm, tv = merged_totals(result)
        np.testing.assert_allclose(tm, mass.sum(axis=1), atol=TOL)
        np.testing.assert_allclose(tv, variance.sum(axis=1), atol=TOL)

    def test_partial_coverage_merge_round_trip(self):
        for n_sources, n_targets, n_groups in [
            (1, 3, 1), (3, 5, 2), (6, 10, 4)
        ]:
            with self.subTest(case=(n_sources, n_targets, n_groups)):
                sources, mass, variance, targets = self._random_case(
                    n_sources, n_targets, n_groups
                )
                result = rebin(sources, mass, variance, targets)
                tm, tv = merged_totals(result)
                np.testing.assert_allclose(
                    tm, mass.sum(axis=1), rtol=1e-12, atol=1e-10
                )
                np.testing.assert_allclose(
                    tv, variance.sum(axis=1), rtol=1e-12, atol=1e-10
                )
                # Diagnostics reported by the engine agree with the merge.
                np.testing.assert_allclose(
                    result.mass_conservation_error["absolute"],
                    np.abs(tm - mass.sum(axis=1)), atol=1e-12
                )
                np.testing.assert_allclose(
                    result.variance_conservation_error["absolute"],
                    np.abs(tv - variance.sum(axis=1)), atol=1e-12
                )

    def test_rebinning_is_information_losing(self):
        # Mass round-trips through any shared layout, but rebuilding the
        # original variance requires the FULL covariance: treating target
        # bins as independent (diagonal only) loses the split covariance.
        sources = [(0.0, 4.0), (4.0, 7.0)]
        targets = [(0.0, 2.0), (2.0, 4.0), (4.0, 5.5), (5.5, 7.0)]
        mass = np.array([[8.0, 6.0]])
        variance = np.array([[10.0, 12.0]])
        forward = rebin(sources, mass, variance, targets)
        # Mass alone round-trips exactly.
        backward_mass = rebin(
            targets, forward.target_mass, forward.target_mass, sources
        )
        np.testing.assert_allclose(backward_mass.target_mass, mass,
                                   atol=TOL)
        # Diagonal-only "independent" rebinning does NOT restore variance.
        naive = rebin(targets, forward.target_mass,
                      forward.target_variance, sources)
        self.assertFalse(
            np.allclose(naive.target_variance, variance, atol=1e-8)
        )
        # ... whereas merging with the full reported covariance does.
        tm, tv = merged_totals(forward)
        np.testing.assert_allclose(tm, mass.sum(axis=1), atol=TOL)
        np.testing.assert_allclose(tv, variance.sum(axis=1), atol=TOL)

    def test_diagnostics_are_per_group(self):
        sources = [(0.0, 2.0)]
        targets = [(0.0, 1.0), (1.0, 2.0)]
        mass = [[2.0], [4.0]]
        variance = [[3.0], [40.0]]
        result = rebin(sources, mass, variance, targets)
        self.assertEqual(result.mass_conservation_error["absolute"].shape,
                         (2,))
        self.assertEqual(
            result.variance_conservation_error["relative"].shape, (2,)
        )
        for diag in (result.mass_conservation_error,
                     result.variance_conservation_error):
            self.assertTrue(np.all(diag["absolute"] < 1e-10))

    def test_matches_reference_on_random_case(self):
        sources, mass, variance, targets = self._random_case(4, 7, 3)
        result = rebin(sources, mass, variance, targets)
        ref = reference_rebin(sources, mass, variance, targets)
        for key in ("target_mass", "target_cov", "uncovered_mass",
                    "uncovered_variance", "uncovered_cov"):
            np.testing.assert_allclose(getattr(result, key), ref[key],
                                       rtol=1e-12, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
