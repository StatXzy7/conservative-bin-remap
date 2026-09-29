import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conservative_remap import RebinError, remap, remap_group
from tests._util import reference_remap


def assert_close(test, got, expected, tol=1e-11):
    test.assertTrue(
        abs(got - expected) <= tol * max(1.0, abs(expected)),
        f"{got!r} !~= {expected!r}",
    )


def assert_matrix_close(test, got, expected, tol=1e-11):
    test.assertEqual(len(got), len(expected))
    for i, row in enumerate(got):
        test.assertEqual(len(row), len(expected[i]))
        for j, v in enumerate(row):
            test.assertTrue(
                abs(v - expected[i][j]) <= tol * max(1.0, abs(expected[i][j])),
                f"[{i}][{j}] {v!r} !~= {expected[i][j]!r}",
            )


class FullCoverageTests(unittest.TestCase):
    def test_identity_full_coverage(self):
        src = [(0, 2), (2, 5), (5, 9)]
        tgt = [(0, 2), (2, 5), (5, 9)]
        totals = [3.0, 7.0, 2.0]
        var = [1.0, 4.0, 0.5]
        r = remap_group(src, tgt, totals, var)
        self.assertEqual(r.totals, totals)
        for i, v in enumerate(var):
            assert_close(self, r.variances[i], v)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert_close(self, r.covariance[i][j], 0.0)
        self.assertEqual(r.uncovered_totals, [])
        assert_close(self, r.diagnostics.total_abs_error, 0.0)
        assert_close(self, r.diagnostics.variance_abs_error, 0.0)

    def test_full_coverage_with_split_correlation(self):
        # One source split evenly across two targets: correlation is 1/2.
        r = remap_group([(0.0, 4.0)], [(0.0, 2.0), (2.0, 4.0)], [8.0], [2.0])
        self.assertEqual(r.totals, [4.0, 4.0])
        assert_matrix_close(self, r.covariance, [[0.5, 0.5], [0.5, 0.5]])

    def test_matches_reference_full_cover(self):
        src = [(0, 10), (10, 20)]
        tgt = [(0, 5), (5, 12), (12, 20)]
        totals, var = [10.0, 20.0], [3.0, 8.0]
        r = remap_group(src, tgt, totals, var)
        _, means, cov = reference_remap(src, tgt, totals, var)
        for i in range(3):
            assert_close(self, r.totals[i], means[i])
        assert_matrix_close(self, r.covariance, cov)


class PartialCoverageTests(unittest.TestCase):
    def setUp(self):
        self.src = [(0.0, 10.0), (10.0, 20.0)]
        self.tgt = [(2.0, 6.0), (6.0, 16.0)]
        self.totals = [10.0, 20.0]
        self.var = [3.0, 8.0]
        self.r = remap_group(self.src, self.tgt, self.totals, self.var)

    def test_target_totals_hand_calculated(self):
        self.assertEqual(self.r.totals, [4.0, 16.0])
        assert_close(self, self.r.variances[0], 0.48)
        assert_close(self, self.r.variances[1], 3.36)

    def test_nonzero_covariance_after_split(self):
        # Source 0 is split between target0, target1 and the left uncovered
        # part; these must be correlated.
        self.assertGreater(self.r.covariance[0][1], 0.0)
        assert_close(self, self.r.covariance[0][1], 3.0 * 0.4 * 0.4)
        assert_close(self, self.r.cross_covariance[0][0], 3.0 * 0.2 * 0.4)
        assert_close(self, self.r.cross_covariance[0][1], 3.0 * 0.2 * 0.4)
        # The right uncovered part only shares source 1 with target 1.
        assert_close(self, self.r.cross_covariance[1][0], 0.0)
        assert_close(self, self.r.cross_covariance[1][1], 8.0 * 0.4 * 0.6)
        # The two disconnected uncovered parts are independent.
        assert_close(self, self.r.uncovered_covariance[0][1], 0.0)

    def test_uncovered_parts(self):
        self.assertEqual(self.r.uncovered_intervals, [(0.0, 2.0), (16.0, 20.0)])
        self.assertEqual(self.r.uncovered_totals, [2.0, 8.0])
        assert_close(self, self.r.uncovered_variances[0], 3.0 * 0.2 ** 2)
        assert_close(self, self.r.uncovered_variances[1], 8.0 * 0.4 ** 2)

    def test_full_matrix_matches_reference(self):
        elems, means, cov = reference_remap(
            self.src, self.tgt, self.totals, self.var, uncovered=True
        )
        m = len(self.tgt)
        # Reference uncovered ordering follows ascending x, same as engine.
        self.assertEqual(elems[m:], self.r.uncovered_intervals)
        for e in range(len(elems)):
            expected_mean = means[e]
            got_mean = self.r.totals[e] if e < m else self.r.uncovered_totals[e - m]
            assert_close(self, got_mean, expected_mean)
        full = [row[:m] + row[m:] for row in
                [self.r.covariance[i] + self.r.cross_covariance[i]
                 for i in range(m)] +
                [self.r.cross_covariance[i] + self.r.uncovered_covariance[i]
                 for i in range(len(elems) - m)]]
        # Build full matrix explicitly instead of the convoluted expression.
        size = m + len(self.r.uncovered_totals)
        full = [[0.0] * size for _ in range(size)]
        for i in range(m):
            for j in range(m):
                full[i][j] = self.r.covariance[i][j]
        for i in range(size - m):
            for j in range(m):
                full[m + i][j] = self.r.cross_covariance[i][j]
                full[j][m + i] = self.r.cross_covariance[i][j]
            for j in range(size - m):
                full[m + i][m + j] = self.r.uncovered_covariance[i][j]
        assert_matrix_close(self, full, cov)

    def test_merge_conservation(self):
        d = self.r.diagnostics
        assert_close(self, d.total_in, 30.0)
        assert_close(self, d.total_out, 30.0)
        assert_close(self, d.variance_in, 11.0)
        assert_close(self, d.variance_out, 11.0)
        assert_close(self, d.total_abs_error, 0.0)
        assert_close(self, d.variance_abs_error, 0.0)
        # Merged scalar variance: 1^T Sigma 1 over targets + uncovered.
        merged = sum(sum(row) for row in self.r.covariance)
        merged += sum(sum(row) for row in self.r.uncovered_covariance)
        merged += 2.0 * sum(sum(row) for row in self.r.cross_covariance)
        assert_close(self, merged, 11.0)
        assert_close(self,
                     math.fsum(self.r.totals) + math.fsum(self.r.uncovered_totals),
                     30.0)


class GapsAndOrderTests(unittest.TestCase):
    def test_targets_with_gap_and_outside(self):
        src = [(0, 10)]
        tgt = [(2, 4), (8, 9), (20, 25)]
        r = remap_group(src, tgt, [10.0], [4.0])
        # Fractions 0.2, 0.1, 0 (outside source), uncovered fraction 0.7.
        self.assertEqual([round(v, 12) for v in r.totals], [2.0, 1.0, 0.0])
        # Gaps covered by targets separate the uncovered runs.
        self.assertEqual(r.uncovered_intervals, [(0.0, 2.0), (4.0, 8.0), (9.0, 10.0)])
        self.assertEqual([round(v, 12) for v in r.uncovered_totals],
                         [2.0, 4.0, 1.0])
        assert_close(self, math.fsum(r.uncovered_totals), 7.0)
        assert_close(self, r.covariance[0][1], 4.0 * 0.2 * 0.1)
        # Outside-source target has zero variance/covariance.
        for j in range(3):
            assert_close(self, r.covariance[2][j], 0.0)

    def test_shuffled_input_preserves_caller_order(self):
        src = [(10.0, 20.0), (0.0, 10.0)]
        tgt = [(6.0, 16.0), (2.0, 6.0)]
        totals = [20.0, 10.0]
        var = [8.0, 3.0]
        r = remap_group(src, tgt, totals, var)
        ordered = remap_group(
            [(0.0, 10.0), (10.0, 20.0)],
            [(2.0, 6.0), (6.0, 16.0)],
            [10.0, 20.0], [3.0, 8.0],
        )
        # Result rows/columns permuted: shuffled target 0 == ordered target 1.
        assert_close(self, r.totals[0], ordered.totals[1])
        assert_close(self, r.totals[1], ordered.totals[0])
        assert_close(self, r.covariance[0][0], ordered.covariance[1][1])
        assert_close(self, r.covariance[1][1], ordered.covariance[0][0])
        assert_close(self, r.covariance[0][1], ordered.covariance[0][1])

    def test_touching_boundaries_no_double_count(self):
        r = remap_group(
            [(0, 5), (5, 10)], [(0, 5), (5, 10)],
            [7.0, 3.0], [2.0, 4.0],
        )
        self.assertEqual(r.totals, [7.0, 3.0])
        assert_matrix_close(self, r.covariance, [[2.0, 0.0], [0.0, 4.0]])
        self.assertEqual(r.uncovered_totals, [])

    def test_zero_total_and_zero_variance_valid(self):
        r = remap_group([(0, 4)], [(0, 2), (2, 4)], [0.0], [0.0])
        self.assertEqual(r.totals, [0.0, 0.0])
        self.assertTrue(all(v == 0.0 for row in r.covariance for v in row))
        assert_close(self, r.diagnostics.rel_total_error, 0.0)
        assert_close(self, r.diagnostics.rel_variance_error, 0.0)

    def test_zero_variance_with_positive_total(self):
        r = remap_group([(0, 3)], [(0, 1), (1, 3)], [9.0], [0.0])
        self.assertEqual(r.totals, [3.0, 6.0])
        self.assertTrue(all(v == 0.0 for row in r.covariance for v in row))


class MultiGroupTests(unittest.TestCase):
    def test_two_groups_share_boundaries(self):
        src = [(0, 10), (10, 20)]
        tgt = [(2, 6), (6, 16)]
        results = remap(src, tgt,
                        [[10.0, 20.0], [5.0, 0.0]],
                        [[3.0, 8.0], [1.0, 2.0]])
        self.assertEqual(len(results), 2)
        assert_close(self, results[0].totals[0], 4.0)
        assert_close(self, results[1].totals[0], 2.0)
        assert_close(self, results[1].uncovered_totals[0], 1.0)
        self.assertEqual(results[0].uncovered_intervals,
                         results[1].uncovered_intervals)

    def test_single_group_flat_input(self):
        results = remap([(0, 4)], [(0, 2), (2, 4)], [8.0], [2.0])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].totals, [4.0, 4.0])


class ValidationTests(unittest.TestCase):
    def _expect_error(self, *args, msg=None):
        with self.assertRaises(RebinError) as ctx:
            remap(*args)
        if msg is not None:
            self.assertIn(msg, str(ctx.exception))

    def test_negative_total(self):
        self._expect_error([(0, 1)], [(0, 1)], [-1e-12], [1.0], msg="negative")

    def test_negative_variance(self):
        self._expect_error([(0, 1)], [(0, 1)], [1.0], [-1.0], msg="negative")

    def test_nonfinite_values(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(RebinError):
                remap_group([(0, 1)], [(0, 1)], [1.0], [bad])
            with self.assertRaises(RebinError):
                remap_group([(0, 1)], [(0, bad)], [1.0], [1.0])

    def test_zero_width_interval(self):
        self._expect_error([(1, 1)], [(0, 1)], [1.0], [1.0], msg="width")
        self._expect_error([(2, 1)], [(0, 1)], [1.0], [1.0], msg="width")

    def test_overlapping_intervals(self):
        self._expect_error(
            [(0, 2), (1, 3)], [(0, 1)], [1.0, 1.0], [1.0, 1.0], msg="overlap")
        self._expect_error(
            [(0, 1)], [(0, 2), (1, 3)], [1.0], [1.0], msg="overlap")

    def test_dimension_mismatch(self):
        self._expect_error([(0, 1), (1, 2)], [(0, 2)], [1.0], [1.0, 1.0])
        self._expect_error([(0, 1)], [(0, 1)], [1.0, 2.0], [1.0])

    def test_bad_interval_shape(self):
        with self.assertRaises(RebinError):
            remap_group([(0, 1, 2)], [(0, 1)], [1.0], [1.0])
        with self.assertRaises(RebinError):
            remap_group([("a", 1)], [(0, 1)], [1.0], [1.0])

    def test_empty_intervals_rejected(self):
        with self.assertRaises(RebinError):
            remap_group([], [(0, 1)], [], [])
        with self.assertRaises(RebinError):
            remap_group([(0, 1)], [], [1.0], [1.0])

    def test_group_count_mismatch(self):
        with self.assertRaises(RebinError):
            remap([(0, 1)], [(0, 1)], [[1.0], [2.0]], [[1.0]])

    def test_failure_does_not_mutate_inputs(self):
        import copy
        src = [(0.0, 2.0), (2.0, 4.0)]
        tgt = [(0.0, 4.0)]
        totals = [1.0, 2.0]
        var = [1.0, -3.0]  # invalid
        src0, tgt0 = copy.deepcopy(src), copy.deepcopy(tgt)
        tot0, var0 = copy.deepcopy(totals), copy.deepcopy(var)
        with self.assertRaises(RebinError):
            remap(src, tgt, totals, var)
        self.assertEqual(src, src0)
        self.assertEqual(tgt, tgt0)
        self.assertEqual(totals, tot0)
        self.assertEqual(var, var0)

    def test_success_does_not_mutate_inputs(self):
        import copy
        src = [(0.0, 2.0)]
        tgt = [(0.0, 1.0), (1.0, 2.0)]
        totals = [4.0]
        var = [2.0]
        snap = copy.deepcopy((src, tgt, totals, var))
        remap(src, tgt, totals, var)
        self.assertEqual(copy.deepcopy((src, tgt, totals, var)), snap)


class RandomizedReferenceTests(unittest.TestCase):
    def test_random_cases_against_reference(self):
        import random
        rng = random.Random(1234)
        for _ in range(40):
            ns = rng.randint(1, 4)
            edges = sorted(rng.uniform(0.0, 100.0) for _ in range(ns + 1))
            while any(b - a < 1e-6 for a, b in zip(edges, edges[1:])):
                edges = sorted(rng.uniform(0.0, 100.0) for _ in range(ns + 1))
            src = list(zip(edges[:-1], edges[1:]))
            nt = rng.randint(1, 5)
            te = sorted(rng.uniform(-10.0, 110.0) for _ in range(2 * nt))
            while any(b - a < 1e-6 for a, b in zip(te[::2], te[1::2])):
                te = sorted(rng.uniform(-10.0, 110.0) for _ in range(2 * nt))
            tgt = list(zip(te[::2], te[1::2]))
            totals = [rng.uniform(0.0, 50.0) for _ in range(ns)]
            var = [rng.uniform(0.0, 10.0) for _ in range(ns)]
            r = remap_group(src, tgt, totals, var)
            elems, means, cov = reference_remap(src, tgt, totals, var,
                                                uncovered=True)
            m = len(tgt)
            u = len(elems) - m
            self.assertEqual(len(r.uncovered_intervals), u)
            self.assertEqual(r.uncovered_intervals, elems[m:])
            size = m + u
            full = [[0.0] * size for _ in range(size)]
            for i in range(m):
                for j in range(m):
                    full[i][j] = r.covariance[i][j]
            for i in range(u):
                for j in range(m):
                    full[m + i][j] = r.cross_covariance[i][j]
                    full[j][m + i] = r.cross_covariance[i][j]
                for j in range(u):
                    full[m + i][m + j] = r.uncovered_covariance[i][j]
            assert_matrix_close(self, full, cov, tol=1e-9)
            all_means = r.totals + r.uncovered_totals
            for e in range(size):
                assert_close(self, all_means[e], means[e], tol=1e-9)
            assert_close(self, r.diagnostics.total_abs_error, 0.0, tol=1e-9)


if __name__ == "__main__":
    unittest.main()
