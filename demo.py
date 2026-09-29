#!/usr/bin/env python3
"""Command-line demonstration for the interval re-binning engine.

Prints inputs, the operation, computed results, hand-calculated expected
values and explicit assertions.  Exits non-zero if any assertion fails.
Run:  python demo.py
"""

from __future__ import annotations

import sys

from rebin import remap

TOL = 1e-12


def check(label, got, expected, tol=TOL):
    ok = abs(got - expected) <= tol
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}: got={got:.6g} "
          f"expected={expected:.6g}")
    if not ok:
        raise AssertionError(f"{label}: {got} != {expected}")


def check_matrix(label, got, expected, tol=TOL):
    flat_g = [x for row in got for x in row]
    flat_e = [x for row in expected for x in row]
    if len(flat_g) != len(flat_e):
        raise AssertionError(f"{label}: shape mismatch")
    for k, (a, b) in enumerate(zip(flat_g, flat_e)):
        check(f"{label}[{k}]", a, b, tol)


def main() -> int:
    print("=" * 68)
    print("INTERVAL RE-BINNING DEMO  (uniform density, independent sources)")
    print("=" * 68)

    # Two non-overlapping sources on [0,10) and [10,30).
    source_edges = [(0.0, 10.0), (10.0, 30.0)]
    # Two measurement groups sharing the same edges.
    totals = [[30.0, 20.0], [10.0, 8.0]]
    variances = [[9.0, 16.0], [4.0, 0.0]]  # group 2 has a zero variance

    # Targets leave a gap [15,20); source S1 is split over T0 / gap / T1.
    target_edges = [(0.0, 15.0), (20.0, 30.0)]

    print("\nINPUT")
    print("  source intervals: S0=[0,10), S1=[10,30)")
    print("  group 1 totals=[30,20] variances=[9,16]")
    print("  group 2 totals=[10,8]  variances=[4,0]")
    print("  target intervals (caller order kept): T0=[0,15), T1=[20,30)")
    print("  overlap fractions: T0 <- (1.0 of S0, 1/4 of S1);")
    print("                     T1 <- (1/2 of S1); gap [15,20) = uncovered")

    result = remap(source_edges, target_edges, totals, variances)
    g1, g2 = result.groups
    d1, d2 = result.diagnostics

    print("\nOPERATION")
    print("  allocate each source total/variance linearly by overlap length,")
    print("  propagate independent variances: Cov(a,b)=sum f_a f_s Var_s")

    print("\nRESULT group 1")
    print(f"  target totals       = {g1.target_totals}")
    print(f"  target covariance   = {g1.target_cov}")
    print(f"  uncovered total/var = {g1.uncovered_total}, "
          f"{g1.uncovered_variance}")
    print(f"  uncovered cov       = {g1.uncovered_cov}")

    print("\nASSERT group 1 (hand calculation)")
    check("T0 total 30 + 20/4", g1.target_totals[0], 35.0)
    check("T1 total 20/2", g1.target_totals[1], 10.0)
    check("uncovered total 20/4", g1.uncovered_total, 5.0)
    check("Var T0 = 9 + 16/16", g1.target_cov[0][0], 10.0)
    check("Var T1 = 16/4", g1.target_cov[1][1], 4.0)
    check("Cov(T0,T1) = 16*(1/4)*(1/2)  <-- non-zero after split",
          g1.target_cov[0][1], 2.0)
    check("Var uncovered = 16/16", g1.uncovered_variance, 1.0)
    check("Cov(U,T0) = 16*(1/4)*(1/4)", g1.uncovered_cov[0], 1.0)
    check("Cov(U,T1) = 16*(1/4)*(1/2)", g1.uncovered_cov[1], 2.0)

    print("\nCONSERVATION group 1 (merge outputs back with the uncovered part)")
    check("input total", d1.input_total, 50.0)
    check("merged total", d1.output_total, 50.0)
    check("total conservation error", d1.total_error, 0.0)
    check("input variance", d1.input_variance, 25.0)
    check("merged variance incl. all covariances", d1.output_variance, 25.0)
    check("variance conservation error", d1.variance_error, 0.0)

    print("\nRESULT group 2 (zero variance source is legal; same fractions)")
    print(f"  target totals={g2.target_totals} cov={g2.target_cov} "
          f"uncovered={g2.uncovered_total}")
    check("T0 total 10 + 8/4", g2.target_totals[0], 12.0)
    check("T1 total 8/2", g2.target_totals[1], 4.0)
    check("uncovered total 8/4", g2.uncovered_total, 2.0)
    check("merged total", d2.output_total, 18.0)
    check("merged variance", d2.output_variance, 4.0)

    print("\nNON-INVERTIBILITY (why rebin -> restore loses the distribution)")
    edges = [(0.0, 2.0)]
    r = remap(edges, [(0.0, 1.0), (1.0, 2.0)], [10.0], [0.0])
    restored = r.groups[0].target_totals  # uniform split: [5, 5]
    print(f"  original [0,2) total 10 concentrated at the left edge,")
    print(f"  split under uniform-density assumption -> {restored}")
    check("restored left half", restored[0], 5.0)
    check("restored right half", restored[1], 5.0)
    print("  merging restores the total (10) but never the original shape:")
    print("  information about in-bin density variation is discarded.")

    print("\n" + "=" * 68)
    print("ALL DEMO ASSERTIONS PASSED")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
