"""Command-line demo for the conservative rebinning engine.

Run with::

    python -m rebinning.demo

Two hand-computed examples are printed:

1. Full coverage of one over-dispersed source interval split into three
   target bins.  The expected numbers are derived analytically in the
   comments below so the demo doubles as an executable specification.
2. Partial coverage plus a gap, showing the uncovered remainder and its
   covariance with the outputs, with two measurement groups sharing the
   same boundaries.
"""

from __future__ import annotations

import numpy as np

from .engine import rebin


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _fmt_row(values) -> str:
    return "  ".join(f"{v:12.6f}" for v in values)


def demo_full_split() -> None:
    _print_header("Example 1: one source split into three target bins (full coverage)")
    # Source: [0, 4), mass M = 4, variance V = 8 (over-dispersed).
    # Targets: [0,1) f=1/4, [1,3) f=2/4, [3,4) f=1/4.
    #
    # Single-source covariance C = (V-M) f f^T + diag(M f), V-M = 4:
    #   masses       = (1, 2, 1)
    #   variances    = (1.25, 3, 1.25)    [8 f^2 + 4 f(1-f)]
    #   cov(0,1)     = 4 * 1/4 * 1/2 = 0.5
    #   cov(0,2)     = 4 * 1/4 * 1/4 = 0.25
    #   cov(1,2)     = 4 * 1/2 * 1/4 = 0.5
    # Sum of all 9 covariance entries = 8 = original V (conservation).
    sources = [(0.0, 4.0)]
    mass = [4.0]
    variance = [8.0]
    targets = [(0.0, 1.0), (1.0, 3.0), (3.0, 4.0)]

    result = rebin(sources, mass, variance, targets)

    print("sources [(start,end), mass, variance]:")
    print("  [0, 4)  M=4.000000  V=8.000000")
    print("targets (caller order kept): [0,1) [1,3) [3,4)")
    print()
    print("target masses    :", _fmt_row(result.target_mass[0]))
    print("expected         :", _fmt_row([1.0, 2.0, 1.0]))
    print()
    print("target covariance matrix (diagonal = variance):")
    for row in result.target_cov[0]:
        print("  " + _fmt_row(row))
    print("expected matrix  :")
    expected = np.array([[1.25, 0.5, 0.25], [0.5, 3.0, 0.5],
                         [0.25, 0.5, 1.25]])
    for row in expected:
        print("  " + _fmt_row(row))
    print()
    print(f"uncovered mass/variance : {result.uncovered_mass[0]:.6f} / "
          f"{result.uncovered_variance[0]:.6f} (expected 0 / 0)")
    print(f"merged total mass       : {result.total_mass[0]:.6f} "
          f"(input 4.000000)")
    print(f"merged total variance   : {result.total_variance[0]:.6f} "
          f"(input 8.000000)")
    print(f"mass abs/rel error      : "
          f"{result.mass_conservation_error['absolute'][0]:.3e} / "
          f"{result.mass_conservation_error['relative'][0]:.3e}")
    print(f"variance abs/rel error  : "
          f"{result.variance_conservation_error['absolute'][0]:.3e} / "
          f"{result.variance_conservation_error['relative'][0]:.3e}")


def demo_partial_groups() -> None:
    _print_header("Example 2: partial coverage, gap, two measurement groups")
    # Sources: [0,2) and [2,6); targets [0,1) and [3,5), caller order.
    # Group A: M = (2, 8), V = (2, 8)  -> V == M (Poisson-like): split
    #          covariances vanish, variances scale linearly with fraction.
    # Group B: M = (2, 4), V = (6, 12).
    #
    # Fractions:
    #   source 0 -> t0 f=1/2 ; source 1 -> t1 f=2/4=1/2
    #   uncovered fractions: f_u = (1/2, 1/2)
    #
    # Group A expected:
    #   target masses = (1, 4), variances = (1, 4), cov(t0,t1) = 0
    #   uncovered mass = 1 + 4 = 5, uncovered variance = 1 + 4 = 5
    #   uncovered cov  = (0, 0)
    # Group B expected (V-M = (4, 8)):
    #   target masses = (1, 2)
    #   var t0 = 4*(1/2)^2 + 2*(1/2) = 1 + 1 = 2
    #   var t1 = 8*(1/2)^2 + 4*(1/2) = 2 + 2 = 4
    #   cov(t0,t1) = 0 (different independent sources)
    #   uncovered mass = 1 + 2 = 3
    #   var u = [6*(1/2)^2 + 2*(1/2)*(1/2)]
    #         + [12*(1/2)^2 + 4*(1/2)*(1/2)]
    #         = (1.5 + 0.5) + (3 + 1) = 6
    #   merged: variances 2+4 + 2*cov(0) + 6 + 2*(1+2) = 18 = input V
    #   uncovered cov = (4 * 1/2 * 1/2, 8 * 1/2 * 1/2) = (1, 2)
    sources = [(0.0, 2.0), (2.0, 6.0)]
    mass = [[2.0, 8.0], [2.0, 4.0]]
    variance = [[2.0, 8.0], [6.0, 12.0]]
    targets = [(0.0, 1.0), (3.0, 5.0)]

    result = rebin(sources, mass, variance, targets)

    print("sources: [0,2), [2,6)   targets: [0,1), [3,5)  (gap [1,3) etc.)")
    for g in range(2):
        print()
        print(f"--- group {g} ---")
        print(f"target masses   : {_fmt_row(result.target_mass[g])}")
        print(f"target variances: {_fmt_row(result.target_variance[g])}")
        print(f"target cov      : "
              f"[{result.target_cov[g, 0, 1]:.6f}] (off-diagonal)")
        print(f"uncovered mass  : {result.uncovered_mass[g]:.6f}")
        print(f"uncovered var   : {result.uncovered_variance[g]:.6f}")
        print(f"uncovered cov   : {_fmt_row(result.uncovered_cov[g])}")
        print(f"merged mass/var : {result.total_mass[g]:.6f} / "
              f"{result.total_variance[g]:.6f} "
              f"(inputs {sum(mass[g]):.1f} / {sum(variance[g]):.1f})")
    print()
    print("expected group A: masses (1,4) vars (1,4) cov 0  "
          "unc 5/5 cov (0,0)")
    print("expected group B: masses (1,2) vars (2,4) cov 0 "
          " unc 3/6 cov (1,2)")
    print()
    print("per-group mass abs errors     :",
          _fmt_row(result.mass_conservation_error["absolute"]))
    print("per-group variance abs errors :",
          _fmt_row(result.variance_conservation_error["absolute"]))


def main() -> None:
    demo_full_split()
    demo_partial_groups()
    _print_header("Note")
    print("Rebinning discards all within-bin density information (uniform")
    print("assumption).  Splitting and re-merging conserves the integrated")
    print("mass and covariance, but it cannot recover the original bin layout")
    print("or any non-uniform density; see README.md for details.")


if __name__ == "__main__":
    main()
