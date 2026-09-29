"""Independent brute-force reference implementation for cross-checking.

The reference enumerates a fine internal grid and never shares code with the
engine under test.
"""

import math


def overlap(a_lo, a_hi, b_lo, b_hi):
    lo = max(a_lo, b_lo)
    hi = min(a_hi, b_hi)
    return max(0.0, hi - lo)


def reference_remap(sources, targets, totals, variances, uncovered=False):
    """Return (means, full_cov) for elements = targets (+ uncovered parts).

    Uncovered parts are built by splitting the source domain at target
    boundaries and merging touching uncovered runs.
    """
    # Build uncovered regions: union of sources minus union of targets.
    cuts = sorted({x for iv in list(sources) + list(targets) for x in iv})
    unc = []
    for k in range(len(cuts) - 1):
        lo, hi = cuts[k], cuts[k + 1]
        if hi <= lo:
            continue
        mid = (lo + hi) / 2.0
        in_src = any(slo < mid < shi for slo, shi in sources)
        in_tgt = any(tlo < mid < thi for tlo, thi in targets)
        if in_src and not in_tgt:
            if unc and unc[-1][1] == lo:
                unc[-1] = (unc[-1][0], hi)
            else:
                unc.append((lo, hi))
    elems = list(targets) + unc if uncovered else list(targets)

    n = len(sources)
    means = [0.0] * len(elems)
    cov = [[0.0] * len(elems) for _ in range(len(elems))]
    for i, (slo, shi) in enumerate(sources):
        sw = shi - slo
        fracs = []
        for e, (elo, ehi) in enumerate(elems):
            f = overlap(slo, shi, elo, ehi) / sw
            fracs.append(f)
            means[e] += totals[i] * f
        for a in range(len(elems)):
            for b in range(len(elems)):
                cov[a][b] += variances[i] * fracs[a] * fracs[b]
    return elems, means, cov
