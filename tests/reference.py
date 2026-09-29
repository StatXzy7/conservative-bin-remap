"""Shared helpers: a deliberately simple independent reference implementation
and tolerance constants for the engine tests."""

import numpy as np

TOL = 1e-10


def overlap_len(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def reference_rebin(sources, mass, variance, targets):
    """Plain-loop reference using the same multinomial split model.

    Returns a dict with arrays identical in meaning to RebinResult.
    Written independently (no einsum, no matrix ops) to cross-check the
    vectorised engine.
    """
    g = len(mass)
    s = len(sources)
    t = len(targets)

    frac = [[overlap_len(*sources[k], *targets[j]) /
             (sources[k][1] - sources[k][0]) for j in range(t)]
            for k in range(s)]
    fu = [1.0 - sum(frac[k]) for k in range(s)]

    tmass = [[sum(mass[h][k] * frac[k][j] for k in range(s))
              for j in range(t)] for h in range(g)]
    tcov = np.zeros((g, t, t))
    umass = [0.0] * g
    uvar = [0.0] * g
    ucov = np.zeros((g, t))
    for h in range(g):
        for k in range(s):
            m, v = mass[h][k], variance[h][k]
            for i in range(t):
                for j in range(t):
                    if i == j:
                        tcov[h, i, j] += (
                            frac[k][i] ** 2 * v + m * frac[k][i] *
                            (1.0 - frac[k][i])
                        )
                    else:
                        tcov[h, i, j] += (
                            frac[k][i] * frac[k][j] * (v - m)
                        )
                ucov[h, i] += frac[k][i] * fu[k] * (v - m)
            uvar[h] += fu[k] ** 2 * v + m * fu[k] * (1.0 - fu[k])
            umass[h] += m * fu[k]
    return {
        "target_mass": np.array(tmass),
        "target_cov": tcov,
        "uncovered_mass": np.array(umass),
        "uncovered_variance": np.array(uvar),
        "uncovered_cov": ucov,
    }
