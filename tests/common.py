"""Independent exact reference implementation using fractions.Fraction.

This is deliberately written from scratch (no overlap shortcuts) so tests
compare the float engine against a logically independent computation rather
than against the same algorithm or mere array shapes.
"""

from fractions import Fraction


def exact_remap(source_edges, target_edges, totals, variances):
    n = len(source_edges)
    m = len(target_edges)
    sources = [(Fraction(lo), Fraction(hi)) for lo, hi in source_edges]
    targets = [(Fraction(lo), Fraction(hi)) for lo, hi in target_edges]
    T = [Fraction(x) for x in totals]
    V = [Fraction(x) for x in variances]

    width = [hi - lo for lo, hi in sources]
    frac = [[Fraction(0)] * n for _ in range(m)]
    for j in range(m):
        for i in range(n):
            ov = min(targets[j][1], sources[i][1]) - max(
                targets[j][0], sources[i][0]
            )
            if ov > 0:
                frac[j][i] = ov / width[i]

    covered = [sum((frac[j][i] for j in range(m)), Fraction(0)) for i in range(n)]
    un_f = [Fraction(1) - c for c in covered]

    t_tot = [sum((frac[j][i] * T[i] for i in range(n)), Fraction(0)) for j in range(m)]
    cov = [[Fraction(0)] * m for _ in range(m)]
    for j in range(m):
        for k in range(m):
            cov[j][k] = sum(
                (frac[j][i] * frac[k][i] * V[i] for i in range(n)), Fraction(0)
            )
    u_tot = sum((un_f[i] * T[i] for i in range(n)), Fraction(0))
    u_var = sum((un_f[i] ** 2 * V[i] for i in range(n)), Fraction(0))
    u_cov = [
        sum((un_f[i] * frac[j][i] * V[i] for i in range(n)), Fraction(0))
        for j in range(m)
    ]
    return {
        "target_totals": t_tot,
        "target_cov": cov,
        "uncovered_total": u_tot,
        "uncovered_variance": u_var,
        "uncovered_cov": u_cov,
    }
