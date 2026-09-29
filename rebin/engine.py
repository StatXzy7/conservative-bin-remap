"""Interval re-binning engine.

Remap one-dimensional cumulative quantities measured on a set of mutually
non-overlapping *source* intervals onto a different set of mutually
non-overlapping *target* intervals.  Density is assumed uniform inside each
source interval, so allocation is purely linear in overlap length; source
measurements are mutually independent.

All arithmetic is Python float (IEEE-754 binary64).  See README.md for the
precision conventions and the discussion of why re-binning is not invertible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Sequence

__all__ = [
    "RemapError",
    "GroupResult",
    "ConservationDiagnostic",
    "RebinResult",
    "remap",
]


class RemapError(ValueError):
    """Raised for invalid input.  Inputs are never mutated when raised."""


@dataclass(frozen=True)
class GroupResult:
    """Result for one measurement group.

    ``target_cov`` is the full m x m covariance matrix of the target totals;
    ``uncovered_cov`` holds Cov(uncovered, target_j) for every target j, in
    caller-provided target order.
    """

    target_totals: list[float]
    target_cov: list[list[float]]
    uncovered_total: float
    uncovered_variance: float
    uncovered_cov: list[float]


@dataclass(frozen=True)
class ConservationDiagnostic:
    """Per-group conservation residuals (output minus input)."""

    input_total: float
    output_total: float
    total_error: float
    input_variance: float
    output_variance: float
    variance_error: float


@dataclass(frozen=True)
class RebinResult:
    source_edges: list[tuple[float, float]]
    target_edges: list[tuple[float, float]]
    groups: list[GroupResult]
    diagnostics: list[ConservationDiagnostic]


def _is_real(x: object) -> bool:
    return isinstance(x, Real) and not isinstance(x, bool)


def _copy_edges(edges: Iterable, name: str) -> list[tuple[float, float]]:
    if not isinstance(edges, (list, tuple)):
        try:
            edges = list(edges)
        except TypeError as exc:
            raise RemapError(f"{name} must be an iterable of (lo, hi) pairs") from exc
    out: list[tuple[float, float]] = []
    for idx, pair in enumerate(edges):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise RemapError(
                f"{name}[{idx}] must be a (lo, hi) pair, got {pair!r}"
            )
        lo, hi = pair
        if not (_is_real(lo) and _is_real(hi)):
            raise RemapError(f"{name}[{idx}] edges must be real numbers")
        lo, hi = float(lo), float(hi)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise RemapError(f"{name}[{idx}] edges must be finite")
        if not lo < hi:
            raise RemapError(
                f"{name}[{idx}] = ({lo}, {hi}) has zero or negative width"
            )
        out.append((lo, hi))
    if not out:
        raise RemapError(f"{name} must contain at least one interval")
    return out


def _check_nonoverlapping(edges: list[tuple[float, float]], name: str) -> None:
    order = sorted(range(len(edges)), key=lambda k: edges[k])
    for a, b in zip(order, order[1:]):
        prev_hi = edges[a][1]
        next_lo = edges[b][0]
        if next_lo < prev_hi:
            raise RemapError(
                f"{name} intervals overlap: {edges[a]} and {edges[b]} "
                "(touching edges are allowed)"
            )


def _copy_groups(
    totals: Sequence, variances: Sequence, n: int
) -> tuple[list[list[float]], list[list[float]]]:
    if isinstance(totals, (list, tuple)) and totals and not isinstance(
        totals[0], (list, tuple)
    ):
        totals = [totals]
        variances = [variances]
    if not isinstance(totals, (list, tuple)) or not isinstance(
        variances, (list, tuple)
    ):
        raise RemapError("totals and variances must be sequences")
    if len(totals) != len(variances):
        raise RemapError(
            f"got {len(totals)} total groups but {len(variances)} variance groups"
        )
    t_out: list[list[float]] = []
    v_out: list[list[float]] = []
    for g, (tg, vg) in enumerate(zip(totals, variances)):
        if not isinstance(tg, (list, tuple)) or not isinstance(vg, (list, tuple)):
            raise RemapError(f"group {g}: totals/variances must be sequences")
        if len(tg) != n:
            raise RemapError(
                f"group {g}: expected {n} totals (one per source), got {len(tg)}"
            )
        if len(vg) != n:
            raise RemapError(
                f"group {g}: expected {n} variances (one per source), "
                f"got {len(vg)}"
            )
        tr: list[float] = []
        vr: list[float] = []
        for i, (t, v) in enumerate(zip(tg, vg)):
            if not (_is_real(t) and _is_real(v)):
                raise RemapError(f"group {g}, index {i}: values must be real numbers")
            t, v = float(t), float(v)
            if not (math.isfinite(t) and math.isfinite(v)):
                raise RemapError(f"group {g}, index {i}: values must be finite")
            if t < 0.0:
                raise RemapError(f"group {g}, index {i}: negative total {t}")
            if v < 0.0:
                raise RemapError(f"group {g}, index {i}: negative variance {v}")
            tr.append(t)
            vr.append(v)
        t_out.append(tr)
        v_out.append(vr)
    if not t_out:
        raise RemapError("at least one measurement group is required")
    return t_out, v_out


def _overlap_fractions(
    sources: list[tuple[float, float]],
    targets: list[tuple[float, float]],
) -> list[list[float]]:
    """Fraction of each source [column] covered by each target [row].

    Half-open intervals: touching boundaries contribute zero overlap, so an
    edge shared by two bins is never allocated twice.
    """
    frac = [[0.0] * len(sources) for _ in targets]
    for j, (tlo, thi) in enumerate(targets):
        for i, (slo, shi) in enumerate(sources):
            ov = min(thi, shi) - max(tlo, slo)
            if ov > 0.0:
                frac[j][i] = ov / (shi - slo)
    return frac


def _run_group(
    frac: list[list[float]],
    totals: list[float],
    variances: list[float],
) -> GroupResult:
    m = len(frac)
    n = len(totals)
    covered = [math.fsum(frac[j][i] for j in range(m)) for i in range(n)]
    uncovered_f = [1.0 - c for c in covered]

    target_totals = [
        math.fsum(frac[j][i] * totals[i] for i in range(n)) for j in range(m)
    ]
    cov = [[0.0] * m for _ in range(m)]
    for j in range(m):
        for k in range(j, m):
            val = math.fsum(
                frac[j][i] * frac[k][i] * variances[i] for i in range(n)
            )
            cov[j][k] = val
            cov[k][j] = val
    uncovered_total = math.fsum(uncovered_f[i] * totals[i] for i in range(n))
    uncovered_variance = math.fsum(
        uncovered_f[i] * uncovered_f[i] * variances[i] for i in range(n)
    )
    uncovered_cov = [
        math.fsum(uncovered_f[i] * frac[j][i] * variances[i] for i in range(n))
        for j in range(m)
    ]
    return GroupResult(
        target_totals=target_totals,
        target_cov=cov,
        uncovered_total=uncovered_total,
        uncovered_variance=uncovered_variance,
        uncovered_cov=uncovered_cov,
    )


def _diagnostic(totals, variances, gr: GroupResult) -> ConservationDiagnostic:
    in_total = math.fsum(totals)
    in_var = math.fsum(variances)
    m = len(gr.target_totals)
    out_total = math.fsum(gr.target_totals + [gr.uncovered_total])
    merged_var_terms = [gr.uncovered_variance]
    for j in range(m):
        merged_var_terms.append(gr.target_cov[j][j])
        for k in range(m):
            if j != k:
                merged_var_terms.append(gr.target_cov[j][k])
        merged_var_terms.append(2.0 * gr.uncovered_cov[j])
    out_var = math.fsum(merged_var_terms)
    return ConservationDiagnostic(
        input_total=in_total,
        output_total=out_total,
        total_error=out_total - in_total,
        input_variance=in_var,
        output_variance=out_var,
        variance_error=out_var - in_var,
    )


def remap(source_edges, target_edges, totals, variances) -> RebinResult:
    """Remap cumulative quantities from source bins onto target bins.

    ``source_edges`` / ``target_edges`` are iterables of ``(lo, hi)`` pairs
    describing mutually non-overlapping intervals (touching is allowed, gaps
    are allowed).  ``totals`` and ``variances`` are either one group (1D
    sequences with one entry per source interval) or several groups sharing
    the same edges (2D sequences, groups first).  Target order is preserved
    exactly as provided.

    Raises :class:`RemapError` on invalid input; caller data is never mutated.
    """
    sources = _copy_edges(source_edges, "source_edges")
    targets = _copy_edges(target_edges, "target_edges")
    _check_nonoverlapping(sources, "source_edges")
    _check_nonoverlapping(targets, "target_edges")
    g_totals, g_vars = _copy_groups(totals, variances, len(sources))

    frac = _overlap_fractions(sources, targets)
    groups = [
        _run_group(frac, g_totals[g], g_vars[g]) for g in range(len(g_totals))
    ]
    diagnostics = [
        _diagnostic(g_totals[g], g_vars[g], groups[g]) for g in range(len(g_totals))
    ]
    return RebinResult(
        source_edges=sources,
        target_edges=targets,
        groups=groups,
        diagnostics=diagnostics,
    )
