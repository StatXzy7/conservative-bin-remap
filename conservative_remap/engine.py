"""Conservative rebinning engine.

Remap one-dimensional cumulative quantities defined on disjoint source
intervals (assumed internally uniform) onto a second set of disjoint target
intervals, tracking the *full* covariance matrix of the redistribution.

Pure Python standard library only (no NumPy).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


class RebinError(ValueError):
    """Raised for any invalid input. Inputs are never modified on failure."""


@dataclass(frozen=True)
class ConservationDiagnostics:
    """Conservation diagnostics for one measurement group."""

    total_in: float
    total_out: float
    variance_in: float
    variance_out: float
    total_abs_error: float
    variance_abs_error: float
    rel_total_error: float
    rel_variance_error: float


@dataclass(frozen=True)
class GroupResult:
    """Result for one measurement group.

    ``covariance`` is the m x m covariance of the target bins in caller order.
    ``uncovered_covariance`` is the u x u covariance of the uncovered
    sub-intervals; cross-covariance between targets and uncovered parts is in
    ``cross_covariance`` (rows: uncovered parts, columns: target bins).
    """

    totals: List[float]
    variances: List[float]
    uncovered_totals: List[float]
    uncovered_variances: List[float]
    covariance: List[List[float]]
    uncovered_covariance: List[List[float]]
    cross_covariance: List[List[float]]
    uncovered_intervals: List[Tuple[float, float]]
    diagnostics: ConservationDiagnostics


def _as_float_list(values, name: str, expected_len: Optional[int] = None) -> List[float]:
    if isinstance(values, (str, bytes)):
        raise RebinError(f"{name} must be a sequence of numbers")
    try:
        out = [float(v) for v in values]
    except TypeError:
        raise RebinError(f"{name} must be a sequence of numbers")
    except (ValueError, OverflowError):
        raise RebinError(f"{name} contains non-numeric or non-finite values")
    if any(not math.isfinite(v) for v in out):
        raise RebinError(f"{name} contains non-finite values")
    if expected_len is not None and len(out) != expected_len:
        raise RebinError(
            f"{name} length {len(out)} does not match interval count {expected_len}"
        )
    return out


def _validate_intervals(intervals, name: str) -> List[Tuple[float, float]]:
    if isinstance(intervals, (str, bytes)):
        raise RebinError(f"{name} must be a sequence of (lo, hi) pairs")
    try:
        n = len(intervals)
    except TypeError:
        raise RebinError(f"{name} must be a sequence of (lo, hi) pairs")
    result: List[Tuple[float, float]] = []
    for i, item in enumerate(intervals):
        if isinstance(item, (str, bytes)):
            raise RebinError(f"{name}[{i}] must be a (lo, hi) pair")
        try:
            if len(item) != 2:
                raise RebinError(f"{name}[{i}] must be a (lo, hi) pair")
            lo, hi = float(item[0]), float(item[1])
        except TypeError:
            raise RebinError(f"{name}[{i}] must be a (lo, hi) pair")
        except (ValueError, OverflowError):
            raise RebinError(f"{name}[{i}] contains non-numeric or non-finite values")
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise RebinError(f"{name}[{i}] contains non-finite values")
        if hi <= lo:
            raise RebinError(
                f"{name}[{i}] = ({lo!r}, {hi!r}) must have strictly positive width"
            )
        result.append((lo, hi))
    for i in range(n):
        for j in range(i + 1, n):
            a_lo, a_hi = result[i]
            b_lo, b_hi = result[j]
            if a_lo < b_hi and b_lo < a_hi:
                raise RebinError(
                    f"{name}[{i}] and {name}[{j}] overlap: "
                    f"({a_lo!r}, {a_hi!r}) vs ({b_lo!r}, {b_hi!r})"
                )
    return result


def _locate(sorted_intervals: List[Tuple[float, float, int]], x: float) -> int:
    """Index in sorted_intervals containing x (half-open), else -1."""
    pos = bisect_right(sorted_intervals, (x, math.inf, math.inf)) - 1
    if pos >= 0:
        lo, hi, _ = sorted_intervals[pos]
        if lo <= x < hi:
            return pos
    return -1


def _build_pieces(
    sources: List[Tuple[float, float]],
    targets: List[Tuple[float, float]],
) -> List[Tuple[float, float, int, int]]:
    """Return elementary (lo, hi, source_orig_idx, target_orig_idx) pieces.

    Boundaries of both interval sets are merged; touching boundaries do not
    create overlap (half-open convention [lo, hi)). Index is -1 outside.
    """
    src_sorted = sorted(
        ((lo, hi, i) for i, (lo, hi) in enumerate(sources)), key=lambda t: t[0]
    )
    tgt_sorted = sorted(
        ((lo, hi, i) for i, (lo, hi) in enumerate(targets)), key=lambda t: t[0]
    )
    cuts = set()
    for lo, hi in sources:
        cuts.add(lo)
        cuts.add(hi)
    for lo, hi in targets:
        cuts.add(lo)
        cuts.add(hi)
    grid = sorted(cuts)

    pieces: List[Tuple[float, float, int, int]] = []
    for k in range(len(grid) - 1):
        lo, hi = grid[k], grid[k + 1]
        if hi <= lo:
            continue
        mid = (lo + hi) * 0.5
        sp = _locate(src_sorted, mid)
        tp = _locate(tgt_sorted, mid)
        sidx = src_sorted[sp][2] if sp >= 0 else -1
        tidx = tgt_sorted[tp][2] if tp >= 0 else -1
        if pieces and pieces[-1][2] == sidx and pieces[-1][3] == tidx:
            p_lo, _, _, _ = pieces[-1]
            pieces[-1] = (p_lo, hi, sidx, tidx)
        else:
            pieces.append((lo, hi, sidx, tidx))
    return pieces


def remap_group(
    source_intervals: Sequence[Sequence[float]],
    target_intervals: Sequence[Sequence[float]],
    totals: Sequence[float],
    variances: Sequence[float],
) -> GroupResult:
    """Remap one measurement group from source to target intervals.

    Uniform within-bin density is assumed: a fraction of a source's total
    equal to the fraction of its width contributes to a target, and the same
    fraction of its independent measurement variance. Covariances arise when
    one source interval is split across several outputs (the shared random
    total induces correlation among the pieces).
    """
    sources = _validate_intervals(list(source_intervals), "source_intervals")
    targets = _validate_intervals(list(target_intervals), "target_intervals")
    n = len(sources)
    m = len(targets)
    if n == 0:
        raise RebinError("source_intervals must contain at least one interval")
    if m == 0:
        raise RebinError("target_intervals must contain at least one interval")
    total_in = _as_float_list(list(totals), "totals", n)
    var_in = _as_float_list(list(variances), "variances", n)
    for i, v in enumerate(total_in):
        if v < 0.0:
            raise RebinError(f"totals[{i}] = {v!r} is negative")
    for i, v in enumerate(var_in):
        if v < 0.0:
            raise RebinError(f"variances[{i}] = {v!r} is negative")

    pieces = [p for p in _build_pieces(sources, targets) if p[2] >= 0]

    # Group uncovered source pieces into maximal touching uncovered intervals.
    # A target-covered gap splits the uncovered region into separate parts.
    uncovered_edges: List[Tuple[float, float]] = []
    for lo, hi, sidx, tidx in pieces:
        if tidx == -1:
            if uncovered_edges and uncovered_edges[-1][1] == lo:
                uncovered_edges[-1] = (uncovered_edges[-1][0], hi)
            else:
                uncovered_edges.append((lo, hi))
    u = len(uncovered_edges)

    # Map every kept piece to an output element index:
    # [0, m) are targets in caller order; [m, m+u) are uncovered parts.
    elem_of_piece: List[int] = []
    for lo, hi, sidx, tidx in pieces:
        if tidx >= 0:
            elem_of_piece.append(tidx)
        else:
            found = -1
            for k, (plo, phi) in enumerate(uncovered_edges):
                if plo <= lo and hi <= phi:
                    found = m + k
                    break
            elem_of_piece.append(found)

    size = m + u
    means = [0.0] * size

    for (lo, hi, sidx, tidx), eidx in zip(pieces, elem_of_piece):
        s_lo, s_hi = sources[sidx]
        frac = (hi - lo) / (s_hi - s_lo)
        means[eidx] += total_in[sidx] * frac

    cov = [[0.0] * size for _ in range(size)]
    for sidx in range(n):
        v = var_in[sidx]
        # Collect (element, fraction) for this source in stable order.
        contribs: List[Tuple[int, float]] = []
        for (lo, hi, si, ti), eidx in zip(pieces, elem_of_piece):
            if si == sidx:
                frac = (hi - lo) / (sources[sidx][1] - sources[sidx][0])
                contribs.append((eidx, frac))
        for a, fa in contribs:
            for b, fb in contribs:
                cov[a][b] += v * fa * fb

    totals_out = means[:m]
    uncovered_totals_out = means[m:]
    target_cov = [row[:m] for row in cov[:m]]
    uncovered_cov = [row[m:] for row in cov[m:]]
    cross_cov = [row[:m] for row in cov[m:]]
    target_var = [target_cov[j][j] for j in range(m)]
    uncovered_var = [uncovered_cov[k][k] for k in range(u)]

    total_in_sum = math.fsum(total_in)
    total_out_sum = math.fsum(means)
    var_in_sum = math.fsum(var_in)
    # Allocating every source fully across target + uncovered pieces means the
    # all-ones quadratic form of cov equals the sum of input variances.
    var_out_sum = math.fsum(cov[i][j] for i in range(size) for j in range(size))

    def _rel(err: float, ref: float) -> float:
        if ref == 0.0:
            return 0.0 if err == 0.0 else math.inf
        return abs(err) / abs(ref)

    t_err = total_out_sum - total_in_sum
    v_err = var_out_sum - var_in_sum
    diag = ConservationDiagnostics(
        total_in=total_in_sum,
        total_out=total_out_sum,
        variance_in=var_in_sum,
        variance_out=var_out_sum,
        total_abs_error=abs(t_err),
        variance_abs_error=abs(v_err),
        rel_total_error=_rel(t_err, total_in_sum),
        rel_variance_error=_rel(v_err, var_in_sum),
    )

    return GroupResult(
        totals=totals_out,
        variances=target_var,
        uncovered_totals=uncovered_totals_out,
        uncovered_variances=uncovered_var,
        covariance=target_cov,
        uncovered_covariance=uncovered_cov,
        cross_covariance=cross_cov,
        uncovered_intervals=list(uncovered_edges),
        diagnostics=diag,
    )


def remap(
    source_intervals: Sequence[Sequence[float]],
    target_intervals: Sequence[Sequence[float]],
    totals,
    variances,
) -> List[GroupResult]:
    """Remap one or several measurement groups sharing the same boundaries.

    ``totals`` / ``variances`` may each be a flat 1-D sequence (one group) or a
    2-D sequence with one row per group. Target order is preserved exactly as
    supplied by the caller. Returns one :class:`GroupResult` per group.
    """
    if isinstance(totals, (str, bytes)) or isinstance(variances, (str, bytes)):
        raise RebinError("totals/variances must be sequences of numbers")
    totals_list = list(totals)
    variances_list = list(variances)

    def _is_numeric_row(row) -> bool:
        if isinstance(row, (str, bytes)):
            return False
        try:
            [float(v) for v in row]
            return True
        except (TypeError, ValueError, OverflowError):
            return False

    groups: List[Tuple[list, list]] = []
    if totals_list and all(
        not isinstance(x, (int, float)) and _is_numeric_row(x) for x in totals_list
    ):
        if len(totals_list) != len(variances_list):
            raise RebinError(
                f"totals has {len(totals_list)} groups but variances has "
                f"{len(variances_list)}"
            )
        for g, (trow, vrow) in enumerate(zip(totals_list, variances_list)):
            t = list(trow)
            v = list(vrow)
            if len(t) != len(v):
                raise RebinError(
                    f"group {g}: totals length {len(t)} != variances length {len(v)}"
                )
            groups.append((t, v))
    else:
        if len(totals_list) != len(variances_list):
            raise RebinError(
                f"totals length {len(totals_list)} != variances length "
                f"{len(variances_list)}"
            )
        groups.append((totals_list, variances_list))

    results: List[GroupResult] = []
    for t, v in groups:
        results.append(remap_group(source_intervals, target_intervals, t, v))
    return results
