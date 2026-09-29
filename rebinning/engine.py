"""Conservative one-dimensional interval rebinning engine.

The engine remaps integrated quantities ("masses") measured on a set of
non-overlapping source intervals onto a caller-supplied set of target
intervals, assuming uniform density inside each source interval.

Statistical model
-----------------
A source interval with mass ``M`` and variance ``V`` is split by area
fractions ``f_k`` (the source density is uniform, so the fraction is just
the fraction of source width covered).  The total mass is shared between
the pieces (a conditional multinomial allocation):

    Var(Y_k)      = f_k^2 * V + M * f_k * (1 - f_k)
    Cov(Y_k, Y_l) = f_k * f_l * (V - M)          for k != l

or, in matrix form for one source::

    C = (V - M) * f f^T + diag(M * f)

Contributions from different source intervals are independent, so their
mass and covariance contributions simply add.  The uncovered remainder of
each source interval is treated as one extra piece with fraction
``1 - sum(f_k)`` and reported in aggregate.

Properties:

* mass is conserved exactly (up to floating point rounding);
* merging all target pieces *and* the uncovered remainder reproduces the
  original total variance ``V``, because the rows of the multinomial
  covariance matrix sum to zero;
* pieces split from the same source interval keep a non-zero covariance
  whenever ``V != M`` (e.g. V > M, over-dispersed independent measurements).

Precision
---------
All arithmetic is IEEE-754 ``float64`` (NumPy default).  Geometric overlap
fractions are computed from half-open interval lengths; a fraction is
clipped into ``[0, 1]`` when it only deviates by floating-point rounding
(relative deviation <= 1e-12), otherwise the input is rejected.  Exact
mathematical conservation therefore holds up to accumulated rounding,
typically ~1e-15 relative; the per-group conservation diagnostics returned
by :func:`rebin` make the achieved residual explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RebinError", "RebinResult", "rebin"]

_FLOAT64_INFO = np.finfo(np.float64)
# A computed fraction is rounded to exactly 0.0/1.0 if it differs only by
# this relative amount; larger deviations indicate bad geometry and error.
_FRACTION_EPS = 1e-12


class RebinError(ValueError):
    """Raised when the inputs describe an invalid rebinning problem."""


@dataclass(frozen=True)
class RebinResult:
    """Result of a rebinning operation.

    Group axis is always axis 0.  For a single measurement group every
    array carries a leading singleton dimension.

    Attributes
    ----------
    target_mass:
        Mass assigned to every target interval, shape ``(G, T)``.
    target_cov:
        Full covariance matrix between target intervals, shape
        ``(G, T, T)``.  Diagonal entries are the target variances.
    uncovered_mass, uncovered_variance:
        Aggregated mass/variance of the source range not covered by any
        target interval, shape ``(G,)``.
    uncovered_cov:
        Covariance of the uncovered remainder with every target interval,
        shape ``(G, T)``.
    total_mass, total_variance:
        Mass/variance obtained after merging every target piece and the
        uncovered remainder, shape ``(G,)``.
    mass_conservation_error, variance_conservation_error:
        Absolute and relative (where well defined) residuals against the
        input totals for every group.
    """

    target_mass: np.ndarray
    target_cov: np.ndarray
    uncovered_mass: np.ndarray
    uncovered_variance: np.ndarray
    uncovered_cov: np.ndarray
    total_mass: np.ndarray
    total_variance: np.ndarray
    mass_conservation_error: dict
    variance_conservation_error: dict

    @property
    def target_variance(self) -> np.ndarray:
        """Variance of every target interval, shape ``(G, T)``."""
        return np.diagonal(self.target_cov, axis1=1, axis2=2)


def _parse_intervals(intervals, name: str) -> np.ndarray:
    """Validate interval input and return an independent float64 copy."""
    try:
        arr = np.asarray(intervals, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RebinError(f"{name} must be convertible to a numeric array") from exc
    if arr.ndim == 1 and arr.size == 0:
        arr = arr.reshape(0, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RebinError(
            f"{name} must be a sequence of (start, end) pairs "
            f"(got shape {arr.shape})"
        )
    if arr.size and not np.all(np.isfinite(arr)):
        raise RebinError(f"{name} contains non-finite values (NaN or infinity)")
    lo = arr[:, 0]
    hi = arr[:, 1]
    zero_width = hi <= lo
    if np.any(zero_width):
        idx = int(np.argmax(zero_width))
        raise RebinError(
            f"{name}[{idx}] has zero or negative width: ({lo[idx]}, {hi[idx]})"
        )
    order = np.argsort(lo, kind="mergesort")
    sorted_hi = hi[order]
    sorted_lo = lo[order]
    # Strictly positive overlap is forbidden; exact touching (hi == next lo)
    # is allowed and must not cause double allocation.
    for pos in range(len(order) - 1):
        if sorted_hi[pos] > sorted_lo[pos + 1]:
            raise RebinError(
                f"{name} intervals overlap: "
                f"({sorted_lo[pos]}, {sorted_hi[pos]}) and "
                f"({sorted_lo[pos + 1]}, {sorted_hi[pos + 1]})"
            )
    return arr.copy()


def _parse_measurements(values, n_intervals: int, name: str) -> np.ndarray:
    """Validate per-interval mass/variance input; return a 2-D copy (G, N)."""
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RebinError(f"{name} must be convertible to a numeric array") from exc
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    elif arr.ndim != 2:
        raise RebinError(f"{name} must be 1- or 2-dimensional (got {arr.ndim} dims)")
    if arr.shape[1] != n_intervals:
        raise RebinError(
            f"{name} has {arr.shape[1]} entries but there are {n_intervals} "
            f"source intervals"
        )
    if not np.all(np.isfinite(arr)):
        raise RebinError(f"{name} contains non-finite values (NaN or infinity)")
    if np.any(arr < 0.0):
        idx = np.unravel_index(int(np.argmax(arr < 0.0)), arr.shape)
        raise RebinError(f"{name}[{idx[0]}, {idx[1]}] is negative: {arr[idx]}")
    return arr.copy()


def _overlap_fractions(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Geometric area fraction of every source inside every target.

    Returns an ``(S, T)`` matrix.  Intervals are treated as half-open
    ``[start, end)`` so touching boundaries produce zero overlap.
    """
    s_lo = source[:, 0, np.newaxis]
    s_hi = source[:, 1, np.newaxis]
    t_lo = target[np.newaxis, :, 0]
    t_hi = target[np.newaxis, :, 1]
    overlap = np.minimum(s_hi, t_hi) - np.maximum(s_lo, t_lo)
    np.maximum(overlap, 0.0, out=overlap)
    width = s_hi - s_lo
    frac = overlap / width
    covered = frac.sum(axis=1, keepdims=True)
    bad = (covered > 1.0 + _FRACTION_EPS) | (frac < 0.0) | (frac > 1.0)
    if np.any(bad):
        s_idx = int(np.argmax(np.any(bad, axis=1)))
        raise RebinError(
            f"target coverage of source interval {s_idx} "
            f"({source[s_idx, 0]}, {source[s_idx, 1]}) exceeds its width; "
            f"targets must not overlap"
        )
    # Snap rounding noise around 0/1 so the conservation identities are exact.
    frac = np.where(np.abs(frac) <= _FRACTION_EPS, 0.0, frac)
    frac = np.where(np.abs(frac - 1.0) <= _FRACTION_EPS, 1.0, frac)
    return frac


def _diagnostic(actual: np.ndarray, expected: np.ndarray) -> dict:
    absolute = np.abs(actual - expected)
    scale = np.maximum(np.abs(expected), 1.0)
    return {"absolute": absolute, "relative": absolute / scale}


def rebin(source_intervals, mass, variance, target_intervals) -> RebinResult:
    """Remap source interval measurements onto target intervals.

    Parameters
    ----------
    source_intervals:
        Sequence of ``(start, end)`` pairs, ``start < end``, mutually
        non-overlapping (exact touching is allowed).  Need not be sorted.
    mass:
        Integrated quantity per source interval.  A 1-D sequence for one
        measurement group, a 2-D ``(groups, sources)`` array when several
        groups share the same boundaries.  Zero is a valid mass.
    variance:
        Independent measurement variance per source interval; same shape
        rules as *mass*.  Zero is a valid variance (a perfectly known mass).
    target_intervals:
        Sequence of ``(start, end)`` pairs for the output bins, mutually
        non-overlapping, with gaps permitted and partial source coverage
        allowed.  Caller order is preserved in the output.

    Returns
    -------
    RebinResult

    Raises
    ------
    RebinError
        For negative/non-finite values, zero-width or overlapping intervals,
        or mismatched dimensions.  Inputs are never modified: validation
        happens before any computation and inputs are copied on entry.
    """
    source = _parse_intervals(source_intervals, "source_intervals")
    target = _parse_intervals(target_intervals, "target_intervals")
    n_source = source.shape[0]
    mass_g = _parse_measurements(mass, n_source, "mass")
    var_g = _parse_measurements(variance, n_source, "variance")
    if mass_g.shape != var_g.shape:
        raise RebinError(
            f"mass shape {mass_g.shape} does not match variance shape "
            f"{var_g.shape}"
        )
    n_target = target.shape[0]

    frac = _overlap_fractions(source, target)  # (S, T)
    uncovered_frac = 1.0 - frac.sum(axis=1)  # (S,)

    # Mass per target: sum over sources of M_s * f_st.
    target_mass = mass_g @ frac  # (G, T)

    # Per-source multinomial covariance piece added to every (i, j):
    #   (V_s - M_s) * f_si * f_sj  +  delta_ij * M_s * f_si
    spread = var_g - mass_g  # (G, S)
    # (G, S, 1) * (1, S, T) broadcast then sum over sources.
    # Off-diagonal and the (V-M) f^2 part of the diagonal, per source:
    target_cov = np.einsum("gs,si,sj->gij", spread, frac, frac)
    # Diagonal-only multinomial allocation term: sum_s M_s f_si.
    diag_add = np.einsum("gs,si->gi", mass_g, frac)
    target_cov += diag_add[:, :, np.newaxis] * np.eye(n_target)

    # Uncovered remainder behaves like one extra piece with fraction f_u.
    # Cov(remainder, target i) summed over sources:
    #   f_si * f_us * (V_s - M_s)   (diagonal term only touches remainder
    #   itself, never a target).
    uncovered_cov = np.einsum(
        "gs,si,s->gi", spread, frac, uncovered_frac
    )
    # Var(remainder) = sum_s [ f_us^2 V_s + M_s f_us (1 - f_us) ].
    uncovered_variance = (
        np.einsum("gs,s->g", var_g, uncovered_frac**2)
        + np.einsum("gs,s->g", mass_g, uncovered_frac * (1.0 - uncovered_frac))
    )
    uncovered_mass = mass_g @ uncovered_frac

    # Merge check: total = sum(target) + remainder.
    total_mass = target_mass.sum(axis=1) + uncovered_mass
    total_variance = (
        target_cov.sum(axis=(1, 2))
        + 2.0 * uncovered_cov.sum(axis=1)
        + uncovered_variance
    )

    input_mass = mass_g.sum(axis=1)
    input_variance = var_g.sum(axis=1)

    return RebinResult(
        target_mass=target_mass,
        target_cov=target_cov,
        uncovered_mass=uncovered_mass,
        uncovered_variance=uncovered_variance,
        uncovered_cov=uncovered_cov,
        total_mass=total_mass,
        total_variance=total_variance,
        mass_conservation_error=_diagnostic(total_mass, input_mass),
        variance_conservation_error=_diagnostic(total_variance, input_variance),
    )

