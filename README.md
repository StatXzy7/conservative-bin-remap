# conservative-bin-remap

A Python engine that conservatively remaps one-dimensional cumulative
quantities measured on one set of disjoint intervals (bins from one
instrument, say) onto a different set of disjoint intervals (a common bin
grid). Within every source interval the density is assumed uniform. The full
covariance of the redistribution is tracked, including the covariance between
covered outputs and the explicitly reported **uncovered** remainder.

Zero third-party dependencies — Python 3.11 standard library only.

## Model

For source interval `i` with width `W_i`, total `T_i` and independent
measurement variance `V_i`, an output element (target bin, or uncovered part)
overlapping it over length `w` receives the fraction `f = w / W_i` of that
source:

```
mean_e      += T_i * f
Cov[e1,e2] += V_i * f1 * f2
```

Each source interval is one independent measurement, so when it is split
across several output elements those pieces are correlated
(`Cov = V_i f_a f_b ≠ 0`); pieces drawn from different sources stay
independent. Intervals are half-open `[lo, hi)`: boundaries that merely touch
never allocate the same width twice.

Uncovered source domain is partitioned into maximal touching runs: a stretch
covered by a target, or a source-free gap, separates two uncovered parts.

## Interface

```python
from conservative_remap import remap, remap_group, RebinError

# Single group
result = remap_group(source_intervals, target_intervals, totals, variances)

# One or several groups sharing the same boundaries (flat list or list of rows)
results = remap(source_intervals, target_intervals, totals_2d, variances_2d)
```

`GroupResult` fields:

- `totals`, `variances` — per target bin, in the exact caller target order.
- `covariance` — full `m × m` covariance matrix of the target bins.
- `uncovered_totals`, `uncovered_variances`, `uncovered_intervals` —
  uncovered source parts in ascending position.
- `uncovered_covariance` — `u × u` covariance among uncovered parts.
- `cross_covariance` — `u × m` covariance between uncovered parts (rows) and
  target bins (columns).
- `diagnostics` — per-group conservation errors (see below).

Target order is never sorted; shuffled source intervals are handled by
sorting internal copies only. Inputs are never modified (inputs are copied
into new lists before validation; all results are new objects).

### Conservation diagnostics

Per group:

- `total_in / total_out`, `variance_in / variance_out`
- `total_abs_error`, `variance_abs_error` and relative versions.

"Out" includes targets **and** uncovered parts. The merged scalar quantity
`1ᵀ Σ 1` over every output element equals `sum(V_i)`, i.e. re-merging all
outputs and uncovered parts restores both the input total and the input total
variance exactly (up to floating-point round-off).

## Validation

`RebinError` is raised for: negative totals or variances, non-finite numbers
(`NaN/±Inf`), zero-width or reversed intervals, overlapping source or target
intervals, dimension mismatches, and empty interval lists. Zero total and
zero variance are valid.

## Precision conventions

- All arithmetic is IEEE-754 `float64` (CPython `float`), half-open
  `[lo, hi)` intervals; overlap fractions use direct width division.
- Conservation sums use `math.fsum`; typical round-off is at the level of a
  few ULPs (~`1e-15` relative). Diagnostics report the observed errors rather
  than promising exact zero.
- Inputs that are only equal to machine tolerance (not bit-identical
  boundaries) are treated as ordinary overlaps.

## Why remapping and restoring cannot recover the original distribution

The model only knows each source bin's **integrated** total; uniform within-
bin density is an assumption, not information. Rebinning replaces every bin
by a piecewise-constant density and re-integrates. Re-merging bins restores
all first and second moments of the aggregates (total and total variance are
conserved by construction), but the internal density structure that existed
inside a bin was never represented and cannot be reconstructed: repeated
remaps smear sharp internal features, align differently, and converge toward
the coarsest piecewise-constant approximation the grids allow. Conservation
of integrated moments does not imply conservation of the pointwise density.

## Demo and tests

```bash
python3 demo.py                                # printed hand-computed demo + assertions
python3 -m unittest discover -s tests -v       # 27 unittest cases
```

Tests compare against an independent brute-force reference
(`tests/_util.py`) over randomized interval layouts, hand-calculated values,
and verify non-zero split covariances, merge conservation, shuffled inputs,
and rejection of every invalid-input class.
