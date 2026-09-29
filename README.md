# conservative-bin-remap

A zero-dependency-service Python engine for conservatively remapping
one-dimensional integrated measurements ("bin contents") between different
interval grids.  No web UI, no network, pure Python + NumPy.

Different instruments report a total quantity per non-overlapping source
interval, each with an independent measurement variance, assuming uniform
density inside a source interval.  The engine allocates those quantities
onto an arbitrary set of non-overlapping target intervals (gaps and partial
coverage are allowed) and reports:

* the mass (integrated total) of every target interval, in caller order;
* the **full** target-to-target covariance matrix (diagonal = variances);
* the mass and variance of the aggregated **uncovered** source remainder;
* the covariance of that remainder with every target interval;
* per-group mass and variance conservation diagnostics.

## Installation / requirements

* Python 3.10+
* NumPy (the only third-party dependency)

```powershell
pip install -r requirements.txt
```

## Demo and tests (Windows, PowerShell)

```powershell
python -m rebinning.demo
python -m unittest discover -s tests -v
```

The demo prints two small examples together with their analytically
hand-computed expected values.

## Callable interface

```python
from rebinning import rebin, RebinError

result = rebin(
    source_intervals=[(0.0, 4.0), (4.0, 7.0)],   # (start, end), any order
    mass=[8.0, 6.0],          # one value per source; 0.0 is valid
    variance=[10.0, 12.0],    # independent variances; 0.0 is valid
    target_intervals=[(0.0, 2.0), (3.0, 5.5)],   # caller order is kept
)

result.target_mass             # shape (G, T), G measurement groups
result.target_cov              # shape (G, T, T), full covariance
result.target_variance         # convenience diagonal accessor
result.uncovered_mass          # shape (G,)
result.uncovered_variance      # shape (G,)
result.uncovered_cov           # shape (G, T)
result.total_mass              # targets + remainder merged
result.total_variance
result.mass_conservation_error     # {"absolute": ..., "relative": ...}
result.variance_conservation_error
```

Several measurement groups may share the same boundaries by passing 2-D
`mass`/`variance` arrays of shape `(groups, sources)`; 1-D input is treated
as a single group.  Inputs are copied on entry and never mutated, including
when validation fails.

Intervals are half-open `[start, end)`; exactly touching boundaries are
allowed and never double-allocated.  Negative mass/variance, non-finite
numbers, zero-width intervals, overlapping intervals and dimension
mismatches raise `rebinning.RebinError`.

## Statistical model and conservation

A source interval with mass `M`, variance `V`, split into area fractions
`f_k` (area fraction = width fraction because density is uniform), follows
a shared-total (conditional multinomial) allocation:

```
Var(Y_k)      = f_k^2 V + M f_k (1 - f_k)
Cov(Y_k, Y_l) = f_k f_l (V - M)        (k != l)
```

or, per source, `C = (V - M) f fᵀ + diag(M f)`.  Contributions from
different (independent) sources add.  The uncovered source range is one
extra piece with fraction `1 − Σf_k`.

Consequences:

* pieces split from the **same** source stay correlated (non-zero
  covariance whenever `V ≠ M`; positive for over-dispersed measurements,
  negative for an unusually precisely known total);
* `V = M` (Poisson-like) pieces are uncorrelated;
* the rows of the multinomial covariance sum to zero, so merging every
  target piece and the uncovered remainder reproduces the original total
  variance; mass is likewise conserved.

## Precision conventions

* All arithmetic is IEEE-754 `float64` via NumPy.
* Overlap fractions are computed from half-open interval lengths.  A
  fraction is snapped to exactly `0`/`1` when it only deviates within
  `1e-12` relative; larger geometric inconsistency (which can only arise
  from overlapping targets) is rejected.
* Conservation residuals are reported per group as both absolute and
  relative errors (`relative = |Δ| / max(|input total|, 1)`).  With
  ordinary inputs they stay at accumulated round-off, ~1e-15 relative.
* The engine performs no Monte Carlo or iterative computation, so results
  are bit-reproducible for identical inputs.

## Why rebinning cannot be undone

Rebinning is a projection that throws away information, not an invertible
transform:

1. **Uniform-density assumption.** Everything inside a source bin is
   treated as flat.  Splitting a bin cannot reconstruct peaks or gradients
   that existed inside it; different true densities that integrate to the
   same `M` produce identical output.
2. **Boundary information is lost.** After mapping onto the target grid,
   the original boundary positions cannot be distinguished from any other
   source grid consistent with the target outputs.
3. **Covariance is required, not optional.** Forward rebinning creates
   correlations between pieces of the same source.  If a later step treats
   the target bins as independent (diagonal variances only), merging does
   **not** restore the input variance.  Only merging with the full
   covariance matrix (plus the uncovered remainder) reproduces the input
   moments.

Conservation means the integrated mass and the joint second moments are
preserved — not that the original distribution, layout or density can be
recovered.

## Project layout

```
rebinning/
  engine.py     core validation + covariance arithmetic
  demo.py       hand-computed command-line examples
tests/          unittest suite (run with the command above)
```
