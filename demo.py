#!/usr/bin/env python3
"""Command-line demo for the conservative interval rebinning engine.

Runs a hand-computed small example end to end, prints inputs / operations /
results, and asserts every number against independently derived values.
Exits non-zero on any failed assertion.
"""

import math
import sys

from conservative_remap import remap

failures = []


def check(name, got, expected, tol=1e-12):
    ok = abs(got - expected) <= tol * max(1.0, abs(expected))
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: got {got!r}, expected {expected!r}")
    if not ok:
        failures.append(name)


def check_mat(name, got, expected, tol=1e-12):
    flat_g = [x for row in got for x in row]
    flat_e = [x for row in expected for x in row]
    if len(flat_g) != len(flat_e):
        print(f"  [FAIL] {name}: shape mismatch {len(flat_g)} vs {len(flat_e)}")
        failures.append(name)
        return
    for k, (g, e) in enumerate(zip(flat_g, flat_e)):
        if abs(g - e) > tol * max(1.0, abs(e)):
            print(f"  [FAIL] {name}[{k}]: got {g!r}, expected {e!r}")
            failures.append(name)
            return
    print(f"  [PASS] {name}: {len(flat_g)} entries within tol={tol:g}")


def print_matrix(label, mat):
    print(f"    {label}:")
    for row in mat:
        print("      " + "  ".join(f"{v:+.6f}" for v in row))


print("=" * 70)
print("Conservative interval rebinning demo")
print("=" * 70)

# --- Scenario 1: hand-computed example ------------------------------------
print("\nScenario 1: partial coverage + one source split across two targets")
src = [(0.0, 10.0), (10.0, 20.0)]
tgt = [(2.0, 6.0), (6.0, 16.0)]
totals = [10.0, 20.0]
variances = [3.0, 8.0]
print(f"  sources (disjoint, uniform density inside): {src}")
print(f"  targets (caller order preserved):          {tgt}")
print(f"  source totals:     {totals}")
print(f"  source variances:  {variances}")
print("  operation: split sources at every target boundary, assign mass and")
print("  independent variance by width fraction; cov = var_i * f_a * f_b.")

r = remap(src, tgt, totals, variances)[0]
print(f"  target totals:              {r.totals}")
print(f"  target variances:           {r.variances}")
print(f"  uncovered totals:           {r.uncovered_totals} "
      f"on {r.uncovered_intervals}")
print(f"  uncovered variances:        {r.uncovered_variances}")
print_matrix("target x target covariance", r.covariance)
print_matrix("uncovered x uncovered covariance", r.uncovered_covariance)
print_matrix("uncovered x target cross-covariance", r.cross_covariance)

# Hand calculation:
#  target 0 [2,6):   4/10*10          = 4.0
#  target 1 [6,16):  4/10*10+6/10*20  = 16.0
#  uncovered [0,2):  2.0 ; [16,20):   8.0
check("target total 0", r.totals[0], 4.0)
check("target total 1", r.totals[1], 16.0)
check("uncovered total L", r.uncovered_totals[0], 2.0)
check("uncovered total R", r.uncovered_totals[1], 8.0)
# variances: diag sums per source
check("target var 0", r.variances[0], 3.0 * 0.4 * 0.4)
check("target var 1", r.variances[1],
      3.0 * 0.4 * 0.4 + 8.0 * 0.6 * 0.6)
check("cov(t0,t1)", r.covariance[0][1], 3.0 * 0.4 * 0.4)
check("cov(uL,t0)", r.cross_covariance[0][0], 3.0 * 0.2 * 0.4)
check("cov(uL,t1)", r.cross_covariance[0][1], 3.0 * 0.2 * 0.4)
check("cov(uR,t0)", r.cross_covariance[1][0], 0.0)
check("cov(uR,t1)", r.cross_covariance[1][1], 8.0 * 0.4 * 0.6)
check("cov(uL,uR)", r.uncovered_covariance[0][1], 0.0)
check("total conservation", r.diagnostics.total_out - r.diagnostics.total_in, 0.0)
check("variance conservation", r.diagnostics.variance_out - r.diagnostics.variance_in, 0.0)

# Re-merge outputs + uncovered: full covariance sum must equal input variance.
m = len(r.totals)
merged_var = sum(sum(row) for row in r.covariance)
merged_var += sum(sum(row) for row in r.uncovered_covariance)
merged_var += 2.0 * sum(sum(row) for row in r.cross_covariance)
check("merged total variance == input", merged_var, sum(variances))
check("merged total == input",
      math.fsum(r.totals) + math.fsum(r.uncovered_totals),
      math.fsum(totals))

# --- Scenario 2: shuffled input order + two measurement groups -------------
print("\nScenario 2: shuffled boundary order, two measurement groups")
src_shuffled = [(10.0, 20.0), (0.0, 10.0)]
tgt_shuffled = [(6.0, 16.0), (2.0, 6.0)]
totals2 = [[10.0, 20.0], [0.0, 5.0]]
variances2 = [[3.0, 8.0], [0.0, 0.0]]
print(f"  sources (shuffled): {src_shuffled}")
print(f"  targets (shuffled): {tgt_shuffled}")
print(f"  totals per group:    {totals2}")
print(f"  variances per group: {variances2}")

results = remap(src_shuffled, tgt_shuffled, totals2, variances2)
# Target order preserved: row 0 is [6,16), row 1 is [2,6).
r0, r1 = results
check("group0 target order [6,16) total", r0.totals[0], 14.0)
check("group0 target order [2,6) total", r0.totals[1], 8.0)
# group1: source (10,20)=0, source (0,10)=5 -> [2,6)=2, [6,10)=2, [0,2)=1
check("group1 target total", math.fsum(r1.totals), 4.0)
check("group1 uncovered total", math.fsum(r1.uncovered_totals), 1.0)
check("group1 all-zero variances",
      sum(x for row in r1.covariance for x in row), 0.0)
for grp, res in enumerate(results):
    d = res.diagnostics
    print(f"  group {grp} conservation: total_in={d.total_in} "
          f"total_out={d.total_out} abs_err={d.total_abs_error:.3e} "
          f"var_in={d.variance_in} var_out={d.variance_out} "
          f"abs_err={d.variance_abs_error:.3e}")

# --- Scenario 3: touching boundaries are never double counted --------------
print("\nScenario 3: exact touching boundaries")
r3 = remap([(0.0, 5.0), (5.0, 10.0)],
           [(0.0, 5.0), (5.0, 10.0)],
           [7.0, 3.0], [2.0, 4.0])[0]
check("touch: total 0", r3.totals[0], 7.0)
check("touch: total 1", r3.totals[1], 3.0)
check_mat("touch: diagonal covariance", r3.covariance, [[2.0, 0.0], [0.0, 4.0]])
check("touch: nothing uncovered", math.fsum(r3.uncovered_totals), 0.0)

# --- Scenario 4: invalid inputs are rejected --------------------------------
print("\nScenario 4: invalid inputs rejected")
from conservative_remap import RebinError

bad_cases = [
    ("negative total", [(0, 1)], [(0, 1)], [-1.0], [1.0]),
    ("negative variance", [(0, 1)], [(0, 1)], [1.0], [-1e-9]),
    ("non-finite", [(0, 1)], [(0.0, float("nan"))], [1.0], [1.0]),
    ("zero-width interval", [(0, 0)], [(0, 1)], [1.0], [1.0]),
    ("overlapping sources", [(0, 2), (1, 3)], [(0, 1)], [1, 1], [1, 1]),
    ("dimension mismatch", [(0, 1), (1, 2)], [(0, 2)], [1.0], [1.0, 1.0]),
]
for name, s, t, tot, var in bad_cases:
    try:
        remap(s, t, tot, var)
    except RebinError as exc:
        print(f"  [PASS] {name}: RebinError -> {exc}")
    else:
        print(f"  [FAIL] {name}: expected RebinError but call succeeded")
        failures.append(name)

print("\n" + "=" * 70)
if failures:
    print(f"DEMO FAILED: {len(failures)} assertion(s): {failures}")
    sys.exit(1)
print("ALL DEMO ASSERTIONS PASSED")
print("=" * 70)
