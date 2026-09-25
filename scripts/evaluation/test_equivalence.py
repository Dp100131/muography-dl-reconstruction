#!/usr/bin/env python3
"""Stage 19 equivalence tests — analytic root, boundary rules, PAVA.

Run:  python scripts/evaluation/test_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import equivalence as E  # noqa: E402


def _logistic(fraction, midpoint=-1.1, slope=3.2, floor=0.10, ceiling=0.96):
    """A smooth, strictly increasing curve in log10 f — the synthetic truth."""
    x = np.log10(np.asarray(fraction, dtype=np.float64))
    return floor + (ceiling - floor) / (1.0 + np.exp(-slope * (x - midpoint)))


def test_analytic_root() -> bool:
    """A curve sampled from a known function must invert to the exact root."""
    fractions = np.array(E.DENSE_FRACTIONS)
    cells = np.tile(_logistic(fractions), (18, 1))
    curve = E.fit_raw_curve(fractions, cells)

    ok = True
    errors = []
    for target_f in (0.03, 0.08, 0.25, 0.60):
        target = float(_logistic(target_f))
        result = E.solve_factor(curve, target, 0.05)
        # PCHIP through 9 nodes is not the logistic itself: compare the solved
        # fraction against the root of the FITTED curve, which must be exact.
        residual = abs(float(curve(result["f_prime"])) - target)
        errors.append(residual)
        if result["status"] != E.FACTOR or residual > 1e-6:
            ok = False
            print(f"    root at f'={target_f}: status={result['status']} "
                  f"residual={residual:.2e} -> FAIL")
    # the interpolation itself must track the generating function closely
    dense = np.linspace(0.011, 0.89, 200)
    interp_error = float(np.max(np.abs(curve(dense) - _logistic(dense))))
    if interp_error > 5e-3:
        ok = False
        print(f"    PCHIP over log10 f departs from the logistic by "
              f"{interp_error:.2e} -> FAIL")

    result = E.solve_factor(curve, float(_logistic(0.25)), 0.05)
    checks = {
        "factor is f'/f": abs(result["factor"]
                              - result["f_prime"] / 0.05) < 1e-12,
        "days are 30 f'": abs(result["equivalent_days"]
                              - 30.0 * result["f_prime"]) < 1e-12,
        "structural cap reported": result["structural_cap"] == 20.0,
        "capped factor never exceeds 1/f":
            result["capped_factor"] <= result["structural_cap"] + 1e-12,
    }
    for label, passed in checks.items():
        if not passed:
            ok = False
            print(f"    {label}: FAIL")
    print(f"  analytic root: max|curve(f') - target| = {max(errors):.2e}, "
          f"PCHIP vs logistic {interp_error:.2e} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_boundaries() -> bool:
    """Both boundary rules of section 6 step 8 must fire on built cases."""
    fractions = np.array(E.DENSE_FRACTIONS)
    curve = E.fit_raw_curve(fractions, np.tile(_logistic(fractions), (6, 1)))
    above = E.solve_factor(curve, 0.999, 0.01)
    below = E.solve_factor(curve, 0.0, 0.01)
    nan = E.solve_factor(curve, float("nan"), 0.01)
    checks = {
        "above the curve gives a lower bound": above["status"] == E.LOWER_BOUND,
        "the lower bound is 0.90/f": abs(above["factor"] - 90.0) < 1e-9,
        "the lower bound is capped at 1/f":
            abs(above["capped_factor"] - 90.0) < 1e-9,
        "the lower bound note is explicit": "lower bound" in above["note"],
        "below the curve gives no gain": below["status"] == E.NO_GAIN,
        "the no-gain factor is 0.01/f": abs(below["factor"] - 1.0) < 1e-9,
        "the no-gain note is explicit": "no measurable gain" in below["note"],
        "a non-finite score is no gain": nan["status"] == E.NO_GAIN,
        "the cap table matches 1/f": all(
            abs(E.structural_cap(level) - 1.0 / level) < 1e-9
            for level in (0.5, 0.1, 0.05, 0.01)),
        "an off-table level still gets 1/f":
            abs(E.structural_cap(0.02) - 50.0) < 1e-9,
    }
    # a factor exceeding the structural cap must be reported capped
    flat = E.fit_raw_curve(fractions, np.tile(np.linspace(0.1, 0.99, 9), (6, 1)))
    capped = E.solve_factor(flat, 0.98, 0.5)
    checks["a factor above 1/f is capped"] = (
        capped["capped_factor"] <= 2.0 + 1e-12 and capped["factor"] > 1.0)

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  boundaries: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_pava() -> bool:
    """PAVA must fix a constructed violation, count it and leave others alone."""
    clean = np.array([0.1, 0.2, 0.3, 0.4])
    violating = np.array([0.1, 0.35, 0.25, 0.4])
    repaired = E.pava(violating)
    checks = {
        "a monotone sequence is untouched":
            bool(np.array_equal(E.pava(clean), clean)),
        "a violation is repaired": E.is_monotone(repaired, 1e-12),
        "the pooled block is the mean of the violators":
            abs(repaired[1] - 0.30) < 1e-12 and abs(repaired[2] - 0.30) < 1e-12,
        "the total is preserved":
            abs(repaired.sum() - violating.sum()) < 1e-12,
        "a fully decreasing sequence collapses to its mean":
            bool(np.allclose(E.pava(np.array([4.0, 3.0, 2.0, 1.0])), 2.5)),
        "a single point is returned as is": E.pava(np.array([7.0])).size == 1,
    }

    fractions = np.array(E.DENSE_FRACTIONS)
    base = _logistic(fractions)
    cells = np.tile(base, (18, 1))
    cells[3, 4], cells[3, 5] = base[5], base[4]        # swap two nodes
    cells[11, 2] = base[3] + 0.05                      # a bump above its right
    curve = E.fit_raw_curve(fractions, cells,
                            [f"g{i // 6}v{i % 6}" for i in range(18)])
    summary = curve.summary()
    checks["exactly the two damaged cells are adjusted"] = (
        summary["n_cells_adjusted_by_pava"] == 2)
    checks["the adjusted cells are named"] = (
        set(summary["adjusted_cells"]) == {"g0v3", "g1v5"})
    checks["the mean curve is monotone afterwards"] = summary["monotone_after_pava"]
    checks["fractions are sorted in the summary"] = (
        summary["fractions"] == sorted(summary["fractions"]))

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  PAVA: {summary['n_cells_adjusted_by_pava']} of 18 cells adjusted "
          f"{summary['adjusted_cells']} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_bootstrap_and_cross_check() -> bool:
    fractions = np.array(E.DENSE_FRACTIONS)
    rng = np.random.default_rng(20260805)
    labels = [f"g{g}v{v}" for g in range(3) for v in range(6)]
    ssim_by_cell = {label: _logistic(fractions) + 0.004 * rng.standard_normal(9)
                    for label in labels}
    enhanced = {label: np.full(20, float(_logistic(0.25))) for label in labels}

    result = E.bootstrap_factor(fractions, ssim_by_cell, enhanced, 0.05,
                               b=300, rng=np.random.default_rng(1))
    boot = result["bootstrap"]
    checks = {
        "the point estimate is a factor": result["status"] == E.FACTOR,
        "the interval brackets the point estimate":
            boot["lo"] <= result["factor"] <= boot["hi"],
        "the interval has width": boot["hi"] > boot["lo"],
        "the factor is near the truth 0.25/0.05 = 5":
            3.5 < result["factor"] < 7.0,
        "boundary replicates are counted, not dropped":
            sum(boot["status_counts"].values()) == boot["b"],
        "the structural cap travels with the result":
            result["structural_cap"] == 20.0,
        "CNR cross-check squares the ratio":
            abs(E.cnr_cross_check(3.0, 1.0) - 9.0) < 1e-12,
        "CNR cross-check guards a zero denominator":
            not np.isfinite(E.cnr_cross_check(3.0, 0.0)),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  bootstrap: N = {result['factor']:.2f} "
          f"[{boot['lo']:.2f}, {boot['hi']:.2f}] "
          f"({boot['b']} replicates, {boot['boundary_fraction']:.1%} at a "
          f"boundary) -> {'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("Stage 19 equivalence tests")
    results = {
        "analytic_root": test_analytic_root(),
        "boundaries": test_boundaries(),
        "pava": test_pava(),
        "bootstrap": test_bootstrap_and_cross_check(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 19 EQUIVALENCE:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
