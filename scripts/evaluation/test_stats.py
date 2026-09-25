#!/usr/bin/env python3
"""Stage 19 statistics tests — coverage, Holm by hand, exact Wilcoxon.

The coverage check is the load-bearing one: 200 synthetic datasets shaped like
the real design (18 cells x 20 correlated pairs) must be covered by the chosen
interval about 95 % of the time. B is reduced to 600 replicates here so the
suite stays under a few seconds; the production runs use B = 10 000, which can
only make the interval smoother, not differently centred.

Run:  python scripts/evaluation/test_stats.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import stats as S  # noqa: E402

N_DATASETS = 200
N_CELLS = 18
K_PAIRS = 20
B_TEST = 600
TRUE_EFFECT = 0.006
TAU_BETWEEN_CELLS = 0.014      # the structural spread reported in section 5.4
SIGMA_WITHIN = 0.004


def _dataset(rng):
    """Cell effects around TRUE_EFFECT plus within-cell Poisson-like jitter."""
    cell_effects = TRUE_EFFECT + TAU_BETWEEN_CELLS * rng.standard_normal(N_CELLS)
    return [cell_effects[i] + SIGMA_WITHIN * rng.standard_normal(K_PAIRS)
            for i in range(N_CELLS)]


def test_coverage() -> bool:
    """Coverage of the pre-registered interval, MEASURED against the t yardstick.

    The nominal level is 95 %. At 18 clusters the nonparametric bootstrap
    genuinely lands near 91 % (measured over 1 000 datasets: BCa 0.913,
    percentile 0.913, Student-t 0.954) -- standard small-sample under-coverage,
    not a defect of this implementation. The acceptance band is therefore
    0.88-0.98 and the t coverage is printed alongside so the dry run records
    the gap instead of hiding it. See the small-sample caveat in stats.py.
    """
    rng = np.random.default_rng(20260805)
    covered, covered_t, methods = 0, 0, {"bca": 0, "percentile": 0}
    t0 = time.perf_counter()
    for _ in range(N_DATASETS):
        cells = _dataset(rng)
        cell_means = np.array([c.mean() for c in cells])
        theta_hat = float(cell_means.mean())
        boot = S.cluster_bootstrap(cells, b=B_TEST, rng=rng)
        interval = S.choose_ci(boot, theta_hat, S.jackknife_over_cells(cells))
        methods[interval["method"]] += 1
        if interval["lo"] <= TRUE_EFFECT <= interval["hi"]:
            covered += 1
        t_lo, t_hi = S.t_interval(cell_means)
        if t_lo <= TRUE_EFFECT <= t_hi:
            covered_t += 1
    coverage = covered / N_DATASETS
    coverage_t = covered_t / N_DATASETS
    ok = 0.88 <= coverage <= 0.98 and 0.92 <= coverage_t <= 0.98
    print(f"  coverage: bootstrap {covered}/{N_DATASETS} = {coverage:.3f} "
          f"(bca {methods['bca']}, percentile {methods['percentile']}, "
          f"B={B_TEST}) | Student-t {coverage_t:.3f} "
          f"| {time.perf_counter() - t0:.1f} s -> {'ok' if ok else 'FAIL'}")
    if coverage < 0.93:
        print("    note: pre-registered bootstrap under-covers at n=18 "
              "(expected, documented in stats.py); the t interval is reported "
              "alongside in every table")
    return ok


def test_bootstrap_mechanics() -> bool:
    """Vectorized and looped paths must agree, and the estimator must be sane."""
    rng = np.random.default_rng(11)
    cells = [np.full(20, float(i)) for i in range(18)]      # zero within-cell var
    boot = S.cluster_bootstrap(cells, b=2000, rng=np.random.default_rng(3))
    grand = float(np.mean([c.mean() for c in cells]))
    ragged = [np.full(20 - (i % 3), float(i)) for i in range(18)]
    boot_ragged = S.cluster_bootstrap(ragged, b=500, rng=np.random.default_rng(3))
    checks = {
        "bootstrap mean tracks the grand mean": abs(boot.mean() - grand) < 0.25,
        "bootstrap has spread": boot.std() > 0.5,
        "ragged input works": boot_ragged.size == 500,
        "seeded bootstrap reproduces": bool(np.array_equal(
            S.cluster_bootstrap(cells, b=200, rng=np.random.default_rng(1)),
            S.cluster_bootstrap(cells, b=200, rng=np.random.default_rng(1)))),
        "constant data gives a degenerate zero-width interval": (
            S.percentile_ci(S.cluster_bootstrap(
                [np.full(5, 2.0)] * 4, b=100, rng=rng)) == (2.0, 2.0)),
        "jackknife leaves one cell out":
            S.jackknife_over_cells(cells).size == 18,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  bootstrap mechanics: {sum(checks.values())}/{len(checks)} "
          f"checks -> {'ok' if ok else 'FAIL'}")
    return ok


def test_ci_selection_rule() -> bool:
    """The 20 % divergence rule must pick the percentile interval when it fires."""
    boot = np.linspace(-3.0, 3.0, 4001)
    jack_flat = np.full(18, 0.0)
    chosen = S.choose_ci(boot, 0.0, jack_flat)
    # a point estimate far from the bootstrap median gives a large bias
    # correction z0, which is exactly when BCa and percentile part company
    biased = S.choose_ci(boot, 1.5, jack_flat)
    # every replicate above the estimate: z0 is undefined -> percentile backup
    degenerate = S.choose_ci(boot, -4.0, jack_flat)
    checks = {
        "symmetric case keeps BCa": chosen["method"] == "bca",
        "symmetric BCa matches percentile":
            abs(chosen["lo"] - chosen["percentile"][0]) < 0.05,
        "divergence fraction reported": np.isfinite(chosen["divergence_fraction"]),
        "large bias correction switches to percentile":
            biased["method"] == "percentile"
            and biased["divergence_fraction"] > 0.20,
        "degenerate z0 falls back to percentile":
            degenerate["method"] == "percentile"
            and degenerate["bca_info"]["degenerate"],
        "excludes_zero is honest":
            (not chosen["excludes_zero"])
            and S.choose_ci(boot + 5.0, 5.0, jack_flat)["excludes_zero"],
        "t interval is available as a yardstick":
            np.isfinite(S.t_interval(np.linspace(0.0, 1.0, 18))[0]),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL "
                  f"(biased divergence={biased['divergence_fraction']:.3f}, "
                  f"method={biased['method']})")
    print(f"  CI selection rule: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_holm_by_hand() -> bool:
    """Worked example: p = [0.010, 0.040, 0.030, 0.005], m = 4."""
    result = S.holm([0.010, 0.040, 0.030, 0.005])
    expected_adjusted = [0.030, 0.060, 0.060, 0.020]
    expected_reject = [True, False, False, True]
    deltas = [abs(a - b) for a, b in zip(result["adjusted"], expected_adjusted)]
    monotone = S.holm([0.001, 0.002, 0.003])
    checks = {
        "adjusted p-values match the hand calculation": max(deltas) < 1e-12,
        "rejections match": result["reject"] == expected_reject,
        "adjusted values are monotone in sorted order": all(
            np.diff([monotone["adjusted"][i]
                     for i in monotone["sorted_order"]]) >= -1e-15),
        "adjusted p never exceeds 1":
            max(S.holm([0.4, 0.6, 0.9])["adjusted"]) <= 1.0,
        "empty family is handled": S.holm([])["m"] == 0,
        "family of 12 uses 0.05/12 at the first step": abs(
            S.holm([0.001] * 12)["per_step_alpha_sorted"][0]
            - 0.05 / 12) < 1e-15,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  Holm: adjusted={[round(v, 4) for v in result['adjusted']]} "
          f"-> {'ok' if ok else 'FAIL'}")
    return ok


def test_wilcoxon_and_gates() -> bool:
    """18 same-sign differences must hit the exact minimum p = 2/2^18."""
    diffs = np.linspace(0.001, 0.02, 18)
    result = S.wilcoxon_exact(diffs)
    minimum = 2.0 / 2.0 ** 18
    negatives = S.wilcoxon_exact(-diffs)
    mixed = S.wilcoxon_exact(np.concatenate([diffs[:9], -diffs[:9]]))
    gate = S.shapiro_gate(np.random.default_rng(4).standard_normal(18))
    non_normal = S.shapiro_gate(np.concatenate(
        [np.zeros(17), [50.0]]))
    checks = {
        "exact method selected at n = 18": result["method"] == "exact",
        "p equals 2/2^18": abs(result["p_value"] - minimum) < 1e-12,
        "minimum attainable p reported":
            abs(result["min_attainable_p"] - minimum) < 1e-15,
        "p survives Holm at 12 tests": result["p_value"] < 0.05 / 12,
        "sign symmetry": abs(negatives["p_value"] - result["p_value"]) < 1e-15,
        "mixed signs are not significant": mixed["p_value"] > 0.5,
        "sign counts reported": (result["positive"] == 18
                                 and negatives["negative"] == 18),
        "all-zero differences are handled": S.wilcoxon_exact(
            np.zeros(18))["p_value"] == 1.0,
        "Shapiro opens on normal data": gate["normal"],
        "Shapiro closes on an outlier": not non_normal["normal"],
        "no zeros are reported when there are none": result["n_zero"] == 0,
    }

    # a zero difference takes SciPy off the exact path, and which way it goes
    # depends on the installed version. The reported method has to be the
    # truth, not the intention: section 5.4 pre-registers the EXACT test, so a
    # sample that cannot have it says "normal-approx" and loses the
    # min_attainable_p that only the exact distribution provides.
    with_zeros = S.wilcoxon_exact(np.concatenate([np.linspace(0.001, 0.02, 16),
                                                  np.zeros(2)]))
    checks["a zero difference is counted"] = with_zeros["n_zero"] == 2
    checks["a sample with zeros is reported as normal-approx"] = (
        with_zeros["method"] == S.NORMAL_APPROX)
    checks["a sample with zeros claims no exact floor"] = not np.isfinite(
        with_zeros["min_attainable_p"])
    checks["the deviation is annotated"] = "exactly zero" in with_zeros["note"]
    checks["the exact path is still labelled exact"] = (
        result["method"] == S.EXACT and result["note"] == "")
    checks["n > 25 is normal-approx too"] = (
        S.wilcoxon_exact(np.linspace(0.001, 0.02, 30))["method"]
        == S.NORMAL_APPROX)
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  Wilcoxon: p = {result['p_value']:.3e} (minimum {minimum:.3e}), "
          f"Shapiro p = {gate['p_value']:.3f} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_mde() -> bool:
    """MDE must scale as sigma/sqrt(n) and tighten as alpha loosens."""
    strict = S.mde(0.004, 18, 0.05 / 12)
    loose = S.mde(0.004, 18, 0.05)
    checks = {
        "strict Holm needs a larger effect": strict > loose,
        "scales linearly in sigma": abs(S.mde(0.008, 18, 0.05 / 12)
                                        - 2.0 * strict) < 1e-12,
        "scales as 1/sqrt(n)": abs(S.mde(0.004, 72, 0.05 / 12)
                                   - 0.5 * strict) < 1e-12,
        "plan-scale magnitude": 0.001 < strict < 0.010,
        "invalid input is NaN": not np.isfinite(S.mde(0.004, 0, 0.05)),
    }
    rows = S.power_table([0.002, 0.004, 0.006])
    checks["power table has one row per sigma"] = len(rows) == 3
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  MDE: sigma_d=0.004 n=18 Holm(12) -> {strict:.5f} "
          f"(alpha=0.05 -> {loose:.5f}); table "
          f"{[round(r['mde_holm_strictest'], 5) for r in rows]} -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_cell_means_and_front_door() -> bool:
    values = [1.0, 3.0, 10.0, 20.0, 30.0]
    cells = ["a", "a", "b", "b", "b"]
    order, means = S.cell_means(values, cells)
    grouped = S.group_by_cell(values, cells)
    diff_by_cell = {f"g{i}v{j}": np.full(20, 0.01 + 0.002 * i - 0.001 * j)
                    for i in range(3) for j in range(6)}
    report = S.compare_cells(diff_by_cell, b=400,
                             rng=np.random.default_rng(20260805))
    checks = {
        "cells in first-appearance order": order == ["a", "b"],
        "means correct": bool(np.allclose(means, [2.0, 20.0])),
        "grouping preserves every value":
            sum(v.size for v in grouped.values()) == 5,
        "length mismatch raises": _raises(S.cell_means, [1.0], ["a", "b"]),
        "front door reports 18 cells": report["n_cells"] == 18,
        "front door counts 360 pairs": report["n_pairs"] == 360,
        "front door runs the exact Wilcoxon":
            report["wilcoxon"]["method"] == "exact",
        "front door interval excludes zero": report["ci"]["excludes_zero"],
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  cell means + front door: estimate={report['estimate']:.5f} "
          f"CI=[{report['ci']['lo']:.5f}, {report['ci']['hi']:.5f}] "
          f"({report['ci']['method']}) -> {'ok' if ok else 'FAIL'}")
    return ok


def _raises(fn, *args) -> bool:
    try:
        fn(*args)
    except Exception:  # noqa: BLE001
        return True
    return False


def main() -> int:
    print("Stage 19 statistics tests")
    results = {
        "coverage": test_coverage(),
        "bootstrap_mechanics": test_bootstrap_mechanics(),
        "ci_selection_rule": test_ci_selection_rule(),
        "holm": test_holm_by_hand(),
        "wilcoxon_and_gates": test_wilcoxon_and_gates(),
        "mde": test_mde(),
        "cell_means": test_cell_means_and_front_door(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 19 STATS:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
