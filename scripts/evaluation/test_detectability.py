#!/usr/bin/env python3
"""Stage 19 detectability tests — analytic fixtures, no data.

A flat background with a disk of known offset has contrast, CNR and z that can
be written down in closed form, so the implementation is checked against
arithmetic rather than against itself. The classification truth table walks
every branch of protocol v1.1 section 5.3 with hand-built statistics dicts.

Run:  python scripts/evaluation/test_detectability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import detectability as D  # noqa: E402
import roi as R  # noqa: E402

SHAPE = (96, 96)


def _disk(shape, row, col, radius):
    rr, cc = np.ogrid[0:shape[0], 0:shape[1]]
    return (rr - row) ** 2 + (cc - col) ** 2 <= radius ** 2


def _ring(shape, row, col, inner, outer):
    rr, cc = np.ogrid[0:shape[0], 0:shape[1]]
    d2 = (rr - row) ** 2 + (cc - col) ** 2
    return (d2 > inner ** 2) & (d2 <= outer ** 2)


# ---------------------------------------------------------------- exact stats
def test_exact_contrast() -> bool:
    """Noiseless fixture: C and CNR must be the arithmetic values to 1e-12."""
    roi = _disk(SHAPE, 48, 48, 8)
    annulus = _ring(SHAPE, 48, 48, 12, 20)
    base, offset = 0.40, 0.05

    img = np.full(SHAPE, base)
    img[roi] = base + offset
    # a deterministic +-sigma checkerboard in the ring: sigma is exact
    sigma = 0.02
    checker = np.indices(SHAPE).sum(axis=0) % 2
    img[annulus] = base + np.where(checker[annulus] == 0, sigma, -sigma)

    stats = D.roi_stats(img, roi, annulus)
    n_bg = int(annulus.sum())
    n_plus = int((checker[annulus] == 0).sum())
    mean_bg = base + sigma * (2 * n_plus - n_bg) / n_bg
    var_bg = (np.sum((np.where(checker[annulus] == 0, sigma, -sigma)
                      + base - mean_bg) ** 2) / (n_bg - 1))
    std_bg = float(np.sqrt(var_bg))
    n_roi = int(roi.sum())
    difference = (base + offset) - mean_bg

    expected = {
        "mean_roi": base + offset,
        "mean_bg": mean_bg,
        "std_bg": std_bg,
        "C": difference / mean_bg,
        "CNR": difference / std_bg,
        "z": difference / (std_bg * np.sqrt(1.0 / n_roi + 1.0 / n_bg)),
    }
    deltas = {key: abs(stats[key] - value) for key, value in expected.items()}
    counts_ok = stats["n_roi"] == n_roi and stats["n_bg"] == n_bg
    ok = counts_ok and max(deltas.values()) < 1e-12
    print(f"  exact contrast: C={stats['C']:.6f} CNR={stats['CNR']:.4f} "
          f"z={stats['z']:.4f} max|delta|={max(deltas.values()):.2e} "
          f"-> {'ok' if ok else 'FAIL'}")
    if not ok:
        for key, delta in deltas.items():
            if delta >= 1e-12:
                print(f"    {key}: got {stats[key]!r} want {expected[key]!r}")
    return ok


def test_sign_convention() -> bool:
    """A brighter ROI gives C > 0, a darker one C < 0; z follows the sign."""
    roi = _disk(SHAPE, 48, 48, 8)
    annulus = _ring(SHAPE, 48, 48, 12, 20)
    rng = np.random.default_rng(19)
    base = np.full(SHAPE, 0.5) + 0.01 * rng.standard_normal(SHAPE)

    bright = base.copy()
    bright[roi] += 0.10
    dark = base.copy()
    dark[roi] -= 0.10
    s_bright = D.roi_stats(bright, roi, annulus)
    s_dark = D.roi_stats(dark, roi, annulus)
    checks = {
        "cavity-like ROI is positive": s_bright["C"] > 0 and s_bright["z"] > 0,
        "dense-body ROI is negative": s_dark["C"] < 0 and s_dark["z"] < 0,
        "CNR carries the same sign as C":
            np.sign(s_bright["CNR"]) == np.sign(s_bright["C"])
            and np.sign(s_dark["CNR"]) == np.sign(s_dark["C"]),
        "|z| exceeds |CNR| for a large ROI":
            abs(s_bright["z"]) > abs(s_bright["CNR"]),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  sign convention: C+={s_bright['C']:.4f} C-={s_dark['C']:.4f} "
          f"-> {'ok' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------ classification
def _stats(c, z):
    return {"C": c, "z": z, "CNR": z, "mean_roi": 0.5, "mean_bg": 0.5,
            "std_bg": 0.01, "n_roi": 100, "n_bg": 400}


def test_classification_truth_table() -> bool:
    """Every branch of section 5.3, including the sign-agreement rule."""
    z_thr, x, y = 4.0, 0.30, 0.50
    reference = _stats(0.20, 8.0)          # a comfortably detected cavity
    cases = [
        ("preserved (identical)", _stats(0.20, 8.0), reference, D.PRESERVED),
        ("preserved (edge of X)", _stats(0.26, 8.0), reference, D.PRESERVED),
        ("deflated past X", _stats(0.12, 8.0), reference,
         D.DETECTED_NOT_PRESERVED),
        ("inflated past Y", _stats(0.40, 8.0), reference, D.INFLATED),
        ("just inside Y is not inflated", _stats(0.29, 8.0), reference,
         D.DETECTED_NOT_PRESERVED),
        ("lost below z_thr", _stats(0.20, 2.0), reference, D.LOST),
        ("lost by sign flip", _stats(-0.20, 8.0), reference, D.LOST),
        ("reference undetected", _stats(0.20, 8.0), _stats(0.20, 1.0),
         D.UNDETECTED_REF),
        ("reference with zero contrast", _stats(0.20, 8.0), _stats(0.0, 8.0),
         D.UNDETECTED_REF),
        ("NaN enhanced z is lost", _stats(0.20, float("nan")), reference,
         D.LOST),
    ]
    ok = True
    for label, enhanced, ref, expected in cases:
        got = D.classify(enhanced, ref, z_thr, x, y)
        if got["label"] != expected:
            ok = False
            print(f"    {label}: got {got['label']!r} want {expected!r} -> FAIL")
        if set(got) != set(D.FLAGS) | {"label"}:
            ok = False
            print(f"    {label}: keys {sorted(got)} -> FAIL")
        # the label must agree with its own flag; DETECTED_NOT_PRESERVED has no
        # flag of its own and means "detected, but neither preserved nor
        # inflated"
        agrees = (got[expected] if expected in D.FLAGS
                  else not (got[D.PRESERVED] or got[D.INFLATED]))
        if not agrees:
            ok = False
            print(f"    {label}: the label disagrees with the flags "
                  f"{ {k: v for k, v in got.items() if k != 'label'} } -> FAIL")

    # a dense body: the reference contrast is negative, sign agreement holds
    dense_ref = _stats(-0.18, -9.0)
    checks = {
        "negative reference preserved":
            D.classify(_stats(-0.19, -9.0), dense_ref, z_thr, x, y)["label"]
            == D.PRESERVED,
        "negative reference, positive enhanced is lost":
            D.classify(_stats(0.19, 9.0), dense_ref, z_thr, x, y)["label"]
            == D.LOST,
        "negative reference inflated":
            D.classify(_stats(-0.40, -9.0), dense_ref, z_thr, x, y)["label"]
            == D.INFLATED,
        "sham above threshold hallucinates":
            D.classify_sham(_stats(0.05, 5.0), z_thr) == D.HALLUCINATED,
        "sham below threshold is clean":
            D.classify_sham(_stats(0.05, 1.0), z_thr) == D.CLEAN_SHAM,
        "sham with negative z also hallucinates":
            D.classify_sham(_stats(-0.05, -5.0), z_thr) == D.HALLUCINATED,
    }

    # X > Y: the two predicates OVERLAP and a case is preserved AND inflated.
    # X is frozen as max(0.30, 3 rel-SD(C_ref over K)), so a noisy reference
    # legitimately produces X = 0.60 against the fixed Y = 0.50.
    wide_x = 0.60
    both = D.classify(_stats(0.31, 8.0), reference, z_thr, wide_x, y)
    neither = D.classify(_stats(0.22, 8.0), reference, z_thr, wide_x, y)
    checks["a case can be preserved AND inflated when X > Y"] = (
        both[D.PRESERVED] and both[D.INFLATED])
    checks["the overlapping case is labelled preserved"] = (
        both["label"] == D.PRESERVED)
    checks["inside both tolerances flags neither inflation"] = (
        neither[D.PRESERVED] and not neither[D.INFLATED])
    checks["a real ROI never sets the sham flag"] = not any(
        D.classify(enh, ref, z_thr, x, y)[D.HALLUCINATED_SHAM]
        for _label, enh, ref, _expected in cases)

    # the rates must count inflation from the FLAG, so the overlapping case is
    # in the inflation numerator even though its label says preserved
    population = [both, neither,
                  D.classify(_stats(0.20, 2.0), reference, z_thr, wide_x, y),
                  D.classify(_stats(0.20, 8.0), _stats(0.20, 1.0), z_thr,
                             wide_x, y)]
    preserved_rate, p_num, p_den = D.flag_rate(population, D.PRESERVED)
    inflated_rate, i_num, i_den = D.flag_rate(population, D.INFLATED)
    lost_rate, l_num, _l_den = D.flag_rate(population, D.LOST)
    checks["flag_rate drops the undetected-reference valve"] = (
        p_den == 3 and i_den == 3)
    checks["preserved counts both overlapping cases"] = (
        p_num == 2 and abs(preserved_rate - 2.0 / 3.0) < 1e-12)
    checks["inflation is counted despite the preserved label"] = (
        i_num == 1 and abs(inflated_rate - 1.0 / 3.0) < 1e-12)
    checks["lost is counted independently"] = (
        l_num == 1 and abs(lost_rate - 1.0 / 3.0) < 1e-12)

    for label, passed in checks.items():
        if not passed:
            ok = False
            print(f"    {label}: FAIL")

    rate, num, den = D.rate_with_denominator(
        [D.PRESERVED, D.PRESERVED, D.LOST, D.UNDETECTED_REF], {D.PRESERVED})
    if not (num == 2 and den == 3 and abs(rate - 2.0 / 3.0) < 1e-12):
        ok = False
        print(f"    denominator excludes UNDETECTED_REF: got {num}/{den} -> FAIL")

    print(f"  classification truth table: {len(cases)} cases + "
          f"{len(checks)} flag checks -> {'ok' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------- octiles
def test_octiles() -> bool:
    """Octiles must be ordered by path length and monotone in the signal."""
    l_a = np.zeros(SHAPE)
    rr, cc = np.ogrid[0:SHAPE[0], 0:SHAPE[1]]
    radius = np.sqrt((rr - 48.0) ** 2 + (cc - 48.0) ** 2)
    core = radius <= 16.0
    l_a[core] = 2.0 * np.sqrt(np.maximum(16.0 ** 2 - radius[core] ** 2, 0.0))

    roi, _ = R.build_roi(l_a)
    annulus = _ring(SHAPE, 48, 48, 22, 32)
    rng = np.random.default_rng(20260805)
    img = 0.5 + 0.005 * rng.standard_normal(SHAPE) + 0.002 * l_a

    rows = D.octile_stats(img, l_a, annulus, roi=roi)
    lengths = [r["l_mean_m"] for r in rows if r["n_roi"] > 0]
    contrasts = [r["C"] for r in rows if r["n_roi"] > 0]
    # a degenerate ROI: the 2 m ray step makes every path length identical, so
    # all nine quantile edges collapse onto one value
    flat_l = np.zeros(SHAPE)
    flat_roi = _disk(SHAPE, 48, 48, 10)
    flat_l[flat_roi] = 40.0
    flat_rows = D.octile_stats(img, flat_l, annulus, roi=flat_roi)
    # three distinct plateaus: the nine quantile edges deduplicate to
    # [20, 40, 60], i.e. two half-open bins, both populated
    steps_l = np.zeros(SHAPE)
    steps_l[_disk(SHAPE, 48, 48, 18)] = 20.0
    steps_l[_disk(SHAPE, 48, 48, 12)] = 40.0
    steps_l[_disk(SHAPE, 48, 48, 6)] = 60.0
    steps_roi = steps_l > 0
    step_rows = D.octile_stats(img, steps_l, annulus, roi=steps_roi)

    checks = {
        "eight octiles": len(rows) == 8,
        "every octile populated": all(r["n_roi"] > 0 for r in rows),
        "path length increases": all(np.diff(lengths) > 0),
        "contrast increases with path length":
            float(np.corrcoef(lengths, contrasts)[0, 1]) > 0.9,
        "octile sizes sum to the ROI":
            sum(r["n_roi"] for r in rows) == int(roi.sum()),
        "the effective bin count travels with every row":
            all(r["n_octiles"] == len(rows) for r in rows),
        "a constant path length collapses to one bin": len(flat_rows) == 1,
        "the collapsed bin holds the whole ROI":
            flat_rows[0]["n_roi"] == int(flat_roi.sum())
            and flat_rows[0]["n_octiles"] == 1,
        "tied edges are deduplicated, never left empty":
            len(step_rows) == 2 and all(r["n_roi"] > 0 for r in step_rows),
        "the merged bins still cover the ROI exactly once":
            sum(r["n_roi"] for r in step_rows) == int(steps_roi.sum()),
        "no merged row reports a NaN statistic":
            all(np.isfinite(r["C"]) and np.isfinite(r["z"])
                for r in flat_rows + step_rows),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  octiles: n={[r['n_roi'] for r in rows]} "
          f"C={[round(c, 4) for c in contrasts]} -> {'ok' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- local SSIM
def test_local_ssim() -> bool:
    """The crop follows the ROI, never falls below 7x7, and is order-safe."""
    roi = _disk(SHAPE, 48, 48, 8)
    rng = np.random.default_rng(5)
    ref = rng.random(SHAPE)
    checks = {
        "identical images give 1": abs(D.local_roi_ssim(ref, ref, roi) - 1.0) < 1e-9,
        "damaged ROI scores lower": True,
        "bbox grown by the margin":
            D.local_roi_bbox(roi, SHAPE) == (32, 65, 32, 65),
    }
    damaged = ref.copy()
    damaged[roi] = 0.0
    checks["damaged ROI scores lower"] = (
        D.local_roi_ssim(damaged, ref, roi) < D.local_roi_ssim(ref, ref, roi))

    tiny = np.zeros(SHAPE, dtype=bool)
    tiny[10, 10] = True
    box = D.local_roi_bbox(tiny, SHAPE)
    checks["single-pixel ROI still yields a >=7x7 crop"] = (
        box[1] - box[0] >= 7 and box[3] - box[2] >= 7)
    checks["single-pixel ROI SSIM is finite"] = np.isfinite(
        D.local_roi_ssim(ref, ref, tiny))
    checks["empty ROI is NaN"] = not np.isfinite(
        D.local_roi_ssim(ref, ref, np.zeros(SHAPE, dtype=bool)))

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  local ROI SSIM: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("Stage 19 detectability tests")
    results = {
        "exact_contrast": test_exact_contrast(),
        "sign_convention": test_sign_convention(),
        "classification": test_classification_truth_table(),
        "octiles": test_octiles(),
        "local_ssim": test_local_ssim(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 19 DETECTABILITY:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
