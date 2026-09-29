#!/usr/bin/env python3
"""Work unit U8 -- self-verifying tests for oe1_pilot_roi.py.

Covers: the marched path-length decomposition against a closed-form
line/sphere and line/cylinder chord on a small synthetic geometry; the
disjoint-by-priority construction of the five ROI masks on synthetic
path-length arrays; the orientation-registration check on a synthetic
asymmetric map (where the true orientation must be picked over both
mirrors); the sign of the pre-registered effect (denser material
winning the overlap must give FEWER expected counts in R_int than the
conduit winning it); (work unit U8b) the pixel-block bootstrap and
verdict logic of the `registration` subcommand on a synthetic asymmetric
campaign/model residual pair, where the true orientation must be CONFIRMED
and a deliberately mirrored one CONTRADICTED; (work unit U8c) the region-sum
check's verdict logic and its symmetry null control; and (work unit U8c
amendment, 2026-09-29) the path-length azimuth-mirror symmetry gate that
replaced the original E_homog relative-difference gate -- a synthetic L map
must PASS with single-step flips excluded from S/S', and FAIL on either a
non-step difference or too large a flipped fraction of the footprint.

Run:  python scripts/analysis/test_oe1_pilot_roi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from surrogate.geometry import Anomaly, VolcanoGeometry  # noqa: E402
from surrogate.imaging import (  # noqa: E402
    DetectorView, RAY_STEP_M, angular_grid, ray_directions,
)
import oe1_pilot_roi as pre  # noqa: E402

SEED = 20260929


# ----------------------------------------------------------- analytic chords
def line_sphere_chord(origin, direction, center, radius) -> float:
    """Closed-form chord length of a unit-direction line through a sphere."""
    oc = origin - center
    b = 2.0 * np.dot(direction, oc)
    c = np.dot(oc, oc) - radius ** 2
    disc = b * b - 4.0 * c
    if disc < 0:
        return 0.0
    sq = np.sqrt(disc)
    t1, t2 = (-b - sq) / 2.0, (-b + sq) / 2.0
    return max(0.0, t2 - t1)


def line_cylinder_chord(origin, direction, cx, cy, radius, z0, z1) -> float:
    """Closed-form chord length through a finite vertical cylinder."""
    dx, dy, dz = direction
    a = dx * dx + dy * dy
    ox, oy, oz = origin[0] - cx, origin[1] - cy, origin[2]
    b = 2.0 * (dx * ox + dy * oy)
    c = ox * ox + oy * oy - radius ** 2
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return 0.0
    sq = np.sqrt(disc)
    t1, t2 = sorted([(-b - sq) / (2 * a), (-b + sq) / (2 * a)])
    if abs(dz) < 1e-12:
        return (t2 - t1) if z0 <= oz <= z1 else 0.0
    za, zb = (z0 - oz) / dz, (z1 - oz) / dz
    zt1, zt2 = min(za, zb), max(za, zb)
    t1, t2 = max(t1, zt1), min(t2, zt2)
    return max(0.0, t2 - t1)


def _synthetic_overlapping_geometry() -> VolcanoGeometry:
    """A small cone with an overlapping conduit (cylinder) and dense body
    (sphere, via equal ellipsoid semi-axes) -- the same overlap topology as
    geo-train-06, scaled down so the ray march stays cheap."""
    return VolcanoGeometry(
        geometry_id="synthetic-overlap-test",
        base_radius_m=300.0, height_m=250.0, rock_density=2.0,
        anomalies=[
            Anomaly("conduit", 1.5, {"cx_m": 0.0, "cy_m": 0.0, "radius_m": 15.0,
                                     "z0_m": 0.0, "z1_m": 200.0}),
            Anomaly("dense_body", 3.0, {"cx_m": 0.0, "cy_m": 0.0, "cz_m": 100.0,
                                        "ax_m": 25.0, "ay_m": 25.0, "az_m": 25.0}),
        ],
    )


# ------------------------------------------------------------------- test 1
def test_chord_length_vs_analytic() -> bool:
    """march_categories' per-pixel path length must match the closed-form
    line/sphere and line/cylinder chord within a couple of RAY_STEP_M
    (discretization) at the pixel whose ray passes near both centers."""
    geo = _synthetic_overlapping_geometry()
    view = DetectorView(geometry_id=geo.geometry_id, view_index=0)
    cats = pre.march_categories(geo, view)

    az, el = angular_grid()
    row = int(np.argmin(np.abs(np.rad2deg(el) - 5.634)))
    col = int(np.argmin(np.abs(np.rad2deg(az) - 0.0)))

    origin = view.position()
    direction = ray_directions(view)[row, col]

    analytic_sphere = line_sphere_chord(origin, direction, np.array([0.0, 0.0, 100.0]), 25.0)
    analytic_cylinder = line_cylinder_chord(origin, direction, 0.0, 0.0, 15.0, 0.0, 200.0)

    measured_body_total = float(cats["l_body_only"][row, col] + cats["l_int"][row, col])
    measured_cond_total = float(cats["l_cond_only"][row, col] + cats["l_int"][row, col])

    tol = 2.5 * RAY_STEP_M
    checks = {
        "sphere chord within tolerance": abs(measured_body_total - analytic_sphere) < tol,
        "cylinder chord within tolerance": abs(measured_cond_total - analytic_cylinder) < tol,
        "sphere chord positive": analytic_sphere > 0.0 and measured_body_total > 0.0,
        "cylinder chord positive": analytic_cylinder > 0.0 and measured_cond_total > 0.0,
        "overlap exists at this pixel": float(cats["l_int"][row, col]) > 0.0,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  chord_length_vs_analytic: analytic sphere={analytic_sphere:.2f} m "
         f"measured={measured_body_total:.2f} m | analytic cylinder={analytic_cylinder:.2f} m "
         f"measured={measured_cond_total:.2f} m -> {'ok' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------- test 2
def test_masks_disjoint_and_priority() -> bool:
    """Synthetic path-length arrays exercising every branch of the priority
    rule; assert_masks_disjoint must accept a correct partition and reject
    a deliberately broken one."""
    shape = (1, 6)
    l_rock = np.array([[100.0, 100.0, 100.0, 50.0, 0.0, 100.0]])
    l_int = np.array([[25.0, 5.0, 5.0, 0.0, 0.0, 25.0]])
    l_cond_only = np.array([[10.0, 30.0, 5.0, 0.0, 0.0, 40.0]])
    l_body_only = np.array([[10.0, 0.0, 25.0, 0.0, 0.0, 0.0]])

    cats = {
        "l_rock": l_rock, "l_int": l_int,
        "l_cond_only": l_cond_only, "l_body_only": l_body_only,
    }
    result = pre.build_masks(cats, min_path_m=20.0)
    masks = result["masks"]
    pre.assert_masks_disjoint(masks)  # must not raise

    expected = {
        "R_int": [True, False, False, False, False, True],
        "R_cond": [False, True, False, False, False, False],   # pixel 5 excluded: R_int wins
        "R_body": [False, False, True, False, False, False],
        "R_rock": [False, False, False, True, False, False],
        "R_sky": [False, False, False, False, True, False],
    }
    checks = {}
    for name, expected_row in expected.items():
        checks[f"{name} matches expected pattern"] = bool(
            np.array_equal(masks[name][0], np.array(expected_row)))

    raised = False
    try:
        pre.assert_masks_disjoint({"a": np.array([[True]]), "b": np.array([[True]])})
    except AssertionError:
        raised = True
    checks["assert_masks_disjoint rejects an overlapping pair"] = raised

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  masks_disjoint_and_priority: {sum(checks.values())}/{len(checks)} checks -> "
         f"{'ok' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------- test 3
def test_orientation_detection_picks_true_orientation() -> bool:
    """A deliberately azimuth- AND elevation-asymmetric synthetic map: the
    orientation check must resolve BOTH axes in favour of the true (not
    mirrored) orientation."""
    rng = np.random.default_rng(SEED)
    shape = (40, 40)
    rows, cols = np.indices(shape)
    # A smooth background plus a strong, off-center, asymmetric bump so the
    # map has no mirror symmetry on either axis (unlike an on-axis cone).
    background = 1000.0 * np.exp(-((rows - 5) ** 2 + (cols - 5) ** 2) / 800.0)
    bump = 400.0 * np.exp(-((rows - 30) ** 2 / 40.0 + (cols - 10) ** 2 / 10.0))
    expected_map = background + bump + 1.0

    # campaign = same structure with small multiplicative noise (Poisson-like)
    campaign = rng.poisson(expected_map).astype(np.float64)

    result = pre.orientation_check(campaign, expected_map)
    checks = {
        "azimuth resolved": result["azimuth_verdict"] == "resolved",
        "elevation resolved": result["elevation_verdict"] == "resolved",
        "true correlation is high": result["corr_true"] > 0.9,
        "true beats az mirror": result["corr_true"] > result["corr_az_mirror"],
        "true beats el mirror": result["corr_true"] > result["corr_el_mirror"],
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  orientation_detection: true={result['corr_true']:.4f} "
         f"az_mirror={result['corr_az_mirror']:.4f} ({result['azimuth_verdict']}) "
         f"el_mirror={result['corr_el_mirror']:.4f} ({result['elevation_verdict']}) -> "
         f"{'ok' if ok else 'FAIL'}")
    return ok


def test_orientation_check_flags_a_wrong_map() -> bool:
    """If the 'expected' map handed in is actually the mirrored one, the
    check must report 'wrong' for that axis, not silently pass."""
    rng = np.random.default_rng(SEED + 1)
    shape = (40, 40)
    rows, cols = np.indices(shape)
    true_map = (1000.0 * np.exp(-((rows - 30) ** 2 / 40.0 + (cols - 10) ** 2 / 10.0)) + 1.0)
    campaign = rng.poisson(true_map).astype(np.float64)

    mirrored_map = np.flip(true_map, axis=1)  # a wrong azimuth convention
    result = pre.orientation_check(campaign, mirrored_map)
    ok = result["azimuth_verdict"] == "wrong"
    if not ok:
        print(f"    expected azimuth_verdict == 'wrong', got {result['azimuth_verdict']!r}: FAIL")
    print(f"  orientation_check_flags_a_wrong_map: azimuth_verdict={result['azimuth_verdict']!r} -> "
         f"{'ok' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------- test 4
def test_expected_effect_sign_denser_wins_fewer_counts() -> bool:
    """On the real geo-train-06 catalog: B1 (denser body wins the overlap)
    must predict FEWER expected counts in R_int than the conduit-wins field
    (less dense material there absorbs fewer muons)."""
    geo = pre.roimod.load_geometry(pre.DEFAULT_GEOMETRY_ID)
    view = DetectorView(geometry_id=pre.DEFAULT_GEOMETRY_ID, view_index=0)
    cats = pre.march_categories(geo, view)
    mask_result = pre.build_masks(cats, pre.DEFAULT_MIN_PATH_M)
    masks = mask_result["masks"]

    x_b1 = pre.build_opacity_field(cats, cats["body_density"])
    x_cw = pre.build_opacity_field(cats, cats["conduit_density"])
    lam_b1 = pre.expected_counts_from_opacity(x_b1, pre.CAMPAIGN_EXPOSURE_DAYS)
    lam_cw = pre.expected_counts_from_opacity(x_cw, pre.CAMPAIGN_EXPOSURE_DAYS)

    table = pre.roi_effect_table(masks, lam_b1, lam_cw, s0=1.0, scale_factor=1.0)
    row = table["R_int"]

    checks = {
        "R_int has pixels": row["n_pixels"] > 0,
        "B1 < conduit-wins in R_int": row["expected_b1_per_1e6_events"] < row["expected_conduit_wins_per_1e6_events"],
        "Delta_max is negative": row["delta_max_b1_minus_conduit_wins"] < 0.0,
        "R_rock/R_sky controls show ~zero effect": (
            abs(table["R_rock"]["delta_max_b1_minus_conduit_wins"]) < 1e-6
            and abs(table["R_sky"]["delta_max_b1_minus_conduit_wins"]) < 1e-6
        ),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  expected_effect_sign: R_int B1={row['expected_b1_per_1e6_events']:.4f} "
         f"CW={row['expected_conduit_wins_per_1e6_events']:.4f} "
         f"Delta_max={row['delta_max_b1_minus_conduit_wins']:.4f} -> {'ok' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------- test 5
def _synthetic_asymmetric_residual_pair(rng, shape=(48, 48)):
    """(campaign_residual, model_residual) sharing a strong, off-axis,
    azimuth-asymmetric bump (plus a symmetric background), and a mask over
    which the true orientation must beat both azimuth mirrors -- the same
    signal shape evaluate_orientation_case builds from a real campaign."""
    rows, cols = np.indices(shape)
    background = 0.05 * np.exp(-((rows - shape[0] / 2) ** 2) / 400.0)
    bump = 0.6 * np.exp(-((rows - 34) ** 2 / 30.0 + (cols - 8) ** 2 / 8.0))
    m = background + bump
    noise = rng.normal(scale=0.03, size=shape)
    c = m + noise
    mask = np.ones(shape, dtype=bool)
    return c, m, mask


def test_registration_bootstrap_confirms_true_orientation() -> bool:
    """block_bootstrap_delta_r_ci + corr_safe, on a synthetic asymmetric
    campaign/model residual pair built the same way evaluate_orientation_case
    does: the TRUE orientation must give delta_r > 0 with a 95% CI excluding
    0 (the CONFIRMED branch of registration_verdict)."""
    rng = np.random.default_rng(SEED + 2)
    c, m, mask = _synthetic_asymmetric_residual_pair(rng)
    m_mirror = pre.mirror_azimuth(m)

    r_true = pre.corr_safe(c[mask], m[mask])
    r_mirror = pre.corr_safe(c[mask], m_mirror[mask])
    delta_r = r_true - r_mirror
    boot = pre.block_bootstrap_delta_r_ci(c, m, m_mirror, mask, block=6,
                                          n_bootstrap=500,
                                          rng=np.random.default_rng(SEED + 3))
    lo, hi = boot["ci95"]

    fake_case = {
        "geometry": "synthetic", "view": 0, "signal_size": 100,
        "insufficient_pixels": False,
        "delta_r": delta_r, "delta_r_ci95": [lo, hi],
        "confirms_true_orientation": bool(delta_r > 0.0 and lo > 0.0),
        "contradicts_true_orientation": bool(delta_r < 0.0 and hi < 0.0),
    }
    verdict = pre.registration_verdict([fake_case])

    checks = {
        "r_true beats r_mirror": r_true > r_mirror,
        "delta_r positive": delta_r > 0.0,
        "CI excludes zero (lower bound > 0)": lo > 0.0,
        "registration_verdict is CONFIRMED": verdict["verdict"] == "CONFIRMED",
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  registration_bootstrap_confirms_true_orientation: r_true={r_true:.4f} "
         f"r_mirror={r_mirror:.4f} delta_r={delta_r:.4f} ci95=[{lo:.4f}, {hi:.4f}] -> "
         f"{'ok' if ok else 'FAIL'}")
    return ok


def test_registration_verdict_contradicted_on_mirrored_map() -> bool:
    """If the 'model residual' handed in is actually the mirrored one (a
    genuine registration bug), block_bootstrap_delta_r_ci + registration_verdict
    must call it CONTRADICTED, not silently pass -- delta_r < 0 with a 95% CI
    excluding 0."""
    rng = np.random.default_rng(SEED + 4)
    c, m_true, mask = _synthetic_asymmetric_residual_pair(rng)
    # the campaign is generated from the TRUE map, but the "model residual"
    # fed to the check is deliberately the mirrored one -- a wrong azimuth
    # convention, exactly what a real registration bug would look like.
    m_wrong = pre.mirror_azimuth(m_true)
    m_wrong_mirror = pre.mirror_azimuth(m_wrong)  # = m_true

    r_true = pre.corr_safe(c[mask], m_wrong[mask])
    r_mirror = pre.corr_safe(c[mask], m_wrong_mirror[mask])
    delta_r = r_true - r_mirror
    boot = pre.block_bootstrap_delta_r_ci(c, m_wrong, m_wrong_mirror, mask, block=6,
                                          n_bootstrap=500,
                                          rng=np.random.default_rng(SEED + 5))
    lo, hi = boot["ci95"]

    fake_case = {
        "geometry": "synthetic", "view": 0, "signal_size": 100,
        "insufficient_pixels": False,
        "delta_r": delta_r, "delta_r_ci95": [lo, hi],
        "confirms_true_orientation": bool(delta_r > 0.0 and lo > 0.0),
        "contradicts_true_orientation": bool(delta_r < 0.0 and hi < 0.0),
    }
    verdict = pre.registration_verdict([fake_case])

    checks = {
        "delta_r negative (wrong map correlates worse)": delta_r < 0.0,
        "CI excludes zero (upper bound < 0)": hi < 0.0,
        "registration_verdict is CONTRADICTED": verdict["verdict"] == "CONTRADICTED",
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  registration_verdict_contradicted_on_mirrored_map: delta_r={delta_r:.4f} "
         f"ci95=[{lo:.4f}, {hi:.4f}] verdict={verdict['verdict']!r} -> "
         f"{'ok' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------- test 6
# Work unit U8c -- region-sum azimuth-registration check. Synthetic cases
# share a small (20, 60) grid: a uniform homogeneous background (lam_homog =
# 80 everywhere, trivially azimuth-symmetric) with the model residual m and
# the campaign built as described per test below.

_U8C_SHAPE = (20, 60)
_U8C_LAM_HOMOG = 80.0


def _u8c_deficit_case(geometry: str, view: int, rng) -> dict:
    """m has a one-sided deficit region (columns 5:15, m = log(0.5) = -0.69 <
    -t) whose azimuth mirror (columns 45:55) stays at the flat background
    (m = 0, well under t0); the campaign is a genuine Poisson draw of the
    TRUE (deficit-applying) expected map, columns 5:15 included."""
    lam_homog = np.full(_U8C_SHAPE, _U8C_LAM_HOMOG)
    lam_anom = lam_homog.copy()
    lam_anom[:, 5:15] = lam_homog[:, 5:15] * 0.5
    m = np.log(lam_anom / lam_homog)
    campaign = rng.poisson(lam_anom).astype(np.float64)
    return {"geometry": geometry, "view": view, "m": m, "lam_anom": lam_anom,
            "campaign": campaign}


def _u8c_clean_case(geometry: str, view: int, rng) -> dict:
    """No anomaly anywhere (m == 0 everywhere): a valid symmetry-null
    reference geometry for the null control."""
    lam_homog = np.full(_U8C_SHAPE, _U8C_LAM_HOMOG)
    m = np.zeros(_U8C_SHAPE)
    campaign = rng.poisson(lam_homog).astype(np.float64)
    return {"geometry": geometry, "view": view, "m": m, "lam_anom": lam_homog,
            "campaign": campaign}


def _u8c_asymmetric_null_case(geometry: str, view: int) -> dict:
    """A deliberately BAD null reference: its own model residual m is flat
    (|m| < t0 everywhere, so it PASSES the null-control eligibility gate),
    but its campaign counts are hugely skewed between columns 5:15 and their
    azimuth mirror 45:55 -- a real azimuth asymmetry in that geometry's own
    campaign, unrelated to the model, that the symmetry control must catch."""
    m = np.zeros(_U8C_SHAPE)
    lam_anom = np.full(_U8C_SHAPE, _U8C_LAM_HOMOG)
    campaign = np.full(_U8C_SHAPE, _U8C_LAM_HOMOG)
    campaign[:, 5:15] = 5000.0
    campaign[:, 45:55] = 5.0
    return {"geometry": geometry, "view": view, "m": m, "lam_anom": lam_anom,
            "campaign": campaign}


def test_registration_sum_confirms_true_orientation() -> bool:
    """(a) A symmetric homogeneous field with a one-sided deficit anomaly, a
    genuine Poisson campaign of the TRUE orientation, plus clean symmetry-null
    reference geometries at the same view -> CONFIRMED."""
    rng = np.random.default_rng(SEED + 9)
    main_case = _u8c_deficit_case("geo-u8c-main", 0, rng)
    cases = [main_case] + [_u8c_clean_case(f"geo-u8c-null-{i}", 0, rng) for i in range(5)]

    result = pre.registration_sum_check(cases)
    checks = {
        "verdict is CONFIRMED": result["verdict"] == "CONFIRMED",
        "at least one eligible sub-test": result["n_eligible_subtests"] >= 1,
        "combined Z >= 3": result["Z"] is not None and result["Z"] >= 3.0,
        "symmetry control passed (|Z_null| < 3)": abs(result["Z_null"]) < 3.0,
        "symmetry control passed (max|z_null| < 4)": result["max_abs_z_null"] < 4.0,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  registration_sum_confirms_true_orientation: verdict={result['verdict']!r} "
         f"Z={result['Z']:.3f} Z_null={result['Z_null']:.3f} "
         f"max|z_null|={result['max_abs_z_null']:.3f} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_registration_sum_contradicted_on_mirrored_campaign() -> bool:
    """(b) The same setup as (a), but the main case's campaign is mirrored in
    azimuth (a genuine registration bug) -> CONTRADICTED."""
    rng = np.random.default_rng(SEED + 10)
    main_case = _u8c_deficit_case("geo-u8c-main", 0, rng)
    main_case["campaign"] = pre.mirror_azimuth(main_case["campaign"])
    cases = [main_case] + [_u8c_clean_case(f"geo-u8c-null-{i}", 0, rng) for i in range(5)]

    result = pre.registration_sum_check(cases)
    checks = {
        "verdict is CONTRADICTED": result["verdict"] == "CONTRADICTED",
        "combined Z <= -3": result["Z"] is not None and result["Z"] <= -3.0,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  registration_sum_contradicted_on_mirrored_campaign: verdict={result['verdict']!r} "
         f"Z={result['Z']:.3f} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_registration_sum_no_anomaly_is_unresolved() -> bool:
    """(c) No anomaly anywhere (m == 0 everywhere) -> no region ever forms,
    so no sub-test is ever eligible -> UNRESOLVED (insufficient counts)."""
    rng = np.random.default_rng(SEED + 11)
    case = _u8c_clean_case("geo-u8c-flat", 0, rng)
    result = pre.registration_sum_check([case])
    checks = {
        "verdict is UNRESOLVED": result["verdict"] == "UNRESOLVED",
        "reason mentions insufficient counts": "insufficient" in result["verdict_reason"],
        "zero eligible sub-tests": result["n_eligible_subtests"] == 0,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  registration_sum_no_anomaly_is_unresolved: verdict={result['verdict']!r} "
         f"reason={result['verdict_reason']!r} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_registration_sum_regions_disjoint_and_mirrored() -> bool:
    """(d) S and its azimuth mirror S' must be disjoint and S' == s(S) exactly."""
    shape = (10, 20)
    m = np.zeros(shape)
    m[3, 2] = 0.2    # |m| > t=0.05 -> in S; its mirror column is 20-1-2=17, m=0 -> |m|<t0
    m[7, 15] = -0.2  # -0.2 < -t -> in D; its mirror column is 20-1-15=4, m=0 -> |m|<t0
    regions = pre.registration_sum_regions(m, t=0.05, t0=0.01)

    expected_excess = np.zeros(shape, dtype=bool); expected_excess[3, 2] = True
    expected_excess_mirror = np.zeros(shape, dtype=bool); expected_excess_mirror[3, 17] = True
    expected_deficit = np.zeros(shape, dtype=bool); expected_deficit[7, 15] = True
    expected_deficit_mirror = np.zeros(shape, dtype=bool); expected_deficit_mirror[7, 4] = True

    checks = {
        "S matches expected": bool(np.array_equal(regions["excess"], expected_excess)),
        "S' matches expected (S' = s(S))": bool(np.array_equal(regions["excess_mirror"], expected_excess_mirror)),
        "D matches expected": bool(np.array_equal(regions["deficit"], expected_deficit)),
        "D' matches expected (D' = s(D))": bool(np.array_equal(regions["deficit_mirror"], expected_deficit_mirror)),
        "S and S' disjoint": not np.any(regions["excess"] & regions["excess_mirror"]),
        "D and D' disjoint": not np.any(regions["deficit"] & regions["deficit_mirror"]),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  registration_sum_regions_disjoint_and_mirrored: {sum(checks.values())}/{len(checks)} "
         f"checks -> {'ok' if ok else 'FAIL'}")
    return ok


def test_registration_sum_symmetry_control_overrides_confirmed() -> bool:
    """(e) The same otherwise-CONFIRMED setup as (a), but one of the null
    reference geometries has a flat/symmetric model residual (so it PASSES
    the null-control eligibility gate) with a hugely azimuth-asymmetric
    campaign of its own -> the symmetry control must override the otherwise
    strong positive signal and report UNRESOLVED (symmetry control failed)."""
    rng = np.random.default_rng(SEED + 12)
    main_case = _u8c_deficit_case("geo-u8c-main", 0, rng)
    bad_null = _u8c_asymmetric_null_case("geo-u8c-bad-null", 0)
    cases = [main_case, bad_null] + [_u8c_clean_case(f"geo-u8c-null-{i}", 0, rng) for i in range(4)]

    result = pre.registration_sum_check(cases)
    checks = {
        "verdict is UNRESOLVED": result["verdict"] == "UNRESOLVED",
        "reason is symmetry control failed": result["verdict_reason"] == "symmetry control failed",
        "the main signal alone would have been strongly positive": result["Z"] is not None and result["Z"] >= 3.0,
        "|Z_null| >= 3 or max|z_null| >= 4": (
            abs(result["Z_null"]) >= 3.0 or result["max_abs_z_null"] >= 4.0),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  registration_sum_symmetry_control_overrides_confirmed: verdict={result['verdict']!r} "
         f"Z={result['Z']:.3f} Z_null={result['Z_null']:.3f} "
         f"max|z_null|={result['max_abs_z_null']:.3f} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_registration_sum_subtest_not_eligible_below_min_n() -> bool:
    """(f) N < 100 -> the sub-test is reported but not eligible (no z/z_s/p)."""
    shape = (5, 5)
    region = np.zeros(shape, dtype=bool); region[0, 0] = True
    region_mirror = np.zeros(shape, dtype=bool); region_mirror[0, 4] = True
    campaign = np.full(shape, 10.0)   # k = 10, N = 20 < 100
    lam_anom = np.ones(shape)

    result = pre.registration_sum_subtest("excess", region, region_mirror, campaign,
                                          lam_anom, min_n=pre.REGISTRATION_SUM_MIN_N)
    checks = {
        "sub-test is not None": result is not None,
        "N below min_n": result["N"] < pre.REGISTRATION_SUM_MIN_N,
        "not eligible": result["eligible"] is False,
        "z is None": result["z"] is None,
        "z_s is None": result["z_s"] is None,
        "binomial_p_value is None": result["binomial_p_value"] is None,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  registration_sum_subtest_not_eligible_below_min_n: N={result['N']} "
         f"eligible={result['eligible']} -> {'ok' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------- test 7
# Work unit U8c amendment (2026-09-29) -- path-length azimuth-mirror symmetry
# gate, replacing the original E_homog relative-difference gate. See
# oe1_pilot_roi.REGISTRATION_SUM_CRITERION["amendment"] for the full
# rationale: a single RAY_STEP_M float32 sampling flip of the analytic ray
# march at the cone surface in an oblique view is not a registration bug and
# must not block the region-sum check.

def _symmetric_l_map(shape=(40, 100), base=500.0, seed_offset=20) -> np.ndarray:
    """A homogeneous rock path length map that is EXACTLY azimuth-mirror
    symmetric by construction (built by mirroring its own left half onto the
    right half). `shape` is chosen so that a couple of flip pixels stay
    comfortably under the 0.1% footprint budget while a couple dozen do
    not."""
    rows, n_az = shape
    half = n_az // 2
    rng = np.random.default_rng(SEED + seed_offset)
    left = base + rng.uniform(-50.0, 50.0, size=(rows, half))
    return np.concatenate([left, left[:, ::-1]], axis=1)


def test_symmetry_gate_passes_on_single_step_flips() -> bool:
    """(a) A perfectly symmetric L map perturbed at ONE pixel by exactly one
    RAY_STEP_M: the gate must PASS (this creates exactly two flip pixels --
    the perturbed pixel and its mirror partner -- each a single step, well
    under the 0.1% footprint budget), and registration_sum_regions must
    exclude both flip pixels from S/S' even where one of them would
    otherwise qualify as excess signal."""
    l_map = _symmetric_l_map()
    row, col = 3, 5
    mirror_col = l_map.shape[1] - 1 - col
    l_map[row, col] += RAY_STEP_M  # single-step, one-sided perturbation

    gate = pre.registration_sum_symmetry_gate(l_map)

    # A spurious "excess" signal placed exactly at the flip pixel, with a
    # clean (|m| < t0) background everywhere else including its mirror --
    # without the exclusion this pixel would qualify for S.
    m = np.zeros(l_map.shape)
    m[row, col] = 0.2
    regions = pre.registration_sum_regions(m, t=0.05, t0=0.01, flip=gate["flip"])

    checks = {
        "gate passes overall": gate["passed"] is True,
        "step-multiple condition holds": gate["is_step_multiple"] is True,
        "flip budget condition holds": gate["within_flip_budget"] is True,
        "exactly two flip pixels": gate["n_flip_pixels"] == 2,
        "perturbed pixel flagged": bool(gate["flip"][row, col]),
        "mirror partner flagged": bool(gate["flip"][row, mirror_col]),
        "flip pixel excluded from S": not regions["excess"][row, col],
        "max |dL| is exactly one step": gate["max_abs_dl_steps"] == 1.0,
        "step residual is zero": gate["max_step_residual"] < 1e-9,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  symmetry_gate_passes_on_single_step_flips: n_flip_pixels="
         f"{gate['n_flip_pixels']}/{gate['footprint_pixels']} "
         f"max_abs_dl_steps={gate['max_abs_dl_steps']:.2f} "
         f"max_step_residual={gate['max_step_residual']:.2e} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_symmetry_gate_fails_on_non_step_difference() -> bool:
    """(b) A perturbation of 0.7 m -- not an integer multiple of RAY_STEP_M
    (2.0 m) -- must fail the gate on the step-multiple condition, even though
    only two pixels flip (well under the fraction budget)."""
    l_map = _symmetric_l_map()
    l_map[2, 7] += 0.7

    gate = pre.registration_sum_symmetry_gate(l_map)
    checks = {
        "gate fails overall": gate["passed"] is False,
        "step-multiple condition fails": gate["is_step_multiple"] is False,
        "flip budget condition alone still holds": gate["within_flip_budget"] is True,
        "step residual is 0.35": abs(gate["max_step_residual"] - 0.35) < 1e-9,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  symmetry_gate_fails_on_non_step_difference: max_abs_dl_steps="
         f"{gate['max_abs_dl_steps']:.4f} max_step_residual={gate['max_step_residual']:.4f} "
         f"-> {'ok' if ok else 'FAIL'}")
    return ok


def test_symmetry_gate_fails_when_flip_fraction_too_high() -> bool:
    """(c) Every flip is individually a valid single step, but flipping more
    than 0.1% of the footprint must still fail the gate on the budget
    condition."""
    l_map = _symmetric_l_map()
    footprint_pixels = int(np.count_nonzero(l_map > 0.0))
    max_allowed = pre.REGISTRATION_SUM_MAX_FLIP_FRACTION * footprint_pixels
    n_perturb = int(max_allowed) + 10  # comfortably over budget, still one-step each

    half = l_map.shape[1] // 2
    perturbed = 0
    for row in range(l_map.shape[0]):
        for col in range(half):
            if perturbed >= n_perturb:
                break
            l_map[row, col] += RAY_STEP_M  # left half only; mirror stays untouched
            perturbed += 1
        if perturbed >= n_perturb:
            break

    gate = pre.registration_sum_symmetry_gate(l_map)
    checks = {
        "gate fails overall": gate["passed"] is False,
        "step-multiple condition still holds": gate["is_step_multiple"] is True,
        "flip budget condition fails": gate["within_flip_budget"] is False,
        "flip count exceeds 0.1% of the footprint": (
            gate["n_flip_pixels"] > pre.REGISTRATION_SUM_MAX_FLIP_FRACTION * gate["footprint_pixels"]),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  symmetry_gate_fails_when_flip_fraction_too_high: n_flip_pixels="
         f"{gate['n_flip_pixels']}/{gate['footprint_pixels']} -> {'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("OE1 pilot ROI pre-registration -- self-verifying tests")
    results = {
        "chord_length_vs_analytic": test_chord_length_vs_analytic(),
        "masks_disjoint_and_priority": test_masks_disjoint_and_priority(),
        "orientation_detection_picks_true_orientation": test_orientation_detection_picks_true_orientation(),
        "orientation_check_flags_a_wrong_map": test_orientation_check_flags_a_wrong_map(),
        "expected_effect_sign_denser_wins_fewer_counts": test_expected_effect_sign_denser_wins_fewer_counts(),
        "registration_bootstrap_confirms_true_orientation": test_registration_bootstrap_confirms_true_orientation(),
        "registration_verdict_contradicted_on_mirrored_map": test_registration_verdict_contradicted_on_mirrored_map(),
        "registration_sum_confirms_true_orientation": test_registration_sum_confirms_true_orientation(),
        "registration_sum_contradicted_on_mirrored_campaign": test_registration_sum_contradicted_on_mirrored_campaign(),
        "registration_sum_no_anomaly_is_unresolved": test_registration_sum_no_anomaly_is_unresolved(),
        "registration_sum_regions_disjoint_and_mirrored": test_registration_sum_regions_disjoint_and_mirrored(),
        "registration_sum_symmetry_control_overrides_confirmed": test_registration_sum_symmetry_control_overrides_confirmed(),
        "registration_sum_subtest_not_eligible_below_min_n": test_registration_sum_subtest_not_eligible_below_min_n(),
        "symmetry_gate_passes_on_single_step_flips": test_symmetry_gate_passes_on_single_step_flips(),
        "symmetry_gate_fails_on_non_step_difference": test_symmetry_gate_fails_on_non_step_difference(),
        "symmetry_gate_fails_when_flip_fraction_too_high": test_symmetry_gate_fails_when_flip_fraction_too_high(),
    }
    passed = all(results.values())
    if not passed:
        failed = [k for k, v in results.items() if not v]
        print(f"failed checks: {', '.join(failed)}")
    print("OE1 PILOT ROI TESTS:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
