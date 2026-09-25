#!/usr/bin/env python3
"""Stage 19 ROI tests -- synthetic fixtures only (seconds, no campaign data).

These cover the pure geometry-free logic: the FWHM threshold arithmetic, the
annulus exclusion rules, the sham sampler constraints, the square-structuring
morphology and the physical DX conversion. One check does exercise the real
ray-caster, but on a small synthetic cone (R = 150 m) so the marched window is
short: it verifies that the measured path length through a sphere approaches
the analytic chord from below.

The checks that need the real 19-geometry catalog and the stored surrogate
opacity (V1-V5 of the execution plan) live in evaluation/test_roi_cluster.py
and run ON THE CLUSTER.

Run:  python scripts/evaluation/test_roi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "simulation"))

import roi as R  # noqa: E402
from surrogate.geometry import Anomaly, VolcanoGeometry  # noqa: E402
from surrogate.imaging import DetectorView  # noqa: E402

SHAPE = (64, 64)


def _disk(shape, row, col, radius):
    rr, cc = np.ogrid[0:shape[0], 0:shape[1]]
    return (rr - row) ** 2 + (cc - col) ** 2 <= radius ** 2


# ----------------------------------------------------------------- morphology
def test_morphology() -> bool:
    """Square dilation must equal the Chebyshev ball; erosion its dual."""
    mask = np.zeros(SHAPE, dtype=bool)
    mask[32, 32] = True
    checks = {}
    for radius in (1, 3, 7):
        grown = R._dilate(mask, radius)
        expected = (2 * radius + 1) ** 2
        checks[f"dilate r={radius} area"] = int(grown.sum()) == expected
        rr, cc = np.ogrid[0:SHAPE[0], 0:SHAPE[1]]
        cheb = np.maximum(np.abs(rr - 32), np.abs(cc - 32)) <= radius
        checks[f"dilate r={radius} shape"] = bool(np.array_equal(grown, cheb))
    solid = _disk(SHAPE, 32, 32, 10)
    checks["erode shrinks"] = int(R._erode(solid, 2).sum()) < int(solid.sum())
    checks["erode is dual of dilate"] = bool(np.array_equal(
        R._erode(solid, 2), ~R._dilate(~solid, 2)))
    # a full-True mask must survive erosion: borders replicate, they are not edges
    checks["replicate border"] = bool(R._erode(np.ones(SHAPE, bool), 3).all())
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  morphology: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------- ROI threshold
def test_roi_threshold() -> bool:
    """max(20 m, 0.5 max L) and the 64-pixel floor."""
    checks = {}

    # FWHM branch: a 100 m peak -> threshold 50 m
    l_a = np.zeros(SHAPE, dtype=np.float32)
    l_a[20:40, 20:40] = 40.0
    l_a[25:35, 25:35] = 100.0
    mask, threshold = R.build_roi(l_a)
    checks["fwhm threshold value"] = abs(threshold - 50.0) < 1e-9
    checks["fwhm selects the core"] = int(mask.sum()) == 100

    # floor branch: a weak anomaly whose half-max is below 20 m
    weak = np.zeros(SHAPE, dtype=np.float32)
    weak[10:22, 10:22] = 30.0
    weak[12:20, 12:20] = 36.0
    mask_w, threshold_w = R.build_roi(weak)
    checks["floor threshold value"] = abs(threshold_w - 20.0) < 1e-9
    checks["floor selects everything above 20 m"] = int(mask_w.sum()) == 144

    # the 64-pixel guard reports, it does not silently drop
    small = np.zeros(SHAPE, dtype=np.float32)
    small[30:37, 30:37] = 100.0            # 49 px
    mask_s, valid_s = R.check_roi_size(R.build_roi(small)[0])
    checks["small ROI flagged invalid"] = (not valid_s) and int(mask_s.sum()) == 49
    checks["large ROI flagged valid"] = R.check_roi_size(mask)[1]
    checks["pixel count"] = R.roi_pixel_count(mask) == 100
    checks["bounding box"] = R.bounding_box(mask) == (25, 35, 25, 35)

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  roi threshold: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------------- annulus
def test_annulus_exclusions() -> bool:
    """Every exclusion rule of protocol v1.1 section 5.2 must actually fire."""
    shape = (128, 128)
    roi = _disk(shape, 64, 40, 6)
    other = _disk(shape, 64, 64, 6)
    sky = np.zeros(shape, dtype=bool)
    sky[:8, :] = True                       # a sky band far from the ROI
    edge = R.silhouette_edge(sky)
    l_rock = np.full(shape, 400.0)

    annulus, info = R.build_annulus(roi, [roi, other], sky, edge, l_rock)
    checks = {
        "annulus is disjoint from its ROI": not (annulus & roi).any(),
        "inner clearance >= 4 px": not (annulus & R._dilate(roi, 3)).any(),
        "outer bound respected": bool((annulus & ~R._dilate(roi, 12)).sum() == 0),
        "other ROI excluded with 3 px margin":
            not (annulus & R._dilate(other, 3)).any(),
        "size reported": info["n_pixels"] == int(annulus.sum()),
    }

    # sky and the dilated silhouette edge
    sky_near = np.zeros(shape, dtype=bool)
    sky_near[:60, :] = True
    edge_near = R.silhouette_edge(sky_near)
    annulus_s, _ = R.build_annulus(roi, [roi], sky_near, edge_near, l_rock)
    checks["sky excluded"] = not (annulus_s & sky_near).any()
    checks["dilated edge excluded"] = not (annulus_s & R._dilate(edge_near, 3)).any()

    # rock-column homogeneity: a wedge 60 % off the ROI median must go
    l_var = np.full(shape, 400.0)
    l_var[:, 70:] = 400.0 * 1.6
    annulus_c, info_c = R.build_annulus(roi, [roi], sky, edge, l_var)
    checks["rock column filter fires"] = not (annulus_c[:, 70:]).any()
    checks["median reported"] = abs(info_c["median_rock_column_m"] - 400.0) < 1e-9

    # widening: strangle the ring so the 12 px version cannot reach 200 px.
    # A 15-row rock band around the ROI leaves 9 usable rows after the 3 px
    # edge exclusion: ~162 ring pixels at 12 px, ~234 at 16 px.
    tight_sky = np.ones(shape, dtype=bool)
    tight_sky[57:72, :] = False
    annulus_w, info_w = R.build_annulus(roi, [roi], tight_sky,
                                        R.silhouette_edge(tight_sky), l_rock)
    checks["widening flag set"] = bool(info_w["widened"])
    checks["widening reported when still short"] = (
        info_w["below_minimum"] == (info_w["n_pixels"] < R.MIN_ANNULUS_PIXELS))

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  annulus: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- sham ROIs
def test_sham_sampler() -> bool:
    """Shams: inside the silhouette, clear of real ROIs and of the edge."""
    shape = (128, 128)
    sky = np.ones(shape, dtype=bool)
    sky[14:114, 14:114] = False             # a square silhouette
    real = [_disk(shape, 40, 40, 7), _disk(shape, 90, 90, 5)]
    rng = np.random.default_rng(20260805)
    shams, info = R.sample_sham_rois_on_silhouette(sky, real, rng, n=10)

    edge = R.silhouette_edge(sky)
    union_real = real[0] | real[1]
    checks = {
        "ten shams obtained": len(shams) == 10,
        "attempts recorded": info["attempts"] >= 10,
    }
    inside = all(not (m & sky).any() for m in shams)
    disjoint = all(not (m & union_real).any() for m in shams)
    big = all(int(m.sum()) >= R.MIN_ROI_PIXELS for m in shams)
    clear_real = all(not (m & R._dilate(union_real, 0)).any() for m in shams)
    # the clearance binds on the WHOLE CLIPPED MASK, not only on the centre: a
    # sham that reaches the silhouette edge or a real ROI carries real
    # structure and would inflate the calibrated z_thr
    forbidden = R._dilate(union_real, 12) | R._dilate(edge, 12) | sky
    centres_ok, masks_ok = True, True
    for mask in shams:
        rows, cols = np.nonzero(mask)
        row, col = int(round(rows.mean())), int(round(cols.mean()))
        if forbidden[row, col]:
            centres_ok = False
        if (mask & forbidden).any():
            masks_ok = False
    checks["shams stay inside the silhouette"] = inside
    checks["no overlap with real ROIs"] = disjoint and clear_real
    checks["at least 64 rock pixels"] = big
    checks["centres respect the 12 px clearance"] = centres_ok
    checks["every sham PIXEL respects the 12 px clearance"] = masks_ok
    checks["the rejection counters are reported"] = all(
        key in info for key in ("rejected_clearance",
                                "rejected_below_min_pixels", "clearance_px"))

    # a silhouette so cramped that the ellipses cannot clear the edge: the
    # sampler must come up short and SAY so, never relax the constraint
    tight_sky = np.ones(shape, dtype=bool)
    tight_sky[56:72, 56:72] = False
    tight, tight_info = R.sample_sham_rois_on_silhouette(
        tight_sky, [_disk(shape, 63, 63, 12)], np.random.default_rng(3), n=10)
    tight_forbidden = (R._dilate(_disk(shape, 63, 63, 12), 12)
                       | R._dilate(R.silhouette_edge(tight_sky), 12)
                       | tight_sky)
    checks["a cramped view yields no violating sham"] = all(
        not (m & tight_forbidden).any() for m in tight)
    checks["a short view reports its shortfall"] = (
        tight_info["obtained"] == len(tight)
        and tight_info["obtained"] < tight_info["requested"])

    # reproducibility and the no-real-ROI fallback
    again, _ = R.sample_sham_rois_on_silhouette(
        sky, real, np.random.default_rng(20260805), n=10)
    checks["seeded sampler reproduces"] = all(
        np.array_equal(a, b) for a, b in zip(shams, again))
    fallback, _ = R.sample_sham_rois_on_silhouette(
        sky, [], np.random.default_rng(7), n=3)
    checks["fallback disk when no real ROI"] = (
        len(fallback) == 3 and all(int(m.sum()) >= R.MIN_ROI_PIXELS
                                   for m in fallback))

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  sham sampler: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------- delta opacity
def test_delta_opacity() -> bool:
    """DX = drho L 100: 0.5 g/cm^3 over 120 m is 6000 g/cm^2."""
    l_a = np.array([[0.0, 60.0], [120.0, 20.0]], dtype=np.float32)
    dx = R.delta_opacity(l_a, -0.5)
    expected = np.array([[0.0, -3000.0], [-6000.0, -1000.0]])
    ok = bool(np.allclose(dx, expected, atol=1e-9))
    print(f"  delta opacity: max|delta| = "
          f"{float(np.abs(dx - expected).max()):.3e} -> {'ok' if ok else 'FAIL'}")
    return ok


# ------------------------------------------- real ray-caster, small synthetic
def test_synthetic_ray_cast() -> bool:
    """The real marcher on a small cone: L_a must approach 2r from below.

    A sphere of radius 60 m entirely inside a cone of R = 150 m, H = 400 m.
    The longest chord is the diameter, 120 m; the 2 m sampling and the finite
    angular grid can only under-shoot, and the sampled length may exceed the
    exact chord by at most one step.
    """
    geo = VolcanoGeometry(
        geometry_id="synthetic-01", base_radius_m=150.0, height_m=400.0,
        rock_density=2.65,
        anomalies=[Anomaly("cavity", 0.2, {"cx_m": 0.0, "cy_m": 0.0,
                                           "cz_m": 200.0, "radius_m": 60.0})])
    view = DetectorView(geo.geometry_id, 0)
    l_rock, lengths = R.cast_path_lengths(geo, view)
    l_a = lengths[0]
    chord = R.analytic_max_chord_m(geo.anomalies[0])

    measured = float(l_a.max())
    mask, _ = R.build_roi(l_a)
    checks = {
        "anomaly is visible": measured > 0.5 * chord,
        "path length never exceeds the chord + one step":
            measured <= chord + R.RAY_STEP_M + 1e-6,
        "path length is a multiple of the step":
            abs(measured / R.RAY_STEP_M - round(measured / R.RAY_STEP_M)) < 1e-4,
        "the anomaly lives inside the silhouette":
            not (mask & R.sky_mask(l_rock)).any(),
        "the cone casts a silhouette": 0 < int(R.sky_mask(l_rock).sum())
                                       < l_rock.size,
        "rock column exceeds the anomaly chord":
            float(l_rock.max()) > chord,
    }
    # sky floor: the analytic prediction must be reproducible
    checks["sky opacity is positive and finite"] = np.isfinite(
        R.sky_opacity(geo, view)) and R.sky_opacity(geo, view) > 0.0
    # the ROI must be wide enough to matter and within the analytic envelope
    box = R.bounding_box(mask)
    width = box[3] - box[2]
    predicted = R.analytic_roi_azimuth_width_px(geo, view, 0)
    checks["ROI azimuth width within +-15 % of sqrt(3) r / D"] = (
        abs(width - predicted) <= 0.15 * predicted + 1.0)

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  synthetic ray cast: max L_a = {measured:.1f} m "
          f"(chord {chord:.1f} m), ROI {int(mask.sum())} px, "
          f"width {width} px vs {predicted:.1f} predicted -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("Stage 19 ROI tests (synthetic)")
    results = {
        "morphology": test_morphology(),
        "roi_threshold": test_roi_threshold(),
        "annulus_exclusions": test_annulus_exclusions(),
        "sham_sampler": test_sham_sampler(),
        "delta_opacity": test_delta_opacity(),
        "synthetic_ray_cast": test_synthetic_ray_cast(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 19 ROI:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
