#!/usr/bin/env python3
"""Self-verifying tests for oe1_geometry_check_report.py — no pytest, prints
PASS/FAIL per check and exits non-zero on any failure.

Covers: the --check-geometry log parser (OVERLAP-CHECK / OVERLAP-CHECK-SUMMARY
/ GeomVol1002 counting) on small synthetic logs, the T1 criteria evaluator
on synthetic resumen rows (clean-pass, a missing-defect anomaly, and a
spurious-overlap anomaly), and the slice boundary/air classifier on a tiny
synthetic geometry with a known analytic answer at each grid point.

Run: python scripts/analysis/test_oe1_geometry_check_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "simulation"))

from surrogate.geometry import Anomaly, VolcanoGeometry  # noqa: E402
from oe1_geometry_check_report import (  # noqa: E402
    parse_check_log, evaluate_t1, classify_slice_mismatches,
    material_class, analytic_material_at, EXPECTED_DEFECTIVE,
)


# ------------------------------------------------------------------ fixtures
def _synthetic_geo() -> VolcanoGeometry:
    """A small cone with one conduit anomaly, scaled down so the grid used
    below stays tiny and every point's analytic class is hand-verifiable."""
    return VolcanoGeometry(
        geometry_id="synthetic-slice-test",
        base_radius_m=100.0, height_m=80.0, rock_density=2.65,
        anomalies=[Anomaly("conduit", 1.8,
                          {"cx_m": 0.0, "cy_m": 0.0, "radius_m": 10.0,
                           "z0_m": 0.0, "z1_m": 60.0})],
    )


CLEAN_LOG = """\
Checking overlaps for volume edifice:0 (G4Cons) ... OK!
OVERLAP-CHECK-HEADER resolution=100000 tolerance=0 maxErr=100 seed=1 g4version=x geometry=geo-x view=0 rng-note=x
Checking overlaps for volume edifice:0 (G4Cons) ... OK!
OVERLAP-CHECK volume=edifice copy=0 solid=G4Cons result=ok
OVERLAP-CHECK volume=detector copy=0 solid=G4Box result=ok
OVERLAP-CHECK-SUMMARY geometry=geo-x view=0 volumes=2 overlapping=0 status=OK
"""

OVERLAP_LOG = """\
Checking overlaps for volume anomaly_0:0 (G4Tubs) ...
-------- WWWW ------- G4Exception-START -------- WWWW -------
*** G4Exception : GeomVol1002
      issued by : G4PVPlacement::CheckOverlaps()
Overlap with volume already placed !
*** This is just a warning message. ***
-------- WWWW -------- G4Exception-END --------- WWWW -------
OVERLAP-CHECK volume=anomaly_0 copy=0 solid=G4Tubs result=overlap
OVERLAP-CHECK volume=anomaly_1 copy=0 solid=G4Ellipsoid result=overlap
OVERLAP-CHECK-SUMMARY geometry=geo-y view=3 volumes=4 overlapping=2 status=OVERLAPS
"""


# --------------------------------------------------------------------- tests
def test_parse_clean_log() -> bool:
    parsed = parse_check_log(CLEAN_LOG)
    ok = (parsed["n_GeomVol1002"] == 0 and parsed["n_GeomNav"] == 0
         and parsed["summary"]["overlapping"] == "0"
         and parsed["summary"]["status"] == "OK"
         and len(parsed["per_volume"]) == 2)
    print(f"  parse clean log: n_GeomVol1002={parsed['n_GeomVol1002']} "
         f"overlapping={parsed['summary']['overlapping']} status={parsed['summary']['status']} "
         f"-> {'ok' if ok else 'FAIL'}")
    return ok


def test_parse_overlap_log() -> bool:
    parsed = parse_check_log(OVERLAP_LOG)
    ok = (parsed["n_GeomVol1002"] == 1 and parsed["summary"]["overlapping"] == "2"
         and parsed["summary"]["status"] == "OVERLAPS"
         and len(parsed["per_volume"]) == 2
         and parsed["per_volume"][0]["result"] == "overlap")
    print(f"  parse overlap log: n_GeomVol1002={parsed['n_GeomVol1002']} "
         f"overlapping={parsed['summary']['overlapping']} -> {'ok' if ok else 'FAIL'}")
    return ok


def _rows_for(geometries_with_overlap_a: set, geometries_with_overlap_b: set) -> list[dict]:
    from oe1_geometry_check_report import DEFAULT_GEOMETRY_IDS
    rows = []
    for geo in DEFAULT_GEOMETRY_IDS:
        for variant, overlap_set in (("A", geometries_with_overlap_a),
                                     ("B", geometries_with_overlap_b)):
            has_overlap = geo in overlap_set
            for view in range(6):
                rows.append({
                    "variant": variant, "geometry": geo, "view": view,
                    "exit_code": 1 if has_overlap else 0,
                    "overlapping": 1 if has_overlap else 0,
                    "volumes": 4, "n_GeomVol1002": 1 if has_overlap else 0,
                    "n_GeomNav": 0, "wall_s": 1.0,
                })
    return rows


def test_t1_clean_pass() -> bool:
    rows = _rows_for(EXPECTED_DEFECTIVE, set())
    ev = evaluate_t1(rows)
    ok = ev["pass"] and ev["n_runs"] == 228 and not ev["b_geometries_with_overlaps"]
    print(f"  T1 clean pass: pass={ev['pass']} n_runs={ev['n_runs']} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_t1_detects_missing_defect() -> bool:
    """Variant A shows overlaps in only 5 of the 6 expected defective
    geometries (one defect went undetected) — must NOT report pass."""
    missing_one = EXPECTED_DEFECTIVE - {"geo-train-06"}
    rows = _rows_for(missing_one, set())
    ev = evaluate_t1(rows)
    ok = (not ev["pass"]) and "geo-train-06" not in ev["a_defective_geometries"]
    print(f"  T1 detects missing defect: pass={ev['pass']} "
         f"a_defective={ev['a_defective_geometries']} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_t1_detects_spurious_b_overlap() -> bool:
    """Variant B shows an overlap where none is expected — must NOT report
    pass, and the geometry must be listed in b_geometries_with_overlaps."""
    rows = _rows_for(EXPECTED_DEFECTIVE, {"geo-val-01"})
    ev = evaluate_t1(rows)
    ok = (not ev["pass"]) and ev["b_geometries_with_overlaps"] == ["geo-val-01"]
    print(f"  T1 detects spurious B overlap: pass={ev['pass']} "
         f"b_overlaps={ev['b_geometries_with_overlaps']} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_t1_detects_exit_code_mismatch() -> bool:
    rows = _rows_for(EXPECTED_DEFECTIVE, set())
    rows[0]["exit_code"] = 1  # rows[0] is A, geo-gen-01 (defective) view 0 -> already 1; flip a clean one
    # find a clean-B row and corrupt its exit code
    for r in rows:
        if r["variant"] == "B" and r["overlapping"] == 0:
            r["exit_code"] = 1
            break
    ev = evaluate_t1(rows)
    ok = not ev["pass"] and any(a["reason"].startswith("exit_code=") for a in ev["anomalies"])
    print(f"  T1 detects exit-code/overlap mismatch: pass={ev['pass']} -> {'ok' if ok else 'FAIL'}")
    return ok


def _grid_1d(lo, hi, step):
    n = int(round((hi - lo) / step)) + 1
    return np.linspace(lo, hi, n)


def test_slice_classifier_interior_and_air() -> bool:
    """Material labels built EXACTLY from the analytic winner (rock/air/
    anomaly_0) at every grid point must never be flagged as an unexplained
    mismatch, regardless of the point's distance to a boundary."""
    geo = _synthetic_geo()
    step = 2.0
    xs = _grid_1d(-30.0, 30.0, step)
    zs = _grid_1d(-10.0, 90.0, step)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    x, y, z = X.ravel(), np.zeros_like(X).ravel(), Z.ravel()
    rho = geo.density(x, y, z)

    r = np.hypot(x, y)
    cone_surface = geo.height_m * np.clip(1.0 - r / geo.base_radius_m, 0.0, None)
    inside_cone = (z >= 0.0) & (z <= cone_surface)
    conduit = geo.anomalies[0]
    in_conduit = ((x - conduit.params["cx_m"]) ** 2 + (y - conduit.params["cy_m"]) ** 2
                 <= conduit.params["radius_m"] ** 2) & (z >= conduit.params["z0_m"]) & (z <= conduit.params["z1_m"])
    material = np.where(~inside_cone, "G4_AIR",
                        np.where(in_conduit, "anomaly_0_mat", "standard_rock"))

    slice_data = {"x_m": x, "y_m": y, "z_m": z, "material": material, "density_g_cm3": rho}
    cls = classify_slice_mismatches(slice_data, geo, plane="y", plane_coord=0.0)
    ok = cls["n_mismatch_unexplained"] == 0
    print(f"  slice classifier (matched materials): n_unexplained={cls['n_mismatch_unexplained']} "
         f"n_boundary={cls['n_mismatch_boundary']} n_points={cls['n_points']} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_slice_classifier_flags_real_mismatch() -> bool:
    """Same regular grid as the previous test (so the inferred half-step
    reflects a real slice resolution), but the material label at one point
    deep inside the conduit (>1 grid step from every boundary) is
    deliberately wrong — it must be the sole unexplained mismatch."""
    geo = _synthetic_geo()
    step = 2.0
    xs = _grid_1d(-30.0, 30.0, step)
    zs = _grid_1d(-10.0, 90.0, step)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    x, y, z = X.ravel(), np.zeros_like(X).ravel(), Z.ravel()
    rho = geo.density(x, y, z)

    r = np.hypot(x, y)
    cone_surface = geo.height_m * np.clip(1.0 - r / geo.base_radius_m, 0.0, None)
    inside_cone = (z >= 0.0) & (z <= cone_surface)
    conduit = geo.anomalies[0]
    in_conduit = ((x - conduit.params["cx_m"]) ** 2 + (y - conduit.params["cy_m"]) ** 2
                 <= conduit.params["radius_m"] ** 2) & (z >= conduit.params["z0_m"]) & (z <= conduit.params["z1_m"])
    material = np.where(~inside_cone, "G4_AIR",
                        np.where(in_conduit, "anomaly_0_mat", "standard_rock"))

    # Point deep inside the conduit: x=0, z=30 (radius 10, z-range 0..60,
    # so this point sits >=8 m from the lateral surface and >=30 m from
    # either cap — well beyond one grid step of 2 m).
    target = np.argmin((x - 0.0) ** 2 + (z - 30.0) ** 2)
    assert in_conduit[target], "fixture bug: target point must be inside the conduit"
    material = material.copy()
    material[target] = "standard_rock"  # deliberately wrong (should be anomaly_0_mat)

    slice_data = {"x_m": x, "y_m": y, "z_m": z, "material": material, "density_g_cm3": rho}
    cls = classify_slice_mismatches(slice_data, geo, plane="y", plane_coord=0.0)
    ok = cls["n_mismatch_unexplained"] == 1 and bool(cls["_arrays"]["unexplained"][target])
    print(f"  slice classifier (forced mismatch): n_unexplained={cls['n_mismatch_unexplained']} "
         f"target_flagged={bool(cls['_arrays']['unexplained'][target])} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_slice_classifier_conduit_respects_axial_extent() -> bool:
    """Regression for the `A | (B & A)` absorption bug: a wrong label at the
    conduit's radial boundary (radial_gap ~ 0) but well past its axial
    extent z1 (here z=66, 6 m past z1=60, with half_step=1 m) must be
    flagged as an unexplained mismatch, not silently absorbed into the
    'near boundary' bucket regardless of z."""
    geo = _synthetic_geo()  # conduit: radius_m=10, z0=0, z1=60
    step = 2.0
    xs = _grid_1d(-30.0, 30.0, step)
    zs = _grid_1d(-10.0, 90.0, step)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    x, y, z = X.ravel(), np.zeros_like(X).ravel(), Z.ravel()
    rho = geo.density(x, y, z)

    r = np.hypot(x, y)
    cone_surface = geo.height_m * np.clip(1.0 - r / geo.base_radius_m, 0.0, None)
    inside_cone = (z >= 0.0) & (z <= cone_surface)
    conduit = geo.anomalies[0]
    in_conduit = ((x - conduit.params["cx_m"]) ** 2 + (y - conduit.params["cy_m"]) ** 2
                 <= conduit.params["radius_m"] ** 2) & (z >= conduit.params["z0_m"]) & (z <= conduit.params["z1_m"])
    material = np.where(~inside_cone, "G4_AIR",
                        np.where(in_conduit, "anomaly_0_mat", "standard_rock"))

    # Target: x=10 (== conduit radius, so radial_gap=0), z=66 (6 m past
    # z1=60, and 6 m short of the cone surface at r=10, cone_surface=72 --
    # both well past the 1 m half-grid-step tolerance). Analytically this
    # is plain rock (inside the cone, outside the conduit's axial range).
    target = np.argmin((x - 10.0) ** 2 + (z - 66.0) ** 2)
    assert abs(x[target] - 10.0) < 1e-9 and abs(z[target] - 66.0) < 1e-9, \
        "fixture bug: target point must land exactly on the grid at (10, 66)"
    assert inside_cone[target] and not in_conduit[target], \
        "fixture bug: target point must be inside the cone but outside the conduit"
    material = material.copy()
    material[target] = "anomaly_0_mat"  # deliberately wrong (should be standard_rock)

    slice_data = {"x_m": x, "y_m": y, "z_m": z, "material": material, "density_g_cm3": rho}
    cls = classify_slice_mismatches(slice_data, geo, plane="y", plane_coord=0.0)
    ok = cls["n_mismatch_unexplained"] == 1 and bool(cls["_arrays"]["unexplained"][target])
    print(f"  slice classifier (conduit axial-extent regression): "
         f"n_unexplained={cls['n_mismatch_unexplained']} "
         f"target_flagged={bool(cls['_arrays']['unexplained'][target])} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_slice_classifier_z_plane_cone_boundary_uses_radius() -> bool:
    """Regression for the z-plane cone-boundary dimension bug: at a fixed z,
    the cone surface is a circle of radius r_boundary(z), so the half-step
    tolerance must be applied radially, not via the y-plane's z-distance
    formula (which, for this geometry's base_radius/height=1.25 ratio,
    would silently widen the tolerance to 1.25x and mask a real mismatch
    that sits just outside the true 1 m radial tolerance)."""
    geo = _synthetic_geo()  # base_radius_m=100, height_m=80 -> ratio 1.25
    z_fixed = 40.0
    # r_boundary(z=40) = 100*(1-40/80) = 50. Two auxiliary points fix the
    # inferred grid step at 2 m (half_step=1 m); the target sits at r=48.8,
    # i.e. 1.2 m inside the true boundary -- outside a correct 1 m radial
    # tolerance, but inside the old, wrongly-scaled 1.25 m tolerance.
    x = np.array([0.0, 2.0, 48.8])
    y = np.array([0.0, 0.0, 0.0])
    z = np.full(3, z_fixed)
    rho = geo.density(x, y, z)

    r = np.hypot(x, y)
    cone_surface = geo.height_m * np.clip(1.0 - r / geo.base_radius_m, 0.0, None)
    inside_cone = (z >= 0.0) & (z <= cone_surface)
    conduit = geo.anomalies[0]
    in_conduit = ((x - conduit.params["cx_m"]) ** 2 + (y - conduit.params["cy_m"]) ** 2
                 <= conduit.params["radius_m"] ** 2) & (z >= conduit.params["z0_m"]) & (z <= conduit.params["z1_m"])
    material = np.where(~inside_cone, "G4_AIR",
                        np.where(in_conduit, "anomaly_0_mat", "standard_rock"))
    assert inside_cone[2] and not in_conduit[2], \
        "fixture bug: target point must be inside the cone but outside the conduit"

    material = material.copy()
    material[2] = "anomaly_0_mat"  # deliberately wrong (should be standard_rock)

    slice_data = {"x_m": x, "y_m": y, "z_m": z, "material": material, "density_g_cm3": rho}
    cls = classify_slice_mismatches(slice_data, geo, plane="z", plane_coord=z_fixed)
    ok = cls["n_mismatch_unexplained"] == 1 and bool(cls["_arrays"]["unexplained"][2])
    print(f"  slice classifier (z-plane radial-boundary regression): "
         f"n_unexplained={cls['n_mismatch_unexplained']} "
         f"target_flagged={bool(cls['_arrays']['unexplained'][2])} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_analytic_material_at_matches_density() -> bool:
    geo = _synthetic_geo()
    label, rho = analytic_material_at(geo, 0.0, 0.0, 30.0)
    ok = label == "anomaly_0" and abs(rho - 1.8) < 1e-9
    label2, rho2 = analytic_material_at(geo, 50.0, 0.0, 5.0)
    ok = ok and label2 == "rock" and abs(rho2 - 2.65) < 1e-9
    label3, rho3 = analytic_material_at(geo, 0.0, 0.0, 200.0)
    ok = ok and label3 == "air"
    print(f"  analytic_material_at: conduit={label}/{rho}, rock={label2}/{rho2}, "
         f"air={label3} -> {'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    tests = [
        test_parse_clean_log, test_parse_overlap_log,
        test_t1_clean_pass, test_t1_detects_missing_defect,
        test_t1_detects_spurious_b_overlap, test_t1_detects_exit_code_mismatch,
        test_slice_classifier_interior_and_air, test_slice_classifier_flags_real_mismatch,
        test_slice_classifier_conduit_respects_axial_extent,
        test_slice_classifier_z_plane_cone_boundary_uses_radius,
        test_analytic_material_at_matches_density,
    ]
    results = []
    for t in tests:
        print(f"{t.__name__}:")
        results.append(t())
    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} tests passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
