#!/usr/bin/env python3
"""Self-verifying tests for oe1_geometry_audit.py — no pytest, prints
PASS/FAIL per check and exits non-zero on any failure.

Run: python scripts/analysis/test_oe1_geometry_audit.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from oe1_geometry_audit import (  # noqa: E402
    AnomalyDef, mc_pair_intersection, conduit_containment, sphere_containment,
    ray_cylinder_chord, ray_ellipsoid_chord, b3_minimal_displacement,
    audit_geometry, build_detail_block, APEX_R_G4,
)

SEED = 20260929


def two_sphere_lens_volume(r: float, d: float) -> float:
    """Exact intersection volume of two equal spheres of radius r whose
    centers are separated by distance d < 2r."""
    return (math.pi / 12.0) * (4 * r + d) * (2 * r - d) ** 2


# ------------------------------------------------------------------- tests
def test_overlapping_spheres_lens() -> bool:
    r, d = 1.0, 1.0
    a = AnomalyDef(0, "cavity", 1.0, {"cx_m": 0.0, "cy_m": 0.0, "cz_m": 0.0, "radius_m": r})
    b = AnomalyDef(1, "cavity", 1.0, {"cx_m": d, "cy_m": 0.0, "cz_m": 0.0, "radius_m": r})
    rng = np.random.default_rng(SEED)
    res = mc_pair_intersection(rng, a, b, R=1000.0, H=1000.0, n_samples=4_000_000)
    expected = two_sphere_lens_volume(r, d)
    n_se = abs(res["volume_m3"] - expected) / max(res["se_m3"], 1e-12)
    ok = res["k"] > 0 and n_se < 4.0
    print(f"  overlapping unit spheres: mc={res['volume_m3']:.5f}+-{res['se_m3']:.5f} "
         f"expected={expected:.5f} ({n_se:.2f} SE) -> {'ok' if ok else 'FAIL'}")
    return ok


def test_cylinder_inside_ellipsoid() -> bool:
    conduit = AnomalyDef(0, "conduit", 1.0,
                         {"cx_m": 0.0, "cy_m": 0.0, "radius_m": 2.0, "z0_m": -3.0, "z1_m": 3.0})
    body = AnomalyDef(1, "dense_body", 1.0,
                      {"cx_m": 0.0, "cy_m": 0.0, "cz_m": 0.0,
                       "ax_m": 10.0, "ay_m": 10.0, "az_m": 10.0})
    rng = np.random.default_rng(SEED)
    res = mc_pair_intersection(rng, conduit, body, R=1000.0, H=1000.0, n_samples=4_000_000)
    expected = conduit.volume()
    rel = abs(res["volume_m3"] - expected) / expected
    n_se = abs(res["volume_m3"] - expected) / max(res["se_m3"], 1e-12)
    ok = rel < 0.01 and n_se < 5.0
    print(f"  cylinder fully inside ellipsoid: mc={res['volume_m3']:.4f} "
         f"expected(cylinder vol)={expected:.4f} rel={rel:.4%} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_disjoint_solids() -> bool:
    a = AnomalyDef(0, "cavity", 1.0, {"cx_m": 0.0, "cy_m": 0.0, "cz_m": 0.0, "radius_m": 5.0})
    b = AnomalyDef(1, "cavity", 1.0, {"cx_m": 1000.0, "cy_m": 0.0, "cz_m": 0.0, "radius_m": 5.0})
    rng = np.random.default_rng(SEED)
    res = mc_pair_intersection(rng, a, b, R=1e6, H=1e6, n_samples=1_000)
    ok = (res["bbox_disjoint"] is True and res["k"] == 0 and res["volume_m3"] == 0.0
         and res["certificate"] is not None
         and res["certificate"]["kind"] == "disjoint_bounding_boxes")
    print(f"  disjoint solids: k={res['k']} volume={res['volume_m3']} "
         f"certificate={res['certificate']} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_conduit_protrusion_from_small_cone() -> bool:
    R, H = 50.0, 100.0
    conduit = AnomalyDef(0, "conduit", 1.0,
                         {"cx_m": 30.0, "cy_m": 0.0, "radius_m": 10.0,
                          "z0_m": 0.0, "z1_m": 80.0})
    row = conduit_containment(conduit, R, H, apex_r=0.0, label="analytic_cone")
    rho0 = 30.0 + 10.0
    R_top = R * (1.0 - 80.0 / H)  # cone radius at z=80
    expected_excess = rho0 - R_top
    ok = (row["state"] == "lateral_protrusion"
         and abs(row["max_radial_excess_m"] - expected_excess) < 1e-9
         and abs(row["excess_height_m"] - 80.0) < 1e-9)
    print(f"  conduit protrusion: excess={row['max_radial_excess_m']:.4f} "
         f"expected={expected_excess:.4f} height={row['excess_height_m']} "
         f"-> {'ok' if ok else 'FAIL'}")
    return ok


def test_sphere_crosses_base() -> bool:
    R, H = 500.0, 400.0
    sph = AnomalyDef(0, "cavity", 1.0, {"cx_m": 0.0, "cy_m": 0.0, "cz_m": 3.0, "radius_m": 5.0})
    row = sphere_containment(sph, R, H, apex_r=0.0, label="analytic_cone")
    ok = row["state"] == "crosses_base" and abs(row["min_z_m"] - (-2.0)) < 1e-9
    print(f"  sphere crosses base: state={row['state']} min_z={row['min_z_m']} "
         f"-> {'ok' if ok else 'FAIL'}")
    return ok


def test_los_chord_hand_calculation() -> bool:
    D = np.array([-100.0, 0.0, 50.0])
    u = np.array([1.0, 0.0, 0.0])
    iv_cyl = ray_cylinder_chord(D, u, cx=0.0, cy=0.0, r=5.0, z0=0.0, z1=100.0)
    ok_cyl = iv_cyl is not None and abs(iv_cyl[0] - 95.0) < 1e-9 and abs(iv_cyl[1] - 105.0) < 1e-9

    D2 = np.array([-50.0, 0.0, 0.0])
    iv_ell = ray_ellipsoid_chord(D2, u, cx=0.0, cy=0.0, cz=0.0, ax=7.0, ay=3.0, az=4.0)
    ok_ell = iv_ell is not None and abs(iv_ell[0] - 43.0) < 1e-9 and abs(iv_ell[1] - 57.0) < 1e-9

    ok = ok_cyl and ok_ell
    print(f"  LOS chord hand calc: cylinder={iv_cyl} (expect 95,105) "
         f"ellipsoid={iv_ell} (expect 43,57) -> {'ok' if ok else 'FAIL'}")
    return ok


def test_b3_displacement_closed_form() -> bool:
    rc, d = 5.0, 8.0
    a_body = 10.0
    conduit = AnomalyDef(0, "conduit", 1.0,
                         {"cx_m": 0.0, "cy_m": 0.0, "radius_m": rc, "z0_m": 0.0, "z1_m": 300.0})
    body = AnomalyDef(1, "dense_body", 1.0,
                      {"cx_m": d, "cy_m": 0.0, "cz_m": 150.0,
                       "ax_m": a_body, "ay_m": a_body, "az_m": 50.0})
    res = b3_minimal_displacement(conduit, body)
    expected_s = (rc + a_body) - d  # circular special case: touch when centers rc+a_body apart
    ok = abs(res["s_star_m"] - expected_s) < 1e-4
    print(f"  B3 displacement (circular case): s*={res['s_star_m']:.6f} "
         f"expected={expected_s:.6f} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_conduit_containment_g4_apex_radius() -> bool:
    """Pins the G4 cone convention (apex radius 0.1 m, DetectorConstruction.cc)
    independently of a full run diffed against the professor's audit: a
    conduit whose top just protrudes past the zero-apex analytic cone but
    is contained once the 0.1 m G4 apex radius is used."""
    R, H = 50.0, 100.0
    conduit = AnomalyDef(0, "conduit", 1.0,
                         {"cx_m": 0.0, "cy_m": 0.0, "radius_m": 0.05,
                          "z0_m": 0.0, "z1_m": H})
    row_analytic = conduit_containment(conduit, R, H, apex_r=0.0, label="analytic_cone")
    row_g4 = conduit_containment(conduit, R, H, apex_r=APEX_R_G4, label="geant4_cone")
    ok = (row_analytic["state"] == "lateral_protrusion"
         and row_g4["state"] == "contained")
    print(f"  conduit containment vs cone apex radius: analytic={row_analytic['state']} "
         f"g4(apex={APEX_R_G4})={row_g4['state']} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_sphere_containment_g4_apex_radius() -> bool:
    """Same as above for sphere_containment: a sphere near the apex that
    protrudes under the zero-apex analytic cone but is contained once the
    0.1 m G4 apex radius is used."""
    R, H = 50.0, 100.0
    sph = AnomalyDef(0, "cavity", 1.0, {"cx_m": 0.0, "cy_m": 0.0, "cz_m": 99.9, "radius_m": 0.05})
    row_analytic = sphere_containment(sph, R, H, apex_r=0.0, label="analytic_cone")
    row_g4 = sphere_containment(sph, R, H, apex_r=APEX_R_G4, label="geant4_cone")
    ok = (row_analytic["state"] == "lateral_protrusion"
         and row_g4["state"] == "contained")
    print(f"  sphere containment vs cone apex radius: analytic={row_analytic['state']} "
         f"g4(apex={APEX_R_G4})={row_g4['state']} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_compound_containment_flags_are_independent() -> bool:
    """A conduit that both crosses the base AND protrudes laterally at its
    top must report both defects as independent flags, even though `state`
    (the mutually-exclusive summary label) can only report one of them."""
    R, H = 50.0, 100.0
    conduit = AnomalyDef(0, "conduit", 1.0,
                         {"cx_m": 30.0, "cy_m": 0.0, "radius_m": 10.0,
                          "z0_m": -5.0, "z1_m": 80.0})
    row = conduit_containment(conduit, R, H, apex_r=0.0, label="analytic_cone")
    # state picks below_base first (precedence chain), but the lateral
    # protrusion at the top must still be visible as an independent flag.
    ok = (row["state"] == "crosses_base" and row["below_base"] is True
         and row["lateral_protrusion"] is True)
    print(f"  compound defect flags: state={row['state']} below_base={row['below_base']} "
         f"lateral_protrusion={row['lateral_protrusion']} -> {'ok' if ok else 'FAIL'}")
    return ok


def _write_synthetic_geometry(catalog_dir: Path, geom_id: str, anomalies: list) -> None:
    geom = {
        "geometry_id": geom_id,
        "edifice": {"shape": "cone", "base_radius_m": 50.0, "height_m": 100.0,
                    "rock_density_g_cm3": 2.65},
        "anomalies": anomalies,
    }
    (catalog_dir / f"{geom_id}.json").write_text(json.dumps(geom), encoding="utf-8")


def test_detail_block_pair_order_independent() -> bool:
    """build_detail_block must find the conduit/dense_body pair regardless
    of which one has the lower JSON index. Here the dense_body is anomaly 0
    and the conduit is anomaly 1 (conduit.index > body.index) — exactly the
    ordering under which the old exact-order match
    `p["indices"] == [conduit.index, body.index]` would never fire, since
    audit_geometry always stores pairwise indices as [i, j] with i < j."""
    geom_id = "synthetic-reversed-order"
    anomalies = [
        {"kind": "dense_body", "density_g_cm3": 3.0,
         "params": {"cx_m": 0.0, "cy_m": 0.0, "cz_m": 50.0,
                    "ax_m": 8.0, "ay_m": 8.0, "az_m": 8.0}},
        {"kind": "conduit", "density_g_cm3": 1.8,
         "params": {"cx_m": 0.0, "cy_m": 0.0, "radius_m": 5.0,
                    "z0_m": 0.0, "z1_m": 90.0}},
    ]
    with tempfile.TemporaryDirectory() as td:
        catalog_dir = Path(td)
        _write_synthetic_geometry(catalog_dir, geom_id, anomalies)
        rng = np.random.default_rng(SEED)
        geom_result = audit_geometry(geom_id, catalog_dir, rng, n_samples=200_000, prof_audit=None)
        detail = build_detail_block(geom_result, catalog_dir, view=0, rng=rng)

    body_index, conduit_index = 0, 1
    ok = ("error" not in detail
         and detail.get("conduit_index") == conduit_index
         and detail.get("body_index") == body_index
         and sorted(detail.get("pair_intersection", {}).get("indices", []))
             == sorted([conduit_index, body_index])
         and detail.get("pair_intersection", {}).get("volume_m3", 0.0) > 0.0)
    print(f"  detail block pair order independence (body index < conduit index): "
         f"detail={'error' if 'error' in detail else detail.get('pair_intersection', {}).get('volume_m3')} "
         f"-> {'ok' if ok else 'FAIL'}")
    return ok


def test_detail_block_missing_pair_is_graceful() -> bool:
    """A geometry with no conduit (or no dense_body) must not crash
    build_detail_block/los_detail with StopIteration; it must return a
    graceful error dict instead."""
    geom_id = "synthetic-no-pair"
    anomalies = [
        {"kind": "dense_body", "density_g_cm3": 3.0,
         "params": {"cx_m": 0.0, "cy_m": 0.0, "cz_m": 50.0,
                    "ax_m": 8.0, "ay_m": 8.0, "az_m": 8.0}},
    ]
    with tempfile.TemporaryDirectory() as td:
        catalog_dir = Path(td)
        _write_synthetic_geometry(catalog_dir, geom_id, anomalies)
        rng = np.random.default_rng(SEED)
        geom_result = audit_geometry(geom_id, catalog_dir, rng, n_samples=1_000, prof_audit=None)
        detail = build_detail_block(geom_result, catalog_dir, view=0, rng=rng)
    ok = isinstance(detail, dict) and "error" in detail
    print(f"  detail block on geometry without a conduit/dense_body pair: "
         f"{detail} -> {'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("OE1 geometry audit — self-verifying tests")
    results = {
        "overlapping_spheres_lens": test_overlapping_spheres_lens(),
        "cylinder_inside_ellipsoid": test_cylinder_inside_ellipsoid(),
        "disjoint_solids": test_disjoint_solids(),
        "conduit_protrusion": test_conduit_protrusion_from_small_cone(),
        "sphere_crosses_base": test_sphere_crosses_base(),
        "los_chord_hand_calculation": test_los_chord_hand_calculation(),
        "b3_displacement_closed_form": test_b3_displacement_closed_form(),
        "conduit_containment_g4_apex_radius": test_conduit_containment_g4_apex_radius(),
        "sphere_containment_g4_apex_radius": test_sphere_containment_g4_apex_radius(),
        "compound_containment_flags_are_independent": test_compound_containment_flags_are_independent(),
        "detail_block_pair_order_independent": test_detail_block_pair_order_independent(),
        "detail_block_missing_pair_is_graceful": test_detail_block_missing_pair_is_graceful(),
    }
    passed = all(results.values())
    if not passed:
        failed = [k for k, v in results.items() if not v]
        print(f"failed checks: {', '.join(failed)}")
    print("OE1 GEOMETRY AUDIT TESTS:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
