#!/usr/bin/env python3
"""OE1 geometry audit — deterministic, seeded diagnostic for the 19-geometry
catalog that feeds both the analytic surrogate engine and the Geant4 build.

Checks per geometry:
  (a) analytic volumes of every anomaly.
  (b) containment of every anomaly inside the analytic cone (apex radius 0)
      and the Geant4 cone (apex radius 0.1 m), with an exact test for the
      conduit and the sphere and a dense surface-sampling test for the
      ellipsoid.
  (c) Monte-Carlo pairwise intersection volume (with SE) for every anomaly
      pair, restricted first to the bounding-box overlap and reported both
      raw and clipped to the Geant4 cone.
  (d) the material assigned in every non-empty pairwise intersection by the
      analytic engine, by Geant4 as received (undefined — overlapping
      siblings), by fix B1 (later anomaly wins, clipped to the edifice) and
      by alternative B2 (conduit priority).
  (e) a consistency check that the B1 CSG rule reproduces the analytic
      surrogate's density field.
  (f) a line-of-sight detail block for one geometry/view (default
      geo-train-06, view 0): chord lengths, column density under B1 and B2,
      and the minimal horizontal displacement (B3) that would remove the
      geo-train-06 overlap.
  (g) a comparison against the professor's independent audit
      (auditoria-oe1.json), geometry by geometry.

Run:
  python scripts/analysis/oe1_geometry_audit.py --out results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "simulation"))

from surrogate.geometry import VolcanoGeometry  # noqa: E402

DEFAULT_CATALOG_DIR = REPO_ROOT / "scripts" / "simulation" / "geometry_catalog" / "geometries"
DEFAULT_GEOMETRY_IDS = [
    "geo-gen-01", "geo-test-01", "geo-test-02", "geo-test-03",
    "geo-train-01", "geo-train-02", "geo-train-03", "geo-train-04",
    "geo-train-05", "geo-train-06", "geo-train-07", "geo-train-08",
    "geo-train-09", "geo-train-10", "geo-train-11", "geo-train-12",
    "geo-val-01", "geo-val-02", "geo-val-03",
]
PROFESSOR_AUDIT_PATH = (
    REPO_ROOT / "muografia-documentada" / "objetivo-1" / "evidencias"
    / "auditoria-oe1.json"
)
# The above default is convenient on a machine where the PROF repo happens to
# be cloned as a sibling directory of this (PUB) repo, but that layout is not
# part of the PUB repo itself. It is overridable via --professor-audit.

APEX_R_ANALYTIC = 0.0
APEX_R_G4 = 0.1  # meters — Geant4 G4Cons apex radius (see DetectorConstruction.cc)
ROCK_DENSITY = 2.65  # g/cm^3, all catalog edifices (edifice.rock_density_g_cm3)
AIR_MAP_THRESHOLD = 0.05  # g/cm^3 — Geant4 maps density below this to G4_AIR

ELLIPSOID_N_THETA = 1440  # 0.25 deg azimuthal step
ELLIPSOID_N_PHI = 721     # 0.25 deg polar step (0..180 deg inclusive)


# --------------------------------------------------------------------- utils
def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel_path_str(path: Path) -> str:
    """String form of `path`, relative to REPO_ROOT when possible.

    Used for path fields embedded in delivered/committed artifacts (the
    audit JSON), so they never leak the local absolute path (and the OS
    username inside it) of whoever ran the script."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def cone_radius(z, R: float, H: float, apex_r: float):
    """R(z) for the linear cone, valid for z in [0, H]."""
    return R - (R - apex_r) * z / H


def is_inside_cone(x, y, z, R: float, H: float, apex_r: float):
    r = np.sqrt(x * x + y * y)
    return (z >= 0.0) & (z <= H) & (r <= cone_radius(z, R, H, apex_r))


# ---------------------------------------------------------- anomaly primitives
@dataclass
class AnomalyDef:
    index: int
    kind: str
    density: float
    params: dict

    def bbox(self):
        p = self.params
        if self.kind == "conduit":
            r = p["radius_m"]
            return (p["cx_m"] - r, p["cx_m"] + r,
                    p["cy_m"] - r, p["cy_m"] + r,
                    p["z0_m"], p["z1_m"])
        if self.kind == "cavity":
            r = p["radius_m"]
            return (p["cx_m"] - r, p["cx_m"] + r,
                    p["cy_m"] - r, p["cy_m"] + r,
                    p["cz_m"] - r, p["cz_m"] + r)
        if self.kind == "dense_body":
            return (p["cx_m"] - p["ax_m"], p["cx_m"] + p["ax_m"],
                    p["cy_m"] - p["ay_m"], p["cy_m"] + p["ay_m"],
                    p["cz_m"] - p["az_m"], p["cz_m"] + p["az_m"])
        raise ValueError(self.kind)

    def volume(self) -> float:
        p = self.params
        if self.kind == "conduit":
            return float(np.pi * p["radius_m"] ** 2 * (p["z1_m"] - p["z0_m"]))
        if self.kind == "cavity":
            return float(4.0 / 3.0 * np.pi * p["radius_m"] ** 3)
        if self.kind == "dense_body":
            return float(4.0 / 3.0 * np.pi * p["ax_m"] * p["ay_m"] * p["az_m"])
        raise ValueError(self.kind)

    def contains(self, x, y, z):
        p = self.params
        if self.kind == "conduit":
            rr = (x - p["cx_m"]) ** 2 + (y - p["cy_m"]) ** 2
            return (rr <= p["radius_m"] ** 2) & (z >= p["z0_m"]) & (z <= p["z1_m"])
        if self.kind == "cavity":
            dd = ((x - p["cx_m"]) ** 2 + (y - p["cy_m"]) ** 2
                  + (z - p["cz_m"]) ** 2)
            return dd <= p["radius_m"] ** 2
        if self.kind == "dense_body":
            nn = (((x - p["cx_m"]) / p["ax_m"]) ** 2
                  + ((y - p["cy_m"]) / p["ay_m"]) ** 2
                  + ((z - p["cz_m"]) / p["az_m"]) ** 2)
            return nn <= 1.0
        raise ValueError(self.kind)

    def margin(self, x, y, z):
        """1 at the center, 0 at the surface, negative outside."""
        p = self.params
        if self.kind == "conduit":
            radial = 1.0 - ((x - p["cx_m"]) ** 2 + (y - p["cy_m"]) ** 2) / p["radius_m"] ** 2
            zc = (p["z0_m"] + p["z1_m"]) / 2.0
            zh = (p["z1_m"] - p["z0_m"]) / 2.0
            axial = 1.0 - np.abs(z - zc) / zh
            return np.minimum(radial, axial)
        if self.kind == "cavity":
            dd = ((x - p["cx_m"]) ** 2 + (y - p["cy_m"]) ** 2
                  + (z - p["cz_m"]) ** 2)
            return 1.0 - dd / p["radius_m"] ** 2
        if self.kind == "dense_body":
            nn = (((x - p["cx_m"]) / p["ax_m"]) ** 2
                  + ((y - p["cy_m"]) / p["ay_m"]) ** 2
                  + ((z - p["cz_m"]) / p["az_m"]) ** 2)
            return 1.0 - nn
        raise ValueError(self.kind)


def load_anomalies(cfg: dict) -> list[AnomalyDef]:
    return [AnomalyDef(i, a["kind"], a["density_g_cm3"], a["params"])
            for i, a in enumerate(cfg.get("anomalies", []))]


# --------------------------------------------------------- containment checks
def _point_seg_dist(px, pz, ax, az, bx, bz):
    dx, dz = bx - ax, bz - az
    L2 = dx * dx + dz * dz
    if L2 <= 1e-15:
        t = 0.0
    else:
        t = ((px - ax) * dx + (pz - az) * dz) / L2
        t = min(1.0, max(0.0, t))
    cx, cz = ax + t * dx, az + t * dz
    return np.hypot(px - cx, pz - cz), cx, cz


def cone_boundary_distance(rho: float, z: float, R: float, H: float, apex_r: float):
    """Unsigned distance from (rho, z) to the physical boundary (base disk +
    lateral surface + top cap for G4) of the solid of revolution, plus the
    'inside' flag. Excludes the symmetry axis (not a physical boundary)."""
    edges = [((0.0, 0.0), (R, 0.0))]          # base
    edges.append(((R, 0.0), (apex_r, H)))      # lateral
    if apex_r > 1e-12:
        edges.append(((apex_r, H), (0.0, H)))  # top cap (G4 only)
    best = None
    for (ax_, az_), (bx_, bz_) in edges:
        d, _, _ = _point_seg_dist(rho, z, ax_, az_, bx_, bz_)
        if best is None or d < best:
            best = d
    inside = (0.0 <= z <= H) and (rho <= cone_radius(np.array(z), R, H, apex_r))
    return best, bool(inside)


def conduit_containment(a: AnomalyDef, R: float, H: float, apex_r: float, label: str):
    p = a.params
    rho0 = float(np.hypot(p["cx_m"], p["cy_m"]) + p["radius_m"])
    z_top = min(p["z1_m"], H)
    margin = cone_radius(z_top, R, H, apex_r) - rho0
    above_apex = p["z1_m"] > H
    below_base = p["z0_m"] < 0.0
    lateral_protrusion = margin < 0.0
    # `state` is a single mutually-exclusive label (kept for backward
    # compatibility with callers that group defects into one bucket); the
    # three flags above are computed independently of each other and of
    # `state`, so a compound defect (e.g. both below_base and a lateral
    # protrusion at the same time) is never silently dropped.
    if margin >= 0 and not above_apex and not below_base:
        state = "contained"
    elif below_base:
        state = "crosses_base"
    elif above_apex:
        state = "above_apex"
    else:
        state = "lateral_protrusion"
    theta = np.arctan2(p["cy_m"], p["cx_m"]) if rho0 > 0 else 0.0
    witness = [p["cx_m"] + p["radius_m"] * np.cos(theta),
               p["cy_m"] + p["radius_m"] * np.sin(theta), z_top]
    return {
        "against": label, "kind": "conduit", "state": state,
        "max_radial_excess_m": float(-margin) if margin < 0 else 0.0,
        "excess_height_m": z_top if margin < 0 else None,
        "margin_m": float(margin),
        "above_apex": bool(above_apex), "below_base": bool(below_base),
        "lateral_protrusion": bool(lateral_protrusion),
        "witness_m": witness,
    }


def sphere_containment(a: AnomalyDef, R: float, H: float, apex_r: float, label: str):
    p = a.params
    rho_c = float(np.hypot(p["cx_m"], p["cy_m"]))
    r = p["radius_m"]
    d, inside = cone_boundary_distance(rho_c, p["cz_m"], R, H, apex_r)
    signed = d if inside else -d
    margin = signed - r
    below_base = (p["cz_m"] - r) < 0.0
    above_apex = (p["cz_m"] + r) > H
    lateral_protrusion = margin < 0.0
    # See the comment in conduit_containment: `state` is exclusive, the
    # flags below are independent so a compound defect is never dropped.
    if margin >= 0:
        state = "contained"
    elif below_base:
        state = "crosses_base"
    elif above_apex:
        state = "above_apex"
    else:
        state = "lateral_protrusion"
    return {
        "against": label, "kind": "cavity", "state": state,
        "boundary_distance_m": float(d), "inside_flag": inside,
        "margin_m": float(margin),
        "min_z_m": float(p["cz_m"] - r), "max_z_m": float(p["cz_m"] + r),
        "below_base": bool(below_base), "above_apex": bool(above_apex),
        "lateral_protrusion": bool(lateral_protrusion),
    }


def ellipsoid_containment(a: AnomalyDef, R: float, H: float, apex_r: float, label: str,
                          n_theta: int = ELLIPSOID_N_THETA, n_phi: int = ELLIPSOID_N_PHI):
    p = a.params
    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0.0, np.pi, n_phi)
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    x = p["cx_m"] + p["ax_m"] * np.sin(ph) * np.cos(th)
    y = p["cy_m"] + p["ay_m"] * np.sin(ph) * np.sin(th)
    z = p["cz_m"] + p["az_m"] * np.cos(ph)
    r = np.hypot(x, y)
    within_h = (z >= 0.0) & (z <= H)
    rc = np.where(within_h, cone_radius(np.clip(z, 0.0, H), R, H, apex_r), np.nan)
    excess = np.where(within_h, r - rc, -np.inf)
    max_excess = float(np.nanmax(excess))
    idx = int(np.nanargmax(excess))
    excess_height = float(z.flat[idx])
    min_z = float(np.min(z))
    max_z = float(np.max(z))
    below_base = min_z < 0.0
    above_apex = max_z > H
    lateral = max_excess > 0.0
    # See the comment in conduit_containment: `state` is exclusive, the
    # flags below (below_base, above_apex, lateral_protrusion) are
    # independent so a compound defect is never dropped.
    if not lateral and not below_base and not above_apex:
        state = "contained"
    elif below_base:
        state = "crosses_base"
    elif above_apex:
        state = "above_apex"
    else:
        state = "lateral_protrusion"
    return {
        "against": label, "kind": "dense_body", "state": state,
        "max_radial_excess_m": max(0.0, max_excess),
        "excess_height_m": excess_height if lateral else None,
        "min_z_m": min_z, "max_z_m": max_z,
        "below_base": bool(below_base), "above_apex": bool(above_apex),
        "lateral_protrusion": bool(lateral),
        "surface_resolution": {"n_theta": n_theta, "n_phi": n_phi,
                               "approx_deg_step": 360.0 / n_theta},
    }


def containment_report(a: AnomalyDef, R: float, H: float):
    checks = []
    for apex_r, label in ((APEX_R_G4, "geant4_cone"), (APEX_R_ANALYTIC, "analytic_cone")):
        if a.kind == "conduit":
            checks.append(conduit_containment(a, R, H, apex_r, label))
        elif a.kind == "cavity":
            checks.append(sphere_containment(a, R, H, apex_r, label))
        elif a.kind == "dense_body":
            checks.append(ellipsoid_containment(a, R, H, apex_r, label))
    return checks


# --------------------------------------------------------------- certificates
def bbox_disjoint(b1, b2) -> bool:
    for lo1, hi1, lo2, hi2 in ((b1[0], b1[1], b2[0], b2[1]),
                               (b1[2], b1[3], b2[2], b2[3]),
                               (b1[4], b1[5], b2[4], b2[5])):
        if hi1 < lo2 or hi2 < lo1:
            return True
    return False


def sphere_sphere_separation(a: AnomalyDef, b: AnomalyDef) -> Optional[float]:
    if a.kind != "cavity" or b.kind != "cavity":
        return None
    pa, pb = a.params, b.params
    d = np.hypot(pa["cx_m"] - pb["cx_m"], pa["cy_m"] - pb["cy_m"], )
    d = float(np.hypot(d, pa["cz_m"] - pb["cz_m"]))
    return d - pa["radius_m"] - pb["radius_m"]


def sphere_cylinder_separation(sph: AnomalyDef, cyl: AnomalyDef) -> Optional[float]:
    if sph.kind != "cavity" or cyl.kind != "conduit":
        return None
    ps, pc = sph.params, cyl.params
    rho_p = float(np.hypot(ps["cx_m"] - pc["cx_m"], ps["cy_m"] - pc["cy_m"]))
    dr = max(0.0, rho_p - pc["radius_m"])
    dz = max(0.0, pc["z0_m"] - ps["cz_m"], ps["cz_m"] - pc["z1_m"])
    dist_to_cyl = float(np.hypot(dr, dz))
    return dist_to_cyl - ps["radius_m"]


def separation_certificate(a: AnomalyDef, b: AnomalyDef, box_a, box_b):
    if bbox_disjoint(box_a, box_b):
        return {"kind": "disjoint_bounding_boxes"}
    s = sphere_sphere_separation(a, b)
    if s is None:
        s = sphere_sphere_separation(b, a)
    if s is not None:
        if s > 0:
            return {"kind": "sphere_sphere_distance", "separation_m": s}
        return None
    s = sphere_cylinder_separation(a, b)
    if s is None:
        s = sphere_cylinder_separation(b, a)
    if s is not None:
        if s > 0:
            return {"kind": "sphere_cylinder_distance", "separation_m": s}
        return None
    return None


# --------------------------------------------------------- pairwise MC (part c)
def mc_pair_intersection(rng: np.random.Generator, a: AnomalyDef, b: AnomalyDef,
                         R: float, H: float, n_samples: int):
    box_a, box_b = a.bbox(), b.bbox()
    cert = separation_certificate(a, b, box_a, box_b)
    if bbox_disjoint(box_a, box_b):
        return {
            "indices": [a.index, b.index], "kinds": [a.kind, b.kind],
            "bbox_disjoint": True, "k": 0, "n_samples": 0,
            "volume_m3": 0.0, "se_m3": 0.0,
            "certificate": cert, "note": "bounding boxes disjoint — exact zero",
        }
    lo = [max(box_a[0], box_b[0]), max(box_a[2], box_b[2]), max(box_a[4], box_b[4])]
    hi = [min(box_a[1], box_b[1]), min(box_a[3], box_b[3]), min(box_a[5], box_b[5])]
    vol_box = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])
    pts = rng.uniform(low=lo, high=hi, size=(n_samples, 3))
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    inside_a = a.contains(x, y, z)
    inside_b = b.contains(x, y, z)
    both = inside_a & inside_b
    k = int(np.count_nonzero(both))
    p_hat = k / n_samples
    volume = vol_box * p_hat
    se = vol_box * np.sqrt(max(p_hat * (1 - p_hat), 0.0) / n_samples)

    inside_g4 = is_inside_cone(x, y, z, R, H, APEX_R_G4)
    both_g4 = both & inside_g4
    k_g4 = int(np.count_nonzero(both_g4))
    volume_g4 = vol_box * (k_g4 / n_samples)
    se_g4 = vol_box * np.sqrt(max((k_g4 / n_samples) * (1 - k_g4 / n_samples), 0.0) / n_samples)

    result = {
        "indices": [a.index, b.index], "kinds": [a.kind, b.kind],
        "bbox_disjoint": False, "k": k, "n_samples": n_samples,
        "bbox_volume_m3": float(vol_box),
        "volume_m3": float(volume), "se_m3": float(se),
        "volume_inside_g4cone_m3": float(volume_g4), "se_inside_g4cone_m3": float(se_g4),
        "certificate": cert,
        "hit_points_xyz": pts[both] if k > 0 else None,
    }
    if k > 0:
        hx, hy, hz = x[both], y[both], z[both]
        result["z_min_m"] = float(np.min(hz))
        result["z_max_m"] = float(np.max(hz))
        result["centroid_m"] = [float(np.mean(hx)), float(np.mean(hy)), float(np.mean(hz))]
        result["fraction_of_a"] = volume / a.volume()
        result["fraction_of_b"] = volume / b.volume()
        ma = a.margin(hx, hy, hz)
        mb = b.margin(hx, hy, hz)
        min_margin = np.minimum(ma, mb)
        wi = int(np.argmax(min_margin))
        result["witness_m"] = [float(hx[wi]), float(hy[wi]), float(hz[wi])]
        result["witness_margin_a"] = float(ma[wi])
        result["witness_margin_b"] = float(mb[wi])
        result["witness_min_margin"] = float(min_margin[wi])
    else:
        result["note"] = ("no intersection detected at N — " +
                          ("separation certificate below" if cert is not None
                           else "no analytic separation certificate available; "
                                "absence of a hit does not certify separation"))
    return result


# ------------------------------------------------------------ material rules
def material_assignment(anomalies: list[AnomalyDef], i: int, j: int):
    """Material assigned in the (i, j) overlap region by each engine.
    i < j is assumed (JSON order)."""
    lo, hi = min(i, j), max(i, j)
    analytic = anomalies[hi].density  # later index wins (surrogate loop order)
    b1 = anomalies[hi].density        # B1 = later anomaly wins (locked decision)
    conduit_idx = None
    for idx in (i, j):
        if anomalies[idx].kind == "conduit":
            conduit_idx = idx
            break
    b2 = anomalies[conduit_idx].density if conduit_idx is not None else anomalies[lo].density
    return {
        "analytic_engine_density_g_cm3": analytic,
        "geant4_as_built": "undefined (overlapping G4PVPlacement siblings; "
                          "navigator/overlap resolution is not deterministic by design)",
        "b1_density_g_cm3": b1,
        "b2_density_g_cm3": b2,
        "b2_rule": ("conduit priority" if conduit_idx is not None
                   else "earlier index (no conduit in this pair)"),
    }


# ------------------------------------------------------------- consistency (e)
def b1_density_field(x, y, z, anomalies: list[AnomalyDef], R: float, H: float):
    inside_g4 = is_inside_cone(x, y, z, R, H, APEX_R_G4)
    rho = np.where(inside_g4, ROCK_DENSITY, 1.225e-3).astype(np.float64)
    for a in anomalies:
        mask = a.contains(x, y, z) & inside_g4
        rho = np.where(mask, a.density, rho)
    return rho, inside_g4


def consistency_check(geo: VolcanoGeometry, anomalies: list[AnomalyDef], R: float, H: float,
                      pair_hit_points: list[np.ndarray], rng: np.random.Generator,
                      n_uniform: int):
    pts_list = [p for p in pair_hit_points if p is not None and len(p)]
    if anomalies:
        lo = np.min([a.bbox()[0::2] for a in anomalies], axis=0)
        hi = np.max([a.bbox()[1::2] for a in anomalies], axis=0)
        lo = [min(lo[0], -1.0), min(lo[1], -1.0), max(lo[2], 0.0)]
        hi = [max(hi[0], 1.0), max(hi[1], 1.0), min(hi[2], H)]
        uniform_pts = rng.uniform(low=lo, high=hi, size=(n_uniform, 3))
        pts_list.append(uniform_pts)
    if not pts_list:
        return {"n_points": 0, "n_mismatch": 0, "classification": {}}
    pts = np.concatenate(pts_list, axis=0)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    analytic_rho = geo.density(x, y, z)
    b1_rho, inside_g4 = b1_density_field(x, y, z, anomalies, R, H)
    inside_analytic = is_inside_cone(x, y, z, R, H, APEX_R_ANALYTIC)

    mismatch = np.abs(analytic_rho - b1_rho) > 1e-9
    n_mismatch = int(np.count_nonzero(mismatch))

    apex_sliver = inside_g4 & (~inside_analytic)
    air_map = (np.abs(analytic_rho) < AIR_MAP_THRESHOLD) | (np.abs(b1_rho) < AIR_MAP_THRESHOLD)

    classified_apex = int(np.count_nonzero(mismatch & apex_sliver))
    classified_air = int(np.count_nonzero(mismatch & (~apex_sliver) & air_map))
    unexplained = int(np.count_nonzero(mismatch & (~apex_sliver) & (~air_map)))

    return {
        "n_points": int(len(pts)),
        "n_mismatch": n_mismatch,
        "classification": {
            "expected_apex_sliver_m_0_1": classified_apex,
            "expected_air_density_mapping": classified_air,
            "unexplained": unexplained,
        },
    }


# --------------------------------------------------------------- ray geometry
def solve_quadratic_leq0(a: float, b: float, c: float):
    """Intervals of t where a*t^2 + b*t + c <= 0."""
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return [(-np.inf, np.inf)] if c <= 0 else []
        t0 = -c / b
        return [(-np.inf, t0)] if b > 0 else [(t0, np.inf)]
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    sq = np.sqrt(disc)
    t1, t2 = sorted([(-b - sq) / (2 * a), (-b + sq) / (2 * a)])
    mid = (t1 + t2) / 2
    if a * mid * mid + b * mid + c <= 0:
        return [(t1, t2)]
    return [(-np.inf, t1), (t2, np.inf)]


def intersect_interval_lists(list_a, list_b):
    out = []
    for lo1, hi1 in list_a:
        for lo2, hi2 in list_b:
            lo, hi = max(lo1, lo2), min(hi1, hi2)
            if lo < hi:
                out.append((lo, hi))
    return out


def bounded(intervals):
    finite = [(lo, hi) for lo, hi in intervals if np.isfinite(lo) and np.isfinite(hi)]
    return finite


def ray_cone_chord(D, u, R: float, H: float, apex_r: float):
    k = (R - apex_r) / H
    R0 = R - k * D[2]
    kr = k * u[2]
    a_lat = u[0] ** 2 + u[1] ** 2 - kr ** 2
    b_lat = 2 * (D[0] * u[0] + D[1] * u[1] + R0 * kr)
    c_lat = D[0] ** 2 + D[1] ** 2 - R0 ** 2
    lat_intervals = solve_quadratic_leq0(a_lat, b_lat, c_lat)
    if abs(u[2]) < 1e-12:
        z_intervals = [(-np.inf, np.inf)] if 0.0 <= D[2] <= H else []
    else:
        ta, tb = (0.0 - D[2]) / u[2], (H - D[2]) / u[2]
        z_intervals = [(min(ta, tb), max(ta, tb))]
    combined = bounded(intersect_interval_lists(lat_intervals, z_intervals))
    if not combined:
        return None
    lo = min(c[0] for c in combined)
    hi = max(c[1] for c in combined)
    return lo, hi


def ray_cylinder_chord(D, u, cx, cy, r, z0, z1):
    a = u[0] ** 2 + u[1] ** 2
    b = 2 * ((D[0] - cx) * u[0] + (D[1] - cy) * u[1])
    c = (D[0] - cx) ** 2 + (D[1] - cy) ** 2 - r ** 2
    radial = solve_quadratic_leq0(a, b, c)
    if abs(u[2]) < 1e-12:
        z_iv = [(-np.inf, np.inf)] if z0 <= D[2] <= z1 else []
    else:
        ta, tb = (z0 - D[2]) / u[2], (z1 - D[2]) / u[2]
        z_iv = [(min(ta, tb), max(ta, tb))]
    combined = bounded(intersect_interval_lists(radial, z_iv))
    if not combined:
        return None
    return min(c[0] for c in combined), max(c[1] for c in combined)


def ray_ellipsoid_chord(D, u, cx, cy, cz, ax, ay, az):
    dx0, dy0, dz0 = D[0] - cx, D[1] - cy, D[2] - cz
    a = (u[0] / ax) ** 2 + (u[1] / ay) ** 2 + (u[2] / az) ** 2
    b = 2 * (dx0 * u[0] / ax ** 2 + dy0 * u[1] / ay ** 2 + dz0 * u[2] / az ** 2)
    c = (dx0 / ax) ** 2 + (dy0 / ay) ** 2 + (dz0 / az) ** 2 - 1.0
    iv = bounded(solve_quadratic_leq0(a, b, c))
    if not iv:
        return None
    return min(c[0] for c in iv), max(c[1] for c in iv)


def los_detail(geo_cfg: dict, anomalies: list[AnomalyDef], view: int):
    """Returns (det, R, H, conduit, body), or None if this geometry does not
    have both a conduit and a dense_body anomaly (the LOS/B3 detail block
    only makes sense for that pair)."""
    R, H = geo_cfg["edifice"]["base_radius_m"], geo_cfg["edifice"]["height_m"]
    conduit = next((a for a in anomalies if a.kind == "conduit"), None)
    body = next((a for a in anomalies if a.kind == "dense_body"), None)
    if conduit is None or body is None:
        return None

    from math import cos, sin, radians
    det = np.array([1000.0 * cos(radians(view * 60.0)),
                    1000.0 * sin(radians(view * 60.0)), 1.5])
    return det, R, H, conduit, body


def compute_los_block(det, target, R, H, conduit: AnomalyDef, body: AnomalyDef):
    u = (target - det)
    u = u / np.linalg.norm(u)
    cone_iv = ray_cone_chord(det, u, R, H, APEX_R_ANALYTIC)
    pc, pcp = conduit.params, body.params
    cond_iv = ray_cylinder_chord(det, u, pc["cx_m"], pc["cy_m"], pc["radius_m"],
                                 pc["z0_m"], pc["z1_m"])
    body_iv = ray_ellipsoid_chord(det, u, pcp["cx_m"], pcp["cy_m"], pcp["cz_m"],
                                  pcp["ax_m"], pcp["ay_m"], pcp["az_m"])

    def L(iv):
        return 0.0 if iv is None else max(0.0, iv[1] - iv[0])

    L_cone = L(cone_iv)
    L_cond = L(cond_iv)
    L_body = L(body_iv)
    if cond_iv is not None and body_iv is not None:
        lo = max(cond_iv[0], body_iv[0])
        hi = min(cond_iv[1], body_iv[1])
        L_int = max(0.0, hi - lo)
    else:
        L_int = 0.0
    L_cond_only = L_cond - L_int
    L_body_only = L_body - L_int
    L_rock_only = L_cone - L_cond_only - L_body_only - L_int

    rho_rock, rho_cond, rho_body = ROCK_DENSITY, conduit.density, body.density
    X_b1 = (rho_rock * L_rock_only + rho_cond * L_cond_only
            + rho_body * (L_body_only + L_int)) * 100.0  # m*g/cm3 -> g/cm2
    X_b2 = (rho_rock * L_rock_only + rho_body * L_body_only
            + rho_cond * (L_cond_only + L_int)) * 100.0
    diff = X_b1 - X_b2
    rel = diff / X_b1 if X_b1 != 0 else float("nan")

    return {
        "detector_pos_m": det.tolist(),
        "target_centroid_m": target.tolist(),
        "direction_unit": u.tolist(),
        "chord_rock_total_m": L_cone,
        "chord_conduit_total_m": L_cond,
        "chord_body_total_m": L_body,
        "chord_conduit_only_m": L_cond_only,
        "chord_body_only_m": L_body_only,
        "chord_intersection_m": L_int,
        "chord_rock_only_m": L_rock_only,
        "column_density_b1_g_cm2": X_b1,
        "column_density_b2_g_cm2": X_b2,
        "column_density_diff_g_cm2": diff,
        "column_density_rel_diff": rel,
        "note": "straight-line estimate: no scattering, no detection efficiency.",
    }


# ------------------------------------------------------ B3 minimal displacement
def _ellipse_point_region_distance(qx, qy, a, b):
    """Distance from point q=(qx,qy) to the axis-aligned ellipse with
    semi-axes (a, b) centered at the origin. Returns (distance, inside).
    distance is the boundary distance; a point inside gets distance 0 for
    'distance to the filled region' purposes (caller decides)."""
    px, py = abs(qx), abs(qy)
    swap = a < b
    if swap:
        a, b = b, a
        px, py = py, px
    if b < 1e-12:
        # degenerate; treat as a segment on the x-axis
        inside = (px <= a) and (py <= 1e-9)
        return (0.0 if inside else np.hypot(max(0.0, px - a), py)), inside

    inside = (px / a) ** 2 + (py / b) ** 2 <= 1.0
    if inside:
        # Distance to the FILLED region is zero; boundary distance not needed.
        return 0.0, True
    if px < 1e-12 and py < 1e-12:
        return b, False  # degenerate: point at center (already excluded by inside)

    a2, b2 = a * a, b * b

    def F(t):
        return ((a * px / (t + a2)) ** 2 + (b * py / (t + b2)) ** 2) - 1.0

    # F is monotonically decreasing for t > -b^2; the point is outside, so
    # F(0) = (px/a)^2+(py/b)^2 - 1 > 0 and F(t) -> -1 as t -> inf.
    lo, hi = 0.0, max(a * px, b * py, 1.0)
    while F(hi) > 0:
        hi *= 2.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if F(mid) > 0:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    x0 = a2 * px / (t + a2)
    y0 = b2 * py / (t + b2)
    dist = float(np.hypot(px - x0, py - y0))
    return dist, False


def b3_minimal_displacement(conduit: AnomalyDef, body: AnomalyDef):
    pc, pb = conduit.params, body.params
    ox, oy, r = pc["cx_m"], pc["cy_m"], pc["radius_m"]
    ex0, ey0, ax_, ay_ = pb["cx_m"], pb["cy_m"], pb["ax_m"], pb["ay_m"]
    dvec = np.array([ex0 - ox, ey0 - oy])
    norm = np.linalg.norm(dvec)
    d_unit = dvec / norm if norm > 1e-9 else np.array([1.0, 0.0])

    def gap(s: float) -> float:
        ex, ey = ex0 + s * d_unit[0], ey0 + s * d_unit[1]
        qx, qy = ox - ex, oy - ey
        dist, inside = _ellipse_point_region_distance(qx, qy, ax_, ay_)
        region_dist = 0.0 if inside else dist
        return region_dist - r

    s_lo, s_hi = 0.0, max(ax_, ay_) + r + norm + 10.0
    g_lo = gap(s_lo)
    tries = 0
    while gap(s_hi) < 0 and tries < 60:
        s_hi *= 1.5
        tries += 1
    g_hi = gap(s_hi)
    if g_lo >= 0:
        return {"s_star_m": 0.0, "note": "already separated at s=0 (unexpected)"}
    for _ in range(80):
        mid = (s_lo + s_hi) / 2
        if gap(mid) < 0:
            s_lo = mid
        else:
            s_hi = mid
    s_star = (s_lo + s_hi) / 2
    return {
        "direction_unit_xy": d_unit.tolist(),
        "s_star_m": float(s_star),
        "gap_at_s_star_m": float(gap(s_star)),
        "gap_before_s_star_m": float(gap(max(0.0, s_star - 0.05))),
        "new_body_center_xy_m": [ex0 + s_star * d_unit[0], ey0 + s_star * d_unit[1]],
        "cross_section_z_m": pb["cz_m"],
    }


def b3_mc_verify(conduit: AnomalyDef, body: AnomalyDef, s_star: float, d_unit,
                 rng: np.random.Generator, n: int = 200_000):
    pc, pb = conduit.params, body.params
    z = pb["cz_m"]

    def count_overlap(s):
        ex = pb["cx_m"] + s * d_unit[0]
        ey = pb["cy_m"] + s * d_unit[1]
        lo_x = min(pc["cx_m"] - pc["radius_m"], ex - pb["ax_m"])
        hi_x = max(pc["cx_m"] + pc["radius_m"], ex + pb["ax_m"])
        lo_y = min(pc["cy_m"] - pc["radius_m"], ey - pb["ay_m"])
        hi_y = max(pc["cy_m"] + pc["radius_m"], ey + pb["ay_m"])
        pts = rng.uniform([lo_x, lo_y], [hi_x, hi_y], size=(n, 2))
        in_circle = (pts[:, 0] - pc["cx_m"]) ** 2 + (pts[:, 1] - pc["cy_m"]) ** 2 <= pc["radius_m"] ** 2
        in_ellipse = ((pts[:, 0] - ex) / pb["ax_m"]) ** 2 + ((pts[:, 1] - ey) / pb["ay_m"]) ** 2 <= 1.0
        return int(np.count_nonzero(in_circle & in_ellipse))

    at_star = count_overlap(s_star)
    before_star = count_overlap(max(0.0, s_star - 0.5))
    return {"overlap_count_at_s_star": at_star, "overlap_count_before_s_star": before_star,
            "n_samples": n}


# ------------------------------------------------------------ audit comparison
def load_professor_audit(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def professor_defects_for(geom_id: str, prof_audit: dict):
    if prof_audit is None:
        return None
    row = next((g for g in prof_audit["geometrias"] if g["geometria"] == geom_id), None)
    if row is None:
        return None
    protrusions = [c["indice"] for c in row["contencion"]
                   if c["estado"] == "salida lateral del cono demostrada"]
    base_crossings = [c["indice"] for c in row["contencion"]
                      if c["estado"] == "salida del cono demostrada bajo la base"]
    pairs = [tuple(p["indices"]) for p in row["pares"]
             if p["estado"] == "intersección demostrada con testigo interior"]
    return {"protrusions": protrusions, "base_crossings": base_crossings,
            "intersecting_pairs": pairs}


def our_defects_for(containment_results, pair_results):
    protrusions = [c["index"] for c in containment_results
                   if c["against"] == "geant4_cone" and c["state"] == "lateral_protrusion"]
    base_crossings = [c["index"] for c in containment_results
                      if c["against"] == "geant4_cone" and c["state"] == "crosses_base"]
    pairs = [tuple(p["indices"]) for p in pair_results if p.get("k", 0) > 0]
    return {"protrusions": sorted(set(protrusions)), "base_crossings": sorted(set(base_crossings)),
            "intersecting_pairs": pairs}


# ------------------------------------------------------------------ per-geom
def audit_geometry(geom_id: str, catalog_dir: Path, rng: np.random.Generator,
                   n_samples: int, prof_audit):
    path = catalog_dir / f"{geom_id}.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    R, H = cfg["edifice"]["base_radius_m"], cfg["edifice"]["height_m"]
    anomalies = load_anomalies(cfg)
    geo = VolcanoGeometry.from_json(path)

    volumes = [{"index": a.index, "kind": a.kind, "density_g_cm3": a.density,
               "volume_m3": a.volume()} for a in anomalies]

    containment = []
    for a in anomalies:
        for row in containment_report(a, R, H):
            row["index"] = a.index
            containment.append(row)

    pairs = []
    hit_pts = []
    for i in range(len(anomalies)):
        for j in range(i + 1, len(anomalies)):
            res = mc_pair_intersection(rng, anomalies[i], anomalies[j], R, H, n_samples)
            hit_pts.append(res.pop("hit_points_xyz", None))
            if res["k"] > 0:
                res["material_assignment"] = material_assignment(anomalies, i, j)
            pairs.append(res)

    consistency = consistency_check(geo, anomalies, R, H, hit_pts, rng, n_samples)

    ours = our_defects_for(containment, pairs)
    prof = professor_defects_for(geom_id, prof_audit)
    if prof is not None:
        agree = {
            "protrusions": sorted(set(ours["protrusions"])) == sorted(set(prof["protrusions"])),
            "base_crossings": sorted(set(ours["base_crossings"])) == sorted(set(prof["base_crossings"])),
            "intersecting_pairs": (sorted(set(map(tuple, ours["intersecting_pairs"])))
                                   == sorted(set(map(tuple, prof["intersecting_pairs"])))),
        }
    else:
        agree = None

    return {
        "geometry_id": geom_id,
        "edifice": {"base_radius_m": R, "height_m": H,
                   "rock_density_g_cm3": cfg["edifice"]["rock_density_g_cm3"]},
        "n_anomalies": len(anomalies),
        "volumes": volumes,
        "containment": containment,
        "pairs": pairs,
        "consistency_b1_vs_analytic": consistency,
        "our_defects": ours,
        "professor_defects": prof,
        "audit_agreement": agree,
    }


# ------------------------------------------------------------------ main CLI
def build_detail_block(geom_result: dict, catalog_dir: Path, view: int,
                       rng: np.random.Generator):
    geom_id = geom_result["geometry_id"]
    path = catalog_dir / f"{geom_id}.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    anomalies = load_anomalies(cfg)
    los = los_detail(cfg, anomalies, view)
    if los is None:
        return {"error": f"geometry {geom_id} has no conduit/dense_body pair "
                         "for the LOS/B3 detail block"}
    det, R, H, conduit, body = los

    # Match the pair by the unordered index set: audit_geometry's pairwise
    # loop always stores indices as [i, j] with i < j (JSON order), which
    # need not equal [conduit.index, body.index] when the dense_body is
    # listed before the conduit in the JSON.
    wanted = sorted([conduit.index, body.index])
    pair01 = next((p for p in geom_result["pairs"]
                  if sorted(p["indices"]) == wanted and p["k"] > 0), None)
    if pair01 is None:
        return {"error": "no MC-detected intersection for the conduit/dense_body pair"}
    target = np.array(pair01["centroid_m"])
    los = compute_los_block(det, target, R, H, conduit, body)

    b3 = b3_minimal_displacement(conduit, body)
    b3_verify = b3_mc_verify(conduit, body, b3["s_star_m"], np.array(b3["direction_unit_xy"]), rng)

    return {
        "geometry_id": geom_id, "view": view,
        "conduit_index": conduit.index, "body_index": body.index,
        "pair_intersection": {k: v for k, v in pair01.items()},
        "line_of_sight": los,
        "b3_displacement": b3,
        "b3_mc_verification": b3_verify,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geometry-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    ap.add_argument("--geometries", nargs="*", default=None)
    ap.add_argument("--mc-samples", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--detail", type=str, default="geo-train-06")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--professor-audit", type=Path, default=PROFESSOR_AUDIT_PATH,
                    help="Path to the PROF repo's independent audit JSON "
                         "(auditoria-oe1.json). Optional: the comparison is "
                         "skipped if the file does not exist. Defaults to "
                         "the sibling-checkout layout used on this machine, "
                         "not a path tracked by this (PUB) repo.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    geom_ids = args.geometries if args.geometries else DEFAULT_GEOMETRY_IDS
    rng = np.random.default_rng(args.seed)
    prof_audit = load_professor_audit(args.professor_audit)

    catalog_sha = {}
    per_geometry = []
    for geom_id in sorted(geom_ids):
        path = args.geometry_dir / f"{geom_id}.json"
        catalog_sha[geom_id] = sha256_of(path)
        result = audit_geometry(geom_id, args.geometry_dir, rng, args.mc_samples, prof_audit)
        per_geometry.append(result)

    detail_result = next((g for g in per_geometry if g["geometry_id"] == args.detail), None)
    detail_block = None
    if detail_result is not None:
        detail_block = build_detail_block(detail_result, args.geometry_dir, args.view, rng)

    n_geo_agree = sum(1 for g in per_geometry
                      if g["audit_agreement"] and all(g["audit_agreement"].values()))
    n_geo_compared = sum(1 for g in per_geometry if g["audit_agreement"] is not None)

    output = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "script_sha256": sha256_of(Path(__file__)),
            "seed": args.seed,
            "mc_samples": args.mc_samples,
            "catalog_dir": rel_path_str(args.geometry_dir),
            "catalog_sha256": catalog_sha,
            "n_geometries": len(per_geometry),
        },
        "geometries": per_geometry,
        "detail": detail_block,
        "audit_comparison_summary": {
            "n_geometries_compared": n_geo_compared,
            "n_geometries_fully_agree": n_geo_agree,
            "professor_audit_path": rel_path_str(args.professor_audit),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print(f"OE1 geometry audit — {len(per_geometry)} geometries, "
         f"N={args.mc_samples}, seed={args.seed}")
    defective = [g["geometry_id"] for g in per_geometry
                if g["our_defects"]["protrusions"] or g["our_defects"]["base_crossings"]
                or g["our_defects"]["intersecting_pairs"]]
    print(f"defective geometries (this audit): {defective}")
    for g in per_geometry:
        if g["audit_agreement"] is not None and not all(g["audit_agreement"].values()):
            print(f"  DISAGREEMENT with professor's audit on {g['geometry_id']}: "
                 f"{g['audit_agreement']}")
    print(f"agreement with professor's audit: {n_geo_agree}/{n_geo_compared} geometries")
    if detail_block is not None and "error" not in detail_block:
        los = detail_block["line_of_sight"]
        # Read the volume from the same pair the detail block actually used
        # (matched by conduit/body index, not by its position in `pairs`):
        # for a geometry with 3+ anomalies the conduit/dense_body pair need
        # not be pairs[0].
        detail_volume = detail_block["pair_intersection"].get("volume_m3", float("nan"))
        print(f"detail {args.detail} view {args.view}: intersection volume = "
             f"{detail_volume:.1f} m3, "
             f"chord_intersection={los['chord_intersection_m']:.2f} m, "
             f"X_b1={los['column_density_b1_g_cm2']:.1f} g/cm2, "
             f"X_b2={los['column_density_b2_g_cm2']:.1f} g/cm2, "
             f"B3 s*={detail_block['b3_displacement']['s_star_m']:.3f} m")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
