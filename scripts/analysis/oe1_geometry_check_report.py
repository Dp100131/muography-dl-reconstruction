#!/usr/bin/env python3
"""OE1 geometry control report — work unit U7, plan step 3, tasks T1-T4.

Evaluates the Geant4 geometry-control campaign (variant A = received
geometry + inert P0 diagnostics, variant B = A + P1 fix B1, 19 catalog
geometries x 6 views) against the four locked criteria:

  T1 (--check-geometry, resolution 100000, tolerance 0, 228 runs):
     B must report 0 overlaps in all 114 runs; A must report overlaps in
     exactly the 6 known-defective geometries (all 6 views each) and 0
     overlaps in the other 13 x 6 = 78 runs.
  T2 (geo-train-06, --dump-geometry / --probe / --slice-y / --slice-z):
     B probes must return exactly the analytic material at every probe
     point; B anomaly volumes must fall within 3 combined standard errors
     of the expected analytic/audit values; B slices must show zero
     non-boundary, non-air material-class mismatches vs. the analytic
     density field (A slices are compared too, as a diagnostic only).
  T3: PNG maps of the two slices for A, B and the analytic engine, plus
     A-vs-analytic / B-vs-analytic mismatch maps.
  T4 (geo-train-06 view 0, 10000 events, 8 threads, normal run): B must
     print RUN DONE, zero GeomVol1002/GeomNav lines, produce a non-empty
     hist/meta pair, events_transported == 10000 and hist sum ==
     detected_in_bins (A is run too, as a reference only — A is expected
     to print GeomVol1002 at construction).

Inputs (read-only; never modified):
  - resultados-t1.tsv (variant, geometry, view, exit_code, wall_s) from the
    T1 sbatch driver.
  - control-geometrico-logs.tar.gz, containing the 228 per-(variant,
    geometry, view) --check-geometry logs (members "A/<geo>_view<v>.log",
    "B/<geo>_view<v>.log").
  - t2/{A,B}/diag.log (DUMP-VOLUME / PROBE lines) and
    t2/{A,B}/<geo>_view<v>_slice_{y,z}<coord>.csv.gz (T2 slices).
  - t4/{A,B}/<geo>_view<v>.meta.json, _hist.csv and run.log (T4 outputs).

Ground truth: the analytic surrogate engine
  scripts/simulation/surrogate/geometry.py (VolcanoGeometry.density),
same catalog JSON. Known, expected (non-defect) differences are
classified rather than counted as mismatches: air density (Geant4
G4_AIR 0.00120479 vs analytic 0.001225 g/cm3, compared as a material
class) and points within half a grid step of any boundary. The cone
apex (Geant4 apex radius 0.1 m vs analytic 0 m) is not given a
dedicated check here — unlike oe1_geometry_audit.py's is_inside_cone,
which does model both apex radii explicitly — it is only covered
incidentally by the generic near-cone-boundary tolerance above, since
0.1 m is well inside a half grid step at this delivery's 1 m grids.
For clipped anomalies other than geo-train-06 (not applicable here),
the 1 mm clip inset is similarly covered only by the generic boundary
tolerance.

Run:
  python scripts/analysis/oe1_geometry_check_report.py \
      --entrega-dir muografia-documentada/objetivo-1/entregas/2026-09-29-geo-train-06/control-geometrico \
      --out-dir    muografia-documentada/objetivo-1/entregas/2026-09-29-geo-train-06/control-geometrico
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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
EXPECTED_DEFECTIVE = {
    "geo-train-06", "geo-train-08", "geo-train-11",
    "geo-test-01", "geo-gen-01", "geo-test-03",
}
DETAIL_GEOMETRY = "geo-train-06"
DETAIL_VIEW = 0

AIR_DENSITY_ANALYTIC = 1.225e-3  # g/cm^3, surrogate.geometry.AIR_DENSITY
AIR_CLASS_THRESHOLD = 0.05       # g/cm^3 — anything below this is "air-class"
# Note: unlike oe1_geometry_audit.py, this script does not model the
# Geant4-vs-analytic cone apex radius difference (0.1 m vs 0 m) with a
# dedicated apex_r-aware cone-radius function; it relies on the generic
# near-cone-boundary half-grid-step tolerance in classify_slice_mismatches
# to absorb it (see module docstring).

# T2 volume expectations (see registro-oe1-geometria; audited figures for
# geo-train-06, variant B, fix B1 = "later anomaly wins", clipped):
EXPECTED_B_VOLUMES_M3 = {
    "anomaly_0": {"value_m3": 387_408.5, "se_m3": 44.7,
                  "note": "471050.4 - 83641.9 (raw conduit minus audited "
                          "MC overlap volume, both analytic/audit exact "
                          "up to the stated SE)"},
    "anomaly_1": {"value_m3": 1_215_148.5, "se_m3": 0.0,
                  "note": "unclipped ellipsoid, exact analytic volume"},
}
T2_VOLUME_SE_MULTIPLE = 3.0
# The diagnostic CLI prints volume_m3 with a fixed 6-significant-figure %g
# (e.g. "1.21515e+06" for an exact analytic volume of 1215148.5 m3): this is
# a display artifact, not a physical uncertainty, but it must still be
# absorbed into the tolerance for any DUMP-VOLUME value read back from a
# log, or an exact analytic volume would spuriously fail a >0-SE check.
CLI_PRINT_SIG_FIGS = 6


def print_precision_floor_m3(value_m3: float) -> float:
    """Half of the smallest printed digit's place value for a 6-significant-
    figure %g rendering of `value_m3` (0 for value_m3 == 0)."""
    if value_m3 == 0:
        return 0.0
    exponent = int(np.floor(np.log10(abs(value_m3))))
    return 0.5 * 10 ** (exponent - CLI_PRINT_SIG_FIGS + 1)

T4_EVENTS_EXPECTED = 10_000


# --------------------------------------------------------------------- utils
def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel_path_str(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT).as_posix())
    except ValueError:
        return Path(path).as_posix()


# ------------------------------------------------------------- T1: check logs
OVERLAP_LINE_RE = re.compile(
    r"OVERLAP-CHECK volume=(?P<volume>\S+) copy=(?P<copy>\d+) "
    r"solid=(?P<solid>\S+) result=(?P<result>ok|overlap)"
)
SUMMARY_LINE_RE = re.compile(
    r"OVERLAP-CHECK-SUMMARY geometry=(?P<geometry>\S+) view=(?P<view>\d+) "
    r"volumes=(?P<volumes>\d+) overlapping=(?P<overlapping>\d+) "
    r"status=(?P<status>OK|OVERLAPS)"
)


def parse_check_log(text: str) -> dict:
    """Extract the per-run overlap-check facts from one --check-geometry log."""
    n_geomvol1002 = text.count("GeomVol1002")
    n_geomnav = text.count("GeomNav")
    per_volume = [m.groupdict() for m in OVERLAP_LINE_RE.finditer(text)]
    summary_match = SUMMARY_LINE_RE.search(text)
    summary = summary_match.groupdict() if summary_match else None
    return {
        "n_GeomVol1002": n_geomvol1002,
        "n_GeomNav": n_geomnav,
        "per_volume": per_volume,
        "summary": summary,
    }


def load_t1_resultados(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            row["view"] = int(row["view"])
            row["exit_code"] = int(row["exit_code"])
            row["wall_s"] = float(row["wall_s"])
            rows.append(row)
    return rows


def build_resumen_solapes(resultados_t1: list[dict], logs_tar: Path) -> list[dict]:
    """Join the exit-code/wall-time TSV with the parsed per-run logs into the
    228-row resumen-solapes table (variant, geometry, view, exit_code,
    overlapping, volumes, n_GeomVol1002, n_GeomNav, wall_s)."""
    with tarfile.open(logs_tar, "r:gz") as tf:
        log_text = {}
        for member in tf.getmembers():
            # Members are stored as "<variant>/<geo>_view<v>.log" (the tar
            # was built with -C on the parent of A/ and B/).
            path_parts = Path(member.name).as_posix().split("/")
            if len(path_parts) < 2 or path_parts[0] not in ("A", "B"):
                continue
            variant = path_parts[0]
            fname = path_parts[-1]
            fm = re.match(r"^(.+)_view(\d+)\.log$", fname)
            if not fm:
                continue
            geo, view = fm.group(1), int(fm.group(2))
            f = tf.extractfile(member)
            if f is None:
                continue
            log_text[(variant, geo, view)] = f.read().decode("utf-8", errors="replace")

    rows = []
    for r in resultados_t1:
        key = (r["variant"], r["geometry"], r["view"])
        text = log_text.get(key)
        if text is None:
            rows.append({
                **r, "overlapping": None, "volumes": None,
                "n_GeomVol1002": None, "n_GeomNav": None,
                "status_line_present": False,
                "error": "log not found in control-geometrico-logs.tar.gz",
            })
            continue
        parsed = parse_check_log(text)
        summary = parsed["summary"]
        rows.append({
            "variant": r["variant"], "geometry": r["geometry"], "view": r["view"],
            "exit_code": r["exit_code"], "wall_s": r["wall_s"],
            "overlapping": int(summary["overlapping"]) if summary else None,
            "volumes": int(summary["volumes"]) if summary else None,
            "n_GeomVol1002": parsed["n_GeomVol1002"],
            "n_GeomNav": parsed["n_GeomNav"],
            "status_line_present": summary is not None,
        })
    rows.sort(key=lambda r: (r["variant"], r["geometry"], r["view"]))
    return rows


def evaluate_t1(resumen: list[dict]) -> dict:
    n_total = len(resumen)
    by_geo = {}
    anomalies = []
    for r in resumen:
        by_geo.setdefault(r["geometry"], {"A": set(), "B": set()})
        if r["overlapping"] is None:
            anomalies.append({"row": r, "reason": "missing OVERLAP-CHECK-SUMMARY line"})
            continue
        has_overlap = r["overlapping"] > 0
        if has_overlap:
            by_geo[r["geometry"]][r["variant"]].add(r["view"])
        # Cross-check exit code against the documented contract (exit 1 on
        # overlap, 0 otherwise) — any mismatch is itself an anomaly.
        expected_exit = 1 if has_overlap else 0
        if r["exit_code"] != expected_exit:
            anomalies.append({
                "row": r,
                "reason": f"exit_code={r['exit_code']} but overlapping="
                         f"{r['overlapping']} (expected exit {expected_exit})",
            })

    a_defective = {g for g, v in by_geo.items() if len(v["A"]) == 6}
    a_clean = {g for g, v in by_geo.items() if len(v["A"]) == 0}
    a_partial = {g for g in by_geo if g not in a_defective and g not in a_clean}
    b_with_overlaps = {g for g, v in by_geo.items() if len(v["B"]) > 0}

    for g in a_partial:
        anomalies.append({
            "row": {"geometry": g},
            "reason": f"variant A overlaps in only {sorted(by_geo[g]['A'])} of "
                     f"6 views (expected all-6 or none)",
        })

    matches_locked_criterion = (
        n_total == 228
        and not b_with_overlaps
        and a_defective == EXPECTED_DEFECTIVE
        and a_clean == (set(DEFAULT_GEOMETRY_IDS) - EXPECTED_DEFECTIVE)
        and not a_partial
        and not any(a["reason"].startswith("exit_code=") for a in anomalies)
    )

    return {
        "n_runs": n_total,
        "expected_n_runs": 228,
        "a_defective_geometries": sorted(a_defective),
        "a_clean_geometries": sorted(a_clean),
        "a_partial_geometries": sorted(a_partial),
        "b_geometries_with_overlaps": sorted(b_with_overlaps),
        "expected_defective_geometries": sorted(EXPECTED_DEFECTIVE),
        "anomalies": anomalies,
        "pass": matches_locked_criterion,
    }


# ---------------------------------------------------------------- T2: probes
DUMP_LINE_RE = re.compile(
    r"DUMP-VOLUME name=(?P<name>\S+) copy=(?P<copy>\d+) mother=(?P<mother>\S+) "
    r"logical=(?P<logical>\S+) solid=(?P<solid>\S+) type=(?P<type>\S+) "
    r"material=(?P<material>\S+) density_g_cm3=(?P<density>[-\d.eE+]+) "
    r"volume_m3=(?P<volume>[-\d.eE+]+) method=(?P<method>\S+)"
    r"(?: n=(?P<n>\d+) se_m3=(?P<se>[-\d.eE+]+))?"
)
PROBE_LINE_RE = re.compile(
    r"PROBE x_m=(?P<x>[-\d.eE+]+) y_m=(?P<y>[-\d.eE+]+) z_m=(?P<z>[-\d.eE+]+) "
    r"volume=(?P<volume>\S+) material=(?P<material>\S+) "
    r"density_g_cm3=(?P<density>[-\d.eE+]+)"
)


def parse_diag_log(text: str) -> dict:
    dumps = [m.groupdict() for m in DUMP_LINE_RE.finditer(text)]
    probes = [m.groupdict() for m in PROBE_LINE_RE.finditer(text)]
    return {"dumps": dumps, "probes": probes}


def analytic_material_at(geo: VolcanoGeometry, x: float, y: float, z: float) -> tuple[str, float]:
    """Returns (label, density) at one point, where label identifies the
    analytic material class: "air", "rock", or "anomaly_<index>" (index
    into geo.anomalies, JSON order)."""
    xa, ya, za = np.array([x]), np.array([y]), np.array([z])
    rho = float(geo.density(xa, ya, za)[0])
    r = np.hypot(x, y)
    cone_surface = geo.height_m * max(0.0, 1.0 - r / geo.base_radius_m)
    inside_cone = 0.0 <= z <= cone_surface
    if not inside_cone:
        return "air", rho
    # "later anomaly wins": walk the anomalies in the same loop order the
    # surrogate uses internally (surrogate.geometry.VolcanoGeometry.density),
    # so the last matching anomaly is the one that actually set rho.
    winner = "rock"
    for idx, a in enumerate(geo.anomalies):
        if VolcanoGeometry._anomaly_mask(a, xa, ya, za)[0]:
            winner = f"anomaly_{idx}"
    return winner, rho


def evaluate_t2_probes(diag_a: dict, diag_b: dict, geo: VolcanoGeometry) -> dict:
    rows = []
    all_b_match = True
    for pa, pb in zip(diag_a["probes"], diag_b["probes"]):
        x, y, z = float(pb["x"]), float(pb["y"]), float(pb["z"])
        expected_label, expected_density = analytic_material_at(geo, x, y, z)
        b_label = "rock" if pb["material"] == "standard_rock" else (
            "air" if pb["material"] == "G4_AIR" else
            f"anomaly_{int(pb['volume'].split('_')[-1])}" if pb["volume"].startswith("anomaly_")
            else pb["volume"]
        )
        a_label = "rock" if pa["material"] == "standard_rock" else (
            "air" if pa["material"] == "G4_AIR" else
            f"anomaly_{int(pa['volume'].split('_')[-1])}" if pa["volume"].startswith("anomaly_")
            else pa["volume"]
        )
        b_ok = (b_label == expected_label)
        all_b_match &= b_ok
        rows.append({
            "x_m": x, "y_m": y, "z_m": z,
            "analytic_material_class": expected_label,
            "analytic_density_g_cm3": expected_density,
            "a_material_class": a_label, "a_density_g_cm3": float(pa["density"]),
            "b_material_class": b_label, "b_density_g_cm3": float(pb["density"]),
            "b_matches_analytic": b_ok,
        })
    return {"rows": rows, "all_b_match_analytic": all_b_match}


def evaluate_t2_volumes(diag_b: dict) -> dict:
    by_name = {d["name"]: d for d in diag_b["dumps"] if d["name"].startswith("anomaly_")}
    rows = []
    all_within = True
    for name, expected in EXPECTED_B_VOLUMES_M3.items():
        d = by_name.get(name)
        if d is None:
            all_within = False
            rows.append({"name": name, "error": "not found in DUMP-VOLUME lines"})
            continue
        observed = float(d["volume"])
        observed_se_mc = float(d["se"]) if d.get("se") else 0.0
        print_floor = print_precision_floor_m3(observed)
        observed_se = float(np.hypot(observed_se_mc, print_floor))
        combined_se = float(np.hypot(observed_se, expected["se_m3"]))
        diff = observed - expected["value_m3"]
        n_se = abs(diff) / combined_se if combined_se > 0 else (0.0 if diff == 0 else float("inf"))
        within = n_se <= T2_VOLUME_SE_MULTIPLE
        all_within &= within
        rows.append({
            "name": name, "expected_m3": expected["value_m3"],
            "expected_se_m3": expected["se_m3"], "observed_m3": observed,
            "observed_se_mc_m3": observed_se_mc,
            "observed_print_precision_floor_m3": print_floor,
            "observed_se_m3": observed_se, "combined_se_m3": combined_se,
            "diff_m3": diff, "n_combined_se": n_se,
            "within_3_se": within, "note": expected["note"],
        })
    return {"rows": rows, "all_within_3_se": all_within}


# ----------------------------------------------------------------- T2: slices
def read_slice_csv_gz(path: Path) -> np.ndarray:
    """Reads a <geo>_view<v>_slice_{y,z}<coord>.csv.gz into a structured
    array with fields x_m, y_m, z_m, volume (str), material (str),
    density_g_cm3."""
    with gzip.open(path, "rt", newline="") as fh:
        reader = csv.DictReader(fh)
        xs, ys, zs, vols, mats, dens = [], [], [], [], [], []
        for row in reader:
            xs.append(float(row["x_m"])); ys.append(float(row["y_m"]))
            zs.append(float(row["z_m"])); vols.append(row["volume"])
            mats.append(row["material"]); dens.append(float(row["density_g_cm3"]))
    return {
        "x_m": np.asarray(xs), "y_m": np.asarray(ys), "z_m": np.asarray(zs),
        "volume": np.asarray(vols), "material": np.asarray(mats),
        "density_g_cm3": np.asarray(dens),
    }


def material_class(material: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Coarse material class per point: "air", "rock", or "anomaly_N" (N =
    trailing integer in the Geant4 volume/material name)."""
    out = np.empty(material.shape, dtype=object)
    for i, (m, d) in enumerate(zip(material, density)):
        if d < AIR_CLASS_THRESHOLD or m == "G4_AIR":
            out[i] = "air"
        elif m == "standard_rock":
            out[i] = "rock"
        else:
            digits = re.search(r"anomaly_(\d+)", m)
            out[i] = f"anomaly_{digits.group(1)}" if digits else m
    return out


def infer_step_from_coords(coords: np.ndarray) -> float:
    uniq = np.unique(coords)
    if len(uniq) < 2:
        return 0.0
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 1e-9]
    return float(np.min(diffs)) if len(diffs) else 0.0


def classify_slice_mismatches(slice_data: dict, geo: VolcanoGeometry,
                              plane: str, plane_coord: float) -> dict:
    """Compares one Geant4 slice against the analytic engine on the same
    grid, classifying every point and counting non-boundary, non-air
    mismatches (the T2 slice locked criterion)."""
    x, y, z = slice_data["x_m"], slice_data["y_m"], slice_data["z_m"]
    g4_class = material_class(slice_data["material"], slice_data["density_g_cm3"])

    analytic_rho = geo.density(x, y, z)
    analytic_class = np.empty(x.shape, dtype=object)
    r = np.hypot(x, y)
    cone_surface = geo.height_m * np.clip(1.0 - r / geo.base_radius_m, 0.0, None)
    inside_cone = (z >= 0.0) & (z <= cone_surface)
    analytic_class[~inside_cone] = "air"
    if np.any(inside_cone):
        winner = np.full(int(np.count_nonzero(inside_cone)), "rock", dtype=object)
        xi, yi, zi = x[inside_cone], y[inside_cone], z[inside_cone]
        for idx, a in enumerate(geo.anomalies):
            mask = VolcanoGeometry._anomaly_mask(a, xi, yi, zi)
            winner[mask] = f"anomaly_{idx}"
        analytic_class[inside_cone] = winner

    # Boundary points: within half a grid step of the cone surface (in z,
    # at fixed r) or of any anomaly surface — the analytic density is not
    # constant there, so both engines legitimately disagree.
    if plane == "y":
        step_x, step_z = infer_step_from_coords(x), infer_step_from_coords(z)
        half_step = 0.5 * max(step_x, step_z, 1e-9)
        # Fixed y, varying (x, z): the cone surface is genuinely a curve
        # z = cone_surface(x) here, so a z-distance tolerance is correct.
        near_cone_boundary = np.abs(z - cone_surface) <= half_step
    else:
        step_x, step_y = infer_step_from_coords(x), infer_step_from_coords(y)
        half_step = 0.5 * max(step_x, step_y, 1e-9)
        # Fixed z, varying (x, y): the cone surface at this height is a
        # circle of radius r_boundary(z), so the tolerance must be applied
        # to the radial gap, not to a z-distance (a z-distance test here is
        # scaled by base_radius_m/height_m relative to the intended
        # half-grid-step tolerance).
        r_boundary = geo.base_radius_m * np.clip(1.0 - z / geo.height_m, 0.0, None)
        in_z_range = (z >= 0.0) & (z <= geo.height_m)
        near_cone_boundary = in_z_range & (np.abs(r - r_boundary) <= half_step)
    near_anomaly_boundary = np.zeros(x.shape, dtype=bool)
    for a in geo.anomalies:
        p = a.params
        if a.kind == "conduit":
            radial_gap = np.abs(np.hypot(x - p["cx_m"], y - p["cy_m"]) - p["radius_m"])
            axial_gap = np.minimum(np.abs(z - p["z0_m"]), np.abs(z - p["z1_m"]))
            near_anomaly_boundary |= (
                (radial_gap <= half_step)
                & (z >= p["z0_m"] - half_step) & (z <= p["z1_m"] + half_step))
            near_anomaly_boundary |= (axial_gap <= half_step) & (
                np.hypot(x - p["cx_m"], y - p["cy_m"]) <= p["radius_m"] + half_step)
        elif a.kind == "dense_body":
            nn = (((x - p["cx_m"]) / p["ax_m"]) ** 2 + ((y - p["cy_m"]) / p["ay_m"]) ** 2
                  + ((z - p["cz_m"]) / p["az_m"]) ** 2)
            grad_scale = 2.0 * np.sqrt(
                ((x - p["cx_m"]) / p["ax_m"] ** 2) ** 2 + ((y - p["cy_m"]) / p["ay_m"] ** 2) ** 2
                + ((z - p["cz_m"]) / p["az_m"] ** 2) ** 2) + 1e-12
            near_anomaly_boundary |= np.abs(nn - 1.0) <= half_step * grad_scale
        elif a.kind == "cavity":
            dist = np.sqrt((x - p["cx_m"]) ** 2 + (y - p["cy_m"]) ** 2 + (z - p["cz_m"]) ** 2)
            near_anomaly_boundary |= np.abs(dist - p["radius_m"]) <= half_step

    is_boundary = near_cone_boundary | near_anomaly_boundary
    is_air_pair = (g4_class == "air") | (analytic_class == "air")
    mismatch = g4_class != analytic_class
    unexplained = mismatch & (~is_boundary) & (~is_air_pair)

    return {
        "plane": plane, "plane_coord": plane_coord,
        "n_points": int(x.size),
        "n_mismatch_total": int(np.count_nonzero(mismatch)),
        "n_mismatch_boundary": int(np.count_nonzero(mismatch & is_boundary)),
        "n_mismatch_air_pair": int(np.count_nonzero(mismatch & (~is_boundary) & is_air_pair)),
        "n_mismatch_unexplained": int(np.count_nonzero(unexplained)),
        "half_step_m": float(half_step),
        "_arrays": {
            "x": x, "y": y, "z": z,
            "g4_class": g4_class, "analytic_class": analytic_class,
            "unexplained": unexplained, "mismatch": mismatch, "boundary": is_boundary,
        },
    }


def find_slice_files(t2_dir: Path, variant: str) -> dict:
    d = t2_dir / variant
    out = {}
    for f in d.glob("*_slice_y*.csv.gz"):
        m = re.search(r"_slice_y(-?[\d.]+)\.csv\.gz$", f.name)
        out["y"] = (f, float(m.group(1)))
    for f in d.glob("*_slice_z*.csv.gz"):
        m = re.search(r"_slice_z(-?[\d.]+)\.csv\.gz$", f.name)
        out["z"] = (f, float(m.group(1)))
    return out


def evaluate_t2_slices(entrega_dir: Path, geo: VolcanoGeometry) -> dict:
    result = {}
    for variant in ("A", "B"):
        files = find_slice_files(entrega_dir / "t2", variant)
        per_plane = {}
        for plane, (path, coord) in files.items():
            data = read_slice_csv_gz(path)
            cls = classify_slice_mismatches(data, geo, plane, coord)
            per_plane[plane] = cls
        result[variant] = per_plane
    b_pass = all(
        result["B"][plane]["n_mismatch_unexplained"] == 0 for plane in result["B"]
    )
    return {"per_variant": result, "b_zero_unexplained_mismatches": b_pass}


# --------------------------------------------------------------------- T3 PNGs
def render_t3_pngs(t2_eval: dict, out_dir: Path) -> list[str]:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
    import plotting_style  # noqa: E402
    plotting_style.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    written = []
    classes = ["air", "rock", "anomaly_0", "anomaly_1"]
    class_to_int = {c: i for i, c in enumerate(classes)}
    colors = ["#e8f0fb", "#b9975b", "#2e7d32", "#c62828"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(classes) + 0.5, 1), cmap.N)

    def to_grid(x, y_or_z, values_int, plane):
        xs = np.unique(x)
        ys = np.unique(y_or_z)
        grid = np.full((ys.size, xs.size), -1, dtype=float)
        xi = np.searchsorted(xs, x)
        yi = np.searchsorted(ys, y_or_z)
        grid[yi, xi] = values_int
        return xs, ys, grid

    for plane in ("y", "z"):
        if plane not in t2_eval["per_variant"]["A"] or plane not in t2_eval["per_variant"]["B"]:
            continue
        a = t2_eval["per_variant"]["A"][plane]["_arrays"]
        b = t2_eval["per_variant"]["B"][plane]["_arrays"]
        coord_a = t2_eval["per_variant"]["A"][plane]["plane_coord"]

        if plane == "y":
            horiz, vert, xlabel, ylabel = a["x"], a["z"], "x (m)", "z (m)"
        else:
            horiz, vert, xlabel, ylabel = a["x"], a["y"], "x (m)", "y (m)"

        analytic_int = np.array([class_to_int.get(c, -1) for c in a["analytic_class"]])
        a_int = np.array([class_to_int.get(c, -1) for c in a["g4_class"]])
        b_int = np.array([class_to_int.get(c, -1) for c in b["g4_class"]])

        xs, ys, grid_analytic = to_grid(horiz, vert, analytic_int, plane)
        _, _, grid_a = to_grid(horiz, vert, a_int, plane)
        _, _, grid_b = to_grid(horiz, vert, b_int, plane)
        _, _, grid_mismatch_a = to_grid(horiz, vert, a["unexplained"].astype(float), plane)
        _, _, grid_mismatch_b = to_grid(horiz, vert, b["unexplained"].astype(float), plane)

        extent = [xs.min(), xs.max(), ys.min(), ys.max()]

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
        for ax, grid, title in zip(
            axes, (grid_analytic, grid_a, grid_b),
            ("motor analítico", "Geant4 variante A", "Geant4 variante B"),
        ):
            im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto",
                           cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_xlabel(xlabel)
            ax.set_title(title, fontsize=10)
        axes[0].set_ylabel(ylabel)
        cbar = fig.colorbar(im, ax=axes, ticks=range(len(classes)), shrink=0.85)
        cbar.ax.set_yticklabels(["aire", "roca", "anomalía 0 (conducto)", "anomalía 1 (cuerpo denso)"])
        fig.suptitle(f"geo-train-06, corte {plane}={coord_a:g} m — mapas de material",
                    fontsize=11)
        out1 = out_dir / f"t3_corte_{plane}{coord_a:g}_materiales.png"
        fig.savefig(out1)
        plt.close(fig)
        written.append(rel_path_str(out1))

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
        for ax, grid, title in zip(axes, (grid_mismatch_a, grid_mismatch_b),
                                   ("A vs. analítico", "B vs. analítico")):
            ax.imshow(grid, origin="lower", extent=extent, aspect="auto",
                     cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xlabel(xlabel)
            ax.set_title(f"discrepancias no explicadas — {title}", fontsize=10)
        axes[0].set_ylabel(ylabel)
        fig.suptitle(f"geo-train-06, corte {plane}={coord_a:g} m — discrepancias "
                    "de clase de material no explicadas (no aire, no borde)",
                    fontsize=10)
        out2 = out_dir / f"t3_corte_{plane}{coord_a:g}_discrepancias.png"
        fig.savefig(out2)
        plt.close(fig)
        written.append(rel_path_str(out2))

    return written


# --------------------------------------------------------------------- T4
def evaluate_t4(entrega_dir: Path) -> dict:
    result = {}
    for variant in ("A", "B"):
        d = entrega_dir / "t4" / variant
        run_log_files = list(d.glob("run.log"))
        meta_files = list(d.glob("*.meta.json"))
        hist_files = list(d.glob("*_hist.csv"))
        row = {"variant": variant}
        if not run_log_files or not meta_files or not hist_files:
            row["error"] = "missing run.log / meta.json / hist.csv"
            result[variant] = row
            continue
        log_text = run_log_files[0].read_text(encoding="utf-8", errors="replace")
        meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
        hist_rows = []
        with hist_files[0].open(encoding="utf-8", newline="") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    hist_rows.append([float(v) for v in line.split(",")])
        hist_arr = np.asarray(hist_rows) if hist_rows else np.zeros((0, 0))
        hist_sum = float(hist_arr.sum())

        run_done = "RUN DONE" in log_text
        n_geomvol1002 = log_text.count("GeomVol1002")
        n_geomnav = log_text.count("GeomNav")
        events_transported = meta.get("events_transported")
        detected_in_bins = meta.get("detected_in_bins")

        row.update({
            "run_done": run_done,
            "n_GeomVol1002": n_geomvol1002,
            "n_GeomNav": n_geomnav,
            "hist_non_empty": hist_arr.size > 0,
            "hist_sum": hist_sum,
            "events_transported": events_transported,
            "detected_in_bins": detected_in_bins,
            "events_transported_matches_expected": events_transported == T4_EVENTS_EXPECTED,
            "hist_sum_matches_detected": (
                detected_in_bins is not None and abs(hist_sum - float(detected_in_bins)) < 1e-6
            ),
        })
        if variant == "B":
            row["pass"] = (
                run_done and n_geomvol1002 == 0 and n_geomnav == 0
                and row["hist_non_empty"]
                and row["events_transported_matches_expected"]
                and row["hist_sum_matches_detected"]
            )
        else:
            row["pass"] = None  # A is reference-only, not judged
            row["reference_note"] = ("variant A is expected to print GeomVol1002 "
                                     "at construction; not a locked criterion")
        result[variant] = row
    return result


# ------------------------------------------------------------------ main CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entrega-dir", type=Path, required=True,
                    help="Directory holding resultados-t1.tsv, "
                         "control-geometrico-logs.tar.gz, t2/, t4/.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    geo = VolcanoGeometry.from_json(args.catalog_dir / f"{DETAIL_GEOMETRY}.json")

    # T1
    resultados_t1 = load_t1_resultados(args.entrega_dir / "resultados-t1.tsv")
    resumen = build_resumen_solapes(resultados_t1, args.entrega_dir / "control-geometrico-logs.tar.gz")
    t1_eval = evaluate_t1(resumen)

    resumen_path = args.out_dir / "resumen-solapes.csv"
    with resumen_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, lineterminator="\n", fieldnames=[
            "variant", "geometry", "view", "exit_code", "overlapping",
            "volumes", "n_GeomVol1002", "n_GeomNav", "wall_s",
        ])
        writer.writeheader()
        for r in resumen:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})

    # T2
    diag_a = parse_diag_log((args.entrega_dir / "t2" / "A" / "diag.log").read_text(encoding="utf-8"))
    diag_b = parse_diag_log((args.entrega_dir / "t2" / "B" / "diag.log").read_text(encoding="utf-8"))
    t2_probes = evaluate_t2_probes(diag_a, diag_b, geo)
    t2_volumes = evaluate_t2_volumes(diag_b)
    t2_slices = evaluate_t2_slices(args.entrega_dir, geo)

    sondeos_path = args.out_dir / "sondeos-volumenes.json"
    sondeos_path.write_text(json.dumps({
        "probes": t2_probes["rows"], "all_b_probes_match_analytic": t2_probes["all_b_match_analytic"],
        "volumes": t2_volumes["rows"], "all_volumes_within_3_se": t2_volumes["all_within_3_se"],
    }, indent=2, default=str), encoding="utf-8", newline="\n")

    cortes_path = args.out_dir / "cortes-comparacion.json"
    cortes_summary = {
        variant: {
            plane: {k: v for k, v in c.items() if k != "_arrays"}
            for plane, c in planes.items()
        }
        for variant, planes in t2_slices["per_variant"].items()
    }
    cortes_path.write_text(json.dumps({
        "per_variant": cortes_summary,
        "b_zero_unexplained_mismatches": t2_slices["b_zero_unexplained_mismatches"],
    }, indent=2, default=str), encoding="utf-8", newline="\n")

    # T3
    t3_files = render_t3_pngs(t2_slices, args.out_dir)

    # T4
    t4_eval = evaluate_t4(args.entrega_dir)

    t2_pass = (
        t2_probes["all_b_match_analytic"]
        and t2_volumes["all_within_3_se"]
        and t2_slices["b_zero_unexplained_mismatches"]
    )
    t4_pass = bool(t4_eval.get("B", {}).get("pass"))

    verdict = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "script_sha256": sha256_of(Path(__file__)),
            "entrega_dir": rel_path_str(args.entrega_dir),
            "detail_geometry": DETAIL_GEOMETRY, "detail_view": DETAIL_VIEW,
        },
        "T1": {"pass": t1_eval["pass"], "detail": {
            k: v for k, v in t1_eval.items() if k != "anomalies"
        } | {"n_anomalies": len(t1_eval["anomalies"])}},
        "T2": {
            "pass": t2_pass,
            "probes_pass": t2_probes["all_b_match_analytic"],
            "volumes_pass": t2_volumes["all_within_3_se"],
            "slices_pass": t2_slices["b_zero_unexplained_mismatches"],
        },
        "T3": {"png_files": t3_files, "n_files": len(t3_files)},
        "T4": {
            "pass": t4_pass,
            "variant_B": {k: v for k, v in t4_eval.get("B", {}).items()},
            "variant_A_reference": {k: v for k, v in t4_eval.get("A", {}).items()},
        },
        "overall_pass": t1_eval["pass"] and t2_pass and t4_pass,
    }
    criterios_path = args.out_dir / "criterios-t1-t4.json"
    criterios_path.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8", newline="\n")

    print(f"T1: pass={t1_eval['pass']} n_runs={t1_eval['n_runs']} "
         f"defective={t1_eval['a_defective_geometries']}")
    print(f"T2: pass={t2_pass} probes={t2_probes['all_b_match_analytic']} "
         f"volumes={t2_volumes['all_within_3_se']} slices={t2_slices['b_zero_unexplained_mismatches']}")
    print(f"T3: wrote {len(t3_files)} PNG files")
    print(f"T4: pass={t4_pass}")
    print(f"overall_pass={verdict['overall_pass']}")
    print(f"wrote {resumen_path}, {sondeos_path}, {cortes_path}, {criterios_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
