#!/usr/bin/env python3
"""OE1 stage 1: synthetic list of technical test T-S and its check.

``generate`` writes a list of 60 mu- of 10 TeV kinetic energy for geo-train-06
view 0 in the same CSV + sidecar format as the C++ writer (its canonical FNV
passes ``muography_sim --check-primaries``; the sidecar carries the extra keys
``synthetic`` and ``clases``, which the C++ reader ignores):

  * 20 lines that cross Z at a depth of at least ``--min-depth-m`` (default
    3 m; the plan asks for >= 1 m) -- rigorous: the line passes within
    (r_in - depth) of the centroid, where r_in is the radius of a ball around
    the centroid that lies inside Z;
  * 20 near misses whose clearance from Z is in [0.5, 1.0] m -- verified with
    perimeter tests of parallel lines displaced by 0.5 m (all miss) and 1.0 m
    (some hit) on the plane perpendicular to the line;
  * 20 lines that miss Z by at least 50 m -- rigorous: the line stays at least
    R_b + 50 m from the centroid, R_b = radius of the bounding sphere of Z.

Every track ends on the 1.2 m panel of the view-0 detector and starts beyond
the far rim exactly like the generator: ``start = impact - dir * R_s / cos(el)``
with R_s = 1000 + R + 300. The expected classification is computed with the
reused ``crossing_reference.py`` and written to ``sintetico-esperado.json``.

``check`` compares the ``_muones.csv`` of an A2 or B2 run of that list with the
expectation (T-S acceptance of the plan). Plan correction S1: WentzelVI displaces
the track laterally inside a step, so the chord in Z is compared with the straight
line within +-1 % only for the central lines; for the shallow ones the acceptable
length is the range spanned by the parallel lines displaced transversally by up to
``--transverse-tolerance-m`` (both limits are written in the expected file).

Deterministic: only ``random.Random.random()`` (stable across Python versions)
is used, never numpy's generator.
"""
import argparse
import math
import os
import random
import sys

import numpy as np

import oe1_stage1_common as C
import oe1_stage1_lists as L
import oe1_stage1_roi as R

ENERGY_GEV = 10000.0
N_PER_CLASS = 20
CASE_CODE = 6
REPLICA = 9
DEFAULT_SEED = 20260930
NEAR_MIN_M = 0.5
NEAR_MAX_M = 1.0
FAR_MIN_M = 50.0
CLASSES = ("atraviesa", "casi_fallo", "lejos")
# PLAN correction S1 (2026-09-30): WentzelVI displaces the track laterally inside a step, so the chord
# length is compared with the straight line only within +-1 % for the CENTRAL lines (those whose length is
# insensitive to a transverse shift); for the shallow ones the acceptable interval is the range of lengths of
# the parallel lines displaced by up to this distance in the plane perpendicular to the track.
TRANSVERSE_TOLERANCE_M = 0.05


def _unit(v):
    return v / np.linalg.norm(v)


def _perp_basis(d):
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = _unit(np.cross(d, ref))
    e2 = np.cross(d, e1)
    return e1, e2


class Scene:
    """Geometry of the synthetic muons for one catalog geometry (view 0)."""

    def __init__(self, cr, cat, view=0):
        self.cr = cr
        self.view = view
        self.r_start = C.DET_DIST_M + cat["base_radius_m"] + 300.0
        self.centroid = np.array(R.Z_CENTROID_M)
        lo, hi = R.z_bounding_box(cr)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        self.r_bound = float(np.linalg.norm(corners - self.centroid, axis=1).max())
        self.r_inscribed = _inscribed_radius(cr, self.centroid)
        self.det = np.array(C.detector_center(view))
        va = C.look_azimuth_rad(view) - math.pi
        self.panel_y = np.array([-math.sin(va), math.cos(va), 0.0])

    def panel_point(self, ua, ub):
        half = C.DET_SIDE_M / 2.0
        return self.det + self.panel_y * ((2.0 * ua - 1.0) * half) + \
            np.array([0.0, 0.0, (2.0 * ub - 1.0) * half])

    def segment(self, panel_pt, from_dir):
        """(start, impact): the track runs from `start` down to the panel point."""
        hor = math.hypot(from_dir[0], from_dir[1])
        start = panel_pt + from_dir * (self.r_start / hor)
        return start, panel_pt

    def z_length(self, panel_pt, from_dir, offset=None):
        start, imp = self.segment(panel_pt, from_dir)
        if offset is not None:
            start = start + offset
            imp = imp + offset
        m = self.cr.model(np.hstack([start, imp])[None, :])
        return float(m["len"][0]), bool(m["deep"][0]), bool(m["graze"][0])

    def lengths_many(self, panel_pt, from_dir, offsets):
        """Chord length in Z (m) of the parallel lines displaced by `offsets` (m, shape (k, 3))."""
        start, imp = self.segment(panel_pt, from_dir)
        offsets = np.asarray(offsets, dtype=np.float64)
        m = self.cr.model(np.hstack([start + offsets, imp + offsets]))
        return np.asarray(m["len"], dtype=np.float64)

    def hits_many(self, panel_pt, from_dir, offsets):
        """Which of the parallel lines displaced by `offsets` (m, shape (k, 3)) reach Z."""
        return self.lengths_many(panel_pt, from_dir, offsets) > 0.0

    def length_interval(self, panel_pt, from_dir, delta, n=360):
        """(min, max) chord length in Z over the parallel lines displaced transversally by up to `delta` m.

        Z is convex, so the length is a concave function of the displacement: the maximum can lie inside
        the disc and the minimum on its rim. The rim is sampled every degree (the length changes by well
        under 0.1 mm between samples for a 5 cm rim) and the interior at half the radius as well.
        """
        offs = [np.zeros((1, 3)), self.ring_offsets(from_dir, 0.5 * delta, n), self.ring_offsets(from_dir, delta, n)]
        lens = self.lengths_many(panel_pt, from_dir, np.vstack(offs))
        return float(lens.min()), float(lens.max())

    def ring_offsets(self, from_dir, radius, n=720):
        e1, e2 = _perp_basis(from_dir)
        ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        return radius * (np.cos(ang)[:, None] * e1[None, :] + np.sin(ang)[:, None] * e2[None, :])

    def distance_to_centroid(self, panel_pt, from_dir):
        return float(np.linalg.norm(np.cross(self.centroid - panel_pt, from_dir)))

    def pixel_angles(self, from_dir):
        elev = math.degrees(math.asin(from_dir[2]))
        rel = C.wrap_angle(math.atan2(from_dir[1], from_dir[0]) - C.look_azimuth_rad(self.view))
        return elev, math.degrees(rel)


def _inscribed_radius(cr, c):
    """Rigorous radius of a ball around ``c`` contained in Z.

    Cylinder: exact (radial wall and z slab). Ellipsoid: with u the normalised
    coordinates, |x - x'| >= s_min |u - u'|, hence the surface is at least
    s_min (1 - |u|) away.
    """
    cyl, ell = cr.CYL, cr.ELL
    radial = cyl["r"] - math.hypot(c[0] - cyl["cx"], c[1] - cyl["cy"])
    slab = min(c[2] - cyl["z0"], cyl["z1"] - c[2])
    u = np.array([(c[0] - ell["cx"]) / ell["ax"], (c[1] - ell["cy"]) / ell["ay"],
                  (c[2] - ell["cz"]) / ell["az"]])
    q = float(np.linalg.norm(u))
    ellip = min(ell["ax"], ell["ay"], ell["az"]) * (1.0 - q)
    return float(max(0.0, min(radial, slab, ellip)))



# --------------------------------------------------------------------------
def _draw_dir_in_plane(rng, d0):
    e1, e2 = _perp_basis(d0)
    a = 2.0 * math.pi * rng.random()
    return math.cos(a) * e1 + math.sin(a) * e2


def make_through(scene, rng, min_depth):
    rho_max = scene.r_inscribed - min_depth
    if rho_max <= 0:
        raise SystemExit("Z is too thin for a %.2f m depth" % min_depth)
    for _ in range(1000):
        p = scene.panel_point(rng.random(), rng.random())
        # aim point: uniform in the ball of radius rho_max around the centroid
        while True:
            v = np.array([2 * rng.random() - 1, 2 * rng.random() - 1, 2 * rng.random() - 1])
            if np.dot(v, v) <= 1.0:
                break
        aim = scene.centroid + rho_max * v
        f = _unit(aim - p)
        if not _in_grid(scene, f):
            continue
        ln, deep, graze = scene.z_length(p, f)
        rho = scene.distance_to_centroid(p, f)
        if ln > 0.0 and deep:
            return {"clase": "atraviesa", "panel": p, "from": f, "lz_recta_m": ln,
                    "profundidad_garantizada_m": scene.r_inscribed - rho,
                    "distancia_al_centroide_m": rho}
    raise SystemExit("could not build a through line")


def _in_grid(scene, f):
    elev, rel = scene.pixel_angles(f)
    return (C.EL_MIN_DEG + 0.5 <= elev <= C.EL_MAX_DEG - 0.5 and abs(rel) <= C.AZ_HALF_DEG - 0.5)


def clearance_estimate(scene, p, f, hi=3.0):
    """Smallest ring radius (m) whose parallel lines reach Z; None if > `hi`.

    Requires that the line itself misses Z. The set of displacements that hit Z
    is convex, so the first ring radius that touches it is the distance between
    the line and Z.
    """
    if not scene.hits_many(p, f, scene.ring_offsets(f, hi)).any():
        return None
    lo = 0.0
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if scene.hits_many(p, f, scene.ring_offsets(f, mid)).any():
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def make_near_miss(scene, rng):
    for _ in range(2000):
        p = scene.panel_point(rng.random(), rng.random())
        d0 = _unit(scene.centroid - p)
        u = _draw_dir_in_plane(rng, d0)

        def from_dir(s):
            return _unit(scene.centroid + s * u - p)

        def hit(s):
            return scene.z_length(p, from_dir(s))[0] > 0.0

        if not hit(0.0):
            continue
        lo, hi = 0.0, 250.0
        if hit(hi):
            continue
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if hit(mid):
                lo = mid
            else:
                hi = mid
        s_star = hi
        target = NEAR_MIN_M + 0.1 + rng.random() * (NEAR_MAX_M - NEAR_MIN_M - 0.2)
        off = target
        for _try in range(14):
            f = from_dir(s_star + off)
            if not _in_grid(scene, f):
                break
            if scene.z_length(p, f)[0] > 0.0:
                off *= 1.2
                continue
            est = clearance_estimate(scene, p, f)
            if est is None:
                off *= 0.7          # farther than the search range: come closer
                continue
            if abs(est - target) > 0.01:
                off *= target / max(est, 1e-3)
                continue
            ring_lo = scene.hits_many(p, f, scene.ring_offsets(f, NEAR_MIN_M))
            ring_hi = scene.hits_many(p, f, scene.ring_offsets(f, NEAR_MAX_M))
            if ring_lo.any() or not ring_hi.any():
                break
            return {"clase": "casi_fallo", "panel": p, "from": f, "lz_recta_m": 0.0,
                    "holgura_minima_m": NEAR_MIN_M, "holgura_maxima_m": NEAR_MAX_M,
                    "holgura_estimada_m": est,
                    "distancia_al_centroide_m": scene.distance_to_centroid(p, f)}
    raise SystemExit("could not build a near-miss line")


def make_far(scene, rng):
    need = scene.r_bound + FAR_MIN_M + 1.0
    for _ in range(2000):
        p = scene.panel_point(rng.random(), rng.random())
        d0 = _unit(scene.centroid - p)
        u = _draw_dir_in_plane(rng, d0)
        offset = need + 4.0 + rng.random() * 200.0
        f = _unit(scene.centroid + offset * u - p)
        if not _in_grid(scene, f):
            continue
        rho = scene.distance_to_centroid(p, f)
        if rho < need:
            continue
        ln, deep, graze = scene.z_length(p, f)
        if ln == 0.0 and not deep and not graze:
            return {"clase": "lejos", "panel": p, "from": f, "lz_recta_m": 0.0,
                    "holgura_minima_m": rho - scene.r_bound,
                    "distancia_al_centroide_m": rho}
    raise SystemExit("could not build a far line")


def build_muons(scene, seed, min_depth):
    rng = random.Random(seed)
    items = []
    for _ in range(N_PER_CLASS):
        items.append(make_through(scene, rng, min_depth))
    for _ in range(N_PER_CLASS):
        items.append(make_near_miss(scene, rng))
    for _ in range(N_PER_CLASS):
        items.append(make_far(scene, rng))
    # deterministic Fisher-Yates so the classes are interleaved
    for i in range(len(items) - 1, 0, -1):
        j = int(rng.random() * (i + 1))
        items[i], items[j] = items[j], items[i]
    return items


def _list_arrays(scene, items):
    n = len(items)
    vals = np.empty((n, 7))
    for k, it in enumerate(items):
        start, imp = scene.segment(it["panel"], it["from"])
        travel = -it["from"]
        vals[k, 0:3] = start
        vals[k, 3:6] = travel
        vals[k, 6] = ENERGY_GEV
        it["start"] = start
        it["travel"] = travel
    ids = np.arange(n, dtype=np.int64)
    pdg = np.full(n, 13, dtype=np.int64)
    charge = np.full(n, -1, dtype=np.int64)
    return ids, vals, pdg, charge


def generate(out_dir, catalog_path, cr_path=None, seed=DEFAULT_SEED, min_depth=3.0, tables_dir=None,
             transverse_tol=TRANSVERSE_TOLERANCE_M, rel_tol=0.01):
    cat = C.read_catalog(catalog_path)
    if cat["geometry_id"] != "geo-train-06":
        raise SystemExit("T-S is defined for geo-train-06 (got %s)" % cat["geometry_id"])
    cr, cr_file, cr_sha = C.load_crossing_reference(cr_path)
    if not C.reference_matches_catalog(cr, cat):
        raise SystemExit("crossing_reference.py primitives differ from the catalog")
    scene = Scene(cr, cat)
    items = build_muons(scene, seed, min_depth)
    ids, vals, pdg, charge = _list_arrays(scene, items)
    os.makedirs(out_dir, exist_ok=True)
    name = "%s_v0_r%d_sintetico.primarios.csv" % (cat["geometry_id"], REPLICA)
    list_path = os.path.join(out_dir, name)
    raw = C.write_primary_csv(list_path, ids, vals, pdg, charge)
    canon = C.canonical_fnv(ids, vals, pdg)
    counts = {c: sum(1 for it in items if it["clase"] == c) for c in CLASSES}
    side = {
        "format": C.LIST_FORMAT, "kind": "sintetico", "synthetic": True,
        "geometry_id": cat["geometry_id"], "view_index": 0,
        "base_radius_m": cat["base_radius_m"], "start_radius_m": scene.r_start,
        "case_code": CASE_CODE, "replica": REPLICA, "source_seed": None,
        "n": len(items), "columns": C.LIST_HEADER.split(","),
        "units": {"id": "Geant4 event ID, rows 0..n-1 in order",
                  "position": "m, world frame (cone axis at the origin, ground at z = 0)",
                  "direction": "unit travel direction", "ekin_gev": "kinetic energy, GeV",
                  "pdg": "13 = mu-, -13 = mu+", "charge_e": "elementary charges"},
        "float_format": "shortest round-trip text (repr); LF line endings",
        "canon_fnv1a64": C.hex16(canon),
        "canon_hash_layout": ("FNV-1a-64 over each record in file order: little-endian int64 id, "
                              "float64 x_m, y_m, z_m, dx, dy, dz, ekin_gev, int32 pdg"),
        "file_sha256": C.sha256_bytes(raw),
        "descripcion": ("lista SINTETICA de la prueba tecnica T-S (no es muestra del modelo de "
                        "fuente): 60 mu- de 10 TeV"),
        "clases": counts, "semilla_sintetica": seed, "profundidad_minima_m": min_depth,
        "generador": "oe1_stage1_synthetic.py",
    }
    C.write_text_new(C.sidecar_path(list_path), C.dumps_json(side))
    # self-check with the same code as the validator (structure + geometry)
    res = L.validate_list(list_path, None, None, do_stats=False)
    if res["veredicto"] != "OK":
        raise SystemExit("la lista sintetica no supera la validacion: %r" % res["problemas"])
    expected = {
        "formato": "oe1-sintetico-esperado/1",
        "lista": name, "sha256_lista": side["file_sha256"], "canon_fnv1a64": side["canon_fnv1a64"],
        "energia_cinetica_gev": ENERGY_GEV, "particula": "mu-", "semilla": seed,
        "catalogo_sha256": cat["sha256"],
        "region": {"cilindro": cr.CYL, "elipsoide": cr.ELL, "centroide_m": list(R.Z_CENTROID_M),
                   "radio_esfera_envolvente_m": scene.r_bound,
                   "radio_esfera_inscrita_m": scene.r_inscribed},
        "criterios": {"cruce": ("crossed == 1; longitud dentro de +-1 % de la recta en las lineas centrales y, "
                                "en las someras, dentro del intervalo de longitudes de las rectas paralelas "
                                "desplazadas hasta la tolerancia transversal (S1: el desplazamiento lateral "
                                "de WentzelVI hace que la cuerda no sea la trayectoria navegada)"),
                      "casi_fallo": "holgura entre 0,5 y 1,0 m: crossed == 0",
                      "lejos": "holgura >= 50 m: crossed == 0",
                      "B": "lz_a0 <= 1 um y lz_rock <= 1 um", "A": "lz_a0 + lz_a1 = lz"},
        "tolerancia_relativa_de_longitud": rel_tol,
        "tolerancia_de_desplazamiento_transversal_m": transverse_tol,
        "crossing_reference": {"ruta": cr_file, "sha256": cr_sha},
        "muones": [],
    }
    for k, it in enumerate(items):
        elev, rel = scene.pixel_angles(it["from"])
        row = {"id": k, "clase": it["clase"],
               "crossed_esperado": 1 if it["clase"] == "atraviesa" else 0,
               "lz_recta_m": it["lz_recta_m"],
               "distancia_al_centroide_m": it["distancia_al_centroide_m"],
               "elevacion_grados": elev, "acimut_relativo_grados": rel,
               "impacto_y_m": float(it["panel"][1]), "impacto_z_m": float(it["panel"][2])}
        for key in ("profundidad_garantizada_m", "holgura_minima_m", "holgura_maxima_m",
                    "holgura_estimada_m"):
            if key in it:
                row[key] = it[key]
        if it["clase"] == "atraviesa":
            lo, hi = scene.length_interval(it["panel"], it["from"], transverse_tol)
            row["lz_desplazada_min_m"], row["lz_desplazada_max_m"] = lo, hi
            # central: the whole displacement interval already lies inside the +-1 % band of the line
            row["central"] = bool(lo >= it["lz_recta_m"] * (1.0 - rel_tol) and
                                  hi <= it["lz_recta_m"] * (1.0 + rel_tol))
        expected["muones"].append(row)
    exp_path = os.path.join(out_dir, "sintetico-esperado.json")
    C.write_text_new(exp_path, C.dumps_json(expected))
    return {"lista": list_path, "esperado": exp_path, "sha256": side["file_sha256"],
            "canon_fnv1a64": side["canon_fnv1a64"], "clases": counts,
            "r_inscrita_m": scene.r_inscribed, "r_envolvente_m": scene.r_bound}


def cmd_generate(args):
    res = generate(args.out_dir, args.catalog, args.crossing_reference, args.seed, args.min_depth_m,
                   transverse_tol=args.transverse_tolerance_m)
    print("lista sintetica: %s" % res["lista"])
    print("  sha256 %s  fnv %s  clases %s" % (res["sha256"], res["canon_fnv1a64"], res["clases"]))
    print("  esfera inscrita %.3f m, esfera envolvente %.3f m" % (res["r_inscrita_m"], res["r_envolvente_m"]))
    print("  clasificacion esperada: %s" % res["esperado"])
    return 0


# --------------------------------------------------------------------------
def check_run(expected, mu, variant, rel_tol=0.01, um=1e-6):
    """T-S acceptance of the plan for one run. Returns (summary, per-muon rows)."""
    rows = []
    n_exp = len(expected["muones"])
    problems = []
    n = len(mu["id"])
    if n != n_exp or not np.array_equal(mu["id"], np.arange(n_exp)):
        problems.append("filas %d != %d esperadas o ids no 0..%d" % (n, n_exp, n_exp - 1))
    n_class_ok = 0
    n_len_ok = 0
    n_len_total = 0
    n_central = n_central_ok = n_shallow = n_shallow_ok = 0
    n_mat_ok = 0
    not_det = 0
    for k, e in enumerate(expected["muones"]):
        if k >= n:
            break
        crossed = mu["crossed"][k]
        lz = mu["lz_m"][k]
        a0 = mu["lz_a0_m"][k]
        a1 = mu["lz_a1_m"][k]
        rock = mu["lz_rock_m"][k]
        other = mu["lz_other_m"][k]
        class_ok = bool(crossed == e["crossed_esperado"])
        if e["crossed_esperado"] == 0:
            class_ok = class_ok and lz == 0.0
        len_ok = None
        rel = None
        lo = hi = None
        central = None
        if e["crossed_esperado"] == 1:
            n_len_total += 1
            ref = e["lz_recta_m"]
            rel = (lz - ref) / ref
            # PLAN S1: +-1 % of the line, widened (shallow lines only) to the range of lengths of the
            # parallel lines displaced transversally; expected files without that range keep +-1 %
            lo = min(ref * (1.0 - rel_tol), e.get("lz_desplazada_min_m", ref * (1.0 - rel_tol)))
            hi = max(ref * (1.0 + rel_tol), e.get("lz_desplazada_max_m", ref * (1.0 + rel_tol)))
            len_ok = bool(lo <= lz <= hi)
            central = bool(e.get("central", True))
            n_len_ok += int(len_ok)
            n_central += int(central)
            n_central_ok += int(central and len_ok)
            n_shallow += int(not central)
            n_shallow_ok += int((not central) and len_ok)
        if variant == "B":
            mat_ok = bool(a0 <= um and rock <= um and other <= um and abs(lz - a1) <= um)
        else:
            mat_ok = bool(rock <= um and other <= um and abs(lz - (a0 + a1)) <= um)
        if e["crossed_esperado"] == 0:
            mat_ok = mat_ok and lz == 0.0
        n_class_ok += int(class_ok)
        n_mat_ok += int(mat_ok)
        det = mu["status"][k] == b"DET"
        not_det += int(not det)
        rows.append({"id": k, "clase": e["clase"], "crossed_esperado": e["crossed_esperado"],
                     "crossed_obtenido": None if np.isnan(crossed) else int(crossed),
                     "lz_recta_m": e["lz_recta_m"], "lz_obtenida_m": float(lz),
                     "diferencia_relativa": rel, "clasificacion_ok": class_ok,
                     "central": central, "longitud_aceptable_min_m": lo, "longitud_aceptable_max_m": hi,
                     "longitud_ok": len_ok, "materiales_ok": mat_ok,
                     "estado": mu["status"][k].decode("ascii")})
    summary = {
        "variante": variant, "muones": n_exp,
        "clasificacion_correcta": n_class_ok,
        # through lines whose length is acceptable (name kept from the +-1 % version; see S1 above)
        "longitud_dentro_de_1pct": n_len_ok,
        "atraviesan_esperados": n_len_total, "materiales_ok": n_mat_ok,
        "lineas_centrales": n_central, "centrales_dentro_de_1pct": n_central_ok,
        "lineas_someras": n_shallow, "someras_dentro_del_intervalo_transversal": n_shallow_ok,
        "tolerancia_desplazamiento_transversal_m": expected.get("tolerancia_de_desplazamiento_transversal_m"),
        "no_detectados": not_det, "tolerancia_relativa": rel_tol, "tolerancia_um_m": um,
        "problemas": problems,
    }
    summary["ok"] = bool(not problems and n_class_ok == n_exp and n_len_ok == n_len_total and
                         n_mat_ok == n_exp)
    return summary, rows


def cmd_check(args):
    expected = C.read_json(args.expected)
    with open(args.muones, "rb") as fh:
        mu = C.parse_muones_bytes(fh.read())
    summary, rows = check_run(expected, mu, args.variant, args.length_tolerance)
    n = summary["muones"]
    label = "T-S (%s2)" % args.variant
    checks = [
        {"descripcion": "el registro tiene %d filas con IDs 0..%d" % (n, n - 1), "ok": not summary["problemas"],
         "evidencia": {"problemas": summary["problemas"]}},
        {"descripcion": "clasificación correcta en los %d muones (atraviesa / casi fallo / lejos)" % n,
         "ok": summary["clasificacion_correcta"] == n, "evidencia": {"correctos": summary["clasificacion_correcta"]}},
        {"descripcion": ("longitud dentro de ±%g %% de la recta en las líneas centrales y dentro del intervalo de "
                         "desplazamiento transversal (±%g m) en las someras" %
                         (100 * summary["tolerancia_relativa"],
                          summary["tolerancia_desplazamiento_transversal_m"] or 0.0)),
         "ok": summary["longitud_dentro_de_1pct"] == summary["atraviesan_esperados"],
         "evidencia": {"dentro": summary["longitud_dentro_de_1pct"], "esperados": summary["atraviesan_esperados"],
                       "centrales": summary["lineas_centrales"],
                       "centrales_dentro_de_1pct": summary["centrales_dentro_de_1pct"],
                       "someras": summary["lineas_someras"],
                       "someras_dentro_del_intervalo": summary["someras_dentro_del_intervalo_transversal"]}},
        {"descripcion": ("en B2 lz_a0 y lz_rock <= 1 µm" if args.variant == "B"
                         else "en A2 lz_a0 + lz_a1 = lz y lz_rock <= 1 µm"),
         "ok": summary["materiales_ok"] == n, "evidencia": {"correctos": summary["materiales_ok"]}},
    ]
    out = {"formato": "oe1-prueba-tecnica/1", "prueba": label, "ok": bool(summary["ok"]),
           "etiqueta": "prueba técnica (DV1), fuera de las 13 corridas", "comprobaciones": checks,
           "entradas": {"esperado_sha256": C.sha256_file(args.expected),
                        "muones_sha256": C.sha256_file(args.muones)},
           "aviso_no_detectados": summary["no_detectados"], "resumen": summary, "muones": rows}
    if args.out:
        C.write_text_new(args.out, C.dumps_json(out))
    print("%s: clasificacion %d/%d, longitud %d/%d, materiales %d/%d, no detectados %d -> %s" %
          (label, summary["clasificacion_correcta"], n, summary["longitud_dentro_de_1pct"],
           summary["atraviesan_esperados"], summary["materiales_ok"], n, summary["no_detectados"],
           "OK" if summary["ok"] else "FALLA"))
    for p in summary["problemas"]:
        print("  - " + p)
    return 0 if summary["ok"] else 3


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="escribe la lista sintetica y la clasificacion esperada")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--catalog", required=True, help="geo-train-06.json")
    g.add_argument("--crossing-reference")
    g.add_argument("--seed", type=int, default=DEFAULT_SEED)
    g.add_argument("--min-depth-m", type=float, default=3.0)
    g.add_argument("--transverse-tolerance-m", type=float, default=TRANSVERSE_TOLERANCE_M,
                   help="desplazamiento transversal que admite la longitud de las lineas someras (S1)")
    g.set_defaults(fn=cmd_generate)
    c = sub.add_parser("check", help="compara un _muones.csv de T-S con lo esperado")
    c.add_argument("--expected", required=True)
    c.add_argument("--muones", required=True)
    c.add_argument("--variant", required=True, choices=("A", "B"))
    c.add_argument("--length-tolerance", type=float, default=0.01)
    c.add_argument("--out")
    c.set_defaults(fn=cmd_check)
    return ap


def main(argv=None):
    C.utf8_streams()
    args = build_parser().parse_args(argv)
    return C.run_command(args.fn, args)


if __name__ == "__main__":
    sys.exit(main())
