#!/usr/bin/env python3
"""OE1 complement R2-01: 12 directed muons into Z, A2 against B2, three transport seeds.

The complement (objetivo-1/COMPLEMENTO_R2_01.md of the director's repository) freezes a list of
12 mu- of 10 TeV aimed at the centre of the view-0 panel. On their straight lines, ids 0-5 reach
Z = conduit AND dense body from the conduit side and ids 6-11 from the dense-body side.
herramientas/ejecutar_complemento_r2_01.sh transports the list with the stage-1 binaries A2 and B2
under three transport seeds: 6 runs, 72 transports.

Subcommands
  lines    before any transport: checks every straight line of the frozen list against the
           catalog with the reused crossing_reference.py (which primitive the line enters first,
           depth, length and entry/exit points in Z, the side kept by parallel lines displaced
           2.5 and 5 cm) and against trayectorias-esperadas.json. Exit 0 = every line as
           planned; 3 = do not transport.
  compare  after the runs: the stage-1 contracts of every run (K6) and of every A/B pair
           (K9 a), plus the files of the executor, then the muon-by-muon A/B comparison per
           seed, split by the planned side: material and length in Z, energy at the Z
           boundaries, detection transition and pixel, consistency over the three seeds.
           Exit 0 = the contracts hold; 3 = blocked (the outputs are still written, marked).

The material of B is a RESULT here, not a contract: the complement asks whether B assigns the
whole of Z to the dense body when the muon arrives through the conduit, with 1 um as the initial
tolerance inherited from T-S. A violation is reported per row together with the lateral
displacement scale of its steps near Z (plan C1), which may explain it but is not a bound.

Python >= 3.9, numpy >= 1.23. Reuses oe1_stage1_common, oe1_stage1_analysis and
oe1_stage1_synthetic (same folder) and crossing_reference.py.
"""
import argparse
import collections
import csv
import io
import json
import math
import os
import sys

import numpy as np

import oe1_stage1_analysis as A
import oe1_stage1_common as C
import oe1_stage1_synthetic as S

LIST_NAME = "geo-train-06_v0_r10_r201.primarios.csv"
EXPECTED_NAME = "trayectorias-esperadas.json"
MASK_NAME = "R_int.mask.txt"
CASE = "geo-train-06"
CASE_CODE = 6
REPLICA = 10
N_MUONS = 12
EKIN_GEV = 10000.0
SEEDS = (1026093011, 1026093012, 1026093013)
VARIANTS = ("A", "B")
SIDE_CONDUIT = "desde_conducto"
SIDE_BODY = "desde_cuerpo_denso"
PLANNED_SIDE = {i: (SIDE_CONDUIT if i < 6 else SIDE_BODY) for i in range(N_MUONS)}
# labels of trayectorias-esperadas.json (the director names the dense-body side after the ellipsoid)
JSON_SIDE = {"desde_conducto": SIDE_CONDUIT, "desde_elipsoide": SIDE_BODY}
# acceptance of the straight lines (COMPLEMENTO_R2_01.md: more than 20 m inside Z and more than 5 m
# between the entries into the two primitives)
MIN_LZ_M = 20.0
MIN_SEPARATION_M = 5.0
POS_TOL_M = 1e-6
RING_RADII_M = (0.025, 0.05)
RING_N = 360
UM_M = A.LZ_TOL_UM_M
# stage-1 manifest: the historical overlap of A on geo-train-06
EXPECTED_WARNINGS_A = "GeomVol1002=1"
# the material of the step before the first deep piece in Z tells the side the navigator came from in A
SIDE_BY_MAT_IN_A = {"A0": SIDE_CONDUIT, "A1": SIDE_BODY}
CATEGORICAL = ("crossed", "mat_in", "mat_out", "status", "detected", "iel", "iaz")
EXIT_OK, EXIT_USAGE, EXIT_BLOCKED = 0, 2, 3


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def num(v):
    """Shortest round-trip text of a float; empty for None or NaN."""
    if v is None:
        return ""
    v = float(v)
    return "" if math.isnan(v) else repr(v)


def text(v):
    return v.decode("ascii") if isinstance(v, bytes) else str(v)


def write_new(path, data):
    with open(path, "xb") as fh:
        fh.write(data)


def write_json(path, obj):
    write_new(path, (json.dumps(C.to_py(obj), indent=1, ensure_ascii=False) + "\n").encode("utf-8"))


def write_table(path, header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow([r[h] for h in header])
    write_new(path, buf.getvalue().encode("utf-8"))


def span(values):
    """'v' when all equal, else 'min..max' (floats), over the finite values."""
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    return num(lo) if lo == hi else "%s..%s" % (num(lo), num(hi))


def tool_record():
    paths = [os.path.abspath(__file__)] + list(C.tool_paths())
    return {"scripts_sha256": C.script_hashes(paths), "entorno": C.env_versions()}


# --------------------------------------------------------------------------
# lines: geometry of the frozen straight lines, before any transport
# --------------------------------------------------------------------------
def side_of(clo, elo):
    """Side from which a line that reaches Z enters it: the primitive entered LAST bounds Z there.

    Entering the cylinder first means the line is inside the conduit when it crosses the ellipsoid
    surface into Z (conduit side); entering the ellipsoid first, the reverse (dense-body side)."""
    if clo < elo:
        return SIDE_CONDUIT
    if elo < clo:
        return SIDE_BODY
    return "simultanea"


def segment_intervals(cr, cyl, ell, a, b):
    """Fractions of the segments a -> b (rows) inside each primitive and inside Z."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    d = b - a
    clo, chi = cr._cylinder_interval(a, d, cyl)
    elo, ehi = cr._ellipsoid_interval(a, d, ell)
    zlo = np.maximum(np.maximum(clo, elo), 0.0)
    zhi = np.minimum(np.minimum(chi, ehi), 1.0)
    length = np.where(zhi > zlo, zhi - zlo, 0.0) * np.sqrt((d * d).sum(axis=1))
    return {"clo": clo, "chi": chi, "elo": elo, "ehi": ehi, "zlo": zlo, "zhi": zhi, "len": length}


def ring_offsets(direction, radius, n=RING_N):
    e1, e2 = S._perp_basis(np.asarray(direction, dtype=np.float64))
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return radius * (np.cos(ang)[:, None] * e1[None, :] + np.sin(ang)[:, None] * e2[None, :])


def line_report(cr, cyl, ell, p0, b):
    """Side, depth, length, separation, entry/exit and ring sensitivity of the segment p0 -> b."""
    p0 = np.asarray(p0, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    g = segment_intervals(cr, cyl, ell, p0, b)
    seg_len = float(np.linalg.norm(b - p0))
    clo, elo = float(g["clo"][0]), float(g["elo"][0])
    zlo, zhi = float(g["zlo"][0]), float(g["zhi"][0])
    deep = bool(cr.model(np.hstack([p0, b])[None, :])["deep"][0])
    side = side_of(clo, elo) if g["len"][0] > 0 else "sin_cruce"
    direction = (b - p0) / seg_len
    offs = np.vstack([ring_offsets(direction, r) for r in RING_RADII_M])
    gr = segment_intervals(cr, cyl, ell, p0[None, :] + offs, b[None, :] + offs)
    ring_sides = [side_of(c, e) if ln > 0 else "sin_cruce" for c, e, ln in zip(gr["clo"], gr["elo"], gr["len"])]
    hit = bool(g["len"][0] > 0)
    return {"lado": side, "profundo": deep, "lz_m": float(g["len"][0]),
            "separacion_entradas_m": abs(elo - clo) * seg_len if hit else None,
            "parte_fuera_de_ambas_primitivas": bool(clo > 0.0 and elo > 0.0),
            "entrada_Z_m": (p0 + zlo * (b - p0)).tolist() if hit else None,
            "salida_Z_m": (p0 + zhi * (b - p0)).tolist() if hit else None,
            "anillo": {"radios_m": list(RING_RADII_M), "rectas": int(len(offs)),
                       "conservan_lado": int(sum(1 for s in ring_sides if s == side)),
                       "lz_min_m": float(gr["len"].min()), "lz_max_m": float(gr["len"].max())}}


def check_lines(list_path, expected_path, catalog_path, cr_path=None):
    cr, cr_file, cr_sha = C.load_crossing_reference(cr_path)
    cat = C.read_catalog(catalog_path)
    cyl, ell = C.region_from_catalog(cat)
    same_primitives = (all(float(cyl[k]) == float(cr.CYL[k]) for k in cr.CYL) and
                       all(float(ell[k]) == float(cr.ELL[k]) for k in cr.ELL))
    table, defects, list_sha, canon = C.read_primary_file(list_path)
    with open(expected_path, "rb") as fh:
        expected = json.loads(fh.read().decode("utf-8"))
    exp_by_id = {int(e["id"]): e for e in expected.get("muones", [])}
    det = np.array(C.detector_center(0))
    r_start = C.DET_DIST_M + cat["base_radius_m"] + 300.0
    rows = []
    for k in range(len(table["id"])):
        i = int(table["id"][k])
        v = table["vals"][k]
        p0, d, ekin = v[:3], v[3:6], float(v[6])
        t_det = float(np.dot(det - p0, d))
        b = p0 + t_det * d                          # point of the line closest to the panel centre
        rep = line_report(cr, cyl, ell, p0, b)
        e = exp_by_id.get(i, {})
        exp_lz = e.get("longitud_recta_Z_m")
        checks = collections.OrderedDict([
            ("id_en_orden", i == k),
            ("mu_menos_de_10_TeV", ekin == EKIN_GEV and int(table["pdg"][k]) == 13 and int(table["charge"][k]) == -1),
            ("pasa_por_el_centro_del_panel", float(np.linalg.norm(det - b)) <= POS_TOL_M and t_det > 0.0),
            ("radio_de_partida", abs(math.hypot(p0[0] - det[0], p0[1] - det[1]) - r_start) <= POS_TOL_M),
            ("parte_fuera_de_ambas_primitivas", rep["parte_fuera_de_ambas_primitivas"]),
            ("cruce_profundo", rep["profundo"]),
            ("lado_planificado", rep["lado"] == PLANNED_SIDE.get(i)),
            ("lado_del_json", rep["lado"] == JSON_SIDE.get(e.get("lado_entrada_recta"))),
            ("lz_mayor_que_20_m", rep["lz_m"] > MIN_LZ_M),
            ("separacion_mayor_que_5_m", (rep["separacion_entradas_m"] or 0.0) > MIN_SEPARATION_M),
            ("lz_igual_al_json", exp_lz is not None and abs(rep["lz_m"] - exp_lz) <= POS_TOL_M),
            ("entrada_igual_al_json", "entrada_Z_m" in e and rep["entrada_Z_m"] is not None and
             float(np.abs(np.array(rep["entrada_Z_m"]) - np.array(e["entrada_Z_m"])).max()) <= POS_TOL_M),
            ("salida_igual_al_json", "salida_Z_m" in e and rep["salida_Z_m"] is not None and
             float(np.abs(np.array(rep["salida_Z_m"]) - np.array(e["salida_Z_m"])).max()) <= POS_TOL_M),
            ("anillo_conserva_el_lado", rep["anillo"]["conservan_lado"] == rep["anillo"]["rectas"]),
        ])
        rows.append({"id": i, "lado_planificado": PLANNED_SIDE.get(i), "medicion": rep,
                     "json": {k_: e.get(k_) for k_ in ("lado_entrada_recta", "longitud_recta_Z_m",
                                                       "separacion_entradas_primitivas_m",
                                                       "longitudes_9_rectas_desplazadas_min_max_m")},
                     "comprobaciones": checks, "ok": all(checks.values())})
    list_ok = (C.primary_defects_clean(defects, len(table["id"])) and len(table["id"]) == N_MUONS)
    ok = list_ok and same_primitives and len(rows) == N_MUONS and all(r["ok"] for r in rows)
    return {"formato": "oe1-r201-rectas/1", "ok": ok,
            "lista": {"archivo": os.path.basename(list_path), "sha256": list_sha, "canon_fnv1a64": C.hex16(canon),
                      "estructura_correcta": list_ok, "n": int(len(table["id"]))},
            "catalogo": {"archivo": os.path.basename(catalog_path), "sha256": cat["sha256"],
                         "primitivas_iguales_a_crossing_reference": same_primitives,
                         "cilindro": cyl, "elipsoide": ell},
            "crossing_reference": {"ruta": cr_file, "sha256": cr_sha},
            "esperado": {"archivo": os.path.basename(expected_path), "sha256": C.sha256_file(expected_path)},
            "criterios": {"lz_min_m": MIN_LZ_M, "separacion_min_m": MIN_SEPARATION_M, "tolerancia_m": POS_TOL_M,
                          "anillos_m": list(RING_RADII_M), "rectas_por_anillo": RING_N,
                          "lado": "el de la primitiva en la que la recta entra en último lugar"},
            "muones": rows, "herramienta": tool_record()}


def cmd_lines(args):
    res = check_lines(args.list, args.expected, args.catalog, args.crossing_reference)
    write_json(args.out, res)
    bad = [r["id"] for r in res["muones"] if not r["ok"]]
    print("R201-LINES ok=%d muones=%d fallan=%s lista=%s" % (int(res["ok"]), len(res["muones"]),
                                                            bad or "ninguno", res["lista"]["sha256"]))
    return EXIT_OK if res["ok"] else EXIT_BLOCKED


# --------------------------------------------------------------------------
# compare: the six runs of the executor
# --------------------------------------------------------------------------
def make_entry(variant, seed, list_sha):
    return {"run_id": "%s-%d" % (variant, seed), "salida": "%s/%d" % (variant, seed), "caso": CASE,
            "variante": variant, "case_code": CASE_CODE, "replica": REPLICA, "N": N_MUONS, "lista": LIST_NAME,
            "lista_sha256": list_sha, "s_tr": seed, "s_src": None, "rol": A.ROLE_MAIN,
            "avisos_esperados": EXPECTED_WARNINGS_A if variant == "A" else ""}


def read_estado(attempt):
    """{(variant, seed): estado} from the executor's estado.tsv, or None if it cannot be read."""
    path = os.path.join(attempt, "estado.tsv")
    if not os.path.isfile(path):
        return None
    rows = list(csv.reader(io.StringIO(A.read_text(path)), delimiter="\t"))
    out = {}
    for r in rows[1:]:
        if len(r) >= 4:
            try:
                out[(r[0], int(r[1]))] = r[3]
            except ValueError:
                out[(r[0], r[1])] = r[3]
    return out


def load_runs(attempt, list_sha, estado):
    runs = {}
    for v in VARIANTS:
        for s in SEEDS:
            run = A.load_run(make_entry(v, s, list_sha), attempt)
            # the executor names the checksum file SHA256SUMS.txt and records its verdict in estado.tsv:
            # PROCESO_OK is written only after the process exited 0 (any other code stops the executor)
            sums = os.path.join(run.dir, "SHA256SUMS.txt")
            if "SHA256SUMS" not in run.files and os.path.isfile(sums):
                run.files["SHA256SUMS"] = sums
            st = (estado or {}).get((v, s))
            run.status = {"run_id": run.run_id, "veredicto": "OK" if st == "PROCESO_OK" else (st or "ausente"),
                          "rc": "0" if st == "PROCESO_OK" else "", "motivo": st or "", "nodo": None,
                          "trabajo": None}
            runs[(v, s)] = run
    return runs


def attempt_contracts(attempt, contracts, list_sha, canon_hex, estado):
    scope = "intento"
    contracts.add("R201", scope, "FINALIZADO.txt presente (el ejecutor terminó las 6 corridas)",
                  os.path.isfile(os.path.join(attempt, "FINALIZADO.txt")), True, {})
    want = {(v, s) for v in VARIANTS for s in SEEDS}
    ok_estado = estado is not None and set(estado) == want and all(x == "PROCESO_OK" for x in estado.values())
    contracts.add("R201", scope, "estado.tsv: las 6 corridas (A, B x 3 semillas) con PROCESO_OK", ok_estado, True,
                  {"estado": {"%s-%s" % k: x for k, x in (estado or {}).items()}})
    checks = {}
    for v in ("A2", "B2"):
        p = os.path.join(attempt, "check-%s.txt" % v)
        checks[v] = A.read_text(p) if os.path.isfile(p) else None
    kv = {}
    if checks["A2"]:
        line = C.find_line(checks["A2"], "OE1-CHECK-PRIMARIES")
        kv = C.parse_kv_line(line) if line else {}
    same = checks["A2"] is not None and checks["A2"] == checks["B2"]
    good = (kv.get("status") == "OK" and kv.get("n") == str(N_MUONS) and kv.get("file_sha256") == list_sha and
            kv.get("canon_fnv1a64") == canon_hex)
    contracts.add("R201", scope, "check-A2.txt == check-B2.txt, con status=OK, n=12, la huella de la lista y su FNV "
                  "canónico", same and good, True,
                  {"identicos": same, "linea": {k: kv.get(k) for k in ("status", "n", "file_sha256", "canon_fnv1a64")}})


def material_b(mu, k):
    """1 um criterion of T-S for B, with the lateral displacement scale of its steps near Z (not a bound)."""
    lz, a0, a1 = mu["lz_m"][k], mu["lz_a0_m"][k], mu["lz_a1_m"][k]
    rock, other = mu["lz_rock_m"][k], mu["lz_other_m"][k]
    vals = (lz, a0, a1, rock, other)
    if any(math.isnan(float(x)) for x in vals):
        return {"ok": False, "fuera_de_A1_m": None, "escala_lateral_m": None, "explicado_por_escala": False}
    outside = float(a0 + rock + other)
    ok = bool(a0 <= UM_M and rock <= UM_M and other <= UM_M and abs(lz - a1) <= UM_M)
    scale = float(A.lateral_length_scale_m(mu["sum_lat_z_mm"][k])) if "sum_lat_z_mm" in mu else None
    explained = (not ok) and scale is not None and outside <= scale
    return {"ok": ok, "fuera_de_A1_m": outside, "escala_lateral_m": scale, "explicado_por_escala": explained}


def observed_side_a(mu, k):
    if mu["crossed"][k] != 1.0:
        return "sin_cruce"
    return SIDE_BY_MAT_IN_A.get(text(mu["mat_in"][k]), "otro:" + text(mu["mat_in"][k]))


PAIR_FIELDS = ("crossed", "mat_in", "mat_out", "lz_m", "lz_a0_m", "lz_a1_m", "lz_rock_m", "lz_other_m",
               "e_in_gev", "e_out_gev", "max_lat_z_mm", "sum_lat_z_mm", "status", "detected", "iel", "iaz",
               "n_entries_z")


def field(mu, name, k):
    v = mu[name][k]
    if mu[name].dtype.kind == "S":
        return text(v)
    if mu[name].dtype.kind in "iu":
        return int(v)
    return num(v)


def pair_rows(runs, expected):
    exp_by_id = {int(e["id"]): e for e in expected.get("muones", [])}
    rows = []
    for s in SEEDS:
        ra, rb = runs[("A", s)], runs[("B", s)]
        if ra.mu is None or rb.mu is None:
            continue
        a, b = ra.mu, rb.mu
        for k in range(min(len(a["id"]), len(b["id"]))):
            i = int(a["id"][k])
            r = collections.OrderedDict()
            r["id"] = i
            r["lado_planificado"] = PLANNED_SIDE.get(i, "")
            r["semilla"] = s
            r["prim_fnv_igual"] = int(a["prim_fnv"][k] == b["prim_fnv"][k])
            r["rng0_igual"] = int(a["rng0"][k] == b["rng0"][k])
            for tag, mu in (("A", a), ("B", b)):
                for f in PAIR_FIELDS:
                    r["%s_%s" % (tag, f)] = field(mu, f, k)
            r["lado_observado_A"] = observed_side_a(a, k)
            mb = material_b(b, k)
            r["B_material_ok"] = int(mb["ok"])
            r["B_fuera_de_A1_m"] = num(mb["fuera_de_A1_m"])
            r["B_escala_lateral_m"] = num(mb["escala_lateral_m"])
            r["B_explicado_por_escala"] = int(mb["explicado_por_escala"])
            r["A_identidad_ok"] = int(abs(float(a["lz_a0_m"][k] + a["lz_a1_m"][k] + a["lz_rock_m"][k] +
                                                a["lz_other_m"][k] - a["lz_m"][k])) <= 1e-9)
            r["transicion"] = "%d%d" % (int(a["detected"][k]), int(b["detected"][k]))
            both = a["detected"][k] == 1 and b["detected"][k] == 1
            r["delta_iel"] = int(b["iel"][k] - a["iel"][k]) if both else ""
            r["delta_iaz"] = int(b["iaz"][k] - a["iaz"][k]) if both else ""
            r["mismo_pixel"] = int(bool(both and r["delta_iel"] == 0 and r["delta_iaz"] == 0)) if both else ""
            for f in ("lz_m", "lz_a0_m", "lz_a1_m", "e_in_gev", "e_out_gev"):
                r["delta_" + f] = num(b[f][k] - a[f][k])
            ref = exp_by_id.get(i, {}).get("longitud_recta_Z_m")
            r["lz_recta_m"] = num(ref)
            r["A_lz_menos_recta_m"] = num(a["lz_m"][k] - ref) if ref is not None else ""
            r["B_lz_menos_recta_m"] = num(b["lz_m"][k] - ref) if ref is not None else ""
            rows.append(r)
    return rows


def consistency(runs, variant, k):
    """Categorical fields that are not identical over the three seeds (empty list = consistent)."""
    vary = []
    mus = [runs[(variant, s)].mu for s in SEEDS]
    if any(m is None or len(m["id"]) <= k for m in mus):
        return ["sin_datos"]
    for f in CATEGORICAL:
        vals = set(field(m, f, k) for m in mus)
        if len(vals) > 1:
            vary.append(f)
    return vary


def trajectory_rows(runs, pairs):
    rows = []
    by_id = collections.defaultdict(list)
    for p in pairs:
        by_id[p["id"]].append(p)
    for i in range(N_MUONS):
        ps = by_id.get(i, [])
        r = collections.OrderedDict()
        r["id"] = i
        r["lado_planificado"] = PLANNED_SIDE[i]
        r["lz_recta_m"] = ps[0]["lz_recta_m"] if ps else ""
        sides = [p["lado_observado_A"] for p in ps]
        r["lado_observado_A"] = sides[0] if sides and len(set(sides)) == 1 else "variable:" + "/".join(sides)
        r["A_mat_in"] = "/".join(p["A_mat_in"] for p in ps)
        r["A_lz_a0_m"] = span(float(p["A_lz_a0_m"] or "nan") for p in ps)
        r["A_lz_a1_m"] = span(float(p["A_lz_a1_m"] or "nan") for p in ps)
        r["B_mat_in"] = "/".join(p["B_mat_in"] for p in ps)
        r["B_lz_a1_m"] = span(float(p["B_lz_a1_m"] or "nan") for p in ps)
        r["B_material_ok"] = "%d/%d" % (sum(p["B_material_ok"] for p in ps), len(ps))
        r["B_fuera_de_A1_max_m"] = span([max((float(p["B_fuera_de_A1_m"] or "nan") for p in ps), default=float("nan"))])
        r["delta_e_out_GeV"] = span(float(p["delta_e_out_gev"] or "nan") for p in ps)
        r["transiciones"] = "/".join(p["transicion"] for p in ps)
        r["pixel_A"] = "/".join("%s,%s" % (p["A_iel"], p["A_iaz"]) for p in ps)
        r["pixel_igual_AB"] = "%d/%d" % (sum(1 for p in ps if p["mismo_pixel"] == 1), len(ps))
        r["max_lat_z_mm_max"] = span([max((float(p[t + "_max_lat_z_mm"] or "nan") for p in ps for t in VARIANTS),
                                          default=float("nan"))])
        for v in VARIANTS:
            vary = consistency(runs, v, i)
            r["consistente_3_semillas_" + v] = "si" if not vary else "no:" + "+".join(vary)
        rows.append(r)
    return rows


def coverage(pairs, trajectories):
    out = collections.OrderedDict()
    for side in (SIDE_CONDUIT, SIDE_BODY):
        ps = [p for p in pairs if p["lado_planificado"] == side]
        ts = [t for t in trajectories if t["lado_planificado"] == side]
        out[side] = {
            "ids": [t["id"] for t in ts],
            "pares": len(ps),
            "pares_con_cruce_en_A_y_B": sum(1 for p in ps if p["A_crossed"] == "1.0" and p["B_crossed"] == "1.0"),
            "pares_con_lado_observado_en_A_igual_al_planificado": sum(1 for p in ps if p["lado_observado_A"] == side),
            "ids_con_lado_en_A_igual_al_planificado_en_las_3_semillas":
                sum(1 for t in ts if t["lado_observado_A"] == side),
            "pares_con_material_de_B_correcto": sum(p["B_material_ok"] for p in ps),
            "pares_de_B_fuera_de_1um_explicados_por_escala_lateral": sum(p["B_explicado_por_escala"] for p in ps),
            "transiciones": dict(collections.Counter(p["transicion"] for p in ps)),
            "pares_detectados_en_ambas_con_mismo_pixel": sum(1 for p in ps if p["mismo_pixel"] == 1),
            "ids_consistentes_en_3_semillas_A": sum(1 for t in ts if t["consistente_3_semillas_A"] == "si"),
            "ids_consistentes_en_3_semillas_B": sum(1 for t in ts if t["consistente_3_semillas_B"] == "si"),
            "A_identidad_ok": sum(p["A_identidad_ok"] for p in ps),
        }
    return out


def relabel_not_applicable(contracts):
    """K6 compares the detection efficiency of a run with the campaign's (informative). A directed list of
    10 TeV muons is not a sample of the source, so the comparison does not apply here."""
    for c in contracts.checks:
        if c["contrato"] == "K6" and c["descripcion"].startswith("eficiencia de detección compatible"):
            c["ok"] = None
            c["evidencia"]["nota"] = "no aplica: lista dirigida de 10 TeV, no es una muestra de la fuente"


def run_compare(args):
    attempt = os.path.abspath(args.attempt)
    inputs = os.path.abspath(args.inputs)
    list_path = os.path.join(inputs, LIST_NAME)
    table, defects, list_sha, canon = C.read_primary_file(list_path)
    with open(os.path.join(inputs, EXPECTED_NAME), "rb") as fh:
        expected = json.loads(fh.read().decode("utf-8"))
    with open(args.mask, "rb") as fh:
        mask_bytes = fh.read()
    roi = _Roi(C.mask_from_text(mask_bytes), C.sha256_bytes(mask_bytes))
    os.makedirs(args.out, exist_ok=False)
    contracts = A.Contracts()
    contracts.DESCRIPTIONS = dict(A.Contracts.DESCRIPTIONS)
    contracts.DESCRIPTIONS["R201"] = ("Archivos del ejecutor del complemento R2-01: FINALIZADO.txt, las 6 corridas "
                                      "con PROCESO_OK, y A2 y B2 leen la misma lista (check-A2.txt = check-B2.txt)",
                                      True)
    contracts.DESCRIPTIONS["K6"] = (A.Contracts.DESCRIPTIONS["K6"][0] + "; aquí N = 12 por corrida", True)
    estado = read_estado(attempt)
    attempt_contracts(attempt, contracts, list_sha, C.hex16(canon), estado)
    runs = load_runs(attempt, list_sha, estado)
    lists = A.ListCache(inputs)
    for v in VARIANTS:
        for s in SEEDS:
            A.k6_checks(runs[(v, s)], roi, contracts, lists)
    relabel_not_applicable(contracts)
    for s in SEEDS:
        A.k9a_checks(runs[("A", s)], runs[("B", s)], contracts)
    pairs = pair_rows(runs, expected)
    trajectories = trajectory_rows(runs, pairs)
    cov = coverage(pairs, trajectories)
    blocked = bool(contracts.failed_blocking())
    tool = tool_record()
    entradas = {"intento": attempt, "lista": {"archivo": LIST_NAME, "sha256": list_sha, "canon_fnv1a64": C.hex16(canon)},
                "esperado": {"archivo": EXPECTED_NAME, "sha256": C.sha256_file(os.path.join(inputs, EXPECTED_NAME))},
                "mascara": {"archivo": os.path.basename(args.mask), "sha256": roi.text_sha},
                "semillas": list(SEEDS)}
    write_json(os.path.join(args.out, "contratos-r201.json"),
               contracts.to_json(extra={"entradas": entradas, "herramienta": tool}))
    write_table(os.path.join(args.out, "pares-r201.csv"), list(pairs[0].keys()) if pairs else ["id"], pairs)
    write_table(os.path.join(args.out, "trayectorias-r201.csv"), list(trajectories[0].keys()), trajectories)
    write_json(os.path.join(args.out, "resumen-r201.json"),
               {"formato": "oe1-r201-resumen/1", "estado": A.STAMP if blocked else "OK", "cobertura": cov,
                "criterio_material_B": "lz_m = lz_a1_m y lz_a0_m, lz_rock_m, lz_other_m <= 1 µm (T-S); la escala "
                                       "lateral 2·√(2·R_max·Σlat) puede explicar un exceso, pero no es una cota",
                "lado_observado_A": "mat_in del primer tramo profundo en Z: A0 = desde el conducto, A1 = desde el "
                                    "cuerpo denso; en B toda Z es A1 y el lado no se observa en el registro agregado",
                "entradas": entradas, "herramienta": tool})
    cmd = " ".join(A._quote(x) for x in [sys.executable] + sys.argv)
    write_new(os.path.join(args.out, "comando-analisis.txt"), (cmd + "\n").encode("utf-8"))
    for side, c in cov.items():
        print("R201 %s: lado_en_A=%d/%d ids, material_B=%d/%d pares, transiciones=%s" % (
            side, c["ids_con_lado_en_A_igual_al_planificado_en_las_3_semillas"], len(c["ids"]),
            c["pares_con_material_de_B_correcto"], c["pares"], c["transiciones"]))
    print("R201-CONTRATOS %s bloqueantes_fallidos=%d" % ("BLOQUEADO" if blocked else "OK",
                                                         len(contracts.failed_blocking())))
    return EXIT_BLOCKED if blocked else EXIT_OK


class _Roi:
    def __init__(self, rint, text_sha):
        self.rint = rint
        self.text_sha = text_sha


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("lines", help="geometry of the 12 frozen straight lines (before any transport)")
    pl.add_argument("--list", required=True)
    pl.add_argument("--expected", required=True)
    pl.add_argument("--catalog", required=True)
    pl.add_argument("--crossing-reference", default=None)
    pl.add_argument("--out", required=True, help="JSON file to create (must not exist)")
    pc = sub.add_parser("compare", help="contracts and A/B comparison of the six runs")
    pc.add_argument("--attempt", required=True, help="the executor's OUTPUT_DIR (intento-NN)")
    pc.add_argument("--inputs", required=True, help="folder complemento-r2-01-v1 (list, sidecar, expected JSON)")
    pc.add_argument("--mask", required=True, help="R_int.mask.txt the runs used")
    pc.add_argument("--out", required=True, help="output folder to create (must not exist)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "lines":
            return cmd_lines(args)
        return run_compare(args)
    except (OSError, ValueError, KeyError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
