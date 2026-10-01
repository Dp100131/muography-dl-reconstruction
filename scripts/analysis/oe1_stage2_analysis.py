#!/usr/bin/env python3
"""OE1 stage 2 (protocol v1.1): manifest, per-block analysis, controls and aggregation.

The design is fixed by objetivo-1/PROTOCOLO_ETAPA_2_OE1_V1_1.md of the director's repository and its
manifest ensayos/etapa-2-v1.1/corridas.csv (55 runs): 24 main lists of 1,000,000 primaries for
geo-train-06 view 0 (replicas 101-124, A and B), 3 control lists of 100,000 for geo-train-01 view 0
(replicas 201-203, A and B) and one exact A/A repeat of geo-train-01-r201-A. Every stage-1 check is
reused by import from oe1_stage1_analysis / oe1_stage1_common, unchanged.

Subcommands
  plan       checks corridas.csv against the protocol (counts, replicas, N, seeds, the A/A repeat) and
             writes, for the runs of one SLURM job, the list table and the stage-1 run manifest that
             principal.sbatch's run_one reads; --print-commands prints the transport commands (dry run).
  block      one main replica (an A/B pair): K6 of A and B, K9 (a, c), R_int and detection transitions,
             pixel changes, discordant ids and the Z crossings of the union U. Exit 3 if a contract fails.
  controls   the seven control runs: K6, K9 (a), K7 (A/A identical) and K8 (A/B identical on geo-train-01).
             K8 is a gate here (protocol section 7), not a note. Exit 3 if the gate fails.
  aggregate  the 24 block summaries: completeness, sums, the conservative interval of the protocol (the
             director's interval() of herramientas/planificar_etapa2_oe1.py, which needs scipy), the
             between-block spread (descriptive only), maps and tables.

Python >= 3.9 and numpy >= 1.23 (plan, block, controls run on SC3); aggregate also needs scipy and,
for the figures, matplotlib.
"""
import argparse
import collections
import csv
import glob
import importlib.util
import io
import json
import math
import os
import re
import sys

import numpy as np

import oe1_stage1_analysis as A
import oe1_stage1_common as C

MAIN_CASE, CONTROL_CASE = "geo-train-06", "geo-train-01"
MAIN_REPLICAS = tuple(range(101, 125))
CONTROL_REPLICAS = (201, 202, 203)
MAIN_N, CONTROL_N = 1000000, 100000
SOURCE_BASE = {MAIN_CASE: 2026100100, CONTROL_CASE: 2026100200}
REPLICA_BASE = {MAIN_CASE: 100, CONTROL_CASE: 200}
CASE_CODES = {MAIN_CASE: 6, CONTROL_CASE: 1}
N_RUNS, N_TRANSPORTS, N_PAIRS_MAIN = 55, 48700000, 24000000
AA_ID, AA_OF = "geo-train-01-r201-AA", "geo-train-01-r201-A"
CORRIDAS_COLUMNS = ("run_id", "geometry_id", "case_code", "view", "replica", "n", "source_seed",
                    "transport_seed", "variant", "repeat_of")
EXPECTED_WARNINGS_A_MAIN = "GeomVol1002=1"
EXIT_OK, EXIT_USAGE, EXIT_BLOCKED = 0, 2, 3


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
class Roi:
    def __init__(self, mask_path):
        with open(mask_path, "rb") as fh:
            data = fh.read()
        self.rint = C.mask_from_text(data)
        self.text_sha = C.sha256_bytes(data)
        self.path = mask_path


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
        w.writerow([r.get(h, "") for h in header])
    write_new(path, buf.getvalue().encode("utf-8"))


def num(v):
    if v is None:
        return ""
    v = float(v)
    return "" if math.isnan(v) else repr(v)


def text(v):
    return v.decode("ascii") if isinstance(v, bytes) else str(v)


def tool_record():
    paths = [os.path.abspath(__file__)] + list(C.tool_paths())
    return {"scripts_sha256": C.script_hashes(paths), "entorno": C.env_versions()}


def list_name(geometry_id, replica, source_seed):
    return "%s_v0_r%d_s%d.primarios.csv" % (geometry_id, replica, source_seed)


def run_folder(row):
    label = "AA" if row["repeat_of"] else row["variant"]
    return "%s/r%d/%s" % (row["geometry_id"], row["replica"], label)


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
def read_corridas(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or tuple(rows[0].keys()) != CORRIDAS_COLUMNS:
        raise ValueError("corridas.csv: columnas inesperadas %r" % (tuple(rows[0].keys()) if rows else ()))
    out = []
    for r in rows:
        d = dict(r)
        for k in ("case_code", "view", "replica", "n", "source_seed", "transport_seed"):
            d[k] = int(d[k])
        out.append(d)
    return out


def check_corridas(rows):
    """Problems of corridas.csv against the protocol (empty list = as designed) and its totals."""
    probs = []
    ids = [r["run_id"] for r in rows]
    if len(rows) != N_RUNS or len(set(ids)) != len(ids):
        probs.append("se esperaban %d corridas con run_id únicos; hay %d (%d únicos)" % (N_RUNS, len(ids), len(set(ids))))
    want = set()
    for case, reps, n in ((MAIN_CASE, MAIN_REPLICAS, MAIN_N), (CONTROL_CASE, CONTROL_REPLICAS, CONTROL_N)):
        for rep in reps:
            for v in ("A", "B"):
                want.add(("%s-r%d-%s" % (case, rep, v), case, rep, v, n, ""))
    want.add((AA_ID, CONTROL_CASE, 201, "A", CONTROL_N, AA_OF))
    have = set((r["run_id"], r["geometry_id"], r["replica"], r["variant"], r["n"], r["repeat_of"]) for r in rows)
    if have != want:
        probs.append("corridas distintas del diseño: faltan %s; sobran %s" % (sorted(want - have)[:5], sorted(have - want)[:5]))
    for r in rows:
        case = r["geometry_id"]
        if case not in SOURCE_BASE:
            probs.append("%s: geometría desconocida %s" % (r["run_id"], case))
            continue
        exp_src = SOURCE_BASE[case] + (r["replica"] - REPLICA_BASE[case])
        if r["source_seed"] != exp_src or r["transport_seed"] != exp_src - 1000000000:
            probs.append("%s: semillas %d/%d distintas de %d/%d" % (r["run_id"], r["source_seed"], r["transport_seed"],
                                                                  exp_src, exp_src - 1000000000))
        if r["case_code"] != CASE_CODES[case] or r["view"] != 0:
            probs.append("%s: case_code o vista inesperados" % r["run_id"])
    transports = sum(r["n"] for r in rows)
    pairs = sum(r["n"] for r in rows if r["geometry_id"] == MAIN_CASE and r["variant"] == "A")
    if transports != N_TRANSPORTS:
        probs.append("transportes %d != %d" % (transports, N_TRANSPORTS))
    if pairs != N_PAIRS_MAIN:
        probs.append("pares principales %d != %d" % (pairs, N_PAIRS_MAIN))
    return probs, {"corridas": len(rows), "transportes": transports, "pares_principales": pairs,
                   "listas": len(set((r["geometry_id"], r["replica"]) for r in rows))}


def read_sha_file(path):
    """{file name: sha256} of a sha256sum-style file (the path part is reduced to its base name)."""
    out = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            sha, name = line.split("  ", 1) if "  " in line else line.split(" *", 1)
            out[re.split(r"[\\/]", name)[-1]] = sha
    return out


def manifest_entries(rows, lists_dir, list_sha, catalog_dir, bin_a, bin_b):
    """Stage-1 manifest entries (oe1_stage1_analysis.MANIFEST_COLUMNS) for the given corridas rows."""
    out = []
    for r in rows:
        name = list_name(r["geometry_id"], r["replica"], r["source_seed"])
        out.append({
            "run_id": r["run_id"], "caso": r["geometry_id"],
            "geometria": os.path.join(catalog_dir, r["geometry_id"] + ".json"), "variante": r["variant"],
            "binario": bin_a if r["variant"] == "A" else bin_b, "s_src": r["source_seed"],
            "s_tr": r["transport_seed"], "case_code": r["case_code"], "lista": os.path.join(lists_dir, name),
            "lista_sha256": list_sha.get(name, "SIN_SHA_CONGELADO"), "N": r["n"], "salida": run_folder(r),
            "rol": A.ROLE_AA if r["repeat_of"] else A.ROLE_MAIN,
            "avisos_esperados": EXPECTED_WARNINGS_A_MAIN if (r["variant"] == "A" and r["geometry_id"] == MAIN_CASE)
            else "ninguno"})
    return out


def transport_command(e, mask, mask_sha, tables):
    region = ["--region-pair", "0,1"] if e["caso"] == MAIN_CASE else ["--region-pair", "none"]
    rep = re.search(r"-r(\d+)", e["run_id"]).group(1)
    return ([e["binario"], "--primaries", e["lista"], "--primaries-sha256", e["lista_sha256"], "--roi-mask", mask,
             "--roi-mask-sha256", mask_sha, "--transport-seed", str(e["s_tr"]), "--variant", e["variante"],
             "--case-code", str(e["case_code"]), "--replica", rep] + region +
            ["--muon-log", "full", "--geometry", e["geometria"], "--view", "0", "--tables", tables, "--seed",
             str(e["s_tr"]), "--threads", "1", "--out", "<intento>/" + e["salida"]])


def cmd_plan(args):
    rows = read_corridas(args.corridas)
    probs, totals = check_corridas(rows)
    print("R2-PLAN corridas=%d transportes=%d pares_principales=%d listas=%d problemas=%d" % (
        totals["corridas"], totals["transportes"], totals["pares_principales"], totals["listas"], len(probs)))
    for p in probs:
        print("PROBLEMA: " + p)
    if probs:
        return EXIT_BLOCKED
    sel = rows
    if args.replicas:
        want = set(int(x) for x in args.replicas.split(","))
        sel = [r for r in rows if r["replica"] in want]
        if not sel:
            print("ninguna corrida con esas réplicas", file=sys.stderr)
            return EXIT_USAGE
    if args.lists_tsv:
        seen, lines = set(), ["nombre\tgeometria\tcase_code\treplica\tsemilla_fuente\tn"]
        for r in sel:
            key = (r["geometry_id"], r["replica"])
            if key in seen:
                continue
            seen.add(key)
            lines.append("%s\t%s\t%d\t%d\t%d\t%d" % (list_name(r["geometry_id"], r["replica"], r["source_seed"]),
                                                     r["geometry_id"], r["case_code"], r["replica"], r["source_seed"], r["n"]))
        write_new(args.lists_tsv, ("\n".join(lines) + "\n").encode("utf-8"))
    if args.manifest_tsv or args.print_commands:
        list_sha = read_sha_file(args.list_sha) if args.list_sha else {}
        entries = manifest_entries(sel, args.lists_dir or "<listas>", list_sha, args.catalog_dir or "<catalogo>",
                                   args.bin_a or "<A2>", args.bin_b or "<B2>")
        if args.manifest_tsv:
            missing = [e["run_id"] for e in entries if e["lista_sha256"] == "SIN_SHA_CONGELADO"]
            if missing:
                print("faltan SHA congelados para %s" % missing[:5], file=sys.stderr)
                return EXIT_BLOCKED
            text_ = "\t".join(A.MANIFEST_COLUMNS) + "\n" + "".join(
                "\t".join(str(e[c]) for c in A.MANIFEST_COLUMNS) + "\n" for e in entries)
            write_new(args.manifest_tsv, text_.encode("utf-8"))
        if args.print_commands:
            for e in entries:
                print("ORDEN %s: %s" % (e["run_id"], " ".join(transport_command(
                    e, args.mask or "<R_int.mask.txt>", args.mask_sha or "<sha>", args.tables or "<tablas>"))))
    return EXIT_OK


# --------------------------------------------------------------------------
# block and controls
# --------------------------------------------------------------------------
def select(entries, caso, replica, variant=None, rol=A.ROLE_MAIN):
    out = [e for e in entries if e["caso"] == caso and e["replica"] == replica and e["rol"] == rol and
           (variant is None or e["variante"] == variant)]
    return out


def resources(run):
    m = run.meta or {}
    st = run.status or {}
    return {"corrida": run.run_id, "pared_s": m.get("wall_seconds"), "cpu_usuario_s": m.get("rusage_user_cpu_seconds"),
            "cpu_sistema_s": m.get("rusage_system_cpu_seconds"), "rss_max_kb": m.get("rusage_max_rss_kb"),
            "nodo": st.get("nodo"), "trabajo": st.get("trabajo"), "veredicto_lanzador": st.get("veredicto")}


CROSS_FIELDS = ("crossed", "mat_in", "mat_out", "lz_m", "lz_a0_m", "lz_a1_m", "lz_rock_m", "lz_other_m",
                "e_in_gev", "e_out_gev", "max_lat_z_mm", "sum_lat_z_mm", "status", "detected", "iel", "iaz", "in_rint")


def field(mu, name, k):
    if name not in mu:
        return ""
    v = mu[name][k]
    if mu[name].dtype.kind == "S":
        return text(v)
    if mu[name].dtype.kind in "iu":
        return int(v)
    return num(v)


def pair_details(pair, roi, ya, yb, union):
    a, b = pair.a.mu, pair.b.mu
    both = pair.det_a & pair.det_b
    moved = both & ((a["iel"] != b["iel"]) | (a["iaz"] != b["iaz"]))
    roi_moved = ya & yb & moved
    anyd, fields = A.discordance_table(pair.a, pair.b, roi)
    idx = np.flatnonzero(anyd)
    classes = A.classify_discordance(pair.a, pair.b, idx) if (pair.caso == MAIN_CASE and len(idx)) else \
        np.array(["no_aplica"] * len(idx), dtype=object)
    disc_rows = []
    for k, cls in zip(idx, classes):
        disc_rows.append({"replica": pair.replica, "id": int(a["id"][k]), "clase": cls,
                          "campos": "+".join(f for f, m in fields.items() if m[k]),
                          "Y_A": int(ya[k]), "Y_B": int(yb[k]),
                          "estado_A": text(a["status"][k]), "estado_B": text(b["status"][k]),
                          "iel_A": int(a["iel"][k]), "iaz_A": int(a["iaz"][k]),
                          "iel_B": int(b["iel"][k]), "iaz_B": int(b["iaz"][k]),
                          "cruza_A": field(a, "crossed", k), "cruza_B": field(b, "crossed", k)})
    cross_rows = []
    for k in np.flatnonzero(union):
        r = collections.OrderedDict([("replica", pair.replica), ("id", int(a["id"][k]))])
        for tag, mu in (("A", a), ("B", b)):
            for f in CROSS_FIELDS:
                r["%s_%s" % (tag, f)] = field(mu, f, k)
        mat_in_a = text(a["mat_in"][k]) if "mat_in" in a else ""
        r["lado_observado_A"] = {"A0": "conducto", "A1": "cuerpo_denso"}.get(mat_in_a, "sin_cruce_en_A" if
                                                                            a["crossed"][k] != 1 else "otro")
        lz, a0, a1 = b["lz_m"][k], b["lz_a0_m"][k], b["lz_a1_m"][k]
        rock, other = b["lz_rock_m"][k], b["lz_other_m"][k]
        if b["crossed"][k] == 1:
            r["B_material_A1_1um"] = int(bool(a0 <= A.LZ_TOL_UM_M and rock <= A.LZ_TOL_UM_M and
                                              other <= A.LZ_TOL_UM_M and abs(lz - a1) <= A.LZ_TOL_UM_M))
        else:
            r["B_material_A1_1um"] = ""
        r["Y_A"], r["Y_B"] = int(ya[k]), int(yb[k])
        cross_rows.append(r)
    move_rows = [{"replica": pair.replica, "id": int(a["id"][k]), "iel_A": int(a["iel"][k]), "iaz_A": int(a["iaz"][k]),
                  "iel_B": int(b["iel"][k]), "iaz_B": int(b["iaz"][k]), "Y_A": int(ya[k]), "Y_B": int(yb[k]),
                  "cruza_union": int(union[k])} for k in np.flatnonzero(moved)]
    summary = {
        "detectados_en_ambas": int(both.sum()),
        "cambian_de_pixel_detectados_en_ambas": int(moved.sum()),
        "en_R_int_en_ambas_pero_otro_pixel": int(roi_moved.sum()),
        "discordantes": int(len(idx)),
        "discordantes_por_campo": {f: int(m.sum()) for f, m in fields.items()},
        "discordantes_por_clase": dict(collections.Counter(str(c) for c in classes)),
    }
    return summary, disc_rows, cross_rows, move_rows


DISC_HEADER = ["replica", "id", "clase", "campos", "Y_A", "Y_B", "estado_A", "estado_B", "iel_A", "iaz_A", "iel_B",
               "iaz_B", "cruza_A", "cruza_B"]
MOVE_HEADER = ["replica", "id", "iel_A", "iaz_A", "iel_B", "iaz_B", "Y_A", "Y_B", "cruza_union"]
CROSS_HEADER = (["replica", "id"] + ["%s_%s" % (t, f) for t in ("A", "B") for f in CROSS_FIELDS] +
                ["lado_observado_A", "B_material_A1_1um", "Y_A", "Y_B"])


def analyse_pair(entries, caso, replica, attempt, roi, lists, contracts, control=False):
    ea = select(entries, caso, replica, "A")
    eb = select(entries, caso, replica, "B")
    if len(ea) != 1 or len(eb) != 1:
        contracts.add("K0", "r%d" % replica, "el manifiesto tiene exactamente una corrida A y una B de la réplica",
                      False, True, {"A": len(ea), "B": len(eb)})
        return None
    ra, rb = A.load_run(ea[0], attempt), A.load_run(eb[0], attempt)
    A.k6_checks(ra, roi, contracts, lists)
    A.k6_checks(rb, roi, contracts, lists)
    same_ids = A.k9a_checks(ra, rb, contracts, control=True) if control else A.k9_checks(ra, rb, contracts)
    out = {"replica": replica, "caso": caso, "N": ea[0]["N"], "corridas": [ra.run_id, rb.run_id],
           "lista": os.path.basename(ea[0]["lista"]), "lista_sha256": ea[0]["lista_sha256"],
           "recursos": [resources(ra), resources(rb)], "runs": (ra, rb)}
    complete = (same_ids and ra.mu is not None and rb.mu is not None and
                len(ra.mu["id"]) == ea[0]["N"] and len(rb.mu["id"]) == ea[0]["N"])
    if not complete:
        out["transiciones"] = None
        return out
    pair = A.Pair(caso, replica, ra, rb)
    stats, ya, yb = A.pair_stats(pair, roi)
    cross, union = A.crossing_summary(pair, ya, yb)
    details, disc_rows, cross_rows, move_rows = pair_details(pair, roi, ya, yb, union)
    out.update({"transiciones": stats, "cruces": cross, "detalle": details, "filas_discordantes": disc_rows,
                "filas_cruces": cross_rows, "filas_cambio_pixel": move_rows})
    return out


def public(res):
    return {k: v for k, v in res.items() if k not in ("runs", "filas_discordantes", "filas_cruces", "filas_cambio_pixel")}


def cmd_block(args):
    entries = A.read_manifest(args.manifest)
    roi = Roi(args.mask)
    lists = A.ListCache(args.lists_dir)
    contracts = A.Contracts()
    res = analyse_pair(entries, MAIN_CASE, args.replica, args.attempt, roi, lists, contracts)
    os.makedirs(args.out, exist_ok=True)
    tag = "r%d" % args.replica
    blocked = bool(contracts.failed_blocking()) or res is None or res.get("transiciones") is None
    body = {"formato": "oe1-etapa2-bloque/1", "replica": args.replica, "estado": A.STAMP if blocked else "OK",
            "mascara_sha256": roi.text_sha, "herramienta": tool_record()}
    if res is not None:
        body.update(public(res))
        ra, rb = res["runs"]
        if ra.hist is not None and rb.hist is not None:
            np.savez_compressed(os.path.join(args.out, "hist-%s.npz" % tag), A=ra.hist, B=rb.hist)
        write_table(os.path.join(args.out, "discordantes-%s.csv" % tag), DISC_HEADER, res.get("filas_discordantes", []))
        write_table(os.path.join(args.out, "cruces-%s.csv" % tag), CROSS_HEADER, res.get("filas_cruces", []))
        write_table(os.path.join(args.out, "cambios-pixel-%s.csv" % tag), MOVE_HEADER, res.get("filas_cambio_pixel", []))
    write_json(os.path.join(args.out, "bloque-%s.json" % tag), contracts.to_json(extra=body))
    t = (res or {}).get("transiciones") or {}
    print("R2-BLOQUE %s estado=%s N=%s n00=%s n01=%s n10=%s n11=%s det_A=%s det_B=%s bloqueantes=%d" % (
        tag, body["estado"], t.get("N"), t.get("n00"), t.get("n01"), t.get("n10"), t.get("n11"), t.get("det_A"),
        t.get("det_B"), len(contracts.failed_blocking())))
    return EXIT_BLOCKED if blocked else EXIT_OK


def cmd_controls(args):
    entries = A.read_manifest(args.manifest)
    roi = Roi(args.mask)
    lists = A.ListCache(args.lists_dir)
    contracts = A.Contracts()
    reps = sorted(set(e["replica"] for e in entries if e["caso"] == CONTROL_CASE and e["rol"] == A.ROLE_MAIN))
    pairs = []
    for rep in reps:
        res = analyse_pair(entries, CONTROL_CASE, rep, args.attempt, roi, lists, contracts, control=True)
        if res is None:
            continue
        ra, rb = res["runs"]
        A.k8_checks(ra, rb, contracts)
        pairs.append(res)
    aa = [e for e in entries if e["rol"] == A.ROLE_AA]
    aa_info = None
    if len(aa) != 1:
        contracts.add("K7", "A/A", "el manifiesto tiene exactamente una repetición A/A", False, True, {"filas": len(aa)})
    else:
        base = select(entries, aa[0]["caso"], aa[0]["replica"], "A")
        rep_run = A.load_run(aa[0], args.attempt)
        A.k6_checks(rep_run, roi, contracts, lists)
        if len(base) == 1:
            base_run = [p for p in pairs if p["replica"] == aa[0]["replica"]]
            base_run = base_run[0]["runs"][0] if base_run else A.load_run(base[0], args.attempt)
            same_input = (base[0]["lista_sha256"] == aa[0]["lista_sha256"] and base[0]["s_tr"] == aa[0]["s_tr"] and
                          base[0]["s_src"] == aa[0]["s_src"] and base[0]["variante"] == aa[0]["variante"] == "A")
            contracts.add("K7", "%s vs %s" % (base_run.run_id, rep_run.run_id),
                          "A/A usa la misma lista, réplica, semillas y --variant A", same_input, True, {})
            A.k7_checks(base_run, rep_run, contracts)
            aa_info = {"corridas": [base_run.run_id, rep_run.run_id], "recursos": [resources(rep_run)]}
        else:
            contracts.add("K7", "A/A", "existe la corrida base de la repetición", False, True, {})
    k8 = [c for c in contracts.checks if c["contrato"] == "K8"]
    k8_ok = bool(k8) and all(c["ok"] is True for c in k8)
    gate = not contracts.failed_blocking() and k8_ok and len(pairs) == len(CONTROL_REPLICAS) and \
        all(p.get("transiciones") for p in pairs)
    body = {"formato": "oe1-etapa2-controles/1", "estado": "OK" if gate else A.STAMP,
            "compuerta": {"bloqueantes_sin_fallos": not contracts.failed_blocking(), "K8_todo_identico": k8_ok,
                          "pares_completos": len(pairs)},
            "pares": [public(p) for p in pairs], "repeticion_AA": aa_info, "mascara_sha256": roi.text_sha,
            "herramienta": tool_record()}
    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, "controles.json"), contracts.to_json(extra=body))
    print("R2-CONTROLES estado=%s bloqueantes=%d K8_identico=%s pares=%d" % (
        body["estado"], len(contracts.failed_blocking()), k8_ok, len(pairs)))
    return EXIT_OK if gate else EXIT_BLOCKED


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------
def load_calculator(path):
    spec = importlib.util.spec_from_file_location("planificar_etapa2_oe1", path)
    mod = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = previous
    return mod


SUM_KEYS = ("N", "n00", "n01", "n10", "n11", "m00", "m01", "m10", "m11", "det_A", "det_B", "sumY_A", "sumY_B")


def aggregate(blocks, expected_replicas, controls, calc):
    by_rep = collections.defaultdict(list)
    for b in blocks:
        by_rep[b.get("replica")].append(b)
    problems = []
    missing = [r for r in expected_replicas if r not in by_rep]
    dup = [r for r, v in by_rep.items() if len(v) > 1]
    extra = [r for r in by_rep if r not in expected_replicas]
    if missing:
        problems.append("faltan bloques: %s" % missing)
    if dup:
        problems.append("bloques repetidos: %s" % dup)
    if extra:
        problems.append("bloques ajenos al diseño: %s" % extra)
    valid = []
    for r in expected_replicas:
        for b in by_rep.get(r, [])[:1]:
            t = b.get("transiciones")
            if b.get("estado") != "OK" or not t:
                problems.append("bloque r%d no válido (estado %s)" % (r, b.get("estado")))
            elif t.get("N") != MAIN_N:
                problems.append("bloque r%d con N=%s" % (r, t.get("N")))
            else:
                valid.append(b)
    if controls is None or controls.get("estado") != "OK":
        problems.append("controles no OK (%s)" % (None if controls is None else controls.get("estado")))
    tot = {k: sum(int(b["transiciones"][k]) for b in valid) for k in SUM_KEYS}
    for k in ("detectados_en_ambas", "cambian_de_pixel_detectados_en_ambas", "en_R_int_en_ambas_pero_otro_pixel",
              "discordantes"):
        tot[k] = sum(int(b["detalle"][k]) for b in valid)
    for k in ("cruces_A", "cruces_B", "cruces_ambas", "cruces_union"):
        tot[k] = sum(int(b["cruces"][k]) for b in valid)
    classes = collections.Counter()
    for b in valid:
        classes.update(b["detalle"]["discordantes_por_clase"])
    tot["discordantes_por_clase"] = dict(classes)
    roi_counts = [tot["n00"], tot["n01"], tot["n10"], tot["n11"]]
    det_counts = [tot["m00"], tot["m01"], tot["m10"], tot["m11"]]
    res = {"bloques_validos": len(valid), "bloques_esperados": len(expected_replicas), "problemas": problems,
           "totales": tot}
    if tot["N"] > 0:
        res["intervalo_R_int"] = calc.interval(roi_counts)
        res["intervalo_deteccion_total_exploratorio"] = calc.interval(det_counts)
        deltas = [(b["transiciones"]["n01"] - b["transiciones"]["n10"]) / b["transiciones"]["N"] * 1e6 for b in valid]
        res["entre_bloques"] = {
            "delta_por_millon_media": float(np.mean(deltas)),
            "delta_por_millon_desviacion": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
            "delta_por_millon_min": float(min(deltas)), "delta_por_millon_max": float(max(deltas)),
            "error_estandar_de_la_media_por_millon": (float(np.std(deltas, ddof=1) / math.sqrt(len(deltas)))
                                                      if len(deltas) > 1 else None),
            "nota": "descriptivo; el protocolo (§3) no usa la desviación entre bloques como intervalo de confianza"}
    interpretable = not problems and len(valid) == len(expected_replicas) and tot["N"] == N_PAIRS_MAIN
    res["interpretable"] = interpretable
    if interpretable:
        res["categoria_de_cierre"] = res["intervalo_R_int"]["decision_numerica"]
    else:
        res["categoria_de_cierre"] = "no_interpretable"
    return res, valid


def block_row(b):
    t, d, c = b["transiciones"], b["detalle"], b["cruces"]
    rec = b.get("recursos", [{}, {}])
    return {"replica": b["replica"], "N": t["N"], "n00": t["n00"], "n01": t["n01"], "n10": t["n10"], "n11": t["n11"],
            "delta_por_millon": repr((t["n01"] - t["n10"]) / t["N"] * 1e6), "sumY_A": t["sumY_A"],
            "sumY_B": t["sumY_B"], "det_A": t["det_A"], "det_B": t["det_B"], "m01": t["m01"], "m10": t["m10"],
            "cambian_pixel": d["cambian_de_pixel_detectados_en_ambas"],
            "R_int_ambas_otro_pixel": d["en_R_int_en_ambas_pero_otro_pixel"], "discordantes": d["discordantes"],
            "cruces_A": c["cruces_A"], "cruces_B": c["cruces_B"], "cruces_union": c["cruces_union"],
            "pared_A_s": num(rec[0].get("pared_s")), "pared_B_s": num(rec[1].get("pared_s")),
            "cpu_A_s": num(rec[0].get("cpu_usuario_s")), "cpu_B_s": num(rec[1].get("cpu_usuario_s")),
            "rss_A_kb": rec[0].get("rss_max_kb"), "rss_B_kb": rec[1].get("rss_max_kb"),
            "nodo": rec[0].get("nodo"), "trabajo": rec[0].get("trabajo"), "estado": b["estado"]}


BLOCK_HEADER = ["replica", "N", "n00", "n01", "n10", "n11", "delta_por_millon", "sumY_A", "sumY_B", "det_A", "det_B",
                "m01", "m10", "cambian_pixel", "R_int_ambas_otro_pixel", "discordantes", "cruces_A", "cruces_B",
                "cruces_union", "pared_A_s", "pared_B_s", "cpu_A_s", "cpu_B_s", "rss_A_kb", "rss_B_kb", "nodo",
                "trabajo", "estado"]


def make_maps(valid, blocks_dir, out_dir, rint, figures):
    ha = np.zeros((C.N_EL, C.N_AZ), dtype=np.int64)
    hb = np.zeros_like(ha)
    used = 0
    for b in valid:
        p = os.path.join(blocks_dir, "hist-r%d.npz" % b["replica"])
        with np.load(p) as z:
            ha += z["A"]
            hb += z["B"]
        used += 1
    n = used * MAIN_N
    np.savez_compressed(os.path.join(out_dir, "mapas-etapa2.npz"), A=ha, B=hb, N=n)
    info = {"bloques": used, "N": n, "suma_A": int(ha.sum()), "suma_B": int(hb.sum()),
            "R_int_A": int(ha[rint].sum()), "R_int_B": int(hb[rint].sum()),
            "celdas_distintas": int((ha != hb).sum()), "max_abs_diferencia_celda": int(np.abs(hb - ha).max())}
    if figures and n:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ext = [-C.AZ_HALF_DEG, C.AZ_HALF_DEG, C.EL_MIN_DEG, C.EL_MAX_DEG]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
        for ax, data, title, cmap in ((axes[0], ha / n, "A / N", "viridis"), (axes[1], hb / n, "B / N", "viridis"),
                                      (axes[2], (hb - ha) / n, "(B − A) / N", "coolwarm")):
            if cmap == "coolwarm":
                lim = max(np.abs(data).max(), 1e-12)
                im = ax.imshow(data, origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim)
            else:
                im = ax.imshow(data, origin="lower", extent=ext, aspect="auto", cmap=cmap)
            ax.contour(rint.astype(float), levels=[0.5], colors="k", linewidths=0.8, origin="lower", extent=ext)
            ax.set_title(title)
            ax.set_xlabel("acimut relativo (°)")
            ax.set_ylabel("elevación (°)")
            fig.colorbar(im, ax=ax, label="cuentas por primario")
        fig.suptitle("geo-train-06, vista 0, N = %d pares; contorno: R_int. (B − A)/N es una diferencia neta y no "
                     "identifica muones individuales" % n, fontsize=9)
        fig.savefig(os.path.join(out_dir, "mapas-etapa2.png"), dpi=130)
        plt.close(fig)
    return info


def cmd_aggregate(args):
    blocks = []
    for p in sorted(glob.glob(os.path.join(args.blocks_dir, "bloque-r*.json"))):
        with open(p, "rb") as fh:
            blocks.append(json.loads(fh.read().decode("utf-8")))
    controls = None
    if args.controls and os.path.isfile(args.controls):
        with open(args.controls, "rb") as fh:
            controls = json.loads(fh.read().decode("utf-8"))
    rows = read_corridas(args.corridas)
    probs, _ = check_corridas(rows)
    expected = sorted(set(r["replica"] for r in rows if r["geometry_id"] == MAIN_CASE))
    calc = load_calculator(args.calculator)
    res, valid = aggregate(blocks, expected, controls, calc)
    res["problemas"] = probs + res["problemas"]
    if probs:
        res["interpretable"] = False
        res["categoria_de_cierre"] = "no_interpretable"
    os.makedirs(args.out, exist_ok=False)
    roi = Roi(args.mask)
    res["mapas"] = make_maps(valid, args.blocks_dir, args.out, roi.rint, args.figures)
    res["margen_por_millon"] = calc.MARGIN * 1e6
    res["calculador"] = {"ruta": os.path.abspath(args.calculator), "sha256": C.sha256_file(args.calculator)}
    res["herramienta"] = tool_record()
    write_json(os.path.join(args.out, "resultados-etapa2.json"), res)
    write_table(os.path.join(args.out, "transiciones-por-bloque.csv"), BLOCK_HEADER, [block_row(b) for b in valid])
    for kind, header in (("discordantes", DISC_HEADER), ("cruces", CROSS_HEADER), ("cambios-pixel", MOVE_HEADER)):
        merged = []
        for b in valid:
            p = os.path.join(args.blocks_dir, "%s-r%d.csv" % (kind, b["replica"]))
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8", newline="") as fh:
                    merged.extend(csv.DictReader(fh))
        write_table(os.path.join(args.out, "%s-agregado.csv" % kind), header, merged)
    iv = res.get("intervalo_R_int", {})
    print("R2-AGREGADO bloques=%d/%d N=%s n01=%s n10=%s intervalo_por_millon=%s categoria=%s problemas=%d" % (
        res["bloques_validos"], res["bloques_esperados"], res["totales"]["N"], res["totales"]["n01"],
        res["totales"]["n10"], iv.get("interval95_conservative_per_million"), res["categoria_de_cierre"],
        len(res["problemas"])))
    return EXIT_OK if res["interpretable"] else EXIT_BLOCKED


# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("plan")
    pp.add_argument("--corridas", required=True)
    pp.add_argument("--replicas", help="comma-separated replicas of one job (default: all)")
    pp.add_argument("--lists-tsv")
    pp.add_argument("--manifest-tsv")
    pp.add_argument("--print-commands", action="store_true")
    for k in ("--list-sha", "--lists-dir", "--catalog-dir", "--bin-a", "--bin-b", "--mask", "--mask-sha", "--tables"):
        pp.add_argument(k)
    for name in ("block", "controls"):
        q = sub.add_parser(name)
        q.add_argument("--manifest", required=True)
        q.add_argument("--attempt", required=True)
        q.add_argument("--lists-dir", required=True)
        q.add_argument("--mask", required=True)
        q.add_argument("--out", required=True)
        if name == "block":
            q.add_argument("--replica", type=int, required=True)
    pa = sub.add_parser("aggregate")
    pa.add_argument("--blocks-dir", required=True)
    pa.add_argument("--controls")
    pa.add_argument("--corridas", required=True)
    pa.add_argument("--calculator", required=True)
    pa.add_argument("--mask", required=True)
    pa.add_argument("--out", required=True)
    pa.add_argument("--figures", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return {"plan": cmd_plan, "block": cmd_block, "controls": cmd_controls, "aggregate": cmd_aggregate}[args.cmd](args)
    except (OSError, ValueError, KeyError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
