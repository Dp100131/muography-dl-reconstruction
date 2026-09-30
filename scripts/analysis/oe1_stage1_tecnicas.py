#!/usr/bin/env python3
"""OE1 stage 1: evaluation of the technical tests (DV1), run outside the 13 runs.

  t-l  Legacy inertness: pilot binary versus the instrumented one in legacy
       mode. hist.csv identical byte for byte, meta identical apart from timing.
  t-r  PROT:17: list mode with ``--muon-log full`` versus ``min``: identical
       histogram and identical rng1 for every id (logging consumes no random
       numbers).
  t-m  Independence from the master engine: same transport seed, different
       master ``--seed``: rng0, rng1 and histogram identical (the whole
       per-muon log is compared).
  t-p  Independence from the previous muons: the first K rows of the list were
       replaced; ids >= K must keep rng1 and the whole outcome.
  t-f  Functional end-to-end rehearsal: evaluates the analysis output
       (contratos.json, tabla-corridas.csv) obtained on the technical runs.
  resumen  Merges the JSON results of the tests above.

Every subcommand prints a verdict, writes ``--out`` (new file) and exits 0 when
the test passes, 3 otherwise. Python >= 3.9, numpy >= 1.23.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

import oe1_stage1_common as C

IDENTITY_COLUMNS = ("caso", "geometry_id", "view", "variant", "replica", "list_sha256")
COMMON_COLUMNS = ("id", "prim_fnv", "rng0", "rng1", "detected", "iel", "iaz", "in_rint", "ekin_plane_gev",
                  "el_plane_deg", "relaz_plane_deg", "status", "end_process", "n_steps")
Z_COLUMNS = ("crossed", "lz_m", "lz_true_m", "lz_rock_m", "lz_a0_m", "lz_a1_m", "lz_other_m", "lc_m", "le_m",
             "rho_lz_gcm2", "n_steps_z", "n_steps_z_both_out", "n_entries_z", "n_graze_z", "mat_in", "mat_out",
             "e_in_gev", "e_in_lo_gev", "e_in_hi_gev", "e_out_gev", "e_out_lo_gev", "e_out_hi_gev")


class Result:
    def __init__(self, name):
        self.name = name
        self.checks = []
        self.inputs = {}

    def add(self, desc, ok, evidence=None):
        self.checks.append({"descripcion": desc, "ok": bool(ok) if ok is not None else None,
                            "evidencia": evidence if evidence is not None else {}})
        return ok

    @property
    def ok(self):
        return all(c["ok"] is not False for c in self.checks) and bool(self.checks)

    def to_json(self):
        return {"formato": "oe1-prueba-tecnica/1", "prueba": self.name, "ok": self.ok,
                "etiqueta": "prueba técnica (DV1), fuera de las 13 corridas",
                "comprobaciones": self.checks, "entradas": self.inputs}


def one(dirpath, pattern):
    found = sorted(glob.glob(os.path.join(dirpath, pattern)))
    return found[0] if len(found) == 1 else None


def load_dir(d, need_muones=True):
    out = {"dir": d}
    out["hist"] = one(d, "*_hist.csv")
    out["meta_path"] = one(d, "*.meta.json")
    out["muones_path"] = one(d, "*_muones.csv")
    out["log_path"] = os.path.join(d, "run.log") if os.path.isfile(os.path.join(d, "run.log")) else None
    out["hist_sha"] = C.sha256_file(out["hist"]) if out["hist"] else None
    out["meta"] = C.read_json(out["meta_path"]) if out["meta_path"] else None
    out["muones_sha"] = C.sha256_file(out["muones_path"]) if out["muones_path"] else None
    out["mu"] = None
    if out["muones_path"]:
        with open(out["muones_path"], "rb") as fh:
            out["mu"] = C.parse_muones_bytes(fh.read())
    out["log"] = ""
    if out["log_path"]:
        with open(out["log_path"], "r", encoding="utf-8", errors="replace") as fh:
            out["log"] = fh.read()
    if need_muones and out["mu"] is None:
        out["problem"] = "no hay _muones.csv en " + d
    return out


def cols_equal(a, b, names):
    """Names of the columns that differ; a column present in only one of the logs differs."""
    bad = []
    for c in names:
        if (c in a) != (c in b):
            bad.append(c)
            continue
        if c not in a:
            continue
        x, y = a[c], b[c]
        if x.shape != y.shape:
            bad.append(c)
        elif x.dtype.kind == "f":
            if not np.array_equal(x, y, equal_nan=True):
                bad.append(c)
        elif not np.array_equal(x, y):
            bad.append(c)
    return bad


def record_inputs(res, **dirs):
    for k, d in dirs.items():
        res.inputs[k] = {"dir": os.path.abspath(d["dir"]), "hist_sha256": d["hist_sha"],
                         "muones_sha256": d["muones_sha"]}


def meta_paths_diff(a, b, allowed=()):
    """Dotted paths that differ, minus those in ``allowed``."""
    out = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                p = path + "." + k if path else k
                if k not in x or k not in y:
                    if p not in allowed:
                        out.append(p)
                else:
                    walk(x[k], y[k], p)
        elif x != y and path not in allowed:
            out.append(path)

    walk(a, b, "")
    return out


def mask_wall(text):
    import re
    return re.sub(r"wall_s=\S+", "wall_s=<t>", text)


# --------------------------------------------------------------------------
def test_legacy(pilot_dir, new_dir, label=None):
    res = Result("T-L" if not label else "T-L (%s)" % label)
    p, n = load_dir(pilot_dir, False), load_dir(new_dir, False)
    record_inputs(res, piloto=p, nuevo=n)
    res.add("hist.csv del binario del piloto y del instrumentado (modo legado) idénticos byte a byte",
            p["hist_sha"] is not None and p["hist_sha"] == n["hist_sha"],
            {"piloto": p["hist_sha"], "nuevo": n["hist_sha"]})
    if p["meta"] is not None and n["meta"] is not None:
        d = meta_paths_diff(C.meta_without_timing(p["meta"]), C.meta_without_timing(n["meta"]))
        res.add("meta.json igual salvo los tiempos y el uso de recursos (%s, ...)" % " y ".join(C.TIMING_META_KEYS),
                not d, {"campos_distintos": d,
                        "claves_de_tiempo_ignoradas": sorted(set(C.timing_meta_paths(p["meta"])) |
                                                             set(C.timing_meta_paths(n["meta"])))})
        res.add("el modo legado no escribe la clave oe1_etapa1", "oe1_etapa1" not in n["meta"], {})
    else:
        res.add("meta.json presente en ambos", False, {})
    res.add("run.log igual una vez enmascarado wall_s= (informativo)",
            mask_wall(p["log"]) == mask_wall(n["log"]), {})
    res.add("el modo legado no escribe _muones.csv", n["muones_path"] is None, {})
    return res


def test_full_vs_min(full_dir, min_dir):
    res = Result("T-R")
    f, m = load_dir(full_dir), load_dir(min_dir)
    record_inputs(res, full=f, min=m)
    if f["mu"] is None or m["mu"] is None:
        res.add("los dos registros existen", False, {})
        return res
    res.add("hist.csv idéntico entre --muon-log full y min", f["hist_sha"] == m["hist_sha"],
            {"full": f["hist_sha"], "min": m["hist_sha"]})
    same_ids = bool(np.array_equal(f["mu"]["id"], m["mu"]["id"]))
    res.add("mismos IDs", same_ids, {})
    if same_ids:
        res.add("rng1 igual para todo id (registrar no consume números aleatorios; PROT:17)",
                bool(np.array_equal(f["mu"]["rng1"], m["mu"]["rng1"])),
                {"distintos": int((f["mu"]["rng1"] != m["mu"]["rng1"]).sum())})
        bad = cols_equal(f["mu"], m["mu"], COMMON_COLUMNS)
        res.add("las columnas de identidad, estado y bin son iguales", not bad, {"columnas_distintas": bad})
    empty = all(bool(np.isnan(m["mu"][c]).all()) for c in ("lz_m", "crossed"))
    res.add("el nivel min deja vacíos los campos de cruce (informativo)", empty, {})
    lvl = (m["meta"] or {}).get("oe1_etapa1", {}).get("muon_log", {}).get("level")
    res.add("el meta de la corrida min declara level = min", lvl == "min", {"level": lvl})
    return res


def test_master_seed(ref_dir, shifted_dir):
    res = Result("T-M")
    r, s = load_dir(ref_dir), load_dir(shifted_dir)
    record_inputs(res, referencia=r, maestra_desplazada=s)
    if r["mu"] is None or s["mu"] is None:
        res.add("los dos registros existen", False, {})
        return res
    res.add("hist.csv idéntico con otra semilla maestra", r["hist_sha"] == s["hist_sha"],
            {"ref": r["hist_sha"], "desplazada": s["hist_sha"]})
    res.add("rng0 igual para todo id", bool(np.array_equal(r["mu"]["rng0"], s["mu"]["rng0"])), {})
    res.add("rng1 igual para todo id", bool(np.array_equal(r["mu"]["rng1"], s["mu"]["rng1"])), {})
    res.add("_muones.csv idéntico byte a byte", r["muones_sha"] == s["muones_sha"],
            {"columnas_distintas": cols_equal(r["mu"], s["mu"], sorted(set(C.muon_columns_present(r["mu"])) |
                                                                       set(C.muon_columns_present(s["mu"]))))})
    allowed = ("seed", "oe1_etapa1.seeds.master_cli", "oe1_etapa1.seeds.master_applied")
    d = meta_paths_diff(C.meta_without_timing(r["meta"]), C.meta_without_timing(s["meta"]), allowed)
    res.add("meta.json igual salvo tiempos y la semilla maestra", not d, {"campos_distintos": d})
    res.add("las semillas maestras de las dos corridas son distintas",
            r["meta"]["seed"] != s["meta"]["seed"], {"ref": r["meta"]["seed"], "desplazada": s["meta"]["seed"]})
    return res


def test_prefix_replacement(base_dir, derived_dir, k):
    res = Result("T-P")
    b, d = load_dir(base_dir), load_dir(derived_dir)
    record_inputs(res, base=b, derivada=d)
    if b["mu"] is None or d["mu"] is None:
        res.add("los dos registros existen", False, {})
        return res
    same_n = len(b["mu"]["id"]) == len(d["mu"]["id"])
    res.add("misma longitud", same_n, {})
    if not same_n:
        return res
    tail = b["mu"]["id"] >= k
    cols = [c for c in C.muon_columns_present(b["mu"]) if c not in IDENTITY_COLUMNS]
    bad = []
    for c in cols:
        if c not in d["mu"]:
            bad.append(c)
            continue
        x, y = b["mu"][c][tail], d["mu"][c][tail]
        equal = np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f" else np.array_equal(x, y)
        if not equal:
            bad.append(c)
    res.add("los ids >= %d conservan rng0, rng1 y todo el resultado" % k, not bad, {"columnas_distintas": bad})
    res.add("rng1 igual en los ids >= %d" % k, bool(np.array_equal(b["mu"]["rng1"][tail], d["mu"]["rng1"][tail])), {})
    head_diff = float((b["mu"]["prim_fnv"][~tail] != d["mu"]["prim_fnv"][~tail]).mean())
    res.add("la sustitución de las primeras %d filas se aplicó (>90 %% cambia el primario)" % k,
            head_diff > 0.9, {"fraccion_que_cambia": head_diff})
    return res


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = [ln for ln in fh if not ln.startswith("#")]
    return list(csv.DictReader(rows))


def test_functional(analysis_dir, expect_zero=("STOPPED", "OTHER", "ABORTED"), expected_runs=5):
    res = Result("T-F")
    ctr_path = os.path.join(analysis_dir, "contratos.json")
    tab_path = os.path.join(analysis_dir, "tabla-corridas.csv")
    if not os.path.isfile(ctr_path):
        res.add("el análisis de extremo a extremo produjo contratos.json", False, {})
        return res
    ctr = C.read_json(ctr_path)
    res.inputs["contratos_sha256"] = C.sha256_file(ctr_path)
    res.add("el análisis terminó sin errores ni contratos bloqueantes fallidos",
            ctr.get("estado") == "OK" and not ctr.get("bloqueantes_fallidos"),
            {"estado": ctr.get("estado"), "bloqueantes_fallidos": ctr.get("bloqueantes_fallidos")})
    res.add("no hay contratos no bloqueantes fallidos (K8, K9 b, avisos)", not ctr.get("no_bloqueantes_fallidos"),
            {"no_bloqueantes_fallidos": ctr.get("no_bloqueantes_fallidos")})
    if not os.path.isfile(tab_path):
        res.add("tabla-corridas.csv presente", False, {})
        return res
    rows = read_csv_rows(tab_path)
    res.add("la tabla de corridas tiene %d filas" % expected_runs, len(rows) == expected_runs,
            {"filas": len(rows)})
    for name in expect_zero:
        bad = {r["corrida"]: r[name] for r in rows if int(r[name]) != 0}
        res.add("estado %s = 0 en todas las corridas" % name, not bad, {"distintas_de_cero": bad})
    per_event = [r["pared_s"] for r in rows if r.get("pared_s") not in ("", "NA")]
    res.inputs["pared_s"] = per_event
    return res


def summarize(paths):
    items = [C.read_json(p) for p in paths]
    return {"formato": "oe1-pruebas-tecnicas/1", "ok": all(i["ok"] for i in items),
            "pruebas": [{"prueba": i["prueba"], "ok": i["ok"],
                         "fallos": [c["descripcion"] for c in i["comprobaciones"] if c["ok"] is False]}
                        for i in items]}


# --------------------------------------------------------------------------
def emit(res, out):
    if out:
        C.write_text_new(out, C.dumps_json(res.to_json()))
    print("%s: %s" % (res.name, "OK" if res.ok else "FALLA"))
    for c in res.checks:
        if c["ok"] is False:
            print("  - " + c["descripcion"])
    return 0 if res.ok else 3


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("t-l")
    a.add_argument("--pilot", required=True)
    a.add_argument("--new", required=True)
    a.add_argument("--label", help="A o B: etiqueta de la variante")
    a.add_argument("--out")
    a = sub.add_parser("t-r")
    a.add_argument("--full", required=True)
    a.add_argument("--min", required=True)
    a.add_argument("--out")
    a = sub.add_parser("t-m")
    a.add_argument("--ref", required=True)
    a.add_argument("--shifted", required=True)
    a.add_argument("--out")
    a = sub.add_parser("t-p")
    a.add_argument("--base", required=True)
    a.add_argument("--derived", required=True)
    a.add_argument("--k", type=int, required=True)
    a.add_argument("--out")
    a = sub.add_parser("t-f")
    a.add_argument("--analysis-dir", required=True)
    a.add_argument("--expected-runs", type=int, default=5)
    a.add_argument("--out")
    a = sub.add_parser("resumen")
    a.add_argument("results", nargs="+")
    a.add_argument("--out")
    return ap


def main(argv=None):
    C.utf8_streams()
    args = build_parser().parse_args(argv)
    return C.run_command(run, args)


def run(args):
    if args.cmd == "t-l":
        return emit(test_legacy(args.pilot, args.new, args.label), args.out)
    if args.cmd == "t-r":
        return emit(test_full_vs_min(args.full, args.min), args.out)
    if args.cmd == "t-m":
        return emit(test_master_seed(args.ref, args.shifted), args.out)
    if args.cmd == "t-p":
        return emit(test_prefix_replacement(args.base, args.derived, args.k), args.out)
    if args.cmd == "t-f":
        return emit(test_functional(args.analysis_dir, expected_runs=args.expected_runs), args.out)
    summary = summarize(args.results)
    if args.out:
        C.write_text_new(args.out, C.dumps_json(summary))
    for p in summary["pruebas"]:
        print("%-5s %s%s" % (p["prueba"], "OK" if p["ok"] else "FALLA",
                             "" if p["ok"] else "  <- " + "; ".join(p["fallos"])))
    print("PRUEBAS TÉCNICAS:", "OK" if summary["ok"] else "FALLA")
    return 0 if summary["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
