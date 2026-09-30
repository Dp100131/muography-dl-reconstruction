#!/usr/bin/env python3
"""OE1 stage 1: validator of the frozen primary lists (step 1, control K1).

Three layers, all in one report (``validacion-listas.json``):

  1. Structure: the same defects the C++ ``muography_sim --check-primaries``
     reports (header, LF only, contiguous unique ids, unit directions within
     1e-12, dz < 0, ekin > 0, PDG +-13 consistent with the charge, duplicate
     initial states) and the canonical FNV-1a-64 recomputed bit-exactly as the
     binary does, compared with the sidecar. ``--cpp-check`` additionally
     compares every counter with the line the binary printed.
  2. Geometry: back-projecting each track hits the 1.2 m panel of the view's
     detector within 1e-9 m and the horizontal start distance equals
     1000 + R + 300 (R = base_radius_m of the catalog).
  3. Source model (pilot generator): chi-square of the elevation rows against
     ``row_weights_30d.txt``, KS of the momentum per band against ``p_cdf.txt``,
     uniformity of the relative azimuth, of the elevation jitter and of the
     impact point, and the mu+ fraction against 1.2794/2.2794. Any p < 1e-4 is
     flagged (investigate before transporting). The p-values use the
     implementations of oe1_stage1_common (no SciPy needed); SciPy, when
     installed, is only used to report a cross-check value.

Subcommands: ``validate`` (default use) and ``derive`` (technical test T-P:
replace the first K rows of a list by rows of another list).

Exit codes: 0 all clear; 3 structural / geometry / sidecar failure; 4 only
statistical flags; 2 usage error.
"""
import argparse
import math
import os
import re
import sys

import numpy as np

import oe1_stage1_common as C

CASE_CODES = {"geo-train-06": 6, "geo-train-01": 1}
NAME_RE = re.compile(r"^(?P<caso>.+)_v(?P<vista>\d+)_r(?P<rep>\d+)_s(?P<semilla>\d+)"
                     r"\.primarios\.csv$")
DEFAULT_ALPHA = 1e-4
DEFAULT_GEOMETRY_TOL_M = 1e-9


# --------------------------------------------------------------------------
# geometry and source model
# --------------------------------------------------------------------------
def list_kinematics(table, view, r_start):
    """Derived quantities of the list, all vectorised.

    Returns a dict with elevation/relative azimuth of the arrival direction
    (degrees), momentum (GeV/c), and the back-projected impact point as the
    generator aimed it (``start + dir * R_start / hypot(dx, dy)``) and as the
    intersection of the straight line with the detector plane.
    """
    v = table["vals"]
    start = v[:, 0:3]
    dvec = v[:, 3:6]
    ekin = v[:, 6]
    hor = np.hypot(dvec[:, 0], dvec[:, 1])
    det = np.array(C.detector_center(view))
    view_az = C.look_azimuth_rad(view) - math.pi
    normal = np.array([math.cos(view_az), math.sin(view_az), 0.0])
    panel_y = np.array([-math.sin(view_az), math.cos(view_az), 0.0])
    # (1) as the generator aimed it
    imp_gen = start + dvec * (r_start / hor)[:, None]
    rel_gen = imp_gen - det
    plane_resid = rel_gen @ normal
    pa = rel_gen @ panel_y
    pb = rel_gen[:, 2]
    # (2) straight line to the detector plane
    denom = dvec @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        t_plane = ((det - start) @ normal) / denom
    imp_plane = start + dvec * t_plane[:, None]
    hdist = np.hypot(imp_plane[:, 0] - start[:, 0], imp_plane[:, 1] - start[:, 1])
    look = C.look_azimuth_rad(view)
    elev = np.degrees(np.arcsin(np.clip(-dvec[:, 2], -1.0, 1.0)))
    rel = np.arctan2(-dvec[:, 1], -dvec[:, 0]) - look
    rel = (rel + math.pi) % (2.0 * math.pi) - math.pi
    rel_deg = np.degrees(rel)
    p_tot = np.sqrt(ekin * ekin + 2.0 * ekin * C.MUON_MASS_GEV)
    return {"elev_deg": elev, "relaz_deg": rel_deg, "p_gev": p_tot,
            "plane_resid": plane_resid, "panel_a": pa, "panel_b": pb,
            "hdist": hdist, "denom": denom}


def geometry_checks(table, view, r_start, tol):
    k = list_kinematics(table, view, r_start)
    out = {}
    finite = np.isfinite(k["plane_resid"]) & np.isfinite(k["hdist"])
    out["filas_no_finitas"] = int((~finite).sum())
    half = C.DET_SIDE_M / 2.0
    excess_a = np.maximum(np.abs(k["panel_a"]) - half, 0.0)
    excess_b = np.maximum(np.abs(k["panel_b"]) - half, 0.0)
    out["desviacion_plano_max_m"] = float(np.abs(k["plane_resid"][finite]).max()) if finite.any() else None
    out["exceso_sobre_panel_max_m"] = float(max(excess_a[finite].max(), excess_b[finite].max())) if finite.any() else None
    out["radio_de_partida_esperado_m"] = r_start
    out["desviacion_radio_de_partida_max_m"] = (
        float(np.abs(k["hdist"][finite] - r_start).max()) if finite.any() else None)
    out["impacto_y_rango_m"] = [float(k["panel_a"][finite].min()), float(k["panel_a"][finite].max())] if finite.any() else None
    out["impacto_z_rango_m"] = [float(k["panel_b"][finite].min() + C.DET_HEIGHT_M),
                                float(k["panel_b"][finite].max() + C.DET_HEIGHT_M)] if finite.any() else None
    out["elevacion_rango_grados"] = [float(k["elev_deg"].min()), float(k["elev_deg"].max())] if len(k["elev_deg"]) else None
    out["acimut_relativo_rango_grados"] = [float(k["relaz_deg"].min()), float(k["relaz_deg"].max())] if len(k["relaz_deg"]) else None
    out["tolerancia_m"] = tol
    ok = (out["filas_no_finitas"] == 0 and out["desviacion_plano_max_m"] is not None and
          out["desviacion_plano_max_m"] <= tol and out["exceso_sobre_panel_max_m"] <= tol and
          out["desviacion_radio_de_partida_max_m"] <= tol)
    out["ok"] = bool(ok)
    return out, k


def chi2_gof(obs, exp, min_expected=5.0):
    """Chi-square goodness of fit merging neighbouring cells until E >= 5."""
    mo, me = [], []
    ao = ae = 0.0
    for o, e in zip(obs, exp):
        ao += o
        ae += e
        if ae >= min_expected:
            mo.append(ao)
            me.append(ae)
            ao = ae = 0.0
    if ae > 0.0:
        if me:
            mo[-1] += ao
            me[-1] += ae
        else:
            mo.append(ao)
            me.append(ae)
    mo = np.array(mo)
    me = np.array(me)
    stat = float(((mo - me) ** 2 / me).sum())
    df = len(mo) - 1
    return stat, df, (C.chi2_sf(stat, df) if df > 0 else 1.0)


def _scipy():
    try:
        import scipy.stats as st
        return st
    except Exception:
        return None


def _add_test(tests, name, n, stat, p, alpha, p_scipy=None, extra=None):
    t = {"prueba": name, "n": int(n), "estadistico": None if stat is None else float(stat),
         "p": float(p), "marcada": bool(p < alpha), "p_scipy": p_scipy}
    if extra:
        t.update(extra)
    tests.append(t)


def statistical_checks(table, kin, tables, alpha):
    """The pilot source-model tests; returns the list of test records."""
    st = _scipy()
    tests = []
    n = len(table["id"])
    elev = kin["elev_deg"]
    centres = tables["centres"]
    half = tables["half_step_deg"]
    # 1. elevation rows against the row weights
    mids = 0.5 * (centres[1:] + centres[:-1])
    row = np.searchsorted(mids, elev)
    obs = np.bincount(row, minlength=len(centres)).astype(np.float64)
    exp = n * tables["weights"] / tables["weights"].sum()
    stat, df, p = chi2_gof(obs, exp)
    p_sp = None
    if st is not None:
        # same merging as above: SciPy only evaluates the tail probability
        p_sp = float(st.chi2.sf(stat, df)) if df > 0 else 1.0
    _add_test(tests, "filas de elevacion frente a row_weights (chi-cuadrado)", n, stat, p, alpha,
              p_sp, {"gl": df})
    # 2. jitter inside the row
    resid = elev - centres[row]
    inside = float(np.abs(resid).max())
    u = (resid / half + 1.0) * 0.5
    d, p = C.ks_uniform_test(np.clip(u, 0.0, 1.0))
    _add_test(tests, "holgura de elevacion dentro de la fila (KS uniforme)", n, d, p, alpha,
              float(st.kstest(np.clip(u, 0, 1), "uniform").pvalue) if st else None,
              {"desviacion_maxima_grados": inside, "semipaso_grados": half})
    # 3. relative azimuth
    u = (kin["relaz_deg"] + C.AZ_HALF_DEG) / (2.0 * C.AZ_HALF_DEG)
    d, p = C.ks_uniform_test(np.clip(u, 0.0, 1.0))
    _add_test(tests, "acimut relativo (KS uniforme en +-30 grados)", n, d, p, alpha,
              float(st.kstest(np.clip(u, 0, 1), "uniform").pvalue) if st else None)
    # 4. impact point on the panel
    half_p = C.DET_SIDE_M / 2.0
    for name, vals in (("y", kin["panel_a"]), ("z", kin["panel_b"])):
        u = (vals + half_p) / C.DET_SIDE_M
        d, p = C.ks_uniform_test(np.clip(u, 0.0, 1.0))
        _add_test(tests, "punto de impacto %s en el panel (KS uniforme)" % name, n, d, p, alpha,
                  float(st.kstest(np.clip(u, 0, 1), "uniform").pvalue) if st else None)
    # 5. charge
    k_plus = int((table["pdg"] == -13).sum())
    p_bin = C.binom_two_sided_p(k_plus, n, tables["prob_mu_plus"])
    _add_test(tests, "fraccion de mu+ frente a %.4f (binomial exacta)" % tables["prob_mu_plus"], n,
              k_plus / float(n) if n else None, p_bin, alpha,
              float(st.binomtest(k_plus, n, tables["prob_mu_plus"]).pvalue) if st else None,
              {"mu_plus": k_plus, "fraccion": k_plus / float(n) if n else None})
    # 6. momentum per band
    edges = tables["edges"]
    band = np.clip(np.searchsorted(edges, elev, side="right") - 1, 0, len(tables["bands"]) - 1)
    for b, q in enumerate(tables["bands"]):
        sel = band == b
        nb = int(sel.sum())
        if nb < 20:
            tests.append({"prueba": "momento banda %d (KS frente a p_cdf)" % b, "n": nb,
                          "estadistico": None, "p": 1.0, "marcada": False, "p_scipy": None,
                          "nota": "menos de 20 muones: no se evalua"})
            continue
        grid = np.arange(len(q)) / float(len(q) - 1)
        uq = np.interp(kin["p_gev"][sel], q, grid, left=0.0, right=1.0)
        d, p = C.ks_uniform_test(uq)
        _add_test(tests, "momento banda %d (KS frente a p_cdf)" % b, nb, d, p, alpha,
                  float(st.kstest(uq, "uniform").pvalue) if st else None)
    return tests


# --------------------------------------------------------------------------
# validation of one list
# --------------------------------------------------------------------------
def _cpp_lines(paths):
    lines = {}
    for p in paths or []:
        with open(p, "r") as fh:
            for line in fh:
                if line.startswith("OE1-CHECK-PRIMARIES"):
                    kv = C.parse_kv_line(line)
                    if "file" in kv:
                        lines[kv["file"]] = kv
    return lines


def compare_with_cpp(kv, defects, n, canon, file_sha):
    """Every counter of the binary's check line against the Python one."""
    rows = []

    def eq(name, a, b):
        rows.append({"campo": name, "python": a, "binario": b, "igual": str(a) == str(b)})

    eq("n", n, kv.get("n"))
    for key in ("lf_only", "header_ok", "ends_with_newline", "bad_rows", "id_min", "id_max",
                "ids_contiguous", "dup_ids", "dup_states", "nonfinite", "norm_fail", "dz_nonneg",
                "ekin_nonpos", "pdg_bad", "charge_mismatch", "n_mu_plus", "n_mu_minus"):
        eq(key, defects[key], kv.get(key))
    eq("canon_fnv1a64", C.hex16(canon), kv.get("canon_fnv1a64"))
    eq("file_sha256", file_sha, kv.get("file_sha256"))
    try:
        dev_b = float(kv.get("max_norm_dev", "nan"))
        rows.append({"campo": "max_norm_dev", "python": defects["max_norm_dev"], "binario": dev_b,
                     "igual": abs(dev_b - defects["max_norm_dev"]) <= 1e-15})
    except ValueError:
        rows.append({"campo": "max_norm_dev", "python": defects["max_norm_dev"],
                     "binario": kv.get("max_norm_dev"), "igual": False})
    return rows


def validate_list(path, tables=None, catalog_dir=None, alpha=DEFAULT_ALPHA, do_stats=True,
                  geometry_tol=DEFAULT_GEOMETRY_TOL_M, cpp_lines=None, strict_names=False):
    name = os.path.basename(path)
    table, d, file_sha, canon = C.read_primary_file(path)
    n = len(table["id"])
    sc, sc_err = C.read_sidecar(path)
    ok_line, check_line = C.format_check_line(name, n, d, canon, file_sha, sc, sc_err)
    res = {"archivo": name, "ruta": os.path.abspath(path), "n": n, "sha256": file_sha,
           "canon_fnv1a64": C.hex16(canon), "linea_check_python": check_line}
    problems = []
    # -- structure
    clean = C.primary_defects_clean(d, n)
    res["estructural"] = {"defectos": d, "limpio": bool(clean), "coincide_con_check_del_binario": ok_line}
    if not clean:
        problems.append("estructura de la lista con defectos")
    # -- sidecar
    side = {"presente": sc is not None, "error": sc_err}
    view = 0
    r_start = None
    if sc is not None:
        side["formato_ok"] = sc.get("format") == C.LIST_FORMAT
        side["n_ok"] = int(sc.get("n", -1)) == n
        side["canon_ok"] = str(sc.get("canon_fnv1a64", "")).lower() == C.hex16(canon)
        side["sha_ok"] = (not sc.get("file_sha256")) or sc.get("file_sha256") == file_sha
        view = int(sc.get("view_index", 0))
        base = sc.get("base_radius_m")
        side["radio_base_m"] = base
        if base is not None:
            r_start = C.DET_DIST_M + float(base) + 300.0
            if "start_radius_m" in sc:
                side["radio_de_partida_ok"] = float(sc["start_radius_m"]) == r_start
            else:
                side["radio_de_partida_ok"] = None
        gid = sc.get("geometry_id")
        cc = sc.get("case_code")
        if gid in CASE_CODES:
            side["case_code_ok"] = cc == CASE_CODES[gid]
        else:
            side["case_code_ok"] = None
        if catalog_dir and gid:
            cat_path = os.path.join(catalog_dir, gid + ".json")
            if os.path.isfile(cat_path):
                cat = C.read_catalog(cat_path)
                side["catalogo_sha256"] = cat["sha256"]
                side["radio_base_catalogo_ok"] = (base is not None and float(base) == cat["base_radius_m"])
                r_start = C.DET_DIST_M + cat["base_radius_m"] + 300.0
            else:
                side["radio_base_catalogo_ok"] = None
        m = NAME_RE.match(name)
        if m:
            side["nombre_ok"] = (m.group("caso") == gid and int(m.group("vista")) == view and
                                 int(m.group("rep")) == sc.get("replica") and
                                 (sc.get("source_seed") is None or int(m.group("semilla")) == sc.get("source_seed")))
        else:
            side["nombre_ok"] = None if not strict_names else False
        bad = [k for k in ("formato_ok", "n_ok", "canon_ok", "sha_ok", "radio_de_partida_ok",
                           "case_code_ok", "radio_base_catalogo_ok", "nombre_ok")
               if side.get(k) is False]
        if bad:
            problems.append("sidecar inconsistente: " + ", ".join(bad))
    else:
        problems.append("sidecar ausente o ilegible")
    res["sidecar"] = side
    # -- comparison with the binary's own line
    if cpp_lines is not None:
        kv = cpp_lines.get(name)
        if kv is None:
            res["comparacion_con_binario"] = {"presente": False}
            problems.append("no hay linea OE1-CHECK-PRIMARIES del binario para " + name)
        else:
            rows = compare_with_cpp(kv, d, n, canon, file_sha)
            res["comparacion_con_binario"] = {"presente": True, "campos": rows,
                                              "todo_igual": all(r["igual"] for r in rows)}
            if not all(r["igual"] for r in rows):
                problems.append("el binario y Python difieren en: " +
                                ", ".join(r["campo"] for r in rows if not r["igual"]))
    # -- geometry
    kin = None
    if r_start is not None and n > 0 and clean:
        geo, kin = geometry_checks(table, view, r_start, geometry_tol)
        res["geometria"] = geo
        if not geo["ok"]:
            problems.append("geometria: la retroproyeccion no cae en el panel o el radio de partida no coincide")
        lo, hi = geo["elevacion_rango_grados"]
        half = tables["half_step_deg"] if tables else 0.0882355
        rng_ok = (lo >= C.EL_MIN_DEG - half - 1e-9 and hi <= C.EL_MAX_DEG + half + 1e-9 and
                  max(abs(geo["acimut_relativo_rango_grados"][0]),
                      abs(geo["acimut_relativo_rango_grados"][1])) <= C.AZ_HALF_DEG + 1e-9)
        geo["rango_angular_ok"] = bool(rng_ok)
        if not rng_ok:
            problems.append("geometria: elevacion o acimut relativo fuera del rango de la rejilla")
    else:
        res["geometria"] = {"omitida": True}
    # -- statistics
    stat_flags = 0
    if do_stats and tables is not None and kin is not None:
        tests = statistical_checks(table, kin, tables, DEFAULT_ALPHA if alpha is None else alpha)
        stat_flags = sum(1 for t in tests if t["marcada"])
        pmin = min(t["p"] for t in tests) if tests else 1.0
        res["estadistica"] = {"alfa": alpha, "pruebas": tests, "p_minimo": pmin, "marcadas": stat_flags,
                              "scipy_disponible": _scipy() is not None,
                              "nota": ("p < alfa exige investigar antes de transportar; con ~22 pruebas "
                                       "por lista la probabilidad de una marca espuria es ~2 por mil")}
    else:
        res["estadistica"] = {"omitida": True}
    res["problemas"] = problems
    if problems:
        res["veredicto"] = "FALLA"
    elif stat_flags:
        res["veredicto"] = "REVISAR_ESTADISTICA"
    else:
        res["veredicto"] = "OK"
    return res


def cmd_validate(args):
    missing = [p for p in list(args.lists) + list(args.cpp_check or []) if not os.path.isfile(p)]
    if args.out and os.path.exists(args.out):
        missing.append(args.out + " (ya existe: no se sobrescribe)")
    if missing:
        for p in missing:
            print("entrada inutilizable: " + p, file=sys.stderr)
        return 2
    try:
        tables = C.load_tables(args.tables) if args.tables else None
    except (OSError, ValueError, KeyError) as exc:
        print("no se pueden leer las tablas del generador en %s: %s" % (args.tables, exc), file=sys.stderr)
        return 2
    cpp = _cpp_lines(args.cpp_check) if args.cpp_check else None
    results = []
    for path in args.lists:
        res = validate_list(path, tables, args.catalog_dir, args.alpha, not args.no_stats,
                            args.geometry_tol, cpp, args.strict_names)
        results.append(res)
        print("%-52s n=%-7d %s  fnv=%s  p_min=%s" % (
            res["archivo"], res["n"], res["veredicto"], res["canon_fnv1a64"],
            ("%.3g" % res["estadistica"]["p_minimo"]) if "p_minimo" in res["estadistica"] else "-"))
        for pr in res["problemas"]:
            print("    - " + pr)
        if "pruebas" in res["estadistica"]:
            for t in res["estadistica"]["pruebas"]:
                if t["marcada"]:
                    print("    * MARCADA p=%.3g: %s" % (t["p"], t["prueba"]))
    cross = {}
    if len(results) > 1:
        fnvs = [r["canon_fnv1a64"] for r in results]
        shas = [r["sha256"] for r in results]
        cross = {"listas": len(results), "fnv_canonicos_distintos": len(set(fnvs)) == len(fnvs),
                 "sha256_distintos": len(set(shas)) == len(shas)}
        if args.expect_distinct and not (cross["fnv_canonicos_distintos"] and cross["sha256_distintos"]):
            for r in results:
                r["problemas"].append("hay listas con el mismo contenido (FNV o SHA-256 repetidos)")
                r["veredicto"] = "FALLA"
    failed = [r for r in results if r["veredicto"] == "FALLA"]
    flagged = [r for r in results if r["veredicto"] == "REVISAR_ESTADISTICA"]
    report = {"formato": "oe1-validacion-listas/1", "listas": results, "cruzada": cross,
              "veredicto": "FALLA" if failed else ("REVISAR_ESTADISTICA" if flagged else "OK"),
              "herramienta": {"scripts_sha256": C.script_hashes(C.tool_paths()),
                              "entorno": C.env_versions()}}
    if args.out:
        C.write_text_new(args.out, C.dumps_json(report))
    if args.print_check:
        for r in results:
            print(r["linea_check_python"])
    print("VALIDACION DE LISTAS:", report["veredicto"])
    if failed:
        return 3
    if flagged:
        return 4
    return 0


# --------------------------------------------------------------------------
# derive (technical test T-P)
# --------------------------------------------------------------------------
def derive_list(base_path, other_path, k, out_path):
    """New list: rows 0..k-1 from ``other``, the rest from ``base``.

    The sidecar keeps the identity fields of ``base`` (geometry, view, radius,
    case code, replica) so the same ``--case-code/--replica`` arguments work;
    ``kind`` becomes ``derivada-T-P``.
    """
    tb, db, _, _ = C.read_primary_file(base_path)
    to, do_, _, _ = C.read_primary_file(other_path)
    for d_, label in ((db, "base"), (do_, "other")):
        if not C.primary_defects_clean(d_, 1):
            raise SystemExit("la lista %s tiene defectos" % label)
    n = len(tb["id"])
    if not 0 < k < n or len(to["id"]) < k:
        raise SystemExit("K debe cumplir 0 < K < N y la otra lista debe tener al menos K filas")
    sb, err = C.read_sidecar(base_path)
    if sb is None:
        raise SystemExit(err)
    vals = tb["vals"].copy()
    pdg = tb["pdg"].copy()
    charge = tb["charge"].copy()
    vals[:k] = to["vals"][:k]
    pdg[:k] = to["pdg"][:k]
    charge[:k] = to["charge"][:k]
    ids = tb["id"]
    raw = C.write_primary_csv(out_path, ids, vals, pdg, charge)
    canon = C.canonical_fnv(ids, vals, pdg)
    side = dict(sb)
    side["kind"] = "derivada-T-P"
    side["derivada_de"] = os.path.basename(base_path)
    side["filas_sustituidas"] = k
    side["sustituidas_con"] = os.path.basename(other_path)
    side["canon_fnv1a64"] = C.hex16(canon)
    side["file_sha256"] = C.sha256_bytes(raw)
    side["n"] = n
    side.pop("binary_sha256", None)
    C.write_text_new(C.sidecar_path(out_path), C.dumps_json(side))
    _, d2, _, _ = C.read_primary_file(out_path)
    if not C.primary_defects_clean(d2, n):
        raise SystemExit("la lista derivada tiene defectos: %r" % d2)
    return {"salida": out_path, "sha256": side["file_sha256"], "canon_fnv1a64": side["canon_fnv1a64"], "n": n, "k": k}


def cmd_derive(args):
    res = derive_list(args.base, args.other, args.replace_first, args.out)
    print("lista derivada %s (N=%d, primeras %d filas sustituidas) sha256=%s fnv=%s" %
          (res["salida"], res["n"], res["k"], res["sha256"], res["canon_fnv1a64"]))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="valida listas de primarios")
    v.add_argument("lists", nargs="+", help="archivos .primarios.csv (cada uno con su .meta.json)")
    v.add_argument("--tables", help="carpeta con row_weights_30d.txt, p_cdf.txt y meta.json")
    v.add_argument("--catalog-dir", help="carpeta del catalogo (cruza base_radius_m)")
    v.add_argument("--out", help="validacion-listas.json (no se sobrescribe)")
    v.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    v.add_argument("--no-stats", action="store_true", help="solo estructura y geometria")
    v.add_argument("--geometry-tol", type=float, default=DEFAULT_GEOMETRY_TOL_M)
    v.add_argument("--cpp-check", nargs="*", help="salida(s) de muography_sim --check-primaries")
    v.add_argument("--expect-distinct", action="store_true",
                   help="exige FNV canonicos y SHA-256 distintos entre las listas")
    v.add_argument("--strict-names", action="store_true")
    v.add_argument("--print-check", action="store_true", help="imprime la linea OE1-CHECK-PRIMARIES de Python")
    v.set_defaults(fn=cmd_validate)
    d = sub.add_parser("derive", help="T-P: sustituye las primeras K filas por las de otra lista")
    d.add_argument("--base", required=True)
    d.add_argument("--other", required=True)
    d.add_argument("--replace-first", type=int, required=True)
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_derive)
    return ap


def main(argv=None):
    C.utf8_streams()
    args = build_parser().parse_args(argv)
    return C.run_command(args.fn, args)


if __name__ == "__main__":
    sys.exit(main())
