"""Tests of the stage-1 analysis on synthetic fixtures.

Every contract path is exercised: a pristine attempt passes, and every blocking
or non-blocking check is broken on purpose (records edited before the files are
written, or files edited afterwards) to prove that the right check fails and
that the exit code and the stamp are the intended ones. Several defects are
injected into different runs of one attempt so the whole suite stays fast; each
test then asserts on the specific check and run.
"""
import contextlib
import csv
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import tokenize
import unittest
import zipfile
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)

import oe1_stage1_analysis as A  # noqa: E402
import oe1_stage1_common as C  # noqa: E402
import fixtures as F  # noqa: E402

N = 200
MASK = F.make_test_mask()
INSIDE = (31, 101)      # a pixel of the test R_int
OUTSIDE = (60, 60)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def set_det(rec, iel, iaz, ekin=50.0):
    rec.update(status="DET", detected=1, iel=iel, iaz=iaz, in_rint=int(MASK[iel, iaz]), plane=True,
               ekin=ekin, el=8.0, rel=1.0, end_vol="detector", end_process="Transportation")


def set_status(rec, status):
    rec.update(status=status, detected=0, iel=-1, iaz=-1, in_rint=-1, plane=False, ekin=0.0, el=0.0, rel=0.0,
               end_vol="edifice", end_process="Decay" if status == "DECAY" else "Transportation")


def touch_conduit(*recs):
    for r in recs:
        r["lc"] = 5.0


def diverge(a, b):
    b["rng1"] = a["rng1"] ^ 0x1234


def load(path):
    return C.read_json(path)


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def edit(path, fn):
    with open(path, "rb") as fh:
        data = fh.read()
    with open(path, "wb") as fh:
        fh.write(fn(data))


def edit_json(path, fn):
    def wrap(data):
        m = json.loads(data.decode())
        fn(m)
        return (json.dumps(m, indent=1) + "\n").encode()

    edit(path, wrap)


_counter = [0]


def analyze(fx, tmp, attempt=None, extra=(), test_mask=True):
    _counter[0] += 1
    out = os.path.join(tmp, "out%03d" % _counter[0])
    attempt = attempt or fx["attempt"]
    argv = ["--attempt", attempt, "--manifest", os.path.join(attempt, "manifiesto-corridas.tsv"),
            "--roi-mask", fx["mask"]["text"], "--roi-frozen", fx["mask"]["frozen"], "--roi-npz", fx["mask"]["npz"],
            "--out", out, "--crossing-reference", F.CROSSING_REFERENCE, "--no-figures"]
    if test_mask:
        argv.append("--test-mask")
    argv += list(extra)
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        rc = A.main(argv)
    return rc, out, buf.getvalue(), err.getvalue()


def checks_of(ctr, contract, scope_part="", desc_part=""):
    out = []
    for scope, lst in ctr["contratos"].get(contract, {}).get("ambitos", {}).items():
        if scope_part in scope:
            out += [c for c in lst if desc_part in c["descripcion"]]
    return out


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="oe1an_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def assertCheckFails(self, ctr, contract, scope_part, desc_part):
        found = checks_of(ctr, contract, scope_part, desc_part)
        self.assertTrue(found, "no check %s / %s / %s" % (contract, scope_part, desc_part))
        self.assertTrue(any(c["ok"] is False for c in found),
                        [(c["ambito"], c["ok"], c["descripcion"]) for c in found])
        return [c for c in found if c["ok"] is False][0]

    def assertCheckPasses(self, ctr, contract, scope_part, desc_part):
        found = checks_of(ctr, contract, scope_part, desc_part)
        self.assertTrue(found, "no check %s / %s / %s" % (contract, scope_part, desc_part))
        self.assertTrue(all(c["ok"] is True for c in found),
                        [(c["ambito"], c["ok"], c["descripcion"]) for c in found])

    def failing(self, ctr):
        return [(c["contrato"], c["ambito"], c["descripcion"]) for grp in ctr["contratos"].values()
                for lst in grp["ambitos"].values() for c in lst if c["ok"] is False]

    def clone(self, fx):
        root = os.path.join(self.tmp, "clone%03d" % len(os.listdir(self.tmp)))
        F.clone_attempt(fx["attempt"], root)
        return root


# --------------------------------------------------------------------------
class PristineTests(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fx = F.build_attempt(os.path.join(cls.tmp, "pristine"), n=N, lateral_columns=True)
        cls.rc, cls.out, cls.stdout, cls.stderr = analyze(cls.fx, cls.tmp, extra=("--high-energy-gev", "0"))
        cls.ctr = load(os.path.join(cls.out, "contratos.json"))
        cls.res = load(os.path.join(cls.out, "resultados-etapa1.json"))

    def test_k7_ignores_wall_time_and_rusage_and_lists_what_it_ignored(self):
        c = checks_of(self.ctr, "K7", "g01-A-r1-rep", "meta.json igual")
        self.assertEqual(len(c), 1)
        self.assertTrue(c[0]["ok"])
        ignored = c[0]["evidencia"]["claves_de_tiempo_ignoradas"]
        # the keys the C++ binary writes (RunAction.cc, plan W3), next to wall_seconds
        for key in ("wall_seconds", "seconds_per_event", "rusage_user_cpu_seconds",
                    "rusage_system_cpu_seconds", "rusage_max_rss_kb"):
            self.assertIn(key, ignored)

    def test_rusage_of_the_meta_is_copied_as_found(self):
        rec = self.res["recursos_del_meta"]
        self.assertEqual(sorted(rec), sorted(self.ctr["contratos"]["K6"]["ambitos"]))
        self.assertEqual(sorted(rec["g06-B-r1"]),
                         ["rusage_max_rss_kb", "rusage_system_cpu_seconds", "rusage_user_cpu_seconds"])

    def test_k8_ignores_the_binary_sha256_that_differs_between_a_and_b(self):
        # the A and B executables differ by construction; everything else recorded in oe1_etapa1
        # (SHA-256 of the catalog JSON and of the tables, region, counts, signature) must be equal
        c = checks_of(self.ctr, "K8", "g01-A-r1 vs g01-B-r1", "meta.json igual")
        self.assertEqual(len(c), 1)
        self.assertTrue(c[0]["ok"], c[0]["evidencia"])
        self.assertIn("SHA-256 del binario", c[0]["descripcion"])

    def test_lateral_displacement_is_reported_not_required_to_be_small(self):
        c = checks_of(self.ctr, "K6", "g06-B-r1", "desplazamiento lateral")
        self.assertEqual(len(c), 1)
        self.assertTrue(c[0]["ok"])
        self.assertGreater(c[0]["evidencia"]["max_lat_mm"], 1.0)           # far above the old 1e-6 mm limit
        self.assertGreater(c[0]["evidencia"]["fraccion_mayor_que_1e-6_mm"], 0.5)
        z = checks_of(self.ctr, "K6", "g06-B-r1", "escala del error")
        self.assertTrue(z and all(x["ok"] is True for x in z))
        row = read_rows(os.path.join(self.out, "tabla-corridas.csv"))[0]
        for col in ("max_lat_mm", "fraccion_max_lat_mayor_1e-6_mm", "max_lat_z_mm", "max_suma_lat_z_mm"):
            self.assertIn(col, row)
        self.assertGreater(float(row["max_lat_mm"]), 1.0)
        lat = self.res["desplazamiento_lateral"]
        self.assertEqual(sorted(lat), sorted(self.ctr["contratos"]["K6"]["ambitos"]))
        self.assertIn("no es la trayectoria navegada", self.res["leyendas"]["cuerda"])

    def test_everything_passes(self):
        self.assertEqual(self.rc, 0, self.stdout + self.stderr)
        ctr = self.ctr
        self.assertEqual(ctr["estado"], "OK")
        self.assertEqual(ctr["bloqueantes_fallidos"], [])
        self.assertEqual(ctr["no_bloqueantes_fallidos"], [])
        self.assertEqual(ctr["no_verificables"], [])
        self.assertFalse(ctr["atribucion_al_solape_bloqueada"])
        for k in ("K0", "K6", "K7", "K8", "K9"):
            self.assertEqual(ctr["contratos"][k]["veredicto"], "OK", k)
        self.assertEqual(len(ctr["contratos"]["K6"]["ambitos"]), 13)
        for name in ("contratos.json", "resultados-etapa1.json", "tabla-corridas.csv", "transiciones.csv",
                     "matriz-clases.csv", "cambios-bin.csv", "muones-discordantes.csv", "cruces-union.csv",
                     "comando-analisis.txt", "mapas-etapa1.npz"):
            self.assertTrue(os.path.isfile(os.path.join(self.out, name)), name)
        self.assertFalse(os.path.exists(os.path.join(self.out, "BLOQUEADO.txt")))
        self.assertIn("CONTRATOS: OK", self.stdout)

    def test_k6_covers_the_listed_contract_items(self):
        wanted = ("archivos presentes", "SHA256SUMS", "estado del guion", "clave oe1_etapa1",
                  "comprobaciones del binario", "filas = N", "IDs únicos", "ninguna fila ABORTED",
                  "histograma reconstruido", "número de filas DET", "in_rint del binario",
                  "max_lat", "region.applicable", "procedencia", "firma de vol", "FNV-1a-64", "RUN DONE")
        descs = [c["descripcion"] for c in self.ctr["contratos"]["K6"]["ambitos"]["g06-B-r1"]]
        for w in wanted:
            self.assertTrue(any(w in d for d in descs), w)
        self.assertTrue(any("B sin GeomVol1002" in d for d in descs))
        a_descs = [c["descripcion"] for c in self.ctr["contratos"]["K6"]["ambitos"]["g06-A-r1"]]
        self.assertTrue(any("avisos de A" in d for d in a_descs))

    def test_numbers(self):
        res = self.res
        self.assertEqual(res["estado"], "OK")
        for caso in ("geo-train-06", "geo-train-01"):
            g = res["casos"][caso]["agrupado"]
            self.assertEqual(g["N"], 3 * N)                       # the A/A repeat is not pooled
            self.assertEqual((g["n01"], g["n10"], g["b"]), (0, 0, 0))
            self.assertEqual(g["delta"], 0.0)
            self.assertEqual(g["n00"] + g["n01"] + g["n10"] + g["n11"], 3 * N)
            self.assertIn("denominador N = 600", g["denominador"])
            self.assertFalse(g["mcnemar_exacto"]["definido"])
            z = C._norm_ppf(0.975)
            self.assertAlmostEqual(g["tango_sup"], z * z / (3 * N + z * z), delta=1e-12)
            self.assertAlmostEqual(g["tango_inf"], -g["tango_sup"], delta=1e-12)
            self.assertEqual(g["por_replica"]["frase"], A.SENTENCE_SD)
            self.assertEqual(g["nota_intervalo"], A.NOTE_INTERVAL)
            self.assertEqual(len(g["por_replica"]["deltas_por_replica"]), 3)
            self.assertIn("nivel", g)
        self.assertEqual(res["casos"]["geo-train-01"]["rol"], "control angular")
        self.assertEqual(res["casos"]["geo-train-06"]["rol"], "afectado")
        self.assertIn("no identifica muones individuales", res["leyendas"]["mapas"])
        self.assertIn("repetición A/A", res["leyendas"]["repeticion_AA"])
        self.assertEqual(len(res["entradas"]["corridas"]), 13)
        self.assertIn("oe1_stage1_analysis.py", res["herramienta"]["scripts_sha256"])
        self.assertTrue(res["herramienta"]["crossing_reference"]["sha256"])
        self.assertEqual(res["entradas"]["solapes_entre_mascaras_descriptivas"], 0)

    def test_stage1_interval_note_says_it_does_not_bound_the_effect(self):
        g = self.res["casos"]["geo-train-06"]["agrupado"]
        self.assertIn("no acota el efecto", g["nota_intervalo"])
        self.assertIn("nota", g)
        self.assertEqual(g["nota"], "diagnóstico; etapa 1 no dimensionada")
        self.assertIn("no dimensionada", self.res["naturaleza"])
        # the flag is a constant: stage 1 is not sized (PROT section 4), whatever the counts are
        self.assertIs(g["intervalo_no_acota_el_efecto"], True)
        # the observed "trivial bound" is gone; the note asserts no inequality
        self.assertNotIn("cota_trivial_observada", g)
        self.assertNotIn("cota_trivial_observada_por_millon", g)
        for symbol in ("≤", "≥", "<", ">", "<=", ">="):
            self.assertNotIn(symbol, g["nota_intervalo"])
            self.assertNotIn(symbol, A.NOTE_INTERVAL)
        self.assertNotIn("más estrecha", g["nota_intervalo"])

    def test_pooled_clopper_pearson_upper_bounds_are_labelled_data_and_assumption(self):
        for caso in ("geo-train-06", "geo-train-01"):
            g = self.res["casos"][caso]["agrupado"]
            cp = g["cotas_superiores_clopper_pearson"]
            self.assertEqual(cp["N"], 3 * N)
            self.assertAlmostEqual(cp["nivel_bilateral"], 0.95)
            self.assertIn("dato", cp["etiqueta"])
            self.assertIn("supuesto", cp["etiqueta"])
            for tag, key in (("p_A", "sumY_A"), ("p_B", "sumY_B")):
                b = cp[tag]
                self.assertEqual(b["suma_Y"], g[key])
                lo, hi = C.clopper_pearson_interval(g[key], 3 * N, 0.05)
                self.assertEqual((b["inf"], b["sup"]), (lo, hi))
                self.assertAlmostEqual(b["sup_por_millon"], hi * 1e6)
                self.assertAlmostEqual(b["estimacion"], g[key] / (3.0 * N))
                self.assertGreaterEqual(b["sup"], b["estimacion"])
        self.assertIn("cotas_clopper_pearson", self.res["leyendas"])

    def test_run_table(self):
        rows = read_rows(os.path.join(self.out, "tabla-corridas.csv"))
        self.assertEqual(len(rows), 13)
        self.assertEqual(rows[-1]["corrida"], "g01-A-r1-rep")
        self.assertEqual(rows[-1]["rol"], "repeticion-AA")
        r = next(x for x in rows if x["corrida"] == "g06-A-r1")
        self.assertEqual(r["N"], str(N))
        self.assertEqual(int(r["DET"]), int(r["detectados"]))
        self.assertEqual(r["avisos_por_codigo"], "GeomVol1002=1")
        self.assertEqual(r["nodo"], "guane01")
        self.assertEqual(r["trabajo"], "30600")
        # PLAN W3: CPU and RSS come from the getrusage keys of the meta (the process itself), not from
        # recursos.json (which run_timed.py writes for the child): 12.25 + 0.1125 s and 91250 kB, not 90000
        self.assertEqual(r["rss_kb"], "91250")
        self.assertAlmostEqual(float(r["cpu_s"]), 12.25 + 0.1125)
        self.assertEqual((r["fuente_cpu"], r["fuente_rss"]), ("meta.rusage", "meta.rusage"))
        self.assertEqual(r["cruces_profundos"], "4")
        self.assertAlmostEqual(float(r["eficiencia"]), int(r["detectados"]) / N)
        self.assertEqual(next(x for x in rows if x["corrida"] == "g06-B-r1")["avisos_por_codigo"], "ninguno")
        self.assertEqual(next(x for x in rows if x["corrida"] == "g01-B-r2")["cruces_profundos"], "NA")

    def test_transition_rows_per_pair_and_pooled(self):
        rows = read_rows(os.path.join(self.out, "transiciones.csv"))
        self.assertEqual(len(rows), 8)                             # 2 cases x (3 replicas + pooled)
        self.assertEqual({r["caso"] for r in rows if r["ambito"] == "agrupado"}, {"geo-train-06", "geo-train-01"})
        for r in rows:
            self.assertEqual(int(r["n00"]) + int(r["n01"]) + int(r["n10"]) + int(r["n11"]), int(r["N"]))
            self.assertIn("denominador N =", r["denominador"])
            self.assertIn("diagnóstico", r["nota"])
        self.assertEqual([r for r in rows if r["caso"] == "geo-train-06"][0]["cruces_union"], "4")
        self.assertEqual([r for r in rows if r["caso"] == "geo-train-01"][0]["cruces_union"], "NA")

    def test_class_matrix_is_diagonal_and_labelled(self):
        rows = read_rows(os.path.join(self.out, "matriz-clases.csv"))
        self.assertTrue(all(r["clase_A"] == r["clase_B"] for r in rows))
        classes = {r["clase_A"] for r in rows}
        self.assertTrue({"R_int", "R_rock", "no detectado"} <= classes)
        self.assertTrue(all(r["etiqueta"] == "control angular" for r in rows if r["caso"] == "geo-train-01"))
        pooled = [r for r in rows if r["caso"] == "geo-train-06" and r["ambito"] == "agrupado"]
        self.assertEqual(sum(int(r["n"]) for r in pooled), 3 * N)
        self.assertEqual(self.res["casos"]["geo-train-06"]["matriz_clases_agrupada"]["R_int"]["R_int"],
                         sum(int(r["n"]) for r in pooled if r["clase_A"] == "R_int"))

    def test_crossings_and_straight_line_check(self):
        cr = self.res["casos"]["geo-train-06"]["cruces_agrupados"]
        self.assertEqual((cr["cruces_A"], cr["cruces_B"], cr["cruces_ambas"], cr["cruces_union"]), (12, 12, 12, 12))
        pair = self.res["casos"]["geo-train-06"]["pares"][0]
        self.assertEqual(pair["cruces"]["tamano_U"], 4)
        self.assertEqual(pair["cruces"]["delta_condicional_U"], 0.0)
        self.assertIn("|U| = 4", pair["cruces"]["nota_U"])
        line = self.res["verificacion_de_la_recta"]["g06-A-r1"]
        self.assertEqual(sum(line["tabla_2x2"].values()), N)      # E >= 0 keeps every muon in the 2x2 table
        rows = read_rows(os.path.join(self.out, "cruces-union.csv"))
        self.assertEqual(len(rows), 12)
        self.assertEqual({r["mat_in_A"] for r in rows}, {"A0"})
        self.assertEqual({r["mat_in_B"] for r in rows}, {"A1"})

    def test_maps_are_saved(self):
        with np.load(os.path.join(self.out, "mapas-etapa1.npz")) as z:
            a, b, d = z["geo-train-06__A_sobre_N"], z["geo-train-06__B_sobre_N"], z["geo-train-06__B_menos_A_sobre_N"]
        self.assertEqual(a.shape, (256, 256))
        self.assertAlmostEqual(float(a.sum()) * 3 * N, round(float(a.sum()) * 3 * N), places=6)
        self.assertTrue(np.array_equal(d, b - a))

    def test_command_file(self):
        with open(os.path.join(self.out, "comando-analisis.txt"), encoding="utf-8") as fh:
            text = fh.read()
        for part in ("comando=", "--test-mask", "script_sha256[oe1_stage1_analysis.py]=", "manifiesto_sha256=",
                     "numpy=", "python="):
            self.assertIn(part, text)

    def test_deterministic_across_runs(self):
        rc2, out2, _, _ = analyze(self.fx, self.tmp, extra=("--high-energy-gev", "0"))
        for name in ("tabla-corridas.csv", "transiciones.csv", "matriz-clases.csv", "cambios-bin.csv",
                     "muones-discordantes.csv", "cruces-union.csv"):
            with open(os.path.join(self.out, name), "rb") as f1, open(os.path.join(out2, name), "rb") as f2:
                self.assertEqual(f1.read(), f2.read(), name)
        c2 = load(os.path.join(out2, "contratos.json"))
        a, b = dict(self.ctr), dict(c2)
        a.pop("fecha_utc")
        b.pop("fecha_utc")
        self.assertEqual(a, b)

    def test_output_folder_must_be_new_and_inputs_must_exist(self):
        rc, out, _, _ = analyze(self.fx, self.tmp)
        self.assertEqual(rc, 0)
        argv = ["--attempt", self.fx["attempt"], "--roi-mask", self.fx["mask"]["text"], "--roi-frozen",
                self.fx["mask"]["frozen"], "--out", out, "--test-mask"]
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc2 = A.main(argv)
        self.assertEqual(rc2, 2)
        self.assertIn("ya existe", err.getvalue())
        rc3, _, _, _ = analyze(self.fx, self.tmp, attempt=os.path.join(self.tmp, "no-existe"))
        self.assertEqual(rc3, 2)

    def test_real_mask_constants_are_enforced_without_the_test_flag(self):
        rc, out, stdout, _ = analyze(self.fx, self.tmp, test_mask=False)
        self.assertEqual(rc, 3)
        self.assertCheckFails(load(os.path.join(out, "contratos.json")), "K0", "", "valores congelados del plan")
        self.assertTrue(os.path.isfile(os.path.join(out, "BLOQUEADO.txt")))
        self.assertIn("BLOQUEADO", stdout)

    def test_muon_fnv_can_be_skipped_but_is_reported_as_unverified(self):
        rc, out, _, _ = analyze(self.fx, self.tmp, extra=("--skip-muon-fnv",))
        self.assertEqual(rc, 0)
        ctr = load(os.path.join(out, "contratos.json"))
        self.assertEqual(len(ctr["no_verificables"]), 13)
        self.assertEqual(ctr["contratos"]["K6"]["veredicto"], "OK_CON_AVISOS")

    def test_figures_are_optional(self):
        argv = ["--attempt", self.fx["attempt"], "--roi-mask", self.fx["mask"]["text"], "--roi-frozen",
                self.fx["mask"]["frozen"], "--roi-npz", self.fx["mask"]["npz"], "--test-mask", "--out",
                os.path.join(self.tmp, "conFiguras"), "--crossing-reference", F.CROSSING_REFERENCE]
        with contextlib.redirect_stdout(io.StringIO()):
            rc = A.main(argv)
        self.assertEqual(rc, 0)
        try:
            import matplotlib  # noqa: F401
            self.assertTrue(os.path.isfile(os.path.join(self.tmp, "conFiguras", "figuras", "mapa-geo-train-06.png")))
        except ImportError:
            self.assertFalse(os.path.exists(os.path.join(self.tmp, "conFiguras", "figuras")))


# --------------------------------------------------------------------------
class DeltaTests(Base):
    """Discordant muons: counts, intervals and derived tables."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        def tweak(records, entries):
            a1, b1 = records["g06-A-r1"], records["g06-B-r1"]
            set_status(a1[60], "DECAY")
            set_det(b1[60], *INSIDE)                     # A misses R_int, B hits it: n01
            set_det(a1[61], 32, 102)
            set_status(b1[61], "MU_CAPTURE")             # A hits, B does not: n10
            set_det(a1[62], *OUTSIDE)
            set_det(b1[62], OUTSIDE[0], OUTSIDE[1] + 1)  # both detected, one bin apart, outside R_int
            for i in (60, 61, 62):
                touch_conduit(a1[i], b1[i])
                diverge(a1[i], b1[i])
            a2, b2 = records["g06-A-r2"], records["g06-B-r2"]
            for i in (70, 71):
                set_status(a2[i], "WORLD_EXIT")
                set_det(b2[i], 30, 100)                  # n01 twice in replica 2
                touch_conduit(a2[i], b2[i])
                diverge(a2[i], b2[i])

        cls.fx = F.build_attempt(os.path.join(cls.tmp, "delta"), n=N, tweak=tweak)
        cls.rc, cls.out, cls.stdout, cls.stderr = analyze(cls.fx, cls.tmp)
        cls.res = load(os.path.join(cls.out, "resultados-etapa1.json"))

    def test_counts_and_deltas(self):
        self.assertEqual(self.rc, 0, self.stdout)
        pairs = self.res["casos"]["geo-train-06"]["pares"]
        p1, p2, p3 = pairs
        self.assertEqual((p1["n01"], p1["n10"], p1["b"], p1["delta"]), (1, 1, 2, 0.0))
        self.assertEqual((p2["n01"], p2["n10"], p2["b"]), (2, 0, 2))
        self.assertAlmostEqual(p2["delta"], 2.0 / N)
        self.assertAlmostEqual(p2["delta_por_millon"], 2.0 / N * 1e6)
        self.assertEqual(p2["denominador"], "cuentas por millón de primarios generados; denominador N = %s" %
                         format(N, ",d").replace(",", " "))
        for p in pairs:
            self.assertEqual(p["n00"] + p["n01"] + p["n10"] + p["n11"], N)
        g = self.res["casos"]["geo-train-06"]["agrupado"]
        self.assertEqual((g["n01"], g["n10"], g["N"]), (3, 1, 3 * N))
        self.assertAlmostEqual(g["delta"], 2.0 / (3 * N))
        lo, hi = C.tango_interval(3, 1, 3 * N)
        self.assertAlmostEqual(g["tango_inf"], lo, places=15)
        self.assertAlmostEqual(g["tango_sup"], hi, places=15)
        self.assertTrue(g["tango_inf"] < g["delta"] < g["tango_sup"])
        mc = g["mcnemar_exacto"]
        self.assertTrue(mc["definido"])
        self.assertAlmostEqual(mc["p_dos_colas"], 2 * (1 + 4) / 16.0, places=12)   # b = 4, smaller count 1
        reps = g["por_replica"]
        self.assertAlmostEqual(reps["media"], float(np.mean([0.0, 2.0 / N, 0.0])))
        self.assertAlmostEqual(reps["DE"], float(np.std([0.0, 2.0 / N, 0.0], ddof=1)))
        self.assertEqual(reps["rango"], [0.0, 2.0 / N])
        self.assertEqual(self.res["casos"]["geo-train-01"]["agrupado"]["b"], 0)

    def test_detection_transitions_and_efficiency(self):
        rows = read_rows(os.path.join(self.out, "transiciones.csv"))
        r1 = next(r for r in rows if r["caso"] == "geo-train-06" and r["ambito"].endswith("1"))
        self.assertEqual((int(r1["m01"]), int(r1["m10"])), (1, 1))       # id 60 gained, id 61 lost
        self.assertEqual(float(r1["delta_eficiencia"]), 0.0)
        r2 = next(r for r in rows if r["caso"] == "geo-train-06" and r["ambito"].endswith("2"))
        self.assertEqual((int(r2["m01"]), int(r2["m10"])), (2, 0))
        self.assertAlmostEqual(float(r2["delta_eficiencia"]), 2.0 / N)

    def test_bin_changes_among_detected_in_both(self):
        rows = read_rows(os.path.join(self.out, "cambios-bin.csv"))
        r1 = [r for r in rows if r["caso"] == "geo-train-06" and r["ambito"] == "replica 1"]
        changed = [r for r in r1 if r["tipo"] == "cambio de bin"]
        self.assertEqual([(r["delta_iEl"], r["delta_iAz"], r["n"]) for r in changed], [("0", "1", "1")])
        self.assertGreater(int(next(r for r in r1 if r["tipo"] == "mismo bin")["n"]), 50)
        tr = read_rows(os.path.join(self.out, "transiciones.csv"))
        self.assertEqual(next(r for r in tr if r["caso"] == "geo-train-06" and r["ambito"] == "agrupado")
                         ["bins_distintos_entre_detectados_en_ambas"], "1")

    def test_class_matrix_lists_the_ids_off_the_diagonal(self):
        rows = read_rows(os.path.join(self.out, "matriz-clases.csv"))
        cell = [r for r in rows if r["caso"] == "geo-train-06" and r["ambito"] == "replica 1" and
                r["clase_A"] == "no detectado" and r["clase_B"] == "R_int"]
        self.assertEqual(len(cell), 1)
        self.assertEqual(cell[0]["n"], "1")
        self.assertEqual(cell[0]["ids_fuera_de_la_diagonal"].split(), ["60"])
        cell = [r for r in rows if r["ambito"] == "replica 1" and r["caso"] == "geo-train-06" and
                r["clase_A"] == "R_int" and r["clase_B"] == "no detectado"]
        self.assertEqual(cell[0]["ids_fuera_de_la_diagonal"].split(), ["61"])
        pooled = next(r for r in rows if r["caso"] == "geo-train-06" and r["ambito"] == "agrupado" and
                      r["clase_A"] == "no detectado" and r["clase_B"] == "R_int")
        self.assertEqual(int(pooled["n"]), 3)
        self.assertIn("r1:60", pooled["ids_fuera_de_la_diagonal"])
        self.assertIn("r2:70", pooled["ids_fuera_de_la_diagonal"])

    def test_discordant_table_and_k9b_classification(self):
        rows = read_rows(os.path.join(self.out, "muones-discordantes.csv"))
        non_crossers = [r for r in rows if r["clase_K9b"] != "cruza_Z"]
        self.assertEqual(len(non_crossers), 5)
        self.assertEqual({r["clase_K9b"] for r in non_crossers}, {"conducto"})
        self.assertEqual(sum(1 for r in rows if r["clase_K9b"] == "cruza_Z"), 12)   # the base crossers diverge
        r60 = next(r for r in rows if r["replica"] == "1" and r["id"] == "60")
        self.assertIn("detected", r60["campos_distintos"])
        self.assertIn("rng1", r60["campos_distintos"])
        self.assertEqual((r60["Y_A"], r60["Y_B"]), ("0", "1"))
        self.assertTrue(r60["ekin_inicial_gev"])                    # read from the list
        ctr = load(os.path.join(self.out, "contratos.json"))
        self.assertFalse(ctr["atribucion_al_solape_bloqueada"])     # all explained
        self.assertEqual(ctr["contratos"]["K9"]["veredicto"], "OK")
        self.assertEqual(sum(x["sin_explicar"] for x in ctr["discordancias_no_cruzan_K9b"]), 0)


# --------------------------------------------------------------------------
class BlockingFileDefectTests(Base):
    """One clone, eleven different file-level defects in eleven different runs."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fx = F.build_attempt(os.path.join(cls.tmp, "base"), n=N)
        root = os.path.join(cls.tmp, "defects")
        F.clone_attempt(cls.fx["attempt"], root)

        def rdir(run_id):
            e = next(x for x in cls.fx["entries"] if x["run_id"] == run_id)
            return os.path.join(root, e["salida"]), "%s_view0" % e["caso"]

        def bump(data):
            lines = data.split(b"\n")
            cells = lines[10].split(b",")
            cells[20] = str(int(cells[20]) + 1).encode()
            lines[10] = b",".join(cells)
            return b"\n".join(lines)

        d, stem = rdir("g06-B-r2")                                        # 1 histogram cell
        edit(os.path.join(d, stem + "_hist.csv"), bump)
        F.write_sums(d)
        d, stem = rdir("g01-A-r2")                                        # 2 leftover .part
        open(os.path.join(d, stem + "_muones.csv.part"), "wb").close()
        d, stem = rdir("g01-B-r1")                                        # 3 no per-muon log
        os.remove(os.path.join(d, stem + "_muones.csv"))
        F.write_sums(d)
        d, stem = rdir("g01-A-r3")                                        # 4 duplicate id

        def dup(data):
            lines = data.split(b"\n")
            f = lines[7].split(b",")
            f[6] = b"5"
            lines[7] = b",".join(f)
            return b"\n".join(lines)

        edit(os.path.join(d, stem + "_muones.csv"), dup)
        F.refresh_muon_log_meta(os.path.join(d, stem))
        d, stem = rdir("g06-B-r3")                                        # 5 warning in B
        edit(os.path.join(d, "run.log"), lambda x: b"*** G4Exception : GeomVol1002\n" + x)
        F.write_sums(d)
        d, stem = rdir("g06-A-r1")                                        # 6 no oe1_etapa1 key
        edit_json(os.path.join(d, stem + ".meta.json"), lambda m: m.pop("oe1_etapa1"))
        F.write_sums(d)
        d, stem = rdir("g06-B-r1")                                        # 7 region not applicable
        edit_json(os.path.join(d, stem + ".meta.json"), lambda m: m["oe1_etapa1"]["region"].update(applicable=False))
        F.write_sums(d)
        d, stem = rdir("g06-A-r2")                                        # 8 wrong geometry signature

        def mislabel(m):
            vols = m["oe1_etapa1"]["geometry_signature"]["volumes"]
            m["oe1_etapa1"]["geometry_signature"]["volumes"] = [v.replace("G4Tubs", "G4SubtractionSolid")
                                                                for v in vols]

        edit_json(os.path.join(d, stem + ".meta.json"), mislabel)
        F.write_sums(d)
        d, stem = rdir("g01-B-r3")                                        # 9 tampered log, sums not refreshed
        edit(os.path.join(d, stem + "_muones.csv"), lambda x: x[:-2] + b"1\n")
        with open(os.path.join(root, "estado", "g06-A-r3.status"), "wb") as fh:    # 10 driver says FAIL
            fh.write(b"g06-A-r3\tFAIL\t1\tno RUN DONE\tguane01\t30600\n")
        d, stem = rdir("g01-B-r2")                                        # 11 provenance

        def swap(m):
            m["oe1_etapa1"]["list"]["sha256"] = "0" * 64
            m["oe1_etapa1"]["variant"] = "A"

        edit_json(os.path.join(d, stem + ".meta.json"), swap)
        F.write_sums(d)
        cls.root = root
        cls.rc, cls.out, cls.stdout, cls.stderr = analyze(cls.fx, cls.tmp, attempt=root)
        cls.ctr = load(os.path.join(cls.out, "contratos.json"))

    def test_exit_code_stamp_and_minimal_output(self):
        self.assertEqual(self.rc, 3)
        self.assertEqual(self.ctr["estado"], A.STAMP)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "BLOQUEADO.txt")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "tabla-corridas.csv")))
        self.assertIn("BLOQUEADO", self.stdout)
        self.assertTrue(self.ctr["bloqueantes_fallidos"])
        self.assertEqual(self.ctr["contratos"]["K6"]["veredicto"], "FALLA")

    def test_histogram_reconstruction(self):
        self.assertCheckFails(self.ctr, "K6", "g06-B-r2", "histograma reconstruido")
        c = self.assertCheckFails(self.ctr, "K6", "g06-B-r2", "número de filas DET")
        self.assertEqual(c["evidencia"]["suma_hist"], c["evidencia"]["DET"] + 1)

    def test_part_file(self):
        self.assertCheckFails(self.ctr, "K6", "g01-A-r2", "archivos presentes")

    def test_missing_muon_log(self):
        self.assertCheckFails(self.ctr, "K6", "g01-B-r1", "archivos presentes")

    def test_duplicate_id(self):
        self.assertCheckFails(self.ctr, "K6", "g01-A-r3", "IDs únicos")

    def test_geometry_warning_in_b(self):
        c = self.assertCheckFails(self.ctr, "K6", "g06-B-r3", "B sin GeomVol1002")
        self.assertEqual(c["evidencia"]["codigos"], {"GeomVol1002": 1})

    def test_missing_oe1_key(self):
        self.assertCheckFails(self.ctr, "K6", "g06-A-r1", "clave oe1_etapa1")

    def test_region_not_applicable(self):
        self.assertCheckFails(self.ctr, "K6", "g06-B-r1", "region.applicable")

    def test_geometry_signature_versus_variant(self):
        self.assertCheckFails(self.ctr, "K6", "g06-A-r2", "firma de vol")

    def test_tampering_is_caught_by_sums_and_fnv(self):
        self.assertCheckFails(self.ctr, "K6", "g01-B-r3", "SHA256SUMS")
        self.assertCheckFails(self.ctr, "K6", "g01-B-r3", "FNV-1a-64")

    def test_driver_status(self):
        self.assertCheckFails(self.ctr, "K6", "g06-A-r3", "estado del guion")

    def test_provenance(self):
        c = self.assertCheckFails(self.ctr, "K6", "g01-B-r2", "procedencia")
        self.assertIn("variant", c["evidencia"]["falla"])
        self.assertIn("lista_sha256", c["evidencia"]["falla"])

    def test_untouched_runs_still_pass_their_checks(self):
        self.assertCheckPasses(self.ctr, "K6", "g01-A-r1", "histograma reconstruido")
        self.assertCheckPasses(self.ctr, "K6", "g06-A-r3", "histograma reconstruido")
        self.assertCheckPasses(self.ctr, "K6", "g06-B-r2", "IDs únicos")


class BlockingRecordDefectTests(Base):
    """Defects that live in the records, so the files remain self-consistent."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        def tweak(records, entries):
            set_status(records["g06-B-r1"][33], "ABORTED")                 # K6 ABORTED
            records["g06-B-r1"][5]["lz_a0"] = 1e-3                           # K9 (c): 1 mm of anomaly_0 in B
            records["g06-B-r2"][8]["prim_fnv"] ^= 1                          # K9 (a)
            records["g06-B-r2"][17]["lz_rock"] = 2e-6                        # K9 (c): 2 micrometres of rock
            records["g06-B-r3"][9]["rng0"] ^= 1                              # K9 (a)
            rec = records["g01-A-r2"][11]                                    # K6 in_rint
            set_det(rec, *INSIDE)
            rec["in_rint"] = 0
            records["g06-A-r3"][40]["max_lat"] = 3e-3                        # lateral displacement: not a defect
            base, rep = records["g01-A-r1"], records["g01-A-r1-rep"]
            rep[20]["rng1"] ^= 7                                             # K7 muon log
            set_det(base[30], 90, 90)
            set_status(rep[30], "DECAY")                                     # K7 histogram and meta

        cls.fx = F.build_attempt(os.path.join(cls.tmp, "records"), n=N, tweak=tweak)
        cls.rc, cls.out, cls.stdout, cls.stderr = analyze(cls.fx, cls.tmp)
        cls.ctr = load(os.path.join(cls.out, "contratos.json"))

    def test_blocked(self):
        self.assertEqual(self.rc, 3)
        self.assertEqual(self.ctr["estado"], A.STAMP)

    def test_aborted_row(self):
        self.assertCheckFails(self.ctr, "K6", "g06-B-r1", "ninguna fila ABORTED")

    def test_in_rint_against_the_offline_mask(self):
        c = self.assertCheckFails(self.ctr, "K6", "g01-A-r2", "in_rint del binario")
        self.assertEqual(c["evidencia"]["distintas"], 1)

    def test_a_large_lateral_displacement_is_reported_and_does_not_block(self):
        # PLAN C1: 3e-3 mm (three thousand times the old limit) is a diagnostic, not a defect
        c = checks_of(self.ctr, "K6", "g06-A-r3", "desplazamiento lateral")
        self.assertEqual(len(c), 1)
        self.assertTrue(c[0]["ok"])
        self.assertGreaterEqual(c[0]["evidencia"]["max_lat_mm"], 3e-3)

    def test_k9_a_primary_fingerprint_and_random_start(self):
        self.assertCheckFails(self.ctr, "K9", "g06-A-r2 vs g06-B-r2", "prim_fnv")
        self.assertCheckFails(self.ctr, "K9", "g06-A-r3 vs g06-B-r3", "rng0")
        self.assertCheckPasses(self.ctr, "K9", "g06-A-r1 vs g06-B-r1", "prim_fnv")

    def test_k9_c_anomaly0_and_rock_length_in_b(self):
        c = self.assertCheckFails(self.ctr, "K9", "g06-A-r1 vs g06-B-r1 (B)", "lz_a0")
        self.assertAlmostEqual(c["evidencia"]["max_lz_a0_m"], 1e-3)
        c = self.assertCheckFails(self.ctr, "K9", "g06-A-r2 vs g06-B-r2 (B)", "lz_rock")
        self.assertAlmostEqual(c["evidencia"]["max_lz_rock_m"], 2e-6)
        self.assertCheckPasses(self.ctr, "K9", "g06-A-r3 vs g06-B-r3 (B)", "lz_rock")

    def test_k7_broken_in_three_ways(self):
        self.assertCheckFails(self.ctr, "K7", "g01-A-r1-rep", "_muones.csv idéntico byte a byte")
        self.assertCheckFails(self.ctr, "K7", "g01-A-r1-rep", "hist.csv idéntico")
        c = self.assertCheckFails(self.ctr, "K7", "g01-A-r1-rep", "meta.json igual salvo")
        self.assertIn("detected_in_bins", c["evidencia"]["campos_distintos"])
        self.assertEqual(self.ctr["contratos"]["K7"]["veredicto"], "FALLA")

    def test_diagnostic_only_writes_stamped_tables_and_exits_3(self):
        rc, out, stdout, _ = analyze(self.fx, self.tmp, extra=("--diagnostic-only",))
        self.assertEqual(rc, 3)
        for name in ("tabla-corridas.csv", "transiciones.csv", "matriz-clases.csv", "cambios-bin.csv",
                     "muones-discordantes.csv", "cruces-union.csv"):
            with open(os.path.join(out, name), encoding="utf-8") as fh:
                self.assertEqual(fh.readline().strip(), "# " + A.STAMP, name)
        res = load(os.path.join(out, "resultados-etapa1.json"))
        self.assertEqual(res["estado"], A.STAMP)
        self.assertTrue(os.path.isfile(os.path.join(out, "BLOQUEADO.txt")))


class NonBlockingTests(Base):
    """Failures that do not block: they are reported and (K8, K9 b) hold back the attribution."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fx = F.build_attempt(os.path.join(cls.tmp, "base"), n=N, lateral_columns=True)

    def test_a_warnings_and_mean_ekin_do_not_block(self):
        root = self.clone(self.fx)
        e = next(x for x in self.fx["entries"] if x["run_id"] == "g06-A-r3")
        d = os.path.join(root, e["salida"])
        edit(os.path.join(d, "run.log"), lambda x: x.replace(b"*** G4Exception : GeomVol1002", b"*** note : none"))
        F.write_sums(d)
        e = next(x for x in self.fx["entries"] if x["run_id"] == "g06-A-r1")
        d = os.path.join(root, e["salida"])
        edit_json(os.path.join(d, "geo-train-06_view0.meta.json"), lambda m: m.update(
            mean_ekin_detected_gev=m["mean_ekin_detected_gev"] * 1.001))
        F.write_sums(d)
        rc, out, stdout, _ = analyze(self.fx, self.tmp, attempt=root)
        self.assertEqual(rc, 0, stdout)
        ctr = load(os.path.join(out, "contratos.json"))
        self.assertEqual(ctr["bloqueantes_fallidos"], [])
        self.assertEqual(len(ctr["no_bloqueantes_fallidos"]), 2)
        self.assertTrue(any("avisos de A" in x for x in ctr["no_bloqueantes_fallidos"]))
        self.assertTrue(any("energía cinética media" in x for x in ctr["no_bloqueantes_fallidos"]))
        self.assertEqual(ctr["contratos"]["K6"]["veredicto"], "REQUIERE_EXPLICACION")
        self.assertFalse(ctr["atribucion_al_solape_bloqueada"])

    def test_aa_on_different_nodes_blocks(self):
        root = self.clone(self.fx)
        with open(os.path.join(root, "estado", "g01-A-r1-rep.status"), "wb") as fh:
            fh.write(b"g01-A-r1-rep\tOK\t0\t\tguane02\t30600\n")
        rc, out, _, _ = analyze(self.fx, self.tmp, attempt=root)
        self.assertEqual(rc, 3)
        self.assertCheckFails(load(os.path.join(out, "contratos.json")), "K7", "g01-A-r1-rep", "mismo nodo")

    def test_unknown_node_is_reported_but_does_not_block(self):
        root = self.clone(self.fx)
        for rid in ("g01-A-r1", "g01-A-r1-rep"):
            with open(os.path.join(root, "estado", rid + ".status"), "wb") as fh:
                fh.write(("%s\tOK\t0\t\n" % rid).encode())
        rc, out, _, _ = analyze(self.fx, self.tmp, attempt=root)
        self.assertEqual(rc, 0)
        self.assertEqual(len(load(os.path.join(out, "contratos.json"))["no_verificables"]), 1)

    def test_k8_control_difference_needs_an_explanation(self):
        def tweak(records, entries):
            records["g01-B-r2"][10]["rng1"] ^= 99                       # the control diverges in one muon
            set_det(records["g01-B-r1"][15], *INSIDE)                     # ...and one muon enters R_int in B
            set_status(records["g01-A-r1"][15], "DECAY")
            set_status(records["g01-A-r1-rep"][15], "DECAY")

        fx = F.build_attempt(os.path.join(self.tmp, "k8"), n=N, tweak=tweak)
        rc, out, stdout, _ = analyze(fx, self.tmp)
        self.assertEqual(rc, 0)                                        # not blocking...
        ctr = load(os.path.join(out, "contratos.json"))
        self.assertEqual(ctr["contratos"]["K8"]["veredicto"], "REQUIERE_EXPLICACION")
        self.assertTrue(ctr["atribucion_al_solape_bloqueada"])         # ...but the attribution is held back
        self.assertCheckFails(ctr, "K8", "g01-A-r2 vs g01-B-r2", "_muones.csv idéntico salvo la columna variant")
        self.assertCheckFails(ctr, "K8", "g01-A-r1 vs g01-B-r1", "hist.csv idéntico entre A y B")
        self.assertCheckPasses(ctr, "K8", "g01-A-r3 vs g01-B-r3", "_muones.csv")
        self.assertIn("ATENCIÓN", stdout)
        disc = read_rows(os.path.join(out, "muones-discordantes.csv"))
        control = sorted((r["replica"], r["id"], r["clase_K9b"]) for r in disc if r["caso"] == "geo-train-01")
        self.assertEqual(control, [("1", "15", "control_sin_region"), ("2", "10", "control_sin_region")])
        res = load(os.path.join(out, "resultados-etapa1.json"))
        self.assertEqual(res["casos"]["geo-train-01"]["agrupado"]["n01"], 1)
        self.assertTrue(res["atribucion_al_solape_bloqueada"])

    def test_k9b_classes_and_the_micron_tolerance(self):
        def tweak(records, entries):
            records["g06-B-r1"][5]["lz_a0"] = 1e-9          # navigation-boundary residue: allowed
            records["g06-B-r1"][17]["lz_rock"] = 5e-7       # below one micrometre: allowed
            a, b = records["g06-A-r1"][90], records["g06-B-r1"][90]
            set_det(a, *OUTSIDE)
            set_det(b, OUTSIDE[0] + 3, OUTSIDE[1])          # moved 3 bins, touching nothing near Z
            diverge(a, b)
            a, b = records["g06-A-r2"][91], records["g06-B-r2"][91]
            set_status(a, "DECAY")
            set_status(b, "DECAY")
            diverge(a, b)
            a["le"] = b["le"] = 6.0                          # touched only the dense body
            a, b = records["g06-A-r3"][100], records["g06-B-r3"][100]
            diverge(a, b)
            b["n_graze"] = 2                                 # grazed Z in B
            a, b = records["g06-A-r3"][101], records["g06-B-r3"][101]
            diverge(a, b)
            a["deep"] = b["deep"] = True                     # crossed Z: the divergence is expected
            a["lz"] = b["lz"] = 3.0
            a["lz_a1"], a["lz_a0"], b["lz_a1"] = 1.8, 1.2, 3.0

        fx = F.build_attempt(os.path.join(self.tmp, "k9b"), n=N, tweak=tweak)
        rc, out, stdout, _ = analyze(fx, self.tmp)
        self.assertEqual(rc, 0, stdout)
        ctr = load(os.path.join(out, "contratos.json"))
        self.assertCheckPasses(ctr, "K9", "g06-A-r1 vs g06-B-r1 (B)", "lz_a0")
        rows = read_rows(os.path.join(out, "muones-discordantes.csv"))
        cls = {(r["replica"], r["id"]): r["clase_K9b"] for r in rows}
        self.assertEqual(cls[("1", "90")], "otra")
        self.assertEqual(cls[("2", "91")], "solo_elipsoide")
        self.assertEqual(cls[("3", "100")], "roce")
        self.assertEqual(cls[("3", "101")], "cruza_Z")
        self.assertTrue(ctr["atribucion_al_solape_bloqueada"])
        self.assertEqual(ctr["contratos"]["K9"]["veredicto"], "REQUIERE_EXPLICACION")
        self.assertEqual(sum(x["sin_explicar"] for x in ctr["discordancias_no_cruzan_K9b"]), 2)


class ManifestTests(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fx = F.build_attempt(os.path.join(cls.tmp, "base"), n=N)

    def rewrite(self, root, fn):
        p = os.path.join(root, "manifiesto-corridas.tsv")
        with open(p, "rb") as fh:
            lines = fh.read().decode().splitlines()
        with open(p, "wb") as fh:
            fh.write(("\n".join(fn(lines)) + "\n").encode())

    def test_missing_repeat_row(self):
        root = self.clone(self.fx)
        self.rewrite(root, lambda ls: [x for x in ls if "g01-A-r1-rep" not in x])
        rc, out, _, _ = analyze(self.fx, self.tmp, attempt=root)
        self.assertEqual(rc, 3)
        ctr = load(os.path.join(out, "contratos.json"))
        self.assertCheckFails(ctr, "K0", "manifiesto", "13 corridas")
        self.assertCheckFails(ctr, "K0", "manifiesto", "repetición A/A de geo-train-01")

    def test_wrong_transport_seed(self):
        root = self.clone(self.fx)

        def fn(lines):
            cols = lines[1].split("\t")
            cols[6] = "12345"
            lines[1] = "\t".join(cols)
            return lines

        self.rewrite(root, fn)
        rc, out, _, _ = analyze(self.fx, self.tmp, attempt=root)
        self.assertEqual(rc, 3)
        self.assertCheckFails(load(os.path.join(out, "contratos.json")), "K0", "manifiesto", "s_tr = s_src - 1e9")

    def test_repeat_with_another_list(self):
        root = self.clone(self.fx)

        def fn(lines):
            for i, ln in enumerate(lines):
                if "g01-A-r1-rep" in ln:
                    cols = ln.split("\t")
                    cols[9] = "f" * 64
                    lines[i] = "\t".join(cols)
            return lines

        self.rewrite(root, fn)
        rc, out, _, _ = analyze(self.fx, self.tmp, attempt=root)
        self.assertEqual(rc, 3)
        self.assertCheckFails(load(os.path.join(out, "contratos.json")), "K0", "manifiesto", "misma lista")

    def test_unexpected_columns(self):
        """An unusable manifest is an unusable input: exit code 2 (a SystemExit with text would exit 1) and the
        output folder is not even created."""
        root = self.clone(self.fx)
        self.rewrite(root, lambda ls: [ls[0].replace("run_id", "id")] + ls[1:])
        rc, out, _, err = analyze(self.fx, self.tmp, attempt=root)
        self.assertEqual(rc, 2)
        self.assertIn("columnas del manifiesto", err)
        self.assertFalse(os.path.exists(out), "a bad manifest must leave no empty output folder")

    def test_other_unusable_manifests_exit_2_without_an_output_folder(self):
        cases = {"empty": lambda ls: [], "short row": lambda ls: ls[:2] + ["g06-B-r1\tgeo-train-06"] + ls[3:],
                 "non-numeric N": lambda ls: [ls[0]] + [ln.replace("\t200\t", "\tmuchos\t") for ln in ls[1:]]}
        for label, fn in cases.items():
            root = self.clone(self.fx)
            self.rewrite(root, fn)
            rc, out, _, err = analyze(self.fx, self.tmp, attempt=root)
            self.assertEqual(rc, 2, label)
            self.assertIn("manifiesto", err, label)
            self.assertFalse(os.path.exists(out), label)
        # a missing file is also code 2 and creates nothing
        rc, out, _, err = analyze(self.fx, self.tmp, attempt=os.path.join(self.tmp, "no-existe"))
        self.assertEqual(rc, 2)
        self.assertFalse(os.path.exists(out))

    def test_technical_layout_with_replica_zero(self):
        """The T-F rehearsal uses replica 0 only: 4 x 1 + 1 = 5 runs."""
        fx = F.build_attempt(os.path.join(self.tmp, "tec"), n=N, replicas=(0,))
        rc, out, stdout, err = analyze(fx, self.tmp, extra=("--replicas", "0"))
        self.assertEqual(rc, 0, stdout + err)
        ctr = load(os.path.join(out, "contratos.json"))
        self.assertEqual(len(ctr["contratos"]["K6"]["ambitos"]), 5)
        res = load(os.path.join(out, "resultados-etapa1.json"))
        self.assertEqual(res["casos"]["geo-train-06"]["agrupado"]["N"], N)
        rows = read_rows(os.path.join(out, "tabla-corridas.csv"))
        self.assertEqual(len(rows), 5)


class UnusableRoiInputTests(Base):
    """An unusable ROI input is an unusable input, like an unusable manifest: exit code 2, one line on stderr and no
    output folder. It used to be a traceback (exit code 1) and an empty --out folder."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fx = F.build_attempt(os.path.join(cls.tmp, "roi-inputs"), n=N)

    def run_with(self, mask=None, frozen=None, npz=None):
        """A.main on the pristine attempt with the named ROI files replaced; (exit code, --out, stderr)."""
        _counter[0] += 1
        out = os.path.join(self.tmp, "out%03d" % _counter[0])
        m = self.fx["mask"]
        argv = ["--attempt", self.fx["attempt"], "--roi-mask", mask or m["text"], "--roi-frozen", frozen or m["frozen"],
                "--roi-npz", npz or m["npz"], "--out", out, "--crossing-reference", F.CROSSING_REFERENCE,
                "--no-figures", "--test-mask"]
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = A.main(argv)
        return rc, out, err.getvalue()

    def write(self, name, data):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    def test_the_pristine_inputs_pass(self):
        """Positive control: the helper itself is not what makes the other tests exit 2."""
        rc, out, err = self.run_with()
        self.assertEqual(rc, 0, err)
        self.assertTrue(os.path.isfile(os.path.join(out, "contratos.json")))

    def test_a_missing_roi_npz_is_exit_2_and_creates_no_output_folder(self):
        missing = os.path.join(self.tmp, "no-existe.npz")
        rc, out, err = self.run_with(npz=missing)
        self.assertEqual(rc, 2)
        self.assertIn(missing, err)
        self.assertNotIn("Traceback", err)
        self.assertFalse(os.path.exists(out), "an unusable input must leave no output folder")

    def test_an_invalid_roi_frozen_json_is_exit_2_and_creates_no_output_folder(self):
        with open(self.fx["mask"]["frozen"], "rb") as fh:
            good = fh.read()
        cases = {"truncated": good[:len(good) // 2], "not-utf8": b"\xff\xfe\x00{", "empty": b"",
                 "a-list": b"[1, 2, 3]", "without-the-keys-k0-reads": b'{"formato": "x"}',
                 "a-key-that-is-not-an-object": b'{"texto": "x", "hash_contenido": {}, "origen": {}}'}
        for label, data in cases.items():
            rc, out, err = self.run_with(frozen=self.write("frozen-%s.json" % label, data))
            self.assertEqual(rc, 2, label)
            self.assertIn("--roi-frozen inutilizable", err, label)
            self.assertFalse(os.path.exists(out), label)

    def test_a_roi_path_that_cannot_be_opened_is_exit_2_and_creates_no_output_folder(self):
        folder = os.path.join(self.tmp, "una-carpeta")
        os.makedirs(folder)
        for key in ("mask", "frozen", "npz"):
            rc, out, err = self.run_with(**{key: folder})
            self.assertEqual(rc, 2, key)
            self.assertIn("inutilizable", err, key)
            self.assertFalse(os.path.exists(out), key)

    def test_an_unusable_roi_npz_is_exit_2_and_creates_no_output_folder(self):
        """An npz that cannot be decoded (truncated, empty, junk, a corrupt member, a bare .npy) or that lacks one of
        the five mask arrays, or holds it with another shape or type, used to fail later, inside load_roi: a traceback
        (BadZipFile, EOFError, ValueError, zlib.error, IndexError...), exit code 1 and an empty --out folder."""
        with open(self.fx["mask"]["npz"], "rb") as fh:
            good = fh.read()
        full = np.zeros((C.N_EL, C.N_AZ), dtype=bool)
        masks = dict(R_int=full, R_cond=full, R_body=full, R_rock=full, R_sky=full)

        def savez(**arrays):
            buf = io.BytesIO()
            np.savez_compressed(buf, **arrays)
            return buf.getvalue()

        def npy(array):
            buf = io.BytesIO()
            np.save(buf, array)
            return buf.getvalue()

        # the central directory stays intact; the compressed bytes of the first member are inverted
        corrupt = bytearray(good)
        info = zipfile.ZipFile(io.BytesIO(good)).infolist()[0]
        name_len, extra_len = struct.unpack("<HH", good[info.header_offset + 26:info.header_offset + 30])
        start = info.header_offset + 30 + name_len + extra_len
        for i in range(start, start + info.compress_size):
            corrupt[i] ^= 0xFF
        cases = {"truncated": good[:len(good) // 2], "empty": b"", "junk-bytes": b"this is not an npz at all",
                 "corrupt-member": bytes(corrupt), "a-bare-npy": npy(full),
                 "without-the-arrays": savez(other=np.zeros((3, 3))),
                 "one-array-missing": savez(**{k: v for k, v in masks.items() if k != "R_sky"}),
                 "wrong-shape": savez(**dict(masks, R_sky=np.zeros((4, 4), dtype=bool))),
                 "text-instead-of-a-mask": savez(**dict(masks, R_sky=np.full((C.N_EL, C.N_AZ), "x")))}
        for label, data in cases.items():
            rc, out, err = self.run_with(npz=self.write("roi-%s.npz" % label, data))
            self.assertEqual(rc, 2, label)
            self.assertIn("--roi-npz inutilizable", err, label)
            self.assertNotIn("Traceback", err, label)
            self.assertFalse(os.path.exists(out), label)

    def test_any_decoding_error_of_the_npz_is_exit_2(self):
        """Damaged archives also make numpy and zipfile raise tokenize.TokenError (the .npy header parser),
        NotImplementedError (zip version or compression method), RuntimeError (the encryption flag) or MemoryError
        (a forged huge shape). Which byte triggers which one depends on the versions, so np.load is replaced instead
        of corrupting a fixed offset."""
        for exc in (tokenize.TokenError("bad header"), NotImplementedError("compression method"),
                    RuntimeError("encrypted"), MemoryError()):
            label = type(exc).__name__
            with mock.patch.object(A.np, "load", side_effect=exc):
                rc, out, err = self.run_with()
            self.assertEqual(rc, 2, label)
            self.assertIn("--roi-npz inutilizable", err, label)
            self.assertNotIn("Traceback", err, label)
            self.assertFalse(os.path.exists(out), label)


class ReplicasOptionTests(Base):
    """--replicas is a comma-separated list of integers; anything else is a usage error (argparse, exit code 2)
    before any file is read. It used to be a bare int() after the parser: a traceback and exit code 1."""

    ARGS = ["--attempt", "a", "--roi-mask", "m", "--roi-frozen", "f"]

    def test_the_default_and_the_good_values_parse(self):
        self.assertEqual(A.parse_replicas("1,2,3"), (1, 2, 3))
        self.assertEqual(A.parse_replicas("0"), (0,))
        args = A.build_parser().parse_args(self.ARGS + ["--out", "o"])
        self.assertEqual(args.replicas, (1, 2, 3))          # the default string goes through the same parser

    def test_a_bad_value_is_a_usage_error_with_exit_code_2_and_creates_nothing(self):
        for bad in ("a,b", "1,,2", "", "1;2", "1,2,"):
            out = os.path.join(self.tmp, "out-replicas")
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as cm:
                    A.main(self.ARGS + ["--out", out, "--replicas", bad])
            self.assertEqual(cm.exception.code, 2, repr(bad))
            self.assertIn("--replicas", err.getvalue(), repr(bad))
            self.assertFalse(os.path.exists(out), repr(bad))


class ListLocationTests(Base):
    def test_lists_dir_is_searched_by_file_name_whatever_the_manifest_separators(self):
        name = "geo-train-06_v0_r1_s2026092901.primarios.csv"
        d = os.path.join(self.tmp, "listas")
        os.makedirs(d)
        open(os.path.join(d, name), "wb").close()
        for foreign in ("C:\\Users\\x\\listas\\" + name, "/home/u/oe1-trabajo/etapa1/listas/prep-01/A2/" + name, name):
            self.assertEqual(A.locate_list({"lista": foreign}, d), os.path.join(d, name), foreign)
        self.assertIsNone(A.locate_list({"lista": "/nowhere/" + name}, os.path.join(self.tmp, "empty")))
        self.assertIsNone(A.locate_list({"lista": "/nowhere/" + name}, None))


class LateralDisplacementTests(Base):
    """PLAN correction C1 (2026-09-30): WentzelVI displaces the track inside a step, so max_lat_mm is a
    diagnostic and sum_lat_z_mm, when the binary records it, is the SCALE of the error of the length in Z (not a
    bound)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        def big(records, entries):
            records["g06-B-r1"][3]["max_lat"] = 33443.568                     # metres: still not a defect

        cls.fx_new = F.build_attempt(os.path.join(cls.tmp, "lat-new"), n=N, lateral_columns=True, tweak=big)
        cls.fx_old = F.build_attempt(os.path.join(cls.tmp, "lat-old"), n=N)

    def k6(self, fx, run_id, edit=None):
        root = self.clone(fx)
        entries = A.read_manifest(os.path.join(root, "manifiesto-corridas.tsv"))
        entry = next(e for e in entries if e["run_id"] == run_id)
        stem = os.path.join(root, entry["salida"], "%s_view0" % entry["caso"])
        if edit:
            edit(stem)
        contracts = A.Contracts()
        roi = A.load_roi(fx["mask"]["text"], fx["mask"]["frozen"], fx["mask"]["npz"], True, contracts)
        A.k6_checks(A.load_run(entry, root), roi, contracts)
        return [c for c in contracts.checks if c["contrato"] == "K6"]

    @staticmethod
    def pick(checks, text):
        found = [c for c in checks if text in c["descripcion"]]
        assert len(found) == 1, (text, len(found))
        return found[0]

    @staticmethod
    def edit_row(stem, row, fn):
        """fn(fields) edits the fields of muon row `row` (0-based) of the log; the meta is refreshed."""
        path = stem + "_muones.csv"
        with open(path, "rb") as fh:
            lines = fh.read().split(b"\n")
        fields = lines[1 + row].split(b",")
        fn(fields)
        lines[1 + row] = b",".join(fields)
        with open(path, "wb") as fh:
            fh.write(b"\n".join(lines))
        F.refresh_muon_log_meta(stem)

    def test_a_displacement_of_metres_is_only_reported(self):
        checks = self.k6(self.fx_new, "g06-B-r1")
        lat = self.pick(checks, "desplazamiento lateral")
        self.assertTrue(lat["ok"])
        self.assertAlmostEqual(lat["evidencia"]["max_lat_mm"], 33443.568)
        self.assertTrue(lat["bloqueante"])
        self.assertTrue(self.pick(checks, "escala del error")["ok"])
        self.assertEqual([c["descripcion"] for c in checks if c["ok"] is False], [])

    def test_a_binary_without_the_bound_columns_is_not_verifiable_and_does_not_block(self):
        checks = self.k6(self.fx_old, "g06-B-r1")
        z = self.pick(checks, "escala del error")
        self.assertIsNone(z["ok"])
        self.assertFalse(z["bloqueante"])
        self.assertEqual(z["evidencia"]["columnas_ausentes"], list(C.MUON_OPTIONAL_COLUMNS))
        self.assertEqual([c["descripcion"] for c in checks if c["ok"] is False], [])
        rc, out, stdout, err = analyze(self.fx_old, self.tmp)
        self.assertEqual(rc, 0, stdout + err)
        ctr = load(os.path.join(out, "contratos.json"))
        self.assertEqual(ctr["estado"], "OK")
        self.assertTrue(ctr["no_verificables"])
        self.assertEqual(ctr["contratos"]["K6"]["veredicto"], "OK_CON_AVISOS")
        row = read_rows(os.path.join(out, "tabla-corridas.csv"))[0]
        self.assertEqual(row["max_lat_z_mm"], "NA")             # the column is absent: no figure, not a zero

    def test_the_bound_must_be_consistent(self):
        def sum_below_max(stem):
            def fn(f):
                f[47], f[48] = b"0.5", b"0.1"
            self.edit_row(stem, 5, fn)

        def negative(stem):
            def fn(f):
                f[47], f[48] = b"0.1", b"-0.2"
            self.edit_row(stem, 5, fn)

        def only_one_of_the_two(stem):
            def fn(f):
                f[48] = b""
            self.edit_row(stem, 5, fn)

        for edit in (sum_below_max, negative, only_one_of_the_two):
            z = self.pick(self.k6(self.fx_new, "g06-B-r1", edit), "escala del error")
            self.assertIs(z["ok"], False, edit.__name__)

    def test_the_aggregates_of_the_meta_must_match_the_columns_and_the_region(self):
        """The corrected binary writes counts.max_lat_z_mm / max_sum_lat_z_mm: they are the maxima of the two
        columns, every row of a run with a region has both values, max_lat_z_mm never exceeds max_lat_mm, and
        a run without a region has neither."""
        def meta_max_differs(stem):
            def fn(m):
                m["oe1_etapa1"]["counts"]["max_lat_z_mm"] += 1.0
            edit_json(stem + ".meta.json", fn)
            F.write_sums(os.path.dirname(stem))

        def meta_sum_differs(stem):
            def fn(m):
                m["oe1_etapa1"]["counts"]["max_sum_lat_z_mm"] = None
            edit_json(stem + ".meta.json", fn)
            F.write_sums(os.path.dirname(stem))

        def above_the_all_steps_maximum(stem):
            def fn(f):
                f[46], f[47], f[48] = b"0.001", b"0.5", b"0.6"       # near-Z displacement > all-steps maximum
            self.edit_row(stem, 4, fn)

        def a_row_without_values(stem):
            def fn(f):
                f[47], f[48] = b"", b""
            self.edit_row(stem, 4, fn)

        for edit in (meta_max_differs, meta_sum_differs, above_the_all_steps_maximum, a_row_without_values):
            z = self.pick(self.k6(self.fx_new, "g06-B-r1", edit), "escala del error")
            self.assertIs(z["ok"], False, edit.__name__)
            self.assertTrue(z["bloqueante"], edit.__name__)

        def values_without_a_region(stem):
            def fn(f):
                f[47], f[48] = b"0.1", b"0.2"
            self.edit_row(stem, 3, fn)

        z = self.pick(self.k6(self.fx_new, "g01-B-r1", values_without_a_region), "escala del error")
        self.assertIs(z["ok"], False)
        # the pristine runs of both cases pass (the meta declares the aggregates: null for geo-train-01)
        for run_id in ("g06-B-r1", "g01-B-r1"):
            z = self.pick(self.k6(self.fx_new, run_id), "escala del error")
            self.assertIs(z["ok"], True, run_id)
            self.assertIs(z["evidencia"]["maximos_iguales_al_meta"], True, run_id)

    def test_a_meta_that_declares_the_aggregates_makes_the_columns_mandatory(self):
        def drop_the_two_columns(stem):
            path = stem + "_muones.csv"
            with open(path, "rb") as fh:
                lines = fh.read().split(b"\n")
            kept = [b",".join(ln.split(b",")[:-2]) if ln else ln for ln in lines]
            with open(path, "wb") as fh:
                fh.write(b"\n".join(kept))
            F.refresh_muon_log_meta(stem)

        checks = self.k6(self.fx_new, "g06-B-r1", drop_the_two_columns)
        z = self.pick(checks, "escala del error")
        self.assertIs(z["ok"], False)
        self.assertTrue(z["bloqueante"])
        self.assertEqual(z["evidencia"]["columnas_ausentes"], list(C.MUON_OPTIONAL_COLUMNS))

    def test_max_lat_must_be_present_finite_and_equal_to_the_meta(self):
        def blank(stem):
            self.edit_row(stem, 4, lambda f: f.__setitem__(46, b""))          # a row with steps

        def meta_differs(stem):
            def fn(m):
                m["oe1_etapa1"]["counts"]["max_lat_mm"] += 1.0
            edit_json(stem + ".meta.json", fn)
            F.write_sums(os.path.dirname(stem))

        def negative(stem):
            self.edit_row(stem, 4, lambda f: f.__setitem__(46, b"-1.5"))

        for edit in (blank, meta_differs, negative):
            lat = self.pick(self.k6(self.fx_new, "g06-B-r1", edit), "desplazamiento lateral")
            self.assertIs(lat["ok"], False, edit.__name__)


class WorkerThreadTests(Base):
    """K6 requires exactly one worker thread; the number the run manager gives it is not part of the
    contract (the master thread is -1), so an unexpected numbering must not block a valid run."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fx = F.build_attempt(os.path.join(cls.tmp, "workers"), n=N)

    def provenance_check(self, ids):
        root = self.clone(self.fx)
        entries = A.read_manifest(os.path.join(root, "manifiesto-corridas.tsv"))
        entry = next(e for e in entries if e["run_id"] == "g06-B-r1")
        d = os.path.join(root, entry["salida"])
        edit_json(os.path.join(d, "%s_view0.meta.json" % entry["caso"]),
                  lambda m: m["oe1_etapa1"]["runtime"].update(worker_thread_ids=ids))
        F.write_sums(d)
        contracts = A.Contracts()
        roi = A.load_roi(self.fx["mask"]["text"], self.fx["mask"]["frozen"], self.fx["mask"]["npz"], True, contracts)
        A.k6_checks(A.load_run(entry, root), roi, contracts)
        found = [c for c in contracts.checks if c["contrato"] == "K6" and c["descripcion"].startswith("procedencia")]
        self.assertEqual(len(found), 1)
        return found[0]

    def test_any_single_worker_number_is_accepted(self):
        for ids in ([0], [1], [7]):
            c = self.provenance_check(ids)
            self.assertTrue(c["ok"], (ids, c["evidencia"]))
            self.assertEqual(c["evidencia"]["worker_thread_ids"], ids)

    def test_no_worker_several_workers_or_the_master_are_rejected(self):
        for ids in ([], [0, 1], [-1], None):
            c = self.provenance_check(ids)
            self.assertFalse(c["ok"], ids)
            self.assertIn("trabajadores", c["evidencia"]["falla"])


class FingerprintTests(Base):
    """W1: K7, K8 and K9 (a) compare columns between runs, so they would pass on nothing if those columns were
    empty, constant or simply not the listed muons. K6 requires real values (non-zero, N different) and
    recomputes prim_fnv from the rows of the located list for every id; K9 (a) is blocking in the control too."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        def tweak(records, entries):
            # 1) duplicate rng0 inside one run, and a zero rng0 in another
            records["g06-B-r2"][8]["rng0"] = records["g06-B-r2"][7]["rng0"]
            records["g01-B-r3"][3]["rng0"] = 0
            # 2) a whole run with an empty (all-zero) rng0 column, identical in A and B: K9 (a) would pass on it
            for rec in records["g06-A-r3"] + records["g06-B-r3"]:
                rec["rng0"] = 0
            # 3) A and B both transported another muon than the listed one (same values in both, still unique and
            #    non-zero): only the recomputation from the list can see it
            for rec in records["g06-A-r1"] + records["g06-B-r1"]:
                rec["prim_fnv"] ^= 0xABCDEF
            # 4) the control: A and B start from different primaries / random states in one muon each
            records["g01-B-r2"][12]["prim_fnv"] ^= 1
            records["g01-B-r1"][14]["rng0"] ^= 1          # the A/A repeat (g01-A-r1-rep) keeps equal to g01-A-r1

        cls.fx = F.build_attempt(os.path.join(cls.tmp, "fingerprints"), n=N, tweak=tweak)
        cls.rc, cls.out, cls.stdout, cls.stderr = analyze(cls.fx, cls.tmp)
        cls.ctr = load(os.path.join(cls.out, "contratos.json"))

    def test_blocked(self):
        self.assertEqual(self.rc, 3)
        self.assertEqual(self.ctr["estado"], A.STAMP)

    def test_duplicate_and_zero_values_are_not_real_values(self):
        c = self.assertCheckFails(self.ctr, "K6", "g06-B-r2", "prim_fnv y rng0")
        self.assertEqual(c["evidencia"]["rng0"]["valores_distintos"], N - 1)
        self.assertEqual(c["evidencia"]["prim_fnv"]["valores_distintos"], N)
        self.assertTrue(c["bloqueante"])
        c = self.assertCheckFails(self.ctr, "K6", "g01-B-r3", "prim_fnv y rng0")
        self.assertEqual(c["evidencia"]["rng0"]["ceros"], 1)

    def test_an_empty_column_shared_by_a_and_b_no_longer_passes_k9_a(self):
        c = self.assertCheckFails(self.ctr, "K6", "g06-A-r3", "prim_fnv y rng0")
        self.assertEqual(c["evidencia"]["rng0"]["ceros"], N)
        k9 = self.assertCheckFails(self.ctr, "K9", "g06-A-r3 vs g06-B-r3", "rng0")
        self.assertTrue(k9["bloqueante"])
        self.assertEqual(k9["evidencia"]["rng0_distintos"], 0)                # A and B ARE equal...
        self.assertFalse(k9["evidencia"]["valores_reales_y_distintos_en_A_y_B"])   # ...on nothing
        self.assertCheckPasses(self.ctr, "K9", "g06-A-r3 vs g06-B-r3", "prim_fnv")

    def test_prim_fnv_is_recomputed_from_the_list_for_every_id(self):
        for run_id in ("g06-A-r1", "g06-B-r1"):
            c = self.assertCheckFails(self.ctr, "K6", run_id, "prim_fnv recalculado")
            self.assertTrue(c["bloqueante"])
            self.assertEqual(c["evidencia"]["filas_que_difieren"], N)
            self.assertEqual(len(c["evidencia"]["primeras"]), 5)
            self.assertNotEqual(c["evidencia"]["primeras"][0]["columna"], c["evidencia"]["primeras"][0]["recalculado"])
        # A equals B (K9 a passes) and the columns are real: the recomputation is what catches it
        self.assertCheckPasses(self.ctr, "K9", "g06-A-r1 vs g06-B-r1", "prim_fnv")
        self.assertCheckPasses(self.ctr, "K6", "g06-A-r1", "prim_fnv y rng0")
        for run_id in ("g06-A-r2", "g06-B-r2", "g01-A-r1", "g01-A-r3"):
            self.assertCheckPasses(self.ctr, "K6", run_id, "prim_fnv recalculado")

    def test_k9_a_is_blocking_in_the_control_too(self):
        c = self.assertCheckFails(self.ctr, "K9", "g01-A-r2 vs g01-B-r2 (control)", "prim_fnv")
        self.assertTrue(c["bloqueante"])
        self.assertEqual(c["evidencia"]["prim_fnv_distintos"], 1)
        c = self.assertCheckFails(self.ctr, "K9", "g01-A-r1 vs g01-B-r1 (control)", "rng0")
        self.assertTrue(c["bloqueante"])
        self.assertEqual(c["evidencia"]["rng0_distintos"], 1)
        self.assertCheckPasses(self.ctr, "K9", "g01-A-r3 vs g01-B-r3 (control)", "prim_fnv")
        # K8 (non-blocking) sees the same difference; the blocking contract is K9
        self.assertEqual(self.ctr["contratos"]["K9"]["veredicto"], "FALLA")

    def test_k7_keeps_working_on_real_values(self):
        # the A/A base and repeat differ only where the tweak said so (rng0 of id 14 changed in both alike)
        self.assertCheckPasses(self.ctr, "K7", "g01-A-r1-rep", "_muones.csv")


class ListVerificationTests(Base):
    """W2: a located list must be the manifest's byte for byte (SHA-256); a list that is not is never used, and
    K0 (blocking) says so."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fx = F.build_attempt(os.path.join(cls.tmp, "lists"), n=N, lateral_columns=True)

    def lists_copy(self):
        dst = os.path.join(self.tmp, "listas-%03d" % len(os.listdir(self.tmp)))
        shutil.copytree(self.fx["lists_dir"], dst)
        return dst

    def test_pristine_has_one_passing_k0_check_per_distinct_list(self):
        rc, out, stdout, err = analyze(self.fx, self.tmp, extra=("--high-energy-gev", "0"))
        self.assertEqual(rc, 0, stdout + err)
        ctr = load(os.path.join(out, "contratos.json"))
        found = checks_of(ctr, "K0", "listas", "lista localizada == manifiesto")
        self.assertEqual(len(found), 6)                            # 2 cases x 3 replicas; the A/A repeat shares one
        self.assertTrue(all(c["ok"] is True and c["bloqueante"] for c in found))
        self.assertEqual(sorted(x for c in found for x in c["evidencia"]["corridas"]),
                         sorted(e["run_id"] for e in self.fx["entries"]))
        res = load(os.path.join(out, "resultados-etapa1.json"))
        self.assertEqual(len(res["entradas"]["listas"]), 6)

    def test_a_list_with_another_sha256_blocks_and_is_never_used(self):
        d = self.lists_copy()
        name = os.path.basename(self.fx["lists"][("geo-train-06", 1)]["path"])
        edit(os.path.join(d, name), lambda data: data.replace(b"\n", b"\r\n", 1))       # one byte more
        rc, out, stdout, err = analyze(self.fx, self.tmp, extra=("--lists-dir", d, "--diagnostic-only",
                                                                    "--high-energy-gev", "0"))
        self.assertEqual(rc, 3)
        ctr = load(os.path.join(out, "contratos.json"))
        c = self.assertCheckFails(ctr, "K0", "listas", name)
        self.assertTrue(c["bloqueante"])
        self.assertEqual(c["evidencia"]["estado"], "sha_distinto")
        self.assertEqual(c["evidencia"]["sha256_del_manifiesto"], [self.fx["lists"][("geo-train-06", 1)]["sha256"]])
        self.assertNotEqual(c["evidencia"]["sha256_del_archivo"], c["evidencia"]["sha256_del_manifiesto"][0])
        self.assertIsNone(c["evidencia"]["ruta_usada"])
        # the other five lists are fine
        self.assertEqual(len([x for x in checks_of(ctr, "K0", "listas", "lista localizada") if x["ok"] is False]), 1)
        # the runs of that list cannot be recomputed, and the recomputation names the reason
        for run_id in ("g06-A-r1", "g06-B-r1"):
            k6 = self.assertCheckFails(ctr, "K6", run_id, "prim_fnv recalculado")
            self.assertIn("SHA-256", k6["evidencia"]["error"])
        self.assertCheckPasses(ctr, "K6", "g06-A-r2", "prim_fnv recalculado")
        # ... and nothing was read from it: the straight-line cross-check is omitted for its runs and the
        # discordance tables carry no initial energy from it
        res = load(os.path.join(out, "resultados-etapa1.json"))
        for run_id in ("g06-A-r1", "g06-B-r1"):
            self.assertIn("SHA-256", res["verificacion_de_la_recta"][run_id]["omitida"])
        self.assertIn("tabla_2x2", res["verificacion_de_la_recta"]["g06-A-r2"])
        self.assertNotIn(name, res["entradas"]["listas"])                   # only verified lists are recorded
        self.assertEqual(len(res["entradas"]["listas"]), 5)

    def test_a_missing_list_blocks(self):
        d = self.lists_copy()
        name = os.path.basename(self.fx["lists"][("geo-train-01", 2)]["path"])
        os.remove(os.path.join(d, name))
        root = self.clone(self.fx)
        p = os.path.join(root, "manifiesto-corridas.tsv")
        with open(p, "rb") as fh:
            text = fh.read().decode()
        # the manifest no longer points at a file that exists (it was written on another system)
        text = text.replace(self.fx["lists"][("geo-train-01", 2)]["path"], "/no/existe/" + name)
        with open(p, "wb") as fh:
            fh.write(text.encode())
        rc, out, _, _ = analyze(self.fx, self.tmp, attempt=root, extra=("--lists-dir", d))
        self.assertEqual(rc, 3)
        ctr = load(os.path.join(out, "contratos.json"))
        c = self.assertCheckFails(ctr, "K0", "listas", name)
        self.assertEqual(c["evidencia"]["estado"], "no_encontrada")
        k6 = self.assertCheckFails(ctr, "K6", "g01-A-r2", "prim_fnv recalculado")
        self.assertIn("--lists-dir", k6["evidencia"]["error"])

    def test_list_cache_returns_only_a_list_that_matches_its_manifest_row(self):
        entry = next(e for e in self.fx["entries"] if e["run_id"] == "g06-A-r1")
        cache = A.ListCache(self.fx["lists_dir"])
        item = cache.get(entry)
        self.assertIsNotNone(item)
        self.assertEqual(item["sha256"], entry["lista_sha256"])
        self.assertIsNone(cache.problem(entry))
        wrong = dict(entry, lista_sha256="0" * 64)
        self.assertIsNone(cache.get(wrong))
        self.assertIn("SHA-256", cache.problem(wrong))
        self.assertEqual(cache.state(wrong)[0], "sha_distinto")
        gone = dict(entry, lista="/nowhere/none.primarios.csv")
        self.assertEqual(A.ListCache(os.path.join(self.tmp, "vacio")).state(gone)[0], "no_encontrada")
        self.assertIn("--lists-dir", A.ListCache(None).problem(gone))


class LateralScaleTests(Base):
    """B4: a K9 (c) violation (lz_a0 or lz_rock above 1 um in B) that stays within the lateral-displacement scale of
    its own steps near Z, lz_a0 + lz_rock <= 2 sqrt(2 R_max sum_lat_z) with R_max = 72.9^2 / 61.6 = 86.27 m, is a
    non-blocking class; any other violation blocks."""

    def build(self, name, tweak):
        return F.build_attempt(os.path.join(self.tmp, name), n=N, lateral_columns=True, tweak=tweak)

    def test_the_scale_is_two_sqrt_two_rmax_lat(self):
        r_max = 72.9 ** 2 / 61.6          # largest radius of curvature of the dense-body ellipsoid
        self.assertEqual(A.R_MAX_M, r_max)
        self.assertAlmostEqual(A.R_MAX_M, 86.27, places=2)
        got = A.lateral_length_scale_m(np.array([0.0, 1.0, 4.0, np.nan]))
        expect = [0.0, 2 * (2 * r_max * 1e-3) ** 0.5, 2 * (2 * r_max * 4e-3) ** 0.5, 0.0]
        for g, e in zip(got, expect):
            self.assertAlmostEqual(g, e, places=12)
        self.assertAlmostEqual(got[1], 0.83077, places=4)              # 1 mm of lateral displacement -> 0.83 m

    @unittest.skipUnless(os.path.isfile(os.path.join(F.CATALOG_DIR, "geo-train-06.json")), "catalog not found")
    def test_rmax_is_the_largest_radius_of_curvature_of_the_catalog_ellipsoid(self):
        """The largest radius of curvature of an ellipsoid is a^2 / c (longest and shortest semi-axis), at the end of
        its shortest axis: here the dense body of geo-train-06."""
        with open(os.path.join(F.CATALOG_DIR, "geo-train-06.json"), "rb") as fh:
            spec = json.loads(fh.read().decode("utf-8"))
        body = next(a for a in spec["anomalies"] if a["kind"] == "dense_body")["params"]
        axes = [body["ax_m"], body["ay_m"], body["az_m"]]
        self.assertAlmostEqual(A.R_MAX_M, max(axes) ** 2 / min(axes), places=9)

    def test_an_explained_violation_is_reported_and_does_not_block(self):
        def tweak(records, entries):
            rec = records["g06-B-r1"][5]
            rec["lz_a0"], rec["lat_z"], rec["sum_lat_z"] = 2e-3, 0.5, 1.0     # 2 mm of A0 vs a 0.83 m scale
            rec = records["g06-B-r2"][9]
            rec["lz_rock"], rec["lat_z"], rec["sum_lat_z"] = 3e-6, 0.005, 0.01  # 3 um of rock vs a 0.083 m scale

        fx = self.build("scale-explained", tweak)
        rc, out, stdout, err = analyze(fx, self.tmp)
        self.assertEqual(rc, 0, stdout + err)
        ctr = load(os.path.join(out, "contratos.json"))
        for scope, key, ident in (("g06-A-r1 vs g06-B-r1 (B)", "max_lz_a0_m", 5),
                                  ("g06-A-r2 vs g06-B-r2 (B)", "max_lz_rock_m", 9)):
            blocking = checks_of(ctr, "K9", scope, "salvo lo que explica")
            self.assertEqual(len(blocking), 1)
            self.assertTrue(blocking[0]["bloqueante"])
            self.assertIs(blocking[0]["ok"], True)                       # not a blocking failure...
            self.assertEqual(blocking[0]["evidencia"]["violaciones"], 1)
            self.assertEqual(blocking[0]["evidencia"]["sin_explicar"], 0)
            self.assertGreater(blocking[0]["evidencia"][key], 1e-6)
            c = self.assertCheckFails(ctr, "K9", scope, "explicadas por el desplazamiento lateral")   # ...but reported
            self.assertFalse(c["bloqueante"])
            self.assertEqual(c["evidencia"]["explicados"], 1)
            self.assertEqual(c["evidencia"]["ids_explicados"], [ident])
            self.assertLessEqual(c["evidencia"]["exceso_max_m"], c["evidencia"]["escala_max_m"])
            self.assertEqual(c["evidencia"]["R_max_m"], A.R_MAX_M)
        self.assertEqual(ctr["bloqueantes_fallidos"], [])
        self.assertEqual(ctr["contratos"]["K9"]["veredicto"], "REQUIERE_EXPLICACION")
        self.assertTrue(ctr["atribucion_al_solape_bloqueada"])           # read before attributing (PROT:55)
        self.assertTrue(any("explicadas por el desplazamiento lateral" in x for x in ctr["no_bloqueantes_fallidos"]))

    def test_a_violation_beyond_the_scale_or_without_lateral_data_blocks(self):
        def tweak(records, entries):
            rec = records["g06-B-r1"][5]
            rec["lz_a0"], rec["lat_z"], rec["sum_lat_z"] = 2.0, 0.5, 1.0        # 2 m against a 0.83 m scale
            rec = records["g06-B-r2"][9]
            rec["lz_rock"] = 2e-3                                              # sum_lat_z is 0 (the fixture default)
            rec = records["g06-B-r3"][11]
            rec["lz_a0"], rec["lz_rock"], rec["lat_z"], rec["sum_lat_z"] = 0.5, 0.5, 0.5, 1.0   # 1.0 m in total

        fx = self.build("scale-unexplained", tweak)
        rc, out, _, _ = analyze(fx, self.tmp)
        self.assertEqual(rc, 3)
        ctr = load(os.path.join(out, "contratos.json"))
        for scope, ident in (("g06-A-r1 vs g06-B-r1 (B)", 5), ("g06-A-r2 vs g06-B-r2 (B)", 9),
                             ("g06-A-r3 vs g06-B-r3 (B)", 11)):
            c = self.assertCheckFails(ctr, "K9", scope, "salvo lo que explica")
            self.assertTrue(c["bloqueante"])
            self.assertEqual(c["evidencia"]["sin_explicar"], 1)
            self.assertEqual(c["evidencia"]["ids_sin_explicar"], [ident])
            self.assertEqual(checks_of(ctr, "K9", scope, "explicadas por el desplazamiento lateral")[0]["ok"], True)

    def test_a_binary_without_the_lateral_columns_cannot_explain_anything(self):
        def tweak(records, entries):
            records["g06-B-r1"][5]["lz_a0"] = 2e-6                  # 2 um: above the tolerance, nothing to explain it

        fx = F.build_attempt(os.path.join(self.tmp, "scale-old"), n=N, tweak=tweak)          # no lateral columns
        rc, out, _, _ = analyze(fx, self.tmp)
        self.assertEqual(rc, 3)
        c = self.assertCheckFails(load(os.path.join(out, "contratos.json")), "K9", "g06-A-r1 vs g06-B-r1 (B)",
                                  "salvo lo que explica")
        self.assertEqual(c["evidencia"]["sin_explicar"], 1)
        self.assertAlmostEqual(c["evidencia"]["max_lz_a0_m"], 2e-6)


class EfficiencySanityTests(Base):
    """S3: an efficiency far from the campaign's (|z| > 5) is flagged, non-blocking."""

    def test_pristine_efficiencies_are_compatible_with_the_campaign(self):
        fx = F.build_attempt(os.path.join(self.tmp, "eff-ok"), n=N)
        rc, out, stdout, err = analyze(fx, self.tmp)
        self.assertEqual(rc, 0, stdout + err)
        ctr = load(os.path.join(out, "contratos.json"))
        found = checks_of(ctr, "K6", "", "eficiencia de detección")
        self.assertEqual(len(found), 13)
        self.assertTrue(all(c["ok"] is True and not c["bloqueante"] for c in found))
        self.assertEqual(ctr["no_bloqueantes_fallidos"], [])

    def test_an_efficiency_far_from_the_campaign_is_flagged_and_does_not_block(self):
        def tweak(records, entries):
            for name in ("g01-A-r3", "g01-B-r3"):                   # half of the detections vanish, alike in A and B
                for rec in records[name]:
                    if rec["status"] == "DET" and rec["id"] % 2 == 0:
                        set_status(rec, "DECAY")

        fx = F.build_attempt(os.path.join(self.tmp, "eff-bad"), n=N, tweak=tweak)
        rc, out, stdout, err = analyze(fx, self.tmp)
        self.assertEqual(rc, 0, stdout + err)
        ctr = load(os.path.join(out, "contratos.json"))
        for run_id in ("g01-A-r3", "g01-B-r3"):
            c = self.assertCheckFails(ctr, "K6", run_id, "eficiencia de detección")
            self.assertFalse(c["bloqueante"])
            self.assertLess(c["evidencia"]["z"], -5.0)
            self.assertEqual(c["evidencia"]["referencia_campana"], A.EFFICIENCY_REFERENCE["geo-train-01"])
        self.assertEqual(ctr["bloqueantes_fallidos"], [])
        self.assertEqual(len(ctr["no_bloqueantes_fallidos"]), 2)
        self.assertEqual(ctr["contratos"]["K6"]["veredicto"], "REQUIERE_EXPLICACION")
        self.assertFalse(ctr["atribucion_al_solape_bloqueada"])         # a K6 note, not a K8 / K9 one
        rows = read_rows(os.path.join(out, "tabla-corridas.csv"))
        z = float(next(r for r in rows if r["corrida"] == "g01-A-r3")["z_eficiencia_vs_campana"])
        self.assertLess(z, -5.0)

    def test_the_references_are_the_campaign_manifest_ratios(self):
        """detected / events_transported of view 0 in results/dataset/campaign-manifest.csv. 0.752094 was the
        pilot's 3e6-event value for geo-train-06, not the campaign's."""
        self.assertEqual(A.EFFICIENCY_REFERENCE["geo-train-06"], 27209253 / 36174382)
        self.assertAlmostEqual(A.EFFICIENCY_REFERENCE["geo-train-06"], 0.752169, places=6)
        self.assertAlmostEqual(A.EFFICIENCY_REFERENCE["geo-train-01"], 26690266 / 36168098, places=5)

    def test_the_limit_is_five_sigma_and_the_z_uses_the_binomial_sigma(self):
        self.assertEqual(A.EFFICIENCY_Z_LIMIT, 5.0)
        ref = A.EFFICIENCY_REFERENCE["geo-train-06"]
        self.assertAlmostEqual(A.efficiency_z("geo-train-06", 100000, 100000), (1 - ref) / (ref * (1 - ref) / 1e5) ** 0.5)
        self.assertAlmostEqual(A.efficiency_z("geo-train-06", int(round(ref * 1000)), 1000),
                               (round(ref * 1000) / 1000 - ref) / (ref * (1 - ref) / 1000) ** 0.5)
        self.assertIsNone(A.efficiency_z("geo-train-99", 10, 100))


class TimingKeysInComparisonsTests(Base):
    """S1: K7 and K8 ignore an explicit allow-list of meta keys and say which; any other numeric difference is one."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fx = F.build_attempt(os.path.join(cls.tmp, "timing"), n=N)

    def run_dir(self, root, run_id):
        e = next(x for x in self.fx["entries"] if x["run_id"] == run_id)
        return os.path.join(root, e["salida"]), "%s_view0.meta.json" % e["caso"]

    def test_only_the_allow_listed_keys_are_ignored_and_reported(self):
        root = self.clone(self.fx)
        for run_id, extra in (("g01-A-r1", 5.0), ("g01-A-r1-rep", 6.0)):
            d, meta = self.run_dir(root, run_id)
            edit_json(os.path.join(d, meta), lambda m, x=extra: m.update(
                cpu_time_used=x, rusage_max_rss_kb=int(x * 1e4)))             # a look-alike key and a listed one
            F.write_sums(d)
        for run_id, extra in (("g01-A-r2", 1.0), ("g01-B-r2", 2.0)):
            d, meta = self.run_dir(root, run_id)
            edit_json(os.path.join(d, meta), lambda m, x=extra: m.update(memory_mb=x, wall_seconds=x))
            F.write_sums(d)
        rc, out, stdout, err = analyze(self.fx, self.tmp, attempt=root)
        ctr = load(os.path.join(out, "contratos.json"))
        k7 = self.assertCheckFails(ctr, "K7", "g01-A-r1-rep", "meta.json igual")
        self.assertEqual(k7["evidencia"]["campos_distintos"], ["cpu_time_used"])       # ignored before: "cpu" matched
        self.assertIn("rusage_max_rss_kb", k7["evidencia"]["claves_de_tiempo_ignoradas"])
        self.assertNotIn("cpu_time_used", k7["evidencia"]["claves_de_tiempo_ignoradas"])
        self.assertIn("rusage_user_cpu_seconds", k7["descripcion"])
        k8 = self.assertCheckFails(ctr, "K8", "g01-A-r2 vs g01-B-r2", "meta.json igual")
        self.assertEqual(k8["evidencia"]["campos_distintos"], ["memory_mb"])
        for key in ("wall_seconds", "seconds_per_event", "rusage_user_cpu_seconds", "rusage_system_cpu_seconds",
                    "rusage_max_rss_kb"):
            self.assertIn(key, k8["evidencia"]["claves_de_tiempo_ignoradas"])
        self.assertIn("oe1_etapa1.binary_sha256", k8["evidencia"]["otras_rutas_ignoradas"])
        # K8 lists the keys it ignored also when nothing differs
        clean = checks_of(ctr, "K8", "g01-A-r3 vs g01-B-r3", "meta.json igual")[0]
        self.assertTrue(clean["ok"])
        self.assertEqual(clean["evidencia"]["claves_de_tiempo_ignoradas"],
                         sorted(("wall_seconds", "seconds_per_event", "rusage_user_cpu_seconds",
                                 "rusage_system_cpu_seconds", "rusage_max_rss_kb")))


class ResourceColumnsTests(Base):
    """S6: cpu_s and rss_kb come from the meta's rusage_* keys, else from recursos.json, and say which."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fx = F.build_attempt(os.path.join(cls.tmp, "resources"), n=N)

    def test_fallback_to_recursos_json_and_na(self):
        root = self.clone(self.fx)

        def run_dir(run_id):
            e = next(x for x in self.fx["entries"] if x["run_id"] == run_id)
            return os.path.join(root, e["salida"]), "%s_view0.meta.json" % e["caso"]

        d, meta = run_dir("g06-A-r2")                                 # no rusage in the meta: recursos.json
        edit_json(os.path.join(d, meta), lambda m: [m.pop(k) for k in list(m) if k.startswith("rusage_")])
        F.write_sums(d)
        d, meta = run_dir("g06-B-r2")                                 # neither source
        edit_json(os.path.join(d, meta), lambda m: [m.pop(k) for k in list(m) if k.startswith("rusage_")])
        os.remove(os.path.join(d, "recursos.json"))
        F.write_sums(d)
        d, meta = run_dir("g06-A-r3")                                 # only the RSS in the meta
        edit_json(os.path.join(d, meta), lambda m: (m.pop("rusage_user_cpu_seconds"), m.pop("rusage_system_cpu_seconds")))
        F.write_sums(d)
        rc, out, stdout, err = analyze(self.fx, self.tmp, attempt=root)
        self.assertEqual(rc, 0, stdout + err)
        rows = {r["corrida"]: r for r in read_rows(os.path.join(out, "tabla-corridas.csv"))}
        r = rows["g06-A-r2"]
        self.assertEqual((r["fuente_cpu"], r["fuente_rss"]), ("recursos.json", "recursos.json"))
        self.assertAlmostEqual(float(r["cpu_s"]), 12.25 + 0.1)
        self.assertEqual(r["rss_kb"], "90000")
        r = rows["g06-B-r2"]
        self.assertEqual((r["fuente_cpu"], r["fuente_rss"], r["cpu_s"], r["rss_kb"]), ("NA", "NA", "NA", "NA"))
        r = rows["g06-A-r3"]
        self.assertEqual((r["fuente_cpu"], r["fuente_rss"]), ("recursos.json", "meta.rusage"))
        self.assertEqual(r["rss_kb"], "91250")
        r = rows["g06-B-r1"]
        self.assertEqual((r["fuente_cpu"], r["fuente_rss"]), ("meta.rusage", "meta.rusage"))
        res = load(os.path.join(out, "resultados-etapa1.json"))
        self.assertIn("recursos.json", res["leyendas"]["recursos"])
        self.assertIn("fuente_cpu", res["leyendas"]["recursos"])

    def test_resources_of_prefers_the_meta(self):
        meta = {"rusage_user_cpu_seconds": 3.0, "rusage_system_cpu_seconds": 0.5, "rusage_max_rss_kb": 1234}
        rec = {"user_s": 9.0, "sys_s": 9.0, "maxrss_kb": 9}
        self.assertEqual(A.resources_of(meta, rec), (3.5, "meta.rusage", 1234, "meta.rusage"))
        self.assertEqual(A.resources_of({}, rec), (18.0, "recursos.json", 9, "recursos.json"))
        self.assertEqual(A.resources_of({}, {}), (None, "NA", None, "NA"))
        self.assertEqual(A.resources_of({"rusage_max_rss_kb": True}, {}), (None, "NA", None, "NA"))   # bool is not a number


if __name__ == "__main__":
    unittest.main()
