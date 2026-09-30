"""Tests of the primary-list validator (oe1_stage1_lists)."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)

import oe1_stage1_common as C  # noqa: E402
import oe1_stage1_lists as L  # noqa: E402
import fixtures as F  # noqa: E402

try:
    import scipy.stats  # noqa: F401
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

DATA = os.path.join(HERE, "data")
# 50 muons written by the binary: the first 50 rows of the PRODUCTION replica-1 list of geo-train-06 (sidecar:
# source_seed 2026092901, replica 1, case_code 6; SC3 job 30510), not technical-seed data. They sit under an
# obviously technical name, never a name a manifest could point to.
REAL50_NAME = "fixture-tecnico-n50.primarios.csv"
REAL50 = os.path.join(DATA, REAL50_NAME)
N = 10000


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = L.main(argv)
    return rc, out.getvalue()


class ListValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="oe1lists_")
        cls.tables = C.load_tables(F.TABLES_DIR)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def make(self, name, seed=5, base_radius=509.7, sidecar_radius=None, n=N, mutate=None, **kw):
        rng = np.random.default_rng(seed)
        ids, vals, pdg, charge = F.sample_source_model(rng, n, self.tables, base_radius, **kw)
        if mutate:
            vals, pdg, charge = mutate(vals, pdg, charge)
        d = tempfile.mkdtemp(dir=self.tmp)
        m = L.NAME_RE.match(name)
        info = F.write_list(d, name, ids, vals, pdg, charge, "geo-train-06", 6, int(m.group("rep")),
                            int(m.group("semilla")), sidecar_radius if sidecar_radius else base_radius)
        return info

    def validate(self, info, **kw):
        return L.validate_list(info["path"], self.tables, F.CATALOG_DIR, **kw)

    # ------------------------------------------------------------------
    def test_valid_list_passes_everything(self):
        info = self.make("geo-train-06_v0_r1_s2026092901.primarios.csv")
        res = self.validate(info)
        self.assertEqual(res["problemas"], [], res["problemas"])
        self.assertEqual(res["veredicto"], "OK", [t for t in res["estadistica"]["pruebas"] if t["marcada"]])
        self.assertLessEqual(res["geometria"]["desviacion_plano_max_m"], 1e-9)
        self.assertLessEqual(res["geometria"]["desviacion_radio_de_partida_max_m"], 1e-9)
        self.assertEqual(res["geometria"]["exceso_sobre_panel_max_m"], 0.0)
        self.assertEqual(res["canon_fnv1a64"], C.hex16(info["canon"]))
        names = [t["prueba"] for t in res["estadistica"]["pruebas"]]
        self.assertEqual(len(names), 6 + 16)      # rows, jitter, relAz, y, z, charge + 16 momentum bands

    @unittest.skipUnless(HAVE_SCIPY, "SciPy not installed")
    def test_internal_p_values_agree_with_scipy(self):
        info = self.make("geo-train-06_v0_r1_s2026092901.primarios.csv", seed=8)
        res = self.validate(info)
        for t in res["estadistica"]["pruebas"]:
            if t.get("p_scipy") is None:
                continue
            self.assertLess(abs(t["p"] - t["p_scipy"]), 0.03 * max(t["p_scipy"], 0.03) + 1e-9, t["prueba"])

    def test_wrong_start_radius_is_a_geometry_failure(self):
        info = self.make("geo-train-06_v0_r1_s2026092901.primarios.csv", base_radius=656.2, sidecar_radius=509.7)
        res = self.validate(info)
        self.assertEqual(res["veredicto"], "FALLA")
        self.assertFalse(res["geometria"]["ok"])
        self.assertGreater(res["geometria"]["desviacion_radio_de_partida_max_m"], 100.0)

    def test_track_missing_the_panel(self):
        def shift(vals, pdg, charge):
            vals = vals.copy()
            vals[:, 1] += 1.0                        # the whole line moves 1 m in y
            return vals, pdg, charge

        res = self.validate(self.make("geo-train-06_v0_r1_s2026092901.primarios.csv", mutate=shift))
        self.assertEqual(res["veredicto"], "FALLA")
        self.assertGreater(res["geometria"]["exceso_sobre_panel_max_m"], 0.3)

    def test_charge_ratio_flagged(self):
        res = self.validate(self.make("geo-train-06_v0_r1_s2026092901.primarios.csv", plus_fraction=0.9))
        self.assertEqual(res["veredicto"], "REVISAR_ESTADISTICA")
        self.assertTrue(any("mu+" in t["prueba"] and t["marcada"] for t in res["estadistica"]["pruebas"]))

    def test_momentum_and_azimuth_and_rows_flagged(self):
        res = self.validate(self.make("geo-train-06_v0_r1_s2026092901.primarios.csv", p_scale=1.3))
        flagged = [t["prueba"] for t in res["estadistica"]["pruebas"] if t["marcada"]]
        self.assertTrue(any("momento" in f for f in flagged), flagged)
        res = self.validate(self.make("geo-train-06_v0_r1_s2026092901.primarios.csv", rel_scale=0.9))
        # relAz compressed: the range check passes but uniformity in +-30 deg fails
        flagged = [t["prueba"] for t in res["estadistica"]["pruebas"] if t["marcada"]]
        self.assertTrue(any("acimut" in f for f in flagged), flagged)
        flat = dict(self.tables)
        flat["weights"] = np.ones_like(self.tables["weights"])
        rng = np.random.default_rng(4)
        ids, vals, pdg, charge = F.sample_source_model(rng, N, flat, 509.7)
        d = tempfile.mkdtemp(dir=self.tmp)
        info = F.write_list(d, "geo-train-06_v0_r1_s2026092901.primarios.csv", ids, vals, pdg, charge,
                            "geo-train-06", 6, 1, 2026092901, 509.7)
        res = self.validate(info)
        self.assertTrue(any("filas" in t["prueba"] and t["marcada"] for t in res["estadistica"]["pruebas"]))

    def test_structural_defects(self):
        info = self.make("geo-train-06_v0_r1_s2026092901.primarios.csv", n=2000)
        path = info["path"]
        with open(path, "rb") as fh:
            raw = fh.read()
        lines = raw.rstrip(b"\n").split(b"\n")
        # duplicated initial state
        dup = lines + [lines[10].replace(b"10,", b"2000,", 1)]
        with open(path, "wb") as fh:
            fh.write(b"\n".join(dup) + b"\n")
        res = L.validate_list(path, self.tables, None, do_stats=False)
        self.assertEqual(res["veredicto"], "FALLA")
        self.assertEqual(res["estructural"]["defectos"]["dup_states"], 1)
        # sidecar no longer matches the content
        self.assertFalse(res["sidecar"]["n_ok"])
        self.assertFalse(res["sidecar"]["canon_ok"])
        # CRLF
        with open(path, "wb") as fh:
            fh.write(raw.replace(b"\n", b"\r\n"))
        res = L.validate_list(path, self.tables, None, do_stats=False)
        self.assertEqual(res["estructural"]["defectos"]["lf_only"], 0)
        self.assertEqual(res["veredicto"], "FALLA")

    def test_sidecar_problems(self):
        info = self.make("geo-train-06_v0_r1_s2026092901.primarios.csv", n=1000)
        sc = C.sidecar_path(info["path"])
        with open(sc, "rb") as fh:
            side = json.loads(fh.read().decode())
        side["canon_fnv1a64"] = "0" * 16
        side["case_code"] = 1
        with open(sc, "wb") as fh:
            fh.write(json.dumps(side).encode())
        res = L.validate_list(info["path"], self.tables, F.CATALOG_DIR, do_stats=False)
        self.assertEqual(res["veredicto"], "FALLA")
        self.assertFalse(res["sidecar"]["canon_ok"])
        self.assertFalse(res["sidecar"]["case_code_ok"])
        os.remove(sc)
        res = L.validate_list(info["path"], self.tables, F.CATALOG_DIR, do_stats=False)
        self.assertFalse(res["sidecar"]["presente"])
        self.assertEqual(res["veredicto"], "FALLA")

    def test_catalog_radius_mismatch(self):
        info = self.make("geo-train-06_v0_r1_s2026092901.primarios.csv", n=1000, base_radius=656.2)
        res = L.validate_list(info["path"], self.tables, F.CATALOG_DIR, do_stats=False)
        self.assertFalse(res["sidecar"]["radio_base_catalogo_ok"])
        self.assertEqual(res["veredicto"], "FALLA")

    def test_parity_with_the_binary_check_line(self):
        recorded = os.path.join(self.tmp, "check-A2.txt")
        with open(recorded, "w") as fh:
            fh.write("OE1-CHECK-PRIMARIES file=" + REAL50_NAME + " n=50 lf_only=1 "
                     "header_ok=1 ends_with_newline=1 bad_rows=0 id_min=0 id_max=49 ids_contiguous=1 dup_ids=0 "
                     "dup_states=0 nonfinite=0 norm_fail=0 max_norm_dev=2.220446049250313e-16 dz_nonneg=0 "
                     "ekin_nonpos=0 pdg_bad=0 charge_mismatch=0 n_mu_plus=32 n_mu_minus=18 "
                     "canon_fnv1a64=73ff04a2e4429cfa file_sha256=0ae76a8d610ca8a37c688768fef5b08cf9c9433e7154f5132381879abd2b0c28 "
                     "sidecar=ok sidecar_n=50 sidecar_fnv_match=1 sidecar_sha_match=1 status=OK\n")
        cpp = L._cpp_lines([recorded])
        res = L.validate_list(REAL50, self.tables, F.CATALOG_DIR, cpp_lines=cpp)
        self.assertEqual(res["comparacion_con_binario"]["todo_igual"], True)
        self.assertEqual(res["problemas"], [])
        # a counter that disagrees is a failure
        bad = dict(cpp[REAL50_NAME])
        bad["n_mu_plus"] = "31"
        res = L.validate_list(REAL50, self.tables, F.CATALOG_DIR,
                              cpp_lines={REAL50_NAME: bad})
        self.assertEqual(res["veredicto"], "FALLA")
        # no line at all for the list
        res = L.validate_list(REAL50, self.tables, F.CATALOG_DIR, cpp_lines={})
        self.assertEqual(res["veredicto"], "FALLA")

    def test_real_list_written_by_the_binary(self):
        res = L.validate_list(REAL50, self.tables, F.CATALOG_DIR)
        self.assertEqual(res["problemas"], [])
        self.assertEqual(res["canon_fnv1a64"], "73ff04a2e4429cfa")
        # the fixture's name does not follow <case>_v<view>_r<k>_s<seed>.primarios.csv: the name is not checked
        # (None), it is neither a match nor a failure; the content checks all pass
        self.assertIsNone(res["sidecar"]["nombre_ok"])
        self.assertLessEqual(res["geometria"]["desviacion_plano_max_m"], 1e-9)

    # ------------------------------------------------------------------
    def test_cli_exit_codes_and_output_file(self):
        ok = self.make("geo-train-06_v0_r1_s2026092901.primarios.csv", n=3000)
        out = os.path.join(self.tmp, "val-ok.json")
        rc, text = run_cli(["validate", ok["path"], "--tables", F.TABLES_DIR, "--catalog-dir", F.CATALOG_DIR,
                            "--out", out])
        self.assertEqual(rc, 0, text)
        with open(out, encoding="utf-8") as fh:
            rep = json.load(fh)
        self.assertEqual(rep["veredicto"], "OK")
        self.assertEqual(rep["listas"][0]["n"], 3000)
        rc, text = run_cli(["validate", ok["path"], "--tables", F.TABLES_DIR, "--out", out])     # never overwrites
        self.assertEqual(rc, 2)
        with open(out, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["veredicto"], "OK")                                   # untouched
        rc, text = run_cli(["validate", os.path.join(self.tmp, "no-such-list.csv"), "--no-stats"])
        self.assertEqual(rc, 2)                                                                  # no traceback
        rc, text = run_cli(["validate", ok["path"], "--tables", os.path.join(self.tmp, "no-tables")])
        self.assertEqual(rc, 2)
        bad = self.make("geo-train-06_v0_r2_s2026092902.primarios.csv", n=3000, base_radius=656.2,
                        sidecar_radius=509.7)
        rc, text = run_cli(["validate", bad["path"], "--tables", F.TABLES_DIR])
        self.assertEqual(rc, 3)
        flagged = self.make("geo-train-06_v0_r3_s2026092903.primarios.csv", n=N, plus_fraction=0.9)
        rc, text = run_cli(["validate", flagged["path"], "--tables", F.TABLES_DIR])
        self.assertEqual(rc, 4)
        self.assertIn("MARCADA", text)
        rc, text = run_cli(["validate", flagged["path"], "--no-stats"])
        self.assertEqual(rc, 0)

    def test_expect_distinct_across_lists(self):
        a = self.make("geo-train-06_v0_r1_s2026092901.primarios.csv", n=800, seed=1)
        b = self.make("geo-train-06_v0_r2_s2026092902.primarios.csv", n=800, seed=2)
        rc, _ = run_cli(["validate", a["path"], b["path"], "--no-stats", "--expect-distinct"])
        self.assertEqual(rc, 0)
        d = tempfile.mkdtemp(dir=self.tmp)
        twin = os.path.join(d, "geo-train-06_v0_r3_s2026092903.primarios.csv")
        shutil.copy(a["path"], twin)
        shutil.copy(C.sidecar_path(a["path"]), C.sidecar_path(twin))
        rc, _ = run_cli(["validate", a["path"], twin, "--no-stats", "--expect-distinct"])
        self.assertEqual(rc, 3)

    def test_derive_replaces_only_the_prefix(self):
        base = self.make("geo-train-06_v0_r1_s2026092901.primarios.csv", n=500, seed=1)
        other = self.make("geo-train-06_v0_r1_s2026092991.primarios.csv", n=500, seed=2)
        d = tempfile.mkdtemp(dir=self.tmp)
        out = os.path.join(d, "derivada.primarios.csv")
        rc, _ = run_cli(["derive", "--base", base["path"], "--other", other["path"], "--replace-first", "60",
                         "--out", out])
        self.assertEqual(rc, 0)
        tb, _, _, _ = C.read_primary_file(base["path"])
        to, _, _, _ = C.read_primary_file(other["path"])
        td, dd, sha, canon = C.read_primary_file(out)
        self.assertTrue(C.primary_defects_clean(dd, 500))
        self.assertTrue(np.array_equal(td["vals"][:60], to["vals"][:60]))
        self.assertTrue(np.array_equal(td["vals"][60:], tb["vals"][60:]))
        sc, _ = C.read_sidecar(out)
        self.assertEqual(sc["canon_fnv1a64"], C.hex16(canon))
        self.assertEqual(sc["kind"], "derivada-T-P")
        self.assertEqual(sc["case_code"], 6)
        res = L.validate_list(out, self.tables, None, do_stats=False)
        self.assertEqual(res["veredicto"], "OK")
        before = C.sha256_file(out)
        rc, _ = run_cli(["derive", "--base", base["path"], "--other", other["path"], "--replace-first", "60",
                         "--out", out])
        self.assertEqual(rc, 2)                                       # never overwrites; one line, no traceback
        self.assertEqual(C.sha256_file(out), before)


class Chi2Tests(unittest.TestCase):
    def test_chi2_gof_merges_small_cells(self):
        obs = np.array([1, 2, 1, 30, 40, 26.0])
        exp = np.array([2, 2, 1, 30, 40, 25.0])
        stat, df, p = L.chi2_gof(obs, exp)
        self.assertEqual(df, 3)                   # cells 0-2 merge (E = 5); 30, 40, 25 stay
        self.assertGreater(p, 0.5)


if __name__ == "__main__":
    unittest.main()
