"""Tests of the T-S synthetic list generator and of its check."""
import contextlib
import copy
import io
import math
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
import oe1_stage1_roi as R  # noqa: E402
import oe1_stage1_synthetic as S  # noqa: E402
import fixtures as F  # noqa: E402

CATALOG = os.path.join(F.CATALOG_DIR, "geo-train-06.json")


def boundary_samples(cr, n, seed=5):
    """Points on the boundary of Z: Z is convex, so rays from the centroid meet it once."""
    cen = np.array(R.Z_CENTROID_M)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    m = cr.model(np.hstack([np.tile(cen, (n, 1)), cen + 300.0 * v]))
    return cen + (np.asarray(m["hi"]) * 300.0)[:, None] * v


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="oe1syn_")
        cls.saved = S.N_PER_CLASS
        S.N_PER_CLASS = 20                         # the real size: 60 muons
        cls.out = os.path.join(cls.tmp, "ts")
        cls.res = S.generate(cls.out, CATALOG, F.CROSSING_REFERENCE)
        cls.cr, _, _ = C.load_crossing_reference(F.CROSSING_REFERENCE)
        cls.expected = C.read_json(cls.res["esperado"])
        cls.table, cls.defects, cls.sha, cls.canon = C.read_primary_file(cls.res["lista"])

    @classmethod
    def tearDownClass(cls):
        S.N_PER_CLASS = cls.saved
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_list_format_and_counts(self):
        self.assertEqual(len(self.table["id"]), 60)
        self.assertTrue(C.primary_defects_clean(self.defects, 60))
        self.assertEqual(self.res["clases"], {"atraviesa": 20, "casi_fallo": 20, "lejos": 20})
        self.assertTrue(np.all(self.table["pdg"] == 13))            # mu-
        self.assertTrue(np.all(self.table["charge"] == -1))
        self.assertTrue(np.all(self.table["vals"][:, 6] == 10000.0))    # 10 TeV kinetic
        sc, err = C.read_sidecar(self.res["lista"])
        self.assertIsNone(err)
        self.assertEqual(sc["format"], C.LIST_FORMAT)
        self.assertEqual((sc["case_code"], sc["replica"], sc["kind"]), (6, 9, "sintetico"))
        self.assertIsNone(sc["source_seed"])
        self.assertTrue(sc["synthetic"])
        self.assertEqual(sc["canon_fnv1a64"], C.hex16(self.canon))
        self.assertEqual(sc["file_sha256"], self.sha)
        # the sidecar reader of the binary needs exactly these keys (PrimaryList.cc ReadSidecar / Load)
        for key in ("format", "n", "canon_fnv1a64", "geometry_id", "view_index", "base_radius_m",
                    "case_code", "replica"):
            self.assertIn(key, sc)
        ok, line = C.format_check_line("x", 60, self.defects, self.canon, self.sha, sc, None)
        self.assertTrue(ok, line)

    def test_geometry_like_the_generator(self):
        res = L.validate_list(self.res["lista"], None, None, do_stats=False)
        self.assertEqual(res["veredicto"], "OK", res["problemas"])
        self.assertLessEqual(res["geometria"]["desviacion_plano_max_m"], 1e-9)
        self.assertLessEqual(res["geometria"]["desviacion_radio_de_partida_max_m"], 1e-9)
        self.assertEqual(res["geometria"]["exceso_sobre_panel_max_m"], 0.0)
        self.assertAlmostEqual(res["geometria"]["radio_de_partida_esperado_m"], 1809.7, places=9)
        # every track descends and starts beyond the far rim (horizontal distance 1809.7 m)
        v = self.table["vals"]
        self.assertTrue(np.all(v[:, 5] < 0))

    def test_classification_matches_the_reference_model(self):
        v = self.table["vals"]
        hor = np.hypot(v[:, 3], v[:, 4])
        imp = v[:, 0:3] + v[:, 3:6] * (1809.7 / hor)[:, None]
        model = self.cr.model(np.hstack([v[:, 0:3], imp]))
        for k, e in enumerate(self.expected["muones"]):
            crossed = bool(model["deep"][k])
            self.assertEqual(crossed, e["crossed_esperado"] == 1, (k, e["clase"]))
            self.assertAlmostEqual(float(model["len"][k]), e["lz_recta_m"], delta=1e-9)
            if e["crossed_esperado"] == 0:
                self.assertEqual(float(model["len"][k]), 0.0)
                self.assertFalse(bool(model["graze"][k]))

    def test_dense_sampling_oracle_agrees(self):
        """Independent of the interval model: dense sampling with the margin() convention."""
        v = self.table["vals"]
        hor = np.hypot(v[:, 3], v[:, 4])
        imp = v[:, 0:3] + v[:, 3:6] * (1809.7 / hor)[:, None]
        for k in (0, 3, 9, 20, 25, 33, 41, 50, 58):
            seg = np.concatenate([v[k, 0:3], imp[k]])
            est, chord, res = self.cr.dense_len(seg, 100001)
            e = self.expected["muones"][k]
            self.assertLessEqual(abs(est - e["lz_recta_m"]), res + 1e-9, (k, e["clase"]))

    def test_clearances_and_depths(self):
        """Near misses are 0.5-1.0 m from Z; through lines are >= 3 m deep; far ones >= 50 m."""
        bd = boundary_samples(self.cr, 400000)
        v = self.table["vals"]
        hor = np.hypot(v[:, 3], v[:, 4])
        imp = v[:, 0:3] + v[:, 3:6] * (1809.7 / hor)[:, None]
        cen = np.array(R.Z_CENTROID_M)
        for k, e in enumerate(self.expected["muones"]):
            f = -v[k, 3:6]
            dist = float(np.linalg.norm(np.cross(bd - imp[k], f), axis=1).min())
            if e["clase"] == "casi_fallo":
                self.assertGreater(dist, 0.5 - 0.03, (k, dist))
                self.assertLess(dist, 1.0 + 0.03, (k, dist))
                self.assertAlmostEqual(dist, e["holgura_estimada_m"], delta=0.03)
            elif e["clase"] == "lejos":
                self.assertGreaterEqual(dist, 50.0, (k, dist))
            else:
                t = float(np.dot(cen - imp[k], f))
                pc = imp[k] + t * f
                depth = float(np.linalg.norm(bd - pc, axis=1).min())
                self.assertGreaterEqual(depth, 3.0 - 0.03, (k, depth))
                self.assertGreaterEqual(e["profundidad_garantizada_m"], 3.0 - 1e-9)

    def test_through_lines_carry_the_transverse_displacement_interval(self):
        """PLAN S1: the acceptable length of a shallow line is the range spanned by displaced parallel lines."""
        exp = self.expected
        self.assertEqual(exp["tolerancia_de_desplazamiento_transversal_m"], S.TRANSVERSE_TOLERANCE_M)
        self.assertEqual(exp["tolerancia_relativa_de_longitud"], 0.01)
        n_central = 0
        for e in exp["muones"]:
            if e["clase"] != "atraviesa":
                self.assertNotIn("lz_desplazada_min_m", e)
                self.assertNotIn("central", e)
                continue
            ref = e["lz_recta_m"]
            self.assertLessEqual(e["lz_desplazada_min_m"], ref + 1e-9)
            self.assertGreaterEqual(e["lz_desplazada_max_m"], ref - 1e-9)
            inside = (e["lz_desplazada_min_m"] >= ref * 0.99 and e["lz_desplazada_max_m"] <= ref * 1.01)
            self.assertEqual(e["central"], inside)
            n_central += int(e["central"])
        self.assertGreater(n_central, 0)                # the deep lines are insensitive to a 5 cm shift

    def test_interval_brackets_lines_displaced_inside_the_disc(self):
        cr, _, _ = C.load_crossing_reference(F.CROSSING_REFERENCE)
        cat = C.read_catalog(CATALOG)
        scene = S.Scene(cr, cat)
        v = self.table["vals"]
        rng = np.random.default_rng(3)
        for k, e in enumerate(self.expected["muones"]):
            if e["clase"] != "atraviesa":
                continue
            f = -v[k, 3:6]
            hor = math.hypot(f[0], f[1])
            panel = v[k, 0:3] + v[k, 3:6] * (scene.r_start / hor)          # impact point on the panel
            e1, e2 = S._perp_basis(f)
            for _ in range(20):
                r = S.TRANSVERSE_TOLERANCE_M * math.sqrt(rng.random())
                a = 2 * math.pi * rng.random()
                off = (r * math.cos(a)) * e1 + (r * math.sin(a)) * e2
                ln = float(scene.lengths_many(panel, f, off[None, :])[0])
                self.assertGreaterEqual(ln, e["lz_desplazada_min_m"] - 1e-4, (k, r))
                self.assertLessEqual(ln, e["lz_desplazada_max_m"] + 1e-4, (k, r))

    def test_inscribed_and_bounding_radius(self):
        sc = self.expected["region"]
        self.assertAlmostEqual(sc["radio_esfera_inscrita_m"], 14.7, delta=1e-3)
        self.assertAlmostEqual(sc["radio_esfera_envolvente_m"], 70.659, places=2)

    def test_deterministic_and_never_overwrites(self):
        out2 = os.path.join(self.tmp, "ts2")
        res2 = S.generate(out2, CATALOG, F.CROSSING_REFERENCE)
        self.assertEqual(res2["sha256"], self.res["sha256"])
        self.assertEqual(res2["canon_fnv1a64"], self.res["canon_fnv1a64"])
        with self.assertRaises(FileExistsError):
            S.generate(out2, CATALOG, F.CROSSING_REFERENCE)

    def test_rejects_other_geometries(self):
        with self.assertRaises(SystemExit):
            S.generate(os.path.join(self.tmp, "x"), os.path.join(F.CATALOG_DIR, "geo-train-01.json"),
                       F.CROSSING_REFERENCE)


class CheckRunTests(unittest.TestCase):
    """T-S acceptance applied to fabricated muones logs."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="oe1syc_")
        cls.saved = S.N_PER_CLASS
        S.N_PER_CLASS = 3
        cls.res = S.generate(os.path.join(cls.tmp, "ts"), CATALOG, F.CROSSING_REFERENCE)
        cls.expected = C.read_json(cls.res["esperado"])

    @classmethod
    def tearDownClass(cls):
        S.N_PER_CLASS = cls.saved
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def fake_log(self, variant, tweak=None):
        consts = {"code": 6, "geometry_id": "geo-train-06", "variant": variant, "replica": 9,
                  "list_sha256": "cd" * 32}
        recs = []
        for k, e in enumerate(self.expected["muones"]):
            r = F.blank_record(k)
            r.update(status="DET", detected=1, iel=34, iaz=135, in_rint=1, plane=True, ekin=9000.0, el=8.0,
                     rel=1.0)
            if e["crossed_esperado"]:
                ln = e["lz_recta_m"] * 1.002
                r.update(deep=True, lz=ln, lz_true=ln, n_z=4, n_entries=1, mat_in="A1", mat_out="A1")
                if variant == "A":
                    r.update(lz_a0=0.3 * ln, lz_a1=0.7 * ln)
                else:
                    r.update(lz_a1=ln)
            recs.append(r)
        if tweak:
            tweak(recs)
        text = C.MUON_HEADER + "\n" + "".join(F.muon_row(consts, r, True, True) for r in recs)
        return C.parse_muones_bytes(text.encode("ascii"))

    def test_both_variants_pass(self):
        for variant in ("A", "B"):
            summary, rows = S.check_run(self.expected, self.fake_log(variant), variant)
            self.assertTrue(summary["ok"], summary)
            self.assertEqual(summary["clasificacion_correcta"], 9)
            self.assertEqual(summary["longitud_dentro_de_1pct"], 3)

    def test_wrong_classification(self):
        def miss_one_crosser(recs):
            for k, e in enumerate(self.expected["muones"]):
                if e["crossed_esperado"]:
                    recs[k].update(deep=False, lz=0.0, lz_a1=0.0, lz_a0=0.0)
                    return

        summary, _ = S.check_run(self.expected, self.fake_log("B", miss_one_crosser), "B")
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["clasificacion_correcta"], 8)

    def test_false_positive_on_a_near_miss(self):
        def spurious(recs):
            for k, e in enumerate(self.expected["muones"]):
                if e["clase"] == "casi_fallo":
                    recs[k].update(deep=True, lz=1.0, lz_a1=1.0)
                    return

        summary, _ = S.check_run(self.expected, self.fake_log("B", spurious), "B")
        self.assertFalse(summary["ok"])

    def test_length_out_of_tolerance(self):
        def stretch(recs):
            for k, e in enumerate(self.expected["muones"]):
                if e["crossed_esperado"]:
                    ln = e["lz_recta_m"] * 1.05
                    recs[k].update(lz=ln, lz_true=ln, lz_a1=ln)
                    return

        summary, rows = S.check_run(self.expected, self.fake_log("B", stretch), "B")
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["longitud_dentro_de_1pct"], 2)

    def test_a_wide_tolerance_produces_shallow_lines_that_the_check_accepts_inside_the_interval(self):
        res = S.generate(os.path.join(self.tmp, "wide"), CATALOG, F.CROSSING_REFERENCE, transverse_tol=1.0)
        wide = C.read_json(res["esperado"])
        through = [e for e in wide["muones"] if e["clase"] == "atraviesa"]
        self.assertTrue(any(not e["central"] for e in through))
        for e in through:
            self.assertLessEqual(e["lz_desplazada_min_m"], e["lz_recta_m"] + 1e-9)
            self.assertGreaterEqual(e["lz_desplazada_max_m"], e["lz_recta_m"] - 1e-9)
        shallow = next(e for e in through if not e["central"])
        # a length at the edge of the interval is accepted for the shallow line and would not be for a central one
        edge = 0.5 * (shallow["lz_desplazada_min_m"] + shallow["lz_recta_m"])
        self.assertLess(edge, 0.99 * shallow["lz_recta_m"])

        def fake(recs):
            recs[shallow["id"]].update(lz=edge, lz_true=edge, lz_a1=edge)

        consts = {"code": 6, "geometry_id": "geo-train-06", "variant": "B", "replica": 9, "list_sha256": "cd" * 32}
        recs = []
        for k, e in enumerate(wide["muones"]):
            r = F.blank_record(k)
            r.update(status="DET", detected=1, iel=34, iaz=135, in_rint=1, plane=True, ekin=9000.0, el=8.0, rel=1.0)
            if e["crossed_esperado"]:
                r.update(deep=True, lz=e["lz_recta_m"], lz_true=e["lz_recta_m"], lz_a1=e["lz_recta_m"], n_z=4,
                         n_entries=1, mat_in="A1", mat_out="A1")
            recs.append(r)
        fake(recs)
        text = C.MUON_HEADER + "\n" + "".join(F.muon_row(consts, r, True, True) for r in recs)
        mu = C.parse_muones_bytes(text.encode("ascii"))
        summary, rows = S.check_run(wide, mu, "B")
        self.assertTrue(summary["ok"], summary)
        self.assertFalse(rows[shallow["id"]]["central"])
        # the same log against the +-1 % rule alone (expected file without the interval) fails
        plain = copy.deepcopy(wide)
        for e in plain["muones"]:
            e.pop("lz_desplazada_min_m", None)
            e.pop("lz_desplazada_max_m", None)
            e.pop("central", None)
        self.assertFalse(S.check_run(plain, mu, "B")[0]["ok"])

    def test_shallow_lines_use_the_transverse_displacement_interval(self):
        exp = copy.deepcopy(self.expected)
        k = next(i for i, e in enumerate(exp["muones"]) if e["crossed_esperado"])
        ref = exp["muones"][k]["lz_recta_m"]
        exp["muones"][k].update(central=False, lz_desplazada_min_m=0.9 * ref, lz_desplazada_max_m=1.1 * ref)

        def stretch(factor):
            def fn(recs):
                recs[k].update(lz=ref * factor, lz_true=ref * factor, lz_a1=ref * factor)
            return fn

        summary, rows = S.check_run(exp, self.fake_log("B", stretch(1.08)), "B")
        self.assertTrue(summary["ok"], summary)
        self.assertEqual((summary["lineas_someras"], summary["someras_dentro_del_intervalo_transversal"]), (1, 1))
        self.assertFalse(rows[k]["central"])
        self.assertAlmostEqual(rows[k]["longitud_aceptable_max_m"], 1.1 * ref)
        # a line whose expected entry has no interval (or is central) keeps the +-1 % rule
        summary, _ = S.check_run(self.expected, self.fake_log("B", stretch(1.08)), "B")
        self.assertFalse(summary["ok"])
        # and the interval is a limit too
        summary, _ = S.check_run(exp, self.fake_log("B", stretch(1.2)), "B")
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["someras_dentro_del_intervalo_transversal"], 0)

    def test_variant_specific_material_rules(self):
        def a0_in_b(recs):
            for k, e in enumerate(self.expected["muones"]):
                if e["crossed_esperado"]:
                    ln = recs[k]["lz"]
                    recs[k].update(lz_a0=0.01, lz_a1=ln - 0.01)      # 1 cm of anomaly_0 in B
                    return

        summary, _ = S.check_run(self.expected, self.fake_log("B", a0_in_b), "B")
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["materiales_ok"], 8)

        def rock_in_a(recs):
            for k, e in enumerate(self.expected["muones"]):
                if e["crossed_esperado"]:
                    recs[k].update(lz_rock=0.5)
                    return

        summary, _ = S.check_run(self.expected, self.fake_log("A", rock_in_a), "A")
        self.assertFalse(summary["ok"])

    def test_wrong_number_of_rows(self):
        log = self.fake_log("B")
        cut = {k: v[:-1] for k, v in log.items()}
        summary, _ = S.check_run(self.expected, cut, "B")
        self.assertFalse(summary["ok"])
        self.assertTrue(summary["problemas"])

    def test_cli(self):
        consts_log = self.fake_log("B")
        text = C.MUON_HEADER + "\n"
        recs = []
        # rebuild the file text from the parsed columns is unnecessary: write it directly
        path = os.path.join(self.tmp, "muones.csv")
        consts = {"code": 6, "geometry_id": "geo-train-06", "variant": "B", "replica": 9, "list_sha256": "cd" * 32}
        for k, e in enumerate(self.expected["muones"]):
            r = F.blank_record(k)
            r.update(status="DET", detected=1, iel=34, iaz=135, in_rint=1, plane=True, ekin=9000.0, el=8.0, rel=1.0)
            if e["crossed_esperado"]:
                r.update(deep=True, lz=e["lz_recta_m"], lz_a1=e["lz_recta_m"])
            recs.append(r)
        with open(path, "wb") as fh:
            fh.write((text + "".join(F.muon_row(consts, r, True, True) for r in recs)).encode("ascii"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = S.main(["check", "--expected", self.res["esperado"], "--muones", path, "--variant", "B",
                         "--out", os.path.join(self.tmp, "ts-B.json")])
        self.assertEqual(rc, 0, out.getvalue())
        self.assertIn("T-S (B2)", out.getvalue())
        self.assertIn("OK", out.getvalue())
        js = C.read_json(os.path.join(self.tmp, "ts-B.json"))
        self.assertEqual((js["formato"], js["prueba"], js["ok"]), ("oe1-prueba-tecnica/1", "T-S (B2)", True))
        self.assertEqual(len(js["comprobaciones"]), 4)
        self.assertTrue(all(c["ok"] for c in js["comprobaciones"]))
        self.assertEqual(len(js["muones"]), 9)


if __name__ == "__main__":
    unittest.main()
