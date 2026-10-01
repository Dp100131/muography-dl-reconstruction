"""Tests of the R2-01 complement tool: straight-line sides and the A/B comparison of the six runs."""
import contextlib
import copy
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
import oe1_r201_compare as R  # noqa: E402
import fixtures as F  # noqa: E402

CATALOG = os.path.join(F.CATALOG_DIR, "geo-train-06.json")


def r201_base(mask, fnv, seed):
    """Twelve detected Z crossers with the primaries of the list (prim_fnv) and their own rng0 per seed."""
    recs = F.base_records(R.N_MUONS, R.CASE, R.REPLICA, seed, mask)
    pix = np.argwhere(mask)
    for i, r in enumerate(recs):
        iel, iaz = (int(x) for x in pix[i % len(pix)])
        lz = 25.0 + i
        r.update(status="DET", detected=1, iel=iel, iaz=iaz, in_rint=int(mask[iel, iaz]), plane=True,
                 ekin=3000.0 + i, el=C.EL_MIN_DEG + iel * (C.EL_MAX_DEG - C.EL_MIN_DEG) / (C.N_EL - 1),
                 rel=-C.AZ_HALF_DEG + iaz * 2 * C.AZ_HALF_DEG / (C.N_AZ - 1), end_vol="detector",
                 end_process="Transportation", deep=True, lz=lz, lz_true=lz * 1.0001, lc=40.0, le=60.0,
                 rho=lz * 100 * 3.285, n_z=3, n_z_bo=0, n_entries=1, have_ein=True, have_eout=True,
                 e_in=(5000.0, 4999.0, 5001.0), e_out=(4990.0, 4989.0, 4991.0))
        r["prim_fnv"] = int(fnv[i])
    return recs


def variant_records(base, variant):
    """A navigates the conduit side (ids 0-5) inside anomaly_0; B assigns the whole of Z to anomaly_1."""
    recs = copy.deepcopy(base)
    for r in recs:
        lz = r["lz"]
        if variant == "A" and r["id"] < 6:
            r.update(mat_in="A0", mat_out="A0", lz_a0=lz, lz_a1=0.0, e_out=(4995.0, 4994.0, 4996.0))
        else:
            r.update(mat_in="A1", mat_out="A1", lz_a0=0.0, lz_a1=lz)
    return recs


class R201Fixture:
    """Attempt folder written like ejecutar_complemento_r2_01.sh, inputs folder and mask."""

    def __init__(self, root, tweak=None, estado=None, check_b_extra=""):
        self.root = root
        self.attempt = os.path.join(root, "intento-01")
        self.inputs = os.path.join(root, "complemento-r2-01-v1")
        tables = C.load_tables(F.TABLES_DIR)
        ids, vals, pdg, charge = F.sample_source_model(np.random.default_rng(7), R.N_MUONS, tables, 509.7)
        info = F.write_list(self.inputs, R.LIST_NAME, ids, vals, pdg, charge, R.CASE, R.CASE_CODE, R.REPLICA,
                            None, 509.7, extra={"kind": "sintetico", "synthetic": True})
        self.list_sha = info["sha256"]
        canon_hex = C.hex16(info["canon"])
        expected = {"muones": [{"id": i, "lado_entrada_recta": "desde_conducto" if i < 6 else "desde_elipsoide",
                                "longitud_recta_Z_m": 25.0 + i} for i in range(R.N_MUONS)]}
        with open(os.path.join(self.inputs, R.EXPECTED_NAME), "wb") as fh:
            fh.write(json.dumps(expected).encode())
        self.mask = F.write_mask_files(os.path.join(root, "roi"))
        fnv = C.primary_fnv_array(vals, pdg)
        lists = {(R.CASE, R.REPLICA): {"canon_hex": canon_hex}}
        for s in R.SEEDS:
            base = r201_base(self.mask["masks"]["R_int"], fnv, s % 1000)
            for v in R.VARIANTS:
                recs = variant_records(base, v)
                if tweak:
                    tweak(v, s, recs)
                entry = R.make_entry(v, s, self.list_sha)
                F.write_run(self.attempt, entry, recs, lists, self.mask, lateral_columns=True)
        rows = ["variante\tsemilla_transporte\tprimarios\testado"]
        for s in R.SEEDS:
            for v in R.VARIANTS:
                rows.append("%s\t%d\t12\t%s" % (v, s, (estado or {}).get((v, s), "PROCESO_OK")))
        self._write("estado.tsv", "\n".join(rows) + "\n")
        self._write("FINALIZADO.txt", "6 procesos terminados; resultados fisicos pendientes de revision.\n")
        line = ("OE1-CHECK-PRIMARIES file=%s n=12 lf_only=1 header_ok=1 ids_contiguous=1 dup_ids=0 "
                "canon_fnv1a64=%s file_sha256=%s sidecar=ok status=OK\n" % (R.LIST_NAME, canon_hex, self.list_sha))
        self._write("check-A2.txt", line)
        self._write("check-B2.txt", line + check_b_extra)

    def _write(self, name, text):
        with open(os.path.join(self.attempt, name), "wb") as fh:
            fh.write(text.encode("utf-8"))

    def compare(self):
        out = os.path.join(self.root, "analisis")
        with contextlib.redirect_stdout(io.StringIO()):
            rc = R.main(["compare", "--attempt", self.attempt, "--inputs", self.inputs, "--mask",
                         self.mask["text"], "--out", out])
        with open(os.path.join(out, "resumen-r201.json"), "rb") as fh:
            summary = json.loads(fh.read().decode("utf-8"))
        with open(os.path.join(out, "contratos-r201.json"), "rb") as fh:
            contracts = json.loads(fh.read().decode("utf-8"))
        with open(os.path.join(out, "pares-r201.csv"), "rb") as fh:
            pairs = list(__import__("csv").DictReader(io.StringIO(fh.read().decode("utf-8"))))
        return rc, summary, contracts, pairs


class LineSideTests(unittest.TestCase):
    def setUp(self):
        self.cr = C.load_crossing_reference()[0]
        self.cyl, self.ell = C.region_from_catalog(C.read_catalog(CATALOG))

    def test_entry_side_follows_the_primitive_entered_last(self):
        # at y = -16 m, z = 125 m a line along +x enters the conduit (x ~ 1.1 m) before the ellipsoid (x ~ 12.8 m)
        rep = R.line_report(self.cr, self.cyl, self.ell, (-100.0, -16.0, 125.0), (100.0, -16.0, 125.0))
        self.assertEqual(rep["lado"], R.SIDE_CONDUIT)
        self.assertTrue(rep["profundo"])
        self.assertAlmostEqual(rep["lz_m"], rep["salida_Z_m"][0] - rep["entrada_Z_m"][0], places=9)
        self.assertEqual(rep["anillo"]["conservan_lado"], rep["anillo"]["rectas"])
        # at y = -35 m the same line is inside the dense body (x ~ -22 m) before it reaches the conduit (x ~ 2.1 m)
        rep = R.line_report(self.cr, self.cyl, self.ell, (-100.0, -35.0, 125.0), (100.0, -35.0, 125.0))
        self.assertEqual(rep["lado"], R.SIDE_BODY)
        # the reverse direction swaps the side of the first line
        rep = R.line_report(self.cr, self.cyl, self.ell, (100.0, -16.0, 125.0), (-100.0, -16.0, 125.0))
        self.assertEqual(rep["lado"], R.SIDE_BODY)
        rep = R.line_report(self.cr, self.cyl, self.ell, (-100.0, 30.0, 125.0), (100.0, 30.0, 125.0))
        self.assertEqual(rep["lado"], "sin_cruce")
        self.assertEqual(rep["lz_m"], 0.0)


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="r201-")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_clean_attempt_passes_and_reports_both_sides(self):
        rc, summary, contracts, pairs = R201Fixture(self.tmp).compare()
        self.assertEqual(rc, R.EXIT_OK, contracts["bloqueantes_fallidos"])
        self.assertEqual(contracts["estado"], "OK")
        self.assertEqual(len(pairs), 36)
        conduit, body = summary["cobertura"][R.SIDE_CONDUIT], summary["cobertura"][R.SIDE_BODY]
        self.assertEqual(conduit["ids_con_lado_en_A_igual_al_planificado_en_las_3_semillas"], 6)
        self.assertEqual(body["ids_con_lado_en_A_igual_al_planificado_en_las_3_semillas"], 6)
        self.assertEqual(conduit["pares_con_material_de_B_correcto"], 18)
        self.assertEqual(conduit["transiciones"], {"11": 18})
        self.assertEqual(conduit["pares_detectados_en_ambas_con_mismo_pixel"], 18)
        row = [p for p in pairs if p["id"] == "0"][0]
        self.assertEqual(float(row["delta_lz_a0_m"]), -25.0)
        self.assertEqual(float(row["delta_e_out_gev"]), -5.0)

    def test_b_material_outside_a1_is_a_result_not_a_block(self):
        def tweak(v, s, recs):
            if v == "B" and s == R.SEEDS[0]:
                recs[2].update(lz_a0=0.5, lz_a1=recs[2]["lz"] - 0.5)
        rc, summary, contracts, pairs = R201Fixture(self.tmp, tweak=tweak).compare()
        self.assertEqual(rc, R.EXIT_OK)
        self.assertEqual(summary["cobertura"][R.SIDE_CONDUIT]["pares_con_material_de_B_correcto"], 17)
        bad = [p for p in pairs if p["B_material_ok"] == "0"]
        self.assertEqual([(p["id"], p["semilla"]) for p in bad], [("2", str(R.SEEDS[0]))])
        self.assertEqual(float(bad[0]["B_fuera_de_A1_m"]), 0.5)

    def test_conduit_id_navigated_as_dense_body_in_a_is_reported(self):
        def tweak(v, s, recs):
            if v == "A":
                recs[0].update(mat_in="A1", mat_out="A1", lz_a0=0.0, lz_a1=recs[0]["lz"])
        rc, summary, contracts, pairs = R201Fixture(self.tmp, tweak=tweak).compare()
        self.assertEqual(rc, R.EXIT_OK)
        self.assertEqual(summary["cobertura"][R.SIDE_CONDUIT]
                         ["ids_con_lado_en_A_igual_al_planificado_en_las_3_semillas"], 5)
        self.assertTrue(all(p["lado_observado_A"] == R.SIDE_BODY for p in pairs if p["id"] == "0"))

    def test_different_primary_between_a_and_b_blocks(self):
        def tweak(v, s, recs):
            if v == "B" and s == R.SEEDS[1]:
                recs[3]["prim_fnv"] ^= 0x1
        rc, summary, contracts, pairs = R201Fixture(self.tmp, tweak=tweak).compare()
        self.assertEqual(rc, R.EXIT_BLOCKED)
        self.assertEqual(summary["estado"], R.A.STAMP)
        self.assertTrue(any("prim_fnv" in x for x in contracts["bloqueantes_fallidos"]))

    def test_failed_process_or_different_list_check_blocks(self):
        rc, _, contracts, _ = R201Fixture(self.tmp, estado={("B", R.SEEDS[2]): "FALLO_4"}).compare()
        self.assertEqual(rc, R.EXIT_BLOCKED)
        self.assertTrue(any("estado.tsv" in x for x in contracts["bloqueantes_fallidos"]))
        shutil.rmtree(self.tmp)
        os.makedirs(self.tmp)
        rc, _, contracts, _ = R201Fixture(self.tmp, check_b_extra="otra linea\n").compare()
        self.assertEqual(rc, R.EXIT_BLOCKED)
        self.assertTrue(any("check-A2.txt" in x for x in contracts["bloqueantes_fallidos"]))


if __name__ == "__main__":
    unittest.main()
