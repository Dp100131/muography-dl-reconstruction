"""Tests of oe1_stage1_roi: export, verify and the K-ROI orientation proof."""
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
import oe1_stage1_roi as R  # noqa: E402
import fixtures as F  # noqa: E402

REAL_NPZ = os.environ.get(F.REAL_NPZ_ENV) or os.path.abspath(os.path.join(
    HERE, *([os.pardir] * 4), "muografia-documentada", "objetivo-1", "entregas", "2026-09-29-geo-train-06",
    "analisis", "roi-mascaras.npz"))


def run_cli(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        rc = R.main(argv)
    return rc, out.getvalue()


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="oe1roi_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def npz(self, mask=None):
        d = tempfile.mkdtemp(dir=self.tmp)
        path = os.path.join(d, "roi-mascaras.npz")
        np.savez_compressed(path, R_int=F.make_test_mask() if mask is None else mask)
        return path

    def test_export_writes_text_hash_and_spec(self):
        npz = self.npz()
        out = os.path.join(tempfile.mkdtemp(dir=self.tmp), "roi")
        res = R.export_mask(npz, out, expect=None, copy_npz=True)
        with open(os.path.join(out, "R_int.mask.txt"), "rb") as fh:
            data = fh.read()
        self.assertEqual(len(data), 256 * 257)
        self.assertTrue(np.array_equal(C.mask_from_text(data), F.make_test_mask()))
        self.assertEqual(res["text_sha256"], C.sha256_bytes(data))
        with open(os.path.join(out, "R_int.mask.txt.sha256")) as fh:
            self.assertEqual(fh.read(), "%s  R_int.mask.txt\n" % C.sha256_bytes(data))
        spec = C.read_json(os.path.join(out, "R_int-congelada.json"))
        self.assertEqual(spec["texto"]["sha256"], C.sha256_bytes(data))
        self.assertEqual(spec["hash_contenido"]["valor"], C.mask_content_sha256(F.make_test_mask()))
        self.assertEqual(spec["forma"], [256, 256])
        self.assertEqual(spec["orientacion"]["indices"], "[iEl, iAz]")
        self.assertEqual(spec["origen"]["commit"], "712c786")
        self.assertEqual(C.sha256_file(os.path.join(out, "roi-mascaras.npz")), C.sha256_file(npz))
        # verify agrees, and detects any tampering
        chk = R.verify_mask(os.path.join(out, "R_int.mask.txt"), os.path.join(out, "R_int-congelada.json"),
                            os.path.join(out, "roi-mascaras.npz"), expect=None)
        self.assertTrue(all(ok for _, ok, _ in chk), [c for c in chk if not c[1]])
        bad = bytearray(data)
        bad[100] = ord("1") if bad[100] == ord("0") else ord("0")
        tampered = os.path.join(out, "tampered.txt")
        with open(tampered, "wb") as fh:
            fh.write(bytes(bad))
        chk = R.verify_mask(tampered, os.path.join(out, "R_int-congelada.json"), None, expect=None)
        self.assertFalse(all(ok for _, ok, _ in chk))

    def test_export_refuses_values_that_are_not_the_frozen_ones(self):
        with self.assertRaises(SystemExit) as cm:
            R.export_mask(self.npz(), os.path.join(tempfile.mkdtemp(dir=self.tmp), "roi"), expect=R.FROZEN)
        self.assertIn("no coincide", str(cm.exception))

    def test_the_text_mask_sha256_of_the_plan_is_hard_coded(self):
        """The exported text file is what the binary loads (--roi-mask-sha256): its SHA-256 is one of the values
        the export and verify commands assert, next to the npz hash and the hash of R_int.tobytes()."""
        self.assertEqual(C.FROZEN_RINT_TEXT_SHA256,
                         "e76dcbe5926f9f4aeb61c556729d15c64124cd50a8577dca6beee9ffa317c5fd")
        self.assertEqual(R.FROZEN["text_sha256"], C.FROZEN_RINT_TEXT_SHA256)

    def test_export_and_verify_assert_the_text_sha256(self):
        mask = F.make_test_mask()
        npz = self.npz(mask)
        text = C.mask_to_text(mask)
        with open(npz, "rb") as fh:
            raw = fh.read()
        summary = C.mask_summary(mask)
        expect = {"npz_sha256": C.sha256_bytes(raw), "content_sha256": C.mask_content_sha256(mask),
                  "shape": (256, 256), "pixels": summary["pixeles"], "rows": tuple(summary["filas"]),
                  "cols": tuple(summary["columnas"]), "text_sha256": "0" * 64}
        refused = os.path.join(tempfile.mkdtemp(dir=self.tmp), "roi")
        with self.assertRaises(SystemExit) as cm:
            R.export_mask(npz, refused, expect=expect)
        self.assertIn("sha256 congelado", str(cm.exception))
        self.assertFalse(os.path.exists(refused))                # nothing is written for a drifted export
        expect["text_sha256"] = C.sha256_bytes(text)
        out = os.path.join(tempfile.mkdtemp(dir=self.tmp), "roi")
        R.export_mask(npz, out, expect=expect)
        args = (os.path.join(out, "R_int.mask.txt"), os.path.join(out, "R_int-congelada.json"), None)
        chk = R.verify_mask(*args, expect=expect)
        self.assertTrue(all(ok for _, ok, _ in chk), [c for c in chk if not c[1]])
        self.assertTrue(any("sha256 del texto == valor congelado del plan" in name for name, _, _ in chk))
        expect["text_sha256"] = "1" * 64
        chk = R.verify_mask(*args, expect=expect)
        failed = [name for name, ok, _ in chk if not ok]
        self.assertEqual(failed, ["sha256 del texto == valor congelado del plan"])

    def test_export_never_overwrites(self):
        npz = self.npz()
        out = os.path.join(tempfile.mkdtemp(dir=self.tmp), "roi")
        R.export_mask(npz, out, expect=None)
        with self.assertRaises(FileExistsError):
            R.export_mask(npz, out, expect=None)

    def test_verify_cli_exit_code(self):
        npz = self.npz()
        out = os.path.join(tempfile.mkdtemp(dir=self.tmp), "roi")
        R.export_mask(npz, out, expect=None)
        rc, _ = run_cli(["verify", "--mask", os.path.join(out, "R_int.mask.txt"),
                         "--frozen", os.path.join(out, "R_int-congelada.json"), "--no-plan-check"])
        self.assertEqual(rc, 0)
        rc, _ = run_cli(["verify", "--mask", os.path.join(out, "R_int.mask.txt"),
                         "--frozen", os.path.join(out, "R_int-congelada.json")])   # the plan's fingerprints
        self.assertEqual(rc, 3)


class MissingInputTests(unittest.TestCase):
    def test_every_subcommand_exits_2_with_one_line(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        none = os.path.join(tmp, "none")
        for argv in (["export", "--npz", none + ".npz", "--out-dir", os.path.join(tmp, "o")],
                     ["verify", "--mask", none + ".txt", "--frozen", none + ".json"],
                     ["kroi", "--npz", none + ".npz", "--out-dir", os.path.join(tmp, "k")]):
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                rc = R.main(argv)
            self.assertEqual(rc, 2, argv)
            self.assertEqual(len(err.getvalue().strip().splitlines()), 1, err.getvalue())


class KroiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="oe1kroi_")
        cls.cr, _, _ = C.load_crossing_reference(F.CROSSING_REFERENCE)
        segs, _ = R.rays_from_detector(0)
        cls.lz = np.asarray(cls.cr.model(segs)["len"]).reshape(256, 256)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def npz(self, rint, l_int=None):
        d = tempfile.mkdtemp(dir=self.tmp)
        path = os.path.join(d, "roi-mascaras.npz")
        arrays = {"R_int": rint}
        if l_int is not None:
            arrays["l_int"] = l_int.astype(np.float32)
        np.savez_compressed(path, **arrays)
        return path

    def test_map_geometry(self):
        lz = self.lz
        self.assertGreater(lz.max(), 40.0)
        rows, cols = np.nonzero(lz > 0)
        # the shadow of Z sits at positive relative azimuth, columns ~130-139, rows ~17-52
        self.assertGreaterEqual(cols.min(), 128)
        self.assertLessEqual(cols.max(), 140)
        self.assertGreaterEqual(rows.min(), 15)
        self.assertLessEqual(rows.max(), 55)

    def test_consistent_mask_passes(self):
        rint = self.lz >= R.KROI_L_INT_THRESHOLD_M
        res, _ = R.kroi_compute(self.cr, rint, self.lz, n_points=2000)
        self.assertEqual(res["veredicto"], "OK", res["comprobaciones"])
        self.assertEqual(res["centroide"]["pixel"], [34, 135])
        self.assertTrue(res["centroide"]["en_R_int"])
        self.assertFalse(res["centroide"]["espejo_en_R_int"])
        self.assertEqual(res["centroide"]["pixel_espejo"], [34, 120])
        self.assertGreater(res["solape"]["masa_LZ_en_R_int"], 0.9)
        self.assertEqual(res["solape"]["masa_LZ_en_espejo_azimut"], 0.0)
        self.assertGreater(res["puntos_de_Z"]["fraccion_en_R_int"], 0.8)
        self.assertEqual(res["puntos_de_Z"]["fraccion_en_espejo"], 0.0)

    def test_mirrored_mask_is_rejected(self):
        rint = (self.lz >= R.KROI_L_INT_THRESHOLD_M)[:, ::-1]
        res, _ = R.kroi_compute(self.cr, rint, None, n_points=500)
        self.assertEqual(res["veredicto"], "CONTRADICE LA MASCARA")
        self.assertFalse(res["centroide"]["en_R_int"])
        self.assertTrue(res["centroide"]["espejo_en_R_int"])

    def test_transposed_or_elevation_flipped_mask_is_rejected(self):
        rint = (self.lz >= R.KROI_L_INT_THRESHOLD_M)[::-1, :]
        res, _ = R.kroi_compute(self.cr, rint, None, n_points=500)
        self.assertEqual(res["veredicto"], "CONTRADICE LA MASCARA")

    def test_cli_exit_codes_and_outputs(self):
        good = self.npz(self.lz >= R.KROI_L_INT_THRESHOLD_M, self.lz)
        out = os.path.join(self.tmp, "k-ok")
        rc, text = run_cli(["kroi", "--npz", good, "--out-dir", out, "--crossing-reference", F.CROSSING_REFERENCE,
                            "--no-figure"])
        self.assertEqual(rc, 0, text)
        js = C.read_json(os.path.join(out, "kroi.json"))
        self.assertEqual(js["veredicto"], "OK")
        self.assertIn("sha256", js["entradas"]["crossing_reference"])
        bad = self.npz((self.lz >= R.KROI_L_INT_THRESHOLD_M)[:, ::-1], self.lz)
        rc, text = run_cli(["kroi", "--npz", bad, "--out-dir", os.path.join(self.tmp, "k-bad"),
                            "--crossing-reference", F.CROSSING_REFERENCE, "--no-figure"])
        self.assertEqual(rc, 3)


@unittest.skipUnless(os.path.isfile(REAL_NPZ), "the pilot's roi-mascaras.npz is not available here")
class RealMaskTests(unittest.TestCase):
    """Runs against the real frozen file (read-only)."""

    def test_real_export_matches_the_plan_and_the_binary_hash(self):
        tmp = tempfile.mkdtemp(prefix="oe1real_")
        try:
            res = R.export_mask(REAL_NPZ, os.path.join(tmp, "roi"), expect=R.FROZEN)
            # the same file the C++ implementer exported and loaded with --roi-mask
            self.assertEqual(res["text_sha256"],
                             "e76dcbe5926f9f4aeb61c556729d15c64124cd50a8577dca6beee9ffa317c5fd")
            self.assertEqual(res["npz_sha256"], C.FROZEN_NPZ_SHA256)
            self.assertEqual(res["content_sha256"], C.FROZEN_RINT_CONTENT_SHA256)
            self.assertEqual(res["summary"]["pixeles"], 195)
            self.assertEqual(res["summary"]["filas"], [17, 50])
            self.assertEqual(res["summary"]["columnas"], [131, 138])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_real_kroi(self):
        cr, _, _ = C.load_crossing_reference(F.CROSSING_REFERENCE)
        raw, arrays = R._load_npz_bytes(REAL_NPZ)
        res, _ = R.kroi_compute(cr, arrays["R_int"], arrays["l_int"], n_points=3000)
        self.assertEqual(res["veredicto"], "OK")
        self.assertEqual(res["centroide"]["pixel"], [34, 135])
        self.assertTrue(res["pixel_esperado_34_135"])
        self.assertGreater(res["solape"]["masa_LZ_en_R_int"], 0.95)
        self.assertEqual(res["solape"]["masa_LZ_en_espejo_azimut"], 0.0)
        self.assertEqual(res["solape"]["pixeles_LZ_positiva_en_R_int"], 195)
        self.assertLess(res["comparacion_con_l_int_del_piloto"]["diferencia_maxima_m"], 2.5)
        self.assertEqual(res["comparacion_con_l_int_del_piloto"]["pixeles_solo_uno_positivo"], 0)


if __name__ == "__main__":
    unittest.main()
