"""Tests of the technical-test evaluators (T-L, T-R, T-M, T-P, T-F)."""
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

import oe1_stage1_analysis as A  # noqa: E402
import oe1_stage1_common as C  # noqa: E402
import oe1_stage1_tecnicas as T  # noqa: E402
import fixtures as F  # noqa: E402

MASK = F.make_test_mask()
K = 25
N = 120


def write_legacy_run(directory, wall=1.0, seed=2026092900, extra_meta=None, hist_bump=False, events=2000):
    os.makedirs(directory)
    hist = np.zeros((256, 256), dtype=np.int64)
    hist[30:34, 100:104] = 3
    if hist_bump:
        hist[5, 5] += 1
    with open(os.path.join(directory, "geo-train-06_view0_hist.csv"), "wb") as fh:
        fh.write(("\n".join(",".join(map(str, row)) for row in hist.tolist()) + "\n").encode())
    meta = {"engine": "geant4-test", "geometry_id": "geo-train-06", "view_index": 0, "seed": seed, "threads": 1,
            "events_transported": events, "ecomug_draws_total": events, "detected_in_bins": int(hist.sum()),
            "mean_ekin_detected_gev": 12.5, "wall_seconds": wall, "seconds_per_event": wall / events,
            "exposure_days": -1.0}
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(directory, "geo-train-06_view0.meta.json"), "wb") as fh:
        fh.write((json.dumps(meta, indent=1, sort_keys=True) + "\n").encode())
    with open(os.path.join(directory, "run.log"), "wb") as fh:
        fh.write(("RUN DONE geo-train-06 view 0: events=%d detected=%d wall_s=%r\n" %
                  (events, hist.sum(), wall)).encode())
    return directory


def write_list_run(directory, recs, level="full", master=1026092990, transport=1026092990, list_sha="ab" * 32,
                   wall=1.0, tweak_rows=None):
    os.makedirs(directory)
    consts = {"code": 6, "geometry_id": "geo-train-06", "variant": "B", "replica": 0, "list_sha256": list_sha}
    full = level == "full"
    text = C.MUON_HEADER + "\n" + "".join(F.muon_row(consts, r, full, True) for r in recs)
    data = text.encode("ascii")
    stem = os.path.join(directory, "geo-train-06_view0")
    with open(stem + "_muones.csv", "wb") as fh:
        fh.write(data)
    hist = np.zeros((256, 256), dtype=np.int64)
    for r in recs:
        if r["status"] == "DET":
            hist[r["iel"], r["iaz"]] += 1
    with open(stem + "_hist.csv", "wb") as fh:
        fh.write(("\n".join(",".join(map(str, row)) for row in hist.tolist()) + "\n").encode())
    meta = {"seed": master, "threads": 1, "geometry_id": "geo-train-06", "events_transported": len(recs),
            "wall_seconds": wall, "seconds_per_event": wall / len(recs),
            "oe1_etapa1": {"muon_log": {"level": level, "rows": len(recs)},
                           "seeds": {"master_cli": master, "master_applied": master % 2147483647,
                                     "transport_seed": transport}}}
    with open(stem + ".meta.json", "wb") as fh:
        fh.write((json.dumps(meta, indent=1, sort_keys=True) + "\n").encode())
    with open(os.path.join(directory, "run.log"), "wb") as fh:
        fh.write(b"RUN DONE\n")
    return directory


def records(seed=3, n=N):
    return F.base_records(n, "geo-train-06", 0, seed, MASK)


def run_cli(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = T.main(argv)
    return rc, out.getvalue()


class TechnicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="oe1tec_")
        cls.n = [0]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def d(self, name):
        self.n[0] += 1
        return os.path.join(self.tmp, "%s%03d" % (name, self.n[0]))

    def failing(self, res):
        return [c["descripcion"] for c in res.checks if c["ok"] is False]

    # ------------------------------------------------------------------ T-L
    def test_legacy_identical(self):
        a = write_legacy_run(self.d("pil"), wall=1.0)
        b = write_legacy_run(self.d("new"), wall=1.3)          # timing differs: allowed
        res = T.test_legacy(a, b)
        self.assertTrue(res.ok, self.failing(res))

    def test_legacy_label_names_the_variant(self):
        a = write_legacy_run(self.d("pil"))
        b = write_legacy_run(self.d("new"))
        self.assertEqual(T.test_legacy(a, b, "A").name, "T-L (A)")
        self.assertEqual(T.test_legacy(a, b).name, "T-L")

    def test_legacy_differences(self):
        a = write_legacy_run(self.d("pil"))
        res = T.test_legacy(a, write_legacy_run(self.d("new"), hist_bump=True))
        self.assertFalse(res.ok)
        self.assertTrue(any("hist.csv" in f for f in self.failing(res)))
        res = T.test_legacy(a, write_legacy_run(self.d("new"), extra_meta={"threads": 8}))
        self.assertTrue(any("meta.json igual" in f for f in self.failing(res)))
        res = T.test_legacy(a, write_legacy_run(self.d("new"), extra_meta={"oe1_etapa1": {}}))
        self.assertTrue(any("oe1_etapa1" in f for f in self.failing(res)))

    # ------------------------------------------------------------------ T-R
    def test_full_versus_min(self):
        recs = records()
        full = write_list_run(self.d("full"), recs, "full")
        mini = write_list_run(self.d("min"), recs, "min")
        res = T.test_full_vs_min(full, mini)
        self.assertTrue(res.ok, self.failing(res))

    def test_full_versus_min_detects_a_changed_rng1(self):
        recs = records()
        other = copy.deepcopy(recs)
        other[40]["rng1"] ^= 5
        res = T.test_full_vs_min(write_list_run(self.d("full"), recs, "full"),
                                 write_list_run(self.d("min"), other, "min"))
        self.assertFalse(res.ok)
        self.assertTrue(any("rng1 igual" in f for f in self.failing(res)))
        # and a histogram difference
        other = copy.deepcopy(recs)
        for r in other:
            if r["status"] == "DET":
                r["iel"] = (r["iel"] + 1) % 256
                break
        res = T.test_full_vs_min(write_list_run(self.d("full"), recs, "full"),
                                 write_list_run(self.d("min"), other, "min"))
        self.assertTrue(any("hist.csv" in f for f in self.failing(res)))

    def test_min_level_must_declare_itself(self):
        recs = records()
        res = T.test_full_vs_min(write_list_run(self.d("full"), recs, "full"),
                                 write_list_run(self.d("min"), recs, "full"))
        self.assertTrue(any("level = min" in f for f in self.failing(res)))

    # ------------------------------------------------------------------ T-M
    def test_master_seed_independence(self):
        recs = records()
        ref = write_list_run(self.d("ref"), recs, master=1026092990)
        shifted = write_list_run(self.d("sh"), recs, master=1026092995)
        res = T.test_master_seed(ref, shifted)
        self.assertTrue(res.ok, self.failing(res))

    def test_master_seed_dependence_is_detected(self):
        recs = records()
        changed = copy.deepcopy(recs)
        changed[7]["rng0"] ^= 1
        res = T.test_master_seed(write_list_run(self.d("ref"), recs, master=1026092990),
                                 write_list_run(self.d("sh"), changed, master=1026092995))
        self.assertFalse(res.ok)
        self.assertTrue(any("rng0" in f for f in self.failing(res)))
        # same master seed: the test itself is void
        res = T.test_master_seed(write_list_run(self.d("ref"), recs, master=5),
                                 write_list_run(self.d("sh"), recs, master=5))
        self.assertTrue(any("distintas" in f for f in self.failing(res)))
        # a meta difference other than the seed
        res = T.test_master_seed(write_list_run(self.d("ref"), recs, master=1, transport=1),
                                 write_list_run(self.d("sh"), recs, master=2, transport=9))
        self.assertTrue(any("meta.json igual" in f for f in self.failing(res)))

    # ------------------------------------------------------------------ T-P
    def test_prefix_replacement(self):
        base = records(seed=3)
        other = records(seed=4)
        derived = copy.deepcopy(base)
        for i in range(K):
            derived[i] = copy.deepcopy(other[i])
            derived[i]["prim_fnv"] ^= 0xFF
            derived[i]["id"] = i
        a = write_list_run(self.d("base"), base, list_sha="aa" * 32)
        b = write_list_run(self.d("der"), derived, list_sha="bb" * 32)        # another list, hence another sha
        res = T.test_prefix_replacement(a, b, K)
        self.assertTrue(res.ok, self.failing(res))

    def test_prefix_replacement_detects_dependence_on_previous_muons(self):
        base = records(seed=3)
        derived = copy.deepcopy(base)
        for i in range(K):
            derived[i]["prim_fnv"] ^= 0xFF
        derived[60]["rng1"] ^= 3                     # a muon after the replaced block changed its stream
        res = T.test_prefix_replacement(write_list_run(self.d("base"), base),
                                        write_list_run(self.d("der"), derived), K)
        self.assertFalse(res.ok)
        self.assertTrue(any("rng1 igual en los ids" in f for f in self.failing(res)))

    def test_prefix_replacement_must_have_replaced_something(self):
        base = records(seed=3)
        res = T.test_prefix_replacement(write_list_run(self.d("base"), base),
                                        write_list_run(self.d("der"), copy.deepcopy(base)), K)
        self.assertTrue(any("se aplic" in f for f in self.failing(res)))

    # ------------------------------------------------------------------ T-F
    def analysis_dir(self, tweak=None, name="tf"):
        fx = F.build_attempt(self.d(name), n=N, tweak=tweak, replicas=(0,))
        out = self.d("analysis")
        argv = ["--attempt", fx["attempt"], "--roi-mask", fx["mask"]["text"], "--roi-frozen", fx["mask"]["frozen"],
                "--roi-npz", fx["mask"]["npz"], "--out", out, "--test-mask", "--replicas", "0", "--no-figures",
                "--crossing-reference", F.CROSSING_REFERENCE]
        with contextlib.redirect_stdout(io.StringIO()):
            rc = A.main(argv)
        return rc, out

    def test_functional_rehearsal_passes(self):
        rc, out = self.analysis_dir()
        self.assertEqual(rc, 0)
        res = T.test_functional(out, expected_runs=5)
        self.assertTrue(res.ok, self.failing(res))

    def test_functional_rehearsal_flags_unexpected_statuses(self):
        def tweak(recs, entries):
            r = recs["g06-B-r0"][12]
            r.update(status="STOPPED", detected=0, iel=-1, iaz=-1, in_rint=-1, plane=False)
            r = recs["g06-A-r0"][12]
            r.update(status="STOPPED", detected=0, iel=-1, iaz=-1, in_rint=-1, plane=False)

        rc, out = self.analysis_dir(tweak, "tf2")
        self.assertEqual(rc, 0)
        res = T.test_functional(out, expected_runs=5)
        self.assertFalse(res.ok)
        self.assertTrue(any("STOPPED" in f for f in self.failing(res)))

    def test_functional_rehearsal_fails_when_contracts_block(self):
        def tweak(recs, entries):
            recs["g01-A-r0-rep"][3]["rng1"] ^= 1

        rc, out = self.analysis_dir(tweak, "tf3")
        self.assertEqual(rc, 3)
        res = T.test_functional(out, expected_runs=5)
        self.assertFalse(res.ok)
        self.assertTrue(any("sin errores" in f for f in self.failing(res)))

    def test_functional_without_analysis_output(self):
        res = T.test_functional(os.path.join(self.tmp, "nada"))
        self.assertFalse(res.ok)

    # ------------------------------------------------------------------ CLI and summary
    def test_cli_writes_json_and_summary(self):
        recs = records()
        full = write_list_run(self.d("full"), recs, "full")
        mini = write_list_run(self.d("min"), recs, "min")
        out1 = os.path.join(self.tmp, "t-r.json")
        rc, text = run_cli(["t-r", "--full", full, "--min", mini, "--out", out1])
        self.assertEqual(rc, 0)
        self.assertIn("T-R: OK", text)
        js = C.read_json(out1)
        self.assertEqual(js["prueba"], "T-R")
        self.assertIn("prueba técnica", js["etiqueta"])
        self.assertTrue(js["entradas"]["full"]["hist_sha256"])
        a = write_legacy_run(self.d("pil"))
        b = write_legacy_run(self.d("new"), hist_bump=True)
        out2 = os.path.join(self.tmp, "t-l.json")
        rc, text = run_cli(["t-l", "--pilot", a, "--new", b, "--label", "B", "--out", out2])
        self.assertEqual(rc, 3)
        rc, text = run_cli(["resumen", out1, out2, "--out", os.path.join(self.tmp, "resumen.json")])
        self.assertEqual(rc, 3)
        self.assertIn("PRUEBAS TÉCNICAS: FALLA", text)
        summary = C.read_json(os.path.join(self.tmp, "resumen.json"))
        self.assertEqual([p["ok"] for p in summary["pruebas"]], [True, False])
        self.assertEqual([p["prueba"] for p in summary["pruebas"]], ["T-R", "T-L (B)"])
        rc, text = run_cli(["resumen", out1])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
