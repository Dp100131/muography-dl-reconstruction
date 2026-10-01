"""Tests of the OE1 stage-2 tool: manifest plan, per-block analysis, controls and aggregation.

The protocol (section 8, item 4) asks for tests with null counts, discordances, repeated or missing ids and a
wrong histogram before any transport. The block and control tests use the stage-1 fixture attempt
(fixtures.build_attempt), whose files have the exact formats of the binary.
"""
import contextlib
import copy
import csv
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
import oe1_stage2_analysis as S2  # noqa: E402
import fixtures as F  # noqa: E402

CALCULATOR_ENV = "OE1_STAGE2_CALCULATOR"


def corridas_rows():
    """The 55 rows of the protocol's manifest, built like herramientas/planificar_etapa2_oe1.py does."""
    rows = []
    for case, count, events, base, repbase in (("geo-train-06", 24, 1000000, 2026100100, 100),
                                               ("geo-train-01", 3, 100000, 2026100200, 200)):
        for i in range(1, count + 1):
            for variant in ("A", "B"):
                rows.append({"run_id": "%s-r%d-%s" % (case, repbase + i, variant), "geometry_id": case,
                             "case_code": int(case[-2:]), "view": 0, "replica": repbase + i, "n": events,
                             "source_seed": base + i, "transport_seed": base + i - 1000000000, "variant": variant,
                             "repeat_of": ""})
    aa = dict(next(r for r in rows if r["run_id"] == "geo-train-01-r201-A"))
    aa.update(run_id="geo-train-01-r201-AA", repeat_of="geo-train-01-r201-A")
    rows.append(aa)
    return rows


def write_corridas(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=S2.CORRIDAS_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def quiet(fn, argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(argv)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="s2plan-")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_protocol_manifest_passes_and_a_changed_seed_fails(self):
        rows = corridas_rows()
        probs, totals = S2.check_corridas(rows)
        self.assertEqual(probs, [])
        self.assertEqual((totals["corridas"], totals["transportes"], totals["pares_principales"], totals["listas"]),
                         (55, 48700000, 24000000, 27))
        bad = copy.deepcopy(rows)
        bad[3]["transport_seed"] += 1
        self.assertTrue(S2.check_corridas(bad)[0])
        self.assertTrue(S2.check_corridas(rows[:-1])[0])

    def test_job_manifest_and_dry_run_commands(self):
        corr = os.path.join(self.tmp, "corridas.csv")
        write_corridas(corr, corridas_rows())
        names = [S2.list_name("geo-train-01", r, 2026100000 + r) for r in (201, 202, 203)]
        sha = os.path.join(self.tmp, "listas.sha256")
        with open(sha, "w", encoding="utf-8") as fh:
            fh.write("".join("%064x  %s\n" % (k + 1, n) for k, n in enumerate(names)))
        tsv = os.path.join(self.tmp, "m.tsv")
        lt = os.path.join(self.tmp, "l.tsv")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = S2.main(["plan", "--corridas", corr, "--replicas", "201,202,203", "--manifest-tsv", tsv,
                          "--lists-tsv", lt, "--list-sha", sha, "--lists-dir", "/tmp/l", "--catalog-dir", "/c",
                          "--bin-a", "/a", "--bin-b", "/b", "--print-commands"])
        self.assertEqual(rc, S2.EXIT_OK)
        import oe1_stage1_analysis as A
        entries = A.read_manifest(tsv)
        self.assertEqual(len(entries), 7)
        aa = [e for e in entries if e["rol"] == A.ROLE_AA]
        self.assertEqual(len(aa), 1)
        self.assertEqual((aa[0]["variante"], aa[0]["binario"], aa[0]["salida"]), ("A", "/a", "geo-train-01/r201/AA"))
        self.assertEqual(aa[0]["lista_sha256"], "%064x" % 1)
        self.assertEqual(out.getvalue().count("ORDEN "), 7)
        self.assertIn("--region-pair none", out.getvalue())
        with open(lt, encoding="utf-8") as fh:
            self.assertEqual(len(fh.read().splitlines()), 4)


class BlockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="s2blk-")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def build(self, tweak=None):
        return F.build_attempt(os.path.join(self.tmp, "at"), n=300, tweak=tweak, lateral_columns=True)

    def block(self, b, out="out"):
        o = os.path.join(self.tmp, out)
        rc = quiet(S2.main, ["block", "--manifest", b["manifest"], "--attempt", b["attempt"], "--lists-dir",
                             b["lists_dir"], "--mask", b["mask"]["text"], "--replica", "1", "--out", o])
        with open(os.path.join(o, "bloque-r1.json"), "rb") as fh:
            return rc, json.loads(fh.read().decode("utf-8")), o

    def test_null_discordances(self):
        rc, js, o = self.block(self.build())
        self.assertEqual(rc, S2.EXIT_OK, js["bloqueantes_fallidos"])
        t = js["transiciones"]
        self.assertEqual((t["N"], t["n01"], t["n10"]), (300, 0, 0))
        self.assertEqual(t["n00"] + t["n01"] + t["n10"] + t["n11"], 300)
        self.assertEqual(js["detalle"]["cambian_de_pixel_detectados_en_ambas"], 0)
        self.assertGreater(js["cruces"]["cruces_union"], 0)
        with np.load(os.path.join(o, "hist-r1.npz")) as z:
            self.assertTrue(np.array_equal(z["A"], z["B"]))

    def test_one_discordance_into_r_int_is_counted(self):
        def tweak(records, entries):
            a, b = records["g06-A-r1"], records["g06-B-r1"]
            k = next(i for i, r in enumerate(a) if r["status"] == "DECAY" and b[i]["status"] == "DECAY")
            iel, iaz = (int(x) for x in np.argwhere(F.make_test_mask())[0])
            b[k].update(status="DET", detected=1, iel=iel, iaz=iaz, in_rint=1, plane=True, ekin=50.0,
                        end_vol="detector", end_process="Transportation")
        rc, js, o = self.block(self.build(tweak))
        self.assertEqual(rc, S2.EXIT_OK, js["bloqueantes_fallidos"])
        self.assertEqual((js["transiciones"]["n01"], js["transiciones"]["n10"]), (1, 0))
        # the four fixture crossers already differ in rng1 (expected for Z crossers); the new one changes Y
        self.assertEqual(js["detalle"]["discordantes"], 5)
        with open(os.path.join(o, "discordantes-r1.csv"), encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["Y_A"] != r["Y_B"]]
        self.assertEqual([(r["Y_A"], r["Y_B"]) for r in rows], [("0", "1")])

    def test_repeated_id_blocks_the_block(self):
        def tweak(records, entries):
            records["g06-B-r1"][5]["id"] = 4
        rc, js, _ = self.block(self.build(tweak))
        self.assertEqual(rc, S2.EXIT_BLOCKED)
        self.assertEqual(js["estado"], S2.A.STAMP)
        self.assertTrue(any("IDs" in x for x in js["bloqueantes_fallidos"]))

    def test_missing_rows_block_the_block(self):
        def tweak(records, entries):
            del records["g06-A-r1"][7]
        rc, js, _ = self.block(self.build(tweak))
        self.assertEqual(rc, S2.EXIT_BLOCKED)
        self.assertIsNone(js["transiciones"])

    def test_wrong_histogram_blocks_the_block(self):
        b = self.build()
        stem = b["paths"]["g06-B-r1"]["stem"]
        hist = C.read_hist_csv(stem + "_hist.csv")
        hist[10, 10] += 1
        with open(stem + "_hist.csv", "wb") as fh:
            fh.write(("\n".join(",".join(map(str, row)) for row in hist.tolist()) + "\n").encode("ascii"))
        F.write_sums(os.path.dirname(stem))
        rc, js, _ = self.block(b)
        self.assertEqual(rc, S2.EXIT_BLOCKED)
        self.assertTrue(any("histograma reconstruido" in x for x in js["bloqueantes_fallidos"]))


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="s2ctl-")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def controls(self, tweak=None):
        b = F.build_attempt(os.path.join(self.tmp, "at"), n=200, tweak=tweak, lateral_columns=True)
        o = os.path.join(self.tmp, "out")
        rc = quiet(S2.main, ["controls", "--manifest", b["manifest"], "--attempt", b["attempt"], "--lists-dir",
                             b["lists_dir"], "--mask", b["mask"]["text"], "--out", o])
        with open(os.path.join(o, "controles.json"), "rb") as fh:
            return rc, json.loads(fh.read().decode("utf-8"))

    def test_identical_controls_pass(self):
        rc, js = self.controls()
        self.assertEqual(rc, S2.EXIT_OK, js["bloqueantes_fallidos"])
        self.assertTrue(js["compuerta"]["K8_todo_identico"])
        self.assertEqual(len(js["pares"]), 3)

    def test_a_b_difference_in_the_control_closes_the_gate(self):
        def tweak(records, entries):
            records["g01-B-r2"][3]["n_steps"] += 1
        rc, js = self.controls(tweak)
        self.assertEqual(rc, S2.EXIT_BLOCKED)
        self.assertFalse(js["compuerta"]["K8_todo_identico"])


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="s2agg-")
        self.blocks = os.path.join(self.tmp, "bloques")
        os.makedirs(self.blocks)
        write_corridas(os.path.join(self.tmp, "corridas.csv"), corridas_rows())
        mask = F.write_mask_files(os.path.join(self.tmp, "roi"))
        self.mask = mask["text"]
        # stand-in calculator: records the counts it receives (the real one needs scipy; see the last test)
        self.calc = os.path.join(self.tmp, "calc.py")
        with open(self.calc, "w", encoding="utf-8") as fh:
            fh.write("MARGIN = 2.736603748e-07\n"
                     "def interval(counts):\n"
                     "    return {'counts_00_01_10_11': list(counts), 'decision_numerica': 'dentro_del_margen',\n"
                     "            'interval95_conservative_per_million': [-0.18, 0.18]}\n")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def write_block(self, rep, n01=0, n10=0, estado="OK"):
        n = S2.MAIN_N
        t = {"N": n, "n00": n - 10 - n01 - n10, "n01": n01, "n10": n10, "n11": 10, "m00": 400000, "m01": 0,
             "m10": 0, "m11": 600000, "det_A": 600000, "det_B": 600000, "sumY_A": 10 + n10, "sumY_B": 10 + n01}
        body = {"replica": rep, "estado": estado, "transiciones": t,
                "detalle": {"detectados_en_ambas": 600000, "cambian_de_pixel_detectados_en_ambas": 0,
                            "en_R_int_en_ambas_pero_otro_pixel": 0, "discordantes": n01 + n10,
                            "discordantes_por_clase": {"cruza_Z": n01 + n10} if n01 + n10 else {}},
                "cruces": {"cruces_A": 2, "cruces_B": 2, "cruces_ambas": 2, "cruces_union": 2},
                "recursos": [{}, {}]}
        with open(os.path.join(self.blocks, "bloque-r%d.json" % rep), "w", encoding="utf-8") as fh:
            json.dump(body, fh)
        h = np.zeros((C.N_EL, C.N_AZ), dtype=np.int64)
        np.savez_compressed(os.path.join(self.blocks, "hist-r%d.npz" % rep), A=h, B=h)

    def aggregate(self, controls_ok=True, calc=None):
        ctl = os.path.join(self.tmp, "controles.json")
        with open(ctl, "w", encoding="utf-8") as fh:
            json.dump({"estado": "OK" if controls_ok else "BLOQUEADO"}, fh)
        out = os.path.join(self.tmp, "agg-%d" % len(os.listdir(self.tmp)))
        rc = quiet(S2.main, ["aggregate", "--blocks-dir", self.blocks, "--controls", ctl, "--corridas",
                             os.path.join(self.tmp, "corridas.csv"), "--calculator", calc or self.calc,
                             "--mask", self.mask, "--out", out])
        with open(os.path.join(out, "resultados-etapa2.json"), "rb") as fh:
            return rc, json.loads(fh.read().decode("utf-8"))

    def test_sums_the_24_blocks_and_hands_the_counts_to_the_calculator(self):
        for rep in S2.MAIN_REPLICAS:
            self.write_block(rep, n01=1 if rep == 105 else 0)
        rc, res = self.aggregate()
        self.assertEqual(rc, S2.EXIT_OK, res["problemas"])
        self.assertEqual(res["totales"]["N"], 24000000)
        self.assertEqual(res["intervalo_R_int"]["counts_00_01_10_11"][1:3], [1, 0])
        self.assertEqual(res["categoria_de_cierre"], "dentro_del_margen")
        self.assertEqual(res["bloques_validos"], 24)

    def test_missing_or_invalid_block_or_failed_controls_is_not_interpretable(self):
        for rep in S2.MAIN_REPLICAS[:-1]:
            self.write_block(rep)
        rc, res = self.aggregate()
        self.assertEqual((rc, res["categoria_de_cierre"]), (S2.EXIT_BLOCKED, "no_interpretable"))
        self.assertTrue(any("faltan bloques" in p for p in res["problemas"]))
        self.write_block(S2.MAIN_REPLICAS[-1], estado="BLOQUEADO")
        rc, res = self.aggregate()
        self.assertEqual(res["categoria_de_cierre"], "no_interpretable")
        self.write_block(S2.MAIN_REPLICAS[-1])
        rc, res = self.aggregate(controls_ok=False)
        self.assertEqual(res["categoria_de_cierre"], "no_interpretable")

    @unittest.skipUnless(os.environ.get(CALCULATOR_ENV), "set %s to the director's calculator" % CALCULATOR_ENV)
    def test_interval_is_the_director_calculator(self):
        for rep in S2.MAIN_REPLICAS:
            self.write_block(rep, n01=1 if rep in (103, 117) else 0)
        rc, res = self.aggregate(calc=os.environ[CALCULATOR_ENV])
        calc = S2.load_calculator(os.environ[CALCULATOR_ENV])
        want = calc.interval([res["totales"]["n00"], 2, 0, res["totales"]["n11"]])
        self.assertEqual(res["intervalo_R_int"]["interval95_conservative_per_million"],
                         want["interval95_conservative_per_million"])
        self.assertEqual(res["categoria_de_cierre"], "inconcluso")


if __name__ == "__main__":
    unittest.main()
