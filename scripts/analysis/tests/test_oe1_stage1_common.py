"""Tests of oe1_stage1_common: hashes, readers, pixel convention, statistics."""
import contextlib
import io
import math
import os
import re
import struct
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)

import oe1_stage1_common as C  # noqa: E402
import fixtures as F  # noqa: E402

try:
    import scipy.stats as SS
    from scipy import optimize as SO
except Exception:  # SciPy is optional (absent on SC3)
    SS = None
    SO = None

DATA = os.path.join(HERE, "data")
# 50 muons written by the instrumented binary (--write-primaries). The rows are the first 50 rows of the PRODUCTION
# replica-1 list of geo-train-06 (sidecar: source_seed 2026092901, replica 1, case_code 6; SC3 job 30510), not
# technical-seed data. The file carries an obviously technical name so that it cannot be mistaken for, or found
# instead of, a production list of the manifest.
REAL50_NAME = "fixture-tecnico-n50.primarios.csv"
REAL50 = os.path.join(DATA, REAL50_NAME)
# Line printed by muography_sim --check-primaries (A2 build, SC3 job 30510) for this list; the binary printed it
# for the same bytes under the file name it had then, only the file= field differs.
CPP_CHECK_LINE_50 = (
    "OE1-CHECK-PRIMARIES file=" + REAL50_NAME + " n=50 lf_only=1 header_ok=1 "
    "ends_with_newline=1 bad_rows=0 id_min=0 id_max=49 ids_contiguous=1 dup_ids=0 dup_states=0 nonfinite=0 "
    "norm_fail=0 max_norm_dev=2.220446049250313e-16 dz_nonneg=0 ekin_nonpos=0 pdg_bad=0 charge_mismatch=0 "
    "n_mu_plus=32 n_mu_minus=18 canon_fnv1a64=73ff04a2e4429cfa "
    "file_sha256=0ae76a8d610ca8a37c688768fef5b08cf9c9433e7154f5132381879abd2b0c28 sidecar=ok sidecar_n=50 "
    "sidecar_fnv_match=1 sidecar_sha_match=1 status=OK")


class HashTests(unittest.TestCase):
    def test_fnv_reference_vectors(self):
        # Landon Curt Noll's FNV-1a 64-bit test vectors.
        self.assertEqual(C.fnv1a64(b""), 0xcbf29ce484222325)
        self.assertEqual(C.fnv1a64(b"a"), 0xaf63dc4c8601ec8c)
        self.assertEqual(C.fnv1a64(b"foobar"), 0x85944171f73967e8)

    def test_canonical_fnv_equals_struct_pack(self):
        rng = np.random.default_rng(3)
        n = 40
        ids = np.arange(n, dtype=np.int64)
        vals = rng.normal(size=(n, 7)) * 1000
        pdg = np.where(rng.random(n) < 0.5, 13, -13).astype(np.int64)
        blob = b"".join(struct.pack("<q7di", int(ids[k]), *[float(x) for x in vals[k]], int(pdg[k]))
                        for k in range(n))
        self.assertEqual(C.canonical_fnv(ids, vals, pdg), C.fnv1a64(blob))

    def test_canonical_fnv_matches_the_cpp_binary(self):
        """The list was written by the instrumented binary; 73ff04a2e4429cfa is ITS hash."""
        table, d, sha, canon = C.read_primary_file(REAL50)
        self.assertEqual(C.hex16(canon), "73ff04a2e4429cfa")
        self.assertEqual(sha, "0ae76a8d610ca8a37c688768fef5b08cf9c9433e7154f5132381879abd2b0c28")

    def test_check_line_is_identical_to_the_binary_line(self):
        table, d, sha, canon = C.read_primary_file(REAL50)
        sc, err = C.read_sidecar(REAL50)
        ok, line = C.format_check_line(os.path.basename(REAL50), len(table["id"]), d, canon, sha, sc, err)
        self.assertTrue(ok)
        self.assertEqual(line, CPP_CHECK_LINE_50)

    def test_parsed_doubles_equal_python_float(self):
        with open(REAL50, "rb") as fh:
            data = fh.read()
        table, _ = C.parse_primary_bytes(data)
        rows = [ln.split(",") for ln in data.decode().splitlines()[1:]]
        for k in (0, 7, 49):
            for j in range(7):
                self.assertEqual(table["vals"][k, j], float(rows[k][1 + j]))


class PrimaryFnvTests(unittest.TestCase):
    """prim_fnv recomputed from the rows of a list, for every row at once (EventAction.cc PrimaryFnv, README)."""

    @staticmethod
    def recipe(row, pdg):
        """The scalar recipe of the README, byte by byte."""
        x, y, z, dx, dy, dz, e = [float(v) for v in row]
        inv = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
        payload = struct.pack("<7d", x * 1000.0, y * 1000.0, z * 1000.0, dx * inv, dy * inv, dz * inv,
                              e * 1000.0) + struct.pack("<i", int(pdg))
        h = 0xcbf29ce484222325
        for b in payload:
            h = ((h ^ b) * 0x100000001b3) & (2 ** 64 - 1)
        return h

    def test_rows_of_the_fnv_helper_match_the_reference_vectors(self):
        rows = np.frombuffer(b"ab", dtype=np.uint8).reshape(2, 1)
        got = C.fnv1a64_rows(rows)
        self.assertEqual([int(v) for v in got], [0xaf63dc4c8601ec8c, C.fnv1a64(b"b")])
        blob = np.frombuffer(b"foobar", dtype=np.uint8).reshape(1, 6)
        self.assertEqual(int(C.fnv1a64_rows(blob)[0]), 0x85944171f73967e8)
        self.assertEqual(len(C.fnv1a64_rows(np.zeros((0, 4), dtype=np.uint8))), 0)

    def test_vectorised_equals_the_scalar_recipe_on_random_rows(self):
        rng = np.random.default_rng(11)
        n = 400
        vals = np.hstack([rng.normal(size=(n, 3)) * 1500.0, rng.normal(size=(n, 3)), rng.random((n, 1)) * 500.0])
        vals[:, 3:6] /= np.sqrt((vals[:, 3:6] ** 2).sum(axis=1))[:, None]
        pdg = np.where(rng.random(n) < 0.5, 13, -13)
        got = C.primary_fnv_array(vals, pdg)
        self.assertEqual(got.dtype, np.uint64)
        for k in range(n):
            self.assertEqual(int(got[k]), self.recipe(vals[k], pdg[k]), k)

    def test_rows_of_the_real_list(self):
        table, d, sha, canon = C.read_primary_file(REAL50)
        got = C.primary_fnv_array(table["vals"], table["pdg"])
        self.assertEqual(len(got), 50)
        for k in range(50):
            self.assertEqual(int(got[k]), self.recipe(table["vals"][k], table["pdg"][k]), k)
        self.assertEqual(len(set(int(v) for v in got)), 50)            # 50 different primaries

    def test_the_norm_is_multiplied_not_divided(self):
        """CLHEP unit() multiplies by 1/norm; dividing gives other last bits for some rows, and the recipe must
        follow the multiplication (that is what the binary hashes)."""
        rng = np.random.default_rng(5)
        differing = 0
        for _ in range(2000):
            v = rng.normal(size=3)
            v /= np.sqrt((v ** 2).sum()) * (1.0 + 1e-13)          # a direction whose norm is not exactly 1
            row = [1.0, 2.0, 3.0, v[0], v[1], v[2], 4.0]
            norm = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            inv = 1.0 / norm
            mult = tuple(float(c) * inv for c in v)
            div = tuple(float(c) / norm for c in v)
            if mult != div:
                differing += 1
                self.assertEqual(int(C.primary_fnv_array(np.array([row]), np.array([13]))[0]),
                                 self.recipe(row, 13))
                if differing >= 5:
                    break
        self.assertGreaterEqual(differing, 1, "no row where multiplying and dividing differ was found")

    def test_every_input_changes_the_hash(self):
        row = np.array([[10.0, -20.0, 30.0, 0.6, 0.0, -0.8, 12.5]])
        base = int(C.primary_fnv_array(row, np.array([13]))[0])
        self.assertNotEqual(base, int(C.primary_fnv_array(row, np.array([-13]))[0]))
        for j in range(7):
            other = row.copy()
            other[0, j] = other[0, j] * (1.0 + 1e-9) if other[0, j] != 0.0 else 1e-9
            self.assertNotEqual(base, int(C.primary_fnv_array(other, np.array([13]))[0]), j)

    def test_hex_column_parsing(self):
        col = np.array([b"0000000000000000", b"ffffffffffffffff", b"0123456789abcdef", b"0123456789ABCDEF",
                        b"fffffffffffffff", b"gggggggggggggggg", b"", b"00000000000000001"], dtype="S17")
        values, valid = C.hex16_column_to_uint64(col)
        self.assertEqual(valid.tolist(), [True, True, True, True, False, False, False, False])
        self.assertEqual([int(v) for v in values[:4]], [0, 2 ** 64 - 1, 0x0123456789abcdef, 0x0123456789abcdef])
        self.assertEqual(len(C.hex16_column_to_uint64(np.zeros(0, dtype="S16"))[0]), 0)


class PrimaryDefectTests(unittest.TestCase):
    def setUp(self):
        with open(REAL50, "rb") as fh:
            self.data = fh.read()

    def parse(self, data):
        return C.parse_primary_bytes(data)[1]

    def test_clean(self):
        self.assertTrue(C.primary_defects_clean(self.parse(self.data), 50))

    def test_crlf(self):
        d = self.parse(self.data.replace(b"\n", b"\r\n"))
        self.assertEqual(d["lf_only"], 0)
        self.assertEqual(d["bad_rows"], 50)       # every data row keeps a CR in its last field
        self.assertEqual(d["header_ok"], 0)

    def test_no_trailing_newline(self):
        d = self.parse(self.data.rstrip(b"\n"))
        self.assertEqual(d["ends_with_newline"], 0)

    def test_header(self):
        d = self.parse(self.data.replace(b"charge_e", b"charge"))
        self.assertEqual(d["header_ok"], 0)

    def test_bad_rows_and_blank_line(self):
        lines = self.data.split(b"\n")
        lines[5] = b"1,2,3"
        lines.insert(7, b"")
        d = self.parse(b"\n".join(lines))
        self.assertEqual(d["bad_rows"], 2)
        self.assertEqual(d["ids_contiguous"], 0)

    def test_duplicate_rows_and_ids(self):
        lines = self.data.rstrip(b"\n").split(b"\n")
        lines.append(lines[3])            # same id and same state
        d = self.parse(b"\n".join(lines) + b"\n")
        self.assertEqual(d["dup_ids"], 1)
        self.assertEqual(d["dup_states"], 1)
        self.assertEqual(d["ids_contiguous"], 0)

    def test_physical_defects(self):
        lines = self.data.rstrip(b"\n").split(b"\n")
        f = lines[2].split(b",")
        f[6] = b"0.5"                     # dz >= 0 (and the norm changes)
        lines[2] = b",".join(f)
        f = lines[3].split(b",")
        f[7] = b"-1.0"                    # ekin <= 0
        lines[3] = b",".join(f)
        f = lines[4].split(b",")
        f[8] = b"11"                      # pdg
        lines[4] = b",".join(f)
        f = lines[5].split(b",")
        f[9] = b"1" if f[8] == b"13" else b"-1"   # charge mismatch
        lines[5] = b",".join(f)
        f = lines[6].split(b",")
        f[1] = b"nan"
        lines[6] = b",".join(f)
        d = self.parse(b"\n".join(lines) + b"\n")
        self.assertGreaterEqual(d["dz_nonneg"], 1)
        self.assertGreaterEqual(d["norm_fail"], 1)
        self.assertEqual(d["ekin_nonpos"], 1)
        self.assertEqual(d["pdg_bad"], 1)
        self.assertEqual(d["charge_mismatch"], 1)
        self.assertEqual(d["nonfinite"], 1)
        self.assertFalse(C.primary_defects_clean(d, 50))

    def test_strict_number_syntax(self):
        lines = self.data.rstrip(b"\n").split(b"\n")
        f = lines[2].split(b",")
        f[1] = b"+1.5"                    # from_chars rejects a leading '+'
        lines[2] = b",".join(f)
        f = lines[3].split(b",")
        f[2] = b" 1.5"                    # and whitespace
        lines[3] = b",".join(f)
        d = self.parse(b"\n".join(lines) + b"\n")
        self.assertEqual(d["bad_rows"], 2)


class SidecarTests(unittest.TestCase):
    def test_missing_sidecar(self):
        sc, err = C.read_sidecar(os.path.join(DATA, "nada.csv"))
        self.assertIsNone(sc)
        self.assertIn("missing", err)

    def test_real_sidecar(self):
        sc, err = C.read_sidecar(REAL50)
        self.assertIsNone(err)
        self.assertEqual(sc["format"], C.LIST_FORMAT)
        self.assertEqual(sc["n"], 50)


class MaskAndPixelTests(unittest.TestCase):
    def test_mask_text_round_trip(self):
        m = F.make_test_mask()
        data = C.mask_to_text(m)
        self.assertEqual(len(data), 256 * 257)
        self.assertTrue(np.array_equal(C.mask_from_text(data), m))
        self.assertEqual(C.mask_content_sha256(m), C.sha256_bytes(m.tobytes()))

    def test_mask_text_is_strict(self):
        data = bytearray(C.mask_to_text(F.make_test_mask()))
        with self.assertRaises(ValueError):
            C.mask_from_text(bytes(data[:-1]))
        bad = bytearray(data)
        bad[5] = ord("2")
        with self.assertRaises(ValueError):
            C.mask_from_text(bytes(bad))
        bad = bytearray(data)
        bad[256] = ord("\r")
        with self.assertRaises(ValueError):
            C.mask_from_text(bytes(bad))

    def test_lround_and_bin_index(self):
        self.assertEqual(C.lround(0.5), 1)
        self.assertEqual(C.lround(1.5), 2)
        self.assertEqual(C.lround(2.5), 3)          # halves away from zero (not banker's rounding)
        self.assertEqual(C.lround(-0.5), -1)
        self.assertEqual(C.bin_index(2.0, 2.0, 47.0, 256), 0)
        self.assertEqual(C.bin_index(47.0, 2.0, 47.0, 256), 255)
        self.assertEqual(C.bin_index(47.5, 2.0, 47.0, 256), -1)
        self.assertEqual(C.bin_index(1.0, 2.0, 47.0, 256), -1)

    def test_centroid_pixel_and_mirror(self):
        det = np.array(C.detector_center(0))
        self.assertEqual(tuple(det), (1000.0, 0.0, 1.5))
        cen = np.array([21.36, -30.71, 137.81])
        mom = (det - cen) / np.linalg.norm(det - cen)
        i_el, i_az, el, rel = C.pixel_of_direction(-mom, 0)
        self.assertEqual((i_el, i_az), (34, 135))
        self.assertAlmostEqual(el, 7.9256, places=3)
        self.assertAlmostEqual(rel, 1.7974, places=3)
        # mirror in azimuth: y -> -y around the look axis
        mom2 = np.array([mom[0], -mom[1], mom[2]])
        m_el, m_az, _, _ = C.pixel_of_direction(-mom2, 0)
        self.assertEqual((m_el, m_az), (34, 255 - 135))

    def test_look_azimuth_and_detector_for_other_views(self):
        self.assertAlmostEqual(C.look_azimuth_rad(0), math.pi)
        d1 = C.detector_center(1)
        self.assertAlmostEqual(d1[0], 1000 * math.cos(math.radians(60)), places=9)


class ParserTests(unittest.TestCase):
    def test_muones_round_trip_and_errors(self):
        ex = F.build_attempt_records_for_parse()
        mu = C.parse_muones_bytes(ex)
        self.assertEqual(len(mu["id"]), 5)
        self.assertEqual(mu["status"][0], b"DET")
        self.assertTrue(np.isnan(mu["lz_m"][2]) or mu["lz_m"][2] == 0.0)
        with self.assertRaises(ValueError):
            C.parse_muones_bytes(ex.replace(b"id,prim_fnv", b"idx,prim_fnv"))
        lines = ex.split(b"\n")
        lines[2] = lines[2] + b",1"
        with self.assertRaises(ValueError):
            C.parse_muones_bytes(b"\n".join(lines))

    def test_optional_lateral_columns_are_read_by_name_wherever_they_are(self):
        ex = F.build_attempt_records_for_parse()
        plain = C.parse_muones_bytes(ex)
        lines = [ln for ln in ex.decode("ascii").split("\n") if ln]
        # appended at the end
        end = "\n".join([lines[0] + ",max_lat_z_mm,sum_lat_z_mm"] + [ln + ",0.5,1.5" for ln in lines[1:]]) + "\n"
        mu = C.parse_muones_bytes(end.encode("ascii"))
        self.assertEqual(C.muon_columns_present(mu), list(C.MUON_COLUMNS) + list(C.MUON_OPTIONAL_COLUMNS))
        self.assertEqual(float(mu["max_lat_z_mm"][0]), 0.5)
        self.assertEqual(float(mu["sum_lat_z_mm"][4]), 1.5)
        self.assertEqual(C.muon_columns_present(plain), list(C.MUON_COLUMNS))
        # inserted after `crossed`, in the opposite order: the base columns are unchanged
        k = lines[0].split(",").index("crossed") + 1

        def insert(line, a, b):
            cells = line.split(",")
            return ",".join(cells[:k] + [a, b] + cells[k:])

        mid = "\n".join([insert(lines[0], "sum_lat_z_mm", "max_lat_z_mm")] +
                        [insert(ln, "2.5", "0.25") for ln in lines[1:]]) + "\n"
        mu2 = C.parse_muones_bytes(mid.encode("ascii"))
        self.assertEqual(float(mu2["sum_lat_z_mm"][0]), 2.5)
        self.assertEqual(float(mu2["max_lat_z_mm"][0]), 0.25)
        for c in C.MUON_COLUMNS:
            self.assertTrue(np.array_equal(mu2[c], plain[c], equal_nan=True) if plain[c].dtype.kind == "f"
                            else np.array_equal(mu2[c], plain[c]), c)

    def test_header_problems_are_rejected(self):
        ex = F.build_attempt_records_for_parse().decode("ascii")
        head, rest = ex.split("\n", 1)
        base = head.split(",")

        def with_header(cols, extra_cells=0):
            body = "\n".join(ln + "," * extra_cells for ln in rest.split("\n") if ln) + "\n"
            return (",".join(cols) + "\n" + body).encode("ascii")

        with self.assertRaises(ValueError):                                   # unknown column
            C.parse_muones_bytes(with_header(base + ["surprise"], 1))
        with self.assertRaises(ValueError):                                   # a base column is missing
            C.parse_muones_bytes(with_header(base[:-1]))
        with self.assertRaises(ValueError):                                   # base columns reordered
            C.parse_muones_bytes(with_header(base[1:] + base[:1]))
        with self.assertRaises(ValueError):                                   # repeated optional column
            C.parse_muones_bytes(with_header(base + ["max_lat_z_mm", "max_lat_z_mm"], 2))
        with self.assertRaises(ValueError):
            C.muon_header_columns("caso,geometry_id\xe9".encode("latin-1"))

    def test_warning_codes(self):
        log = ("x\n*** G4Exception : GeomVol1002\n  issued by\n*** G4Exception : GeomNav0003\n"
               "*** G4Exception : GeomVol1002\n")
        c = C.count_warning_codes(log)
        self.assertEqual(c, {"GeomVol1002": 2, "GeomNav0003": 1})
        self.assertEqual(C.warnings_as_text(c), "GeomNav0003=1,GeomVol1002=2")
        self.assertEqual(C.warnings_as_text({}), "ninguno")


class RunCommandTests(unittest.TestCase):
    def test_missing_or_existing_files_become_exit_code_2(self):
        def boom(a):
            raise FileNotFoundError("no such thing")

        def exists(a):
            raise FileExistsError("already there")

        for fn in (boom, exists):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(C.run_command(fn, 1), 2)
            self.assertIn("inutilizable", err.getvalue())
        self.assertEqual(C.run_command(lambda a: 7, 1), 7)

    def test_other_errors_still_propagate(self):
        def bad(a):
            raise ValueError("a real bug must not be hidden")

        with self.assertRaises(ValueError):
            C.run_command(bad, 1)


class TimingKeyTests(unittest.TestCase):
    """The A/A and A/B comparisons ignore an explicit allow-list of top-level keys, nothing else."""
    META = {"wall_seconds": 1.0, "seconds_per_event": 0.1, "seed": 7, "engine": "geant4",
            "rusage_user_cpu_seconds": 0.9, "rusage_system_cpu_seconds": 0.05, "rusage_max_rss_kb": 91234,
            "oe1_etapa1": {"runtime": {"cpu_seconds": 3.5, "cpu_model": "AMD EPYC", "threads_cli": 1,
                                       "run_manager_class": "G4TaskRunManager", "worker_thread_ids": [0]},
                           "counts": {"max_lat_mm": 12.5, "status": {"DET": 3}},
                           "muon_log": {"bytes": 10, "fnv1a64": "00"}}}

    def test_the_allow_list_is_exactly_what_the_binary_writes(self):
        self.assertEqual(sorted(C.TIMING_META_KEYS),
                         ["rusage_max_rss_kb", "rusage_system_cpu_seconds", "rusage_user_cpu_seconds",
                          "seconds_per_event", "wall_seconds"])
        self.assertFalse(hasattr(C, "TIMING_KEY_RE"))            # no substring rule any more

    def test_paths_are_the_listed_top_level_keys(self):
        self.assertEqual(C.timing_meta_paths(self.META),
                         ["rusage_max_rss_kb", "rusage_system_cpu_seconds", "rusage_user_cpu_seconds",
                          "seconds_per_event", "wall_seconds"])

    def test_removal_keeps_everything_else(self):
        clean = C.meta_without_timing(self.META)
        self.assertEqual(sorted(clean), ["engine", "oe1_etapa1", "seed"])
        self.assertEqual(clean["oe1_etapa1"], self.META["oe1_etapa1"])   # cpu_seconds deeper in the tree stays
        self.assertIn("wall_seconds", self.META)                       # the input is not modified

    def test_substring_lookalikes_are_compared_not_ignored(self):
        """The old substring rule swallowed any key that looked like a time or a resource: ru_*, cpu, user, memory,
        second... A real difference in such a key must show up now."""
        meta = {"seed": 1, "ru_utime": 1.5, "ru_stime": 0.2, "ru_maxrss": 90000, "ru_minflt": 12345,
                "ru_nvcsw": 7, "cpu_count": 8, "user_seconds": 3.0, "memory_mb": 512, "elapsed": 9.5,
                "events_transported": 10, "rusage": {"utime_s": 1.0, "maxrss_kb": 5}}
        self.assertEqual(C.timing_meta_paths(meta), [])
        self.assertEqual(C.meta_without_timing(meta), meta)

    def test_the_same_name_deeper_in_the_tree_is_not_ignored(self):
        meta = {"wall_seconds": 1.0, "oe1_etapa1": {"wall_seconds": 2.0, "rusage_max_rss_kb": 5}}
        self.assertEqual(C.timing_meta_paths(meta), ["wall_seconds"])
        self.assertEqual(C.meta_without_timing(meta), {"oe1_etapa1": {"wall_seconds": 2.0, "rusage_max_rss_kb": 5}})

    def test_strings_booleans_and_blocks_under_a_listed_name_are_never_ignored(self):
        meta = {"wall_seconds": "12.5", "seconds_per_event": True, "rusage_max_rss_kb": {"a": 1},
                "rusage_user_cpu_seconds": None, "rusage_system_cpu_seconds": [1.0]}
        self.assertEqual(C.timing_meta_paths(meta), [])
        self.assertEqual(C.meta_without_timing(meta), meta)

    def test_the_keys_the_binary_writes_are_time_keys_and_nothing_else_is(self):
        """RunAction.cc writes getrusage(RUSAGE_SELF) as top-level keys next to wall_seconds (plan W3); the
        new oe1_etapa1 entries (hashes, near-Z counts, lateral displacement) are not durations."""
        meta = {"wall_seconds": 1.0, "seconds_per_event": 0.1, "rusage_user_cpu_seconds": 0.9,
                "rusage_system_cpu_seconds": 0.05, "rusage_max_rss_kb": 91234, "seed": 1,
                "oe1_etapa1": {"binary_sha256": "0" * 64,
                               "inputs": {"geometry_json": {"file": "g.json", "sha256": "1" * 64},
                                          "tables": {"row_weights_30d.txt": "2" * 64, "p_cdf.txt": "3" * 64,
                                                     "meta.json": "4" * 64}},
                               "region": {"applicable": True, "explicit_none": False, "near_margin_m": 1.0,
                                          "eps_m": 1e-6},
                               "counts": {"max_lat_mm": 9.0, "max_lat_z_mm": 4.0, "max_sum_lat_z_mm": 4.0,
                                          "near_z_steps": 19, "near_z_muons": 10},
                               "checks": {"variant_matches_geometry": True}}}
        self.assertEqual(C.timing_meta_paths(meta),
                         ["rusage_max_rss_kb", "rusage_system_cpu_seconds", "rusage_user_cpu_seconds",
                          "seconds_per_event", "wall_seconds"])
        self.assertEqual(C.meta_without_timing(meta)["oe1_etapa1"], meta["oe1_etapa1"])


class CppContractTests(unittest.TestCase):
    """Names that the C++ binary writes and this package expects (read from the sources)."""
    APP = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, "simulation", "geant4_app"))

    def source(self, rel):
        path = os.path.join(self.APP, rel)
        if not os.path.isfile(path):
            self.skipTest("C++ sources not next to the tools: %s" % path)
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    def test_muon_log_columns_are_the_base_columns_then_the_optional_ones(self):
        text = self.source(os.path.join("include", "MuonRecord.hh"))
        body = text.split("inline const char* MuonCsvHeader()")[1].split("}")[0]
        header = "".join(re.findall(r'"([^"]*)"', body))
        self.assertEqual(header.split(","), list(C.MUON_COLUMNS) + list(C.MUON_OPTIONAL_COLUMNS))
        m = re.search(r"constexpr int kMuonCsvColumns = (\d+);", text)
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), len(C.MUON_COLUMNS) + len(C.MUON_OPTIONAL_COLUMNS))
        # the header the binary writes is accepted by the reader
        C.muon_header_columns(header.encode("ascii"))

    def test_rusage_keys_of_the_binary_are_ignored_by_the_comparisons(self):
        text = self.source(os.path.join("src", "RunAction.cc"))
        keys = re.findall(r'meta\["(rusage_[A-Za-z_]+)"\]', text)
        self.assertEqual(sorted(keys), ["rusage_max_rss_kb", "rusage_system_cpu_seconds",
                                        "rusage_user_cpu_seconds"])
        for k in keys:
            self.assertIn(k, C.TIMING_META_KEYS)
        self.assertEqual(C.meta_without_timing(dict.fromkeys(keys, 1.5)), {})

    def test_the_binary_writes_the_lateral_aggregates_the_analysis_reads(self):
        text = self.source(os.path.join("src", "RunAction.cc"))
        for key in ("max_lat_mm", "max_lat_z_mm", "max_sum_lat_z_mm", "binary_sha256", "variant_matches_geometry"):
            self.assertIn('"%s"' % key, text, key)


@unittest.skipIf(SS is None, "SciPy not installed")
class StatisticsVersusScipy(unittest.TestCase):
    def test_chi2_sf(self):
        for df in (1, 3, 50, 255, 1000):
            for x in (0.5, df * 0.6, df, df * 1.5, df * 2 + 20):
                a, b = C.chi2_sf(x, df), SS.chi2.sf(x, df)
                self.assertAlmostEqual(a, b, delta=1e-10 * max(1.0, b) + 1e-12 * b)

    def test_ks_uniform_pvalue(self):
        rng = np.random.default_rng(1)
        for n in (300, 3000, 30000):
            u = rng.random(n)
            d, p = C.ks_uniform_test(u)
            ref = SS.kstest(u, "uniform")
            self.assertAlmostEqual(d, ref.statistic, places=12)
            self.assertLess(abs(p - ref.pvalue), 0.02 * max(ref.pvalue, 0.05))
        # a clearly non-uniform sample gives a tiny p
        u = np.sort(rng.random(5000)) ** 1.3
        self.assertLess(C.ks_uniform_test(u)[1], 1e-6)

    def test_binomial_exact(self):
        for k, n, p in ((55959, 100000, 0.5612), (10, 20, 0.5), (3, 40, 0.3), (0, 10, 0.5), (200, 300, 0.7)):
            self.assertAlmostEqual(C.binom_two_sided_p(k, n, p), SS.binomtest(k, n, p).pvalue, delta=1e-8)

    def test_mcnemar_exact_versus_scipy(self):
        for b, c in ((1, 0), (0, 3), (5, 5), (7, 2), (40, 15), (150, 86), (86, 150)):
            got = C.mcnemar_exact(b, c)
            self.assertTrue(got["definido"])
            ref = SS.binomtest(min(b, c), b + c, 0.5).pvalue
            self.assertAlmostEqual(got["p_dos_colas"], ref, delta=1e-12 + 1e-9 * ref)


class ClopperPearsonTests(unittest.TestCase):
    """Exact two-sided Clopper-Pearson limits without SciPy."""

    @staticmethod
    def binom_cdf(k, n, p):
        """P(X <= k | n, p) by direct summation in log space (independent of the incomplete beta function)."""
        total = 0.0
        for i in range(k + 1):
            total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) +
                              i * math.log(p) + (n - i) * math.log1p(-p))
        return total

    def brute_upper(self, k, n, alpha):
        lo, hi = k / float(n), 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.binom_cdf(k, n, mid) > alpha / 2.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def test_no_successes_is_the_closed_form(self):
        for n in (10, 1000, 300000):
            lo, hi = C.clopper_pearson_interval(0, n)
            self.assertEqual(lo, 0.0)
            self.assertAlmostEqual(hi, 1.0 - 0.025 ** (1.0 / n), delta=1e-15)
        self.assertAlmostEqual(C.clopper_pearson_upper(0, 300000) * 3e5, 3.68888, places=4)

    def test_all_successes_is_the_mirror_case(self):
        lo, hi = C.clopper_pearson_interval(20, 20)
        self.assertEqual(hi, 1.0)
        self.assertAlmostEqual(lo, 0.025 ** (1.0 / 20), delta=1e-12)

    def test_published_values(self):
        # 95 % limits: 1/10 -> [0.00253, 0.44502]; 5/20 -> [0.0866, 0.4910]
        lo, hi = C.clopper_pearson_interval(1, 10)
        self.assertAlmostEqual(lo, 0.0025285785, places=8)
        self.assertAlmostEqual(hi, 0.4450161170, places=8)
        lo, hi = C.clopper_pearson_interval(5, 20)
        self.assertAlmostEqual(lo, 0.0866, places=3)
        self.assertAlmostEqual(hi, 0.4910, places=3)
        # the Poisson limit for a few events in a huge sample: 3 events -> 8.767 / n (97.5 % quantile)
        self.assertAlmostEqual(C.clopper_pearson_upper(3, 300000) * 3e5, 8.767, delta=2e-3)

    def test_against_direct_binomial_summation(self):
        for k, n in ((1, 40), (2, 300), (7, 100), (30, 60), (55, 60), (1, 3000), (12, 5000)):
            for alpha in (0.05, 0.10, 0.01):
                got = C.clopper_pearson_upper(k, n, alpha)
                self.assertAlmostEqual(got, self.brute_upper(k, n, alpha), delta=1e-9 * max(1.0, got), msg=(k, n))
                # it solves P(X <= k | p_upper) = alpha / 2
                self.assertAlmostEqual(self.binom_cdf(k, n, got), alpha / 2.0, delta=1e-9)

    def test_lower_limit_by_symmetry(self):
        for k, n in ((1, 40), (7, 100), (30, 60)):
            lo, hi = C.clopper_pearson_interval(k, n)
            # P(X >= k | p_lower) = alpha / 2 = 1 - P(X <= k - 1 | p_lower)
            self.assertAlmostEqual(1.0 - self.binom_cdf(k - 1, n, lo), 0.025, delta=1e-9)
            self.assertLess(lo, k / float(n))
            self.assertGreater(hi, k / float(n))

    def test_monotone_and_ordered(self):
        n = 1000
        prev = -1.0
        for k in range(0, 30):
            lo, hi = C.clopper_pearson_interval(k, n)
            self.assertLessEqual(lo, k / float(n))
            self.assertGreaterEqual(hi, k / float(n))
            self.assertGreater(hi, prev)
            prev = hi

    def test_invalid_arguments(self):
        for k, n in ((-1, 10), (11, 10), (0, 0)):
            with self.assertRaises(ValueError):
                C.clopper_pearson_upper(k, n)

    @unittest.skipIf(SS is None, "SciPy not installed")
    def test_against_scipy(self):
        for k, n in ((0, 300000), (1, 300000), (3, 300000), (2, 50), (10, 10), (77, 100)):
            lo, hi = C.clopper_pearson_interval(k, n, 0.05)
            ref = SS.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
            self.assertAlmostEqual(lo, ref.low, delta=1e-9 * max(1.0, ref.low) + 1e-13)
            self.assertAlmostEqual(hi, ref.high, delta=1e-9 * max(1.0, ref.high) + 1e-13)


class McNemarAndTangoTests(unittest.TestCase):
    def test_b_zero(self):
        got = C.mcnemar_exact(0, 0)
        self.assertFalse(got["definido"])
        self.assertEqual(got["p_dos_colas"], 1.0)

    def test_mcnemar_small_exact_values(self):
        # b = 1 discordant pair: P(X <= 0) = 1/2, two-sided = 1
        self.assertAlmostEqual(C.mcnemar_exact(1, 0)["p_dos_colas"], 1.0)
        # 5 vs 0: 2 * (1/2)^5 = 0.0625
        self.assertAlmostEqual(C.mcnemar_exact(5, 0)["p_dos_colas"], 0.0625, places=12)
        # 10 vs 0: 2 * (1/2)^10
        self.assertAlmostEqual(C.mcnemar_exact(0, 10)["p_dos_colas"], 2 * 0.5 ** 10, places=15)

    def test_agresti_table_10_1(self):
        """Agresti, Categorical Data Analysis (2nd ed., 2002), Table 10.1 (presidential
        approval, n = 1600): 794, 150, 86, 570. The published McNemar statistic is
        z = (86 - 150)/sqrt(86 + 150) = -4.17; the sample difference is 0.04."""
        n01, n10, n = 150, 86, 1600
        z = (n10 - n01) / math.sqrt(n01 + n10)
        self.assertAlmostEqual(z, -4.17, places=2)
        # Tango's statistic at delta = 0 reduces to McNemar's z (Tango 1998)
        self.assertAlmostEqual(C._tango_z(n01, n10, n, 0.0), (n01 - n10) / math.sqrt(n01 + n10), places=10)
        lo, hi = C.tango_interval(n01, n10, n)
        wald_se = math.sqrt(n01 + n10 - (n01 - n10) ** 2 / n) / n
        self.assertAlmostEqual((n01 - n10) / n, 0.04, places=12)
        self.assertAlmostEqual(lo, 0.04 - 1.96 * wald_se, delta=1.5e-3)
        self.assertAlmostEqual(hi, 0.04 + 1.96 * wald_se, delta=1.5e-3)
        self.assertGreater(lo, 0.0)                       # 0 is excluded: significant at 5 %
        self.assertLess(C.mcnemar_exact(n01, n10)["p_dos_colas"], 1e-4)

    def test_wikipedia_mcnemar_example(self):
        """McNemar's test, Wikipedia example (Test 1 vs Test 2): b = 121, c = 59 -> chi2 = 21.36."""
        b, c = 121, 59
        self.assertAlmostEqual((b - c) ** 2 / (b + c), 21.36, places=2)
        self.assertLess(C.mcnemar_exact(b, c)["p_dos_colas"], 1e-4)

    def test_tango_no_discordant_pairs_closed_form(self):
        """With n01 = n10 = 0 the score limits are +-z^2/(N + z^2)."""
        z = C._norm_ppf(0.975)
        for n in (100, 3000, 300000):
            lo, hi = C.tango_interval(0, 0, n)
            expect = z * z / (n + z * z)
            self.assertAlmostEqual(hi, expect, delta=1e-12)
            self.assertAlmostEqual(lo, -expect, delta=1e-12)
        self.assertAlmostEqual(C.tango_interval(0, 0, 300000)[1] * 1e6, 12.8047, places=3)

    def test_tango_limits_are_the_score_roots(self):
        z = C._norm_ppf(0.975)
        for n01, n10, n in ((1, 0, 300000), (0, 2, 300000), (5, 3, 1000), (40, 60, 5000), (2, 2, 100)):
            lo, hi = C.tango_interval(n01, n10, n)
            self.assertAlmostEqual(C._tango_z(n01, n10, n, lo), z, places=6)
            self.assertAlmostEqual(C._tango_z(n01, n10, n, hi), -z, places=6)
            self.assertLessEqual(lo, (n01 - n10) / n)
            self.assertGreaterEqual(hi, (n01 - n10) / n)

    def test_tango_symmetry(self):
        lo, hi = C.tango_interval(7, 2, 500)
        lo2, hi2 = C.tango_interval(2, 7, 500)
        self.assertAlmostEqual(lo, -hi2, places=12)
        self.assertAlmostEqual(hi, -lo2, places=12)

    @unittest.skipIf(SO is None, "SciPy not installed")
    def test_tango_statistic_versus_numerical_mle(self):
        """Z(delta) with the closed-form restricted MLE equals a numerical maximisation."""

        def z_brute(n01, n10, n, d):
            s = n - n01 - n10

            def nll(q):
                t = 0.0
                if n01:
                    t += n01 * math.log(q + d)
                if n10:
                    t += n10 * math.log(q)
                if s:
                    t += s * math.log(1 - 2 * q - d)
                return -t

            eps = 1e-14
            r = SO.minimize_scalar(nll, bounds=(max(0.0, -d) + eps, (1 - d) / 2 - eps), method="bounded",
                                   options={"xatol": 1e-15})
            return (n01 - n10 - n * d) / math.sqrt(n * (2 * r.x + d - d * d))

        rng = np.random.default_rng(3)
        worst = 0.0
        for _ in range(150):
            n = int(rng.integers(30, 4000))
            n01 = int(rng.integers(0, max(1, n // 6)))
            n10 = int(rng.integers(0, max(1, n // 6)))
            d = float(rng.uniform(-0.25, 0.25))
            worst = max(worst, abs(C._tango_z(n01, n10, n, d) - z_brute(n01, n10, n, d)))
        self.assertLess(worst, 1e-5)

    def test_norm_ppf(self):
        self.assertAlmostEqual(C._norm_ppf(0.975), 1.959963984540054, places=12)
        self.assertAlmostEqual(C._norm_ppf(0.5), 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
