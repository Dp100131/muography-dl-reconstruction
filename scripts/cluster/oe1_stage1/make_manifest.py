#!/usr/bin/env python3
"""OE1 stage 1: writes manifiesto-corridas.tsv (13 rows) and entradas-esperadas.sha256.

Python >= 3.9, standard library only (it also runs on the SC3 login node).

    python3 make_manifest.py --lists-dir DIR --catalog-dir DIR --tables-dir DIR \\
        --bin-a .../build/A2/muography_sim --bin-b .../build/B2/muography_sim \\
        --roi-mask .../analisis/roi/R_int.mask.txt \\
        --out manifiesto-corridas.tsv --inputs-sha256 entradas-esperadas.sha256

Columns (tab separated, header included):
  run_id caso geometria variante binario s_src s_tr case_code lista lista_sha256 N salida rol avisos_esperados

Rows: g06-{A,B}-r{1,2,3}, g01-{A,B}-r{1,2,3} and g01-A-r1-rep (role repeticion-AA: the same list and
the same s_tr as g01-A-r1). s_tr = s_src - 1e9 (1026092901..03), below 2^31 - 1. `salida` is relative
to the attempt folder: <caso>/<A|B|A-repeticion>/<s_src>. `avisos_esperados` is GeomVol1002=1 for A on
geo-train-06 and `ninguno` everywhere else.

--technical builds the manifest of the end-to-end rehearsal T-F instead: the technical lists
(replica 0, s_src 2026092990) of both cases, i.e. g06-{A,B}-r0, g01-{A,B}-r0 and g01-A-r0-rep.

The region pair is not a column: principal.sbatch (run_one) derives it from `caso` and always passes it
explicitly, because muography_sim exits 2 when a geometry with two or more anomalies gets no
--region-pair (plan W2): geo-train-06 -> `--region-pair 0,1` (conduit 0, dense body 1), geo-train-01
(no anomalies, no region) -> `--region-pair none`.

The file entradas-esperadas.sha256 has the sha256 of every fixed input (both binaries, both catalog
files, the three generator tables, the R_int mask, every list and its sidecar) with the SAME path
strings that principal.sbatch uses, so `sha256sum -c` and the per-run comparison work unchanged.
Neither output file is ever overwritten.

The manifest refuses (exit code 2) anything that is not what the plan fixes:
  * every production list has N = 100 000 (technical lists, 2000) in its sidecar AND in its rows;
  * the catalog files are the frozen ones: geo-train-06 c8f6c135..., geo-train-01 ff3c69ab...;
  * the three generator tables are the ones the pilot ran with (their SHA-256 in the pilot's
    entradas.sha256, which agree between the original and the corrected run).
"""
import argparse
import hashlib
import json
import os
import sys

COLUMNS = ("run_id", "caso", "geometria", "variante", "binario", "s_src", "s_tr", "case_code", "lista",
           "lista_sha256", "N", "salida", "rol", "avisos_esperados")
CASES = (("geo-train-06", "g06", 6), ("geo-train-01", "g01", 1))
S_SRC_BASE = 2026092900
S_SRC_TECHNICAL = 2026092990
S_TR_OFFSET = 1000000000
LIST_FORMAT = "oe1-primarios/1"
TABLE_FILES = ("row_weights_30d.txt", "p_cdf.txt", "meta.json")
PRODUCTION_N = 100000        # primaries per production run (PROT: 1e5)
TECHNICAL_N = 2000           # primaries per technical list (DV1)
# Catalog files frozen by the plan (step 1) and the generator tables of the pilot: SHA-256 taken from the
# pilot delivery's piloto/{original,corregida}/entradas.sha256 (both runs list the same three) and from the
# catalog files of the plan. Hard-coded on purpose: the manifest is what freezes the inputs of the attempt.
CATALOG_SHA256 = {
    "geo-train-06": "c8f6c1352af1f7564356cffbd98e26fd6a5881e5ece15a0ab6f2ed60c8ea26ed",
    "geo-train-01": "ff3c69ab7e37b51c5d4e1318d298a9ddc35446367e6ea4abf5c322e3b86f07b1",
}
TABLE_SHA256 = {
    "row_weights_30d.txt": "93ce67abe51c59ea3ac5c07dc99fd1615ae147df82c16d1bad64f5c0712d2c0c",
    "p_cdf.txt": "7161c8adf11711139840007dff26178221c4cfbff8fc3253cabdda50c533f6ae",
    "meta.json": "7ac4b6a9353af701b56e5a97ee783ed59072d58df52d80ddbdc0f5f70147d35d",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fail(msg):
    sys.stderr.write("make_manifest.py: %s\n" % msg)
    raise SystemExit(2)


def list_info(lists_dir, caso, code, rep, s_src, want_n):
    name = "%s_v0_r%d_s%d.primarios.csv" % (caso, rep, s_src)
    path = os.path.join(lists_dir, name)
    if not os.path.isfile(path):
        fail("missing list " + path)
    side_path = path + ".meta.json"
    if not os.path.isfile(side_path):
        fail("missing sidecar " + side_path)
    with open(side_path, "r") as fh:
        side = json.load(fh)
    checks = (("format", LIST_FORMAT), ("geometry_id", caso), ("view_index", 0), ("case_code", code),
              ("replica", rep), ("source_seed", s_src))
    for key, want in checks:
        if side.get(key) != want:
            fail("%s: sidecar %s = %r, expected %r" % (name, key, side.get(key), want))
    sha = sha256_file(path)
    if side.get("file_sha256") not in (None, sha):
        fail("%s: the sidecar's file_sha256 does not match the file" % name)
    if not isinstance(side.get("n"), int) or side["n"] <= 0:
        fail("%s: the sidecar has no valid n" % name)
    if side["n"] != want_n:
        fail("%s: the sidecar says n = %d, expected N = %d" % (name, side["n"], want_n))
    with open(path, "rb") as fh:
        rows = sum(1 for _ in fh) - 1                       # the header is not a muon
    if rows != want_n:
        fail("%s: the file has %d rows, expected N = %d" % (name, rows, want_n))
    return {"path": path, "sidecar": side_path, "sha256": sha, "n": side["n"]}


def build(args):
    lists_dir = os.path.abspath(args.lists_dir)
    catalog_dir = os.path.abspath(args.catalog_dir)
    tables_dir = os.path.abspath(args.tables_dir)
    bins = {"A": os.path.abspath(args.bin_a), "B": os.path.abspath(args.bin_b)}
    mask = os.path.abspath(args.roi_mask)
    reps = (0,) if args.technical else (1, 2, 3)
    rows = []
    fixed = [bins["A"], bins["B"], mask] + [os.path.join(tables_dir, t) for t in TABLE_FILES]
    for caso, short, code in CASES:
        fixed.append(os.path.join(catalog_dir, caso + ".json"))
    for path in fixed:
        if not os.path.isfile(path):
            fail("missing input " + path)
    # the frozen inputs: the catalog files of the plan and the tables the pilot ran with
    for caso, short, code in CASES:
        got = sha256_file(os.path.join(catalog_dir, caso + ".json"))
        if got != CATALOG_SHA256[caso]:
            fail("catalog %s.json has sha256 %s, the plan freezes %s" % (caso, got, CATALOG_SHA256[caso]))
    for name in TABLE_FILES:
        got = sha256_file(os.path.join(tables_dir, name))
        if got != TABLE_SHA256[name]:
            fail("table %s has sha256 %s, the pilot ran with %s" % (name, got, TABLE_SHA256[name]))
    want_n = TECHNICAL_N if args.technical else PRODUCTION_N
    for caso, short, code in CASES:
        geom = os.path.join(catalog_dir, caso + ".json")
        for rep in reps:
            s_src = S_SRC_TECHNICAL if args.technical else S_SRC_BASE + rep
            info = list_info(lists_dir, caso, code, rep, s_src, want_n)
            fixed += [info["path"], info["sidecar"]]
            for var in ("A", "B"):
                rows.append(_row("%s-%s-r%d" % (short, var, rep), caso, geom, var, bins[var], s_src, code, info,
                                 "principal", var))
            if caso == "geo-train-01" and rep == reps[0]:
                rows.append(_row("%s-A-r%d-rep" % (short, rep), caso, geom, "A", bins["A"], s_src, code, info,
                                 "repeticion-AA", "A-repeticion"))
    for path in fixed:
        if not os.path.isfile(path):
            fail("missing input " + path)
    expected = 4 * len(reps) + 1
    if len(rows) != expected:
        fail("built %d rows, expected %d" % (len(rows), expected))
    return rows, fixed


def _row(run_id, caso, geom, var, binary, s_src, code, info, rol, folder):
    warn = "GeomVol1002=1" if (var == "A" and caso == "geo-train-06") else "ninguno"
    return (run_id, caso, geom, var, binary, str(s_src), str(s_src - S_TR_OFFSET), str(code), info["path"],
            info["sha256"], str(info["n"]), "%s/%s/%d" % (caso, folder, s_src), rol, warn)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lists-dir", required=True)
    ap.add_argument("--catalog-dir", required=True)
    ap.add_argument("--tables-dir", required=True)
    ap.add_argument("--bin-a", required=True)
    ap.add_argument("--bin-b", required=True)
    ap.add_argument("--roi-mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs-sha256", required=True)
    ap.add_argument("--technical", action="store_true")
    args = ap.parse_args(argv)
    for p in (args.out, args.inputs_sha256):
        if os.path.exists(p):
            fail("refusing to overwrite " + p)
    rows, fixed = build(args)
    with open(args.out, "x", newline="\n") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            assert len(r) == len(COLUMNS)
            fh.write("\t".join(r) + "\n")
    with open(args.inputs_sha256, "x", newline="\n") as fh:
        for path in fixed:
            fh.write("%s  %s\n" % (sha256_file(path), path))
    print("wrote %s (%d rows) and %s (%d files)" % (args.out, len(rows), args.inputs_sha256, len(fixed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
