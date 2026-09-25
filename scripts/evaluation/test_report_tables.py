#!/usr/bin/env python3
"""Stage 19 table tests — a golden mini-CSV reproduced character for character.

The descriptive half of the master tables is pure arithmetic, so it is
compared against an expected Markdown string exactly. The inferential half is
checked structurally (columns, matched pairs, Holm ordering, verdict logic)
because a bootstrap interval is not a stable golden across NumPy versions.

Run:  python scripts/evaluation/test_report_tables.py
"""

from __future__ import annotations

import csv
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import report_tables as RT  # noqa: E402

# 2 geometries x 2 views x 2 realizations: the smallest table with real cells
GEOMETRIES = ("geo-test-01", "geo-test-02")
VIEWS = (0, 1)
KS = (0, 1)
LEVEL = 0.05
SSIM = {"B": 0.3700, "C": 0.4464, "D": 0.7777, "E": 0.7787, "F": 0.7834}


def _write_golden_csv(path, wobble_amplitude: float = 0.0):
    """wobble_amplitude = 0 makes every value identical, so the golden table
    (including the degenerate bootstrap interval) is exact and free of any
    dependence on the NumPy random stream."""
    rows = []
    for geometry in GEOMETRIES:
        for view in VIEWS:
            # arm A: the reference, one detected anomaly per view
            rows.append({"split": "test", "arm": "A", "arm_label": "referencia",
                         "geometry": geometry, "view": view, "fraction": 1.0,
                         "k": -1, "scope": "roi", "anomaly_idx": 0,
                         "anomaly_kind": "cavity", "C": 0.2000, "z": 9.0,
                         "roi_valid": 1, "ssim": "", "psnr_db": ""})
            # an ROI below the 64 px floor: reported with roi_valid = 0 and
            # explicit NaN statistics, and excluded from every rate
            rows.append({"split": "test", "arm": "A", "arm_label": "referencia",
                         "geometry": geometry, "view": view, "fraction": 1.0,
                         "k": -1, "scope": "roi", "anomaly_idx": 1,
                         "anomaly_kind": "conduit", "C": "nan", "z": "nan",
                         "roi_valid": 0, "ssim": "", "psnr_db": ""})
            rows.append({"split": "test", "arm": "A", "arm_label": "referencia",
                         "geometry": geometry, "view": view, "fraction": 1.0,
                         "k": -1, "scope": "sham", "anomaly_idx": 0,
                         "anomaly_kind": "sham", "C": 0.01, "z": 1.2,
                         "ssim": "", "psnr_db": ""})
            for k in KS:
                for arm, value in SSIM.items():
                    # the wobble cancels within a cell, so the cell mean is
                    # exactly the nominal value and only the pair-level SD moves
                    wobble = wobble_amplitude * ((view + k) % 2 - 0.5) * 2.0
                    rows.append({
                        "split": "test", "arm": arm, "arm_label": arm,
                        "geometry": geometry, "view": view, "fraction": LEVEL,
                        "k": k, "scope": "global", "anomaly_idx": -1,
                        "anomaly_kind": "", "ssim": f"{value + wobble:.6g}",
                        "psnr_db": "20", "C": "", "z": ""})
                    rows.append({
                        "split": "test", "arm": arm, "arm_label": arm,
                        "geometry": geometry, "view": view, "fraction": LEVEL,
                        "k": k, "scope": "roi", "anomaly_idx": 0,
                        "anomaly_kind": "cavity", "ssim": "", "psnr_db": "",
                        "roi_valid": 1,
                        "C": f"{0.2000 * (0.6 if arm == 'B' else 1.0):.6g}",
                        "z": "8.0" if arm != "B" else "1.0"})
                    rows.append({
                        "split": "test", "arm": arm, "arm_label": arm,
                        "geometry": geometry, "view": view, "fraction": LEVEL,
                        "k": k, "scope": "roi", "anomaly_idx": 1,
                        "anomaly_kind": "conduit", "ssim": "", "psnr_db": "",
                        "roi_valid": 0, "C": "nan", "z": "nan"})
                    rows.append({
                        "split": "test", "arm": arm, "arm_label": arm,
                        "geometry": geometry, "view": view, "fraction": LEVEL,
                        "k": k, "scope": "sham", "anomaly_idx": 0,
                        "anomaly_kind": "sham", "ssim": "", "psnr_db": "",
                        "C": "0.01", "z": "1.5" if arm != "B" else "6.0"})

    columns = ["split", "arm", "arm_label", "geometry", "view", "fraction",
               "k", "scope", "anomaly_idx", "anomaly_kind", "ssim", "psnr_db",
               "C", "z", "roi_valid"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return len(rows)


EXPECTED_LEVELS = "\n".join([
    "| Nivel | Brazo | Media ± DE (n pares) | Media por celda | IC 95 % (método) | n celdas |",
    "|---|---|---|---|---|---|",
    "| 5 % | crudo | 0.3700 ± 0.0000 (n=8) | 0.3700 | [0.3700, 0.3700] (percentile) | 4 |",
    "| 5 % | crudo 1/f | 0.4464 ± 0.0000 (n=8) | 0.4464 | [0.4464, 0.4464] (percentile) | 4 |",
    "| 5 % | clásica congelada | 0.7777 ± 0.0000 (n=8) | 0.7777 | [0.7777, 0.7777] (percentile) | 4 |",
    "| 5 % | U-Net L1 | 0.7787 ± 0.0000 (n=8) | 0.7787 | [0.7787, 0.7787] (percentile) | 4 |",
    "| 5 % | cWGAN-GP | 0.7834 ± 0.0000 (n=8) | 0.7834 | [0.7834, 0.7834] (percentile) | 4 |",
])


def test_golden_levels_table() -> bool:
    """The descriptive table must match the expected Markdown exactly."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.csv"
        n_rows = _write_golden_csv(path)
        rows = RT.load_metrics(path)
        table = RT.table_levels(rows, "ssim", b=200, seed=1)

    ok = table == EXPECTED_LEVELS
    print(f"  golden levels table ({n_rows} source rows): "
          f"{'ok' if ok else 'FAIL'}")
    if not ok:
        got = table.splitlines()
        want = EXPECTED_LEVELS.splitlines()
        for index in range(max(len(got), len(want))):
            g = got[index] if index < len(got) else "<missing>"
            w = want[index] if index < len(want) else "<missing>"
            if g != w:
                print(f"    line {index}:\n      got  {g}\n      want {w}")
    if ok:
        # and with variation the same code must report a real SD and a
        # non-degenerate interval -- the golden fixture is deliberately flat,
        # not accidentally so
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            _write_golden_csv(path, 0.001)
            varied = RT.table_levels(RT.load_metrics(path), "ssim", b=200,
                                     seed=1)
        ok = "± 0.0011" in varied and "[0.3700, 0.3700]" not in varied
        print(f"  varying fixture yields a non-degenerate table: "
              f"{'ok' if ok else 'FAIL'}")
    return ok


def test_loading_and_pairing() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.csv"
        _write_golden_csv(path, 0.001)
        rows = RT.load_metrics(path)
        cwgan = RT.select(rows, arm="F", scope="global", fraction=LEVEL)
        classical = RT.select(rows, arm="D", scope="global", fraction=LEVEL)
        diffs, matched, dropped = RT.paired_differences(cwgan, classical)
        unmatched, m2, d2 = RT.paired_differences(cwgan[:4], classical)

    checks = {
        "numeric columns are typed": isinstance(cwgan[0]["ssim"], float),
        "integer columns are typed": isinstance(cwgan[0]["view"], int),
        "empty float cells become NaN":
            math.isnan(RT.select(rows, arm="A", scope="roi")[0]["ssim"]),
        "float selection tolerates rounding": len(cwgan) == 8,
        "pairing matches every realization": matched == 8,
        "pairing drops nothing when complete": dropped == 0,
        "four cells": len(diffs) == 4,
        "the difference is the frozen margin":
            abs(float(np.mean([v.mean() for v in diffs.values()]))
                - (SSIM["F"] - SSIM["D"])) < 1e-9,
        "unmatched rows are counted": d2 == 4 and m2 == 4,
        "the cell key is geometry|view":
            sorted(diffs)[0] == "geo-test-01|v0",
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  loading and pairing: {sum(checks.values())}/{len(checks)} "
          f"checks -> {'ok' if ok else 'FAIL'}")
    return ok


def test_primary_and_verdicts() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "metrics.csv"
        _write_golden_csv(path, 0.001)
        rows = RT.load_metrics(path)
        primary = RT.table_primary(rows, "ssim", b=200, seed=1)
        thresholds = dict(RT.DEFAULT_THRESHOLDS)
        thresholds["z_thr"] = 4.0
        verdicts = RT.table_verdicts(rows, thresholds, "ssim", b=200, seed=1)
        detect = RT.table_detectability(rows, thresholds, b=200, seed=1)
        calibration = RT.calibrate_z_threshold(rows)

    lines = [line for line in primary.splitlines() if line.startswith("| 5 %")]
    checks = {
        "three comparisons at the measured level": len(lines) == 3,
        "the family is announced as 12 hypotheses": "de 12 hipótesis" in primary,
        "levels without data are still listed":
            primary.count("sin datos") == 9,
        "the cWGAN minus classical margin is printed":
            f"{SSIM['F'] - SSIM['D']:.4f}" in primary,
        "the verdict table names S1, S2 and S3":
            all(token in verdicts for token in ("S1", "S2", "S3")),
        "provisional thresholds are flagged":
            "PROVISIONALES" in verdicts,
        "S2 compares against the frozen floor": "piso" in verdicts,
        "the detectability table reports a hallucination column":
            "Alucinación (sham)" in detect,
        "the reference shams calibrate z_thr":
            abs(calibration["z_thr"] - 1.2) < 1e-9 and calibration["n"] == 4,
    }
    # the classification itself: arm B is below z_thr -> lost; F is preserved
    rates_f = RT.detectability_rates(rows, "F", LEVEL, 4.0, thresholds)
    rates_b = RT.detectability_rates(rows, "B", LEVEL, 4.0, thresholds)
    checks["a matching arm is fully preserved"] = rates_f["preserved"] == 1.0
    checks["an under-threshold arm is fully lost"] = rates_b["lost"] == 1.0
    checks["a deflated contrast is not preserved"] = rates_b["preserved"] == 0.0
    checks["the sham denominator is counted"] = rates_f["n_sham"] == 8
    checks["hallucination uses the sham rows"] = (
        rates_b["hallucinated"] == 1.0 and rates_f["hallucinated"] == 0.0)

    # W3: the roi_valid = 0 rows must be OUT of every rate and counted apart
    checks["excluded ROIs stay out of the denominator"] = (
        rates_f["denominator"] == 8 and rates_b["denominator"] == 8)
    checks["excluded ROIs are counted"] = (
        rates_f["excluded_roi"] == 8 and rates_b["excluded_roi"] == 8)
    checks["the table carries a dedicated excluded-ROI column"] = (
        "ROI excluidas (<64 px)" in detect)
    checks["the table carries a dedicated excluded-ROI row"] = (
        "**ROI excluidas (todas las filas)**" in detect)
    checks["a table with no roi_valid column raises"] = _raises(
        ValueError, RT.detectability_rates,
        [{**row, "roi_valid": None} for row in rows], "F", LEVEL, 4.0,
        thresholds)

    # W5: S1 needs the Holm rejection, not just an interval away from zero
    comparisons, holm = RT.primary_family(rows, "ssim", b=200, seed=1)
    item, rejected = RT.find_comparison(comparisons, holm, LEVEL, RT.ARM_CLASSICAL)
    checks["the verdict table prints the Holm p next to S1"] = (
        "Holm p" in verdicts)
    checks["S1 and the primary table read the same comparison"] = (
        item is not None
        and abs(item["report"]["estimate"] - (SSIM["F"] - SSIM["D"])) < 1e-9)
    # this fixture is exactly the case W5 is about: four cells give a Wilcoxon
    # p of 0.125, so the interval excludes zero (it is degenerate: every cell
    # carries the same +0.0057) while Holm over the family of 12 refuses. The
    # old rule read the interval alone and would have declared S1 met.
    level_line = next(line for line in verdicts.splitlines()
                      if line.startswith(f"| {LEVEL * 100:g} %"))
    checks["the fixture is the one W5 is about"] = (
        item is not None and item["report"]["ci"]["excludes_zero"]
        and not rejected)
    checks["S1 follows Holm, not the interval alone"] = (
        "NO CUMPLE (IC excluye 0 pero no sobrevive Holm)" in level_line)
    checks["the level verdict is therefore not met"] = (
        level_line.rstrip().endswith("NO CUMPLE |"))

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  primary and verdicts: {sum(checks.values())}/{len(checks)} "
          f"checks -> {'ok' if ok else 'FAIL'}")
    return ok


def test_fragment_writing() -> bool:
    """The CLI must emit five fragments plus the calibration JSON."""
    import contextlib
    import io

    saved_argv = sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "cache"
        cache.mkdir()
        _write_golden_csv(cache / "metrics.csv", 0.001)
        out = Path(tmp) / "fragments"
        buffer = io.StringIO()
        try:
            sys.argv = ["report_tables.py", "--cache", str(cache),
                        "--out", str(out),
                        "--thresholds", str(Path(tmp) / "missing.json"),
                        "--bootstrap", "200"]
            with contextlib.redirect_stdout(buffer):
                status = RT.main()
        finally:
            sys.argv = saved_argv
        names = sorted(p.name for p in out.glob("*")) if out.exists() else []
        checks = {
            "the CLI exits cleanly": status == 0,
            "five fragments plus the calibration": names == [
                "01_levels.md", "02_primary.md", "03_verdicts.md",
                "04_detectability.md", "05_generalization.md",
                "z_thr_calibration.json"],
            "the provisional warning is printed":
                "PROVISIONAL" in buffer.getvalue(),
            "every fragment is non-empty": all(
                (out / name).stat().st_size > 0 for name in names),
        }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL (files: {names})")
    print(f"  fragment writing: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_geometry_split() -> bool:
    """Protocol v1.1 section 5.4: the primary family is the TEST geometries
    only (n = 18); geo-gen-01 is reported apart, never inside the Holm family.

    Locks the split helper and the CLI contract: a metrics.csv carrying a
    geometry_split column must route the generalization rows out of the four
    primary fragments and into 05_generalization; a cache without the column
    (the validation dry run) must keep every row in the test family so the
    rehearsal tables are unchanged.
    """
    tagged = [
        {"geometry": "geo-test-01", "geometry_split": "test", "arm": "F"},
        {"geometry": "geo-test-02", "geometry_split": "test", "arm": "F"},
        {"geometry": "geo-gen-01", "geometry_split": "generalization",
         "arm": "F"},
    ]
    test_rows, gen_rows = RT.split_by_geometry(tagged)
    untagged = [{"geometry": "geo-test-01", "arm": "F"},
                {"geometry": "geo-test-02", "arm": "F"}]
    test_only, gen_empty = RT.split_by_geometry(untagged)
    checks = {
        "test rows keep only geometry_split == test":
            [r["geometry"] for r in test_rows] == ["geo-test-01",
                                                   "geo-test-02"],
        "generalization rows are separated out":
            [r["geometry"] for r in gen_rows] == ["geo-gen-01"],
        "no rows are lost across the partition":
            len(test_rows) + len(gen_rows) == len(tagged),
        "a cache without the column keeps every row as test":
            len(test_only) == len(untagged) and gen_empty == [],
        "the generalization fragment is a placeholder when empty":
            "geometry_split" in RT.table_generalization([]),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  geometry split (section 5.4): {sum(checks.values())}/"
          f"{len(checks)} checks -> {'ok' if ok else 'FAIL'}")
    return ok


def _raises(exception, fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except exception:
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"    unexpected {type(exc).__name__}: {exc}")
        return False
    return False


def test_test_split_needs_frozen_thresholds() -> bool:
    """W4: on the test split, unfrozen thresholds ABORT, they do not warn."""
    provisional = {**RT.DEFAULT_THRESHOLDS, "provisional": True,
                   "missing_file": None}
    frozen = {**RT.DEFAULT_THRESHOLDS, "provisional": False,
              "missing_file": None}
    absent = {**RT.DEFAULT_THRESHOLDS, "provisional": True,
              "missing_file": "results/evaluation/protocol_thresholds.json"}
    checks = {
        "provisional thresholds abort on test": _raises(
            RuntimeError, RT.assert_thresholds_are_frozen,
            {"split": "test"}, provisional),
        "a missing thresholds file aborts on test": _raises(
            RuntimeError, RT.assert_thresholds_are_frozen,
            {"split": "test"}, absent),
        "the refusal says which file is missing":
            "protocol_thresholds.json" in _message(
                RuntimeError, RT.assert_thresholds_are_frozen,
                {"split": "test"}, absent),
        "frozen thresholds pass on test": not _raises(
            RuntimeError, RT.assert_thresholds_are_frozen,
            {"split": "test"}, frozen),
        "validation is still free to rehearse with provisional numbers":
            not _raises(RuntimeError, RT.assert_thresholds_are_frozen,
                        {"split": "val"}, provisional),
        "an absent summary does not block the dry run": not _raises(
            RuntimeError, RT.assert_thresholds_are_frozen, {}, provisional),
    }

    # end to end through the CLI: a test-split cache plus provisional
    # thresholds must fail before a single fragment is written
    written = None
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "cache"
        cache.mkdir()
        _write_golden_csv(cache / "metrics.csv", 0.001)
        (cache / "summary.json").write_text('{"split": "test"}',
                                            encoding="utf-8")
        out = Path(tmp) / "fragments"
        saved_argv = sys.argv
        try:
            sys.argv = ["report_tables.py", "--cache", str(cache), "--out",
                        str(out), "--thresholds", str(Path(tmp) / "gone.json"),
                        "--bootstrap", "50"]
            aborted = _raises(RuntimeError, RT.main)
        finally:
            sys.argv = saved_argv
        written = sorted(p.name for p in out.glob("*")) if out.exists() else []
    checks["the CLI aborts on a test cache with no frozen thresholds"] = aborted
    checks["nothing is written when it aborts"] = written == []

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  frozen-threshold gate: {sum(checks.values())}/{len(checks)} "
          f"checks -> {'ok' if ok else 'FAIL'}")
    return ok


def _message(exception, fn, *args) -> str:
    try:
        fn(*args)
    except exception as exc:
        return str(exc)
    except Exception:  # noqa: BLE001
        return ""
    return ""


def main() -> int:
    print("Stage 19 report-table tests")
    results = {
        "golden_levels": test_golden_levels_table(),
        "loading_and_pairing": test_loading_and_pairing(),
        "primary_and_verdicts": test_primary_and_verdicts(),
        "geometry_split": test_geometry_split(),
        "frozen_threshold_gate": test_test_split_needs_frozen_thresholds(),
        "fragment_writing": test_fragment_writing(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 19 REPORT TABLES:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
