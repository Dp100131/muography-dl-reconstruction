#!/usr/bin/env python3
"""Stage 19 figure tests — synthetic CSV in, PNG files out (matplotlib Agg).

Figures cannot be asserted pixel by pixel without becoming brittle, so the
checks are: the code runs on a table shaped like the real one, every family
produces a non-empty file, and the helpers that carry NUMBERS into the figures
(the cell summary and the equivalence projection) return the right values.

Run:  python scripts/evaluation/test_report_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import report_figures as RF  # noqa: E402
import report_tables as RT  # noqa: E402

LEVELS = (0.50, 0.10, 0.05, 0.01)
DENSE = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 0.90)
# mirrors run_evaluation.OCTILE_COLUMNS; kept literal so this test does not
# import the driver (which would pull in torch-adjacent modules for nothing)
OCTILE_COLUMNS = ["split", "arm", "model_run", "geometry", "view", "fraction",
                  "k", "anomaly_idx", "anomaly_kind", "octile", "n_roi",
                  "l_lo_m", "l_hi_m", "l_mean_m", "delta_opacity_mean", "C",
                  "z"]
ARM_LEVEL_SSIM = {
    "B": {0.50: 0.8718, 0.10: 0.5261, 0.05: 0.3700, 0.01: 0.1022},
    "C": {0.50: 0.8910, 0.10: 0.5965, 0.05: 0.4464, 0.01: 0.1271},
    "D": {0.50: 0.8914, 0.10: 0.7869, 0.05: 0.7777, 0.01: 0.7438},
    "E": {0.50: 0.9021, 0.10: 0.7832, 0.05: 0.7787, 0.01: 0.7723},
    "F": {0.50: 0.9054, 0.10: 0.7997, 0.05: 0.7834, 0.01: 0.7695},
}


def _write_csv(path, n_geometries=3, n_views=6, n_k=4):
    rng = np.random.default_rng(20260805)
    columns = RT.NUMERIC_COLUMNS + RT.INTEGER_COLUMNS + (
        "split", "arm", "arm_label", "model_run", "geometry", "geometry_split",
        "convention", "scope", "anomaly_kind", "notes")
    columns = list(dict.fromkeys(columns))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()

        def row(**values):
            base = {column: "" for column in columns}
            base.update({"split": "val", "scope": "global", "anomaly_idx": -1})
            base.update(values)
            writer.writerow(base)

        for g in range(n_geometries):
            geometry = f"geo-val-{g + 1:02d}"
            for view in range(n_views):
                for level in LEVELS:
                    for k in range(n_k):
                        for arm, table in ARM_LEVEL_SSIM.items():
                            row(arm=arm, geometry=geometry, view=view,
                                fraction=level, k=k,
                                ssim=table[level]
                                + 0.004 * rng.standard_normal(),
                                psnr_db=20.0 + 4.0 * rng.standard_normal())
                for level in DENSE:
                    for k in range(n_k):
                        row(arm="I", geometry=geometry, view=view,
                            fraction=level, k=k, convention="corrected",
                            ssim=float(np.clip(0.10 + 0.9 / (1.0 + np.exp(
                                -3.2 * (np.log10(level) + 1.1))), 0.0, 1.0))
                            + 0.003 * rng.standard_normal(),
                            psnr_db=22.0)
    return path


def _write_octiles(path):
    rng = np.random.default_rng(11)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OCTILE_COLUMNS)
        writer.writeheader()
        for arm in ("D", "E", "F"):
            for octile in range(8):
                length = 20.0 + 12.0 * octile
                writer.writerow({
                    "split": "val", "arm": arm, "model_run": "",
                    "geometry": "geo-val-01", "view": 0, "fraction": 0.05,
                    "k": 0, "anomaly_idx": 0, "anomaly_kind": "cavity",
                    "octile": octile, "n_roi": 60,
                    "l_lo_m": length - 6, "l_hi_m": length + 6,
                    "l_mean_m": length,
                    "delta_opacity_mean": -0.5 * length * 100.0,
                    "C": -0.02 * octile,
                    "z": -1.0 - 1.4 * octile + rng.standard_normal()})
    return path


def test_helpers_carry_the_numbers() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        rows = RT.load_metrics(_write_csv(Path(tmp) / "metrics.csv"))
        summary = RF._cell_summary(rows, "F", 0.05, "ssim", 400, 20260805)
        missing = RF._cell_summary(rows, "Z", 0.05, "ssim", 100, 1)
    checks = {
        "the cell mean tracks the frozen value":
            abs(summary["estimate"] - ARM_LEVEL_SSIM["F"][0.05]) < 0.005,
        "18 cells are recovered": summary["n_cells"] == 18,
        "the interval brackets the estimate":
            summary["lo"] <= summary["estimate"] <= summary["hi"],
        "all 72 pairs are kept for the strip plot":
            summary["pairs"].size == 18 * 4,
        "an absent arm returns None": missing is None,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  helpers: estimate {summary['estimate']:.4f} "
          f"[{summary['lo']:.4f}, {summary['hi']:.4f}] over "
          f"{summary['n_cells']} cells -> {'ok' if ok else 'FAIL'}")
    return ok


def _factors_payload():
    """A minimal factor_analysis.to_json() payload, built by hand.

    The figure must consume the analysis, so the test hands it exactly the
    shape factor_analysis emits: a curve summary per (convention, metric) and
    the solved rows. The projection drawn is whatever `f_prime` says, which is
    the point -- the figure has no root finder of its own any more.
    """
    fractions = list(DENSE)
    means = [float(np.clip(0.10 + 0.9 / (1.0 + np.exp(
        -3.2 * (np.log10(f) + 1.1))), 0.0, 1.0)) for f in fractions]
    rows = []
    for level, f_prime in ((0.50, 0.75), (0.10, 0.30), (0.05, 0.20),
                           (0.01, 0.10)):
        rows.append({
            "method": "cWGAN-GP", "arm": "F", "level": level,
            "convention": "corrected", "metric": "ssim", "status": "factor",
            "f_prime": f_prime, "factor": f_prime / level,
            "incremental_factor": f_prime / level - 1.0,
            "equivalent_days": 30.0 * f_prime,
            "structural_cap": 1.0 / level,
            "score": float(np.interp(np.log10(f_prime),
                                     np.log10(fractions), means)),
            "n_cells": 18,
            "bootstrap": {"b": 200, "lo": f_prime / level * 0.9,
                          "hi": f_prime / level * 1.1, "two_stage": True},
        })
    return {"curves": {"corrected|ssim": {"fractions": fractions,
                                          "mean_curve": means, "n_cells": 18}},
            "factor_rows": rows}


def test_figures_are_written() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = RT.load_metrics(_write_csv(root / "metrics.csv"))
        octiles = RF.load_octiles(_write_octiles(root / "octiles.csv"))
        out = root / "figures"
        out.mkdir()
        counts = {
            "quality_vs_level": RF.figure_quality_vs_level(
                rows, out / "stage20_quality_vs_level.png", b=150,
                ceiling=0.905),
            "per_sample": RF.figure_per_sample(
                rows, out / "stage20_per_sample.png"),
            "equivalence": RF.figure_equivalence(
                _factors_payload(), out / "stage22_equivalence.png"),
            "detectability": RF.figure_detectability(
                octiles, out / "stage21_detectability.png", z_thr=4.2),
            "gallery": RF.figure_gallery(
                [{"label": "entrada", "image": np.random.default_rng(1).random(
                    (32, 32)), "contour": None, "caption": "C=0.20"},
                 {"label": "referencia", "image": np.random.default_rng(2).random(
                     (32, 32)), "contour": np.zeros((32, 32), bool),
                  "caption": "z=9.1"}],
                out / "stage21_gallery.png"),
        }
        empty = RF.figure_equivalence({"curves": {}, "factor_rows": []},
                                      out / "unused.png")
        files = sorted(p.name for p in out.glob("*.png"))
        sizes = {p.name: p.stat().st_size for p in out.glob("*.png")}

    checks = {
        "five figures written": len(files) == 5,
        "all five families reported drawing something":
            all(count > 0 for count in counts.values()),
        "five arms on the quality curve": counts["quality_vs_level"] == 5,
        "five violins": counts["per_sample"] == 5,
        "three arms on the detectability plot": counts["detectability"] == 3,
        "no file is empty": all(size > 5_000 for size in sizes.values()),
        "names follow the stage2X convention": files == [
            "stage20_per_sample.png", "stage20_quality_vs_level.png",
            "stage21_detectability.png", "stage21_gallery.png",
            "stage22_equivalence.png"],
        "an empty factor payload draws nothing": empty == 0,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL (files={files}, counts={counts})")
    print(f"  figures: {len(files)} PNG "
          f"({min(sizes.values()) // 1024}-{max(sizes.values()) // 1024} KiB) "
          f"-> {'ok' if ok else 'FAIL'}")
    return ok


def test_ceiling_and_gallery_wiring() -> bool:
    """The ceiling is READ from ceiling.json; the gallery is built from cache."""
    import cache as C
    import run_evaluation as RE

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # the regenerated ceiling, exactly as evaluation/ceiling.py writes it
        (root / "ceiling.json").write_text(
            json.dumps({"ssim_cell_mean": 0.9047, "ssim_cell_sd": 0.0021}),
            encoding="utf-8")
        ceiling = RF.load_ceiling(root / "ceiling.json")
        absent = RF.load_ceiling(root / "not-there.json")

        # a one-geometry, one-view cache with every gallery arm
        cache = root / "cache"
        geometry = "geo-val-01"
        rng = np.random.default_rng(3)
        ref_counts = rng.poisson(2500.0, size=(32, 32)).astype(np.uint32)
        C.save_masks(cache / geometry / "masks.npz",
                     views=np.array([0], dtype=np.int32),
                     rois=np.ones((1, 1, 32, 32), dtype=bool),
                     ref_counts=ref_counts[np.newaxis])
        C.save_predictions(cache / geometry / "arm_A.npz",
                           rng.random((1, 32, 32)).astype(np.float32),
                           view=np.array([0]), fraction=np.array([1.0]),
                           k=np.array([-1]),
                           model_run=np.array([""], dtype="U32"))
        for arm in ("D", "E", "F"):
            C.save_predictions(cache / geometry / f"arm_{arm}.npz",
                               rng.random((1, 32, 32)).astype(np.float32),
                               view=np.array([0]), fraction=np.array([0.05]),
                               k=np.array([0]),
                               model_run=np.array(["run"], dtype="U32"))

        case = {"geometry": geometry, "view": 0, "fraction": 0.05, "k": 0,
                "anomaly_idx": 0, "anomaly_kind": "cavity", "arm": "F",
                "C": 0.21, "CNR": 3.4, "z": 8.1, "local_ssim": 0.42}
        panels = gallery = RF.gallery_panels(cache, case)
        out = root / "figures"
        out.mkdir()
        drawn = RF.figure_gallery(panels, out / "stage21_gallery_01.png",
                                  "peor caso")

        # the worst-case picker must order by local SSIM and skip invalid ROIs
        metric_rows = [
            {"arm": "F", "scope": "roi", "roi_valid": 1, "local_ssim": 0.9,
             "geometry": geometry, "view": 0},
            {"arm": "F", "scope": "roi", "roi_valid": 1, "local_ssim": 0.4,
             "geometry": geometry, "view": 1},
            {"arm": "F", "scope": "roi", "roi_valid": 0, "local_ssim": 0.1,
             "geometry": geometry, "view": 2},
        ]
        picked = RF.worst_cases(metric_rows, "F", limit=2)

    labels = [panel["label"] for panel in gallery]
    checks = {
        "the ceiling is read from the JSON": abs(ceiling - 0.9047) < 1e-12,
        "a missing ceiling file is None, not a literal": absent is None,
        "the gallery has an input panel": any(
            label.startswith("entrada") for label in labels),
        "the input panel says it was regenerated": any(
            "regenerada" in (panel.get("caption") or "") for panel in gallery),
        "the gallery has classical, L1, cWGAN and the reference":
            {"clásica", "U-Net L1", "cWGAN-GP", "referencia 30 d"}
            <= set(labels),
        "the ROI contour travels with every panel": all(
            panel["contour"] is not None for panel in gallery),
        "the case statistics are printed on its own arm": any(
            "z=8.1" in (panel.get("caption") or "") for panel in gallery),
        "the figure is drawn": drawn == len(panels),
        "the worst cases come first":
            [row["local_ssim"] for row in picked] == [0.4, 0.9],
        "an invalid ROI is not a worst case": all(
            row["roi_valid"] == 1 for row in picked),
        "the arm-I seed is the one the driver uses":
            RE.thinning_seed(2, 0, 0, 0).entropy
            == [20260805 * 10 ** 6 + 22 * 10 ** 4 + 2 * 10 ** 2, 0, 0],
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL (panels={labels})")
    print(f"  ceiling and gallery: {len(gallery)} panels {labels} -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_cli() -> bool:
    """The CLI entry point, called in-process (a subprocess would re-pay the
    matplotlib and SciPy import cost for nothing)."""
    import contextlib
    import io

    saved_argv = sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "cache"
        cache.mkdir()
        _write_csv(cache / "metrics.csv", n_geometries=1, n_views=2, n_k=1)
        _write_octiles(cache / "octiles.csv")
        # the CLI must PREFER the factor tables on disk over recomputing them,
        # so the figure and the Stage 22 table are the same numbers
        factors_path = Path(tmp) / "factor_tables.json"
        factors_path.write_text(json.dumps(_factors_payload()),
                                encoding="utf-8")
        ceiling_path = Path(tmp) / "ceiling.json"
        ceiling_path.write_text(json.dumps({"ssim_cell_mean": 0.9051}),
                                encoding="utf-8")
        out = Path(tmp) / "out"
        buffer = io.StringIO()
        try:
            sys.argv = ["report_figures.py", "--cache", str(cache),
                        "--out", str(out), "--bootstrap", "60",
                        "--z-thr", "4.2",
                        "--ceiling-json", str(ceiling_path),
                        "--factors", str(factors_path)]
            with contextlib.redirect_stdout(buffer):
                status = RF.main()
        finally:
            sys.argv = saved_argv
        files = sorted(p.name for p in out.glob("*.png")) if out.exists() else []
    checks = {
        "the CLI exits cleanly": status == 0,
        "four figures emitted": len(files) == 4,
        "the CLI reports what it wrote": "wrote" in buffer.getvalue(),
        "the CLI reuses the factor tables instead of recomputing":
            "factor_tables.json" in buffer.getvalue(),
        "the CLI reports the ceiling it read":
            "0.9051" in buffer.getvalue(),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL (files={files})")
    print(f"  CLI: {files} -> {'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("Stage 19 figure tests (matplotlib Agg, synthetic tables)")
    results = {
        "helpers": test_helpers_carry_the_numbers(),
        "figures": test_figures_are_written(),
        "ceiling_and_gallery": test_ceiling_and_gallery_wiring(),
        "cli": test_cli(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 19 REPORT FIGURES:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
