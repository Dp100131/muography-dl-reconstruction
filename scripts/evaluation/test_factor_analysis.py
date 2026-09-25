#!/usr/bin/env python3
"""Stage 22 factor tests — a curve whose inverse is known in closed form.

The fixture's dense grid follows q(f) = a + b log10(f) EXACTLY. PCHIP
reproduces data that is linear in its abscissa, and the abscissa here is
log10 f', so the interpolated curve IS the line and Brent must return the
analytic root to machine precision. Each enhanced arm is then handed the value
of that line at a chosen f', so the expected factor N = f'/f is an integer
written into the fixture rather than a number read off the output.

Run:  python scripts/evaluation/test_factor_analysis.py
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

import equivalence as EQ  # noqa: E402
import factor_analysis as FA  # noqa: E402
import report_tables as RT  # noqa: E402

GEOMETRIES = ("geo-val-01", "geo-val-02")
VIEWS = (0, 1)
KS = (0, 1, 2)
INTERCEPT, SLOPE = 0.30, 0.35          # ssim  = 0.30 + 0.35 (log10 f + 2)
PSNR_INTERCEPT, PSNR_SLOPE = 12.0, 6.0  # psnr = 12.0 + 6.0 (log10 f + 2)

# level -> {arm: the f' whose curve value the arm is given}. The factor is
# therefore f'/level, exactly.
TARGETS = {
    0.05: {"C": 0.05, "D": 0.10, "E": 0.15, "F": 0.20},
    0.01: {"D": 0.05, "E": 0.05},
}
# CNR per anomaly type at f = 0.05: arm C is the denominator of the ratio
CNR = {"C": {"cavity": 2.0, "conduit": 1.0},
       "F": {"cavity": 4.0, "conduit": 3.0}}

COLUMNS = ["split", "arm", "arm_label", "model_run", "geometry",
           "geometry_split", "geo_index", "view", "fraction", "model_fraction",
           "convention", "k", "scope", "anomaly_idx", "anomaly_kind",
           "roi_valid", "ssim", "psnr_db", "C", "CNR", "z", "notes"]


def curve(fraction: float, metric: str = "ssim") -> float:
    log_f = float(np.log10(fraction))
    if metric == "ssim":
        return INTERCEPT + SLOPE * (log_f + 2.0)
    return PSNR_INTERCEPT + PSNR_SLOPE * (log_f + 2.0)


def _write_csv(path):
    rows = []

    def row(**values):
        base = {column: "" for column in COLUMNS}
        base.update({"split": "val", "scope": "global", "anomaly_idx": -1})
        base.update(values)
        rows.append(base)

    for geometry in GEOMETRIES:
        for view in VIEWS:
            for fraction in EQ.DENSE_FRACTIONS:
                for k in KS:
                    for convention in ("raw", "corrected"):
                        row(arm="I", geometry=geometry, view=view,
                            fraction=f"{fraction:.10g}", k=k,
                            convention=convention,
                            ssim=f"{curve(fraction, 'ssim'):.12g}",
                            psnr_db=f"{curve(fraction, 'psnr_db'):.12g}")
            for level, targets in TARGETS.items():
                for arm, target in targets.items():
                    for k in KS:
                        row(arm=arm, geometry=geometry, view=view,
                            fraction=f"{level:.10g}", k=k,
                            convention="corrected",
                            ssim=f"{curve(target, 'ssim'):.12g}",
                            psnr_db=f"{curve(target, 'psnr_db'):.12g}")
            # the two boundary cases of section 6 step 8
            for k in KS:
                row(arm="F", geometry=geometry, view=view, fraction="0.01",
                    k=k, convention="corrected", ssim="0.99", psnr_db="30")
                row(arm="C", geometry=geometry, view=view, fraction="0.01",
                    k=k, convention="corrected", ssim="0.10", psnr_db="5")
            # ROI rows for the per-anomaly-type CNR factors
            for arm, per_kind in CNR.items():
                for anomaly_idx, (kind, value) in enumerate(per_kind.items()):
                    for k in KS:
                        row(arm=arm, geometry=geometry, view=view,
                            fraction="0.05", k=k, scope="roi",
                            anomaly_idx=anomaly_idx, anomaly_kind=kind,
                            roi_valid=1, CNR=f"{value:.10g}", C="0.2",
                            z="9.0")
            # an excluded ROI must never enter the CNR ratio
            row(arm="F", geometry=geometry, view=view, fraction="0.05", k=0,
                scope="roi", anomaly_idx=9, anomaly_kind="cavity",
                roi_valid=0, CNR="nan", C="nan", z="nan")

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


TEST_B = 60          # the interval is not what these tests assert; see B_DEFAULT
_ANALYSIS = None


def _analysis():
    """Built once: the same fixture drives both assertion sets."""
    global _ANALYSIS
    if _ANALYSIS is None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            n_rows = _write_csv(path)
            rows = RT.load_metrics(path)
            _ANALYSIS = (FA.FactorAnalysis(rows, b=TEST_B, seed=1), n_rows)
    return _ANALYSIS


def test_known_curve_gives_the_known_factor() -> bool:
    """The whole point: an analytic curve must invert to the analytic factor."""
    analysis, n_rows = _analysis()

    by_key = {(row["level"], row["method"], row["convention"], row["metric"]):
              row for row in analysis.rows}
    checks = {}
    for metric in ("ssim", "psnr_db"):
        for arm, target in TARGETS[0.05].items():
            method = next(name for name, a in FA.METHODS if a == arm)
            item = by_key[(0.05, method, "corrected", metric)]
            expected = target / 0.05
            checks[f"f' is exact ({metric}, {method})"] = (
                abs(item["f_prime"] - target) < 1e-8)
            checks[f"N = f'/f ({metric}, {method})"] = (
                abs(item["factor"] - expected) < 1e-6)
            checks[f"days = 30 f' ({metric}, {method})"] = (
                abs(item["equivalent_days"] - 30.0 * target) < 1e-6)

    identity = by_key[(0.05, FA.RAW_IDENTITY, "corrected", "ssim")]
    cwgan = by_key[(0.05, "cWGAN-GP", "corrected", "ssim")]
    classical = by_key[(0.05, "mejor clásica", "corrected", "ssim")]
    checks["the raw identity really is an identity"] = (
        abs(identity["factor"] - 1.0) < 1e-6)
    checks["the incremental column subtracts the raw identity"] = (
        abs(cwgan["incremental_factor"] - 3.0) < 1e-6
        and abs(classical["incremental_factor"] - 1.0) < 1e-6)
    checks["the structural cap travels in every row"] = all(
        np.isfinite(row["structural_cap"]) for row in analysis.rows)
    checks["the cap is 1/f from the frozen table"] = (
        cwgan["structural_cap"] == 20.0
        and by_key[(0.01, "U-Net L1", "corrected", "ssim")]["structural_cap"]
        == 100.0)

    # both conventions produce a curve, and the raw one is fitted too
    checks["both conventions are fitted"] = (
        analysis.curve("raw", "ssim") is not None
        and analysis.curve("corrected", "ssim") is not None)
    checks["the raw convention gives the same analytic factor"] = abs(
        by_key[(0.05, "cWGAN-GP", "raw", "ssim")]["factor"] - 4.0) < 1e-6

    # the PAVA guard must find nothing to repair on a monotone fixture
    summary = analysis.curve("corrected", "ssim").summary()
    checks["PAVA leaves a monotone curve alone"] = (
        summary["n_cells_adjusted_by_pava"] == 0
        and summary["monotone_after_pava"])
    checks["the curve carries four cells"] = summary["n_cells"] == 4

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  known curve ({n_rows} rows): f'={cwgan['f_prime']:.6f} "
          f"N={cwgan['factor']:.4f} (expected 4) incremental "
          f"{cwgan['incremental_factor']:.4f} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_boundaries_and_bootstrap() -> bool:
    """Boundary statuses, the two-stage draw and the CNR anchor."""
    analysis, _n_rows = _analysis()

    by_key = {(row["level"], row["method"], row["convention"], row["metric"]):
              row for row in analysis.rows}
    above = by_key[(0.01, "cWGAN-GP", "corrected", "ssim")]
    below = by_key[(0.01, FA.RAW_IDENTITY, "corrected", "ssim")]
    missing = by_key[(0.50, "cWGAN-GP", "corrected", "ssim")]
    cwgan = by_key[(0.05, "cWGAN-GP", "corrected", "ssim")]

    checks = {
        "a score above the grid is a LOWER BOUND, not a factor":
            above["status"] == EQ.LOWER_BOUND
            and abs(above["f_prime"] - 0.90) < 1e-12,
        "a score below the grid is NO GAIN":
            below["status"] == EQ.NO_GAIN,
        "a level with no enhanced rows says so":
            missing["status"] == "sin datos" and missing["n_cells"] == 0,
        "the bootstrap is the TWO-STAGE cluster scheme":
            cwgan["bootstrap"]["two_stage"] is True
            and cwgan["bootstrap"]["k_per_cell"] == len(KS),
        "B travels with the result":
            cwgan["bootstrap"]["b"] == TEST_B and analysis.b == TEST_B,
        "the interval brackets the point estimate":
            cwgan["bootstrap"]["lo"] <= cwgan["factor"]
            <= cwgan["bootstrap"]["hi"],
        "the default B is the pre-registered 10 000": FA.B_DEFAULT == 10_000,
    }

    # per-anomaly-type CNR, from the analytic Poisson anchor
    cnr = {(row["level"], row["method"], row["anomaly_kind"]): row
           for row in analysis.cnr_rows}
    cavity = cnr.get((0.05, "cWGAN-GP", "cavity"))
    conduit = cnr.get((0.05, "cWGAN-GP", "conduit"))
    checks["a CNR factor exists per anomaly type"] = (
        cavity is not None and conduit is not None)
    if cavity and conduit:
        checks["N_CNR is the squared CNR ratio"] = (
            abs(cavity["factor"] - 4.0) < 1e-9
            and abs(conduit["factor"] - 9.0) < 1e-9)
        checks["the CNR rows carry the structural cap"] = (
            cavity["structural_cap"] == 20.0)
        checks["excluded ROIs stay out of the CNR ratio"] = (
            cavity["n_cells"] == 4)

    text = FA.markdown_tables(analysis)
    checks["every factor table prints the caps"] = (
        "Cota estructural" in text and "f=0.01 → 100×" in text)
    checks["the incremental column is in the table"] = (
        "N incremental" in text)
    checks["the CNR table names the analytic anchor"] = (
        "(razón CNR)²" in text)
    checks["the bootstrap scheme is stated with its B"] = (
        "dos etapas" in text and f"B = {TEST_B}" in text)
    payload = json.loads(json.dumps(analysis.to_json(), default=float))
    checks["the JSON carries the rows and the scheme"] = (
        len(payload["factor_rows"]) == len(analysis.rows)
        and payload["bootstrap"]["b"] == TEST_B)

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  boundaries and bootstrap: {sum(checks.values())}/{len(checks)} "
          f"checks -> {'ok' if ok else 'FAIL'}")
    return ok


def test_cli() -> bool:
    """The CLI writes both artefacts and refuses an unfrozen test cache."""
    import contextlib
    import io

    saved_argv = sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "cache"
        cache.mkdir()
        _write_csv(cache / "metrics.csv")
        (cache / "summary.json").write_text('{"split": "val"}',
                                            encoding="utf-8")
        out = Path(tmp) / "artifacts"
        buffer = io.StringIO()
        try:
            sys.argv = ["factor_analysis.py", "--cache", str(cache), "--out",
                        str(out), "--bootstrap", "20"]
            with contextlib.redirect_stdout(buffer):
                status = FA.main()
        finally:
            sys.argv = saved_argv
        names = sorted(p.name for p in out.glob("*")) if out.exists() else []

        # a TEST cache with PROVISIONAL thresholds must abort before writing.
        # The repo now ships a frozen protocol_thresholds.json, so force the
        # provisional condition for this assertion rather than relying on its
        # absence.
        (cache / "summary.json").write_text('{"split": "test"}',
                                            encoding="utf-8")
        aborted = False
        saved_loader = FA.RT.load_thresholds
        FA.RT.load_thresholds = lambda *a, **k: {"provisional": True}
        try:
            FA.build(cache, b=10)
        except RuntimeError:
            aborted = True
        finally:
            FA.RT.load_thresholds = saved_loader

    checks = {
        "the CLI exits cleanly": status == 0,
        "both artefacts are written":
            names == ["factor_tables.json", "factor_tables.md"],
        "the CLI reports B": "B = 20" in buffer.getvalue(),
        "a test cache without frozen thresholds aborts": aborted,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL (files={names})")
    print(f"  CLI: {names} -> {'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("Stage 22 factor-analysis tests (analytic curve, analytic factor)")
    results = {
        "known_curve": test_known_curve_gives_the_known_factor(),
        "boundaries": test_boundaries_and_bootstrap(),
        "cli": test_cli(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 22 FACTOR ANALYSIS TESTS:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
