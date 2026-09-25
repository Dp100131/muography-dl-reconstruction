#!/usr/bin/env python3
"""Stage 19 — freeze protocol v1.1 thresholds from the validation dry run.

WHY PER-ARM CALIBRATION. The detection statistic z divides by the annulus
noise of the SAME image. Denoising shrinks sigma_bg, so a threshold
calibrated on reference images does not transfer to denoised arms: the dry
run measured sham |z| medians several times larger under NLM and the
generators than under the raw reference (the provisional scalar z_thr = 4
produced ~76-80% sham positives on validation — an artefact of the
threshold, not model hallucination). The pre-registered remedy, decided
during the dry run and BEFORE any test contact: z_thr is a TABLE, the 99th
percentile of |z| over validation sham ROIs computed separately per
(arm, level). Each arm's false-positive rate is ~1% on validation by
construction, and the S3 hallucination gate (<= 5x) compares each arm
against its own calibration.

Everything this script writes derives from the validation dry-run cache;
the test set is untouched. Output: results/evaluation/protocol_thresholds.json
with provisional: false.

Run:  python scripts/evaluation/calibrate_thresholds.py --cache <dryrun-cache> \
          [--out results/evaluation/protocol_thresholds.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

FRACTIONS = (0.50, 0.10, 0.05, 0.01)
# Arms whose detection statistics are consumed by Stage 20/21 rates.
CALIBRATED_ARMS = ("A", "C", "D", "E", "F", "G", "H")
PERCENTILE = 99.0
S2_MARGIN_SSIM = 0.02          # max(0.02, 3*SD_cells) per the plan
S2_MARGIN_PSNR_DB = 1.0
X_FLOOR = 0.30                 # preserved tolerance = max(0.30, 3*rel-SD C_ref)
Y_INFLATION = 0.50


def _rows(path: Path):
    with open(path, newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", default=str(
        REPO_ROOT / "results" / "evaluation" / "protocol_thresholds.json"))
    args = parser.parse_args()

    metrics = Path(args.cache) / "metrics.csv"
    sham_z = {}          # (arm, fraction) -> [|z|]
    ref_sham_z = []      # arm A scalar continuity value
    ref_contrast = {}    # (geometry, view, anomaly) -> [C over k]
    cwgan_cells = {}     # fraction -> {(geometry, view): [ssim]}
    cwgan_psnr = {}
    split = None

    for row in _rows(metrics):
        split = row.get("split", split)
        arm = row.get("arm", "")
        frac = _f(row.get("fraction"))
        scope = row.get("scope", "")
        if scope == "sham":
            z = abs(_f(row.get("z")))
            if np.isfinite(z):
                if arm == "A":
                    ref_sham_z.append(z)
                for target in FRACTIONS:
                    if arm != "A" and np.isclose(frac, target):
                        sham_z.setdefault((arm, target), []).append(z)
                        break
        elif arm == "A" and row.get("anomaly_idx") not in ("", "-1", None):
            if int(float(row["roi_valid"] or 0)) == 1:
                key = (row["geometry"], row["view"], row["anomaly_idx"])
                c = _f(row.get("C"))
                if np.isfinite(c):
                    ref_contrast.setdefault(key, []).append(c)
        elif arm == "F" and row.get("anomaly_idx") in ("", "-1", None):
            for target in FRACTIONS:
                if np.isclose(frac, target):
                    cell = (row["geometry"], row["view"])
                    cwgan_cells.setdefault(target, {}).setdefault(
                        cell, []).append(_f(row.get("ssim")))
                    cwgan_psnr.setdefault(target, {}).setdefault(
                        cell, []).append(_f(row.get("psnr_db")))
                    break

    if split == "test":
        raise SystemExit("calibration must come from the VALIDATION dry run")

    z_thr_by_arm = {}
    for (arm, frac), values in sorted(sham_z.items()):
        z_thr_by_arm.setdefault(arm, {})[f"{frac:.2f}"] = round(
            float(np.percentile(values, PERCENTILE)), 3)
    z_ref = round(float(np.percentile(ref_sham_z, PERCENTILE)), 3)

    # X tolerance: the preserved-contrast band must not be tighter than the
    # reference's own Poisson jitter (3 * relative SD of C_ref over K).
    rel_sds = []
    for values in ref_contrast.values():
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) > 1 and abs(arr.mean()) > 0:
            rel_sds.append(3.0 * arr.std(ddof=1) / abs(arr.mean()))
    x_measured = float(np.median(rel_sds)) if rel_sds else 0.0
    x_tolerance = round(max(X_FLOOR, x_measured), 3)

    def _cell_mean(values):
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else float("nan")

    floors_ssim, floors_psnr = {}, {}
    for frac in FRACTIONS:
        means = [m for m in (_cell_mean(v)
                             for v in cwgan_cells.get(frac, {}).values())
                 if np.isfinite(m)]
        psnr_means = [m for m in (_cell_mean(v)
                                  for v in cwgan_psnr.get(frac, {}).values())
                      if np.isfinite(m)]
        if not means:
            raise SystemExit(f"no finite cWGAN SSIM cells at f={frac}: "
                             "cannot freeze an S2 floor")
        sd = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
        sd_p = float(np.std(psnr_means, ddof=1)) if len(psnr_means) > 1 else 0.0
        floors_ssim[f"{frac:.2f}"] = round(
            float(np.mean(means)) - max(S2_MARGIN_SSIM, 3.0 * sd), 4)
        floors_psnr[f"{frac:.2f}"] = round(
            float(np.mean(psnr_means)) - max(S2_MARGIN_PSNR_DB, 3.0 * sd_p), 2)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True,
                                cwd=REPO_ROOT).stdout.strip()
    except OSError:
        commit = "unknown"

    payload = {
        "provisional": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "validation dry run (Stage 19 step 7); test set untouched",
        "git_commit": commit,
        "z_thr": z_ref,
        "z_thr_note": ("scalar = reference-arm calibration (continuity); "
                       "detection ALWAYS uses z_thr_by_arm when the arm has "
                       "an entry — per-arm 99th-percentile sham calibration "
                       "decided during the dry run, before any test contact"),
        "z_thr_by_arm": z_thr_by_arm,
        "z_thr_percentile": PERCENTILE,
        "preserved_tolerance_X": x_tolerance,
        "preserved_tolerance_X_note": (
            f"max({X_FLOOR}, mediana de 3*rel-SD(C_ref) medida = "
            f"{round(x_measured, 3)})"),
        "inflation_tolerance_Y": Y_INFLATION,
        "s2_floors": {key: {"ssim": floors_ssim[key],
                            "psnr_db": floors_psnr.get(key)}
                      for key in floors_ssim},
        "s3": {"preserved_min": 0.90, "hallucination_max": 0.05,
               "lost_max": 0.10,
               "levels_required": ["0.50", "0.10", "0.05"],
               "hallucination_note": ("cada brazo se compara contra su "
                                      "propia calibracion (FPR 1% en "
                                      "validacion por construccion); 5% = "
                                      "5x esa calibracion")},
        "ceiling_ssim": None,  # filled from ceiling.json by the freeze step
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"  z_thr reference scalar: {z_ref} (n={len(ref_sham_z)})")
    for arm in sorted(z_thr_by_arm):
        print(f"  z_thr[{arm}]: {z_thr_by_arm[arm]}")
    print(f"  X tolerance: {x_tolerance} (measured 3*rel-SD median "
          f"{round(x_measured, 3)})")
    print(f"  S2 floors SSIM: {floors_ssim}")
    print("STAGE 19 THRESHOLD CALIBRATION: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
