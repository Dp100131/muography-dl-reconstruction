#!/usr/bin/env python3
"""Stage 20 — THE evaluation driver. One pass, maximum cache.

This is the ONLY file in the repository authorized to set the held-out guard
MUOGRAPHY_ALLOW_TEST (protocol v1.1 section 5.4). Every other module takes
arrays.

    --split val    free to use, as often as needed: this is how Stage 19 is
                   developed and how every threshold of protocol v1.1 is
                   calibrated before anything irreversible happens
    --split test   opens the test split AND geo-gen-01, exactly once, and
                   records the fact in results/evaluation/test_access_log.json:
                   timestamp, git commit, argv, host, SLURM job id, then the
                   SHA-256 of every shard read and every cache file written

PRINCIPLE: WHAT IS NOT CACHED IS UNMEASURABLE FOREVER. Stages 21 and 22 are
pure post-processing of this cache -- zero additional reads -- so anything a
later analysis might want has to be written here.

Arms (execution plan section 2.2):

  A  reference               the 30-day image; source of C_ref and of the
                             sham statistics that calibrate z_thr
  B  raw, shared scale       no model, no amplitude correction
  C  raw, 1/f corrected      no model, deterministic amplitude restoration
  D  classical, frozen       the Stage 13 winner per level, 1/f corrected,
                             K = 20 (selected on validation with K = 5; the
                             per-level oracle biases IN FAVOUR of the baseline
                             and is reported as conservative)
  E  U-Net L1                the Stage 19.5 retrains
  F  cWGAN-GP                the Stage 17 finals, one model per level
  G  seed replicas           e17-final-f5-s1/s2 at f = 0.05
  H  cross-level matrix      every cWGAN on every level, EXPLORATORY (the
                             diagonal is arm F and is not recomputed)
  I  dense thinning grid     9 fractions x K = 20, metrics only, seeded so the
                             whole grid is regenerable bit for bit

CRASH PROTOCOL. A job that dies half way through the test pass has already
opened the split, so the run has to be finishable without pretending to open
it twice. `--resume` keeps every geometry whose cache files AND tidy rows are
both present and verified against the manifest, redoes the rest, appends to
the tables instead of truncating them, and writes a `restart` event carrying
the reason and the sha256 of every file the dead attempt left behind.

Run:  python scripts/evaluation/run_evaluation.py --split val --skip-gpu
      python scripts/evaluation/run_evaluation.py --split val \
          --out-root results/evaluation/val
      python scripts/evaluation/run_evaluation.py --split test \
          --out-root "$MUOGRAPHY_RUNS/e20-cache"          # ONCE. EVER.
      python scripts/evaluation/run_evaluation.py --split test \
          --out-root "$MUOGRAPHY_RUNS/e20-cache" \
          --resume --resume-reason "SLURM time limit on geo-test-03"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

import cache as cache_module  # noqa: E402
import detectability as det  # noqa: E402
import predict as pred  # noqa: E402
import roi as roi_module  # noqa: E402
import texture as tex  # noqa: E402
from metrics import compute_psnr, compute_ssim  # noqa: E402
from muography_dataset import (ALLOW_TEST_ENV, SHARD_DIR,  # noqa: E402
                               geometry_ids_for_split, load_shard,
                               preprocess_pair, preprocess_pair_corrected)
from surrogate.imaging import DetectorView  # noqa: E402

ACCESS_LOG = REPO_ROOT / "results" / "evaluation" / "test_access_log.json"
DEFAULT_CACHE_ROOT = REPO_ROOT / "results" / "evaluation"

MASTER_SEED = 20260805
THINNING_STAGE = 22          # section 6 step 2: SeedSequence([seed(22, fi, k), geo, view])
SHAM_STAGE = 21              # sham masks share the Stage 21 stream
K_REALIZATIONS = 20
N_SHAM_PER_VIEW = 10
FROZEN_FRACTIONS = (0.50, 0.10, 0.05, 0.01)
DENSE_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 0.90)
TEXTURE_FRACTIONS = (0.05, 0.01)

SHARED = "shared"
REF_EQUIVALENT = "ref_equivalent"
RAW = "raw"
CORRECTED = "corrected"

# --- arms -------------------------------------------------------------------
ARM_REFERENCE = "A"
ARM_RAW_SHARED = "B"
ARM_RAW_CORRECTED = "C"
ARM_CLASSICAL = "D"
ARM_UNET_L1 = "E"
ARM_CWGAN = "F"
ARM_SEED_REPLICAS = "G"
ARM_CROSS_LEVEL = "H"
ARM_THINNING_GRID = "I"

ALL_ARMS = (ARM_REFERENCE, ARM_RAW_SHARED, ARM_RAW_CORRECTED, ARM_CLASSICAL,
            ARM_UNET_L1, ARM_CWGAN, ARM_SEED_REPLICAS, ARM_CROSS_LEVEL,
            ARM_THINNING_GRID)
GPU_ARMS = (ARM_UNET_L1, ARM_CWGAN, ARM_SEED_REPLICAS, ARM_CROSS_LEVEL)
CACHED_ARMS = (ARM_REFERENCE, ARM_CLASSICAL, ARM_UNET_L1, ARM_CWGAN,
               ARM_SEED_REPLICAS, ARM_CROSS_LEVEL)
SHAM_ARMS = (ARM_REFERENCE, ARM_RAW_CORRECTED, ARM_CLASSICAL, ARM_UNET_L1,
             ARM_CWGAN)
OCTILE_ARMS = (ARM_REFERENCE, ARM_RAW_CORRECTED, ARM_CLASSICAL, ARM_UNET_L1,
               ARM_CWGAN)
# Texture is the pre-registered arbiter of the 1 % verdict (Stage 18), and an
# arbiter needs a scale. "The cWGAN restores more high-frequency residual power
# than the U-Net L1" is only interpretable against the un-enhanced input (arm C)
# and against the frozen classical (arm D): without them the secondary family
# has no vs-raw and no vs-classical comparison and the two numbers float free.
TEXTURE_ARMS = (ARM_RAW_CORRECTED, ARM_CLASSICAL, ARM_UNET_L1, ARM_CWGAN)

ARM_LABELS = {
    ARM_REFERENCE: "referencia 30 d",
    ARM_RAW_SHARED: "crudo (escala compartida)",
    ARM_RAW_CORRECTED: "crudo (corregido 1/f)",
    ARM_CLASSICAL: "clasica congelada (corregido 1/f)",
    ARM_UNET_L1: "U-Net L1",
    ARM_CWGAN: "cWGAN-GP",
    ARM_SEED_REPLICAS: "replicas de semilla (f=0.05)",
    ARM_CROSS_LEVEL: "matriz cruzada (exploratoria)",
    ARM_THINNING_GRID: "grilla densa de thinning",
}

# Frozen classical winners (execution plan section 2.2, protocol v1.1 H12).
# Selected on validation with K = 5; run here with K = 20.
FROZEN_CLASSICAL = {
    0.50: "nlm_log_h0.5",
    0.10: "nlm_anscombe_h1.2",
    0.05: "nlm_anscombe_h1.2",
    0.01: "gaussian_s2.5",
}
CLASSICAL_CONVENTION = CORRECTED

UNET_L1_RUNS = {0.50: "e19-unetl1-f50", 0.10: "e19-unetl1-f10",
                0.05: "e19-unetl1-f5", 0.01: "e19-unetl1-f1"}
# PRE-REGISTERED CONTINUITY COMPARISON (execution plan 1.1 note and 2.1 E19.5).
# The Stage 18 table quotes 0.7747 at f = 0.01 for a U-Net L1 trained on the
# SHARED input contract, while the correct paired comparison against the cWGAN
# (which is ref_equivalent) is the 0.7723 twin. Both are evaluated in the same
# pass: the paired one is the primary comparison and this one carries the
# continuity with the published Stage 18 number. Declaring it here, before the
# test is opened, is what keeps it a comparison rather than a choice made
# afterwards among two available numbers.
UNET_L1_CONTINUITY_RUNS = (("e19-unetl1-f1-shared", 0.01),)
CWGAN_RUNS = {0.50: "e17-final-f50", 0.10: "e17-final-f10",
              0.05: "e17-final-f5", 0.01: "e17-final-f1"}
SEED_REPLICA_RUNS = ("e17-final-f5-s1", "e17-final-f5-s2")
SEED_REPLICA_FRACTION = 0.05

FRACTION_ATOL = 1e-9         # fractions are compared with np.isclose, never ==

CSV_COLUMNS = [
    "split", "arm", "arm_label", "model_run", "geometry", "geometry_split",
    "geo_index", "view", "fraction", "model_fraction", "convention", "k",
    "scope", "anomaly_idx", "anomaly_kind", "delta_rho", "delta_opacity_max",
    "ssim", "psnr_db", "local_ssim", "roi_pixels", "annulus_pixels",
    "roi_valid", "mean_roi", "mean_bg", "std_bg", "n_roi", "n_bg",
    "C", "CNR", "z", "variance_ratio", "nyquist_band_ratio",
    "residual_high_freq_power", "residual_low_freq_power", "residual_std",
    "notes",
]
OCTILE_COLUMNS = [
    "split", "arm", "model_run", "geometry", "view", "fraction", "k",
    "anomaly_idx", "anomaly_kind", "octile", "n_octiles", "n_roi", "l_lo_m",
    "l_hi_m", "l_mean_m", "delta_opacity_mean", "C", "z",
]


# ------------------------------------------------------- held-out guard logic
# Snapshot at IMPORT time: if the variable is already set when this process
# starts, somebody else opened the guard and this run must not proceed.
GUARD_PRESET_AT_IMPORT = os.environ.get(ALLOW_TEST_ENV) == "1"


def assert_guard_not_preset(preset: bool | None = None) -> None:
    """Refuse to run on test when the guard was opened outside this module."""
    preset = GUARD_PRESET_AT_IMPORT if preset is None else bool(preset)
    if preset:
        raise RuntimeError(
            f"{ALLOW_TEST_ENV} was already set when run_evaluation.py started. "
            f"This module is the only authorized call-site of the held-out "
            f"guard (protocol v1.1 section 5.4); a pre-set variable means the "
            f"single-use accounting is no longer trustworthy. Unset it and "
            f"re-run.")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=REPO_ROOT).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_access_log(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def access_log_open(path, argv) -> dict:
    """Append the OPENING entry before the guard is set. Append-only, always."""
    now = datetime.now(timezone.utc)
    entry = {
        "event": "open",
        # microseconds AND the pid: two opens in the same second must not
        # collide, or the close entry could not be matched to its open
        "access_id": f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{os.getpid()}",
        "utc": now.isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "argv": list(argv),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
    }
    entries = _read_access_log(path)
    entries.append(entry)
    cache_module.write_json_atomic(path, entries)
    return entry


def access_log_restart(path, argv, reason: str, partial_hashes: dict) -> dict:
    """Append a RESTART entry: the crash protocol of a single-use split.

    A job that dies half way through the test pass leaves the split already
    opened and a partial cache on disk. Re-running it as a plain second `open`
    would make the log read as two independent readings of the held-out data --
    the exact claim the log exists to defend -- when what happened was one
    reading finished in two attempts. A `restart` says so explicitly, names the
    reason and carries the sha256 of every file the previous attempt left
    behind, so the continuation is auditable: the surviving cache is evidence,
    not something to be taken on trust.
    """
    now = datetime.now(timezone.utc)
    entry = {
        "event": "restart",
        "access_id": f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{os.getpid()}",
        "utc": now.isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "argv": list(argv),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "reason": str(reason),
        "partial_cache": dict(sorted(partial_hashes.items())),
        "n_partial_files": len(partial_hashes),
    }
    entries = _read_access_log(path)
    entries.append(entry)
    cache_module.write_json_atomic(path, entries)
    return entry


def access_log_close(path, entry: dict, shard_hashes: dict,
                     cache_hashes: dict, summary: dict) -> None:
    """Append the CLOSING entry with the hash of everything read and written."""
    entries = _read_access_log(path)
    entries.append({
        "event": "close",
        "opened_by": entry.get("event", "open"),
        "access_id": entry["access_id"],
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shards_read": dict(sorted(shard_hashes.items())),
        "cache_written": dict(sorted(cache_hashes.items())),
        "summary": summary,
    })
    cache_module.write_json_atomic(path, entries)


def resolve_geometries(split: str, access_path=ACCESS_LOG, argv=None,
                       resume: bool = False, resume_reason: str = "",
                       partial_hashes=None):
    """[(geometry_id, split_label)] plus the access-log entry for a test pass.

    THIS FUNCTION HOLDS THE ONLY ASSIGNMENT OF THE HELD-OUT GUARD IN THE
    REPOSITORY. On --split test it opens the test split AND geo-gen-01,
    because the execution plan reads both in the same single pass. A resumed
    pass writes a `restart` entry instead of a second `open`.
    """
    if split == "test":
        assert_guard_not_preset()
        entry = (access_log_restart(access_path, argv or sys.argv,
                                    resume_reason, partial_hashes or {})
                 if resume else access_log_open(access_path, argv or sys.argv))
        os.environ[ALLOW_TEST_ENV] = "1"        # <- the one call-site
        pairs = [(gid, "test") for gid in geometry_ids_for_split("test")]
        pairs += [(gid, "generalization")
                  for gid in geometry_ids_for_split("generalization")]
        return pairs, entry
    return [(gid, split) for gid in geometry_ids_for_split(split)], None


# ------------------------------------------------------------ crash protocol
def partial_cache_hashes(out_root) -> dict:
    """{path relative to the cache root: sha256} of what a dead job left."""
    root = Path(out_root)
    if not root.exists():
        return {}
    return {path.relative_to(root).as_posix(): sha256_of(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def expected_cache_keys(geometry_id: str, arms) -> list:
    """The cache files a completed geometry must have left in the manifest."""
    return [f"{geometry_id}/masks.npz"] + [
        f"{geometry_id}/arm_{arm}.npz" for arm in arms if arm in CACHED_ARMS]


def geometries_in_table(metrics_path) -> set:
    """The geometry ids that already have rows in a partial metrics.csv."""
    path = Path(metrics_path)
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {row.get("geometry") for row in csv.DictReader(handle)
                if row.get("geometry")}


def completed_geometries(out_root, manifest_path, metrics_path, geometries,
                         arms) -> list:
    """Geometries whose cache AND whose tidy rows are BOTH already there.

    Both halves are required. A geometry with cache files but no rows would
    lose its metrics; one with rows but no cache would leave Stages 21 and 22
    unable to redraw it -- and the cache is written only after the whole
    geometry is done, so "cache present" is the marker of completion. Every
    listed file is also re-hashed against the manifest: a half-written npz from
    a job killed mid-write is not a completed geometry.
    """
    manifest = cache_module.read_manifest(manifest_path)
    tracked = manifest.get("files", {})
    broken = {key for key, _reason in
              cache_module.verify_manifest(manifest_path, out_root)}
    with_rows = geometries_in_table(metrics_path)
    done = []
    for geometry_id, _split in geometries:
        keys = expected_cache_keys(geometry_id, arms)
        if not all(key in tracked and key not in broken for key in keys):
            continue
        if geometry_id not in with_rows:
            continue
        done.append(geometry_id)
    return done


def prune_tables(path, columns, keep_geometries) -> int:
    """Drop the rows of every geometry that is NOT being resumed from.

    The tidy tables are written row by row while the cache file of a geometry
    is written only at the end of its loop, so a crash leaves partial rows for
    the geometry that was in flight. Appending to that file would duplicate
    them. Rewriting it down to the completed geometries first makes the append
    exact, and the row counts of the summary stay meaningful.
    """
    path = Path(path)
    if not path.exists():
        return 0
    keep = set(keep_geometries)
    with open(path, newline="", encoding="utf-8") as handle:
        kept = [row for row in csv.DictReader(handle)
                if row.get("geometry") in keep]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in kept:
            writer.writerow({column: row.get(column, "") for column in columns})
    return len(kept)


# ----------------------------------------------------------------- arm setup
def resolve_arms(requested, skip_gpu: bool):
    """The ordered arm list actually executed, after --arms and --skip-gpu."""
    if requested in (None, [], ["all"]):
        arms = list(ALL_ARMS)
    else:
        unknown = [a for a in requested if a not in ALL_ARMS]
        if unknown:
            raise ValueError(f"unknown arms {unknown}; available {list(ALL_ARMS)}")
        arms = [a for a in ALL_ARMS if a in set(requested)]
    if skip_gpu:
        arms = [a for a in arms if a not in GPU_ARMS]
    if ARM_REFERENCE not in arms:
        # every other arm is scored AGAINST the reference of the same view
        arms.insert(0, ARM_REFERENCE)
    return arms


def thinning_seed(fraction_index: int, k: int, geo_index: int, view: int):
    """SeedSequence([seed(22, fi, k), geo, view]) — section 6 step 2, verbatim."""
    base = (MASTER_SEED * 10 ** 6 + THINNING_STAGE * 10 ** 4
            + fraction_index * 10 ** 2 + k)
    return np.random.SeedSequence([base, geo_index, view])


def sham_seed(geo_index: int, view: int):
    """Sham masks are geometry-and-view specific and independent of the arm."""
    base = MASTER_SEED * 10 ** 6 + SHAM_STAGE * 10 ** 4
    return np.random.SeedSequence([base, geo_index, view])


def fraction_index_of(fraction, grid):
    """Index of `fraction` in `grid` under np.isclose, or None.

    The fractions travel through JSON, a CSV and a float32 shard header, so
    0.05 is not necessarily the same double everywhere. Equality on floats
    would silently drop a whole level from a table -- and a level that is
    quietly absent is far worse than one that raises.
    """
    for index, value in enumerate(grid):
        if np.isclose(float(value), float(fraction), rtol=0.0,
                      atol=FRACTION_ATOL):
            return index
    return None


def is_frozen_fraction(fraction) -> bool:
    return fraction_index_of(fraction, FROZEN_FRACTIONS) is not None


def frozen_classical_for(fraction) -> str:
    """The frozen classical winner of a level, matched with np.isclose."""
    levels = list(FROZEN_CLASSICAL)
    index = fraction_index_of(fraction, levels)
    if index is None:
        raise KeyError(f"no frozen classical winner for fraction {fraction!r}; "
                       f"the frozen levels are {levels}")
    return FROZEN_CLASSICAL[levels[index]]


def model_runs_for_arm(arm: str):
    """[(run_id, model_fraction, evaluated_fractions)] for a model arm."""
    if arm == ARM_UNET_L1:
        return ([(UNET_L1_RUNS[f], f, (f,)) for f in FROZEN_FRACTIONS]
                + [(run, f, (f,)) for run, f in UNET_L1_CONTINUITY_RUNS])
    if arm == ARM_CWGAN:
        return [(CWGAN_RUNS[f], f, (f,)) for f in FROZEN_FRACTIONS]
    if arm == ARM_SEED_REPLICAS:
        return [(run, SEED_REPLICA_FRACTION, (SEED_REPLICA_FRACTION,))
                for run in SEED_REPLICA_RUNS]
    if arm == ARM_CROSS_LEVEL:
        # the diagonal is arm F; recomputing it would duplicate 1 920 forwards
        return [(CWGAN_RUNS[trained], trained,
                 tuple(f for f in FROZEN_FRACTIONS if f != trained))
                for trained in FROZEN_FRACTIONS]
    return []


def input_scaling_for(run_id: str) -> str:
    """The training input convention of a run, from its archived config.

    RAISES rather than defaulting. The convention decides whether the model is
    fed log1p(low)/scale or log1p(low/f)/scale, and the two differ by the whole
    amplitude restoration -- feeding a `shared` model the ref-equivalent input
    does not fail, it just produces a wrong image with a plausible SSIM. That
    is exactly the failure a single irreversible pass cannot afford, and the
    Stage 18 contract mismatch (execution plan 1.1) is the proof that this
    particular mistake is not hypothetical. A run with no archived record is a
    run whose contract nobody knows.
    """
    record = pred.run_config(run_id)
    if record is None:
        raise FileNotFoundError(
            f"no archived config record for run {run_id!r} in "
            f"{pred.RECORDS_DIR}: the input contract cannot be assumed. "
            f"Restore the record, or add the run to the retrain roster.")
    scaling = record.get("config", {}).get("input_scaling")
    if scaling is None:
        raise KeyError(f"the archived config of {run_id!r} declares no "
                       f"'input_scaling': the input contract is undefined")
    return str(scaling)


def preprocess_for(scaling: str, low_counts, ref_counts, fraction: float):
    if scaling == REF_EQUIVALENT:
        return preprocess_pair_corrected(low_counts, ref_counts, fraction)
    return preprocess_pair(low_counts, ref_counts)


# ------------------------------------------------------------------ row build
def _blank_row():
    return {column: "" for column in CSV_COLUMNS}


def _fmt(value, digits=6):
    if value is None:
        return ""
    value = float(value)
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}g}"


def global_row(context, image, x_ref, notes="") -> dict:
    row = _blank_row()
    row.update(context)
    row["scope"] = "global"
    row["anomaly_idx"] = -1
    row["ssim"] = _fmt(compute_ssim(image, x_ref))
    row["psnr_db"] = _fmt(compute_psnr(x_ref, image), 5)
    row["notes"] = notes
    return row


STAT_COLUMNS = ("mean_roi", "mean_bg", "std_bg", "C", "CNR", "z")


def roi_rows(context, image, x_ref, masks, geo) -> list:
    """One row per anomaly, ALWAYS -- an unmeasurable ROI is reported, not skipped.

    Protocol v1.1 section 5.2: an ROI below 64 px is excluded from the rates
    "Y se reporta, nunca se descarta en silencio". A skipped row is a silent
    discard: the instance simply would not appear in the table and nobody
    reading it could tell whether it was excluded or never existed. So every
    (anomaly, view) emits a row; when the mask or its annulus is empty the row
    carries roi_valid = 0, an explicit `nan` in every statistic and a note
    saying why. The rates of Stage 21 filter on roi_valid and count the
    exclusions separately.
    """
    rows = []
    for index, (mask, annulus) in enumerate(zip(masks["rois"], masks["annuli"])):
        anomaly = geo.anomalies[index]
        delta_rho = roi_module.anomaly_delta_rho(geo, index)
        row = _blank_row()
        row.update(context)
        row["scope"] = "roi"
        row["anomaly_idx"] = index
        row["anomaly_kind"] = anomaly.kind
        row["delta_rho"] = _fmt(delta_rho, 4)
        # SIGNED, exactly like octiles.csv delta_opacity_mean: the sign is the
        # sign of the density contrast and carries the direction of the
        # expected detection (cavity/conduit positive, dense body negative)
        row["delta_opacity_max"] = _fmt(
            float(roi_module.delta_opacity(
                masks["l_anomalies"][index].max(), delta_rho)), 6)
        row["roi_pixels"] = masks["roi_pixels"][index]
        row["annulus_pixels"] = masks["annulus_info"][index]["n_pixels"]

        measurable = bool(mask.any() and annulus.any())
        row["roi_valid"] = int(bool(masks["roi_valid"][index]) and measurable)
        notes = []
        if measurable:
            stats = det.roi_stats(image, mask, annulus)
            for key in STAT_COLUMNS:
                row[key] = _fmt(stats[key])
            row["n_roi"] = stats["n_roi"]
            row["n_bg"] = stats["n_bg"]
            row["local_ssim"] = _fmt(det.local_roi_ssim(image, x_ref, mask))
        else:
            for key in STAT_COLUMNS:
                row[key] = "nan"
            row["local_ssim"] = "nan"
            row["n_roi"] = int(mask.sum())
            row["n_bg"] = int(annulus.sum())
            notes.append("no measurable statistic: "
                         + ("empty ROI" if not mask.any() else "empty annulus"))
        if not masks["roi_valid"][index]:
            notes.append("roi below 64 px: excluded from the rates, reported")
        if masks["annulus_info"][index]["widened"]:
            notes.append("annulus widened to 16 px")
        if masks["annulus_info"][index]["below_minimum"]:
            notes.append("annulus below 200 px")
        row["notes"] = "; ".join(notes)
        rows.append(row)
    return rows


def sham_rows(context, image, masks) -> list:
    rows = []
    for index, sham in enumerate(masks["shams"]):
        annulus, info = masks["sham_annuli"][index]
        if not sham.any() or not annulus.any():
            continue
        stats = det.roi_stats(image, sham, annulus)
        row = _blank_row()
        row.update(context)
        row["scope"] = "sham"
        row["anomaly_idx"] = index
        row["anomaly_kind"] = "sham"
        row["roi_pixels"] = int(sham.sum())
        row["annulus_pixels"] = info["n_pixels"]
        for key in ("mean_roi", "mean_bg", "std_bg", "C", "CNR", "z"):
            row[key] = _fmt(stats[key])
        row["n_roi"] = stats["n_roi"]
        row["n_bg"] = stats["n_bg"]
        rows.append(row)
    return rows


def texture_row(context, image, x_ref, masks) -> dict:
    union = np.zeros_like(masks["sky"])
    for mask in masks["rois"]:
        union |= mask
    scalars = tex.texture_scalars(image, x_ref, union if union.any() else None)
    row = _blank_row()
    row.update(context)
    row["scope"] = "texture"
    row["anomaly_idx"] = -1
    for key, value in scalars.items():
        row[key] = _fmt(value)
    row["notes"] = ("texture over the union of the ROIs" if union.any()
                    else "texture over the whole image (no ROI in this view)")
    return row


def octile_rows(context, image, masks, geo) -> list:
    rows = []
    for index, (mask, annulus) in enumerate(zip(masks["rois"], masks["annuli"])):
        if not mask.any() or not annulus.any():
            continue
        l_a = masks["l_anomalies"][index]
        delta_rho = roi_module.anomaly_delta_rho(geo, index)
        for entry in det.octile_stats(image, l_a, annulus, roi=mask):
            row = {column: "" for column in OCTILE_COLUMNS}
            for key in ("split", "arm", "model_run", "geometry", "view",
                        "fraction", "k"):
                row[key] = context.get(key, "")
            row["anomaly_idx"] = index
            row["anomaly_kind"] = geo.anomalies[index].kind
            row["octile"] = entry["octile"]
            # the EFFECTIVE bin count: ties in the quantized path length can
            # leave fewer than eight, and Stage 21 must quote what it got
            row["n_octiles"] = entry["n_octiles"]
            row["n_roi"] = entry["n_roi"]
            row["l_lo_m"] = _fmt(entry["l_lo_m"], 5)
            row["l_hi_m"] = _fmt(entry["l_hi_m"], 5)
            row["l_mean_m"] = _fmt(entry["l_mean_m"], 5)
            row["delta_opacity_mean"] = _fmt(
                float(roi_module.delta_opacity(entry["l_mean_m"], delta_rho)), 6)
            row["C"] = _fmt(entry["C"])
            row["z"] = _fmt(entry["z"])
            rows.append(row)
    return rows


# ---------------------------------------------------------------------- main
def build_view_masks(geo, geometry_id: str, view_index: int, geo_index: int,
                     n_sham: int):
    """Ray-cast ONCE per (geometry, view) and reuse for every arm."""
    view = DetectorView(geometry_id, view_index)
    rng = np.random.default_rng(sham_seed(geo_index, view_index))
    masks = roi_module.view_masks(geo, view, rng=rng, n_sham=n_sham)
    masks["sham_annuli"] = [
        roi_module.build_annulus(sham, list(masks["rois"]) + [sham],
                                 masks["sky"], masks["edge"], masks["l_rock"])
        for sham in masks["shams"]]
    return masks


def pack_masks(mask_store, geo) -> dict:
    """Everything a later stage needs about the geometry, in one npz.

    Stages 21 and 22 draw ROI contours, recompute octiles and locate the blind
    panels' anomalies; without the masks in the cache they would have to
    re-cast the rays (cheap) or re-open a shard (forbidden).

    THE INTEGER REFERENCE COUNTS TRAVEL WITH THE MASKS. Arm I is stored as
    metrics only, on the promise that the whole dense grid is regenerable bit
    for bit from its seeds -- but a seeded binomial thinning needs the counts
    it thins, and those live in the shard, which after Stage 20 may never be
    opened again. Without them arm I is not regenerable, the promise is empty,
    and the Stage 21 gallery has no input panel to put beside the enhanced one:
    arms B and C are not cached either. 256 x 256 uint32 per view is ~6.3 MiB
    over the four geometries, against a cache budget of 0.8-1.0 GB.
    """
    views = np.array([view for view, _ in mask_store], dtype=np.int32)
    n_anomalies = len(geo.anomalies)
    stack = lambda getter: np.stack([getter(masks) for _v, masks in mask_store])
    sham_counts = [len(masks["shams"]) for _view, masks in mask_store]
    centroids = np.full((len(mask_store), max(n_anomalies, 1), 2), np.nan)
    for row, (_view, masks) in enumerate(mask_store):
        for index, mask in enumerate(masks["rois"]):
            if mask.any():
                rows, cols = np.nonzero(mask)
                centroids[row, index] = (rows.mean(), cols.mean())
    return {
        "views": views,
        "sky": stack(lambda m: m["sky"]),
        "edge": stack(lambda m: m["edge"]),
        "l_rock": stack(lambda m: m["l_rock"].astype(np.float16)),
        "rois": stack(lambda m: np.stack(m["rois"])) if n_anomalies
        else np.zeros((len(mask_store), 0, 1, 1), dtype=bool),
        "annuli": stack(lambda m: np.stack(m["annuli"])) if n_anomalies
        else np.zeros((len(mask_store), 0, 1, 1), dtype=bool),
        "l_anomalies": stack(lambda m: np.stack(
            [m["l_anomalies"][i].astype(np.float16)
             for i in range(n_anomalies)])) if n_anomalies
        else np.zeros((len(mask_store), 0, 1, 1), dtype=np.float16),
        # the rejection sampler can come up short on a cramped silhouette, so
        # the sham stack is only rectangular when every view got the same
        # count; the per-view counts always travel so a short view is visible
        "shams": (stack(lambda m: np.stack(m["shams"])) if sham_counts
                  and len(set(sham_counts)) == 1 and sham_counts[0]
                  else np.zeros((len(mask_store), 0, 1, 1), dtype=bool)),
        "sham_counts": np.array(sham_counts, dtype=np.int32),
        "roi_centroids": centroids,
        "roi_valid": np.array([[bool(v) for v in masks["roi_valid"]]
                               for _view, masks in mask_store], dtype=bool)
        if n_anomalies else np.zeros((len(mask_store), 0), dtype=bool),
        # fixed-width unicode, never dtype=object: an object array can only be
        # read back through pickle and the cache is loaded with allow_pickle=False
        "anomaly_kinds": np.array([a.kind for a in geo.anomalies], dtype="U32"),
        "anomaly_delta_rho": np.array(
            [roi_module.anomaly_delta_rho(geo, i) for i in range(n_anomalies)],
            dtype=np.float64),
        # the integer 30-day counts, one per view, in the same row order as
        # `views`: the seed input of arm I and the source of the B / C panels
        "ref_counts": (stack(lambda m: np.asarray(m["ref_counts"],
                                                  dtype=np.uint32))
                       if mask_store and all("ref_counts" in masks
                                             for _view, masks in mask_store)
                       else np.zeros((len(mask_store), 0, 0), dtype=np.uint32)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--arms", nargs="+", default=None,
                        help=f"subset of {list(ALL_ARMS)}, or 'all'")
    parser.add_argument("--runs-root", default=None,
                        help="re-root the checkpoint paths of "
                             "models/weights-manifest.json")
    parser.add_argument("--manifest", default=str(pred.DEFAULT_MANIFEST))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--k", type=int, default=K_REALIZATIONS)
    parser.add_argument("--n-sham", type=int, default=N_SHAM_PER_VIEW,
                        help="frozen at 10 for the test pass (protocol v1.1)")
    parser.add_argument("--views", nargs="+", type=int, default=None)
    parser.add_argument("--skip-gpu", action="store_true",
                        help="classical and raw arms only; never imports torch")
    parser.add_argument("--access-log", default=str(ACCESS_LOG))
    parser.add_argument("--resume", action="store_true",
                        help="crash protocol: keep the geometries whose cache "
                             "AND rows are both already there, redo the rest, "
                             "and record a `restart` event instead of a second "
                             "opening of the split")
    parser.add_argument("--resume-reason", default="",
                        help="why the previous attempt stopped; REQUIRED with "
                             "--resume on the test split, it goes into the "
                             "access log verbatim")
    args = parser.parse_args()

    arms = resolve_arms(args.arms, args.skip_gpu)
    out_root = Path(args.out_root) if args.out_root else \
        DEFAULT_CACHE_ROOT / args.split
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / cache_module.MANIFEST_NAME
    metrics_path = out_root / "metrics.csv"
    octiles_path = out_root / "octiles.csv"

    if args.resume and args.split == "test" and not args.resume_reason.strip():
        parser.error("--resume on the test split needs --resume-reason: the "
                     "restart entry of the access log has to say what "
                     "happened, or the continuation is not auditable")

    t_start = time.perf_counter()

    # ---- PRE-FLIGHT, before anything irreversible: every generator hashed
    # against the manifest (main + the Stage 19.5 supplement) and every
    # archived config present. Doing this after the guard opened would leave a
    # permanent opening entry behind a KeyError that produced no data.
    run_ids = [run_id for arm in arms
               for run_id, _f, _levels in model_runs_for_arm(arm)]
    checkpoints = pred.verify_all_checkpoints(run_ids, args.manifest,
                                              args.runs_root)
    scalings = {run_id: input_scaling_for(run_id) for run_id in run_ids}
    for run_id in dict.fromkeys(run_ids):
        print(f"  pre-flight {run_id}: sha256 "
              f"{checkpoints[run_id]['sha256'][:12]}..., input "
              f"{scalings[run_id]}, val SSIM "
              f"{checkpoints[run_id]['best_val_ssim']}", flush=True)

    # ---- generators (lazy: --skip-gpu never touches torch), still before the
    # split is resolved: a broken architecture must fail while nothing is open
    models, provenance = {}, {}
    for run_id in dict.fromkeys(run_ids):
        model, info = pred.load_generator(
            run_id, args.manifest, args.runs_root, args.device)
        models[run_id] = model
        provenance[run_id] = {**info, "input_scaling": scalings[run_id]}
        print(f"  loaded {run_id} ({info['generator']}, "
              f"sha256 {info['sha256'][:12]}...)", flush=True)

    partial = partial_cache_hashes(out_root) if args.resume else {}
    geometries, access_entry = resolve_geometries(
        args.split, args.access_log, sys.argv, resume=args.resume,
        resume_reason=args.resume_reason, partial_hashes=partial)
    print(f"Stage 20 evaluation — split={args.split} arms={arms} "
          f"geometries={[g for g, _ in geometries]} K={args.k} "
          f"out={out_root}")

    resumed = (completed_geometries(out_root, manifest_path, metrics_path,
                                    geometries, arms) if args.resume else [])
    if args.resume:
        kept_metrics = prune_tables(metrics_path, CSV_COLUMNS, resumed)
        kept_octiles = prune_tables(octiles_path, OCTILE_COLUMNS, resumed)
        print(f"  RESUMING after {args.resume_reason!r}: keeping {resumed} "
              f"({kept_metrics} metric rows, {kept_octiles} octile rows) from "
              f"{len(partial)} partial cache files; redoing "
              f"{[g for g, _s in geometries if g not in resumed]}")
        geometries = [(g, s) for g, s in geometries if g not in resumed]

    shard_hashes, cache_hashes = {}, {}
    counts = {"global": 0, "roi": 0, "sham": 0, "texture": 0, "octile": 0}
    fresh = not metrics_path.exists() or metrics_path.stat().st_size == 0
    fresh_octiles = (not octiles_path.exists()
                     or octiles_path.stat().st_size == 0)
    mode = "a" if args.resume else "w"

    with open(metrics_path, mode, newline="", encoding="utf-8") as metrics_file, \
            open(octiles_path, mode, newline="",
                 encoding="utf-8") as octiles_file:
        writer = csv.DictWriter(metrics_file, fieldnames=CSV_COLUMNS)
        if fresh or mode == "w":
            writer.writeheader()
        octile_writer = csv.DictWriter(octiles_file, fieldnames=OCTILE_COLUMNS)
        if fresh_octiles or mode == "w":
            octile_writer.writeheader()

        def emit(rows):
            for row in (rows if isinstance(rows, list) else [rows]):
                writer.writerow(row)
                counts[row["scope"]] = counts.get(row["scope"], 0) + 1

        for geo_index, (geometry_id, geometry_split) in enumerate(geometries):
            shard_path = SHARD_DIR / f"{geometry_id}.npz"
            shard_hashes[shard_path.name] = sha256_of(shard_path)
            data, _meta = load_shard(geometry_id)
            geo = roi_module.load_geometry(geometry_id)
            fractions = [float(f) for f in data["fractions"]]
            view_indices = (list(range(data["ref_counts"].shape[0]))
                            if args.views is None else list(args.views))
            # both reset per geometry: `blocks` collects the images to cache
            # and `mask_store` the per-view geometry, and each is written to
            # its own file under this geometry's folder at the end of the loop
            blocks, mask_store = {}, []
            print(f"\n{geometry_id} ({geometry_split}) — "
                  f"{len(geo.anomalies)} anomalies, views {view_indices}",
                  flush=True)

            for view in view_indices:
                t_view = time.perf_counter()
                ref_counts = np.asarray(data["ref_counts"][view])
                masks = build_view_masks(geo, geometry_id, view, geo_index,
                                         args.n_sham)
                scale = pred.shared_scale(ref_counts)
                x_ref = (np.log1p(ref_counts.astype(np.float32))
                         / np.float32(scale))

                # the integer counts travel into masks.npz so arm I stays
                # regenerable and Stage 21 can rebuild the B / C input panels
                masks["ref_counts"] = ref_counts
                mask_store.append((view, masks))
                base = {"split": args.split, "geometry": geometry_id,
                        "geometry_split": geometry_split, "geo_index": geo_index,
                        "view": view}

                # ---- arm A: the reference itself
                if ARM_REFERENCE in arms:
                    context = {**base, "arm": ARM_REFERENCE,
                               "arm_label": ARM_LABELS[ARM_REFERENCE],
                               "model_run": "", "fraction": 1.0,
                               "model_fraction": "", "convention": "reference",
                               "k": -1}
                    blocks.setdefault(ARM_REFERENCE, []).append(
                        (view, 1.0, -1, "", x_ref))
                    emit(global_row(context, x_ref, x_ref,
                                    "self-comparison: SSIM = 1 by construction"))
                    emit(roi_rows(context, x_ref, x_ref, masks, geo))
                    emit(sham_rows(context, x_ref, masks))
                    for row in octile_rows(context, x_ref, masks, geo):
                        octile_writer.writerow(row)
                        counts["octile"] += 1

                # ---- arms B, C, D: no model / classical, per frozen level
                for fraction_index, fraction in enumerate(fractions):
                    if not is_frozen_fraction(fraction):
                        continue
                    for k in range(min(args.k, data["low_counts"].shape[2])):
                        low = np.asarray(data["low_counts"][view,
                                                            fraction_index, k])
                        variants = []
                        if ARM_RAW_SHARED in arms:
                            image, _ = preprocess_pair(low, ref_counts)
                            variants.append((ARM_RAW_SHARED, "", RAW, image))
                        if ARM_RAW_CORRECTED in arms:
                            image, _ = preprocess_pair_corrected(low, ref_counts,
                                                                 fraction)
                            variants.append((ARM_RAW_CORRECTED, "", CORRECTED,
                                             image))
                        if ARM_CLASSICAL in arms:
                            key = frozen_classical_for(fraction)
                            image = pred.classical_denoise(
                                key, None, low, fraction, CLASSICAL_CONVENTION,
                                scale)
                            variants.append((ARM_CLASSICAL, key,
                                             CLASSICAL_CONVENTION, image))
                            blocks.setdefault(ARM_CLASSICAL, []).append(
                                (view, fraction, k, key, image))

                        textured = (fraction_index_of(fraction,
                                                      TEXTURE_FRACTIONS)
                                    is not None)
                        for arm, run_label, convention, image in variants:
                            context = {**base, "arm": arm,
                                       "arm_label": ARM_LABELS[arm],
                                       "model_run": run_label,
                                       "fraction": fraction,
                                       "model_fraction": "",
                                       "convention": convention, "k": k}
                            emit(global_row(context, image, x_ref))
                            emit(roi_rows(context, image, x_ref, masks, geo))
                            if arm in SHAM_ARMS:
                                emit(sham_rows(context, image, masks))
                            if arm in OCTILE_ARMS:
                                for row in octile_rows(context, image, masks,
                                                       geo):
                                    octile_writer.writerow(row)
                                    counts["octile"] += 1
                            if arm in TEXTURE_ARMS and textured:
                                emit(texture_row(context, image, x_ref, masks))

                # ---- arms E, F, G, H: learned generators
                for arm in arms:
                    for run_id, model_fraction, levels in model_runs_for_arm(arm):
                        scaling = scalings[run_id]
                        for fraction in levels:
                            fraction_index = fraction_index_of(fraction,
                                                               fractions)
                            if fraction_index is None:
                                continue
                            n_k = min(args.k, data["low_counts"].shape[2])
                            batch = np.stack([
                                preprocess_for(
                                    scaling,
                                    np.asarray(data["low_counts"][
                                        view, fraction_index, k]),
                                    ref_counts, fraction)[0]
                                for k in range(n_k)])
                            outputs = pred.predict_batch(
                                models[run_id], batch, args.device, args.batch)
                            blocks.setdefault(arm, []).extend(
                                (view, fraction, k, run_id, outputs[k])
                                for k in range(n_k))
                            for k in range(n_k):
                                context = {**base, "arm": arm,
                                           "arm_label": ARM_LABELS[arm],
                                           "model_run": run_id,
                                           "fraction": fraction,
                                           "model_fraction": model_fraction,
                                           "convention": scaling, "k": k}
                                emit(global_row(context, outputs[k], x_ref))
                                emit(roi_rows(context, outputs[k], x_ref, masks,
                                              geo))
                                if arm in SHAM_ARMS:
                                    emit(sham_rows(context, outputs[k], masks))
                                if arm in OCTILE_ARMS:
                                    for row in octile_rows(context, outputs[k],
                                                           masks, geo):
                                        octile_writer.writerow(row)
                                        counts["octile"] += 1
                                if (arm in TEXTURE_ARMS
                                        and fraction_index_of(
                                            fraction, TEXTURE_FRACTIONS)
                                        is not None):
                                    emit(texture_row(context, outputs[k], x_ref,
                                                     masks))

                # ---- arm I: the dense thinning grid (metrics only, seeded)
                if ARM_THINNING_GRID in arms:
                    ref64 = ref_counts.astype(np.int64)
                    for fraction_index, fraction in enumerate(DENSE_FRACTIONS):
                        for k in range(args.k):
                            rng = np.random.default_rng(
                                thinning_seed(fraction_index, k, geo_index, view))
                            low = rng.binomial(ref64, fraction)
                            for convention in (RAW, CORRECTED):
                                image = (preprocess_pair(low, ref_counts)[0]
                                         if convention == RAW
                                         else preprocess_pair_corrected(
                                             low, ref_counts, fraction)[0])
                                context = {**base, "arm": ARM_THINNING_GRID,
                                           "arm_label":
                                               ARM_LABELS[ARM_THINNING_GRID],
                                           "model_run": "", "fraction": fraction,
                                           "model_fraction": "",
                                           "convention": convention, "k": k}
                                emit(global_row(
                                    context, image, x_ref,
                                    "regenerable: SeedSequence([seed(22, fi, k), "
                                    "geo_index, view])"))

                print(f"  view {view}: {time.perf_counter() - t_view:.1f} s "
                      f"(rows so far {sum(counts.values())})", flush=True)

            # ---- cache the geometry-derived masks (Stage 21 reads no shard)
            mask_path = out_root / geometry_id / "masks.npz"
            cache_module.save_masks(mask_path, **pack_masks(mask_store, geo))
            cache_module.manifest_update(manifest_path, mask_path, out_root)
            cache_hashes[f"{geometry_id}/masks.npz"] = sha256_of(mask_path)
            print(f"  cached masks -> {mask_path.relative_to(out_root)} "
                  f"({mask_path.stat().st_size / 2 ** 20:.1f} MiB)", flush=True)

            # ---- cache the predictions of this geometry, one file per arm
            for arm, entries in blocks.items():
                if arm not in CACHED_ARMS:
                    continue
                path = out_root / geometry_id / f"arm_{arm}.npz"
                cache_module.save_predictions(
                    path, np.stack([image for *_head, image in entries]),
                    view=np.array([v for v, _f, _k, _r, _i in entries]),
                    fraction=np.array([f for _v, f, _k, _r, _i in entries]),
                    k=np.array([k for _v, _f, k, _r, _i in entries]),
                    # (view, fraction, k) does NOT identify an image: arm E now
                    # holds two runs at f = 0.01 (the paired retrain and the
                    # shared-contract continuity twin) and arm H holds one row
                    # per (trained level, evaluated level). Without the run id
                    # the two collide in the cache and no later stage can say
                    # which generator produced which image.
                    model_run=np.array([r for _v, _f, _k, r, _i in entries],
                                       dtype="U32"))
                cache_module.manifest_update(manifest_path, path, out_root)
                cache_hashes[f"{geometry_id}/arm_{arm}.npz"] = sha256_of(path)
                print(f"  cached arm {arm}: {len(entries)} images -> "
                      f"{path.relative_to(out_root)} "
                      f"({path.stat().st_size / 2 ** 20:.1f} MiB)", flush=True)

    # the tidy tables are part of the cache: Stages 21 and 22 read nothing else
    for path in (metrics_path, octiles_path):
        cache_module.manifest_update(manifest_path, path, out_root)
        cache_hashes[path.name] = sha256_of(path)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": "20 (single test pass)" if args.split == "test"
        else "19 (validation dry run)",
        "split": args.split,
        "arms": arms,
        "geometries": [{"geometry_id": g, "split": s} for g, s in geometries],
        "resume": {"resumed": bool(args.resume),
                   "reason": args.resume_reason,
                   "kept_geometries": list(resumed),
                   "n_partial_cache_files": len(partial)},
        "k_realizations": int(args.k),
        "n_sham_per_view": int(args.n_sham),
        "frozen_fractions": list(FROZEN_FRACTIONS),
        "dense_fractions": list(DENSE_FRACTIONS),
        "frozen_classical": {f"{f:.2f}": key
                             for f, key in FROZEN_CLASSICAL.items()},
        "classical_convention": CLASSICAL_CONVENTION,
        "unet_l1_continuity_runs": [{"run_id": run, "fraction": f}
                                    for run, f in UNET_L1_CONTINUITY_RUNS],
        "model_provenance": provenance,
        "input_scalings": scalings,
        "seed_derivations": {
            "thinning_grid": "SeedSequence([20260805e6 + 22e4 + fi*100 + k, "
                             "geo_index, view])",
            "sham_masks": "SeedSequence([20260805e6 + 21e4, geo_index, view])",
        },
        "row_counts": counts,
        "outputs": {"metrics_csv": str(metrics_path),
                    "octiles_csv": str(octiles_path),
                    "cache_manifest": str(manifest_path)},
        "shards_read": dict(sorted(shard_hashes.items())),
        "cache_written": dict(sorted(cache_hashes.items())),
        "wall_seconds": round(time.perf_counter() - t_start, 1),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": git_commit(),
    }
    summary_path = out_root / "summary.json"
    cache_module.write_json_atomic(summary_path, summary)
    cache_module.manifest_update(manifest_path, summary_path, out_root)
    cache_hashes[summary_path.name] = sha256_of(summary_path)

    # ---- FINAL STEP: re-hash everything the manifest claims. The manifest is
    # what Stage 23 puts next to each assertion and what a re-run from the
    # cache is checked against, so it has to be verified while the job that
    # wrote it is still alive -- a truncated npz found months later is
    # indistinguishable from a good one until somebody hashes it.
    problems = cache_module.verify_manifest(manifest_path, out_root)
    for key, reason in problems:
        print(f"  CACHE VERIFY FAILURE {key}: {reason}", file=sys.stderr)

    if access_entry is not None:
        access_log_close(args.access_log, access_entry, shard_hashes,
                         cache_hashes,
                         {"row_counts": counts,
                          "wall_seconds": summary["wall_seconds"],
                          "out_root": str(out_root),
                          "manifest_problems": [list(p) for p in problems]})

    print(f"\nwrote {metrics_path} ({sum(counts.values())} rows)")
    print(f"wrote {octiles_path}")
    print(f"wrote {summary_path}")
    print(f"rows by scope: {counts}")
    print(f"cache manifest: "
          f"{len(cache_module.read_manifest(manifest_path).get('files', {}))} "
          f"files, {len(problems)} problem(s)")
    print(f"wall: {summary['wall_seconds'] / 60.0:.1f} min")
    print("STAGE 20 EVALUATION:", "PASSED" if not problems else "FAILED")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
