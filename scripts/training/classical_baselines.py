#!/usr/bin/env python3
"""Stage 13.1 + 13.2 — classical denoising baselines on the VALIDATION split.

The no-model baseline of Stage 11 is dominated by AMPLITUDE, not by noise:
preprocess_pair normalizes input and reference with the same scale, so a low
statistics input is systematically ~f times darker. Rescaling counts by the
known 1/f before log1p is deterministic and available at inference time, so a
classical method compared WITHOUT it would be a straw man. Every method is
therefore evaluated under both input conventions:

  raw       : denoise the counts, then preprocess_pair (shared scale)
  corrected : denoise the counts, divide by f, then the same shared scale
              (preprocess_pair_corrected semantics)

Stage 13.2 adds a dense equivalence grid (fractions generated on the fly by
binomial thinning from ref_counts — the frozen shards are never touched) and
the sensitivity of the oracle scale log1p(ref.max()) against the operational
estimator log1p(max(low)/f).

Outputs: results/dataset/stats/stage13_baselines.json
         figures/results/stage13_baselines.png
         figures/results/stage13_equivalence_grid.png

Run:  python scripts/training/classical_baselines.py --workers 6
      python scripts/training/classical_baselines.py --quick        # smoke test
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from skimage.filters import gaussian as gaussian_filter
from skimage.filters import median as median_filter
from skimage.morphology import disk
from skimage.restoration import denoise_nl_means, estimate_sigma

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

from metrics import compute_psnr, compute_ssim, summarize  # noqa: E402
from muography_dataset import (SHARD_DIR, geometry_ids_for_split,  # noqa: E402
                               load_shard)
from plotting_style import apply_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

MASTER_SEED = 20260805
STAGE = 13
STATS_PATH = REPO_ROOT / "results" / "dataset" / "stats" / "stage13_baselines.json"
FIG_DIR = REPO_ROOT / "figures" / "results"

FROZEN_FRACTIONS = (0.50, 0.10, 0.05, 0.01)
EXTRA_FRACTIONS = (0.75, 0.30, 0.20, 0.02)  # generated on the fly (Stage 13.2)
K_INDICES = (0, 1, 2, 3, 4)                 # 5 of the 20 realizations
CONVENTIONS = ("raw", "corrected")
GAUSSIAN_SIGMAS = (0.8, 1.5, 2.5)
MEDIAN_RADII = (1, 2, 3)
NLM_H_MULTIPLIERS = (0.5, 0.8, 1.2)
NLM_PATCH_SIZE = 5
NLM_PATCH_DISTANCE = 6
ANSCOMBE_OFFSET = 3.0 / 8.0

try:  # optional wheel (bm3d + PyWavelets); absence is documented, not fatal
    import bm3d as _bm3d
    HAVE_BM3D = True
except ImportError:
    _bm3d = None
    HAVE_BM3D = False

try:  # estimate_sigma imports pywt lazily inside the call — probe it up front
    import pywt as _pywt  # noqa: F401
    HAVE_PYWT = True
except ImportError:
    HAVE_PYWT = False


def seed_for(stage: int, config_idx: int, rep_idx: int) -> int:
    """Master seed formula (protocol PA-1)."""
    return MASTER_SEED * 10 ** 6 + stage * 10 ** 4 + config_idx * 10 ** 2 + rep_idx


# ------------------------------------------------------------------ denoisers
def _anscombe(counts: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(np.maximum(counts, 0.0) + ANSCOMBE_OFFSET)


def _inverse_anscombe(values: np.ndarray) -> np.ndarray:
    return np.maximum((values / 2.0) ** 2 - ANSCOMBE_OFFSET, 0.0)


def _nlm(image: np.ndarray, h_mult: float) -> np.ndarray:
    sigma = float(estimate_sigma(image, channel_axis=None))
    return denoise_nl_means(image, h=h_mult * sigma, sigma=sigma,
                            fast_mode=True, patch_size=NLM_PATCH_SIZE,
                            patch_distance=NLM_PATCH_DISTANCE,
                            preserve_range=True)


def denoise(method: str, params: dict, counts: np.ndarray) -> np.ndarray:
    """counts -> denoised counts (float32), one entry point per method."""
    x = counts.astype(np.float32)
    if method == "identity":
        return x
    if method == "gaussian":
        return gaussian_filter(x, sigma=params["sigma"], preserve_range=True
                               ).astype(np.float32)
    if method == "median":
        return median_filter(x, footprint=disk(params["radius"])
                             ).astype(np.float32)
    if method == "nlm_log":
        return np.expm1(_nlm(np.log1p(x), params["h_mult"])).astype(np.float32)
    if method == "nlm_anscombe":
        return _inverse_anscombe(_nlm(_anscombe(x), params["h_mult"])
                                 ).astype(np.float32)
    if method == "bm3d_anscombe":
        if not HAVE_BM3D:
            raise RuntimeError("bm3d is not installed")
        # Anscombe stabilizes Poisson noise to unit variance.
        return _inverse_anscombe(
            _bm3d.bm3d(_anscombe(x), sigma_psd=1.0)).astype(np.float32)
    raise ValueError(f"unknown method {method!r}")


def method_grid() -> list:
    """[(key, method, params)] — the Stage 13.1 configuration grid."""
    grid = [("identity", "identity", {})]
    grid += [(f"gaussian_s{s:g}", "gaussian", {"sigma": s})
             for s in GAUSSIAN_SIGMAS]
    grid += [(f"median_r{r}", "median", {"radius": r}) for r in MEDIAN_RADII]
    if HAVE_PYWT:
        grid += [(f"nlm_log_h{h:g}", "nlm_log", {"h_mult": h})
                 for h in NLM_H_MULTIPLIERS]
        grid += [(f"nlm_anscombe_h{h:g}", "nlm_anscombe", {"h_mult": h})
                 for h in NLM_H_MULTIPLIERS]
    if HAVE_BM3D:
        grid.append(("bm3d_anscombe", "bm3d_anscombe", {}))
    return grid


# ------------------------------------------------------------------- workers
_SHARD_CACHE: dict = {}


def _cached_shard(gid: str):
    if gid not in _SHARD_CACHE:
        _SHARD_CACHE[gid] = load_shard(gid)[0]
    return _SHARD_CACHE[gid]


def _scored(low_den: np.ndarray, fraction: float, scale: float,
            x_ref: np.ndarray) -> dict:
    """SSIM/PSNR of one denoised input under both conventions."""
    out = {}
    for convention in CONVENTIONS:
        counts = low_den if convention == "raw" else low_den / np.float32(fraction)
        x_low = np.log1p(counts) / scale
        out[convention] = (compute_ssim(x_low, x_ref), compute_psnr(x_ref, x_low))
    return out


def evaluate_unit(job: tuple) -> dict:
    """One (geometry, view) work unit: full method grid + equivalence grid."""
    gid, geo_index, view, k_indices, extra_fractions = job
    data = _cached_shard(gid)
    ref = np.asarray(data["ref_counts"][view])
    fractions = [float(f) for f in data["fractions"]]
    scale = float(np.log1p(np.float32(ref.max())))
    x_ref = (np.log1p(ref.astype(np.float32)) / scale)

    grid = method_grid()
    methods = {key: {c: {f"{f:.2f}": {"ssim": [], "psnr": []}
                         for f in fractions} for c in CONVENTIONS}
               for key, _, _ in grid}
    equivalence = {}
    oracle = {}

    # ---- Stage 13.1: full method grid on the frozen shard realizations
    for fi, fraction in enumerate(fractions):
        fkey = f"{fraction:.2f}"
        for k in k_indices:
            low = np.asarray(data["low_counts"][view, fi, k])
            for key, method, params in grid:
                den = denoise(method, params, low)
                scores = _scored(den, fraction, scale, x_ref)
                for convention, (ssim, psnr) in scores.items():
                    methods[key][convention][fkey]["ssim"].append(ssim)
                    methods[key][convention][fkey]["psnr"].append(psnr)

    # ---- Stage 13.2: dense equivalence grid + scale-oracle sensitivity
    def _record_equivalence(fraction: float, low: np.ndarray, source: str):
        fkey = f"{fraction:.2f}"
        cell = equivalence.setdefault(
            fkey, {"source": source,
                   "raw": {"ssim": [], "psnr": []},
                   "corrected": {"ssim": [], "psnr": []}})
        scores = _scored(low.astype(np.float32), fraction, scale, x_ref)
        for convention, (ssim, psnr) in scores.items():
            cell[convention]["ssim"].append(ssim)
            cell[convention]["psnr"].append(psnr)
        # operational estimator scale replaces the oracle in BOTH images
        scale_est = float(np.log1p(np.float32(low.max()) / np.float32(fraction)))
        if scale_est > 0.0:
            x_ref_e = np.log1p(ref.astype(np.float32)) / scale_est
            x_low_e = np.log1p(low.astype(np.float32) / np.float32(fraction)) / scale_est
            oracle.setdefault(fkey, {"oracle": [], "estimator": []})
            oracle[fkey]["oracle"].append(scores["corrected"][0])
            oracle[fkey]["estimator"].append(compute_ssim(x_low_e, x_ref_e))

    for fi, fraction in enumerate(fractions):
        for k in k_indices:
            _record_equivalence(fraction, np.asarray(data["low_counts"][view, fi, k]),
                                "shard")
    ref64 = ref.astype(np.int64)
    for i, fraction in enumerate(extra_fractions):
        for k in k_indices:
            # base seed from the protocol formula, spawned per (geometry, view)
            # so the realizations are independent across the campaign
            ss = np.random.SeedSequence([seed_for(STAGE, 90 + i, k), geo_index, view])
            rng = np.random.default_rng(ss)
            _record_equivalence(fraction, rng.binomial(ref64, fraction), "on-the-fly")

    return {"methods": methods, "equivalence": equivalence, "oracle": oracle}


# ------------------------------------------------------------------ reduction
def _merge(dst: dict, src: dict) -> None:
    for key, value in src.items():
        if isinstance(value, dict):
            _merge(dst.setdefault(key, {}), value)
        elif isinstance(value, list):
            dst.setdefault(key, []).extend(value)
        else:
            dst[key] = value


def _cell(ssim_values, psnr_values) -> dict:
    s, p = summarize(ssim_values, digits=4), summarize(psnr_values, digits=2)
    return {"n": s["n"], "ssim_mean": s["mean"], "ssim_std": s["std"],
            "psnr_mean_db": p["mean"], "psnr_std_db": p["std"]}


# -------------------------------------------------------------------- figures
def figure_baselines(methods: dict, fraction_keys, path: Path) -> None:
    labels, identity_raw, identity_cor, best_val, best_name = [], [], [], [], []
    for fkey in fraction_keys:
        labels.append(f"{float(fkey) * 100:g} %")
        identity_raw.append(methods["identity"]["raw"][fkey]["ssim_mean"])
        identity_cor.append(methods["identity"]["corrected"][fkey]["ssim_mean"])
        candidates = [(m["corrected"][fkey]["ssim_mean"], key)
                      for key, m in methods.items() if key != "identity"]
        value, name = max(candidates) if candidates else (float("nan"), "-")
        best_val.append(value)
        best_name.append(name)

    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - 0.27, identity_raw, 0.26, label="identidad (crudo)")
    ax.bar(x, identity_cor, 0.26, label="identidad (corregido 1/f)")
    ax.bar(x + 0.27, best_val, 0.26, label="mejor clásica (corregido 1/f)")
    for xi, value, name in zip(x, best_val, best_name):
        if np.isfinite(value):
            ax.text(xi + 0.27, value + 0.012, name, ha="center", fontsize=7,
                    rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Fracción de eventos")
    ax.set_ylabel("SSIM en validación")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Etapa 13.1 — mejor método clásico por nivel frente a la identidad")
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def figure_equivalence(equivalence: dict, path: Path) -> None:
    fractions = sorted(float(f) for f in equivalence)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for convention, marker, label in (
            ("raw", "o", "crudo (escala compartida)"),
            ("corrected", "s", "corregido 1/f (equivalente a referencia)")):
        means = [equivalence[f"{f:.2f}"][convention]["ssim_mean"] for f in fractions]
        stds = [equivalence[f"{f:.2f}"][convention]["ssim_std"] for f in fractions]
        ax.errorbar(fractions, means, yerr=stds, marker=marker, capsize=4,
                    label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Fracción de eventos conservada")
    ax.set_ylabel("SSIM entrada vs. referencia")
    ax.set_title("Etapa 13.2 — grilla de equivalencia sin modelo (media ± DE)")
    ax.legend(loc="upper left")
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=str(STATS_PATH))
    parser.add_argument("--quick", action="store_true",
                        help="1 geometry, 1 view, k in {0,1} — smoke test")
    args = parser.parse_args()

    apply_style()
    geometry_ids = geometry_ids_for_split("val")
    views = list(range(6))
    k_indices = list(K_INDICES)
    extra_fractions = list(EXTRA_FRACTIONS)
    if args.quick:
        geometry_ids, views, k_indices = geometry_ids[:1], [0], [0, 1]

    missing = [g for g in geometry_ids if not (SHARD_DIR / f"{g}.npz").exists()]
    if missing:
        print(f"missing shards: {missing}")
        print("STAGE 13 CLASSICAL BASELINES: FAILED")
        return 1

    jobs = [(gid, gi, view, k_indices, extra_fractions)
            for gi, gid in enumerate(geometry_ids) for view in views]
    grid = method_grid()
    print(f"split=val geometries={len(geometry_ids)} views={len(views)} "
          f"k={k_indices} methods={len(grid)} conventions={len(CONVENTIONS)} "
          f"work_units={len(jobs)} workers={args.workers}")
    if not HAVE_BM3D:
        print("note: bm3d not installed — bm3d_anscombe skipped "
              "(optional per the Stage 13 plan)")
    if not HAVE_PYWT:
        print("WARNING: PyWavelets not installed — NLM methods skipped "
              "(the strongest classical baselines; install pywavelets)")

    t0 = time.perf_counter()
    merged: dict = {}
    workers = max(1, min(args.workers, len(jobs)))
    if workers == 1:
        for done, job in enumerate(jobs, start=1):
            _merge(merged, evaluate_unit(job))
            print(f"  unit {done}/{len(jobs)} done", flush=True)
    else:
        with mp.Pool(workers) as pool:
            for done, result in enumerate(
                    pool.imap_unordered(evaluate_unit, jobs), start=1):
                _merge(merged, result)
                print(f"  unit {done}/{len(jobs)} done", flush=True)
    elapsed = time.perf_counter() - t0

    # ---- reduce
    fraction_keys = [f"{f:.2f}" for f in FROZEN_FRACTIONS]
    methods_out = {}
    for key, method, params in grid:
        methods_out[key] = {"method": method, "params": params}
        for convention in CONVENTIONS:
            methods_out[key][convention] = {
                fkey: _cell(merged["methods"][key][convention][fkey]["ssim"],
                            merged["methods"][key][convention][fkey]["psnr"])
                for fkey in fraction_keys}

    equivalence_out = {}
    for fkey, cell in merged["equivalence"].items():
        equivalence_out[fkey] = {"source": cell["source"]}
        for convention in CONVENTIONS:
            equivalence_out[fkey].update(
                {convention: _cell(cell[convention]["ssim"],
                                   cell[convention]["psnr"])})

    oracle_out = {}
    for fkey, cell in merged["oracle"].items():
        o, e = summarize(cell["oracle"]), summarize(cell["estimator"])
        oracle_out[fkey] = {
            "n": o["n"],
            "ssim_oracle_scale": o["mean"],
            "ssim_estimator_scale": e["mean"],
            "delta_estimator_minus_oracle": round(e["mean"] - o["mean"], 4),
        }

    # ---- self-checks
    identity_raw = [methods_out["identity"]["raw"][f]["ssim_mean"]
                    for f in fraction_keys]
    monotone = bool(all(np.diff(identity_raw) < 0))  # fractions descending
    low_levels = [f for f in fraction_keys if float(f) <= 0.10]
    corrected_wins = bool(all(
        methods_out["identity"]["corrected"][f]["ssim_mean"]
        >= methods_out["identity"]["raw"][f]["ssim_mean"] for f in low_levels))
    best_at_5 = max(
        ((m["corrected"]["0.05"]["ssim_mean"], key)
         for key, m in methods_out.items() if key != "identity"),
        default=(float("nan"), "-"))

    summary = {
        "stage": "13.1+13.2",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": "val",
        "geometry_ids": geometry_ids,
        "views": views,
        "k_realizations_used": k_indices,
        "n_per_cell": len(geometry_ids) * len(views) * len(k_indices),
        "quick_mode": bool(args.quick),
        "conventions": {
            "raw": "log1p(denoise(low)) / log1p(ref.max())",
            "corrected": "log1p(denoise(low)/f) / log1p(ref.max())",
        },
        "nlm_settings": {"patch_size": NLM_PATCH_SIZE,
                         "patch_distance": NLM_PATCH_DISTANCE,
                         "fast_mode": True,
                         "h_multipliers_of_estimated_sigma": list(NLM_H_MULTIPLIERS)},
        "bm3d_available": HAVE_BM3D,
        "methods": methods_out,
        "equivalence_grid": equivalence_out,
        "equivalence_seed_derivation":
            "SeedSequence([20260805e6 + 13e4 + (90+i)*100 + k, geo_index, view])",
        "scale_oracle_sensitivity": oracle_out,
        "self_check": {
            "identity_ssim_monotone_in_fraction": monotone,
            "corrected_ge_raw_for_f_le_0.10": corrected_wins,
            "best_classical_ssim_at_0.05": best_at_5[0],
            "best_classical_method_at_0.05": best_at_5[1],
        },
        "runtime_minutes": round(elapsed / 60.0, 2),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_baselines(methods_out, fraction_keys, FIG_DIR / "stage13_baselines.png")
    figure_equivalence(equivalence_out, FIG_DIR / "stage13_equivalence_grid.png")

    print(f"\nwrote {out_path}")
    print(f"wrote {FIG_DIR / 'stage13_baselines.png'}")
    print(f"wrote {FIG_DIR / 'stage13_equivalence_grid.png'}")
    print(f"identity SSIM monotone in fraction: {monotone}")
    print(f"corrected >= raw for f <= 0.10:     {corrected_wins}")
    print(f"best classical at f=0.05: {best_at_5[1]} -> SSIM {best_at_5[0]}")
    print(f"elapsed: {elapsed / 60.0:.1f} min")

    passed = monotone and corrected_wins
    print("STAGE 13 CLASSICAL BASELINES:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
