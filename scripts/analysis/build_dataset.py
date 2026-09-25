#!/usr/bin/env python3
"""Stage 11 — build the training-pair dataset by Poisson thinning.

For every campaign view (19 geometries x 6 views): reference counts (100%)
plus K=20 realizations per fraction {50%, 10%, 5%, 1%} sampled as
Poisson(lambda * f) — statistically EXACT event thinning (a thinned Poisson
process is Poisson). Seeds derived per (geometry, view, fraction, k) from
the master seed and recorded in shard metadata.

Also computes the no-model baseline: SSIM/PSNR of preprocessed input vs
reference per fraction (mean +/- std) with a monotonicity check, plus count
histograms and an example-pair figure.

Run:  python scripts/analysis/build_dataset.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

from muography_dataset import preprocess_pair  # noqa: E402
from plotting_style import apply_style, watermark_surrogate  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

import os

# Engine selection: surrogate campaign by default; set MUOGRAPHY_ENGINE=geant4
# to build from the Geant4 campaign (binomial thinning branch).
_ENGINE = os.environ.get("MUOGRAPHY_ENGINE", "surrogate")
CAMPAIGN_DIR = REPO_ROOT / "data" / (
    "campaign-geant4" if _ENGINE == "geant4" else "campaign")
SHARD_DIR = REPO_ROOT / "dataset" / "shards"
STATS_DIR = REPO_ROOT / "results" / "dataset" / "stats"
FIG_DIR = REPO_ROOT / "figures" / "results"
MASTER_SEED = 20260805
FRACTIONS = np.array([0.50, 0.10, 0.05, 0.01])
K_REALIZATIONS = 20
N_VIEWS = 6


def build_shard(geo_dir: Path, geo_index: int):
    gid = geo_dir.name
    ref_counts, lam_maps, view_meta = [], [], []
    for vi in range(N_VIEWS):
        data = np.load(geo_dir / f"{gid}_view{vi}.npz")
        ref_counts.append(data["counts_ref"].astype(np.uint32))
        # Surrogate views carry the analytic rate map (lam); Geant4 views
        # carry only real transported counts.
        lam_maps.append(data["lam"].astype(np.float64) if "lam" in data else None)
        view_meta.append(json.loads(
            (geo_dir / f"{gid}_view{vi}.meta.json").read_text(encoding="utf-8")))

    ref_counts = np.stack(ref_counts)
    h, w = ref_counts.shape[1:]

    low = np.empty((N_VIEWS, len(FRACTIONS), K_REALIZATIONS, h, w), dtype=np.uint16)
    seeds_used = {}
    for vi in range(N_VIEWS):
        for fi, frac in enumerate(FRACTIONS):
            ss = np.random.SeedSequence(
                [MASTER_SEED, 11, geo_index, vi, fi]).spawn(K_REALIZATIONS)
            for k, s in enumerate(ss):
                rng = np.random.default_rng(s)
                if lam_maps[vi] is not None:
                    # Poisson thinning: exact for a Poisson rate map.
                    sample = rng.poisson(lam_maps[vi] * frac)
                else:
                    # Binomial thinning: exact subsampling of REAL events
                    # (each detected event kept independently with prob=frac).
                    sample = rng.binomial(ref_counts[vi].astype(np.int64), frac)
                low[vi, fi, k] = np.minimum(sample, np.iinfo(np.uint16).max)
            seeds_used[f"view{vi}_frac{frac}"] = [int(x.generate_state(1)[0]) for x in ss]

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SHARD_DIR / f"{gid}.npz",
                        ref_counts=ref_counts, low_counts=low,
                        fractions=FRACTIONS)
    meta = {
        "geometry_id": gid,
        "n_views": N_VIEWS,
        "fractions": FRACTIONS.tolist(),
        "k_realizations": K_REALIZATIONS,
        "master_seed": MASTER_SEED,
        "thinning": ("Poisson(lambda*f) — exact thinning equivalence"
                     if lam_maps[0] is not None else
                     "Binomial(counts, f) — exact real-event subsampling"),
        "seed_derivation": "SeedSequence([master, 11, geo_index, view, frac_index]).spawn(K)",
        "seeds": seeds_used,
        "source_views": [m.get("git_commit", m.get("engine", "unknown"))
                         for m in view_meta],
        # Surrogate metas carry exposure_days; Geant4 merged metas are
        # exposure-calibrated by construction (30 d via --days, see runbook).
        "exposure_days_reference": view_meta[0].get("exposure_days", 30.0),
    }
    (SHARD_DIR / f"{gid}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return ref_counts, low


def baseline_metrics(ref_counts, low_counts):
    """SSIM/PSNR of preprocessed low vs ref for every (view, frac, k)."""
    out = {f: {"ssim": [], "psnr": []} for f in range(len(FRACTIONS))}
    for vi in range(ref_counts.shape[0]):
        for fi in range(len(FRACTIONS)):
            for k in range(low_counts.shape[2]):
                x_low, x_ref = preprocess_pair(low_counts[vi, fi, k],
                                               ref_counts[vi])
                out[fi]["ssim"].append(
                    structural_similarity(x_low, x_ref, data_range=1.0))
                out[fi]["psnr"].append(
                    peak_signal_noise_ratio(x_ref, x_low, data_range=1.0))
    return out


def main() -> int:
    apply_style()
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    geo_dirs = sorted(p for p in CAMPAIGN_DIR.glob("geo-*") if p.is_dir())
    assert len(geo_dirs) == 19, f"expected 19 geometries, found {len(geo_dirs)}"

    t0 = time.perf_counter()
    agg = {fi: {"ssim": [], "psnr": []} for fi in range(len(FRACTIONS))}
    n_pairs = 0
    example = None

    for geo_index, geo_dir in enumerate(geo_dirs):
        ref, low = build_shard(geo_dir, geo_index)
        n_pairs += low.shape[0] * low.shape[1] * low.shape[2]
        metrics = baseline_metrics(ref, low)
        for fi in agg:
            agg[fi]["ssim"].extend(metrics[fi]["ssim"])
            agg[fi]["psnr"].extend(metrics[fi]["psnr"])
        if geo_dir.name == "geo-train-08":
            example = (ref[0], [low[0, fi, 0] for fi in range(len(FRACTIONS))])
        print(f"{geo_dir.name}: shard built", flush=True)

    # ---- aggregate baseline table + monotonicity
    table = {}
    for fi, frac in enumerate(FRACTIONS):
        s, p = np.array(agg[fi]["ssim"]), np.array(agg[fi]["psnr"])
        table[f"{frac:.2f}"] = {
            "n": int(len(s)),
            "ssim_mean": round(float(s.mean()), 4),
            "ssim_std": round(float(s.std(ddof=1)), 4),
            "psnr_mean_db": round(float(p.mean()), 2),
            "psnr_std_db": round(float(p.std(ddof=1)), 2),
        }
    ssim_means = [table[f"{f:.2f}"]["ssim_mean"] for f in FRACTIONS]
    monotone = bool(all(np.diff(ssim_means) < 0))  # fractions are descending

    summary = {
        "n_geometries": len(geo_dirs),
        "n_views_per_geometry": N_VIEWS,
        "fractions": FRACTIONS.tolist(),
        "k_realizations": K_REALIZATIONS,
        "total_pairs": n_pairs,
        "baseline_no_model": table,
        "ssim_monotonically_decreasing_with_subsampling": monotone,
        "build_minutes": round((time.perf_counter() - t0) / 60.0, 1),
    }
    (STATS_DIR / "stage11_baseline.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    # ---- figures
    fracs_pct = (FRACTIONS * 100).astype(int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ssim_std = [table[f"{f:.2f}"]["ssim_std"] for f in FRACTIONS]
    psnr_m = [table[f"{f:.2f}"]["psnr_mean_db"] for f in FRACTIONS]
    psnr_s = [table[f"{f:.2f}"]["psnr_std_db"] for f in FRACTIONS]
    ax1.errorbar(fracs_pct, ssim_means, yerr=ssim_std, marker="o", capsize=4)
    ax1.set_xscale("log"); ax1.set_xlabel("Fracción de eventos [%]")
    ax1.set_ylabel("SSIM entrada vs. referencia")
    ax1.set_title("Línea base sin modelo — SSIM (media ± DE)")
    ax2.errorbar(fracs_pct, psnr_m, yerr=psnr_s, marker="s", capsize=4, color="C1")
    ax2.set_xscale("log"); ax2.set_xlabel("Fracción de eventos [%]")
    ax2.set_ylabel("PSNR [dB]")
    ax2.set_title("Línea base sin modelo — PSNR (media ± DE)")
    if _ENGINE != "geant4":
        watermark_surrogate(fig)
    fig.savefig(FIG_DIR / "stage11_baseline_metrics.png")
    plt.close(fig)

    if example is not None:
        ref0, lows = example
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
        x_ref = np.log1p(ref0.astype(np.float32))
        vmax = x_ref.max()
        axes[0].imshow(x_ref, origin="lower", vmax=vmax)
        axes[0].set_title("Referencia (100 %)")
        for ax, fi in zip(axes[1:], range(len(FRACTIONS))):
            ax.imshow(np.log1p(lows[fi].astype(np.float32)), origin="lower", vmax=vmax)
            ax.set_title(f"{fracs_pct[fi]} % de eventos")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Pares de entrenamiento por submuestreo — geo-train-08, vista 0 "
                     "(escala log1p común)")
        if _ENGINE != "geant4":
            watermark_surrogate(fig)
        fig.savefig(FIG_DIR / "stage11_example_pairs.png")
        plt.close(fig)

    # Freeze guard: rebuilding on a different numpy/platform yields DIFFERENT
    # binomial realizations. If a freeze exists, verify we reproduced it.
    splits_path = REPO_ROOT / "results" / "dataset" / "splits.json"
    frozen_ok = True
    if splits_path.exists():
        import hashlib
        frozen = json.loads(splits_path.read_text(encoding="utf-8"))
        expected = frozen.get("shard_sha256", {})
        mismatched = []
        for gid, exp in expected.items():
            shard = SHARD_DIR / f"{gid}.npz"
            if shard.exists():
                h = hashlib.sha256(shard.read_bytes()).hexdigest()
                if h != exp:
                    mismatched.append(gid)
        if mismatched:
            frozen_ok = False
            print(f"WARNING: {len(mismatched)}/{len(expected)} rebuilt shards "
                  "DIFFER from the frozen dataset hashes in splits.json.")
            print("This build does NOT reproduce the frozen release — expected "
                  "when numpy or the platform differs from the freeze "
                  "environment. Do NOT train or freeze from these shards; "
                  "restore the released ones.")

    print(json.dumps(summary, indent=2))
    print("STAGE 11 DATASET BUILD:",
          "PASSED" if (monotone and frozen_ok) else
          ("FAILED (monotonicity)" if not monotone else
           "PASSED WITH WARNING (frozen-hash mismatch)"))
    return 0 if monotone else 1


if __name__ == "__main__":
    sys.exit(main())
