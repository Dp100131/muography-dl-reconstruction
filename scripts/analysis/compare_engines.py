#!/usr/bin/env python3
"""E9 cross-validation: Geant4 pilot view vs surrogate engine, same geometry.

Both images share the acceptance convention (protocol lambda_open row
weights), so differences isolate TRANSPORT physics the surrogate does not
model: multiple scattering (angular blur), energy-loss straggling (softened
transmission edge) and muon decay in flight.

Inputs:
    --geant4-hist  data/campaign-geant4/pilot/geo-train-01_view0_hist.csv
    --surrogate    data/campaign/geo-train-01/geo-train-01_view0.npz
Outputs:
    figures/results/stage09_engine_comparison.png
    results/simulation/stage-09-engine-comparison.json (summary metrics)

Usage:
    python scripts/analysis/compare_engines.py --geant4-hist <csv> [--surrogate <npz>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
from plotting_style import apply_style, watermark_surrogate  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from skimage.metrics import structural_similarity  # noqa: E402

EL_MIN, EL_MAX = 2.0, 47.0


def edge_width_deg(profile: np.ndarray, elev: np.ndarray) -> float:
    """Width (deg) of the 20%-80% rise of the volcano silhouette edge."""
    prof = profile / max(profile.max(), 1e-12)
    lo_idx = np.argmax(prof >= 0.2)
    hi_idx = np.argmax(prof >= 0.8)
    if hi_idx <= lo_idx:
        return float("nan")
    return float(elev[hi_idx] - elev[lo_idx])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geant4-hist", required=True)
    parser.add_argument("--surrogate",
                        default=str(REPO / "data/campaign/geo-train-01/geo-train-01_view0.npz"))
    parser.add_argument("--out-fig",
                        default=str(REPO / "figures/results/stage09_engine_comparison.png"))
    parser.add_argument("--out-json",
                        default=str(REPO / "results/simulation/stage-09-engine-comparison.json"))
    args = parser.parse_args()

    g4 = np.loadtxt(args.geant4_hist, delimiter=",")          # (256 el, 256 az)
    surro = np.load(args.surrogate)
    ref = surro["counts_ref"].astype(np.float64)
    assert g4.shape == ref.shape, f"shape mismatch {g4.shape} vs {ref.shape}"

    elev = np.linspace(EL_MIN, EL_MAX, g4.shape[0])

    # --- global statistics
    total_ratio = float(g4.sum() / ref.sum())

    # --- normalized images for structural comparison (log1p, common scale)
    def norm(img):
        x = np.log1p(img)
        return x / max(x.max(), 1e-12)

    ssim = float(structural_similarity(norm(g4), norm(ref), data_range=1.0))

    # --- elevation profiles (azimuth-summed) and silhouette edge width
    prof_g4 = g4.sum(axis=1)
    prof_ref = ref.sum(axis=1)
    edge_g4 = edge_width_deg(prof_g4, elev)
    edge_ref = edge_width_deg(prof_ref, elev)

    # --- per-row ratio in the open-sky region (top rows): normalization check
    open_sky = slice(int(0.85 * len(elev)), None)
    open_ratio = float(prof_g4[open_sky].sum() / prof_ref[open_sky].sum())

    summary = {
        "geant4_hist": str(args.geant4_hist),
        "surrogate_npz": str(args.surrogate),
        "total_counts_geant4": int(g4.sum()),
        "total_counts_surrogate": int(ref.sum()),
        "total_ratio_g4_over_surrogate": round(total_ratio, 4),
        "open_sky_ratio": round(open_ratio, 4),
        "ssim_log_normalized": round(ssim, 4),
        "silhouette_edge_width_deg_geant4": round(edge_g4, 3),
        "silhouette_edge_width_deg_surrogate": round(edge_ref, 3),
        "interpretation": {
            "open_sky_ratio": "≈1 esperado (misma normalización de protocolo); "
                              "desviación = espectro EcoMug vs Guan + decaimiento en vuelo",
            "edge_width": "ensanchamiento Geant4 = dispersión múltiple + straggling "
                          "(física ausente en el surrogate)",
        },
    }

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, img, title in ((axes[0], ref, "Surrogate (analítico)"),
                           (axes[1], g4, "Geant4 (transporte completo)")):
        im = ax.imshow(np.log1p(img), origin="lower", aspect="auto",
                       extent=[-30, 30, EL_MIN, EL_MAX], cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Azimut relativo [°]")
        ax.set_ylabel("Elevación [°]")
        fig.colorbar(im, ax=ax, label="log(1+cuentas)")

    axes[2].plot(elev, prof_ref / prof_ref.max(), label="Surrogate", lw=1.8)
    axes[2].plot(elev, prof_g4 / prof_g4.max(), label="Geant4", lw=1.8)
    axes[2].set_xlabel("Elevación [°]")
    axes[2].set_ylabel("Perfil normalizado (suma en azimut)")
    axes[2].set_title(f"Silueta: borde {edge_ref:.1f}° vs {edge_g4:.1f}°")
    axes[2].legend()
    fig.suptitle("E9 cruzada — geo-train-01 vista 0, 30 días de exposición")
    fig.tight_layout()
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)

    Path(args.out_json).write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
