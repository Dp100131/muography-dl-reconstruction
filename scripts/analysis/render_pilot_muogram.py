#!/usr/bin/env python3
"""Stage 8 pilot — render the Geant4 pilot muogram from the angular histogram.

Reads the CSV histogram written by the Geant4 application (one row per
elevation bin, azimuth along the columns) plus its .meta.json sidecar, and
writes a two-panel figure: the log-scaled count map and the azimuth-summed
elevation profile that shows the edifice silhouette.

Usage:
    python scripts/analysis/render_pilot_muogram.py \
        --hist data/pilot-geant4/geo-train-01_view0_hist.csv \
        --out figures/results/stage08_pilot_muogram_geant4.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
from plotting_style import apply_style  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hist", default=str(REPO / "data/pilot-geant4/geo-train-01_view0_hist.csv"))
    parser.add_argument("--out", default=str(REPO / "figures/results/stage08_pilot_muogram_geant4.png"))
    args = parser.parse_args()

    hist_path = Path(args.hist)
    meta_path = hist_path.with_name(hist_path.name.replace("_hist.csv", ".meta.json"))
    meta = json.loads(meta_path.read_text())
    grid = meta["grid"]
    az_half = float(grid["az_half_deg"])
    el_min, el_max = float(grid["el_min_deg"]), float(grid["el_max_deg"])

    counts = np.loadtxt(hist_path, delimiter=",")  # (n_el, n_az)
    assert counts.shape == (grid["n_el"], grid["n_az"]), counts.shape
    elev = np.linspace(el_min, el_max, counts.shape[0])
    profile = counts.sum(axis=1)
    profile /= max(profile.max(), 1e-12)

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})

    im = axes[0].imshow(np.log1p(counts), origin="lower", aspect="auto",
                        extent=[-az_half, az_half, el_min, el_max], cmap="viridis")
    axes[0].set_xlabel("Azimut relativo [°]")
    axes[0].set_ylabel("Elevación [°]")
    axes[0].set_title("Mapa de cuentas (Geant4, 30 días)")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], label="log(1 + cuentas)")

    axes[1].plot(elev, profile, lw=1.8)
    axes[1].set_xlabel("Elevación [°]")
    axes[1].set_ylabel("Perfil normalizado (suma en azimut)")
    axes[1].set_title("Silueta del edificio")
    axes[1].set_xlim(el_min, el_max)
    axes[1].set_ylim(0, 1.05)

    detected = int(meta["detected_in_bins"])
    transported = int(meta["events_transported"])
    fig.suptitle(f"{meta['geometry_id']} vista {meta['view_index']} — "
                 f"{transported/1e6:.2f} M eventos transportados, "
                 f"{detected/1e6:.2f} M detectados, distancia {meta['detector']['distance_m']:.0f} m")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out} ({counts.sum():.0f} counts, max bin {counts.max():.0f})")


if __name__ == "__main__":
    main()
