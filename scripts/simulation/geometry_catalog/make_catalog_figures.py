#!/usr/bin/env python3
"""Stage 7 — visual inspection figures for the geometry catalog.

Renders a vertical density cross-section (x-z plane at y=0) for every
geometry in the catalog, as a grid figure plus one detailed figure for a
representative geometry. This is the documented surrogate for the Geant4
OpenGL/Qt viewer inspection (which remains pending for the codirector with
the real Geant4 build).

Run:  python scripts/simulation/geometry_catalog/make_catalog_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from surrogate.geometry import VolcanoGeometry  # noqa: E402
from plotting_style import apply_style, watermark_surrogate  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

GEO_DIR = Path(__file__).resolve().parent / "geometries"
FIG_DIR = REPO_ROOT / "figures" / "results"


def slice_plane_y(geo: VolcanoGeometry) -> float:
    """Pick the y-plane that actually intersects the first anomaly.

    A fixed y=0 slice misses anomalies placed off-axis (found during visual
    inspection) — slicing through each geometry's first anomaly guarantees
    the inspection figure shows what the catalog declares.
    """
    if not geo.anomalies:
        return 0.0
    return float(geo.anomalies[0].params.get("cy_m", 0.0))


def cross_section(geo: VolcanoGeometry, n: int = 400):
    half = geo.base_radius_m * 1.05
    y0 = slice_plane_y(geo)
    x = np.linspace(-half, half, n)
    z = np.linspace(0.0, geo.height_m * 1.1, n)
    xx, zz = np.meshgrid(x, z)
    rho = geo.density(xx, np.full_like(xx, y0), zz)
    return x, z, rho, y0


def main() -> None:
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(GEO_DIR.glob("*.json"))

    fig, axes = plt.subplots(4, 5, figsize=(16, 11))
    for ax, path in zip(axes.ravel(), paths):
        geo = VolcanoGeometry.from_json(path)
        x, z, rho, y0 = cross_section(geo, n=250)
        im = ax.pcolormesh(x, z, rho, vmin=0.0, vmax=3.3, shading="auto")
        ax.set_title(f"{geo.geometry_id}  (y = {y0:.0f} m)", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(paths):]:
        ax.axis("off")
    fig.suptitle("Catálogo de geometrías volcánicas — corte de densidad por el plano "
                 "de la primera anomalía (19 configuraciones)", fontsize=13)
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, label="Densidad [g/cm³]")
    cbar.ax.tick_params(labelsize=8)
    watermark_surrogate(fig)
    fig.savefig(FIG_DIR / "stage07_catalog_cross_sections.png")
    plt.close(fig)

    # Detailed view of the richest geometry (3 anomaly types)
    geo = VolcanoGeometry.from_json(GEO_DIR / "geo-train-08.json")
    x, z, rho, y0 = cross_section(geo, n=500)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(x, z, rho, vmin=0.0, vmax=3.3, shading="auto")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"{geo.geometry_id} (corte y = {y0:.0f} m): {geo.description}")
    fig.colorbar(im, ax=ax, label="Densidad [g/cm³]")
    watermark_surrogate(fig)
    fig.savefig(FIG_DIR / "stage07_geometry_detail.png")
    plt.close(fig)

    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
