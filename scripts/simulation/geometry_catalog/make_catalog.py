#!/usr/bin/env python3
"""Stage 7 — generate the versioned catalog of 19 volcano geometries.

Deterministic (seeded) so the catalog is reproducible from this script alone
(PA-1). Coverage requirements from the provisional Phase B parameters:
- every anomaly type (conduit / cavity / dense_body) appears alone and in
  combinations across the catalog,
- one anomaly-free control geometry,
- one fully-held-out generalization geometry (geo-gen-01).

Run:  python scripts/simulation/geometry_catalog/make_catalog.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "geometries"
CATALOG_SEED = 20260805

# (role, index, anomaly recipe) — 19 geometries total
# recipe letters: C=conduit, V=cavity(ies), D=dense body
RECIPES = [
    ("train", 1, ""),      # control: edificio homogéneo
    ("train", 2, "C"),
    ("train", 3, "V"),
    ("train", 4, "D"),
    ("train", 5, "CV"),
    ("train", 6, "CD"),
    ("train", 7, "VD"),
    ("train", 8, "CVD"),
    ("train", 9, "C"),
    ("train", 10, "VV"),
    ("train", 11, "CVV"),
    ("train", 12, "D"),
    ("val", 1, "C"),
    ("val", 2, "V"),
    ("val", 3, "CD"),
    ("test", 1, "CV"),
    ("test", 2, "D"),
    ("test", 3, "CVD"),
    ("gen", 1, "CVV"),     # geometría de generalización, excluida de todo
]


def build_geometry(rng: np.random.Generator, role: str, idx: int, recipe: str) -> dict:
    base_radius = float(rng.uniform(500.0, 700.0))
    height = float(rng.uniform(350.0, 450.0))
    anomalies = []
    desc_parts = []

    for kind_letter in recipe:
        if kind_letter == "C":
            radius = float(rng.uniform(15.0, 40.0))
            density = float(rng.uniform(1.8, 2.2))
            cx = float(rng.uniform(-40.0, 40.0))
            cy = float(rng.uniform(-40.0, 40.0))
            z1 = float(rng.uniform(0.75, 0.95) * height)
            anomalies.append({
                "kind": "conduit", "density_g_cm3": round(density, 3),
                "params": {"cx_m": round(cx, 1), "cy_m": round(cy, 1),
                           "radius_m": round(radius, 1),
                           "z0_m": 0.0, "z1_m": round(z1, 1)},
            })
            desc_parts.append(f"conducto r={radius:.0f} m ρ={density:.2f}")
        elif kind_letter == "V":
            radius = float(rng.uniform(20.0, 50.0))
            density = float(rng.uniform(0.0, 0.5))
            # place well inside the edifice at mid-height
            frac_h = rng.uniform(0.25, 0.55)
            cz = float(frac_h * height)
            max_r = base_radius * (1.0 - cz / height) - radius - 20.0
            rr = float(rng.uniform(0.1, 0.6) * max(max_r, 30.0))
            ang = float(rng.uniform(0, 2 * np.pi))
            anomalies.append({
                "kind": "cavity", "density_g_cm3": round(density, 3),
                "params": {"cx_m": round(rr * np.cos(ang), 1),
                           "cy_m": round(rr * np.sin(ang), 1),
                           "cz_m": round(cz, 1), "radius_m": round(radius, 1)},
            })
            desc_parts.append(f"cavidad r={radius:.0f} m ρ={density:.2f}")
        elif kind_letter == "D":
            ax_ = float(rng.uniform(40.0, 90.0))
            ay_ = float(rng.uniform(40.0, 90.0))
            az_ = float(rng.uniform(30.0, 70.0))
            density = float(rng.uniform(3.0, 3.3))
            frac_h = rng.uniform(0.15, 0.45)
            cz = float(frac_h * height)
            max_r = base_radius * (1.0 - cz / height) - max(ax_, ay_) - 20.0
            rr = float(rng.uniform(0.1, 0.5) * max(max_r, 30.0))
            ang = float(rng.uniform(0, 2 * np.pi))
            anomalies.append({
                "kind": "dense_body", "density_g_cm3": round(density, 3),
                "params": {"cx_m": round(rr * np.cos(ang), 1),
                           "cy_m": round(rr * np.sin(ang), 1),
                           "cz_m": round(cz, 1), "ax_m": round(ax_, 1),
                           "ay_m": round(ay_, 1), "az_m": round(az_, 1)},
            })
            desc_parts.append(f"cuerpo denso ρ={density:.2f}")

    gid = f"geo-{role}-{idx:02d}"
    return {
        "geometry_id": gid,
        "catalog_seed": CATALOG_SEED,
        "role": role,
        "edifice": {
            "shape": "cone",
            "base_radius_m": round(base_radius, 1),
            "height_m": round(height, 1),
            "rock_density_g_cm3": 2.65,
        },
        "anomalies": anomalies,
        "description": ("edificio homogéneo (control)" if not desc_parts
                        else " + ".join(desc_parts)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = np.random.SeedSequence(CATALOG_SEED).spawn(len(RECIPES))
    for (role, idx, recipe), seed in zip(RECIPES, seeds):
        rng = np.random.default_rng(seed)
        cfg = build_geometry(rng, role, idx, recipe)
        path = OUT_DIR / f"{cfg['geometry_id']}.json"
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"{cfg['geometry_id']}: {cfg['description']}")
    print(f"\n{len(RECIPES)} geometries written to {OUT_DIR}")


if __name__ == "__main__":
    main()
