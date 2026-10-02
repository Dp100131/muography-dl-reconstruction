#!/usr/bin/env python3
"""Build the direction/spectrum tables for the table-driven Geant4 generator.

Design (S0 calibration, campaign runbook):
- DIRECTION weighting per elevation row follows the PROTOCOL convention
  lambda_open = Phi_int(theta) * A * d_omega * T (flat detector area), taken
  from the surrogate campaign itself (lam / transmission â€” open-sky rate map,
  azimuth-independent per row). This keeps the Geant4 reference images on the
  same acceptance convention as the frozen protocol and the surrogate, so the
  E9 cross-comparison measures TRANSPORT physics only.
- MOMENTUM spectrum comes from the REAL EcoMug sample (1e7 muons, stage 6),
  as inverse CDFs conditional on elevation band. The ~29% EcoMug-vs-Guan
  spectral difference stays documented as a systematic.
- Charge ratio mu+/mu- = measured EcoMug value.

Outputs (small text files, versioned and shipped in the deploy bundle):
    scripts/simulation/geant4_app/tables-az-edges-v1/row_weights_30d.txt   # 256 rows: elev_deg lambda_open_30d
    scripts/simulation/geant4_app/tables-az-edges-v1/p_cdf.txt             # bands x quantiles inverse CDF
    scripts/simulation/geant4_app/tables-az-edges-v1/meta.json

Usage:
    python scripts/analysis/build_generator_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN_VIEW = REPO / "data/campaign/geo-train-01/geo-train-01_view0.npz"
ECOMUG_CSV = REPO / "scripts/simulation/ecomug_standalone/muons.csv"
OUT_DIR = REPO / "scripts/simulation/geant4_app/tables-az-edges-v1"

N_EL = 256
EL_MIN, EL_MAX = 2.0, 47.0
N_BANDS = 16          # elevation bands for conditional momentum spectra
N_QUANTILES = 257     # inverse-CDF sampling points per band


def build_row_weights() -> np.ndarray:
    meta = json.loads(CAMPAIGN_VIEW.with_suffix(".meta.json").read_text())
    grid = meta.get("angular_grid", {})
    if (grid.get("azimuth_convention") != "az-edges-v1" or
            grid.get("n_az") != 256 or grid.get("n_el") != N_EL or
            grid.get("az_edges_deg") != [-30.0, 30.0] or
            grid.get("az_step_deg") != 60/256 or
            meta.get("exposure_days") != 30):
        raise ValueError("requires a NEW 30-day surrogate view with az-edges-v1 metadata")
    z = np.load(CAMPAIGN_VIEW)
    lam = z["lam"].astype(np.float64)
    trans = z["transmission"].astype(np.float64)
    if lam.shape != (256, 256) or trans.shape != lam.shape:
        raise ValueError("unexpected angular map shape")
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_open = np.where(trans > 0, lam / trans, np.nan)
    # Open-sky rate is azimuth-independent: check row constancy, then reduce.
    row_mean = np.nanmean(lam_open, axis=1)
    row_std = np.nanstd(lam_open, axis=1)
    rel = np.nanmax(row_std / row_mean)
    assert rel < 1e-6, f"lambda_open not azimuth-constant (max rel std {rel:.2e})"
    row_total = row_mean * lam_open.shape[1]  # total open-sky counts per row, 30 d
    return row_total


def build_p_cdfs() -> tuple[np.ndarray, np.ndarray, float]:
    data = np.loadtxt(ECOMUG_CSV, delimiter=",", skiprows=1,
                      usecols=(0, 1, 3))            # p_gev, theta_rad, charge
    p, theta, charge = data[:, 0], data[:, 1], data[:, 2]
    elev = np.degrees(theta) - 90.0                  # EcoMug theta in [90,180] downward
    # Keep the angular window only
    mask = (elev >= EL_MIN) & (elev <= EL_MAX)
    p, elev, charge = p[mask], elev[mask], charge[mask]

    edges = np.linspace(EL_MIN, EL_MAX, N_BANDS + 1)
    qs = np.linspace(0.0, 1.0, N_QUANTILES)
    cdfs = np.empty((N_BANDS, N_QUANTILES))
    for b in range(N_BANDS):
        sel = p[(elev >= edges[b]) & (elev < edges[b + 1])]
        # Near-horizontal bands are naturally sparse (EcoMug's cos-theta
        # weighting); ~1500 muons still yield stable percentile tables and
        # those rows carry the least open-sky weight anyway.
        assert len(sel) > 1000, f"band {b}: only {len(sel)} muons"
        cdfs[b] = np.quantile(sel, qs)
    charge_ratio = float((charge > 0).sum() / max((charge < 0).sum(), 1))
    return edges, cdfs, charge_ratio


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError("use a new output directory for angular tables")
    OUT_DIR.mkdir(parents=True)

    row_total = build_row_weights()
    el_centers = np.linspace(EL_MIN, EL_MAX, N_EL)
    with open(OUT_DIR / "row_weights_30d.txt", "w", encoding="ascii") as fh:
        fh.write("# elev_center_deg lambda_open_counts_30days_per_row\n")
        for e, w in zip(el_centers, row_total):
            fh.write(f"{e:.6f} {w:.6e}\n")

    edges, cdfs, charge_ratio = build_p_cdfs()
    with open(OUT_DIR / "p_cdf.txt", "w", encoding="ascii") as fh:
        fh.write(f"# {len(edges)-1} elevation bands, {cdfs.shape[1]} quantiles"
                 " (momentum GeV/c, inverse CDF per band)\n")
        fh.write("# band_edges_deg: " + " ".join(f"{e:.3f}" for e in edges) + "\n")
        for b in range(cdfs.shape[0]):
            fh.write(" ".join(f"{v:.6e}" for v in cdfs[b]) + "\n")

    total_30d = float(row_total.sum())
    meta = {
        "azimuth_convention": "az-edges-v1",
        "n_azimuth_columns": 256,
        "azimuth_min_deg": -30.0, "azimuth_max_deg": 30.0,
        "direction_convention": "protocol lambda_open = Phi_int*A*d_omega "
                                "(flat area, Guan-normalized, from surrogate campaign)",
        "spectrum_source": "real EcoMug sample (stage 6, 1e7 muons), "
                           "conditional on elevation band",
        "charge_ratio_mu_plus_minus": round(charge_ratio, 4),
        "total_open_sky_counts_30days_per_view": total_30d,
        "n_elevation_rows": N_EL,
        "n_spectrum_bands": cdfs.shape[0],
        "source_view": CAMPAIGN_VIEW.name,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
