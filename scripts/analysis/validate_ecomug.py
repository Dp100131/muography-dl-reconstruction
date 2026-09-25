#!/usr/bin/env python3
"""Stage 6 (real generator) — validate an EcoMug CSV against Guan et al. 2015.

Consumes the CSV from scripts/simulation/ecomug_standalone/ecomug_gen and
applies the SAME analysis as validate_flux_sampler.py (chi2, residuals,
anchors), so the surrogate-sampler validation and the real-EcoMug validation
are directly comparable.

Usage:
    ./ecomug_gen 10000000 20260805 > muons.csv
    python scripts/analysis/validate_ecomug.py --csv scripts/simulation/ecomug_standalone/muons.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from validate_flux_sampler import (  # noqa: E402
    FIG_DIR, OUT_DIR, chi2_and_residuals,
    expected_counts_energy, expected_counts_zenith,
)
from plotting_style import apply_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    p = data["p_gev"]
    theta = data["theta_rad"]
    # EcoMug zenith convention: downward-going muons have theta in [pi/2, pi]
    # measured from +z. Convert to conventional zenith angle from vertical.
    zenith = np.pi - theta if np.median(theta) > np.pi / 2 else theta
    cos_z = np.cos(zenith)

    # EcoMug sky mode generates the flux THROUGH a horizontal surface, so the
    # analytic expectation must carry the cos(theta) acceptance factor.
    e_edges = np.logspace(np.log10(0.5), np.log10(100.0), 41)
    obs_e, _ = np.histogram(p, bins=e_edges)
    exp_e = expected_counts_energy(e_edges, weight_cos=True) * obs_e.sum()
    chi2_e, ndf_e, res_e = chi2_and_residuals(obs_e.astype(float), exp_e)

    c_edges = np.linspace(np.cos(np.deg2rad(70.0)), 1.0, 36)
    obs_c, _ = np.histogram(cos_z, bins=c_edges)
    exp_c = expected_counts_zenith(c_edges, weight_cos=True) * obs_c.sum()
    chi2_c, ndf_c, res_c = chi2_and_residuals(obs_c.astype(float), exp_c)

    summary = {
        "engine": "EcoMug (real generator)",
        "csv": args.csv,
        "n_muons": int(len(p)),
        "energy_spectrum": {
            "chi2_ndf": round(chi2_e / ndf_e, 3),
            "max_abs_rel_residual": round(float(np.nanmax(np.abs(res_e))), 4),
            "acceptance_lt_10pct": bool(np.nanmax(np.abs(res_e)) < 0.10),
        },
        "zenith_distribution": {
            "chi2_ndf": round(chi2_c / ndf_c, 3),
            "max_abs_rel_residual": round(float(np.nanmax(np.abs(res_c))), 4),
            "acceptance_lt_10pct": bool(np.nanmax(np.abs(res_c)) < 0.10),
        },
        "mean_momentum_gev": round(float(p.mean()), 3),
        "charge_ratio_mu_plus_minus": round(
            float((data["charge"] > 0).sum() / max((data["charge"] < 0).sum(), 1)), 3),
        "note": "Comparación contra Guan et al. 2015; EcoMug implementa su propia "
                "parametrización — desviaciones moderadas entre modelos son esperables "
                "y se documentan, no se ocultan.",
    }

    e_centers = np.sqrt(e_edges[:-1] * e_edges[1:])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(e_centers, obs_e, drawstyle="steps-mid", label="EcoMug")
    ax.loglog(e_centers, exp_e, "--", label="Guan et al. (2015)")
    ax.set_xlabel("Momento del muón [GeV/c]")
    ax.set_ylabel("Cuentas por bin")
    ax.set_title("EcoMug vs. parametrización de Guan — espectro")
    ax.legend()
    fig.savefig(FIG_DIR / "stage06_ecomug_spectrum.png")
    plt.close(fig)

    (OUT_DIR / "stage06_ecomug_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
