#!/usr/bin/env python3
"""Stage 6 — physical validation of the muon flux model and sampler.

Validates the surrogate sampler against the analytic Guan et al. (2015)
parametrization and PDG (2022) anchors:

  1. Momentum/energy spectrum: sampled histogram vs analytic expectation,
     chi2/ndf and per-bin relative residuals (acceptance: <10% in 0.5-100 GeV).
  2. Zenith distribution: sampled vs analytic, plus cos^n(theta) fit
     (PDG anchor: n ~ 2 for ~3 GeV muons).
  3. Integral anchors: vertical integral flux above ~1 GeV vs the canonical
     "~1 muon / cm^2 / min" (~70 m^-2 s^-1 sr^-1), mean energy ~4 GeV.

Outputs: figures (figures/results/) + JSON summary consumed by the
stage-06 report. Run:  python scripts/analysis/validate_flux_sampler.py [--n 10000000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from surrogate import flux_model as fm  # noqa: E402
from plotting_style import apply_style, watermark_surrogate  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = REPO_ROOT / "figures" / "results"
OUT_DIR = REPO_ROOT / "data" / "output" / "validation"
MASTER_SEED = 20260805


def expected_counts_energy(edges: np.ndarray, cth_range=(0.0, 1.0),
                           weight_cos: bool = False) -> np.ndarray:
    """Analytic expected fraction per energy bin (marginal over cos_theta).

    weight_cos=True applies the cos(theta) acceptance factor of a HORIZONTAL
    generation surface (flux THROUGH the surface, EcoMug sky mode). The
    per-solid-angle sampler (surrogate) uses weight_cos=False.
    """
    c_grid = np.linspace(cth_range[0] + 1e-6, cth_range[1], 400)
    w = c_grid if weight_cos else np.ones_like(c_grid)
    e_centers = np.sqrt(edges[:-1] * edges[1:])
    frac = np.empty(len(e_centers))
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        e_sub = np.logspace(np.log10(lo), np.log10(hi), 24)
        pdf = fm.diff_flux(e_sub[np.newaxis, :], c_grid[:, np.newaxis])
        frac[i] = np.trapezoid(np.trapezoid(pdf, e_sub, axis=1) * w, c_grid)
    return frac / frac.sum()


def expected_counts_zenith(edges: np.ndarray, e_range=(0.5, 1.0e4),
                           weight_cos: bool = False) -> np.ndarray:
    """Analytic expected fraction per cos(theta) bin (marginal over E).

    See expected_counts_energy for the weight_cos semantics.
    """
    e_grid = np.logspace(np.log10(e_range[0]), np.log10(e_range[1]), 600)
    frac = np.empty(len(edges) - 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        c_sub = np.linspace(max(lo, 1e-6), hi, 24)
        w = c_sub if weight_cos else np.ones_like(c_sub)
        pdf = fm.diff_flux(e_grid[np.newaxis, :], c_sub[:, np.newaxis])
        frac[i] = np.trapezoid(np.trapezoid(pdf, e_grid, axis=1) * w, c_sub)
    return frac / frac.sum()


def chi2_and_residuals(observed: np.ndarray, expected: np.ndarray):
    mask = expected > 0
    chi2 = float(np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask]))
    ndf = int(mask.sum() - 1)
    rel = np.full_like(expected, np.nan)
    rel[mask] = (observed[mask] - expected[mask]) / expected[mask]
    return chi2, ndf, rel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000_000)
    args = parser.parse_args()

    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(np.random.SeedSequence(MASTER_SEED).spawn(1)[0])
    t0 = time.perf_counter()
    energy, theta = fm.sample_muons(args.n, rng)
    t_sample = time.perf_counter() - t0
    cos_theta = np.cos(theta)

    # ---------------- 1. energy spectrum ----------------
    e_edges = np.logspace(np.log10(0.5), np.log10(100.0), 41)
    obs_e, _ = np.histogram(energy, bins=e_edges)
    in_range = obs_e.sum()
    exp_e_frac = expected_counts_energy(e_edges)
    exp_e = exp_e_frac * in_range
    chi2_e, ndf_e, res_e = chi2_and_residuals(obs_e.astype(float), exp_e)
    max_res_e = float(np.nanmax(np.abs(res_e)))

    # ---------------- 2. zenith distribution ----------------
    # 0-70 deg -> cos(theta) in [cos(70deg), 1]
    c_edges = np.linspace(np.cos(np.deg2rad(70.0)), 1.0, 36)
    obs_c, _ = np.histogram(cos_theta, bins=c_edges)
    exp_c_frac = expected_counts_zenith(c_edges)
    exp_c = exp_c_frac * obs_c.sum()
    chi2_c, ndf_c, res_c = chi2_and_residuals(obs_c.astype(float), exp_c)
    max_res_c = float(np.nanmax(np.abs(res_c)))

    # cos^n fit on the angular DISTRIBUTION of arrivals per solid angle:
    # dN/dOmega ~ integral flux(theta) ~ cos^n(theta). Estimate n by log-log
    # regression of the per-cos-bin density against cos(theta).
    c_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
    density = obs_c / np.diff(c_edges)  # per unit cos(theta), azimuth-integrated
    good = density > 0
    n_exp = float(np.polyfit(np.log(c_centers[good]), np.log(density[good]), 1)[0])

    # ---------------- 3. integral anchors ----------------
    mean_e = float(energy.mean())
    phi_v_05 = fm.open_sky_vertical_integral_flux()          # cm^-2 s^-1 sr^-1, E>0.5
    phi_v_1 = float(fm.integral_flux(1.0, 1.0).ravel()[0])
    # canonical "1 muon/cm^2/min" for a horizontal detector: integrate
    # Phi*cos(theta) over the upper hemisphere
    c_grid = np.linspace(1e-3, 1.0, 200)
    phi_int_c = fm.integral_flux(np.full_like(c_grid, 0.5), c_grid)
    rate_horiz = 2.0 * np.pi * np.trapezoid(phi_int_c * c_grid, c_grid)  # cm^-2 s^-1
    rate_per_min = float(rate_horiz * 60.0)

    summary = {
        "engine": "surrogate v0.9 (Guan 2015 + Chirkin)",
        "master_seed": MASTER_SEED,
        "n_sampled": args.n,
        "sampling_seconds": round(t_sample, 1),
        "energy_spectrum": {
            "range_gev": [0.5, 100.0], "bins": len(e_edges) - 1,
            "chi2": round(chi2_e, 1), "ndf": ndf_e,
            "chi2_ndf": round(chi2_e / ndf_e, 3),
            "max_abs_rel_residual": round(max_res_e, 4),
            "acceptance_lt_10pct": bool(max_res_e < 0.10),
        },
        "zenith_distribution": {
            "range_deg": [0.0, 70.0], "bins": len(c_edges) - 1,
            "chi2": round(chi2_c, 1), "ndf": ndf_c,
            "chi2_ndf": round(chi2_c / ndf_c, 3),
            "max_abs_rel_residual": round(max_res_c, 4),
            "acceptance_lt_10pct": bool(max_res_c < 0.10),
            "cos_exponent_fit": round(n_exp, 3),
            "pdg_anchor_exponent": 2.0,
        },
        "anchors": {
            "mean_energy_gev": round(mean_e, 3),
            "pdg_mean_energy_gev": 4.0,
            "vertical_integral_flux_e_gt_0p5_cm2_s_sr": float(f"{phi_v_05:.4e}"),
            "vertical_integral_flux_e_gt_1_cm2_s_sr": float(f"{phi_v_1:.4e}"),
            "horizontal_rate_muons_per_cm2_min": round(rate_per_min, 3),
            "pdg_anchor_rate": 1.0,
        },
    }

    # ---------------- figures ----------------
    e_centers = np.sqrt(e_edges[:-1] * e_edges[1:])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.loglog(e_centers, obs_e, drawstyle="steps-mid", label="Muestreado (10⁷ muones)")
    ax1.loglog(e_centers, exp_e, "--", label="Esperado analítico (Guan et al. 2015)")
    ax1.set_ylabel("Cuentas por bin")
    ax1.set_title("Espectro de energía muestreado vs. parametrización analítica")
    ax1.legend()
    ax2.semilogx(e_centers, 100 * res_e, ".-")
    ax2.axhspan(-10, 10, alpha=0.15, color="green")
    ax2.set_xlabel("Energía del muón [GeV]")
    ax2.set_ylabel("Residuo [%]")
    watermark_surrogate(fig)
    fig.savefig(FIG_DIR / "stage06_energy_spectrum.png")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    theta_deg = np.rad2deg(np.arccos(c_centers))
    ax1.plot(theta_deg, obs_c, drawstyle="steps-mid", label="Muestreado")
    ax1.plot(theta_deg, exp_c, "--", label="Esperado analítico")
    ax1.set_ylabel("Cuentas por bin")
    ax1.set_title(f"Distribución cenital (ajuste cosⁿθ: n = {n_exp:.2f}; PDG ≈ 2)")
    ax1.legend()
    ax2.plot(theta_deg, 100 * res_c, ".-")
    ax2.axhspan(-10, 10, alpha=0.15, color="green")
    ax2.set_xlabel("Ángulo cenital θ [grados]")
    ax2.set_ylabel("Residuo [%]")
    watermark_surrogate(fig)
    fig.savefig(FIG_DIR / "stage06_zenith_distribution.png")
    plt.close(fig)

    out = OUT_DIR / "stage06_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    ok = (summary["energy_spectrum"]["acceptance_lt_10pct"]
          and summary["zenith_distribution"]["acceptance_lt_10pct"])
    print("STAGE 6 SAMPLER VALIDATION:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
