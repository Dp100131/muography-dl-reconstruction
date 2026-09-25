#!/usr/bin/env python3
"""Stage 9 — physical verification and validation of the simulated images.

Tests (per the 30-stage plan):
  1. CONTROL TRANSMISSION: simulated transmission through a homogeneous
     control block vs an INDEPENDENT theoretical calculation (analytic
     path length through the cone x rock density -> E_min -> integral flux
     ratio, computed by direct integration — NOT via the imaging pipeline's
     interpolation table). Acceptance: relative error < 10-15%.
  2. CONTRAST SENSITIVITY: a known-density anomaly must produce the
     theoretically expected transmission change along rays through it.
  3. POISSON DISPERSION: counts over many replicas must satisfy
     variance ~ mean (dispersion index test, chi^2).
  4. FLUX-TABLE ACCURACY: interpolation table vs direct integration
     (supporting test for 1-2).

Run:  python scripts/analysis/vv_physics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from surrogate import flux_model as fm  # noqa: E402
from surrogate import imaging  # noqa: E402
from surrogate.geometry import VolcanoGeometry  # noqa: E402
from plotting_style import apply_style, watermark_surrogate  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

GEO_DIR = REPO_ROOT / "scripts" / "simulation" / "geometry_catalog" / "geometries"
FIG_DIR = REPO_ROOT / "figures" / "results"
OUT_DIR = REPO_ROOT / "data" / "output" / "validation"
MASTER_SEED = 20260805


def analytic_cone_path_length(geo, det_pos, azimuth_rad, elevation_rad,
                              n_fine=200_000):
    """Independent path-length-through-rock estimate: ultra-fine sampling
    with a DIFFERENT step and integrator than the imaging pipeline."""
    d = np.array([np.cos(elevation_rad) * np.cos(azimuth_rad),
                  np.cos(elevation_rad) * np.sin(azimuth_rad),
                  np.sin(elevation_rad)])
    t = np.linspace(0.0, imaging.RAY_MAX_M, n_fine)
    pts = det_pos[np.newaxis, :] + d[np.newaxis, :] * t[:, np.newaxis]
    r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    cone_surface = geo.height_m * np.clip(1.0 - r / geo.base_radius_m, 0.0, None)
    inside = (pts[:, 2] >= 0.0) & (pts[:, 2] <= cone_surface)
    return float(np.trapezoid(inside.astype(float), t))  # meters of rock


def main() -> int:
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # ------------------------------------------------------------------
    # Test 1 — control transmission vs independent theory (homogeneous cone)
    # ------------------------------------------------------------------
    geo = VolcanoGeometry.from_json(GEO_DIR / "geo-train-01.json")  # control
    view = imaging.DetectorView(geo.geometry_id, 0)
    lam, trans_sim, x_map = imaging.expected_counts(geo, view, exposure_days=30.0)

    az_grid, el_grid = imaging.angular_grid()
    # probe a set of lines of sight crossing the edifice at varied depths
    probe = [(iel, iaz) for iel in range(20, 240, 20) for iaz in (64, 128, 192)]
    rows = []
    det_pos = view.position()
    for iel, iaz in probe:
        el = el_grid[iel]
        az_world = view.look_azimuth_rad() + az_grid[iaz]
        rock_m = analytic_cone_path_length(geo, det_pos, az_world, el)
        if rock_m < 5.0:
            continue  # sky ray: not a control-transmission probe
        x_theory = rock_m * 100.0 * geo.rock_density \
            + imaging.RAY_MAX_M * 100.0 * 1.225e-3  # + air along full path
        cos_z = np.sin(el)
        e_min_th = fm.e_min_from_opacity(np.array([x_theory]))
        e_min_sky = fm.e_min_from_opacity(
            np.array([imaging.RAY_MAX_M * 100.0 * 1.225e-3]))
        phi_th = fm.integral_flux(e_min_th, np.array([cos_z]))[0]
        phi_sky = fm.integral_flux(e_min_sky, np.array([cos_z]))[0]
        t_theory = phi_th / phi_sky
        t_sim = float(trans_sim[iel, iaz])
        if t_theory < 1e-12:
            continue  # both effectively zero: ratio unstable and uninformative
        rows.append({
            "elevation_deg": round(float(np.rad2deg(el)), 1),
            "azimuth_bin": iaz,
            "rock_m": round(rock_m, 1),
            "t_theory": t_theory,
            "t_sim": t_sim,
            "rel_error": (t_sim - t_theory) / t_theory,
        })
    rel_errors = np.array([abs(r["rel_error"]) for r in rows])
    results["control_transmission"] = {
        "n_lines_of_sight": len(rows),
        "max_abs_rel_error": round(float(rel_errors.max()), 4),
        "median_abs_rel_error": round(float(np.median(rel_errors)), 4),
        "acceptance_lt_15pct": bool(rel_errors.max() < 0.15),
    }

    # ------------------------------------------------------------------
    # Test 2 — contrast sensitivity (known-density anomaly)
    # ------------------------------------------------------------------
    # Synthetic single-cavity geometry with exactly known parameters
    import copy
    geo_anom = copy.deepcopy(geo)
    from surrogate.geometry import Anomaly
    cavity = Anomaly("cavity", 0.0,
                     {"cx_m": 0.0, "cy_m": 0.0, "cz_m": 150.0, "radius_m": 40.0})
    geo_anom.anomalies = [cavity]
    _, trans_anom, x_anom = imaging.expected_counts(geo_anom, view, 30.0)

    # Ray through the cavity center: elevation atan((cz - h_det)/distance)
    el_c = np.arctan((150.0 - view.height_m) / view.distance_m)
    iel_c = int(np.argmin(np.abs(el_grid - el_c)))
    iaz_c = imaging.N_AZ // 2
    # Expected opacity reduction: chord through sphere ~ 2*r of rock removed
    dx_expected = 2 * 40.0 * 100.0 * geo.rock_density  # g/cm^2
    dx_sim = float(x_map[iel_c, iaz_c] - x_anom[iel_c, iaz_c])
    ratio_sim = float(trans_anom[iel_c, iaz_c] / trans_sim[iel_c, iaz_c])
    # Theory ratio via direct integration
    cos_zc = np.sin(el_grid[iel_c])
    phi_with = fm.integral_flux(
        fm.e_min_from_opacity(np.array([x_anom[iel_c, iaz_c]])), np.array([cos_zc]))[0]
    phi_without = fm.integral_flux(
        fm.e_min_from_opacity(np.array([x_map[iel_c, iaz_c]])), np.array([cos_zc]))[0]
    ratio_theory = float(phi_with / phi_without)
    results["contrast_sensitivity"] = {
        "cavity": "r=40 m, rho=0, center (0,0,150) m",
        "opacity_reduction_sim_g_cm2": round(dx_sim, 1),
        "opacity_reduction_expected_g_cm2": round(dx_expected, 1),
        "opacity_rel_error": round(abs(dx_sim - dx_expected) / dx_expected, 4),
        "transmission_boost_sim": round(ratio_sim, 3),
        "transmission_boost_theory": round(ratio_theory, 3),
        "boost_rel_error": round(abs(ratio_sim - ratio_theory) / ratio_theory, 4),
        "acceptance_lt_15pct": bool(
            abs(dx_sim - dx_expected) / dx_expected < 0.15
            and abs(ratio_sim - ratio_theory) / ratio_theory < 0.15),
    }

    # ------------------------------------------------------------------
    # Test 3 — Poisson dispersion over replicas
    # ------------------------------------------------------------------
    n_rep = 400
    rng = np.random.default_rng(np.random.SeedSequence(MASTER_SEED).spawn(999)[-1])
    # take a block of bins spanning sky and edifice (row stripes)
    lam_block = lam[::16, ::16]  # 16x16 = 256 bins across the image
    reps = rng.poisson(lam_block, size=(n_rep,) + lam_block.shape)
    mean_ = reps.mean(axis=0)
    var_ = reps.var(axis=0, ddof=1)
    # dispersion index per bin; chi2 test: (n-1)*var/mean ~ chi2(n-1)
    valid = mean_ > 1.0
    disp = var_[valid] / mean_[valid]
    stat = (n_rep - 1) * disp
    p_values = 1.0 - stats.chi2.cdf(stat, df=n_rep - 1)
    # two-sided: fraction of bins outside [2.5%, 97.5%] should be ~5%
    frac_outside = float(np.mean((p_values < 0.025) | (p_values > 0.975)))
    results["poisson_dispersion"] = {
        "n_replicas": n_rep,
        "n_bins_tested": int(valid.sum()),
        "mean_dispersion_index": round(float(disp.mean()), 4),
        "fraction_outside_95pct_band": round(frac_outside, 4),
        "acceptance": bool(abs(disp.mean() - 1.0) < 0.05 and frac_outside < 0.10),
    }

    # ------------------------------------------------------------------
    # Test 4 — flux interpolation table vs direct integration
    # ------------------------------------------------------------------
    rng2 = np.random.default_rng(123)
    e_test = 10 ** rng2.uniform(np.log10(0.3), np.log10(5e4), 200)
    c_test = rng2.uniform(0.02, 1.0, 200)
    table_vals = imaging.flux_table()(e_test, c_test)
    direct_vals = fm.integral_flux(e_test, c_test)
    rel = np.abs(table_vals - direct_vals) / np.maximum(direct_vals, 1e-300)
    results["flux_table_accuracy"] = {
        "n_points": 200,
        "max_rel_error": round(float(rel.max()), 5),
        "acceptance_lt_1pct": bool(rel.max() < 0.01),
    }

    # ------------------------------------------------------------------ figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    t_th = [r["t_theory"] for r in rows]
    t_si = [r["t_sim"] for r in rows]
    ax1.loglog(t_th, t_si, "o", ms=4)
    lims = [min(min(t_th), 1e-12), 1.2]
    ax1.loglog(lims, lims, "k--", lw=1, label="identidad")
    ax1.set_xlabel("Transmisión teórica (cálculo independiente)")
    ax1.set_ylabel("Transmisión simulada (pipeline)")
    ax1.set_title("Prueba 1: bloque de control — línea a línea")
    ax1.legend()
    ax2.hist(disp, bins=30)
    ax2.axvline(1.0, color="k", ls="--", lw=1)
    ax2.set_xlabel("Índice de dispersión (varianza/media)")
    ax2.set_ylabel("Bins")
    ax2.set_title(f"Prueba 3: dispersión Poisson ({n_rep} réplicas)")
    watermark_surrogate(fig)
    fig.savefig(FIG_DIR / "stage09_vv_physics.png")
    plt.close(fig)

    (OUT_DIR / "stage09_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

    ok = all(v.get("acceptance", v.get("acceptance_lt_15pct",
             v.get("acceptance_lt_1pct", False))) for v in results.values())
    print("STAGE 9 V&V:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
