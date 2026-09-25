"""Cosmic-ray muon flux model — surrogate engine v0.9.

Physics implemented:
- Differential sea-level muon flux: Guan et al. (2015) modified Gaisser
  parametrization (arXiv:1509.06176), valid down to ~sub-GeV energies and
  large zenith angles, with the Chirkin cos(theta*) Earth-curvature
  correction.
- Muon absorption in standard rock: continuous-slowing-down approximation
  dE/dX = a + b*E with a = 2.17e-3 GeV cm^2/g, b = 4.0e-6 cm^2/g
  (PDG / Groom, Mokhov & Striganov muon tables, standard rock <Z/A>=0.5,
  rho = 2.65 g/cm^3). E_min(X) = (a/b) * (exp(b*X) - 1).

NOT implemented (documented surrogate limitations — the full-transport
Geant4 engine replaces this module for the definitive dataset):
multiple scattering, energy-loss fluctuations, secondary backgrounds,
detector response beyond geometric acceptance.

Units: energy GeV, opacity g/cm^2, angles rad, flux cm^-2 s^-1 sr^-1 GeV^-1.
"""

from __future__ import annotations

import numpy as np

# CSDA parameters, standard rock (PDG/Groom) — cited in stage-06 report
A_LOSS_GEV_CM2_G = 2.17e-3
B_LOSS_CM2_G = 4.0e-6

MUON_MASS_GEV = 0.1056583745

# Integration grid for integral fluxes (log-spaced, GeV)
_E_GRID = np.logspace(np.log10(0.10), np.log10(2.0e5), 4000)


def cos_theta_star(cos_theta: np.ndarray) -> np.ndarray:
    """Chirkin correction for Earth curvature (as used by Guan et al. 2015)."""
    p1, p2, p3, p4, p5 = 0.102573, -0.068287, 0.958633, 0.0407253, 0.817285
    c = np.asarray(cos_theta, dtype=np.float64)
    num = c**2 + p1**2 + p2 * c**p3 + p4 * c**p5
    den = 1.0 + p1**2 + p2 + p4
    return np.sqrt(num / den)


def diff_flux(energy_gev: np.ndarray, cos_theta: np.ndarray) -> np.ndarray:
    """Guan et al. (2015) modified Gaisser differential flux.

    Phi(E, theta) [cm^-2 s^-1 sr^-1 GeV^-1], E = muon energy at sea level.
    Broadcasting: energy and cos_theta must be broadcastable.
    """
    e = np.asarray(energy_gev, dtype=np.float64)
    cs = cos_theta_star(cos_theta)
    e_eff = e * (1.0 + 3.64 / (e * cs**1.29))
    term = 1.0 / (1.0 + 1.1 * e * cs / 115.0) + 0.054 / (1.0 + 1.1 * e * cs / 850.0)
    return 0.14 * e_eff**-2.7 * term


def e_min_from_opacity(opacity_g_cm2: np.ndarray) -> np.ndarray:
    """Minimum sea-level energy to cross opacity X (CSDA, standard rock)."""
    x = np.asarray(opacity_g_cm2, dtype=np.float64)
    return (A_LOSS_GEV_CM2_G / B_LOSS_CM2_G) * np.expm1(B_LOSS_CM2_G * x)


def opacity_from_e(energy_gev: np.ndarray) -> np.ndarray:
    """Inverse of e_min_from_opacity: CSDA range in g/cm^2 for energy E."""
    e = np.asarray(energy_gev, dtype=np.float64)
    return np.log1p(B_LOSS_CM2_G * e / A_LOSS_GEV_CM2_G) / B_LOSS_CM2_G


def integral_flux(e_min_gev: np.ndarray, cos_theta: np.ndarray) -> np.ndarray:
    """Integral flux above E_min at zenith theta [cm^-2 s^-1 sr^-1].

    Vectorized trapezoidal integration on the shared log grid. e_min and
    cos_theta must be broadcastable to a common shape.
    """
    e_min = np.atleast_1d(np.asarray(e_min_gev, dtype=np.float64))
    cth = np.atleast_1d(np.asarray(cos_theta, dtype=np.float64))
    e_min_b, cth_b = np.broadcast_arrays(e_min, cth)
    shape = e_min_b.shape

    e = _E_GRID[np.newaxis, :]                                # (1, N)
    phi = diff_flux(e, cth_b.reshape(-1, 1))                  # (M, N)
    mask = e >= e_min_b.reshape(-1, 1)
    integrand = np.where(mask, phi, 0.0)
    out = np.trapezoid(integrand, _E_GRID, axis=1)
    return out.reshape(shape)


def open_sky_vertical_integral_flux() -> float:
    """Integral vertical flux above 0.5 GeV — PDG anchor (~70 m^-2 s^-1 sr^-1)."""
    return float(integral_flux(0.5, 1.0).ravel()[0])


def sample_muons(n: int, rng: np.random.Generator,
                 e_range=(0.5, 1.0e4), cos_theta_range=(0.0, 1.0)):
    """Sample (E, theta) from the Guan distribution weighted by solid angle.

    Target density over (E, cos_theta): Phi(E, theta) (per unit E, per unit
    cos_theta after the sr -> d(cos)d(phi) reduction; the azimuthal factor is
    uniform and irrelevant to marginals). Method: inverse-CDF on a dense 2D
    grid (exact to grid resolution, which is far finer than analysis bins).

    Returns (energy_gev, theta_rad) arrays of length n.
    """
    n_e, n_c = 1200, 600
    e_grid = np.logspace(np.log10(e_range[0]), np.log10(e_range[1]), n_e)
    c_grid = np.linspace(cos_theta_range[0] + 1e-6, cos_theta_range[1], n_c)

    pdf = diff_flux(e_grid[np.newaxis, :], c_grid[:, np.newaxis])  # (n_c, n_e)

    # Cell-integrated probabilities (trapezoid weights on both axes)
    de = np.gradient(e_grid)
    dc = np.gradient(c_grid)
    cell = pdf * de[np.newaxis, :] * dc[:, np.newaxis]
    flat = cell.ravel()
    cdf = np.cumsum(flat)
    cdf /= cdf[-1]

    idx = np.searchsorted(cdf, rng.random(n))
    ic, ie = np.unravel_index(idx, cell.shape)

    # Jitter uniformly inside each cell to avoid grid artifacts
    e_out = e_grid[ie] * np.exp(rng.uniform(-0.5, 0.5, n) * np.gradient(np.log(e_grid))[ie])
    c_out = c_grid[ic] + rng.uniform(-0.5, 0.5, n) * dc[ic]
    c_out = np.clip(c_out, 1e-6, 1.0)
    return e_out, np.arccos(c_out)
