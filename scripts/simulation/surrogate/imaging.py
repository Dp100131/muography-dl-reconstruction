"""Muographic image formation — surrogate engine v0.9.

Pipeline per (geometry, detector view):
  1. Ray-cast the density field on an angular grid (azimuth x elevation)
     from the detector position -> opacity map X [g/cm^2].
  2. Open-sky baseline: same path length through air only -> X_air.
  3. Transmission flux per bin: Phi_int(E_min(X), zenith) with E_min from
     the CSDA range in standard rock (flux_model).
  4. Expected counts: lambda = Phi_int * A_det * dOmega * T_exposure.
  5. Counts: Poisson(lambda). Subsampling fraction f: Poisson(lambda*f)
     (exact Poisson-thinning equivalence).

Detector model (documented simplifications): flat 1.44 m^2 (MuTe-like)
acceptance without angular efficiency; no instrumental background; no
momentum threshold beyond the flux integration cutoff.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from . import flux_model as fm
from .geometry import VolcanoGeometry, AIR_DENSITY

ENGINE_VERSION = "surrogate-v0.9"

# Angular grid (provisional Phase B parameters v0.9)
N_AZ, N_EL = 256, 256
AZ_HALF_DEG = 30.0
EL_MIN_DEG, EL_MAX_DEG = 2.0, 47.0
DETECTOR_AREA_CM2 = 1.44e4          # 120 x 120 cm (MuTe-like)
RAY_STEP_M = 2.0
RAY_MAX_M = 3000.0


@dataclass
class DetectorView:
    geometry_id: str
    view_index: int          # 0..5 -> azimuth around the cone axis
    distance_m: float = 1000.0
    height_m: float = 1.5

    @property
    def azimuth_deg(self) -> float:
        return self.view_index * 60.0

    def position(self) -> np.ndarray:
        a = np.deg2rad(self.azimuth_deg)
        return np.array([self.distance_m * np.cos(a),
                         self.distance_m * np.sin(a),
                         self.height_m])

    def look_azimuth_rad(self) -> float:
        """Detector looks toward the cone axis."""
        return np.deg2rad(self.azimuth_deg + 180.0)


def angular_grid():
    az = np.deg2rad(np.linspace(-AZ_HALF_DEG, AZ_HALF_DEG, N_AZ))
    el = np.deg2rad(np.linspace(EL_MIN_DEG, EL_MAX_DEG, N_EL))
    return az, el


def ray_directions(view: DetectorView):
    """Unit direction per bin, shape (N_EL, N_AZ, 3). Rows = elevation."""
    az, el = angular_grid()
    az_world = view.look_azimuth_rad() + az[np.newaxis, :]
    el_g = el[:, np.newaxis]
    cos_el = np.cos(el_g)
    dirs = np.empty((N_EL, N_AZ, 3))
    dirs[..., 0] = cos_el * np.cos(az_world)
    dirs[..., 1] = cos_el * np.sin(az_world)
    dirs[..., 2] = np.sin(el_g)
    return dirs


def opacity_map(geo: VolcanoGeometry, view: DetectorView,
                chunk_rows: int = 32) -> np.ndarray:
    """Line-integral opacity X [g/cm^2] per angular bin (N_EL, N_AZ).

    Optimization: rock can only exist inside the edifice bounding cylinder,
    so the dense ray-march is bounded to [D - 1.25R, D + 1.25R]; the rest of
    the path up to RAY_MAX_M is pure air and added analytically. Total path
    length is therefore identical for every ray (consistent with the
    open-sky baseline).
    """
    origin = view.position()
    dirs = ray_directions(view)

    t_lo = max(0.0, view.distance_m - 1.25 * geo.base_radius_m)
    t_hi = min(RAY_MAX_M, view.distance_m + 1.25 * geo.base_radius_m)
    t = np.arange(t_lo + 0.5 * RAY_STEP_M, t_hi, RAY_STEP_M, dtype=np.float32)
    air_outside_cm = (RAY_MAX_M - (t_hi - t_lo)) * 100.0

    x_map = np.empty((N_EL, N_AZ))
    step_cm = RAY_STEP_M * 100.0
    for r0 in range(0, N_EL, chunk_rows):
        r1 = min(r0 + chunk_rows, N_EL)
        d = dirs[r0:r1].astype(np.float32)                # (rows, N_AZ, 3)
        pts = origin.astype(np.float32)[np.newaxis, np.newaxis, np.newaxis, :] \
            + d[:, :, np.newaxis, :] * t[np.newaxis, np.newaxis, :, np.newaxis]
        rho = geo.density(pts[..., 0], pts[..., 1], pts[..., 2])
        x_map[r0:r1] = rho.sum(axis=2) * step_cm
    return x_map + air_outside_cm * AIR_DENSITY


def air_baseline_opacity() -> np.ndarray:
    """Open-sky opacity for the same total path length through air only."""
    x_air = RAY_MAX_M * 100.0 * AIR_DENSITY
    return np.full((N_EL, N_AZ), x_air)


class FluxTable:
    """Bilinear interpolation table of log10(Phi_int) over (log10 E_min, cos zenith).

    Avoids re-integrating the spectrum for every angular bin. Built once per
    process; accuracy validated in Stage 9 against direct integration.
    """

    def __init__(self):
        self.log_e = np.linspace(np.log10(0.1), np.log10(9.0e4), 400)
        self.cos_z = np.linspace(0.005, 1.0, 160)
        phi = fm.integral_flux(10.0 ** self.log_e[np.newaxis, :],
                               self.cos_z[:, np.newaxis])
        self._interp = RegularGridInterpolator(
            (self.cos_z, self.log_e), np.log10(np.maximum(phi, 1e-300)),
            bounds_error=False, fill_value=None)

    def __call__(self, e_min: np.ndarray, cos_z: np.ndarray) -> np.ndarray:
        pts = np.stack([np.clip(cos_z, self.cos_z[0], 1.0),
                        np.clip(np.log10(np.maximum(e_min, 0.1)),
                                self.log_e[0], self.log_e[-1])], axis=-1)
        return 10.0 ** self._interp(pts)


_FLUX_TABLE: FluxTable | None = None


def flux_table() -> FluxTable:
    global _FLUX_TABLE
    if _FLUX_TABLE is None:
        _FLUX_TABLE = FluxTable()
    return _FLUX_TABLE


def expected_counts(geo: VolcanoGeometry, view: DetectorView,
                    exposure_days: float):
    """Return (lambda_map, transmission_map, opacity_map) for a view."""
    x_rock = opacity_map(geo, view)
    x_air = air_baseline_opacity()

    _, el = angular_grid()
    cos_zenith = np.sin(el)[:, np.newaxis] * np.ones((1, N_AZ))  # zenith = 90deg - elevation

    ft = flux_table()
    phi_obj = ft(fm.e_min_from_opacity(x_rock), cos_zenith)
    phi_sky = ft(fm.e_min_from_opacity(x_air), cos_zenith)
    transmission = phi_obj / phi_sky

    az, el_ = angular_grid()
    d_az = az[1] - az[0]
    d_el = el_[1] - el_[0]
    d_omega = d_az * d_el * np.cos(el_)[:, np.newaxis]  # sr per bin

    t_sec = exposure_days * 86400.0
    lam = phi_obj * DETECTOR_AREA_CM2 * d_omega * t_sec
    return lam, transmission, x_rock


def git_commit_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def view_metadata(geo: VolcanoGeometry, view: DetectorView,
                  exposure_days: float, seed: int) -> dict:
    return {
        "engine": ENGINE_VERSION,
        "git_commit": git_commit_hash(),
        "geometry_id": geo.geometry_id,
        "view": asdict(view),
        "angular_grid": {
            "n_az": N_AZ, "n_el": N_EL,
            "az_half_deg": AZ_HALF_DEG,
            "el_range_deg": [EL_MIN_DEG, EL_MAX_DEG],
        },
        "detector_area_cm2": DETECTOR_AREA_CM2,
        "ray_step_m": RAY_STEP_M,
        "exposure_days": exposure_days,
        "poisson_seed": seed,
        "physics": {
            "flux": "Guan et al. 2015 + Chirkin",
            "absorption": "CSDA standard rock a=2.17e-3 GeV cm2/g, b=4.0e-6 cm2/g",
            "not_included": ["multiple scattering", "energy-loss fluctuations",
                             "instrumental background", "detector efficiency"],
        },
    }


def save_view(out_dir: Path, geo: VolcanoGeometry, view: DetectorView,
              exposure_days: float, seed: int):
    """Compute and persist one view: lambda, transmission, reference counts."""
    lam, trans, x_rock = expected_counts(geo, view, exposure_days)
    rng = np.random.default_rng(seed)
    counts_ref = rng.poisson(lam).astype(np.uint32)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{geo.geometry_id}_view{view.view_index}"
    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, lam=lam.astype(np.float32),
                        transmission=trans.astype(np.float32),
                        opacity=x_rock.astype(np.float32),
                        counts_ref=counts_ref)
    meta = view_metadata(geo, view, exposure_days, seed)
    meta["total_expected_counts"] = float(lam.sum())
    meta["total_reference_counts"] = int(counts_ref.sum())
    (out_dir / f"{stem}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return npz_path, meta
