"""Parametric volcano density field — surrogate engine v0.9.

A geometry JSON defines a conical edifice on flat terrain plus internal
anomalies. The SAME JSON files drive the future Geant4 implementation
(Stage 7 acceptance: instantiate every geometry from a config file), so the
catalog is engine-agnostic.

Coordinates: meters. Origin at the cone axis on the ground plane (z=0).
Density: g/cm^3. Everything above the terrain and outside the edifice is air
(density ~0 for opacity purposes: 400 m of air ~ 0.05 g/cm^2 * ... actually
air contributes ~1e-3 of rock per km — included for correctness).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

AIR_DENSITY = 1.225e-3  # g/cm^3 at sea level (constant approximation)


@dataclass
class Anomaly:
    kind: str          # "conduit" | "cavity" | "dense_body"
    density: float     # g/cm^3
    params: dict = field(default_factory=dict)


@dataclass
class VolcanoGeometry:
    geometry_id: str
    base_radius_m: float
    height_m: float
    rock_density: float
    anomalies: list[Anomaly]
    description: str = ""

    @classmethod
    def from_json(cls, path: str | Path) -> "VolcanoGeometry":
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            geometry_id=cfg["geometry_id"],
            base_radius_m=cfg["edifice"]["base_radius_m"],
            height_m=cfg["edifice"]["height_m"],
            rock_density=cfg["edifice"]["rock_density_g_cm3"],
            anomalies=[Anomaly(a["kind"], a["density_g_cm3"], a["params"])
                       for a in cfg.get("anomalies", [])],
            description=cfg.get("description", ""),
        )

    # ------------------------------------------------------------------ field
    def density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Vectorized density [g/cm^3] at points (x, y, z) in meters."""
        r = np.sqrt(x**2 + y**2)
        # Cone: surface z = H * (1 - r/R) for r <= R
        cone_surface = self.height_m * np.clip(1.0 - r / self.base_radius_m, 0.0, None)
        inside = (z >= 0.0) & (z <= cone_surface)

        rho = np.where(inside, self.rock_density, AIR_DENSITY).astype(np.float64)

        for a in self.anomalies:
            mask = self._anomaly_mask(a, x, y, z) & inside
            rho = np.where(mask, a.density, rho)
        return rho

    @staticmethod
    def _anomaly_mask(a: Anomaly, x, y, z) -> np.ndarray:
        p = a.params
        if a.kind == "conduit":
            # Vertical cylinder from z0 to z1 centered at (cx, cy)
            return ((x - p["cx_m"])**2 + (y - p["cy_m"])**2 <= p["radius_m"]**2) \
                & (z >= p["z0_m"]) & (z <= p["z1_m"])
        if a.kind == "cavity":
            # Sphere at (cx, cy, cz)
            return ((x - p["cx_m"])**2 + (y - p["cy_m"])**2
                    + (z - p["cz_m"])**2) <= p["radius_m"]**2
        if a.kind == "dense_body":
            # Axis-aligned ellipsoid at (cx, cy, cz) with semi-axes (ax, ay, az)
            return (((x - p["cx_m"]) / p["ax_m"])**2
                    + ((y - p["cy_m"]) / p["ay_m"])**2
                    + ((z - p["cz_m"]) / p["az_m"])**2) <= 1.0
        raise ValueError(f"unknown anomaly kind: {a.kind}")

    # -------------------------------------------------------------- utilities
    def summary_row(self) -> dict:
        kinds = ",".join(sorted({a.kind for a in self.anomalies})) or "none"
        return {
            "geometry_id": self.geometry_id,
            "base_radius_m": self.base_radius_m,
            "height_m": self.height_m,
            "rock_density": self.rock_density,
            "n_anomalies": len(self.anomalies),
            "anomaly_kinds": kinds,
            "description": self.description,
        }
