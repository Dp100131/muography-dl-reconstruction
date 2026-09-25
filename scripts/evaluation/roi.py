"""Stage 19 — geometry-derived regions of interest (protocol v1.1 section 5.2).

Every detectability number of Stage 21 is measured inside a mask that comes
from the SAME ray-caster that produced the images, never from a threshold on
the images themselves. That is the whole point: a mask derived from the data
would move with the noise and would make "the anomaly is preserved" circular.

The angular grid and the ray directions are IMPORTED from
scripts/simulation/surrogate/imaging.py -- they are never re-implemented here.
The march mirrors imaging.opacity_map exactly: same origin, same directions, same
t-window [max(0, D - 1.25 R), min(RAY_MAX_M, D + 1.25 R)], same 2.0 m step,
same chunking. Only the accumulator changes: instead of the density field it
integrates the indicator of one anomaly (intersected with the edifice, exactly
as VolcanoGeometry.density does before overriding the rock density).

Definitions (protocol v1.1 section 5.2, PRE-REGISTERED):

    L_a(p)     [m]  path length of ray p inside anomaly a
    L_rock(p)  [m]  path length of ray p inside the edifice cone
    sky(p)          L_rock(p) == 0
    ROI(a)          {p : L_a(p) >= max(20 m, 0.5 max L_a)}   (FWHM rule)
    annulus         Chebyshev distance 4..12 px from the ROI, minus sky, minus
                    the dilated silhouette edge, minus other dilated ROIs,
                    minus pixels whose rock column differs by more than 50 %
                    from the ROI median; widened to 16 px if it holds < 200 px
    DX = drho L 100 [g/cm^2]  physical opacity perturbation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "simulation"))

from surrogate.geometry import AIR_DENSITY, VolcanoGeometry  # noqa: E402
from surrogate.imaging import (N_AZ, N_EL, RAY_MAX_M,  # noqa: E402
                               RAY_STEP_M, DetectorView, angular_grid,
                               ray_directions)

GEOMETRY_DIR = REPO_ROOT / "scripts" / "simulation" / "geometry_catalog" / "geometries"

# --- ROI / annulus constants (protocol v1.1 section 5.2, PRE-REGISTERED) ----
ROI_MIN_PATH_M = 20.0        # 10 ray steps: below this the mask is sampling noise
ROI_FWHM_FRACTION = 0.5      # half of the maximum path length through the anomaly
MIN_ROI_PIXELS = 64          # one 7x7 SSIM window plus margin
ANNULUS_INNER_PX = 4
ANNULUS_OUTER_PX = 12
ANNULUS_OUTER_WIDE_PX = 16   # fallback when the ring holds < MIN_ANNULUS_PIXELS
MIN_ANNULUS_PIXELS = 200
EXCLUSION_DILATION_PX = 3    # silhouette edge and neighbouring ROIs
ROCK_COLUMN_TOLERANCE = 0.5  # |L_rock - median_ROI(L_rock)| <= 50 % of the median

# --- sham ROI constants (calibrates z_thr at a 1 % false positive rate) ----
SHAM_PER_VIEW = 10
SHAM_MIN_CLEARANCE_PX = 12   # from every real ROI and from the silhouette edge
SHAM_FALLBACK_RADIUS_PX = 6  # 12 x 12 disk when the view has no real ROI
SHAM_MAX_ATTEMPTS = 4000


# ------------------------------------------------------------------ loading
def load_geometry(geometry_id: str) -> VolcanoGeometry:
    """Catalog geometry by id.

    scripts/simulation/surrogate/geometry.py exposes the loader as the classmethod
    VolcanoGeometry.from_json(path); this wrapper only resolves the catalog
    path so callers do not hard-code it. The parsing itself is not duplicated.
    """
    path = GEOMETRY_DIR / f"{geometry_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"unknown geometry {geometry_id!r}: {path}")
    return VolcanoGeometry.from_json(path)


# ------------------------------------------------------------------ marching
def _inside_edifice(geo: VolcanoGeometry, x, y, z) -> np.ndarray:
    """Cone predicate of VolcanoGeometry.density, without anomaly overrides.

    Mirrors surrogate/geometry.py: the cone surface is z = H (1 - r/R) for
    r <= R and the interior is 0 <= z <= surface. Kept as a separate function
    because density() folds the anomaly overrides into the same pass and the
    decomposition check (V4) needs the two terms apart. test_roi_cluster.py
    asserts agreement with geo.density on a random point cloud.
    """
    r = np.sqrt(x ** 2 + y ** 2)
    cone_surface = geo.height_m * np.clip(1.0 - r / geo.base_radius_m, 0.0, None)
    return (z >= 0.0) & (z <= cone_surface)


def _march_window(geo: VolcanoGeometry, view: DetectorView):
    """(t samples, air path outside the window [cm]) -- imaging.opacity_map."""
    t_lo = max(0.0, view.distance_m - 1.25 * geo.base_radius_m)
    t_hi = min(RAY_MAX_M, view.distance_m + 1.25 * geo.base_radius_m)
    t = np.arange(t_lo + 0.5 * RAY_STEP_M, t_hi, RAY_STEP_M, dtype=np.float32)
    air_outside_cm = (RAY_MAX_M - (t_hi - t_lo)) * 100.0
    return t, air_outside_cm


def _cast(geo: VolcanoGeometry, view: DetectorView, chunk_rows: int = 32):
    """Yield (row_slice, inside_edifice, x, y, z) chunks of the mirrored march."""
    origin = view.position().astype(np.float32)
    dirs = ray_directions(view)
    t, _ = _march_window(geo, view)
    for r0 in range(0, N_EL, chunk_rows):
        r1 = min(r0 + chunk_rows, N_EL)
        d = dirs[r0:r1].astype(np.float32)                    # (rows, N_AZ, 3)
        pts = (origin[np.newaxis, np.newaxis, np.newaxis, :]
               + d[:, :, np.newaxis, :] * t[np.newaxis, np.newaxis, :, np.newaxis])
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        yield slice(r0, r1), _inside_edifice(geo, x, y, z), x, y, z


def cast_path_lengths(geo: VolcanoGeometry, view: DetectorView,
                      chunk_rows: int = 32):
    """(L_rock, {index: L_a}) in a SINGLE march -- the expensive entry point.

    Stage 20 pays the ray-casting cost once per (geometry, view); splitting it
    into two passes would double the wall time the execution plan says is
    dominated by exactly this loop.
    """
    l_rock = np.zeros((N_EL, N_AZ), dtype=np.float32)
    l_anomalies = {i: np.zeros((N_EL, N_AZ), dtype=np.float32)
                   for i in range(len(geo.anomalies))}
    for rows, inside, x, y, z in _cast(geo, view, chunk_rows):
        l_rock[rows] = inside.sum(axis=2, dtype=np.float32) * RAY_STEP_M
        for i, anomaly in enumerate(geo.anomalies):
            mask = VolcanoGeometry._anomaly_mask(anomaly, x, y, z) & inside
            l_anomalies[i][rows] = mask.sum(axis=2, dtype=np.float32) * RAY_STEP_M
    return l_rock, l_anomalies


def anomaly_path_lengths(geo: VolcanoGeometry, view: DetectorView,
                         chunk_rows: int = 32) -> dict:
    """{anomaly_index: float32 (N_EL, N_AZ)} path length [m] inside each anomaly.

    The anomaly indicator is intersected with the edifice exactly as
    VolcanoGeometry.density does, so a lobe of an anomaly that sticks out of
    the cone contributes nothing -- as it contributes nothing to the opacity.
    """
    return cast_path_lengths(geo, view, chunk_rows)[1]


def rock_path_length(geo: VolcanoGeometry, view: DetectorView,
                     chunk_rows: int = 32) -> np.ndarray:
    """Path length [m] inside the edifice cone, anomalies included as rock."""
    return cast_path_lengths(geo, view, chunk_rows)[0]


def sky_mask(l_rock: np.ndarray) -> np.ndarray:
    """Pixels whose ray never enters the edifice."""
    return l_rock == 0.0


def sky_opacity(geo: VolcanoGeometry, view: DetectorView) -> float:
    """Opacity [g/cm^2] of an open-sky pixel under imaging.opacity_map.

    imaging.opacity_map sums n_t samples of AIR_DENSITY over the marched
    window and then adds (RAY_MAX_M - (t_hi - t_lo)) * 100 cm of air
    analytically. Because n_t * RAY_STEP_M is not exactly (t_hi - t_lo), the
    floor is weakly geometry dependent -- it is NOT the round
    RAY_MAX_M * 100 * AIR_DENSITY = 367.5 g/cm^2. Reproduced here bit for bit
    so V4 can assert the sky floor instead of assuming a literal.
    """
    t, air_outside_cm = _march_window(geo, view)
    return float(AIR_DENSITY * (t.size * RAY_STEP_M * 100.0 + air_outside_cm))


def predicted_opacity(geo: VolcanoGeometry, view: DetectorView,
                      l_rock: np.ndarray, l_anomalies: dict) -> np.ndarray:
    """X [g/cm^2] rebuilt from the path-length decomposition (check V4).

        X = 100 [ rho_rock L_rock + sum_a (rho_a - rho_rock) L_a
                  + rho_air (L_window - L_rock) ] + rho_air air_outside

    Assumes the anomalies of a geometry do not overlap (true for the whole
    catalog: the closest pair is 100 m apart with radii 26 + 41 m). Where they
    did overlap, density() would apply the LAST anomaly only and this sum
    would over-count -- which is precisely what V4 would catch.
    """
    t, air_outside_cm = _march_window(geo, view)
    window_m = t.size * RAY_STEP_M
    x = geo.rock_density * l_rock.astype(np.float64)
    for i, anomaly in enumerate(geo.anomalies):
        x = x + (anomaly.density - geo.rock_density) * l_anomalies[i].astype(np.float64)
    x = x + AIR_DENSITY * (window_m - l_rock.astype(np.float64))
    return x * 100.0 + AIR_DENSITY * air_outside_cm


def reproduced_opacity(geo: VolcanoGeometry, view: DetectorView,
                       chunk_rows: int = 32) -> np.ndarray:
    """X [g/cm^2] rebuilt by re-marching and evaluating geo.density() directly.

    Unlike predicted_opacity (a linear path-length decomposition that assumes
    non-overlapping anomalies), this reproduces imaging.opacity_map bit for
    bit: at every sample point density() applies the SAME sequential-override
    semantics as the stored surrogate opacity (the last anomaly wins where two
    overlap, e.g. the conduit and cavity of geo-test-01), so V4 holds to
    floating point regardless of overlap. This validates that roi.py's grid,
    ray directions and marched window reproduce the physics; the per-anomaly
    path lengths (validated by V2/V3 and the overlays) are what Stage 21/22
    consume, and they are correct per anomaly whether or not anomalies overlap.
    """
    step_cm = RAY_STEP_M * 100.0
    t, air_outside_cm = _march_window(geo, view)
    x = np.zeros((N_EL, N_AZ), dtype=np.float64)
    for rows, _inside, px, py, pz in _cast(geo, view, chunk_rows):
        rho = geo.density(px, py, pz)                      # (rows, N_AZ, n_t)
        x[rows] = rho.sum(axis=2, dtype=np.float64) * step_cm
    return x + AIR_DENSITY * air_outside_cm


def delta_opacity(l_a: np.ndarray, delta_rho: float) -> np.ndarray:
    """Physical perturbation DX = drho * L * 100 [g/cm^2] (L in m, drho g/cm^3)."""
    return np.asarray(l_a, dtype=np.float64) * float(delta_rho) * 100.0


def anomaly_delta_rho(geo: VolcanoGeometry, index: int) -> float:
    """rho_anomaly - rho_rock: the signed density contrast of one instance."""
    return float(geo.anomalies[index].density - geo.rock_density)


# ------------------------------------------------------- morphology (square SE)
def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a (2r+1) square structuring element = Chebyshev ball.

    Separable pure-NumPy implementation (rows then columns): 2(2r+1) shifted
    OR passes instead of the (2r+1)^2 of the naive form, exact for a square SE
    and free of any SciPy version behaviour. Borders replicate, so the image
    frame never acts as an object boundary.
    """
    if radius <= 0:
        return np.array(mask, dtype=bool)
    out = np.array(mask, dtype=bool)
    for axis in (0, 1):
        padded = np.pad(out, [(radius, radius) if a == axis else (0, 0)
                              for a in (0, 1)], mode="edge")
        acc = np.zeros_like(out)
        for shift in range(2 * radius + 1):
            if axis == 0:
                acc |= padded[shift:shift + out.shape[0], :]
            else:
                acc |= padded[:, shift:shift + out.shape[1]]
        out = acc
    return out


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary erosion with the same square structuring element (De Morgan)."""
    return ~_dilate(~np.asarray(mask, dtype=bool), radius)


def silhouette_edge(sky: np.ndarray) -> np.ndarray:
    """Sky-side boundary of the silhouette: sky pixels that touch rock."""
    sky = np.asarray(sky, dtype=bool)
    return sky & ~_erode(sky, 1)


# ------------------------------------------------------------------ ROI build
def roi_threshold(l_a: np.ndarray) -> float:
    """max(20 m, 0.5 max L_a) -- the value actually used, exposed for the log."""
    return float(max(ROI_MIN_PATH_M, ROI_FWHM_FRACTION * float(np.max(l_a))))


def build_roi(l_a: np.ndarray):
    """(mask, threshold) for one anomaly under the pre-registered FWHM rule."""
    threshold = roi_threshold(l_a)
    return np.asarray(l_a) >= threshold, threshold


def roi_pixel_count(roi: np.ndarray) -> int:
    return int(np.count_nonzero(roi))


def check_roi_size(roi: np.ndarray, minimum: int = MIN_ROI_PIXELS):
    """(mask, valid) -- an ROI below the floor is EXCLUDED and REPORTED.

    Protocol v1.1 section 5.2: never dropped in silence. The caller writes the
    flag into the tidy table so the exclusion is visible in the final report.
    """
    return np.asarray(roi, dtype=bool), roi_pixel_count(roi) >= int(minimum)


def bounding_box(mask: np.ndarray):
    """(row0, row1, col0, col1) half-open bounding box, or None if empty."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1] + 1), int(cols[0]), int(cols[-1] + 1)


def build_annulus(roi: np.ndarray, all_rois, sky: np.ndarray,
                  edge: np.ndarray, l_rock: np.ndarray):
    """(mask, info) background ring around one ROI (protocol v1.1 section 5.2).

    all_rois may include `roi` itself; it is identified by object identity or
    by equality and is not used as an exclusion against itself.
    """
    roi = np.asarray(roi, dtype=bool)
    sky = np.asarray(sky, dtype=bool)
    edge = np.asarray(edge, dtype=bool)
    l_rock = np.asarray(l_rock, dtype=np.float64)

    forbidden = sky | _dilate(edge, EXCLUSION_DILATION_PX)
    for other in all_rois:
        other = np.asarray(other, dtype=bool)
        if other.shape == roi.shape and np.array_equal(other, roi):
            continue
        forbidden |= _dilate(other, EXCLUSION_DILATION_PX)

    column_ok = np.ones_like(roi)
    median_column = float("nan")
    if roi.any():
        median_column = float(np.median(l_rock[roi]))
        if median_column > 0.0:
            column_ok = (np.abs(l_rock - median_column)
                         <= ROCK_COLUMN_TOLERANCE * median_column)

    inner = _dilate(roi, ANNULUS_INNER_PX - 1)   # Chebyshev distance <= 3
    for outer_px in (ANNULUS_OUTER_PX, ANNULUS_OUTER_WIDE_PX):
        ring = _dilate(roi, outer_px) & ~inner
        annulus = ring & ~forbidden & column_ok
        if np.count_nonzero(annulus) >= MIN_ANNULUS_PIXELS:
            break
    n_pixels = roi_pixel_count(annulus)
    info = {
        "n_pixels": n_pixels,
        "outer_px": int(outer_px),
        "widened": bool(outer_px == ANNULUS_OUTER_WIDE_PX),
        "below_minimum": bool(n_pixels < MIN_ANNULUS_PIXELS),
        "median_rock_column_m": median_column,
    }
    return annulus, info


# ----------------------------------------------------------------- sham ROIs
def _ellipse_mask(shape, row: int, col: int, semi_row: float, semi_col: float):
    rr, cc = np.ogrid[0:shape[0], 0:shape[1]]
    return (((rr - row) / max(semi_row, 0.5)) ** 2
            + ((cc - col) / max(semi_col, 0.5)) ** 2) <= 1.0


def sample_sham_rois_on_silhouette(sky, real_rois, rng,
                                   n: int = SHAM_PER_VIEW,
                                   clearance: int = SHAM_MIN_CLEARANCE_PX,
                                   min_pixels: int = MIN_ROI_PIXELS):
    """Sham ROIs on a synthetic or real silhouette -- pure-mask entry point.

    Rejection sampling: a candidate centre must sit inside the rock silhouette,
    at Chebyshev distance >= `clearance` from every real ROI and from the
    silhouette edge. The shape is an ellipse whose axes are drawn from the
    empirical (height, width) of this view's real ROI bounding boxes, so a
    sham has the same size and aspect statistics as a real detection and the
    calibrated z_thr is not an artefact of a different mask geometry.

    THE CLEARANCE IS CHECKED ON THE WHOLE CLIPPED MASK, not only on the centre.
    A centre 12 px clear of the silhouette edge says nothing about an ellipse
    with a 40 px semi-axis: the mask can still reach the edge or a real ROI,
    and such a sham carries real structure, inflating the 99th percentile of
    |z| and therefore the calibrated detection threshold. The whole point of
    z_thr is that a sham holds NO anomaly, so every pixel of the accepted mask
    must respect the same exclusion the centre does.
    """
    sky = np.asarray(sky, dtype=bool)
    rock = ~sky
    shape = sky.shape
    real = [np.asarray(m, dtype=bool) for m in real_rois]

    forbidden = sky.copy() | _dilate(silhouette_edge(sky), clearance)
    for mask in real:
        forbidden |= _dilate(mask, clearance)
    candidates = np.flatnonzero(~forbidden)
    if candidates.size == 0:
        return [], {"requested": int(n), "obtained": 0, "attempts": 0,
                    "reason": "no admissible centre"}

    axes = []
    for mask in real:
        box = bounding_box(mask)
        if box is not None:
            axes.append((0.5 * (box[1] - box[0]), 0.5 * (box[3] - box[2])))
    if not axes:
        axes = [(float(SHAM_FALLBACK_RADIUS_PX), float(SHAM_FALLBACK_RADIUS_PX))]

    shams, attempts = [], 0
    rejected_small, rejected_clearance = 0, 0
    while len(shams) < n and attempts < SHAM_MAX_ATTEMPTS:
        attempts += 1
        flat = int(candidates[rng.integers(candidates.size)])
        row, col = divmod(flat, shape[1])
        semi_row, semi_col = axes[int(rng.integers(len(axes)))]
        ellipse = _ellipse_mask(shape, row, col, semi_row, semi_col)
        sham = ellipse & rock
        if np.count_nonzero(sham) < min_pixels:
            rejected_small += 1
            continue
        # the clipped mask, not the centre: `forbidden` is
        # sky | dilate(silhouette edge, clearance) | dilate(union real, clearance)
        if (sham & forbidden).any():
            rejected_clearance += 1
            continue
        shams.append(sham)
    info = {"requested": int(n), "obtained": len(shams), "attempts": attempts,
            "rejected_below_min_pixels": int(rejected_small),
            "rejected_clearance": int(rejected_clearance),
            "clearance_px": int(clearance),
            "axes_pool": [[round(a, 2), round(b, 2)] for a, b in axes]}
    return shams, info


def sample_sham_rois(geo: VolcanoGeometry, view: DetectorView, real_rois, rng,
                     n: int = SHAM_PER_VIEW, l_rock: np.ndarray | None = None):
    """Sham ROIs for one (geometry, view); casts the silhouette if not given."""
    if l_rock is None:
        l_rock = rock_path_length(geo, view)
    return sample_sham_rois_on_silhouette(sky_mask(l_rock), real_rois, rng, n=n)


# --------------------------------------------------------------- convenience
def view_masks(geo: VolcanoGeometry, view: DetectorView, rng=None,
               n_sham: int = SHAM_PER_VIEW) -> dict:
    """Everything Stage 20 needs for one (geometry, view), cast exactly once.

    Returns a dict with l_rock, sky, edge, l_anomalies, rois, annuli, the
    per-ROI validity flags and (when an rng is given) the sham masks.
    """
    l_rock, l_anomalies = cast_path_lengths(geo, view)
    sky = sky_mask(l_rock)
    edge = silhouette_edge(sky)

    rois, thresholds, valid = [], [], []
    for i in range(len(geo.anomalies)):
        mask, threshold = build_roi(l_anomalies[i])
        mask, ok = check_roi_size(mask)
        rois.append(mask)
        thresholds.append(threshold)
        valid.append(bool(ok))

    annuli, annulus_info = [], []
    for mask in rois:
        annulus, info = build_annulus(mask, rois, sky, edge, l_rock)
        annuli.append(annulus)
        annulus_info.append(info)

    shams, sham_info = ([], {"requested": 0, "obtained": 0})
    if rng is not None:
        shams, sham_info = sample_sham_rois_on_silhouette(sky, rois, rng, n=n_sham)

    return {
        "l_rock": l_rock,
        "l_anomalies": l_anomalies,
        "sky": sky,
        "edge": edge,
        "rois": rois,
        "roi_thresholds": thresholds,
        "roi_valid": valid,
        "roi_pixels": [roi_pixel_count(m) for m in rois],
        "annuli": annuli,
        "annulus_info": annulus_info,
        "shams": shams,
        "sham_info": sham_info,
    }


# --------------------------------------------------------- analytic envelopes
def analytic_max_chord_m(anomaly) -> float:
    """Safe upper bound on the path length through one anomaly [m] (check V2).

    sphere      2 r
    ellipsoid   2 max(semi-axes)
    cylinder    sqrt((2r)^2 + (z1 - z0)^2)  -- the space diagonal, an upper
                bound for any ray including the near-vertical ones
    """
    p = anomaly.params
    if anomaly.kind == "cavity":
        return 2.0 * float(p["radius_m"])
    if anomaly.kind == "dense_body":
        return 2.0 * max(float(p["ax_m"]), float(p["ay_m"]), float(p["az_m"]))
    if anomaly.kind == "conduit":
        return float(np.hypot(2.0 * float(p["radius_m"]),
                              float(p["z1_m"]) - float(p["z0_m"])))
    raise ValueError(f"unknown anomaly kind: {anomaly.kind}")


def transverse_radius_m(anomaly, view=None) -> float:
    """Horizontal radius perpendicular to the line of sight (angular-width check).

    Spheres and cylinders are isotropic in the horizontal plane, so the
    transverse radius is just their radius. An axis-aligned ellipsoid is not:
    its half-width along the horizontal unit direction perpendicular to the
    view is sqrt((ax u_x)^2 + (ay u_y)^2). Using max(ax, ay) over-estimates
    the azimuth width whenever the larger semi-axis lies along the line of
    sight (measured on geo-test-02/03: -22 to -26 %), so the projection is
    taken when the view is known; max(ax, ay) remains the view-agnostic bound.
    """
    p = anomaly.params
    if anomaly.kind in ("cavity", "conduit"):
        return float(p["radius_m"])
    if anomaly.kind == "dense_body":
        ax, ay = float(p["ax_m"]), float(p["ay_m"])
        if view is None:
            return max(ax, ay)
        # horizontal transverse unit vector = perpendicular to the look azimuth
        look = view.look_azimuth_rad()
        ux, uy = -np.sin(look), np.cos(look)
        return float(np.hypot(ax * ux, ay * uy))
    raise ValueError(f"unknown anomaly kind: {anomaly.kind}")


def anomaly_centre_m(anomaly) -> np.ndarray:
    p = anomaly.params
    if anomaly.kind == "conduit":
        return np.array([p["cx_m"], p["cy_m"],
                         0.5 * (p["z0_m"] + p["z1_m"])], dtype=float)
    return np.array([p["cx_m"], p["cy_m"], p["cz_m"]], dtype=float)


def azimuth_rad_per_pixel() -> float:
    az, _ = angular_grid()
    return float(az[1] - az[0])


def elevation_rad_per_pixel() -> float:
    _, el = angular_grid()
    return float(el[1] - el[0])


def analytic_roi_azimuth_width_px(geo: VolcanoGeometry, view: DetectorView,
                                  index: int) -> float:
    """Expected ROI width in azimuth pixels at the half-maximum cut.

    The chord through a sphere or a cylinder cross-section is
    L(b) = 2 sqrt(r^2 - b^2); it falls to half its peak at b = r sqrt(3)/2, so
    the FULL width at half maximum is sqrt(3) r, NOT 2 r. Comparing the
    measured mask against 2 r would build in a 13 % bias, which is the same
    order as the +-15 % tolerance of V3, so the sqrt(3) factor is used.
    """
    anomaly = geo.anomalies[index]
    distance = float(np.linalg.norm(anomaly_centre_m(anomaly) - view.position()))
    width_rad = np.sqrt(3.0) * transverse_radius_m(anomaly, view) / distance
    return float(width_rad / azimuth_rad_per_pixel())
