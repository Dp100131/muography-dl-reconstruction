#!/usr/bin/env python3
"""OE1 pilot ROI pre-registration -- geo-train-06 view 0, variants A/B (work unit U8).

The geo-train-06 catalog geometry places a conduit (anomaly 0) and a dense
ellipsoidal body (anomaly 1) as overlapping solids. As received in Geant4
(variant A) the two anomalies are full sibling volumes with no CSG clipping
between them, so the material a track sees in the overlap depends on which
sibling the navigator visits first -- undefined by the geometry itself (see
muografia-documentada/objetivo-1/entregas/2026-09-29-geo-train-06/
diagnostico-geo-train-06.md, sections 3 and 5, and its cited Geant4
G4Exception GeomVol1002 warning). Fix B1 (variant B) makes the dense body
occupy the overlap and clips the conduit by it: anomaly_i = (A_i n cone) -
union_{j>i} A_j, which is exactly the priority rule the analytic surrogate
engine already applies (surrogate.geometry.VolcanoGeometry.density: later
JSON index wins). A static navigator probe at the overlap centroid of the
received geometry already returns anomaly_1 (the body), so B1 is also the
best available analytic proxy for what the *as-received* campaign actually
measured.

This script pre-registers the ROIs used to compare variants A and B in the
pilot (work unit U9, not run by this script) BEFORE that pilot exists, per
the locked rule that ROIs must be committed before any pilot run and that
"no difference" is never reported as equivalence.

Reuse (not re-implemented here):
  - angular grid, ray directions, flux integration, expected-count pipeline:
    scripts/simulation/surrogate/imaging.py (angular_grid, ray_directions,
    DetectorView, opacity_map, air_baseline_opacity, flux_table).
  - the marched ray-cast window and per-sample "inside the edifice" test:
    scripts/evaluation/roi.py (_cast, _march_window), whose docstring
    documents that this window/step/chunking is the SAME as
    imaging.opacity_map, and whose pixel registration against the Geant4
    binIndex convention is protocol v1.1. The joint conduit/body overlap
    decomposition below (R_int/R_cond/R_body) is new: roi.py's own
    cast_path_lengths() returns each anomaly's raw path length independently
    (it does not need the pairwise overlap for its FWHM-based ROI, which is
    a different, adaptive-threshold scheme used elsewhere in the pipeline).

Per-pixel path-length decomposition (single ray march, priority masks):
    L_int(p)          path length inside conduit AND body (the overlap)
    L_cond_only(p)     path length inside conduit, outside body
    L_body_only(p)     path length inside body, outside conduit
    L_rock_only(p)     path length inside the cone, outside both anomalies
    L_rock(p)          total chord inside the cone (= sum of the above four)

Material fields (same marched window, only the assignment in the overlap
and in the two exclusive regions changes):
    B1 field           overlap -> body density (received-geometry proxy)
    conduit-wins field overlap -> conduit density (upper bound for what
                       variant A could resolve to at any given point; NOT a
                       proposed fix, only an envelope for the expected A-B
                       gap)
    homogeneous field  no anomalies at all (rock density everywhere inside
                       the cone); reference for the sky/rock control check

Expected counts per event are obtained from the surrogate lambda map (same
flux/opacity pipeline as imaging.expected_counts) divided by the open-sky
normalisation S0 (muons/view/30 d, from the Geant4 flux tables), then scaled
by campaign_detected_total / lambda_B1_total for this view -- an explicit,
approximate correction for whatever the real Geant4 transport and detector
model removes that the surrogate flux integral does not model (efficiency,
multiple scattering, energy thresholds). This is stated as an approximation
everywhere it is used; it is not a claim that the surrogate reproduces
Geant4 exactly.

Run:
    python scripts/analysis/oe1_pilot_roi.py preregister \\
        --campaign-hist <path to geo-train-06_view0_hist.csv> \\
        --out-dir muografia-documentada/objetivo-1/entregas/2026-09-29-geo-train-06/analisis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "simulation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from surrogate.geometry import AIR_DENSITY, VolcanoGeometry  # noqa: E402
from surrogate import flux_model as fm  # noqa: E402
from surrogate.imaging import (  # noqa: E402
    DETECTOR_AREA_CM2, N_AZ, N_EL, RAY_STEP_M,
    DetectorView, air_baseline_opacity, angular_grid, flux_table,
    opacity_map,
)
import roi as roimod  # noqa: E402  (Stage 19 ray-march reuse: _cast, _march_window)
from plotting_style import apply_style, watermark_surrogate  # noqa: E402

GEOMETRY_DIR = REPO_ROOT / "scripts" / "simulation" / "geometry_catalog" / "geometries"
TABLES_META_PATH = REPO_ROOT / "scripts" / "simulation" / "geant4_app" / "tables" / "meta.json"

DEFAULT_GEOMETRY_ID = "geo-train-06"
DEFAULT_MIN_PATH_M = 20.0
DEFAULT_SEED = 20260929

# geo-train-06 Geant4 training campaign duration (context: 30 days,
# 36 174 382 events transported, 27 209 253 detected -- see
# muografia-documentada/objetivo-1/entregas/2026-09-29-geo-train-06/
# diagnostico-geo-train-06.md section 4). events_transported is, to within
# 0.003%, the open-sky normalisation S0 read from TABLES_META_PATH below, so
# S0 doubles as "number of transported events in the campaign period" for
# the per-event conversion.
CAMPAIGN_EXPOSURE_DAYS = 30.0

# R_rock control: total anomaly path (overlap + conduit-only + body-only)
# must be below this to call a pixel "anomaly-free rock" (protocol constant
# for this pre-registration, not exposed on the CLI).
ROCK_CONTROL_MAX_ANOMALY_PATH_M = 1.0

# Pilot parameters fixed by work unit U9's design (not run by this script;
# recorded here only so the pre-registered effect sizes are on the same
# basis as the pilot that will consume them).
PILOT_EVENTS_PER_RUN = 1_000_000
PILOT_N_SEEDS = 3

# Threshold (in Steiger z units, see steiger_z_dependent below) for the
# orientation registration check: the gap between the true orientation's
# correlation and a mirrored variant's must exceed this many standard errors
# (under H0: equal population correlation) to call that axis "resolved" one
# way or the other. ~3 is a conservative, standard threshold (one-sided
# p < 0.0014).
Z_THRESHOLD = 3.0


# --------------------------------------------------------- registration (U8b)
# Cross-geometry, empirical azimuth-registration check (work unit U8b). The
# geo-train-06 Steiger test above cannot resolve azimuth because that
# geometry's anomalies sit close enough to the axis, and cover so small a
# pixel fraction, that the axisymmetric cone silhouette dominates the
# correlation. The geometries below break that symmetry with an off-axis,
# strongly under-dense cavity (kind "cavity"; among the 5 geometries actually
# used below, density 0.068-0.464 g/cm^3, radius 27.9-48.0 m -- see
# scripts/simulation/geometry_catalog/geometries/):
#
#   geo-train-03  cavity only
#   geo-train-05  conduit + cavity (centres 66.7 m apart in xy, both radii
#                 sum to 57.6 m -> no 3-D overlap)
#   geo-train-07  cavity + dense_body (~130 m apart in xy vs. a 118.3 m
#                 radius sum -> no overlap)
#   geo-train-10  two cavities (78.9 m 3-D centre separation vs. a 73.6 m
#                 radius sum -> no overlap)
#   geo-val-02    cavity only
#
# geo-train-11 and geo-train-08 are EXCLUDED even though they also carry
# off-axis cavities: their anomalies' bounding spheres/cylinders are close
# enough that which one "wins" a shared point is ambiguous under the
# sequential-override rule (the same class of ambiguity work unit U8
# diagnosed for geo-train-06), so a single analytic opacity field is not an
# unambiguous proxy for what Geant4 actually transported there. geo-train-06
# itself is excluded because it is the case under test, and geo-test-*/
# geo-gen-* are excluded because only train/val geometries may be used for
# this pre-pilot check (protocol: test/generalisation geometries are held
# out of every analysis step that could feed back into a design choice).
REGISTRATION_GEOMETRIES = [
    "geo-train-03", "geo-train-05", "geo-train-07", "geo-train-10", "geo-val-02",
]
REGISTRATION_VIEWS = [0, 1, 2, 3, 4, 5]
DEFAULT_MIN_COUNTS = 20
DEFAULT_BLOCK_PX = 16
DEFAULT_N_BOOTSTRAP = 2000

# |model residual| above which a pixel counts as carrying azimuth "signal"
# (the effect the check is trying to detect), independent of min_counts.
SIGNAL_ABS_THRESHOLD_M = 0.05

# A case only "qualifies" to be checked against the CONFIRMED bar if it has
# at least this many signal pixels; below this the true and mirrored maps are
# too similar (small anomaly footprint) for the correlation gap to be
# informative either way.
REGISTRATION_MIN_SIGNAL_PIXELS = 50

# Written into the JSON verbatim, BEFORE any (geometry, view) case is
# computed -- the pre-stated criterion for work unit U8b (see task context).
# "case" below always means one evaluated (geometry, view) pair on the
# AZIMUTH axis; the ELEVATION mirror is a positive control only (elevation is
# already resolved per work unit U8) and does not enter this verdict.
REGISTRATION_CRITERION = {
    "confirmed": (
        "azimuth registration CONFIRMED if EVERY (geometry, view) case with "
        f"signal_size >= {REGISTRATION_MIN_SIGNAL_PIXELS} has delta_r > 0 "
        "with its 95% block-bootstrap CI excluding 0, AND no evaluated case "
        "(any signal_size) has delta_r < 0 with its 95% CI excluding 0."
    ),
    "contradicted": (
        "CONTRADICTED if ANY evaluated case has delta_r < 0 with its 95% CI "
        "excluding 0 (the mirrored azimuth orientation fits the campaign "
        "statistically better than the true one -- a registration bug, not "
        "a symmetry artefact)."
    ),
    "unresolved": (
        "otherwise (no case reaches signal_size, or a qualifying case's CI "
        "includes 0 -- inconclusive at this sample size, not evidence either way)."
    ),
    "delta_r_definition": "delta_r = corr(c, m) - corr(c, m_mirrored_in_azimuth)",
    "min_signal_pixels": REGISTRATION_MIN_SIGNAL_PIXELS,
    "signal_definition": f"number of evaluated pixels with |m| > {SIGNAL_ABS_THRESHOLD_M}",
    "exit_code_on_contradicted": 1,
}


# --------------------------------------------------------------------- utils
def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel_path_str(path: Path) -> str:
    """`path` relative to REPO_ROOT, POSIX-style (forward slashes), computed
    via os.path.relpath so it works even for a path outside the repo tree
    (using '..' segments, e.g. the campaign histogram under a sibling repo)
    -- it never silently falls back to an absolute, personal path in a
    committed artifact. Raises if `path` is on a different drive than
    REPO_ROOT (Windows only), rather than leaking an absolute path."""
    resolved = Path(path).resolve()
    try:
        rel = os.path.relpath(resolved, REPO_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"cannot express {resolved} relative to REPO_ROOT {REPO_ROOT} "
            "(different drive?) -- refusing to fall back to an absolute path"
        ) from exc
    return Path(rel).as_posix()


def write_text_lf(path: Path, text: str) -> None:
    """Write text with LF line endings regardless of platform default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


# ---------------------------------------------------------------- geometry
def find_anomaly_index(geo: VolcanoGeometry, kind: str) -> int:
    matches = [i for i, a in enumerate(geo.anomalies) if a.kind == kind]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one anomaly of kind {kind!r} in "
            f"{geo.geometry_id}, found {len(matches)}"
        )
    return matches[0]


# ------------------------------------------------------------------ marching
def march_categories(geo: VolcanoGeometry, view: DetectorView,
                     chunk_rows: int = 32) -> dict:
    """Single ray march -> per-pixel path lengths [m] for the five
    non-overlapping categories that partition the marched window, plus the
    total cone chord.

    Reuses roi._cast for the angular grid, ray directions, marched t-window
    and the "inside the edifice" test (identical to imaging.opacity_map, per
    roi.py's own docstring); only the accumulator differs from
    roi.cast_path_lengths, which returns each anomaly's raw path length
    independently and does not need their pairwise overlap.
    """
    conduit_idx = find_anomaly_index(geo, "conduit")
    body_idx = find_anomaly_index(geo, "dense_body")
    conduit = geo.anomalies[conduit_idx]
    body = geo.anomalies[body_idx]

    l_rock = np.zeros((N_EL, N_AZ), dtype=np.float64)
    l_int = np.zeros((N_EL, N_AZ), dtype=np.float64)
    l_cond_only = np.zeros((N_EL, N_AZ), dtype=np.float64)
    l_body_only = np.zeros((N_EL, N_AZ), dtype=np.float64)

    for rows, inside, x, y, z in roimod._cast(geo, view, chunk_rows):
        mask_c = VolcanoGeometry._anomaly_mask(conduit, x, y, z) & inside
        mask_b = VolcanoGeometry._anomaly_mask(body, x, y, z) & inside
        overlap = mask_c & mask_b
        l_rock[rows] = inside.sum(axis=2, dtype=np.float64) * RAY_STEP_M
        l_int[rows] = overlap.sum(axis=2, dtype=np.float64) * RAY_STEP_M
        l_cond_only[rows] = (mask_c & ~mask_b).sum(axis=2, dtype=np.float64) * RAY_STEP_M
        l_body_only[rows] = (mask_b & ~mask_c).sum(axis=2, dtype=np.float64) * RAY_STEP_M

    l_rock_only = l_rock - l_int - l_cond_only - l_body_only
    t, air_outside_cm = roimod._march_window(geo, view)
    window_m = float(t.size) * RAY_STEP_M

    return {
        "l_rock": l_rock, "l_int": l_int,
        "l_cond_only": l_cond_only, "l_body_only": l_body_only,
        "l_rock_only": l_rock_only,
        "window_m": window_m, "air_outside_cm": air_outside_cm,
        "conduit_idx": conduit_idx, "body_idx": body_idx,
        "conduit_density": conduit.density, "body_density": body.density,
        "rock_density": geo.rock_density,
    }


def build_opacity_field(cats: dict, rho_overlap: float) -> np.ndarray:
    """Column density X [g/cm^2] with the overlap assigned `rho_overlap`;
    the two exclusive anomaly regions keep their own density, everything
    else is rock or air, exactly matching imaging.opacity_map's convention
    (bounded window + analytic air outside it). Validated bit-for-bit
    against imaging.opacity_map for the B1 case (rho_overlap = body
    density) by the caller.
    """
    x = (cats["rock_density"] * cats["l_rock_only"]
        + cats["conduit_density"] * cats["l_cond_only"]
        + cats["body_density"] * cats["l_body_only"]
        + rho_overlap * cats["l_int"]
        + AIR_DENSITY * (cats["window_m"] - cats["l_rock"]))
    return x * 100.0 + AIR_DENSITY * cats["air_outside_cm"]


def build_homogeneous_field(cats: dict) -> np.ndarray:
    """Column density with no anomalies at all (rock density throughout the
    cone) -- the sky/rock control reference."""
    x = cats["rock_density"] * cats["l_rock"] + AIR_DENSITY * (cats["window_m"] - cats["l_rock"])
    return x * 100.0 + AIR_DENSITY * cats["air_outside_cm"]


def expected_counts_from_opacity(x_rock: np.ndarray, exposure_days: float) -> np.ndarray:
    """Lambda map [expected counts] from a precomputed opacity map, reusing
    the SAME flux/dOmega pipeline as imaging.expected_counts -- only the
    opacity-map computation itself is decoupled from it, so callers can feed
    in the B1 / conduit-wins / homogeneous fields above without
    re-implementing the angular grid, the flux table lookup or the solid
    angle per bin."""
    x_air = air_baseline_opacity()
    _, el = angular_grid()
    cos_zenith = np.sin(el)[:, np.newaxis] * np.ones((1, N_AZ))
    ft = flux_table()
    phi_obj = ft(fm.e_min_from_opacity(x_rock), cos_zenith)
    az, el_ = angular_grid()
    d_az = az[1] - az[0]
    d_el = el_[1] - el_[0]
    d_omega = d_az * d_el * np.cos(el_)[:, np.newaxis]
    t_sec = exposure_days * 86400.0
    return phi_obj * DETECTOR_AREA_CM2 * d_omega * t_sec


# --------------------------------------------------------------------- masks
def build_masks(cats: dict, min_path_m: float) -> dict:
    """Five DISJOINT boolean masks by priority (protocol for this
    pre-registration; distinct from roi.py's adaptive FWHM ROI, which is not
    used here because the signal of interest is the conduit/body overlap
    itself, not a per-anomaly detectability ROI)."""
    l_rock = cats["l_rock"]
    l_int = cats["l_int"]
    l_cond_only = cats["l_cond_only"]
    l_body_only = cats["l_body_only"]
    total_anomaly = l_int + l_cond_only + l_body_only

    r_int = l_int >= min_path_m
    r_cond = (~r_int) & (l_cond_only >= min_path_m)
    r_body = (~r_int) & (~r_cond) & (l_body_only >= min_path_m)
    r_rock = (l_rock >= min_path_m) & (total_anomaly < ROCK_CONTROL_MAX_ANOMALY_PATH_M)
    r_sky = l_rock == 0.0

    masks = {"R_int": r_int, "R_cond": r_cond, "R_body": r_body,
             "R_rock": r_rock, "R_sky": r_sky}
    pixel_counts = {name: int(np.count_nonzero(m)) for name, m in masks.items()}
    return {"masks": masks, "pixel_counts": pixel_counts}


def assert_masks_disjoint(masks: dict) -> None:
    names = list(masks.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.any(masks[names[i]] & masks[names[j]]):
                raise AssertionError(f"masks {names[i]!r} and {names[j]!r} overlap")


# ------------------------------------------------------------- effect table
def poisson_se(expected_count: float) -> float:
    return float(np.sqrt(max(expected_count, 0.0)))


def paired_mean_se_3seeds(expected_count: float, n_seeds: int = PILOT_N_SEEDS) -> float:
    """SE of the mean of `n_seeds` independent paired differences, each with
    Var ~= 2*expected_count under independent Poisson runs for A and B."""
    return float(np.sqrt(2.0 * max(expected_count, 0.0) / n_seeds))


def roi_effect_table(masks: dict, lam_b1: np.ndarray, lam_cw: np.ndarray,
                     s0: float, scale_factor: float) -> dict:
    table = {}
    per_event_scale = scale_factor / s0 * PILOT_EVENTS_PER_RUN
    for name, mask in masks.items():
        b1_count = float((lam_b1[mask] * per_event_scale).sum())
        cw_count = float((lam_cw[mask] * per_event_scale).sum())
        delta_max = b1_count - cw_count
        se_run = poisson_se(b1_count)
        se_mean = paired_mean_se_3seeds(b1_count)
        ratio = abs(delta_max) / se_mean if se_mean > 0 else float("nan")
        table[name] = {
            "n_pixels": int(np.count_nonzero(mask)),
            "expected_b1_per_1e6_events": b1_count,
            "expected_conduit_wins_per_1e6_events": cw_count,
            "delta_max_b1_minus_conduit_wins": delta_max,
            "poisson_se_per_run": se_run,
            "predicted_se_mean_3seeds": se_mean,
            "ratio_abs_delta_max_over_se_mean": ratio,
        }
    return table


# -------------------------------------------------------- registration check
def steiger_z_dependent(r_y1: float, r_y2: float, r12: float, n_pixels: int) -> float:
    """Steiger's (1980) z-test for two correlations that share one variable
    (here: the campaign vector, log1p) and are computed against two
    DEPENDENT, overlapping predictor maps -- the true expected map and a
    mirrored variant of it -- from the SAME sample, not from independent
    samples. Uses the asymptotic covariance of the two Fisher z-transforms
    (Steiger, 1980, eq. 3; Cohen & Cohen, 1983, ch. 3; the "steiger1980"
    method for one dependent, overlapping pair in R's `cocor` package).
    `r12` = corr(true map, mirrored map) over the same pixel mask.

    This replaces the independent-sample formula (SE = sqrt(2/(n-3))), which
    understates the true variance here: corr_true and corr_mirror are NOT
    independent draws, they are the same campaign vector correlated against
    two highly correlated (r12 near +-1) predictor maps, so their sampling
    covariance is not zero as the independent-sample formula assumes.
    """
    n_eff = max(int(n_pixels), 4)
    z1 = np.arctanh(np.clip(r_y1, -0.999999, 0.999999))
    z2 = np.arctanh(np.clip(r_y2, -0.999999, 0.999999))
    rm = (r_y1 + r_y2) / 2.0
    num = r12 * (1.0 - 2.0 * rm ** 2) - 0.5 * rm ** 2 * (1.0 - 2.0 * rm ** 2 - r12 ** 2)
    den = (1.0 - rm ** 2) ** 2
    c = num / den if den > 0.0 else 0.0
    se = np.sqrt(max((2.0 * (1.0 - c)) / (n_eff - 3), 1e-300))
    return float((z1 - z2) / se)


def _orientation_verdict(r_true: float, r_mirror: float, z: float) -> str:
    """'resolved' if the true orientation is significantly better, 'wrong'
    if the mirror is significantly better (a real registration bug), else
    'inconclusive' (cannot be told apart at this sample size -- expected
    when the underlying field has that mirror symmetry, e.g. an on-axis
    cone silhouette in azimuth)."""
    if r_true > r_mirror and z > Z_THRESHOLD:
        return "resolved"
    if r_mirror > r_true and (-z) > Z_THRESHOLD:
        return "wrong"
    return "inconclusive"


def orientation_check(campaign: np.ndarray, expected_map: np.ndarray) -> dict:
    """Correlation (log1p counts, pixels with expected > 0) between the
    campaign histogram and `expected_map`, plus the same correlation with
    `expected_map` mirrored in azimuth and in elevation. The true-vs-mirror
    gap is tested with Steiger's (1980) dependent-correlation z-test (see
    steiger_z_dependent), using r12 = corr(true map, mirrored map) over the
    same pixel mask -- NOT the independent-two-sample Fisher-z formula,
    because corr_true and corr_mirror share the same campaign sample."""
    mask = expected_map > 0.0
    log_camp = np.log1p(campaign[mask])
    log_true = np.log1p(expected_map[mask])
    map_az = np.flip(expected_map, axis=1)
    map_el = np.flip(expected_map, axis=0)
    log_az = np.log1p(map_az[mask])
    log_el = np.log1p(map_el[mask])

    r_true = float(np.corrcoef(log_camp, log_true)[0, 1])
    r_az = float(np.corrcoef(log_camp, log_az)[0, 1])
    r_el = float(np.corrcoef(log_camp, log_el)[0, 1])
    r12_az = float(np.corrcoef(log_true, log_az)[0, 1])
    r12_el = float(np.corrcoef(log_true, log_el)[0, 1])
    n = int(np.count_nonzero(mask))
    z_az = steiger_z_dependent(r_true, r_az, r12_az, n)
    z_el = steiger_z_dependent(r_true, r_el, r12_el, n)
    return {
        "n_pixels_expected_gt0": n,
        "corr_true": r_true, "corr_az_mirror": r_az, "corr_el_mirror": r_el,
        "corr_true_vs_az_mirror_map": r12_az, "corr_true_vs_el_mirror_map": r12_el,
        "steiger_z_true_vs_az_mirror": z_az, "steiger_z_true_vs_el_mirror": z_el,
        "azimuth_verdict": _orientation_verdict(r_true, r_az, z_az),
        "elevation_verdict": _orientation_verdict(r_true, r_el, z_el),
    }


def registration_check(campaign: np.ndarray, masks: dict, s0: float,
                       fields: dict) -> dict:
    orientation = {name: orientation_check(campaign, m) for name, m in fields.items()}

    campaign_per_event = campaign / s0
    sky_mean = float(campaign_per_event[masks["R_sky"]].mean()) if masks["R_sky"].any() else float("nan")
    rock_mean = float(campaign_per_event[masks["R_rock"]].mean()) if masks["R_rock"].any() else float("nan")

    ratio_per_roi = {}
    lam_b1 = fields["B1"]
    for name, mask in masks.items():
        expected_sum = float(lam_b1[mask].sum())
        campaign_sum = float(campaign[mask].sum())
        ratio_per_roi[name] = (campaign_sum / expected_sum) if expected_sum > 0 else None

    verdicts = [o["azimuth_verdict"] for o in orientation.values()] + \
        [o["elevation_verdict"] for o in orientation.values()]
    any_wrong = "wrong" in verdicts

    return {
        "orientation": orientation,
        "independent_code_level_check": {
            "note": ("Pixel registration is additionally guaranteed by exact "
                    "equivalence of the binning formula, independent of the "
                    "statistical correlation above: both the surrogate "
                    "angular grid and the Geant4 histogram bin index use "
                    "numpy.linspace(min, max, N) CENTER binning over the "
                    "same [az_half, el_min, el_max] parameters, azimuth "
                    "measured relative to the same look direction."),
            "surrogate_file": "scripts/simulation/surrogate/imaging.py",
            "surrogate_function": "angular_grid, ray_directions",
            "geant4_file": "scripts/simulation/geant4_app/src/SteppingAction.cc",
            "geant4_function": "binIndex (line 24), relAz/iAz/iEl (lines 47-58)",
        },
        "sky_vs_rock_control": {
            "mean_counts_per_event_R_sky": sky_mean,
            "mean_counts_per_event_R_rock": rock_mean,
            "sky_much_higher_than_rock": bool(sky_mean > 10.0 * rock_mean)
            if np.isfinite(sky_mean) and np.isfinite(rock_mean) else None,
        },
        "campaign_to_expected_b1_ratio_per_roi": ratio_per_roi,
        "any_orientation_contradicted": any_wrong,
    }


# --------------------------------------------------- registration (U8b) core
def homogeneous_opacity_map(geo: VolcanoGeometry, view: DetectorView) -> np.ndarray:
    """Column density X [g/cm^2] with NO anomalies (rock density throughout
    the cone), for ANY geometry/anomaly combination.

    Same convention as build_homogeneous_field (U8), but that function
    consumes march_categories' conduit/dense_body-specific decomposition;
    this one only needs the total rock chord L_rock, so it reuses
    roi.rock_path_length (itself roi._cast + roi._march_window, the same
    march imaging.opacity_map performs) unmodified, without re-deriving a
    per-anomaly breakdown this check does not need.
    """
    l_rock = roimod.rock_path_length(geo, view).astype(np.float64)
    t, air_outside_cm = roimod._march_window(geo, view)
    window_m = float(t.size) * RAY_STEP_M
    x = geo.rock_density * l_rock + AIR_DENSITY * (window_m - l_rock)
    return x * 100.0 + AIR_DENSITY * air_outside_cm


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, NaN (not a crash) if either side has zero
    variance (e.g. a degenerate block-bootstrap resample)."""
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _block_ids(shape: tuple, block: int) -> np.ndarray:
    """Integer id per block*block tile covering `shape`, row-major."""
    n_col_blocks = (shape[1] + block - 1) // block
    rows = (np.arange(shape[0]) // block)[:, np.newaxis]
    cols = (np.arange(shape[1]) // block)[np.newaxis, :]
    return rows * n_col_blocks + cols


def case_seed(base_seed: int, geometry_id: str, view_index: int) -> int:
    """Deterministic block-bootstrap sub-seed for one (geometry, view) case.

    Derived from the pair's own identity (a hash of "geometry:view"), not
    from its position in the --geometries/--views lists. A running case
    counter would make a case's seed -- and therefore its bootstrap CI --
    depend on which and how many other geometries/views were requested in
    the same run and in what order, so the same case run alone or as part
    of a different subset would not reproduce the same CI even though the
    point estimates (delta_r, steiger_z) are seed-independent and always
    reproduce.
    """
    digest = hashlib.sha256(f"{geometry_id}:{view_index}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big")
    return (base_seed + offset) % (2**31 - 1)


def block_bootstrap_delta_r_ci(campaign_res: np.ndarray, model_res: np.ndarray,
                               model_res_mirror: np.ndarray, mask: np.ndarray,
                               block: int, n_bootstrap: int, rng: np.random.Generator) -> dict:
    """95% percentile CI of Delta r = corr(c, m) - corr(c, m_mirrored) under a
    pixel-BLOCK bootstrap: block x block spatial tiles are resampled WITH
    REPLACEMENT (not individual pixels), because neighbouring pixels share a
    ray-march footprint and an i.i.d.-pixel bootstrap would understate the
    true sampling variance of a spatially correlated residual map."""
    ids = _block_ids(mask.shape, block)
    flat_mask = mask.ravel()
    flat_ids = ids.ravel()[flat_mask]
    flat_c = campaign_res.ravel()[flat_mask]
    flat_m = model_res.ravel()[flat_mask]
    flat_mm = model_res_mirror.ravel()[flat_mask]

    unique_ids, inverse = np.unique(flat_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(unique_ids.size)]
    n_blocks = unique_ids.size

    deltas = np.full(n_bootstrap, np.nan, dtype=np.float64)
    for b in range(n_bootstrap):
        picks = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([groups[p] for p in picks]) if n_blocks > 0 else np.empty(0, dtype=int)
        if idx.size < 2:
            continue
        r1 = corr_safe(flat_c[idx], flat_m[idx])
        r2 = corr_safe(flat_c[idx], flat_mm[idx])
        if np.isfinite(r1) and np.isfinite(r2):
            deltas[b] = r1 - r2

    valid = deltas[np.isfinite(deltas)]
    n_failed = int(n_bootstrap - valid.size)
    if valid.size == 0:
        return {"ci95": [float("nan"), float("nan")], "n_bootstrap": int(n_bootstrap),
                "n_blocks": int(n_blocks), "n_failed_resamples": n_failed}
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return {"ci95": [float(lo), float(hi)], "n_bootstrap": int(n_bootstrap),
            "n_blocks": int(n_blocks), "n_failed_resamples": n_failed}


def evaluate_orientation_case(geometry_id: str, view_index: int, campaign_path: Path,
                              min_counts: float, block: int, n_bootstrap: int,
                              seed: int) -> dict:
    """Full azimuth + elevation (positive control) registration check for one
    (geometry, view), evaluated on pixels with E_homog > 0, inside the cone
    footprint, and campaign counts >= min_counts.

    Returns {'case_azimuth': {...}, 'case_elevation': {...}, 'maps': {...}}
    where 'maps' carries the full (N_EL, N_AZ) campaign/model residual arrays
    (azimuth-mirrored variant included) ONLY for the plotting step -- not
    written to the JSON report.
    """
    geo = roimod.load_geometry(geometry_id)
    view = DetectorView(geometry_id=geometry_id, view_index=view_index)

    l_rock = roimod.rock_path_length(geo, view).astype(np.float64)
    footprint = l_rock > 0.0

    x_anom = opacity_map(geo, view)
    x_homog = homogeneous_opacity_map(geo, view)
    lam_anom = expected_counts_from_opacity(x_anom, CAMPAIGN_EXPOSURE_DAYS)
    lam_homog = expected_counts_from_opacity(x_homog, CAMPAIGN_EXPOSURE_DAYS)

    campaign = load_campaign_hist(campaign_path)

    evaluated = footprint & (lam_homog > 0.0) & (campaign >= min_counts)
    n_evaluated = int(np.count_nonzero(evaluated))

    m = np.log(lam_anom / lam_homog)
    m_mirror_az = mirror_azimuth(m)
    m_mirror_el = mirror_elevation(m)

    if n_evaluated >= 2 and float(lam_anom[evaluated].sum()) > 0.0:
        s = float(campaign[evaluated].sum()) / float(lam_anom[evaluated].sum())
    else:
        s = float("nan")
    c = np.log((campaign + 0.5) / (max(s, 1e-300) * lam_homog))

    signal_size = int(np.count_nonzero(evaluated & (np.abs(m) > SIGNAL_ABS_THRESHOLD_M)))

    rng = np.random.default_rng(seed)

    def _axis_case(axis_name: str, m_mirror: np.ndarray) -> dict:
        if n_evaluated < 2:
            return {
                "geometry": geometry_id, "view": view_index, "axis": axis_name,
                "n_pixels_evaluated": n_evaluated, "signal_size": signal_size,
                "insufficient_pixels": True,
            }
        c_ev, m_ev, mm_ev = c[evaluated], m[evaluated], m_mirror[evaluated]
        r_true = corr_safe(c_ev, m_ev)
        r_mirror = corr_safe(c_ev, mm_ev)
        r12 = corr_safe(m_ev, mm_ev)
        z = steiger_z_dependent(r_true, r_mirror, r12, n_evaluated)
        delta_r = r_true - r_mirror
        boot = block_bootstrap_delta_r_ci(c, m, m_mirror, evaluated, block, n_bootstrap, rng)
        lo, hi = boot["ci95"]
        ci_excludes_zero = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0))
        return {
            "geometry": geometry_id, "view": view_index, "axis": axis_name,
            "n_pixels_evaluated": n_evaluated, "signal_size": signal_size,
            "insufficient_pixels": False,
            "scale_factor_s": s,
            "r_true": r_true, "r_mirror": r_mirror,
            "r12_true_vs_mirror": r12,
            "steiger_z": z,
            "delta_r": delta_r,
            "delta_r_ci95": [lo, hi],
            "delta_r_ci95_n_bootstrap": boot["n_bootstrap"],
            "delta_r_ci95_n_blocks": boot["n_blocks"],
            "delta_r_ci95_n_failed_resamples": boot["n_failed_resamples"],
            "ci_excludes_zero": ci_excludes_zero,
            "confirms_true_orientation": bool(ci_excludes_zero and delta_r > 0.0),
            "contradicts_true_orientation": bool(ci_excludes_zero and delta_r < 0.0),
        }

    return {
        "case_azimuth": _axis_case("azimuth", m_mirror_az),
        "case_elevation": _axis_case("elevation", m_mirror_el),
        "maps": {
            "campaign_residual_c": c, "model_residual_m": m,
            "model_residual_m_mirror_az": m_mirror_az, "evaluated_mask": evaluated,
        },
        "n_evaluated_pixels": n_evaluated,
        "campaign_hist_path": rel_path_str(campaign_path),
        "campaign_hist_sha256": sha256_of(campaign_path),
    }


def registration_verdict(azimuth_cases: list) -> dict:
    """Apply REGISTRATION_CRITERION to the azimuth cases (see its docstring:
    elevation is a positive control, never part of this verdict)."""
    usable = [c for c in azimuth_cases if not c.get("insufficient_pixels")]
    contradicting = [c for c in usable if c["contradicts_true_orientation"]]
    qualifying = [c for c in usable if c["signal_size"] >= REGISTRATION_MIN_SIGNAL_PIXELS]
    qualifying_confirmed = [c for c in qualifying if c["confirms_true_orientation"]]

    if contradicting:
        verdict = "CONTRADICTED"
    elif qualifying and len(qualifying_confirmed) == len(qualifying):
        verdict = "CONFIRMED"
    else:
        verdict = "UNRESOLVED"

    return {
        "verdict": verdict,
        "n_cases_total": len(azimuth_cases),
        "n_cases_usable": len(usable),
        "n_cases_qualifying_by_signal_size": len(qualifying),
        "n_cases_qualifying_confirmed": len(qualifying_confirmed),
        "n_cases_contradicting": len(contradicting),
        "contradicting_cases": [
            {"geometry": c["geometry"], "view": c["view"],
             "delta_r": c["delta_r"], "delta_r_ci95": c["delta_r_ci95"]}
            for c in contradicting
        ],
    }


def mirror_azimuth(field: np.ndarray) -> np.ndarray:
    """Flip a (N_EL, N_AZ) map along the azimuth (column) axis."""
    return np.flip(field, axis=1)


def mirror_elevation(field: np.ndarray) -> np.ndarray:
    """Flip a (N_EL, N_AZ) map along the elevation (row) axis."""
    return np.flip(field, axis=0)


# ------------------------------------------------------- registration-sum (U8c)
# Region-SUM azimuth-registration check (work unit U8c): sums campaign counts
# over whole regions instead of correlating per-pixel values, trading spatial
# resolution for statistical power. Work unit U8b's per-pixel correlation
# check found 0/30 cases reaching its signal-size floor because the
# anomaly-affected pixels sit deep in the opaque interior, where campaign
# counts are too sparse for the per-pixel filter (C >= 20) to keep them; this
# check applies NO per-pixel count filter, only a minimum on the SUM.
EXCLUDED_OVERLAP_GEOMETRIES = ("geo-train-06", "geo-train-08", "geo-train-11")

REGISTRATION_SUM_VIEWS = [0, 1, 2, 3, 4, 5]

# Pre-stated region thresholds on the model residual m = log(E_anom/E_homog)
# (work unit U8c design, fixed before any case is computed).
REGISTRATION_SUM_EXCESS_DEFICIT_T = 0.05
REGISTRATION_SUM_BASELINE_T0 = 0.01
REGISTRATION_SUM_MIN_N = 100
REGISTRATION_SUM_Z_COMBINED_THRESHOLD = 3.0
REGISTRATION_SUM_Z_NULL_STOUFFER_THRESHOLD = 3.0
REGISTRATION_SUM_Z_NULL_MAX_THRESHOLD = 4.0

# The homogeneous field's own azimuth-mirror symmetry, in EXPECTED COUNTS
# (E_homog = lam_homog), is a NONLINEAR function of the homogeneous rock path
# length L (opacity -> flux -> counts): a single float32 ray-step sampling
# flip of L at the cone surface in an oblique view (see
# registration_sum_symmetry_gate below) can inflate this relative difference
# far past what a real azimuth-registration bug would need to produce -- this
# is exactly what the first real run's gate failure (max relative difference
# 9.154e-03) traced back to (see REGISTRATION_SUM_CRITERION["amendment"]).
# This tolerance therefore no longer gates the command directly; it now
# bounds the POST-exclusion invariant checked by
# homogeneous_symmetry_after_exclusion (symmetry_max_rel_diff_after_exclusion),
# which must hold once flip pixels are excluded.
REGISTRATION_SUM_SYMMETRY_TOL = 1e-9

# Path-length azimuth-mirror symmetry gate (amendment, work unit U8c):
# |L(p) - L(mirror(p))| must be within this fraction of an integer multiple
# of RAY_STEP_M to count as a genuine single/multi-step float32 sampling
# flip rather than a registration bug.
REGISTRATION_SUM_STEP_MULTIPLE_TOL = 1e-6

# Path-length azimuth-mirror symmetry gate: at most this fraction of the cone
# footprint may flip (L(p) != L(mirror(p))) before the case is rejected.
REGISTRATION_SUM_MAX_FLIP_FRACTION = 0.001

# Written into the JSON verbatim, BEFORE any case is computed -- the
# pre-stated criterion for work unit U8c (see task context / this module's
# top-level constants above for the exact numeric thresholds it references).
REGISTRATION_SUM_CRITERION = {
    "t_excess_deficit_threshold": REGISTRATION_SUM_EXCESS_DEFICIT_T,
    "t0_baseline_threshold": REGISTRATION_SUM_BASELINE_T0,
    "min_n_per_subtest": REGISTRATION_SUM_MIN_N,
    "z_combined_threshold": REGISTRATION_SUM_Z_COMBINED_THRESHOLD,
    "z_null_stouffer_threshold": REGISTRATION_SUM_Z_NULL_STOUFFER_THRESHOLD,
    "z_null_max_threshold": REGISTRATION_SUM_Z_NULL_MAX_THRESHOLD,
    "symmetry_gate": (
        "Before any region sum is computed, each case's homogeneous rock "
        "path length L = roi.rock_path_length(geo, view) is compared to its "
        "azimuth mirror L_m = L[:, ::-1] over the cone footprint (pixels "
        "where L > 0 or L_m > 0). flip(p) = (L(p) != L_m(p)), symmetric by "
        "construction. A case PASSES iff every nonzero |L(p) - L_m(p)| is an "
        f"integer multiple of RAY_STEP_M within {REGISTRATION_SUM_STEP_MULTIPLE_TOL:.0e} "
        "of the nearest integer step count, AND flip.sum() <= "
        f"{REGISTRATION_SUM_MAX_FLIP_FRACTION} * footprint_pixels. Any "
        "failing case stops the command before any region sum is computed, "
        "with the same exit code as the checks below."
    ),
    "region_definitions": {
        "excess_S": ("m(p) > t and |m(s(p))| < t0 and not flip(p), where s is "
                    "the azimuth mirror (column j -> N_az-1-j) and flip(p) is "
                    "the symmetry gate's per-pixel flip mask"),
        "deficit_D": "m(p) < -t and |m(s(p))| < t0 and not flip(p)",
        "mirrors": ("S' = s(S), D' = s(D); disjoint from S, D by construction "
                   "since t0 < t; flip is mirror-symmetric by construction "
                   "(flip(p) == flip(s(p))), so S' and D' also exclude flip "
                   "pixels"),
    },
    "subtest_statistic": (
        "k = campaign counts summed over the sub-test's region (S or D); "
        "N = campaign counts summed over the region union its mirror (S "
        "union S', D union D'); no per-pixel count filter. Eligible iff "
        "N >= min_n_per_subtest (never depends on the direction of the "
        "imbalance). z = (k - N/2) / sqrt(N/4); signed z_s = +z for an "
        "excess sub-test and -z for a deficit sub-test, so z_s > 0 means the "
        "imbalance points the way the code-level registration predicts. The "
        "exact two-sided binomial p-value (scipy.stats.binomtest(k, N, 0.5)) "
        "and p_model = E_anom(S) / (E_anom(S) + E_anom(S')) are also "
        "recorded; p_model is context only, never used by the verdict."
    ),
    "null_control": (
        "for every eligible sub-test, the same (region, mirror) pixel pair "
        "is evaluated on the campaign map of the SAME view of every OTHER "
        "geometry in the case list whose own model residual satisfies "
        "|m| < t0 on the whole region union; z_null uses the same unsigned "
        "formula, only if that geometry's own N >= min_n_per_subtest."
    ),
    "combination": (
        "Z = Stouffer sum(z_s)/sqrt(n) over eligible sub-tests (independent "
        "geometries/views); Z_null = Stouffer sum(z_null)/sqrt(n_null) over "
        "all null evaluations, plus max|z_null|. Null evaluations reuse the "
        "same campaign maps across several region pairs and are therefore "
        "NOT independent, which is why the max|z_null| gate is applied in "
        "addition to the Stouffer combination."
    ),
    "confirmed": (
        "CONFIRMED if at least one sub-test is eligible, the symmetry null "
        "control passes (|Z_null| < z_null_stouffer_threshold AND "
        "max|z_null| < z_null_max_threshold), the combined Z is >= "
        "z_combined_threshold, AND no eligible sub-test has "
        "z_s <= -z_combined_threshold."
    ),
    "contradicted": (
        "CONTRADICTED if the symmetry null control passes and the combined "
        "Z is <= -z_combined_threshold."
    ),
    "unresolved": (
        "UNRESOLVED if no sub-test is eligible (insufficient counts), OR the "
        "symmetry null control fails (|Z_null| >= z_null_stouffer_threshold "
        "OR max|z_null| >= z_null_max_threshold), OR neither the CONFIRMED "
        "nor the CONTRADICTED condition above holds."
    ),
    "exit_code_on_contradicted": 1,
    "amendment": (
        "Symmetry gate amended on 2026-09-29 before any region sum was computed: "
        "the first run stopped at the original floating-point gate (max relative "
        "difference 9.154e-03), traced to single-step (2.0 m) float32 sampling "
        "flips of the analytic ray march at the cone surface in oblique views; "
        "flipped pixels are now excluded and the gate requires exact step "
        "multiples on at most 0.1 % of the footprint."
    ),
}


def discover_registration_sum_geometries(
        exclude: tuple = EXCLUDED_OVERLAP_GEOMETRIES) -> list:
    """Every geo-train-*/geo-val-* catalog geometry id, sorted, minus
    `exclude` -- enumerated from the catalog file names themselves (work unit
    U8c design item 1), not from a hand-maintained list. geo-test-*/geo-gen-*
    are never included: only train/val geometries may feed a design/analysis
    step, per the same protocol as REGISTRATION_GEOMETRIES above."""
    ids = []
    for path in sorted(GEOMETRY_DIR.glob("geo-*.json")):
        gid = path.stem
        if (gid.startswith("geo-train-") or gid.startswith("geo-val-")) and gid not in exclude:
            ids.append(gid)
    return ids


def registration_sum_symmetry_gate(
        l_rock: np.ndarray,
        step: float = RAY_STEP_M,
        step_tol: float = REGISTRATION_SUM_STEP_MULTIPLE_TOL,
        max_flip_fraction: float = REGISTRATION_SUM_MAX_FLIP_FRACTION) -> dict:
    """Path-length azimuth-mirror symmetry gate (amendment, work unit U8c;
    see REGISTRATION_SUM_CRITERION["amendment"]), replacing the original
    E_homog relative-difference gate.

    E_homog (the homogeneous field's expected-count map) is a NONLINEAR
    function of the homogeneous rock path length L (opacity -> flux ->
    counts), so a single float32 ray-step sampling flip of L at the cone
    surface in an oblique view can appear as a large relative difference in
    E_homog even though it is not a registration bug -- exactly what the
    first real run's gate failure (max relative difference 9.154e-03) traced
    back to. L itself is linear and directly comparable step by step, so the
    gate is evaluated on L, not on E_homog.

    `flip` = (L != L_mirror) is symmetric by construction: if L(p) !=
    L_mirror(p) at a pixel p, then at p's own mirror site, L(mirror(p)) =
    L_mirror(p) and L_mirror(mirror(p)) = L(p), so L(mirror(p)) !=
    L_mirror(mirror(p)) too -- flip(p) == flip(mirror(p)) always.
    `footprint` is the union of both sides' nonzero rock path (pixels where
    L or its mirror sees the cone at all).

    The case PASSES iff every nonzero |L - L_mirror| is an integer multiple
    of `step` within `step_tol` of the nearest integer step count, AND
    flip.sum() <= max_flip_fraction * footprint_pixels.
    """
    l_mirror = l_rock[:, ::-1]
    footprint = (l_rock > 0.0) | (l_mirror > 0.0)
    footprint_pixels = int(np.count_nonzero(footprint))
    flip = l_rock != l_mirror
    n_flip_pixels = int(np.count_nonzero(flip))

    d = np.abs(l_rock - l_mirror)
    nonzero = d > 0.0
    if np.any(nonzero):
        step_counts = d[nonzero] / step
        step_residuals = np.abs(step_counts - np.round(step_counts))
        max_abs_dl_steps = float(np.max(step_counts))
        max_step_residual = float(np.max(step_residuals))
        is_step_multiple = bool(np.all(step_residuals < step_tol))
    else:
        max_abs_dl_steps = 0.0
        max_step_residual = 0.0
        is_step_multiple = True

    within_flip_budget = n_flip_pixels <= max_flip_fraction * footprint_pixels

    return {
        "flip": flip,
        "footprint": footprint,
        "footprint_pixels": footprint_pixels,
        "n_flip_pixels": n_flip_pixels,
        "max_abs_dl_steps": max_abs_dl_steps,
        "max_step_residual": max_step_residual,
        "is_step_multiple": is_step_multiple,
        "within_flip_budget": within_flip_budget,
        "passed": bool(is_step_multiple and within_flip_budget),
    }


def homogeneous_symmetry_after_exclusion(lam_homog: np.ndarray, footprint: np.ndarray,
                                         flip: np.ndarray) -> float:
    """Max relative |E_homog(p) - E_homog(mirror(p))| / E_homog(p) over
    footprint pixels NOT flagged by `flip` (work unit U8c amendment).

    Once the single/multi-step sampling-flip pixels are excluded, E_homog at
    a pixel depends only on L at that pixel and on its (azimuth-independent)
    elevation row, so it must be exactly mirror-symmetric on the complement;
    registration_sum_case_maps asserts this rather than only reporting it."""
    keep = footprint & ~flip
    if not np.any(keep):
        return 0.0
    lam_homog_mirror = mirror_azimuth(lam_homog)
    rel_diff = np.abs(lam_homog[keep] - lam_homog_mirror[keep]) / lam_homog[keep]
    return float(np.max(rel_diff))


def registration_sum_case_maps(geometry_id: str, view_index: int) -> dict:
    """Model residual m = log(E_anom / E_homog) and the path-length
    azimuth-mirror symmetry gate (work unit U8c amendment, see
    registration_sum_symmetry_gate and REGISTRATION_SUM_CRITERION
    ["amendment"]), for one (geometry, view). Reuses opacity_map /
    homogeneous_opacity_map / expected_counts_from_opacity unmodified -- no
    ray march is repeated here (only roi.rock_path_length, itself a single
    march, is added for the cone footprint and the symmetry gate)."""
    geo = roimod.load_geometry(geometry_id)
    view = DetectorView(geometry_id=geometry_id, view_index=view_index)

    l_rock = roimod.rock_path_length(geo, view).astype(np.float64)
    footprint = l_rock > 0.0

    x_anom = opacity_map(geo, view)
    x_homog = homogeneous_opacity_map(geo, view)
    lam_anom = expected_counts_from_opacity(x_anom, CAMPAIGN_EXPOSURE_DAYS)
    lam_homog = expected_counts_from_opacity(x_homog, CAMPAIGN_EXPOSURE_DAYS)

    valid = lam_homog > 0.0
    m = np.full(lam_homog.shape, np.nan, dtype=np.float64)
    m[valid] = np.log(lam_anom[valid] / lam_homog[valid])

    gate = registration_sum_symmetry_gate(l_rock)
    flip = gate["flip"]

    # Old metric, kept for information only -- no longer used to gate the
    # command (see REGISTRATION_SUM_CRITERION["amendment"]): E_homog is a
    # nonlinear function of L, so a single-step L flip can inflate this
    # relative difference well past what a real registration bug would need.
    lam_homog_mirror = mirror_azimuth(lam_homog)
    sym_mask = footprint & valid
    if np.any(sym_mask):
        rel_diff = np.abs(lam_homog[sym_mask] - lam_homog_mirror[sym_mask]) / lam_homog[sym_mask]
        symmetry_max_rel_diff = float(np.max(rel_diff))
    else:
        symmetry_max_rel_diff = 0.0

    symmetry_max_rel_diff_after_exclusion = homogeneous_symmetry_after_exclusion(
        lam_homog, gate["footprint"], flip)
    if symmetry_max_rel_diff_after_exclusion > REGISTRATION_SUM_SYMMETRY_TOL:
        raise AssertionError(
            f"{geometry_id} view {view_index}: E_homog is not exactly "
            "azimuth-symmetric after excluding flip pixels (max relative "
            f"difference {symmetry_max_rel_diff_after_exclusion:.3e} exceeds "
            f"tolerance {REGISTRATION_SUM_SYMMETRY_TOL:.1e}) -- this points "
            "to a bug in the exclusion logic itself, not the expected "
            "float32 sampling artefact."
        )

    return {
        "geometry": geometry_id, "view": view_index,
        "m": m, "lam_anom": lam_anom,
        "flip": flip,
        "footprint_pixels": gate["footprint_pixels"],
        "n_flip_pixels": gate["n_flip_pixels"],
        "max_abs_dl_steps": gate["max_abs_dl_steps"],
        "max_step_residual": gate["max_step_residual"],
        "symmetry_gate_passed": gate["passed"],
        "symmetry_max_rel_diff": symmetry_max_rel_diff,
        "symmetry_max_rel_diff_after_exclusion": symmetry_max_rel_diff_after_exclusion,
    }


def registration_sum_regions(m: np.ndarray, t: float, t0: float,
                             flip: np.ndarray | None = None) -> dict:
    """Excess (S) and deficit (D) regions plus their azimuth mirrors (S', D'),
    per work unit U8c's pre-stated rule, now also excluding the symmetry
    gate's flip pixels (work unit U8c amendment): S = {p : m(p) > t,
    |m(s(p))| < t0, not flip(p)} and likewise for D. `flip` defaults to "no
    exclusion" (all-False) so callers that pre-date the amendment (and the
    synthetic cases in the test suite that do not carry a flip mask) are
    unaffected. S and S' (and D and D') are disjoint BY CONSTRUCTION whenever
    t0 < t: a pixel in S has |m(mirror)| < t0 < t, so that mirror pixel
    cannot itself satisfy m > t and be in S -- the flip exclusion only
    shrinks S/D further and cannot create a new overlap. Asserted here, not
    just claimed. Because flip is mirror-symmetric by construction
    (flip(p) == flip(s(p))), mirroring S/D below automatically carries the
    exclusion over to S'/D' as well."""
    valid = ~np.isnan(m)
    m_mirror = mirror_azimuth(m)
    valid_pair = valid & ~np.isnan(m_mirror)
    not_flip = ~flip if flip is not None else np.ones(m.shape, dtype=bool)

    excess = valid_pair & (m > t) & (np.abs(m_mirror) < t0) & not_flip
    deficit = valid_pair & (m < -t) & (np.abs(m_mirror) < t0) & not_flip
    excess_mirror = mirror_azimuth(excess)
    deficit_mirror = mirror_azimuth(deficit)

    if np.any(excess & excess_mirror):
        raise AssertionError("excess region S and its azimuth mirror S' overlap")
    if np.any(deficit & deficit_mirror):
        raise AssertionError("deficit region D and its azimuth mirror D' overlap")

    return {"excess": excess, "excess_mirror": excess_mirror,
            "deficit": deficit, "deficit_mirror": deficit_mirror}


def registration_sum_subtest(kind: str, region: np.ndarray, region_mirror: np.ndarray,
                             campaign: np.ndarray, lam_anom: np.ndarray,
                             min_n: int) -> dict | None:
    """One region-sum sub-test (excess or deficit): k/N counts, eligibility
    (N >= min_n, independent of the direction of the imbalance), the signed
    z_s, the exact two-sided binomial p-value, and p_model (context only, not
    used by the verdict). Returns None if the region is empty (nothing to
    test), so an empty sub-test never silently becomes "ineligible" noise in
    the report."""
    n_pixels = int(np.count_nonzero(region))
    if n_pixels == 0:
        return None
    union = region | region_mirror
    k = float(campaign[region].sum())
    N = float(campaign[union].sum())
    k_int, N_int = int(round(k)), int(round(N))
    eligible = N_int >= min_n

    e_s = float(lam_anom[region].sum())
    e_s_mirror = float(lam_anom[region_mirror].sum())
    e_denom = e_s + e_s_mirror
    p_model = (e_s / e_denom) if e_denom > 0.0 else None

    result = {
        "kind": kind, "n_pixels": n_pixels,
        "n_pixels_mirror": int(np.count_nonzero(region_mirror)),
        "k": k_int, "N": N_int, "eligible": eligible, "p_model": p_model,
    }
    if not eligible:
        result.update({"z": None, "z_s": None, "binomial_p_value": None})
        return result

    z = (k_int - N_int / 2.0) / np.sqrt(N_int / 4.0)
    z_s = z if kind == "excess" else -z
    p_value = float(stats.binomtest(k_int, N_int, 0.5).pvalue)
    result.update({"z": float(z), "z_s": float(z_s), "binomial_p_value": p_value})
    return result


def registration_sum_null_evaluations(region: np.ndarray, union: np.ndarray,
                                      other_cases: list, t0: float, min_n: int) -> list:
    """Evaluate the SAME (region, union) pixel pair on every `other_case`'s
    own campaign map, but only where that other geometry's own model residual
    is itself below the baseline threshold t0 everywhere on `union` (i.e. it
    carries no anomaly signal there and can serve as a symmetry null)."""
    evaluations = []
    for other in other_cases:
        m_other = other["m"]
        union_vals = m_other[union]
        if union_vals.size == 0 or not np.all(np.isfinite(union_vals)):
            continue
        if not np.all(np.abs(union_vals) < t0):
            continue
        campaign_other = other["campaign"]
        k_null = float(campaign_other[region].sum())
        N_null = float(campaign_other[union].sum())
        k_int, N_int = int(round(k_null)), int(round(N_null))
        if N_int < min_n:
            continue
        z_null = (k_int - N_int / 2.0) / np.sqrt(N_int / 4.0)
        evaluations.append({
            "geometry": other["geometry"], "view": other["view"],
            "k": k_int, "N": N_int, "z_null": float(z_null),
        })
    return evaluations


def registration_sum_verdict(eligible_subtests: list, null_evaluations: list) -> dict:
    """Apply the REGISTRATION_SUM_CRITERION thresholds (see its docstring)."""
    n_eligible = len(eligible_subtests)
    n_null = len(null_evaluations)
    z_null_sum = sum(e["z_null"] for e in null_evaluations)
    Z_null = (z_null_sum / np.sqrt(n_null)) if n_null > 0 else 0.0
    max_abs_z_null = max((abs(e["z_null"]) for e in null_evaluations), default=0.0)

    if n_eligible == 0:
        Z = None
        verdict, reason = "UNRESOLVED", "insufficient counts (no eligible sub-test reached N >= min_n)"
    else:
        Z = sum(st["z_s"] for st in eligible_subtests) / np.sqrt(n_eligible)
        any_eligible_negative = any(st["z_s"] <= -REGISTRATION_SUM_Z_COMBINED_THRESHOLD
                                    for st in eligible_subtests)
        if (abs(Z_null) >= REGISTRATION_SUM_Z_NULL_STOUFFER_THRESHOLD
                or max_abs_z_null >= REGISTRATION_SUM_Z_NULL_MAX_THRESHOLD):
            verdict, reason = "UNRESOLVED", "symmetry control failed"
        elif Z >= REGISTRATION_SUM_Z_COMBINED_THRESHOLD and not any_eligible_negative:
            verdict, reason = "CONFIRMED", "combined Z reaches the threshold and no sub-test contradicts it"
        elif Z <= -REGISTRATION_SUM_Z_COMBINED_THRESHOLD:
            verdict, reason = "CONTRADICTED", "combined Z indicates the imbalance points against the prediction"
        else:
            verdict, reason = "UNRESOLVED", "combined Z is between the confirmation and contradiction thresholds"

    return {
        "verdict": verdict, "verdict_reason": reason,
        "n_eligible_subtests": n_eligible,
        "Z": (float(Z) if Z is not None else None),
        "n_null_evaluations": n_null,
        "Z_null": float(Z_null),
        "max_abs_z_null": float(max_abs_z_null),
    }


def registration_sum_check(cases: list, t: float = REGISTRATION_SUM_EXCESS_DEFICIT_T,
                           t0: float = REGISTRATION_SUM_BASELINE_T0,
                           min_n: int = REGISTRATION_SUM_MIN_N) -> dict:
    """Full region-sum registration check (work unit U8c), decoupled from
    geometry/campaign I/O so it can run on synthetic maps in the test suite.

    `cases`: list of {"geometry": str, "view": int, "m": ndarray,
    "lam_anom": ndarray, "campaign": ndarray}, plus an optional "flip"
    ndarray (work unit U8c amendment: the symmetry gate's per-pixel flip
    mask, excluded from every region -- see registration_sum_regions).
    Cases without a "flip" key are evaluated with no exclusion, so synthetic
    cases built before the amendment are unaffected. The null control (see
    registration_sum_null_evaluations) pairs each eligible sub-test with
    every OTHER case at the SAME `view`.
    """
    by_view: dict = {}
    for case in cases:
        by_view.setdefault(case["view"], []).append(case)

    per_case = []
    subtests_by_case = []
    for case in cases:
        regions = registration_sum_regions(case["m"], t, t0, case.get("flip"))
        case_subtests = {}
        for kind, region_key, mirror_key in (
                ("excess", "excess", "excess_mirror"),
                ("deficit", "deficit", "deficit_mirror")):
            region, region_mirror = regions[region_key], regions[mirror_key]
            result = registration_sum_subtest(kind, region, region_mirror,
                                              case["campaign"], case["lam_anom"], min_n)
            case_subtests[kind] = result
            if result is not None and result["eligible"]:
                subtests_by_case.append({
                    "geometry": case["geometry"], "view": case["view"], "kind": kind,
                    "region": region, "region_mirror": region_mirror,
                    "union": region | region_mirror, "z_s": result["z_s"],
                })
        per_case.append({
            "geometry": case["geometry"], "view": case["view"],
            "symmetry_max_rel_diff": case.get("symmetry_max_rel_diff"),
            "symmetry_max_rel_diff_after_exclusion":
                case.get("symmetry_max_rel_diff_after_exclusion"),
            "n_flip_pixels": case.get("n_flip_pixels"),
            "footprint_pixels": case.get("footprint_pixels"),
            "max_abs_dl_steps": case.get("max_abs_dl_steps"),
            "max_step_residual": case.get("max_step_residual"),
            "subtests": case_subtests,
        })

    eligible_subtests = []
    null_evaluations = []
    for st in subtests_by_case:
        others = [c for c in by_view[st["view"]] if c["geometry"] != st["geometry"]]
        evals = registration_sum_null_evaluations(st["region"], st["union"], others, t0, min_n)
        for e in evals:
            e["source_geometry"] = st["geometry"]
            e["source_view"] = st["view"]
            e["source_kind"] = st["kind"]
        null_evaluations.extend(evals)
        eligible_subtests.append({
            "geometry": st["geometry"], "view": st["view"], "kind": st["kind"], "z_s": st["z_s"],
        })

    verdict = registration_sum_verdict(eligible_subtests, null_evaluations)
    return {
        "per_case": per_case,
        "eligible_subtests": eligible_subtests,
        "null_evaluations": null_evaluations,
        **verdict,
    }


def plot_registration_sum(out_path: Path, result: dict) -> None:
    """z_s per eligible sub-test (with +-3 reference lines) and the null z
    distribution (with +-4 reference lines), work unit U8c."""
    apply_style()
    import matplotlib.pyplot as plt

    subtests = result["eligible_subtests"]
    nulls = result["null_evaluations"]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))

    ax = axes[0]
    if subtests:
        labels = [f"{s['geometry']} v{s['view']}\n({s['kind']})" for s in subtests]
        z_values = [s["z_s"] for s in subtests]
        colors = ["#2ca02c" if z >= 3.0 else ("#d62728" if z <= -3.0 else "#1f77b4")
                 for z in z_values]
        ax.bar(range(len(z_values)), z_values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    else:
        ax.text(0.5, 0.5, "sin sub-pruebas elegibles", ha="center", va="center",
               transform=ax.transAxes)
    ax.axhline(3.0, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(-3.0, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_ylabel("z_s (sub-prueba elegible)")
    ax.set_title("z_s por sub-prueba de suma de regiones")

    ax2 = axes[1]
    if nulls:
        z_null_values = [e["z_null"] for e in nulls]
        ax2.hist(z_null_values, bins=min(30, max(5, len(z_null_values) // 2)),
                color="#7f7f7f", edgecolor="white")
    else:
        ax2.text(0.5, 0.5, "sin evaluaciones nulas", ha="center", va="center",
                transform=ax2.transAxes)
    ax2.axvline(4.0, color="black", linestyle="--", linewidth=1.0)
    ax2.axvline(-4.0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("z_null (control de simetría)")
    ax2.set_ylabel("frecuencia")
    ax2.set_title("Distribución de z_null (control nulo)")

    fig.suptitle("Registro de orientación azimutal por suma de regiones (U8c)")
    watermark_surrogate(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_registration(out_path: Path, maps: dict, geometry_id: str, view_index: int) -> None:
    """Campaign residual, true model residual and azimuth-mirrored model
    residual, on the SAME diverging colour scale, for the (geometry, view)
    case with the largest azimuth signal_size."""
    apply_style()
    import matplotlib.pyplot as plt

    az, el = angular_grid()
    az_half_deg = float(np.rad2deg(az[-1]))
    el_min_deg, el_max_deg = float(np.rad2deg(el[0])), float(np.rad2deg(el[-1]))
    extent = [-az_half_deg, az_half_deg, el_min_deg, el_max_deg]

    mask = maps["evaluated_mask"]
    c_map = np.where(mask, maps["campaign_residual_c"], np.nan)
    m_map = np.where(mask, maps["model_residual_m"], np.nan)
    mm_map = np.where(mask, maps["model_residual_m_mirror_az"], np.nan)

    finite_vals = np.concatenate([
        np.abs(c_map[np.isfinite(c_map)]), np.abs(m_map[np.isfinite(m_map)]),
        np.abs(mm_map[np.isfinite(mm_map)]),
    ])
    vmax = float(np.percentile(finite_vals, 99)) if finite_vals.size else 1.0
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))
    panels = [
        ("Residuo de campaña c", c_map),
        ("Residuo del modelo m (orientación real)", m_map),
        ("Residuo del modelo m (azimut espejado)", mm_map),
    ]
    im = None
    for ax, (title, field) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", aspect="auto", extent=extent,
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Azimut relativo [°]")
        ax.grid(False)
    axes[0].set_ylabel("Elevación [°]")
    fig.suptitle(f"Registro de orientación azimutal -- {geometry_id}, vista {view_index} "
                f"(caso con mayor señal)")
    fig.colorbar(im, ax=list(axes), shrink=0.85, label="log(razón) [adimensional]")
    watermark_surrogate(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------- I/O
def load_campaign_hist(path: Path) -> np.ndarray:
    counts = np.loadtxt(path, delimiter=",")
    if counts.shape != (N_EL, N_AZ):
        raise ValueError(
            f"campaign histogram {path} has shape {counts.shape}, "
            f"expected ({N_EL}, {N_AZ})"
        )
    return counts.astype(np.float64)


def plot_preregistration(out_path: Path, lam_b1: np.ndarray, campaign: np.ndarray,
                         masks: dict, geometry_id: str, view_index: int) -> None:
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    az, el = angular_grid()
    az_half_deg = float(np.rad2deg(az[-1]))
    el_min_deg, el_max_deg = float(np.rad2deg(el[0])), float(np.rad2deg(el[-1]))
    extent = [-az_half_deg, az_half_deg, el_min_deg, el_max_deg]

    colors = {"R_int": "#d62728", "R_cond": "#ff7f0e",
             "R_body": "#9467bd", "R_rock": "#2ca02c", "R_sky": "#1f77b4"}
    labels = {"R_int": "R_int (solape conducto-cuerpo)",
             "R_cond": "R_cond (conducto exclusivo)",
             "R_body": "R_body (cuerpo denso exclusivo)",
             "R_rock": "R_rock (control roca)",
             "R_sky": "R_sky (control cielo)"}
    # R_int is the pilot's central comparison and sits immediately adjacent
    # to R_cond/R_body's boundary (it is carved out of them by priority), so
    # it is drawn LAST (on top, higher zorder, thicker line) -- otherwise the
    # two neighbouring contours visually erase it.
    draw_order = ["R_cond", "R_body", "R_rock", "R_sky", "R_int"]
    linewidths = {"R_int": 2.4}
    zorders = {"R_int": 5}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    axes[0].imshow(np.log1p(lam_b1), origin="lower", aspect="auto",
                  extent=extent, cmap="viridis")
    axes[0].set_title(f"Mapa esperado B1 ({geometry_id}, vista {view_index})")

    axes[1].imshow(np.log1p(campaign), origin="lower", aspect="auto",
                  extent=extent, cmap="viridis")
    axes[1].set_title("Histograma de campaña (Geant4, geometría recibida)")

    for ax in axes:
        for name in draw_order:
            mask = masks[name]
            if not mask.any():
                continue
            ax.contour(mask.astype(float), levels=[0.5],
                      colors=[colors[name]], extent=extent,
                      linewidths=linewidths.get(name, 1.2),
                      zorder=zorders.get(name, 1))
        ax.set_xlabel("Azimut relativo [°]")
        ax.set_ylabel("Elevación [°]")
        ax.grid(False)

    handles = [Patch(facecolor="none", edgecolor=colors[n], label=labels[n]) for n in masks]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    watermark_surrogate(fig)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------ command
def cmd_preregister(args: argparse.Namespace) -> int:
    geometry_id = args.geometry
    geometry_path = GEOMETRY_DIR / f"{geometry_id}.json"
    geo = roimod.load_geometry(geometry_id)
    view = DetectorView(geometry_id=geometry_id, view_index=args.view)

    campaign = load_campaign_hist(args.campaign_hist)
    tables_meta = json.loads(TABLES_META_PATH.read_text(encoding="utf-8"))
    s0 = float(tables_meta["total_open_sky_counts_30days_per_view"])

    cats = march_categories(geo, view)
    mask_result = build_masks(cats, args.min_path_m)
    masks = mask_result["masks"]
    assert_masks_disjoint(masks)

    x_b1 = build_opacity_field(cats, cats["body_density"])
    x_cw = build_opacity_field(cats, cats["conduit_density"])
    x_homog = build_homogeneous_field(cats)

    x_ref = opacity_map(geo, view)
    b1_consistency_max_diff = float(np.max(np.abs(x_b1 - x_ref)))

    lam_b1 = expected_counts_from_opacity(x_b1, CAMPAIGN_EXPOSURE_DAYS)
    lam_cw = expected_counts_from_opacity(x_cw, CAMPAIGN_EXPOSURE_DAYS)
    lam_homog = expected_counts_from_opacity(x_homog, CAMPAIGN_EXPOSURE_DAYS)

    campaign_detected_total = float(campaign.sum())
    lam_b1_total = float(lam_b1.sum())
    scale_factor = campaign_detected_total / lam_b1_total

    effect_table = roi_effect_table(masks, lam_b1, lam_cw, s0, scale_factor)

    fields = {"B1": lam_b1, "conduit_wins": lam_cw, "homogeneous": lam_homog}
    reg_check = registration_check(campaign, masks, s0, fields)

    if reg_check["any_orientation_contradicted"]:
        print("REGISTRATION CHECK FAILED: a mirrored orientation correlates "
             "significantly BETTER than the true one -- this indicates a "
             "pixel-registration bug, not a symmetry artefact. Stopping "
             "without writing pre-registration outputs.", file=sys.stderr)
        print(json.dumps(reg_check["orientation"], indent=2, default=str), file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = args.out_dir / "roi-mascaras.npz"
    np.savez_compressed(
        npz_path,
        R_int=masks["R_int"], R_cond=masks["R_cond"], R_body=masks["R_body"],
        R_rock=masks["R_rock"], R_sky=masks["R_sky"],
        l_rock=cats["l_rock"].astype(np.float32),
        l_int=cats["l_int"].astype(np.float32),
        l_cond_only=cats["l_cond_only"].astype(np.float32),
        l_body_only=cats["l_body_only"].astype(np.float32),
        l_rock_only=cats["l_rock_only"].astype(np.float32),
    )

    png_path = args.out_dir / "roi-preregistro.png"
    plot_preregistration(png_path, lam_b1, campaign, masks, geometry_id, args.view)

    total_pixels = N_EL * N_AZ
    classified = sum(mask_result["pixel_counts"].values())

    report = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "script_sha256": sha256_of(Path(__file__)),
            "geometry_id": geometry_id,
            "geometry_path": rel_path_str(geometry_path),
            "geometry_sha256": sha256_of(geometry_path),
            "view": args.view,
            "min_path_m": args.min_path_m,
            "seed": args.seed,
            "seed_used": False,
            "campaign_hist_path": rel_path_str(args.campaign_hist),
            "campaign_hist_sha256": sha256_of(args.campaign_hist),
            "tables_meta_path": rel_path_str(TABLES_META_PATH),
            "tables_meta_sha256": sha256_of(TABLES_META_PATH),
            "campaign_exposure_days": CAMPAIGN_EXPOSURE_DAYS,
            "pilot_events_per_run": PILOT_EVENTS_PER_RUN,
            "pilot_n_seeds": PILOT_N_SEEDS,
        },
        "notes": {
            "seed_usage": ("This subcommand is fully deterministic (no "
                          "sampling); --seed is accepted for interface "
                          "consistency with a future randomized check and "
                          "is not otherwise used."),
        },
        "angular_grid": {
            "n_az": N_AZ, "n_el": N_EL,
            "az_half_deg": float(np.rad2deg(angular_grid()[0][-1])),
            "el_range_deg": [float(np.rad2deg(angular_grid()[1][0])),
                            float(np.rad2deg(angular_grid()[1][-1]))],
        },
        "roi_definitions": {
            "priority_order": ["R_int", "R_cond", "R_body"],
            "R_int": f"L_int(p) >= {args.min_path_m} m (path length inside conduit AND body)",
            "R_cond": f"not R_int; L_cond_only(p) >= {args.min_path_m} m",
            "R_body": f"not R_int, not R_cond; L_body_only(p) >= {args.min_path_m} m",
            "R_rock": (f"L_rock(p) >= {args.min_path_m} m (total cone chord) AND "
                      f"L_int+L_cond_only+L_body_only < {ROCK_CONTROL_MAX_ANOMALY_PATH_M} m"),
            "R_sky": "L_rock(p) == 0",
        },
        "pixel_counts": {**mask_result["pixel_counts"],
                        "unclassified": total_pixels - classified,
                        "total": total_pixels},
        "material_fields": {
            "B1": {"description": "overlap assigned to the dense body (received-geometry proxy)",
                  "rho_overlap_g_cm3": cats["body_density"],
                  "lam_sum_30d": lam_b1_total,
                  "consistency_vs_imaging_opacity_map_max_abs_diff_g_cm2": b1_consistency_max_diff},
            "conduit_wins": {"description": "overlap assigned to the conduit (upper-bound envelope, not a proposed fix)",
                            "rho_overlap_g_cm3": cats["conduit_density"],
                            "lam_sum_30d": float(lam_cw.sum())},
            "homogeneous": {"description": "no anomalies (reference for the sky/rock control)",
                           "lam_sum_30d": float(lam_homog.sum())},
        },
        "scale_factor": {
            "definition": ("campaign_detected_total / lambda_B1_sum_30d -- an "
                          "APPROXIMATE correction of the idealized surrogate "
                          "flux integral to the Geant4-detected scale for "
                          "this view; not a claim of exact physical equivalence."),
            "campaign_detected_total": campaign_detected_total,
            "lambda_b1_sum_30d": lam_b1_total,
            "value": scale_factor,
        },
        "open_sky_normalisation": {
            "s0_muons_per_view_30d": s0,
            "source": rel_path_str(TABLES_META_PATH),
        },
        "roi_effect_table": effect_table,
        "registration_check": reg_check,
        "outputs": {
            "masks_npz": rel_path_str(npz_path),
            "figure_png": rel_path_str(png_path),
        },
    }

    json_path = args.out_dir / "roi-preregistro.json"
    write_text_lf(json_path, json.dumps(report, indent=2, default=str) + "\n")

    print(f"OE1 pilot ROI pre-registration -- {geometry_id} view {args.view}")
    print(f"pixel counts: {mask_result['pixel_counts']}")
    print(f"scale factor (campaign detected / surrogate B1 expected): {scale_factor:.4f}")
    for name, row in effect_table.items():
        print(f"  {name}: B1={row['expected_b1_per_1e6_events']:.4f} "
             f"CW={row['expected_conduit_wins_per_1e6_events']:.4f} "
             f"Delta_max={row['delta_max_b1_minus_conduit_wins']:.4f} "
             f"SE_mean={row['predicted_se_mean_3seeds']:.4f} "
             f"ratio={row['ratio_abs_delta_max_over_se_mean']:.4f}")
    for name, o in reg_check["orientation"].items():
        print(f"  orientation[{name}]: true={o['corr_true']:.6f} "
             f"az_mirror={o['corr_az_mirror']:.6f} (z={o['steiger_z_true_vs_az_mirror']:.3f}, "
             f"{o['azimuth_verdict']}) el_mirror={o['corr_el_mirror']:.6f} "
             f"(z={o['steiger_z_true_vs_el_mirror']:.3f}, {o['elevation_verdict']})")
    print(f"wrote {npz_path}")
    print(f"wrote {png_path}")
    print(f"wrote {json_path}")
    return 0


def cmd_registration(args: argparse.Namespace) -> int:
    geometries = args.geometries
    views = args.views

    cases_azimuth, cases_elevation, per_case_records = [], [], []
    maps_by_case = {}
    input_hashes = {}

    for geometry_id in geometries:
        geometry_path = GEOMETRY_DIR / f"{geometry_id}.json"
        input_hashes[f"geometry:{geometry_id}"] = {
            "path": rel_path_str(geometry_path), "sha256": sha256_of(geometry_path),
        }
        for view_index in views:
            campaign_path = args.campaign_dir / geometry_id / f"{geometry_id}_view{view_index}_hist.csv"
            if not campaign_path.exists():
                print(f"WARNING: missing campaign histogram, skipping: {campaign_path}",
                     file=sys.stderr)
                continue
            # A sub-seed derived from the (geometry, view) pair's own identity
            # (case_seed), not from its position in --geometries/--views: the
            # block bootstrap of a given case is then reproducible on its own,
            # regardless of which other geometries/views were requested in the
            # same run or in what order.
            effective_seed = case_seed(args.seed, geometry_id, view_index)
            result = evaluate_orientation_case(
                geometry_id, view_index, campaign_path,
                args.min_counts, args.block, args.bootstrap, effective_seed,
            )
            input_hashes[f"campaign:{geometry_id}:view{view_index}"] = {
                "path": result["campaign_hist_path"], "sha256": result["campaign_hist_sha256"],
            }
            cases_azimuth.append(result["case_azimuth"])
            cases_elevation.append(result["case_elevation"])
            maps_by_case[(geometry_id, view_index)] = result["maps"]
            per_case_records.append({
                "geometry": geometry_id, "view": view_index, "seed": effective_seed,
                "azimuth": result["case_azimuth"], "elevation": result["case_elevation"],
            })

    verdict = registration_verdict(cases_azimuth)

    usable_elevation = [c for c in cases_elevation if not c.get("insufficient_pixels")]
    elevation_control = {
        "n_cases_usable": len(usable_elevation),
        "all_confirm_true_orientation": bool(usable_elevation) and all(
            c["confirms_true_orientation"] for c in usable_elevation),
        "any_contradict_true_orientation": any(
            c["contradicts_true_orientation"] for c in usable_elevation),
        "note": ("Elevation is already resolved (work unit U8's Steiger test, "
                "z ~ 456 on geo-train-06); this cross-geometry re-check is a "
                "POSITIVE CONTROL for the method itself, not part of the "
                "azimuth verdict above -- if it failed to confirm here, that "
                "would indicate a bug in this check, not a real elevation "
                "registration problem."),
    }

    usable_azimuth = [c for c in cases_azimuth if not c.get("insufficient_pixels")]
    best_case = max(usable_azimuth, key=lambda c: c["signal_size"], default=None)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / "registro-orientacion.png"
    if best_case is not None:
        plot_registration(png_path, maps_by_case[(best_case["geometry"], best_case["view"])],
                          best_case["geometry"], best_case["view"])
    else:
        print("WARNING: no case had >=2 evaluated pixels; skipping figure.", file=sys.stderr)

    report = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "script_sha256": sha256_of(Path(__file__)),
            "campaign_dir": rel_path_str(args.campaign_dir),
            "geometries": geometries,
            "views": views,
            "min_counts": args.min_counts,
            "block_px": args.block,
            "n_bootstrap": args.bootstrap,
            "seed": args.seed,
            "campaign_exposure_days": CAMPAIGN_EXPOSURE_DAYS,
        },
        "criterion": REGISTRATION_CRITERION,
        "per_case": per_case_records,
        "azimuth_verdict": verdict,
        "elevation_positive_control": elevation_control,
        "input_hashes": input_hashes,
        "outputs": {
            "figure_png": rel_path_str(png_path) if best_case is not None else None,
            "figure_case": ({"geometry": best_case["geometry"], "view": best_case["view"]}
                           if best_case is not None else None),
        },
    }

    json_path = args.out_dir / "registro-orientacion.json"
    write_text_lf(json_path, json.dumps(report, indent=2, default=str) + "\n")

    print("OE1 azimuth registration check (work unit U8b)")
    print(f"geometries: {geometries}")
    print(f"views: {views}")
    for rec in per_case_records:
        a = rec["azimuth"]
        if a.get("insufficient_pixels"):
            print(f"  {rec['geometry']} view{rec['view']}: insufficient evaluated pixels")
            continue
        print(f"  {rec['geometry']} view{rec['view']}: n={a['n_pixels_evaluated']} "
             f"signal={a['signal_size']} delta_r={a['delta_r']:.4f} "
             f"ci95=[{a['delta_r_ci95'][0]:.4f}, {a['delta_r_ci95'][1]:.4f}] "
             f"z={a['steiger_z']:.3f}")
    print(f"VERDICT: {verdict['verdict']} "
         f"(qualifying={verdict['n_cases_qualifying_by_signal_size']}, "
         f"confirmed={verdict['n_cases_qualifying_confirmed']}, "
         f"contradicting={verdict['n_cases_contradicting']})")
    print(f"elevation positive control: all_confirm={elevation_control['all_confirm_true_orientation']}")
    print(f"wrote {json_path}")
    if best_case is not None:
        print(f"wrote {png_path}")

    if verdict["verdict"] == "CONTRADICTED":
        print("REGISTRATION CHECK CONTRADICTED: a mirrored azimuth orientation fits "
             "the campaign significantly better than the true one for at least one "
             "evaluated case -- see registro-orientacion.json.", file=sys.stderr)
        return 1
    return 0


def cmd_registration_sum(args: argparse.Namespace) -> int:
    geometries = args.geometries
    views = args.views

    case_maps = {}
    input_hashes = {}
    for geometry_id in geometries:
        geometry_path = GEOMETRY_DIR / f"{geometry_id}.json"
        input_hashes[f"geometry:{geometry_id}"] = {
            "path": rel_path_str(geometry_path), "sha256": sha256_of(geometry_path)}
        for view_index in views:
            case_maps[(geometry_id, view_index)] = registration_sum_case_maps(geometry_id, view_index)

    symmetry_records = [
        {
            "geometry": g, "view": v,
            "max_rel_diff": cm["symmetry_max_rel_diff"],
            "max_rel_diff_after_exclusion": cm["symmetry_max_rel_diff_after_exclusion"],
            "n_flip_pixels": cm["n_flip_pixels"],
            "footprint_pixels": cm["footprint_pixels"],
            "max_abs_dl_steps": cm["max_abs_dl_steps"],
            "max_step_residual": cm["max_step_residual"],
            "gate_passed": cm["symmetry_gate_passed"],
        }
        for (g, v), cm in case_maps.items()
    ]
    worst = max((r["max_rel_diff"] for r in symmetry_records), default=0.0)
    failing = [r for r in symmetry_records if not r["gate_passed"]]
    if failing:
        print("SYMMETRY CHECK FAILED: the path-length azimuth-mirror gate (work "
             "unit U8c amendment) rejected at least one case -- either a "
             "flipped pixel's |L - L_mirror| is not an integer multiple of "
             f"RAY_STEP_M ({RAY_STEP_M} m) within tolerance, or more than "
             f"{REGISTRATION_SUM_MAX_FLIP_FRACTION * 100:.2f}% of the cone "
             "footprint flipped -- this indicates a real azimuth-registration "
             "bug, not the expected single-step float32 sampling artefact. "
             "Stopping without computing region sums.", file=sys.stderr)
        for rec in sorted(failing, key=lambda r: -r["n_flip_pixels"]):
            print(f"  {rec['geometry']} view{rec['view']}: n_flip_pixels="
                 f"{rec['n_flip_pixels']}/{rec['footprint_pixels']} "
                 f"max_abs_dL_steps={rec['max_abs_dl_steps']:.3e} "
                 f"max_step_residual={rec['max_step_residual']:.3e}", file=sys.stderr)
        return 1

    cases = []
    missing = []
    for geometry_id in geometries:
        for view_index in views:
            campaign_path = args.campaign_dir / geometry_id / f"{geometry_id}_view{view_index}_hist.csv"
            if not campaign_path.exists():
                missing.append(str(campaign_path))
                continue
            campaign = load_campaign_hist(campaign_path)
            input_hashes[f"campaign:{geometry_id}:view{view_index}"] = {
                "path": rel_path_str(campaign_path), "sha256": sha256_of(campaign_path)}
            cm = case_maps[(geometry_id, view_index)]
            cases.append({
                "geometry": geometry_id, "view": view_index,
                "m": cm["m"], "lam_anom": cm["lam_anom"], "campaign": campaign,
                "flip": cm["flip"],
                "symmetry_max_rel_diff": cm["symmetry_max_rel_diff"],
                "symmetry_max_rel_diff_after_exclusion":
                    cm["symmetry_max_rel_diff_after_exclusion"],
                "n_flip_pixels": cm["n_flip_pixels"],
                "footprint_pixels": cm["footprint_pixels"],
                "max_abs_dl_steps": cm["max_abs_dl_steps"],
                "max_step_residual": cm["max_step_residual"],
            })
    for path_str in missing:
        print(f"WARNING: missing campaign histogram, skipping case: {path_str}", file=sys.stderr)

    result = registration_sum_check(cases)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / "registro-orientacion-suma.png"
    plot_registration_sum(png_path, result)

    report = {
        "metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "script_sha256": sha256_of(Path(__file__)),
            "campaign_dir": rel_path_str(args.campaign_dir),
            "geometries": geometries,
            "views": views,
            "campaign_exposure_days": CAMPAIGN_EXPOSURE_DAYS,
            "n_cases_requested": len(geometries) * len(views),
            "n_cases_evaluated": len(cases),
        },
        "criterion": REGISTRATION_SUM_CRITERION,
        "symmetry_check": {
            "gate": "path_length_symmetry (work unit U8c amendment; see criterion.amendment)",
            "step_multiple_tolerance": REGISTRATION_SUM_STEP_MULTIPLE_TOL,
            "max_flip_fraction": REGISTRATION_SUM_MAX_FLIP_FRACTION,
            "post_exclusion_tolerance": REGISTRATION_SUM_SYMMETRY_TOL,
            "max_rel_diff_overall": worst,
            "n_cases_failing_gate": len(failing),
            "per_case": symmetry_records,
        },
        "per_case": result["per_case"],
        "eligible_subtests_summary": result["eligible_subtests"],
        "null_evaluations": result["null_evaluations"],
        "combination": {
            "n_eligible_subtests": result["n_eligible_subtests"],
            "Z": result["Z"],
            "n_null_evaluations": result["n_null_evaluations"],
            "Z_null": result["Z_null"],
            "max_abs_z_null": result["max_abs_z_null"],
        },
        "verdict": {"verdict": result["verdict"], "reason": result["verdict_reason"]},
        "input_hashes": input_hashes,
        "outputs": {"figure_png": rel_path_str(png_path)},
    }

    json_path = args.out_dir / "registro-orientacion-suma.json"
    write_text_lf(json_path, json.dumps(report, indent=2, default=str) + "\n")

    print("OE1 azimuth registration check by region sums (work unit U8c)")
    print(f"geometries: {geometries}")
    print(f"views: {views}")
    print(f"cases evaluated: {len(cases)} / {len(geometries) * len(views)} requested")
    for case in result["per_case"]:
        for kind, sub in case["subtests"].items():
            if sub is None or not sub["eligible"]:
                continue
            print(f"  {case['geometry']} view{case['view']} [{kind}]: "
                 f"k={sub['k']} N={sub['N']} z_s={sub['z_s']:.3f} p={sub['binomial_p_value']:.3e}")
    print(f"combined Z = {result['Z']} over {result['n_eligible_subtests']} eligible sub-test(s)")
    print(f"null control: Z_null = {result['Z_null']:.3f}, max|z_null| = {result['max_abs_z_null']:.3f} "
         f"over {result['n_null_evaluations']} evaluation(s)")
    print(f"VERDICT: {result['verdict']} ({result['verdict_reason']})")
    print(f"wrote {json_path}")
    print(f"wrote {png_path}")

    if result["verdict"] == "CONTRADICTED":
        print("REGISTRATION-SUM CHECK CONTRADICTED: the region-sum imbalance points "
             "against the predicted azimuth orientation -- see "
             "registro-orientacion-suma.json.", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preregister",
                         help="compute and commit the five ROI masks and their "
                              "pre-pilot expected effect sizes")
    pre.add_argument("--geometry", default=DEFAULT_GEOMETRY_ID,
                     help="geometry catalog id (default: %(default)s)")
    pre.add_argument("--view", type=int, default=0,
                     help="detector view index (default: %(default)s)")
    pre.add_argument("--campaign-hist", type=Path, required=True,
                     help="path to the merged campaign *_hist.csv for this geometry/view")
    pre.add_argument("--out-dir", type=Path, required=True,
                     help="directory to write roi-mascaras.npz, roi-preregistro.json, roi-preregistro.png")
    pre.add_argument("--min-path-m", type=float, default=DEFAULT_MIN_PATH_M,
                     help="minimum path length [m] for the R_int/R_cond/R_body/R_rock thresholds "
                          "(default: %(default)s)")
    pre.add_argument("--seed", type=int, default=DEFAULT_SEED,
                     help="recorded for interface consistency; not used (this subcommand is "
                          "deterministic; default: %(default)s)")

    reg = sub.add_parser("registration",
                         help="empirical, cross-geometry azimuth-registration check "
                              "(work unit U8b) against the Geant4 training campaign")
    reg.add_argument("--campaign-dir", type=Path, required=True,
                     help="root of the merged Geant4 campaign "
                          "(<campaign-dir>/<geometry>/<geometry>_view<v>_hist.csv)")
    reg.add_argument("--geometries", nargs="+", default=list(REGISTRATION_GEOMETRIES),
                     help="train/val catalog geometry ids to check "
                          "(default: %(default)s)")
    reg.add_argument("--views", nargs="+", type=int, default=list(REGISTRATION_VIEWS),
                     help="detector view indices to check (default: %(default)s)")
    reg.add_argument("--min-counts", type=float, default=DEFAULT_MIN_COUNTS,
                     help="minimum campaign counts for a pixel to be evaluated "
                          "(default: %(default)s)")
    reg.add_argument("--block", type=int, default=DEFAULT_BLOCK_PX,
                     help="block-bootstrap tile size in pixels (default: %(default)s)")
    reg.add_argument("--bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP,
                     help="number of block-bootstrap resamples (default: %(default)s)")
    reg.add_argument("--seed", type=int, default=DEFAULT_SEED,
                     help="RNG seed for the block bootstrap (default: %(default)s)")
    reg.add_argument("--out-dir", type=Path, required=True,
                     help="directory to write registro-orientacion.json/.png")

    regsum = sub.add_parser("registration-sum",
                            help="region-sum azimuth-registration check (work unit U8c) "
                                 "against the Geant4 training campaign")
    regsum.add_argument("--campaign-dir", type=Path, required=True,
                        help="root of the merged Geant4 campaign "
                             "(<campaign-dir>/<geometry>/<geometry>_view<v>_hist.csv)")
    regsum.add_argument("--geometries", nargs="+",
                        default=discover_registration_sum_geometries(),
                        help="train/val catalog geometry ids to check (default: every "
                             "geo-train-*/geo-val-* catalog geometry except "
                             + ", ".join(EXCLUDED_OVERLAP_GEOMETRIES))

    regsum.add_argument("--views", nargs="+", type=int, default=list(REGISTRATION_SUM_VIEWS),
                        help="detector view indices to check (default: %(default)s)")
    regsum.add_argument("--out-dir", type=Path, required=True,
                        help="directory to write registro-orientacion-suma.json/.png")

    args = ap.parse_args()
    if args.command == "preregister":
        return cmd_preregister(args)
    if args.command == "registration":
        return cmd_registration(args)
    if args.command == "registration-sum":
        return cmd_registration_sum(args)
    ap.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
