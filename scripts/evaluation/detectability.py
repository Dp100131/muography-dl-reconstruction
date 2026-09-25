"""Stage 19 — anomaly detectability (protocol v1.1 sections 5.2 and 5.3).

Every quantity is measured on the PREPROCESSED [0, 1] images, the space the
generator actually emits. The preprocessing log1p(counts)/scale is monotone in
the counts and the counts are monotone in the transmission, so the SIGN of the
contrast survives the transform: a cavity stays brighter than its background
and a dense body stays darker. That is the only property the classification
rules need from the transform.

    C    = (mu_ROI - mu_BG) / mu_BG
    CNR  = (mu_ROI - mu_BG) / sigma_BG
    z    = (mu_ROI - mu_BG) / (sigma_BG sqrt(1/n_ROI + 1/n_BG))

The detection threshold z_thr is NOT assumed: it is the 99th percentile of |z|
over the sham ROIs measured on the VALIDATION references during the dry run
and frozen as a number in protocol v1.1. That absorbs the spatial correlation
of the images, so the hallucination rate is calibrated to a 1 % false positive
rate by construction rather than by a Gaussian argument that does not hold.

"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

from metrics import compute_ssim  # noqa: E402
import roi as roi_module  # noqa: E402

# --- classification constants (protocol v1.1 section 5.3, PRE-REGISTERED) ---
# X is frozen in the dry run as max(0.30, 3 * rel-SD(C_ref over K)); 0.30 is
# the floor, never the final number unless the measured jitter is smaller.
PRESERVED_TOLERANCE_X = 0.30
INFLATION_TOLERANCE_Y = 0.50

PRESERVED = "preserved"
LOST = "lost"
INFLATED = "inflated"
DETECTED_NOT_PRESERVED = "detected_not_preserved"
UNDETECTED_REF = "undetected_ref"
HALLUCINATED = "hallucinated"
HALLUCINATED_SHAM = "hallucinated_sham"
CLEAN_SHAM = "clean_sham"
LABELS = (PRESERVED, LOST, INFLATED, DETECTED_NOT_PRESERVED, UNDETECTED_REF)
# the five outcomes are INDEPENDENT predicates, not a partition: see classify()
FLAGS = (PRESERVED, INFLATED, LOST, HALLUCINATED_SHAM, UNDETECTED_REF)

# ROI SSIM crop: the bounding box grown by this margin, floored at the SSIM
# window so structural_similarity never sees a patch smaller than 7x7.
LOCAL_SSIM_MARGIN_PX = 8
LOCAL_SSIM_MIN_SIDE = 7
N_OCTILES = 8


def _finite_stats(values: np.ndarray):
    values = np.asarray(values, dtype=np.float64).ravel()
    return values.size, (float(values.mean()) if values.size else float("nan"))


def roi_stats(img: np.ndarray, roi: np.ndarray, annulus: np.ndarray) -> dict:
    """Contrast, CNR and the detection statistic of one ROI against its ring.

    sigma_BG uses the sample standard deviation (ddof=1): the ring is a sample
    of the background, not the population, and n_BG >= 200 makes the
    difference numerically irrelevant but the convention explicit.
    """
    img = np.asarray(img, dtype=np.float64)
    roi = np.asarray(roi, dtype=bool)
    annulus = np.asarray(annulus, dtype=bool)

    n_roi, mean_roi = _finite_stats(img[roi])
    n_bg, mean_bg = _finite_stats(img[annulus])
    std_bg = (float(np.asarray(img[annulus], dtype=np.float64).std(ddof=1))
              if n_bg > 1 else float("nan"))

    difference = mean_roi - mean_bg
    contrast = (difference / mean_bg
                if np.isfinite(mean_bg) and mean_bg != 0.0 else float("nan"))
    cnr = difference / std_bg if std_bg and np.isfinite(std_bg) and std_bg > 0 \
        else float("nan")
    if n_roi > 0 and n_bg > 1 and np.isfinite(std_bg) and std_bg > 0:
        z = difference / (std_bg * np.sqrt(1.0 / n_roi + 1.0 / n_bg))
    else:
        z = float("nan")
    return {"mean_roi": mean_roi, "mean_bg": mean_bg, "std_bg": std_bg,
            "n_roi": int(n_roi), "n_bg": int(n_bg),
            "C": float(contrast), "CNR": float(cnr), "z": float(z)}


def _same_sign(a: float, b: float) -> bool:
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return (a > 0 and b > 0) or (a < 0 and b < 0)


def is_detected(stats: dict, reference: dict, z_thr: float) -> bool:
    """|z| >= z_thr AND sign(C) == sign(C_ref) (protocol v1.1 section 5.3)."""
    z = stats.get("z", float("nan"))
    if not np.isfinite(z) or abs(z) < z_thr:
        return False
    return _same_sign(stats.get("C", float("nan")),
                      reference.get("C", float("nan")))


def classify(stats_enh: dict, stats_ref: dict, z_thr: float,
             x_tolerance: float = PRESERVED_TOLERANCE_X,
             y_tolerance: float = INFLATION_TOLERANCE_Y) -> dict:
    """INDEPENDENT flags plus a primary label, per section 5.3.

    The five outcomes of the protocol are predicates, not a partition, and the
    frozen constants do not make them exclusive: `preserved` is
    |C - C_ref|/|C_ref| <= X and `inflated` is |C| > (1 + Y)|C_ref|, so any
    X > Y admits a case that is BOTH -- with X = 0.6 and Y = 0.5 a contrast at
    1.55 C_ref is inside the preservation tolerance and past the inflation
    tolerance at the same time. X is frozen in the dry run as
    max(0.30, 3 rel-SD(C_ref over K)) and can therefore exceed Y = 0.50, so
    collapsing the two into one label would silently drop inflated cases from
    the inflation rate exactly when the reference jitter is largest. The rates
    of Stage 21 read the flags; the label exists for the tidy table and for the
    galleries, where one word per case is what fits.

    Returned keys: preserved, inflated, lost, hallucinated_sham,
    undetected_ref (booleans) and label (one of LABELS).

    UNDETECTED_REF is the "not detectable even at 100 %" valve: the instance is
    removed from the denominator of the preservation rate and reported apart
    (candidate: the conduit of geo-test-01, the weakest of the nine).
    """
    flags = {name: False for name in FLAGS}

    z_ref = stats_ref.get("z", float("nan"))
    c_ref = stats_ref.get("C", float("nan"))
    if not np.isfinite(z_ref) or abs(z_ref) < z_thr or not np.isfinite(c_ref) \
            or c_ref == 0.0:
        flags[UNDETECTED_REF] = True
        return {**flags, "label": UNDETECTED_REF}
    if not is_detected(stats_enh, stats_ref, z_thr):
        flags[LOST] = True
        return {**flags, "label": LOST}

    c_enh = stats_enh["C"]
    flags[PRESERVED] = bool(abs(c_enh - c_ref) / abs(c_ref) <= x_tolerance)
    flags[INFLATED] = bool(abs(c_enh) > (1.0 + y_tolerance) * abs(c_ref))
    label = (PRESERVED if flags[PRESERVED]
             else (INFLATED if flags[INFLATED] else DETECTED_NOT_PRESERVED))
    return {**flags, "label": label}


def classify_sham(stats_sham: dict, z_thr: float) -> str:
    """A sham ROI holds no anomaly: any |z| >= z_thr is a structural
    hallucination. The denominator is 10 shams x 18 cells x 20 realizations
    per level = 3 600 measurements."""
    z = stats_sham.get("z", float("nan"))
    if np.isfinite(z) and abs(z) >= z_thr:
        return HALLUCINATED
    return CLEAN_SHAM


def rate_with_denominator(labels, numerator_labels, exclude=(UNDETECTED_REF,)):
    """(rate, numerator, denominator) with the excluded valve removed."""
    kept = [label for label in labels if label not in exclude]
    numerator = sum(1 for label in kept if label in numerator_labels)
    denominator = len(kept)
    rate = numerator / denominator if denominator else float("nan")
    return float(rate), int(numerator), int(denominator)


def flag_rate(classifications, flag: str):
    """(rate, numerator, denominator) over classify() dicts, valve excluded.

    Counts the FLAG, not the label, so an inflated case is counted as inflated
    whether or not it also fell inside the preservation tolerance.
    """
    kept = [item for item in classifications if not item[UNDETECTED_REF]]
    numerator = sum(1 for item in kept if item[flag])
    denominator = len(kept)
    rate = numerator / denominator if denominator else float("nan")
    return float(rate), int(numerator), int(denominator)


# ------------------------------------------------------------------- octiles
def octile_stats(img: np.ndarray, l_a: np.ndarray, annulus: np.ndarray,
                 roi: np.ndarray | None = None, n_bins: int = N_OCTILES):
    """Per-octile contrast and z inside one ROI, ordered by path length.

    Stage 21 needs the detectability AS A FUNCTION of the physical
    perturbation; with six real instances a one-dimensional curve across
    anomalies would be dishonest, so the curve is built WITHIN each ROI by
    slicing its own path-length distribution into octiles. The background
    statistics are the ROI's own annulus, shared by every octile.

    The 2 m ray step quantizes L_a, so several octile edges can coincide on a
    small or a flat-topped ROI. Duplicate edges are removed and any bin left
    empty is merged into its neighbour, so a row never reports n_roi = 0 with
    NaN statistics; the EFFECTIVE number of bins travels in `n_octiles` and is
    what Stage 21 must quote, since it may be fewer than eight.
    """
    img = np.asarray(img, dtype=np.float64)
    l_a = np.asarray(l_a, dtype=np.float64)
    if roi is None:
        roi = roi_module.build_roi(l_a)[0]
    roi = np.asarray(roi, dtype=bool)
    annulus = np.asarray(annulus, dtype=bool)

    background = img[annulus]
    n_bg = background.size
    mean_bg = float(background.mean()) if n_bg else float("nan")
    std_bg = float(background.std(ddof=1)) if n_bg > 1 else float("nan")

    values = l_a[roi]
    rows = []
    if values.size == 0 or not np.isfinite(std_bg) or std_bg <= 0:
        return rows
    edges = _octile_edges(values, roi, l_a, n_bins)
    n_effective = edges.size - 1
    for index in range(n_effective):
        lo, hi = float(edges[index]), float(edges[index + 1])
        selection = roi & (l_a >= lo) & (l_a < hi)
        n_roi = int(selection.sum())
        mean_roi = float(img[selection].mean()) if n_roi else float("nan")
        difference = mean_roi - mean_bg
        rows.append({
            "octile": index, "n_octiles": int(n_effective), "n_roi": n_roi,
            "l_lo_m": lo, "l_hi_m": hi,
            "l_mean_m": float(l_a[selection].mean()) if n_roi else float("nan"),
            "C": (float(difference / mean_bg)
                  if n_roi and mean_bg else float("nan")),
            "z": (float(difference / (std_bg * np.sqrt(1.0 / n_roi
                                                       + 1.0 / n_bg)))
                  if n_roi else float("nan")),
        })
    return rows


def _octile_edges(values: np.ndarray, roi: np.ndarray, l_a: np.ndarray,
                  n_bins: int) -> np.ndarray:
    """Deduplicated quantile edges with empty bins merged into their neighbour.

    Returns at least two edges; the last one is nudged past the maximum so the
    half-open bins cover the whole ROI exactly once.
    """
    unique = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if unique.size < 2:
        unique = np.array([float(values.min())] * 2, dtype=np.float64)
    bounds = list(np.asarray(unique, dtype=np.float64))
    bounds[-1] = float(np.nextafter(bounds[-1], np.inf))

    def populated(lo, hi):
        return bool((roi & (l_a >= lo) & (l_a < hi)).any())

    merged = [bounds[0]]
    for index in range(1, len(bounds)):
        if index < len(bounds) - 1 and not populated(merged[-1], bounds[index]):
            continue                      # absorb the empty bin into the next
        merged.append(bounds[index])
    # a trailing empty bin cannot be absorbed forward: drop its lower edge
    while len(merged) > 2 and not populated(merged[-2], merged[-1]):
        merged.pop(-2)
    return np.asarray(merged, dtype=np.float64)


# ---------------------------------------------------------------- local SSIM
def local_roi_bbox(roi: np.ndarray, shape=None, margin: int = LOCAL_SSIM_MARGIN_PX):
    """Bounding box of the ROI grown by `margin`, clipped, at least 7x7."""
    box = roi_module.bounding_box(roi)
    if box is None:
        return None
    height, width = (shape if shape is not None else np.asarray(roi).shape)
    r0, r1, c0, c1 = box
    r0, r1 = max(0, r0 - margin), min(height, r1 + margin)
    c0, c1 = max(0, c0 - margin), min(width, c1 + margin)
    if r1 - r0 < LOCAL_SSIM_MIN_SIDE:
        r0 = max(0, min(r0, height - LOCAL_SSIM_MIN_SIDE))
        r1 = min(height, r0 + LOCAL_SSIM_MIN_SIDE)
    if c1 - c0 < LOCAL_SSIM_MIN_SIDE:
        c0 = max(0, min(c0, width - LOCAL_SSIM_MIN_SIDE))
        c1 = min(width, c0 + LOCAL_SSIM_MIN_SIDE)
    return r0, r1, c0, c1


def local_roi_ssim(pred: np.ndarray, ref: np.ndarray, roi: np.ndarray) -> float:
    """SSIM restricted to the ROI neighbourhood, under the frozen parameters.

    Global SSIM is dominated by the sky and by the smooth rock column; a
    method can win it while flattening the very structure the thesis is about.
    The local crop is the ROI bounding box grown by 8 px, so the 7x7 window
    still sees context on every side of the anomaly.
    """
    pred = np.asarray(pred)
    box = local_roi_bbox(roi, shape=pred.shape)
    if box is None:
        return float("nan")
    r0, r1, c0, c1 = box
    if r1 - r0 < LOCAL_SSIM_MIN_SIDE or c1 - c0 < LOCAL_SSIM_MIN_SIDE:
        return float("nan")
    return compute_ssim(pred[r0:r1, c0:c1], np.asarray(ref)[r0:r1, c0:c1])
