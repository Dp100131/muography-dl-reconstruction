"""Stage 19 — residual texture metrics (the pre-registered 1 % arbiter).

Stage 18 committed to two texture diagnostics for the level where SSIM cannot
adjudicate between the cWGAN and the U-Net L1 (the gap is -0.0028, about
1.9 sigma): the radially averaged power spectrum of the residual and the
variance ratio inside the anomaly mask. A purely L1 generator regresses to the
conditional mean, which suppresses high-frequency residual power and drives
the variance ratio below 1; an adversarial generator restores texture, which
is either fidelity or hallucination -- the ROI detectability rates of Stage 21
say which, and these two scalars say how much.

`nyquist_band_ratio` reproduces scripts/training/engine.py::nyquist_ratio with
the same 0.45 / 0.40 band constants, so the Stage 20 numbers stay comparable
with the per-iteration checkerboard diagnostic logged during training. It is
duplicated rather than imported because engine.py imports torch at module
level and the evaluation modules must run on a CPU box without it.
"""

from __future__ import annotations

import numpy as np

N_RADIAL_BINS = 64
MAX_NORMALIZED_FREQUENCY = float(np.sqrt(2.0))   # the spectrum corner
NYQUIST_RING = 0.45      # engine.nyquist_ratio default
NYQUIST_FLOOR = 0.40     # engine.nyquist_ratio default


def _normalized_radius(height: int, width: int) -> np.ndarray:
    """Radius in units of Nyquist, matching engine.nyquist_ratio exactly."""
    fy = np.fft.fftshift(np.fft.fftfreq(height)) / 0.5
    fx = np.fft.fftshift(np.fft.fftfreq(width)) / 0.5
    return np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def power_spectrum(residual: np.ndarray) -> np.ndarray:
    """|FFT(residual - mean)|^2, fftshifted. The mean is removed so bin 0 is
    not a DC spike that swamps the low-frequency end of the radial profile."""
    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim == 4:
        residual = residual[:, 0]
    if residual.ndim == 3:
        residual = residual.mean(axis=0)
    residual = residual - residual.mean()
    spectrum = np.fft.fftshift(np.fft.fft2(residual))
    return np.abs(spectrum) ** 2


def radial_power_spectrum(residual: np.ndarray, n_bins: int = N_RADIAL_BINS):
    """(bin centres, mean power) over normalized frequency in [0, sqrt(2)].

    Bins are uniform in radius, so the outer bins hold many more modes than
    the inner ones -- exactly what a radially averaged spectrum should do. An
    empty bin returns NaN rather than 0 so it can never be mistaken for a
    measured null.
    """
    power = power_spectrum(residual)
    radius = _normalized_radius(*power.shape)
    edges = np.linspace(0.0, MAX_NORMALIZED_FREQUENCY, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    index = np.digitize(radius.ravel(), edges) - 1
    flat = power.ravel()
    profile = np.full(n_bins, np.nan)
    for b in range(n_bins):
        selection = index == b
        if selection.any():
            profile[b] = float(flat[selection].mean())
    return centres, profile


def variance_ratio(pred: np.ndarray, ref: np.ndarray, mask=None) -> float:
    """Var(pred[mask]) / Var(ref[mask]) with the sample variance (ddof=1).

    < 1 means the prediction is smoother than the reference (regression to the
    mean); > 1 means it carries more variance than the truth. Both are
    failures of a different kind, so the metric is reported signed, never as
    |1 - ratio|.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        pred, ref = pred[mask], ref[mask]
    pred, ref = pred.ravel(), ref.ravel()
    if pred.size < 2 or ref.size < 2:
        return float("nan")
    denominator = float(ref.var(ddof=1))
    if denominator <= 0.0:
        return float("nan")
    return float(pred.var(ddof=1) / denominator)


def nyquist_band_ratio(residual: np.ndarray, ring: float = NYQUIST_RING,
                       floor: float = NYQUIST_FLOOR) -> float:
    """Mean power above `ring` x Nyquist over the floor below `floor`.

    Mirror of scripts/training/engine.py::nyquist_ratio (same constants, same
    normalization) so Stage 20 continues the training-time series. The one
    difference is that this version removes the mean first, which cannot
    change the result: the DC mode sits at radius 0, inside the floor band but
    with a weight of one mode out of thousands.
    """
    power = power_spectrum(residual)
    radius = _normalized_radius(*power.shape)
    band = radius >= ring
    base = radius < floor
    if not band.any() or not base.any():
        return float("nan")
    denominator = float(power[base].mean())
    if denominator <= 0.0:
        return float("nan")
    return float(power[band].mean() / denominator)


def texture_scalars(pred: np.ndarray, ref: np.ndarray, mask=None) -> dict:
    """The three numbers Stage 21 stores per (pair, method) in the tidy table."""
    residual = np.asarray(pred, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    centres, profile = radial_power_spectrum(residual)
    high = centres >= NYQUIST_RING
    low = centres < NYQUIST_FLOOR
    with np.errstate(invalid="ignore"):
        high_power = float(np.nanmean(profile[high])) if high.any() else float("nan")
        low_power = float(np.nanmean(profile[low])) if low.any() else float("nan")
    return {
        "variance_ratio": variance_ratio(pred, ref, mask),
        "nyquist_band_ratio": nyquist_band_ratio(residual),
        "residual_high_freq_power": high_power,
        "residual_low_freq_power": low_power,
        "residual_std": float(residual.std(ddof=1)) if residual.size > 1
        else float("nan"),
    }
