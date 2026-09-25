"""Frozen image-quality metrics — the single measurement interface of Phase C/D.

Parameters are FROZEN (protocol section 5.1) and must never be re-tuned:

    SSIM: skimage.metrics.structural_similarity(a, b, data_range=1.0)
          -> win_size=7, uniform (box) window, K1=0.01, K2=0.03
    PSNR: skimage.metrics.peak_signal_noise_ratio(ref, x, data_range=1.0)

These are exactly the calls that produced
results/dataset/stats/stage11_baseline.json, so every Phase C number stays
comparable with the frozen no-model baseline. Do NOT enable gaussian_weights:
the Wang 11x11 gaussian variant differs by <= 0.002 and consistency with the
frozen table wins over convention.

Stage 19 layers bootstrap confidence intervals, paired tests and ROI-restricted
detectability ON TOP of these wrappers — it does not change them.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DATA_RANGE = 1.0
WIN_SIZE = 7
K1 = 0.01
K2 = 0.03


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM between two [0, 1] images under the frozen parameters."""
    # Parameters passed explicitly so the frozen contract does not depend on
    # skimage defaults staying what they are today.
    return float(structural_similarity(a, b, data_range=DATA_RANGE,
                                       win_size=WIN_SIZE, K1=K1, K2=K2,
                                       gaussian_weights=False))


def compute_psnr(ref: np.ndarray, x: np.ndarray) -> float:
    """PSNR [dB] of x against the reference ref, both in [0, 1]."""
    return float(peak_signal_noise_ratio(ref, x, data_range=DATA_RANGE))


def summarize(values, digits: int = 4) -> dict:
    """{n, mean, std} with the sample standard deviation (ddof=1).

    Rounded the same way scripts/analysis/build_dataset.py rounds its baseline
    table so aggregated JSON stays byte-comparable across scripts.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), digits),
        "std": round(std, digits),
    }
