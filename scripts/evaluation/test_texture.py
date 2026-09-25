#!/usr/bin/env python3
"""Stage 19 texture tests — white noise, low-passed noise, identity.

Run:  python scripts/evaluation/test_texture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import texture as T  # noqa: E402

SIZE = 256


def _box_filter(field: np.ndarray, radius: int) -> np.ndarray:
    """Circular box average — a low-pass with no scipy dependency."""
    out = np.zeros_like(field)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out += np.roll(np.roll(field, dy, axis=0), dx, axis=1)
    return out / float((2 * radius + 1) ** 2)


def test_white_noise_is_flat() -> bool:
    """A white field must give a radial spectrum with a small coefficient of
    variation across bins (the plan's 'flat spectrum' acceptance)."""
    rng = np.random.default_rng(20260805)
    field = rng.standard_normal((SIZE, SIZE))
    centres, profile = T.radial_power_spectrum(field)
    valid = np.isfinite(profile)
    cv = float(np.nanstd(profile[valid]) / np.nanmean(profile[valid]))
    checks = {
        "64 bins": profile.size == 64,
        "bins span [0, sqrt(2)]": (centres[0] > 0.0
                                   and centres[-1] < T.MAX_NORMALIZED_FREQUENCY),
        "coefficient of variation < 0.2": cv < 0.20,
        "no bin is zero": bool(np.nanmin(profile[valid]) > 0.0),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  white noise: CV across bins = {cv:.4f} "
          f"({int(valid.sum())} populated bins) -> {'ok' if ok else 'FAIL'}")
    return ok


def test_lowpass_cuts_high_frequencies() -> bool:
    """A 5x5 box filter must drop the high-frequency power by more than 5x."""
    rng = np.random.default_rng(1234)
    field = rng.standard_normal((SIZE, SIZE))
    smoothed = _box_filter(field, 2)

    _, white = T.radial_power_spectrum(field)
    centres, low = T.radial_power_spectrum(smoothed)
    high_band = centres >= 0.5
    ratio = float(np.nanmean(white[high_band]) / np.nanmean(low[high_band]))
    ring_white = T.nyquist_band_ratio(field)
    ring_low = T.nyquist_band_ratio(smoothed)
    checks = {
        "high-frequency power drops by more than 5x": ratio > 5.0,
        "low-frequency power is preserved":
            0.5 < float(np.nanmean(low[centres < 0.1])
                        / np.nanmean(white[centres < 0.1])) < 2.0,
        "nyquist band ratio falls": ring_low < ring_white,
        "white ring ratio is near unity": 0.5 < ring_white < 2.0,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  low-pass: high-band attenuation = {ratio:.1f}x, "
          f"nyquist ratio {ring_white:.3f} -> {ring_low:.4f} "
          f"-> {'ok' if ok else 'FAIL'}")
    return ok


def test_variance_ratio() -> bool:
    rng = np.random.default_rng(7)
    ref = rng.random((64, 64))
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:40, 10:40] = True
    checks = {
        "identical images give exactly 1":
            abs(T.variance_ratio(ref, ref, mask) - 1.0) < 1e-12,
        "no mask also gives 1": abs(T.variance_ratio(ref, ref) - 1.0) < 1e-12,
        "a scaled prediction gives the squared factor":
            abs(T.variance_ratio(0.5 * ref, ref, mask) - 0.25) < 1e-12,
        "an over-smoothed prediction falls below 1":
            T.variance_ratio(_box_filter(ref, 2), ref, mask) < 1.0,
        "a constant reference is NaN":
            not np.isfinite(T.variance_ratio(ref, np.zeros((64, 64)), mask)),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  variance ratio: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_texture_scalars() -> bool:
    """The bundle must be finite and must separate smooth from textured."""
    rng = np.random.default_rng(99)
    ref = 0.5 + 0.05 * rng.standard_normal((128, 128))
    smooth = _box_filter(ref, 3)
    noisy = ref + 0.05 * rng.standard_normal((128, 128))
    a = T.texture_scalars(smooth, ref)
    b = T.texture_scalars(noisy, ref)
    checks = {
        "all scalars finite": all(np.isfinite(v) for v in a.values())
                              and all(np.isfinite(v) for v in b.values()),
        "smoothing lowers the variance ratio": a["variance_ratio"] < 0.9,
        "added noise raises the variance ratio": b["variance_ratio"] > 1.1,
        "keys are stable": set(a) == {"variance_ratio", "nyquist_band_ratio",
                                      "residual_high_freq_power",
                                      "residual_low_freq_power", "residual_std"},
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  texture scalars: smooth vr={a['variance_ratio']:.3f} "
          f"noisy vr={b['variance_ratio']:.3f} -> {'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("Stage 19 texture tests")
    results = {
        "white_noise_flat": test_white_noise_is_flat(),
        "lowpass_cut": test_lowpass_cuts_high_frequencies(),
        "variance_ratio": test_variance_ratio(),
        "texture_scalars": test_texture_scalars(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 19 TEXTURE:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
