#!/usr/bin/env python3
"""Stage 19 ceiling tests — synthetic rate fields only (no shards).

The real regeneration runs on the cluster during the dry run
(`python evaluation/ceiling.py --split val`) and must land on 0.905 +- 0.005.
Here the estimator is checked on rate fields whose answer is known by
construction: the ceiling must be finite, bounded by 1 and MONOTONE IN THE
RATE, because a brighter exposure is a smaller relative Poisson fluctuation.

Run:  python scripts/evaluation/test_ceiling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import ceiling as C  # noqa: E402

SIZE = 96


def _ceiling(lam_field, k=4, geo_index=0, view=0):
    ssim_values, _, _ = C.ceiling_for_view(lam_field, geo_index, view, k)
    return float(np.mean(ssim_values)), ssim_values


def test_monotone_in_rate() -> bool:
    """A higher counting rate must give a higher ceiling."""
    rates = [5.0, 50.0, 500.0, 5000.0]
    ceilings = [_ceiling(np.full((SIZE, SIZE), rate))[0] for rate in rates]
    checks = {
        "all finite": all(np.isfinite(v) for v in ceilings),
        "all inside (0, 1]": all(0.0 < v <= 1.0 for v in ceilings),
        "monotone in the rate": all(np.diff(ceilings) > 0),
        "a low rate is far from 1": ceilings[0] < 0.5,
        "a high rate approaches 1": ceilings[-1] > 0.9,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  monotone in rate: "
          f"{[f'{r:g}->{v:.4f}' for r, v in zip(rates, ceilings)]} -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_structure_raises_the_ceiling() -> bool:
    """Structure raises SSIM at a fixed mean rate: the covariance term grows."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    flat = np.full((SIZE, SIZE), 200.0)
    structured = 200.0 * (1.0 + 0.9 * np.sin(xx / 7.0) * np.cos(yy / 9.0))
    structured *= 200.0 / structured.mean()      # match the mean rate exactly
    flat_ceiling, _ = _ceiling(flat)
    structured_ceiling, _ = _ceiling(structured)
    checks = {
        "structured field scores higher": structured_ceiling > flat_ceiling,
        "the mean rate is matched":
            abs(structured.mean() - flat.mean()) < 1e-6,
        "both finite": np.isfinite(flat_ceiling) and np.isfinite(structured_ceiling),
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  structure: flat {flat_ceiling:.4f} vs structured "
          f"{structured_ceiling:.4f} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_reproducibility_and_guards() -> bool:
    lam = np.full((SIZE, SIZE), 300.0)
    first = C.ceiling_for_view(lam, 0, 0, 3)[0]
    again = C.ceiling_for_view(lam, 0, 0, 3)[0]
    other_view = C.ceiling_for_view(lam, 0, 1, 3)[0]
    other_geo = C.ceiling_for_view(lam, 1, 0, 3)[0]
    empty = C.ceiling_for_view(np.zeros((8, 8)), 0, 0, 3)[0]
    checks = {
        "the same seed reproduces": first == again,
        "a different view gives different draws": first != other_view,
        "a different geometry gives different draws": first != other_geo,
        "an all-zero field is refused, not crashed": empty == [],
        "K draws are returned": len(first) == 3,
        "the seed formula matches the protocol":
            C.seed_for(19, 1, 0) == 20260805 * 10 ** 6 + 19 * 10 ** 4 + 100,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  reproducibility: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("Stage 19 ceiling tests (synthetic rates; the val regeneration is "
          "cluster-only)")
    results = {
        "monotone_in_rate": test_monotone_in_rate(),
        "structure": test_structure_raises_the_ceiling(),
        "reproducibility": test_reproducibility_and_guards(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 19 CEILING TESTS:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
