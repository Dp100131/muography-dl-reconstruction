#!/usr/bin/env python3
"""Stage 13 dataset tests — loader contract, split guard and throughput.

NumPy checks always run. The checks that need a torch Dataset are skipped with
a printed note when torch is absent (the framework acta is still pending).

Run:  python scripts/training/test_dataset.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

import muography_dataset as mds  # noqa: E402

EXPECTED_SPLIT_SIZES = {"train": 12, "val": 3, "test": 3, "generalization": 1}
MIN_PAIRS_PER_SECOND = 100.0
VAL_GEOMETRY = "geo-val-01"

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def test_split_guard() -> bool:
    """Held-out splits must refuse to resolve without the explicit opt-in."""
    saved = os.environ.pop(mds.ALLOW_TEST_ENV, None)
    ok = True
    try:
        for split in mds.HELD_OUT_SPLITS:
            try:
                mds.geometry_ids_for_split(split)
            except RuntimeError as exc:
                if "Stage 20" not in str(exc):
                    ok = False
                    print(f"  split guard {split!r}: message lacks the "
                          f"single-use warning -> FAIL")
            else:
                ok = False
                print(f"  split guard {split!r}: did NOT raise -> FAIL")
        # train/val must stay reachable without the opt-in
        for split in ("train", "val"):
            mds.geometry_ids_for_split(split)
    finally:
        if saved is not None:
            os.environ[mds.ALLOW_TEST_ENV] = saved
    if ok:
        print("  split guard: test/generalization raise, train/val open -> ok")
    return ok


def test_no_leakage() -> bool:
    """Splits are disjoint by geometry and have the frozen sizes."""
    saved = os.environ.get(mds.ALLOW_TEST_ENV)
    os.environ[mds.ALLOW_TEST_ENV] = "1"  # names only, no pixel is read
    try:
        sets = {name: set(mds.geometry_ids_for_split(name))
                for name in EXPECTED_SPLIT_SIZES}
    finally:
        if saved is None:
            os.environ.pop(mds.ALLOW_TEST_ENV, None)
        else:
            os.environ[mds.ALLOW_TEST_ENV] = saved

    ok = True
    for name, expected in EXPECTED_SPLIT_SIZES.items():
        if len(sets[name]) != expected:
            ok = False
            print(f"  no-leakage: split {name} has {len(sets[name])} "
                  f"geometries, expected {expected} -> FAIL")
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sets[a] & sets[b]
            if shared:
                ok = False
                print(f"  no-leakage: {a} and {b} share {sorted(shared)} -> FAIL")

    pairs = json.loads(
        (REPO_ROOT / "results" / "dataset" / "splits.json").read_text(encoding="utf-8")
    )["pairs_per_split"]
    if ok:
        print(f"  no-leakage: 12/3/3/1 geometries, disjoint; pairs "
              f"{pairs['train']}/{pairs['val']}/{pairs['test']}/"
              f"{pairs['generalization']} -> ok")
    return ok


def test_loader_throughput() -> bool:
    """Materialized shard access must sustain >= 100 pairs/s (Stage 13.0)."""
    data, _ = mds.load_shard(VAL_GEOMETRY)
    n_views, n_fracs, k_max = data["low_counts"].shape[:3]
    rng = np.random.default_rng(20260805130000)
    picks = [(int(rng.integers(n_views)), int(rng.integers(n_fracs)),
              int(rng.integers(k_max))) for _ in range(200)]

    t0 = time.perf_counter()
    for vi, fi, k in picks:
        mds.preprocess_pair(data["low_counts"][vi, fi, k], data["ref_counts"][vi])
    dt = time.perf_counter() - t0

    rate = len(picks) / dt
    ok = rate >= MIN_PAIRS_PER_SECOND
    print(f"  loader throughput: {rate:.1f} pairs/s "
          f"({1000.0 * dt / len(picks):.2f} ms/pair, n={len(picks)}) "
          f"-> {'ok' if ok else 'FAIL (< %.0f/s)' % MIN_PAIRS_PER_SECOND}")
    return ok


def test_corrected_amplitude() -> bool:
    """1/f rescaling must bring the input amplitude onto the reference."""
    data, _ = mds.load_shard(VAL_GEOMETRY)
    ref = data["ref_counts"][0]
    rng = np.random.default_rng(20260805130001)
    ok = True
    # log1p concavity biases the rescaled mean at very low counts (Jensen);
    # at f=0.01 (mean ~4 counts/px) the systematic gap is ~0.09, not noise.
    tolerance = {0.50: 0.05, 0.10: 0.05, 0.05: 0.05, 0.01: 0.12}
    for fraction in (0.50, 0.10, 0.05, 0.01):
        low = rng.binomial(ref.astype(np.int64), fraction)
        x_low, x_ref = mds.preprocess_pair_corrected(low, ref, fraction)
        rel = abs(float(x_low.mean()) - float(x_ref.mean())) / float(x_ref.mean())
        if rel > tolerance[fraction]:
            ok = False
            print(f"  corrected amplitude f={fraction:.2f}: relative mean gap "
                  f"{rel:.4f} > {tolerance[fraction]} -> FAIL")
        else:
            print(f"  corrected amplitude f={fraction:.2f}: "
                  f"mean(x_low)={x_low.mean():.4f} vs mean(x_ref)="
                  f"{x_ref.mean():.4f} (rel {rel:.4f}) -> ok")
    return ok


def test_shapes_and_dtypes() -> bool:
    if not HAVE_TORCH:
        print("  shapes/dtypes: SKIPPED (torch not installed)")
        return True
    ds = mds.MuographyPairs(geometry_ids=[VAL_GEOMETRY], fractions=[0.05])
    x_low, x_ref = ds[0]
    ok = (tuple(x_low.shape) == (1, 256, 256)
          and tuple(x_ref.shape) == (1, 256, 256)
          and x_low.dtype == torch.float32 and x_ref.dtype == torch.float32
          and len(ds) == 6 * 20)
    print(f"  shapes/dtypes: {tuple(x_low.shape)} {x_low.dtype}, "
          f"len={len(ds)} (expected 120) "
          f"max(x_ref)={float(x_ref.max()):.4f} -> {'ok' if ok else 'FAIL'}")
    return ok


def test_hflip_axis() -> bool:
    """hflip mirrors azimuth (last axis) only, and only sometimes."""
    if not HAVE_TORCH:
        print("  hflip axis: SKIPPED (torch not installed)")
        return True
    plain = mds.MuographyPairs(geometry_ids=[VAL_GEOMETRY], fractions=[0.05])
    flipped = mds.MuographyPairs(geometry_ids=[VAL_GEOMETRY], fractions=[0.05],
                                 hflip=True, seed=20260805130002)
    n_flipped = n_kept = 0
    ok = True
    for i in range(40):
        a_low, a_ref = plain[i]
        b_low, b_ref = flipped[i]
        if torch.equal(a_low, b_low) and torch.equal(a_ref, b_ref):
            n_kept += 1
            continue
        azimuth = torch.flip(a_low, [-1])
        elevation = torch.flip(a_low, [-2])
        if not (torch.equal(azimuth, b_low)
                and torch.equal(torch.flip(a_ref, [-1]), b_ref)):
            ok = False
            print(f"  hflip axis: item {i} is neither identity nor an "
                  f"azimuth mirror -> FAIL")
            break
        if torch.equal(elevation, b_low) and not torch.equal(elevation, azimuth):
            ok = False
            print(f"  hflip axis: item {i} matches an ELEVATION flip -> FAIL")
            break
        n_flipped += 1
    if ok and (n_flipped == 0 or n_kept == 0):
        ok = False
        print(f"  hflip axis: degenerate draw (flipped={n_flipped}, "
              f"kept={n_kept}) -> FAIL")
    if ok:
        print(f"  hflip axis: azimuth-only mirror, {n_flipped} flipped / "
              f"{n_kept} kept out of 40 -> ok")
    return ok


def test_rethin_draw() -> bool:
    """On-the-fly re-thinning (Stage 16 c11): a fresh input, the same target."""
    if not HAVE_TORCH:
        print("  rethin draw: SKIPPED (torch not installed)")
        return True
    common = {"geometry_ids": [VAL_GEOMETRY], "fractions": [0.05],
              "input_scaling": mds.REF_EQUIVALENT}
    stored = mds.MuographyPairs(**common)
    redrawn = mds.MuographyPairs(**common, rethin=True, seed=20260805161100)
    twin = mds.MuographyPairs(**common, rethin=True, seed=20260805161100)

    a_low, a_ref = stored[0]
    b_low, b_ref = redrawn[0]
    t_low, _ = twin[0]
    redrawn.set_epoch(1)
    c_low, c_ref = redrawn[0]

    # A redraw is another realization of the SAME binomial process, so the
    # reference-equivalent amplitude must land where the stored realization
    # does; a wrong parameterization (raw counts, 1/f applied twice) would
    # move this mean by far more than the tolerance.
    delta = abs(float(b_low.mean()) - float(a_low.mean()))
    checks = {
        "input differs from the stored realization":
            not torch.equal(a_low, b_low),
        "target is untouched": torch.equal(a_ref, b_ref)
                               and torch.equal(b_ref, c_ref),
        "the draw varies with the epoch": not torch.equal(b_low, c_low),
        "same seed and epoch reproduce the draw": torch.equal(b_low, t_low),
        "amplitude matches the stored draw": delta < 0.01,
    }
    ok = all(checks.values())
    print(f"  rethin draw: |mean(x_low_rethin) - mean(x_low_stored)| = "
          f"{delta:.4f} -> {'ok' if ok else 'FAIL'}")
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    return ok


def main() -> int:
    print("Stage 13 dataset tests"
          + ("" if HAVE_TORCH else "  [torch missing: Dataset checks skipped]"))
    results = {
        "split_guard": test_split_guard(),
        "no_leakage": test_no_leakage(),
        "loader_throughput": test_loader_throughput(),
        "corrected_amplitude": test_corrected_amplitude(),
        "shapes_and_dtypes": test_shapes_and_dtypes(),
        "hflip_axis": test_hflip_axis(),
        "rethin_draw": test_rethin_draw(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 13 DATASET TESTS:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
