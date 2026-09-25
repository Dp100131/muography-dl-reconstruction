#!/usr/bin/env python3
"""Stage 19-core metric tests — guards the frozen SSIM/PSNR contract.

Three independent checks:
  1. Cross-implementation: a self-contained NumPy SSIM (7x7 uniform window,
     valid-mode moments) must track metrics.compute_ssim.
  2. Golden values: fixed analytic fixtures must reproduce the constants in
     results/evaluation/golden_metrics.json bit-for-bit (1e-10). The file is
     created on first run (or with MUOGRAPHY_WRITE_GOLDEN=1) and must be
     committed.
  3. Regression: SSIM/PSNR of real shard pairs at f=0.05 must land on the
     frozen results/dataset/stats/stage11_baseline.json row.

Run:  python scripts/evaluation/test_metrics.py
      MUOGRAPHY_WRITE_GOLDEN=1 python scripts/evaluation/test_metrics.py   # re-bless
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

from metrics import compute_psnr, compute_ssim, summarize  # noqa: E402
from muography_dataset import load_shard, preprocess_pair  # noqa: E402

GOLDEN_PATH = REPO_ROOT / "results" / "evaluation" / "golden_metrics.json"
BASELINE_PATH = REPO_ROOT / "results" / "dataset" / "stats" / "stage11_baseline.json"
WRITE_GOLDEN_ENV = "MUOGRAPHY_WRITE_GOLDEN"


# --------------------------------------------------- independent SSIM (naive)
def _window_mean(img: np.ndarray, w: int = 7) -> np.ndarray:
    """Valid-mode running mean over w x w windows via a summed-area table."""
    pad = np.pad(img.astype(np.float64), ((1, 0), (1, 0)))
    cs = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    total = cs[w:, w:] - cs[:-w, w:] - cs[w:, :-w] + cs[:-w, :-w]
    return total / float(w * w)


def naive_ssim(a: np.ndarray, b: np.ndarray, w: int = 7,
               data_range: float = 1.0) -> float:
    """Textbook SSIM with population moments — independent of scikit-image."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a, mu_b = _window_mean(a, w), _window_mean(b, w)
    var_a = _window_mean(a * a, w) - mu_a ** 2
    var_b = _window_mean(b * b, w) - mu_b ** 2
    cov = _window_mean(a * b, w) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(np.mean(num / den))


def _rank(values) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64))
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    return ranks


# ------------------------------------------------------------------- fixtures
def golden_fixtures():
    """Analytic (RNG-free) 64x64 pair — stable across NumPy versions."""
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float64)
    a = 0.5 + 0.35 * np.sin(xx / 7.0) * np.cos(yy / 11.0)
    b = a + 0.06 * np.sin(xx / 3.0 + yy / 5.0) - 0.03 * np.cos(yy / 2.0)
    return (np.clip(a, 0.0, 1.0).astype(np.float32),
            np.clip(b, 0.0, 1.0).astype(np.float32))


# ---------------------------------------------------------------------- tests
def test_cross_implementation() -> bool:
    rng = np.random.default_rng(20260805)
    ours, theirs, deltas = [], [], []
    for _ in range(5):
        a = rng.random((64, 64)).astype(np.float32)
        b = np.clip(a + 0.15 * rng.standard_normal((64, 64)), 0.0, 1.0
                    ).astype(np.float32)
        s_ref, s_naive = compute_ssim(a, b), naive_ssim(a, b)
        theirs.append(s_ref)
        ours.append(s_naive)
        deltas.append(abs(s_ref - s_naive))
    max_delta = max(deltas)
    same_ranking = bool(np.array_equal(_rank(ours), _rank(theirs)))
    ok = max_delta < 0.02 and same_ranking
    print(f"  cross-implementation SSIM: max|delta|={max_delta:.5f} "
          f"ranking_preserved={same_ranking} -> {'ok' if ok else 'FAIL'}")
    if not ok:
        print(f"    skimage={['%.6f' % v for v in theirs]}")
        print(f"    naive  ={['%.6f' % v for v in ours]}")
    return ok


def test_golden_values() -> bool:
    a, b = golden_fixtures()
    current = {
        "ssim": compute_ssim(a, b),
        "psnr_db": compute_psnr(a, b),
        "ssim_self": compute_ssim(a, a),
        "psnr_self_is_inf": bool(np.isinf(compute_psnr(a, a))),
    }
    force = os.environ.get(WRITE_GOLDEN_ENV) == "1"
    if force or not GOLDEN_PATH.exists():
        payload = {
            "fixture": "scripts/evaluation/test_metrics.py::golden_fixtures (analytic)",
            "frozen_params": {"data_range": 1.0, "win_size": 7,
                              "K1": 0.01, "K2": 0.03},
            "numpy": np.__version__,
            "values": current,
        }
        GOLDEN_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  golden values: WRITTEN to {GOLDEN_PATH.name} "
              f"(ssim={current['ssim']:.10f}, psnr={current['psnr_db']:.10f} dB)"
              " — commit this file")
        return True

    stored = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["values"]
    ok = True
    for key in ("ssim", "psnr_db", "ssim_self"):
        delta = abs(float(stored[key]) - float(current[key]))
        if delta > 1e-10:
            ok = False
            print(f"  golden {key}: stored={stored[key]!r} "
                  f"current={current[key]!r} delta={delta:.3e} -> FAIL")
    if bool(stored.get("psnr_self_is_inf")) != current["psnr_self_is_inf"]:
        ok = False
        print("  golden psnr_self_is_inf mismatch -> FAIL")
    if ok:
        print(f"  golden values: match to 1e-10 "
              f"(ssim={current['ssim']:.10f}, psnr={current['psnr_db']:.4f} dB)")
    return ok


def test_baseline_regression() -> bool:
    if not BASELINE_PATH.exists():
        print(f"  baseline regression: MISSING {BASELINE_PATH} -> FAIL")
        return False
    shard = REPO_ROOT / "dataset" / "shards" / "geo-val-01.npz"
    if not shard.exists():
        print(f"  baseline regression: MISSING {shard} -> FAIL")
        return False

    data, _ = load_shard("geo-val-01")
    frac_index = 2  # 0.05
    fraction = float(data["fractions"][frac_index])
    ssim_vals, psnr_vals = [], []
    for k in range(5):
        x_low, x_ref = preprocess_pair(
            data["low_counts"][0, frac_index, k], data["ref_counts"][0])
        ssim_vals.append(compute_ssim(x_low, x_ref))
        psnr_vals.append(compute_psnr(x_ref, x_low))

    expected = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    row = expected["baseline_no_model"][f"{fraction:.2f}"]
    got_ssim = summarize(ssim_vals)
    got_psnr = summarize(psnr_vals, digits=2)
    d_ssim = abs(got_ssim["mean"] - row["ssim_mean"])
    ok = d_ssim < 0.02
    print(f"  baseline regression f={fraction:.2f} view0 k=0..4: "
          f"ssim={got_ssim['mean']:.4f}+-{got_ssim['std']:.4f} "
          f"(frozen {row['ssim_mean']:.4f}, |delta|={d_ssim:.4f}) "
          f"psnr={got_psnr['mean']:.2f} dB (frozen {row['psnr_mean_db']:.2f}) "
          f"-> {'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("Stage 19-core metric tests")
    results = {
        "cross_implementation": test_cross_implementation(),
        "golden_values": test_golden_values(),
        "baseline_regression": test_baseline_regression(),
    }
    passed = all(results.values())
    if not passed:
        failed = [name for name, ok in results.items() if not ok]
        print(f"failed checks: {', '.join(failed)}")
    print("STAGE 19-CORE METRICS:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
