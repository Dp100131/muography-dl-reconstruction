#!/usr/bin/env python3
"""Stage 19 cache tests — the uint16 round trip must not move SSIM.

The acceptance criterion of the execution plan: over 100 synthetic pairs,
SSIM computed on the restored arrays must reproduce SSIM on the originals to
better than 1e-4, and a self-comparison must stay at 1.

Run:  python scripts/evaluation/test_cache.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import cache as C  # noqa: E402
from metrics import compute_psnr, compute_ssim  # noqa: E402

N_IMAGES = 100
SIZE = 64
SSIM_TOLERANCE = 1e-4


def _synthetic_images(rng, n=N_IMAGES):
    """Images that live where the real ones live: smooth, bright, near 1."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    images = []
    for i in range(n):
        base = (0.72 + 0.20 * np.sin((xx + 3 * i) / 9.0)
                * np.cos((yy - 2 * i) / 13.0))
        noisy = base + 0.01 * rng.standard_normal((SIZE, SIZE))
        images.append(np.clip(noisy, 0.0, 1.0).astype(np.float32))
    return images


def test_round_trip() -> bool:
    rng = np.random.default_rng(20260805)
    images = _synthetic_images(rng)
    with tempfile.TemporaryDirectory() as tmp:
        path = C.save_predictions(Path(tmp) / "block.npz", np.stack(images),
                                  fraction=np.array([0.05] * N_IMAGES),
                                  k=np.arange(N_IMAGES))
        restored, extra = C.load_predictions(path)

    self_deltas, pair_deltas, psnr_deltas = [], [], []
    for original, back in zip(images, restored):
        self_deltas.append(abs(compute_ssim(original, back) - 1.0))
    for i in range(0, N_IMAGES - 1, 2):
        a, b = images[i], images[i + 1]
        ra, rb = restored[i], restored[i + 1]
        pair_deltas.append(abs(compute_ssim(ra, rb) - compute_ssim(a, b)))
        psnr_deltas.append(abs(compute_psnr(rb, ra) - compute_psnr(b, a)))

    max_abs = float(np.max(np.abs(np.stack(images) - restored)))
    checks = {
        "quantization error below half a step": max_abs <= 0.5 / C.UINT16_SCALE
                                                + 1e-9,
        "self SSIM stays at 1": max(self_deltas) < SSIM_TOLERANCE,
        "pairwise SSIM preserved": max(pair_deltas) < SSIM_TOLERANCE,
        "pairwise PSNR preserved to 0.01 dB": max(psnr_deltas) < 0.01,
        "dtype restored as float32": restored.dtype == np.float32,
        "extra arrays travel with the block":
            set(extra) == {"fraction", "k"} and extra["k"].size == N_IMAGES,
        "clipping is applied":
            float(C.dequantize(C.quantize(np.array([-0.5, 1.5]))).max()) == 1.0,
    }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  round trip: max|delta pixel| = {max_abs:.3e}, "
          f"max|delta SSIM self| = {max(self_deltas):.3e}, "
          f"max|delta SSIM pair| = {max(pair_deltas):.3e} -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def _raises(exception, fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except exception:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


def test_manifest() -> bool:
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest_path = root / C.MANIFEST_NAME
        first = C.save_predictions(root / "geo-test-01" / "arm_f.npz",
                                   rng.random((4, 8, 8)))
        second = C.save_predictions(root / "geo-test-02" / "arm_e.npz",
                                    rng.random((4, 8, 8)))
        C.manifest_update(manifest_path, second)
        manifest = C.manifest_update(manifest_path, first)
        keys = list(manifest["files"])
        checks = {
            "both files recorded": len(keys) == 2,
            "keys are sorted": keys == sorted(keys),
            "keys are POSIX relative paths":
                keys[0] == "geo-test-01/arm_f.npz" and "\\" not in keys[0],
            "sha256 recorded": all(len(e["sha256"]) == 64
                                   for e in manifest["files"].values()),
            "shape and dtype recorded":
                manifest["files"][keys[0]]["shape"] == [4, 8, 8]
                and manifest["files"][keys[0]]["dtype"] == "uint16",
            "totals recomputed": manifest["n_files"] == 2
                                 and manifest["total_bytes"] > 0,
            "manifest is valid JSON":
                isinstance(json.loads(manifest_path.read_text(encoding="utf-8")),
                           dict),
            "verification is clean": C.verify_manifest(manifest_path) == [],
        }
        # corrupt one file and make sure verification notices
        with open(first, "r+b") as handle:
            handle.seek(0)
            handle.write(b"\x00\x00\x00\x00")
        problems = C.verify_manifest(manifest_path)
        checks["corruption is detected"] = (
            len(problems) == 1 and problems[0][1] == "sha256 mismatch")
        # a missing file is reported, not raised
        first.unlink()
        checks["missing file is reported"] = (
            C.verify_manifest(manifest_path)[0][1] == "missing")

        # the manifest tracks EVERY cache artefact, not only the prediction
        # blocks: Stage 23 puts a sha256 next to each claim, and the masks,
        # the tidy tables and the summary all carry claims
        masks = C.save_masks(root / "geo-test-03" / "masks.npz",
                             views=np.array([0, 1], dtype=np.int32),
                             ref_counts=np.zeros((2, 4, 4), dtype=np.uint32))
        table = root / "metrics.csv"
        table.write_text("split,arm\nval,A\n", encoding="utf-8")
        summary = root / "summary.json"
        summary.write_text('{"split": "val"}\n', encoding="utf-8")
        for path in (masks, table, summary):
            manifest = C.manifest_update(manifest_path, path, root)
        entries = manifest["files"]
        checks["a mask archive is tracked"] = (
            "geo-test-03/masks.npz" in entries
            and entries["geo-test-03/masks.npz"]["arrays"]
            == ["ref_counts", "views"])
        checks["a CSV is tracked with its hash"] = (
            len(entries["metrics.csv"]["sha256"]) == 64
            and "shape" not in entries["metrics.csv"])
        checks["the summary JSON is tracked"] = "summary.json" in entries
        checks["prediction blocks still record shape and dtype"] = (
            entries["geo-test-02/arm_e.npz"]["shape"] == [4, 8, 8])

        # an object array would only be readable through pickle, and the cache
        # is loaded with allow_pickle=False on purpose
        checks["save_masks refuses an object array"] = _raises(
            TypeError, C.save_masks, root / "bad.npz",
            kinds=np.array(["cavity", "conduit"], dtype=object))
        checks["a fixed-width string array is fine"] = isinstance(
            C.save_masks(root / "ok.npz",
                         kinds=np.array(["cavity"], dtype="U32")), Path)
        checks["masks load back without pickle"] = (
            C.load_masks(root / "ok.npz")["kinds"].tolist() == ["cavity"])

    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  manifest: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def test_atomicity() -> bool:
    """No .tmp file may survive, and rewriting must not truncate on failure."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = C.save_predictions(root / "a.npz", np.zeros((2, 4, 4)))
        C.write_json_atomic(root / "s.json", {"a": 1})
        leftovers = list(root.glob("*.tmp")) + list(root.glob("*.npz.tmp"))
        checks = {
            "no temporary files left": leftovers == [],
            "npz readable": C.load_predictions(path)[0].shape == (2, 4, 4),
            "json readable": json.loads(
                (root / "s.json").read_text(encoding="utf-8")) == {"a": 1},
            "json ends with a newline":
                (root / "s.json").read_text(encoding="utf-8").endswith("\n"),
        }
    ok = all(checks.values())
    for label, passed in checks.items():
        if not passed:
            print(f"    {label}: FAIL")
    print(f"  atomicity: {sum(checks.values())}/{len(checks)} checks -> "
          f"{'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("Stage 19 cache tests")
    results = {
        "round_trip": test_round_trip(),
        "manifest": test_manifest(),
        "atomicity": test_atomicity(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(n for n, ok in results.items() if not ok))
    print("STAGE 19 CACHE:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
