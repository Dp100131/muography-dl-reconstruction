#!/usr/bin/env python3
"""Convert muography_sim CSV+meta output into the campaign NPZ contract.

The surrogate campaign stores per view:
    NPZ:  counts_ref (uint32, 256x256)  [+ lam/transmission/opacity, surrogate-only]
    JSON: sidecar metadata

Geant4 output has real transported counts, so only counts_ref is stored; the
dataset builder subsamples Geant4 counts by binomial thinning (exact for
event subsampling) instead of Poisson thinning of lam.

Usage:
    python csv_to_npz.py --hist geo-x_view0_hist.csv --out-dir data/campaign-geant4/geo-x
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

N_AZ = N_EL = 256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hist", required=True,
                        help="path to <stem>_hist.csv (meta sidecar expected next to it)")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    hist_path = Path(args.hist)
    stem = hist_path.name.removesuffix("_hist.csv")
    meta_path = hist_path.with_name(stem + ".meta.json")

    counts = np.loadtxt(hist_path, delimiter=",", dtype=np.uint64)
    assert counts.shape == (N_EL, N_AZ), f"bad histogram shape {counts.shape}"
    assert counts.max() < 2**32, "counts overflow uint32"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["npz_contract"] = "counts_ref only (real transported events); "\
                           "subsample with binomial thinning"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, counts_ref=counts.astype(np.uint32))

    sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    meta["npz_sha256"] = sha
    (out_dir / f"{stem}.meta.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    total = int(counts.sum())
    print(f"{npz_path} written: total_counts={total} sha256={sha[:16]}...")


if __name__ == "__main__":
    main()
