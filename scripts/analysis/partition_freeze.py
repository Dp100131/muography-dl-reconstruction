#!/usr/bin/env python3
"""Stage 12 — dataset partitioning, leakage check, hashing and freeze.

Split BY GEOMETRY (every realization of a geometry stays in one split),
generalization geometry fully excluded, automated no-leakage asserts,
SHA-256 of every shard, splits.json + freeze manifest for the git tag.

Run:  python scripts/analysis/partition_freeze.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SHARD_DIR = REPO_ROOT / "dataset" / "shards"
OUT_DIR = REPO_ROOT / "results" / "dataset"

SPLIT_OF_PREFIX = {"geo-train-": "train", "geo-val-": "val",
                   "geo-test-": "test", "geo-gen-": "generalization"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-utc", default=None,
                        help="preserve an existing freeze timestamp instead of now()")
    args = parser.parse_args()

    shards = sorted(SHARD_DIR.glob("*.npz"))
    assert len(shards) == 19, f"expected 19 shards, found {len(shards)}"

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": [],
                                    "generalization": []}
    hashes = {}
    pair_counts = {}
    for shard in shards:
        gid = shard.stem
        split = next(v for k, v in SPLIT_OF_PREFIX.items() if gid.startswith(k))
        splits[split].append(gid)
        hashes[gid] = sha256_of(shard)
        meta = json.loads(
            (SHARD_DIR / f"{gid}.meta.json").read_text(encoding="utf-8"))
        pair_counts[gid] = (meta["n_views"] * len(meta["fractions"])
                            * meta["k_realizations"])

    # ---------------- no-leakage asserts (the acceptance criterion)
    all_ids = [g for ids in splits.values() for g in ids]
    assert len(all_ids) == len(set(all_ids)), "LEAKAGE: geometry in >1 split"
    for a in splits:
        for b in splits:
            if a < b:
                inter = set(splits[a]) & set(splits[b])
                assert not inter, f"LEAKAGE between {a}/{b}: {inter}"
    assert splits["generalization"] == ["geo-gen-01"], "generalization set wrong"
    assert len(splits["train"]) == 12 and len(splits["val"]) == 3 \
        and len(splits["test"]) == 3

    n_train = sum(pair_counts[g] for g in splits["train"])
    freeze = {
        "dataset_version": "v1.0",
        "engine": "geant4-11.4.2 (SC3 campaign, 114/114 views, table-driven "
                  "generator, binomial thinning)",
        "frozen_utc": args.frozen_utc or datetime.now(
            timezone.utc).isoformat(timespec="seconds"),
        "split_policy": "by geometry (all realizations of a geometry share its split)",
        "splits": splits,
        "pairs_per_split": {s: sum(pair_counts[g] for g in ids)
                            for s, ids in splits.items()},
        "train_pairs_criterion_ge_5000": bool(n_train >= 5000),
        "shard_sha256": hashes,
        "pending_for_oe1_close": [
            "firma de cierre OE1 por director y codirector (PA-4)",
            "respaldo externo real (disco externo + OneDrive institucional)",
        ],
    }
    (OUT_DIR / "splits.json").write_text(json.dumps(freeze, indent=2),
                                         encoding="utf-8")
    print(json.dumps({k: v for k, v in freeze.items() if k != "shard_sha256"},
                     indent=2))
    print("STAGE 12 PARTITION/FREEZE: PASSED (no leakage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
