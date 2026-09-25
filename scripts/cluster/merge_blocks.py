#!/usr/bin/env python3
"""Merge completed campaign blocks into per-view histograms (stdlib only).

Counting histograms are additive, so the merge is an exact integer sum.
Writes <geo>_view<k>_hist.csv + merged .meta.json per view under
$MUOGRAPHY_SCRATCH/campaign-merged/, plus a campaign manifest CSV with
SHA-256 of every merged histogram. Refuses to merge a view whose block set
is incomplete unless --allow-partial is given.

Usage:
    python3 scripts/cluster/merge_blocks.py --blocks-per-view N [--allow-partial]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

N_VIEWS = 6
N_AZ = N_EL = 256


def load_hist(path: Path) -> list[list[int]]:
    with open(path, newline="") as fh:
        rows = [[int(v) for v in row] for row in csv.reader(fh)]
    assert len(rows) == N_EL and all(len(r) == N_AZ for r in rows), \
        f"bad shape in {path}"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks-per-view", type=int, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    scratch = Path(os.environ.get(
        "MUOGRAPHY_SCRATCH", f"/scratch/{os.environ['USER']}/muography"))
    campaign = scratch / "campaign"
    out_root = scratch / "campaign-merged"
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    incomplete = []

    for geo_dir in sorted(campaign.iterdir()):
        if not geo_dir.is_dir():
            continue
        geo = geo_dir.name
        for view in range(N_VIEWS):
            view_dir = geo_dir / f"view{view}"
            blocks = sorted(view_dir.glob("block[0-9][0-9][0-9][0-9]"))
            if len(blocks) < args.blocks_per_view:
                incomplete.append(f"{geo}/view{view}: "
                                  f"{len(blocks)}/{args.blocks_per_view}")
                if not args.allow_partial:
                    continue
            if not blocks:
                continue

            stem = f"{geo}_view{view}"
            total = [[0] * N_AZ for _ in range(N_EL)]
            events = draws = detected = 0
            wall = 0.0
            seeds = []
            for b in blocks:
                hist = load_hist(b / f"{stem}_hist.csv")
                for i in range(N_EL):
                    row = total[i]
                    brow = hist[i]
                    for j in range(N_AZ):
                        row[j] += brow[j]
                meta = json.loads((b / f"{stem}.meta.json").read_text())
                events += meta["events_transported"]
                draws += meta["ecomug_draws_total"]
                detected += meta["detected_in_bins"]
                wall += meta["wall_seconds"]
                seeds.append(meta["seed"])

            out_dir = out_root / geo
            out_dir.mkdir(exist_ok=True)
            hist_path = out_dir / f"{stem}_hist.csv"
            with open(hist_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerows(total)
            sha = hashlib.sha256(hist_path.read_bytes()).hexdigest()

            merged_meta = {
                "engine": "geant4 (merged blocks)",
                "geometry_id": geo, "view_index": view,
                "blocks_merged": len(blocks),
                "blocks_expected": args.blocks_per_view,
                "events_transported": events,
                "ecomug_draws_total": draws,
                "detected_in_bins": detected,
                "wall_seconds_total": wall,
                "block_seeds": seeds,
                "hist_sha256": sha,
                "merged_at": datetime.now(timezone.utc).isoformat(),
            }
            (out_dir / f"{stem}.meta.json").write_text(
                json.dumps(merged_meta, indent=1) + "\n")
            manifest_rows.append([geo, view, len(blocks), events,
                                  detected, sha])

    manifest = out_root / "campaign-manifest.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["geometry", "view", "blocks", "events_transported",
                    "detected", "hist_sha256"])
        w.writerows(manifest_rows)

    print(f"merged views: {len(manifest_rows)} -> {out_root}")
    if incomplete:
        print("INCOMPLETE:", *incomplete, sep="\n  ")
        if not args.allow_partial:
            sys.exit(1)


if __name__ == "__main__":
    main()
