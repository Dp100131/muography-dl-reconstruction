#!/usr/bin/env python3
"""Idempotent campaign launcher for SC3 (stdlib only — cluster-safe).

Reads the campaign plan (geometries x views x blocks), inspects which blocks
already exist and verify (block.sha256), and submits ONLY the missing or
corrupt ones via sbatch, honoring a submission cap. Run it as many times as
needed until 100% of blocks are published; it never duplicates work.

Seed derivation (documented, reproducible without numpy):
    seed(geo_idx, view, block) = MASTER*1_000_000 + geo_idx*10_000 + view*1_000 + block

Usage (from $MUOGRAPHY_SIM_HOME on guane):
    python3 scripts/cluster/submit_missing.py --events-per-view 60000000 \
        --events-per-block 2000000 [--max-submit 20] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

MASTER_SEED = 20260805
N_VIEWS = 6


def read_catalog(catalog_dir: Path) -> list[str]:
    geos = sorted(p.stem for p in catalog_dir.glob("geo-*.json"))
    if not geos:
        sys.exit(f"no geometries found in {catalog_dir}")
    return geos


def block_ok(block_dir: Path) -> bool:
    sha_file = block_dir / "block.sha256"
    if not sha_file.exists():
        return False
    try:
        line = sha_file.read_text().split()
        digest, fname = line[0], line[-1]
        target = block_dir / Path(fname).name
        if not target.exists():
            return False
        import hashlib
        h = hashlib.sha256(target.read_bytes()).hexdigest()
        return h == digest
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=30.0,
                        help="calibrated exposure per view (block = full view; "
                             "the app Poisson-samples N from the S0 tables)")
    parser.add_argument("--max-submit", type=int, default=4,
                        help="cap sbatch calls per invocation (QoS normal: "
                             "MaxSubmitPU=4 — rerun as the queue drains)")
    parser.add_argument("--partition", default="legacy",
                        help="SLURM partition (legacy: 24c, amd: 256c)")
    parser.add_argument("--cpus", type=int, default=None,
                        help="cpus-per-task (default: 24 legacy, 64 amd)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cpus = args.cpus or {"legacy": 24, "amd": 64}.get(args.partition, 8)

    home = Path(os.environ.get("MUOGRAPHY_SIM_HOME", Path.cwd()))
    scratch = Path(os.environ.get(
        "MUOGRAPHY_SCRATCH", f"/scratch/{os.environ['USER']}/muography"))
    catalog_dir = home / "scripts/simulation/geometry_catalog/geometries"
    geometries = read_catalog(catalog_dir)

    # Views already queued or running (job names b-<geoidx>-<view>): skipping
    # them prevents the in-flight resubmission race that wastes whole views.
    try:
        q = subprocess.run(["squeue", "-h", "-u", os.environ["USER"],
                            "-o", "%j"], capture_output=True, text=True)
        in_flight = set(q.stdout.split())
    except Exception:
        in_flight = set()

    blocks_per_view = 1  # one full-view job per block (fits QoS walls easily)
    total = len(geometries) * N_VIEWS * blocks_per_view
    done = missing = submitted = 0

    (home / "logs").mkdir(exist_ok=True)

    for gi, geo in enumerate(geometries):
        for view in range(N_VIEWS):
            view_dir = scratch / "campaign" / geo / f"view{view}"
            for blk in range(blocks_per_view):
                bdir = view_dir / f"block{blk:04d}"
                if bdir.exists() and block_ok(bdir):
                    done += 1
                    continue
                job_name = f"b-{gi:02d}-{view}"
                if job_name in in_flight:
                    continue  # someone is already computing this view
                missing += 1
                if submitted >= args.max_submit:
                    continue
                seed = (MASTER_SEED * 1_000_000 + gi * 10_000
                        + view * 1_000 + blk)
                env = (f"GEO={geo},VIEW={view},BLOCK={blk},"
                       f"DAYS={args.days},SEED={seed},CHAIN=1")
                cmd = ["sbatch", f"--partition={args.partition}",
                       f"--cpus-per-task={cpus}",
                       f"--job-name={job_name}",
                       "--export=ALL," + env,
                       str(home / "scripts/cluster/campaign_block.sbatch")]
                if args.dry_run:
                    print("DRY:", " ".join(cmd))
                else:
                    # At the QoS submit limit sbatch fails — stop this round;
                    # the next finishing chained job will refill the queue.
                    r = subprocess.run(cmd, cwd=home)
                    if r.returncode != 0:
                        print("sbatch refused (submit limit?) — stopping round")
                        submitted = args.max_submit
                        continue
                submitted += 1

    print(json.dumps({
        "geometries": len(geometries), "blocks_per_view": blocks_per_view,
        "total_blocks": total, "completed": done, "missing": missing,
        "submitted_now": submitted,
        "progress_pct": round(100 * done / total, 2),
    }, indent=1))


if __name__ == "__main__":
    main()
