#!/usr/bin/env python3
"""Claim-based queue worker for the Phase C grids (execution plan 2.10).

The QoS allows 4 submitted jobs, so a 12-configuration grid cannot be one job
per configuration. Two of these workers run instead (one per GPU of felix),
each pulling the next unclaimed configuration out of the grid until nothing is
left. The claim is a directory: os.mkdir is atomic on NFS, which is the whole
reason the protocol is built around it (same pattern as the Phase B campaign
launcher, scripts/cluster/submit_missing.py).

    python scripts/training/run_queue.py --grid scripts/training/configs/e16_grid.json \
        --phase e16 [--reclaim-stale-minutes 30] [--dry-run]

Per configuration:

  <runs>/<run_id>/.claim/             the claim itself (mkdir = the lock)
  <runs>/<run_id>/.claim/owner.json   host, job id, worker, utc, failures
  <runs>/<run_id>/.claim/heartbeat    touched before the run starts
  <runs>/<run_id>/.claim/abandoned    written after MAX_FAILURES failed tries

A configuration is DONE when its tracked record carries a summary.json whose
stop_reason is terminal (early_stop / epochs / max_iters). stop_reason
"stop_file" is NOT terminal: it means the job was signalled before its wall
clock and checkpointed, so the claim is released and the run continues in the
NEXT worker the chain driver submits (--requeue only covers node failure and
preemption; a job that ends at its time limit is not requeued by SLURM).

Liveness is not the heartbeat alone -- a real run takes hours, and a claim
whose only proof of life were a file touched at launch would be reclaimed by
the other worker mid-training. The freshness of a claim is therefore the most
recent of: the heartbeat, the run's metrics.jsonl and its last.pt.

Two files coordinate with the batch script, both under $MUOGRAPHY_RUNS:

  .worker-<w>-current   the run id this worker is training right now, so the
                        SIGUSR1 trap stops THAT run and not the other
                        worker's run
  .worker-<w>-stop      written by the trap; the loop reads it and exits
                        instead of starting a configuration it cannot finish

Exit codes:  0 nothing left to do for this worker
             3 ran at least one configuration, others are claimed elsewhere
             1 a configuration failed and the grid could not be advanced
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

# The completion semantics live in ONE place: the driver and the worker must
# never disagree about what "done" means.
from chain_driver import (  # noqa: E402
    RECORDS_DIR, is_complete, outcome, phase_of_run, prune_run_checkpoints,
    read_record, runs_root, train_command, utc_now)

CLAIM_DIR = ".claim"
MAX_FAILURES = 2  # after this many failed tries the config is left for a human
CLAIM_VERIFY_SECONDS = 2.0


def load_grid(path: Path) -> dict:
    grid = json.loads(path.read_text(encoding="utf-8"))
    for key in ("base_config", "runs"):
        if key not in grid:
            raise KeyError(f"{path}: grid is missing {key!r}")
    return grid


def claim_paths(run_dir: Path) -> tuple:
    claim = run_dir / CLAIM_DIR
    return claim, claim / "owner.json", claim / "heartbeat", claim / "abandoned"


def last_activity(run_dir: Path) -> float:
    """Newest mtime among the claim heartbeat and the run's own output."""
    _, _, heartbeat, _ = claim_paths(run_dir)
    candidates = [heartbeat, run_dir / "metrics.jsonl",
                  run_dir / "checkpoints" / "last.pt"]
    stamps = [path.stat().st_mtime for path in candidates if path.exists()]
    return max(stamps) if stamps else 0.0


def read_owner(owner_path: Path) -> dict:
    try:
        return json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def epochs_written(run_id: str) -> int:
    """Epochs the run has actually logged — the only evidence of progress."""
    record = read_record(RECORDS_DIR, run_id)
    return len(record["metrics"]) if record else 0


def strike_needed(epochs_before: int, epochs_after: int) -> bool:
    """Does a failed run consume one of its MAX_FAILURES tries?

    Only when it produced NOTHING. A run killed at the wall clock after
    training for hours has made progress and is resumable: charging it a
    strike would abandon a healthy configuration after two time limits, which
    is the normal way a 2-day job ends when a level needs longer than that.
    """
    return epochs_after <= epochs_before


class Worker:
    def __init__(self, grid_path: Path, phase: str, worker: str,
                 reclaim_stale_minutes: float, dry_run: bool = False):
        self.grid_path = grid_path
        self.grid = load_grid(grid_path)
        self.phase = phase
        self.worker = worker
        self.stale_seconds = reclaim_stale_minutes * 60.0
        self.dry_run = dry_run
        self.root = runs_root()
        self.identity = {
            "host": socket.gethostname(),
            "job_id": os.environ.get("SLURM_JOB_ID", "none"),
            "worker": worker,
            "pid": os.getpid(),
        }
        self.ran = []
        self.failed = []
        self.pruned = []
        # --dry-run trains nothing, so is_done() never flips: without this the
        # loop would re-claim the same configuration forever.
        self.visited = set()
        self.current_path = self.root / f".worker-{worker}-current"
        self.stop_path = self.root / f".worker-{worker}-stop"

    # ------------------------------------------------------ signal handshake
    def set_current(self, run_id) -> None:
        """Publish the run the SIGUSR1 trap of the batch script must stop."""
        self.root.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            self.current_path.unlink(missing_ok=True)
        else:
            self.current_path.write_text(run_id + "\n", encoding="utf-8")

    def stop_requested(self) -> bool:
        return self.stop_path.exists()

    # ------------------------------------------------------------------ claim
    def is_done(self, run_id: str) -> bool:
        return is_complete(read_record(RECORDS_DIR, run_id))

    def try_claim(self, run_id: str) -> bool:
        """os.mkdir is the lock; a stale claim is torn down and re-taken."""
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        claim, owner_path, heartbeat, abandoned = claim_paths(run_dir)

        if abandoned.exists():
            return False

        failures = 0
        if claim.exists():
            age = time.time() - last_activity(run_dir)
            if age < self.stale_seconds:
                return False
            owner = read_owner(owner_path)
            # Carried across the reclaim on purpose: a configuration that keeps
            # dying must reach MAX_FAILURES instead of resetting its counter
            # every time the claim goes stale.
            failures = int(owner.get("failures", 0))
            print(f"reclaiming {run_id}: no activity for {age / 60:.1f} min "
                  f"(previous owner {owner.get('host')}/{owner.get('job_id')}, "
                  f"failures {failures})", flush=True)
            shutil.rmtree(claim, ignore_errors=True)

        try:
            os.mkdir(claim)
        except FileExistsError:
            return False  # the other worker won the race
        except OSError as error:
            print(f"cannot claim {run_id}: {error}", flush=True)
            return False

        payload = dict(self.identity, run_id=run_id, phase=self.phase,
                       claimed_utc=utc_now(), failures=failures)
        owner_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        heartbeat.touch()

        # The reclaim path (rmtree + mkdir) has a narrow window in which two
        # workers could both end up inside the same claim. Re-reading the owner
        # after a short pause closes it: the loser sees someone else's identity
        # and steps away instead of training the same run twice.
        time.sleep(CLAIM_VERIFY_SECONDS)
        current = read_owner(owner_path)
        if current.get("pid") != self.identity["pid"] or \
                current.get("host") != self.identity["host"]:
            print(f"claim on {run_id} lost to {current.get('host')}/"
                  f"{current.get('job_id')} — skipping", flush=True)
            return False
        return True

    def release(self, run_id: str) -> None:
        shutil.rmtree(self.root / run_id / CLAIM_DIR, ignore_errors=True)

    def record_failure(self, run_id: str, detail: str) -> None:
        run_dir = self.root / run_id
        claim, owner_path, heartbeat, abandoned = claim_paths(run_dir)
        owner = read_owner(owner_path)
        failures = int(owner.get("failures", 0)) + 1
        owner.update({"failures": failures, "last_failure_utc": utc_now(),
                      "last_failure": detail})
        claim.mkdir(parents=True, exist_ok=True)
        owner_path.write_text(json.dumps(owner, indent=2), encoding="utf-8")
        heartbeat.touch()
        if failures >= MAX_FAILURES:
            abandoned.write_text(
                json.dumps({"run_id": run_id, "failures": failures,
                            "utc": utc_now(), "detail": detail}, indent=2),
                encoding="utf-8")
            print(f"ABANDONED {run_id} after {failures} failed attempts: "
                  f"{detail}. Delete {abandoned} once the cause is fixed.",
                  flush=True)
        else:
            print(f"{run_id} failed ({detail}); attempt {failures} of "
                  f"{MAX_FAILURES}, the claim goes stale in "
                  f"{self.stale_seconds / 60:.0f} min and can be retried.",
                  flush=True)

    # -------------------------------------------------------------- execution
    def execute(self, entry: dict) -> str:
        """Run one configuration. Returns done | stopped | failed."""
        run_id = entry["run_id"]
        self.visited.add(run_id)
        command = train_command(self.grid["base_config"], entry)
        print(f"=== {self.phase} {run_id} (worker {self.worker}) "
              f"{utc_now()}\n    " + " ".join(command), flush=True)
        if self.dry_run:
            self.release(run_id)
            return "done"

        before = epochs_written(run_id)
        started = time.time()
        self.set_current(run_id)
        try:
            proc = subprocess.run(command, cwd=str(REPO_ROOT))
        finally:
            self.set_current(None)
        elapsed = (time.time() - started) / 60.0
        after = epochs_written(run_id)
        record = read_record(RECORDS_DIR, run_id)
        oc = outcome(record)
        print(f"--- {run_id}: rc={proc.returncode} "
              f"stop_reason={oc['stop_reason']} "
              f"best_val_ssim={oc['best_val_ssim']} "
              f"epochs {before}->{after} ({elapsed:.1f} min)", flush=True)

        if oc["complete"]:
            self.ran.append(run_id)
            self.prune(run_id)
            self.release(run_id)
            return "done"
        if oc["stop_reason"] == "stop_file":
            # The job was signalled: the run is checkpointed and continues in
            # the next worker, so the claim is handed back immediately.
            self.ran.append(run_id)
            self.release(run_id)
            return "stopped"
        self.failed.append(run_id)
        if strike_needed(before, after):
            self.record_failure(run_id, f"exit {proc.returncode}, "
                                        f"stop_reason={oc['stop_reason']}, "
                                        f"no new epoch")
        else:
            # Progress was made and the checkpoint is on disk: this is a wall
            # clock or a node death, not a broken configuration. Hand the claim
            # back without spending a strike.
            print(f"{run_id} died after writing {after - before} epoch(s) — "
                  "resumable, no strike charged", flush=True)
            self.release(run_id)
        return "failed"

    def prune(self, run_id: str) -> None:
        """Free the checkpoints a finished run no longer needs.

        /home is a 20 GB hard quota and a checkpoint pair is ~780 MB: without
        this, twelve Stage 16 arms alone would fill it and the runs after them
        would die mid-checkpoint on "Disk quota exceeded". best.pt survives
        only where it is cited downstream (see KEEP_BEST_PHASES).
        """
        result = prune_run_checkpoints(self.root / run_id,
                                       phase_of_run(run_id))
        if result["removed"]:
            self.pruned.append(result)
            print(f"pruned {run_id}: {', '.join(result['removed'])} "
                  f"({result['freed_mb']:.0f} MB freed)", flush=True)

    # ------------------------------------------------------------------- loop
    def run(self) -> int:
        entries = self.grid["runs"]
        print(f"queue worker {self.worker}: phase={self.phase} "
              f"grid={self.grid_path.name} configs={len(entries)} "
              f"runs_root={self.root}", flush=True)
        # Leftovers from the previous job of this worker: a stale stop file
        # would end the round immediately, a stale current file would point
        # the signal trap at a run nobody is training.
        self.root.mkdir(parents=True, exist_ok=True)
        self.stop_path.unlink(missing_ok=True)
        self.set_current(None)

        while True:
            if self.stop_requested():
                print("stop requested before claiming a new configuration — "
                      "exiting cleanly", flush=True)
                break
            pending = [entry for entry in entries
                       if entry["run_id"] not in self.visited
                       and not self.is_done(entry["run_id"])]
            if not pending:
                break
            claimed = None
            for entry in pending:
                if self.try_claim(entry["run_id"]):
                    claimed = entry
                    break
            if claimed is None:
                print("nothing claimable left for this worker "
                      f"({len(pending)} config(s) held elsewhere or abandoned)",
                      flush=True)
                break
            result = self.execute(claimed)
            if result == "stopped":
                print("stop requested: exiting so the job can be requeued",
                      flush=True)
                break

        self.set_current(None)
        remaining = [entry["run_id"] for entry in entries
                     if not self.is_done(entry["run_id"])]
        print(json.dumps({
            "phase": self.phase, "worker": self.worker, "ran": self.ran,
            "failed": self.failed, "remaining": remaining,
            "pruned_mb": round(sum(item["freed_mb"] for item in self.pruned), 1),
        }), flush=True)
        if not remaining:
            return 0
        if self.failed and not self.ran:
            return 1
        return 3 if self.ran else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", required=True,
                        help="grid JSON (scripts/training/configs/e16_grid.json, ...)")
    parser.add_argument("--phase", required=True, help="e16 | e17 | e18")
    parser.add_argument("--worker", default=os.environ.get("WORKER", "a"))
    parser.add_argument("--reclaim-stale-minutes", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="claim and print the commands, train nothing")
    args = parser.parse_args()

    grid_path = Path(args.grid)
    if not grid_path.is_absolute():
        grid_path = REPO_ROOT / grid_path
    worker = Worker(grid_path, args.phase, args.worker,
                    args.reclaim_stale_minutes, dry_run=args.dry_run)
    return worker.run()


if __name__ == "__main__":
    sys.exit(main())
