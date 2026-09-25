#!/usr/bin/env python3
"""Phase C chain driver — the mechanical state machine for Stages 15 to 18.

It runs on a CPU node (stdlib only; no torch), looks at what the records
already say, and takes exactly ONE step of the pre-registered Phase C
protocol. It is idempotent: running it twice in a row changes nothing the
second time, and it never submits a job whose name is already in the queue.

    python scripts/training/chain_driver.py            # decide and act
    python scripts/training/chain_driver.py --dry-run  # decide and print, no writes

What it decides ON ITS OWN (every threshold below is pre-registered, none of
them is chosen here):

  E15  an attempt PASSES with best val SSIM > 0.7777 (the best classical bar
       at 5%), is STABLE-BUT-NOT-SUPERIOR in (0.770, 0.7777], and is VETOED if
       the divergence check fails. Any pass closes the stage; otherwise all
       launched attempts must finish and at least one must be stable. Up to
       three attempts (seed reps 0/1/2); a stalled attempt is resubmitted with
       --resume auto, which continues the same run rather than starting a new
       one.
  E16  12 OFAT configs through two queue workers; then the selection rule:
       adopt an axis when its best val SSIM beats the accepted E15 result by
       more than 0.003, best score wins inside an axis (PSNR breaks ties), any
       diverged run is vetoed. Writes e17_final.json and e16_results.csv with
       every number that produced the decision.
  E17  4 levels + 2 extra 5% seeds from the selected config, then the SHA-256
       weights manifest.
  E18  the three pre-registered ablations (lambda_L1 = 0, no skips, depth 4).

Where it STOPS and asks for a human:

  * three E15 attempts finished and none is even stable  -> HALTED
  * the conditioned (log f) model of E17: the conditioning channel is not
    implemented in the dataset or the engine, so the driver records the tau
    gate, prints a warning and DEFERS it (the plan's literature clause covers
    the omission; training it needs a dated addendum).
  * anything the pre-registration does not decide mechanically -- the
    checkerboard criterion and the "no sustained collapse" criterion have no
    pre-registered numeric threshold, so their numbers are REPORTED in the
    tables and the log, never turned into an automatic veto.

Nothing here touches dataset/shards, deletes a record, or opens the test split.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDS_DIR = REPO_ROOT / "results" / "training" / "records"
TABLES_DIR = REPO_ROOT / "results" / "training"
CONFIG_DIR = REPO_ROOT / "scripts" / "training" / "configs"
BENCH_PATH = REPO_ROOT / "results" / "training" / "bench-felix.json"
LOG_PATH = REPO_ROOT / "results" / "training" / "chain-driver.log"
STATE_PATH = TABLES_DIR / "phase_c_chain_state.json"
MANIFEST_PATH = REPO_ROOT / "models" / "weights-manifest.json"

BASE_CONFIG = CONFIG_DIR / "e15_reference.json"
E16_GRID = CONFIG_DIR / "e16_grid.json"
E17_GRID = CONFIG_DIR / "e17_runs.json"
E17_FINAL = CONFIG_DIR / "e17_final.json"
E18_GRID = CONFIG_DIR / "e18_runs.json"

E15_SBATCH = "scripts/cluster/e15_initial.sbatch"
QUEUE_SBATCH = "scripts/cluster/queue_worker.sbatch"

# ----------------------------------------------------------- frozen constants
MASTER_SEED = 20260805
# stop_reason values the engine writes when a run has FINISHED. "stop_file" is
# not among them: it means the job was signalled 300 s before its wall clock,
# checkpointed and exited, and the run must be resumed.
TERMINAL_STOP_REASONS = ("early_stop", "epochs", "max_iters")

E15_ATTEMPTS = 3
E15_RUN_IDS = tuple(f"e15-c0-attempt{n}" for n in range(1, E15_ATTEMPTS + 1))
E15_JOB_NAMES = {f"e15-c0-attempt{n}": f"muo-e15-a{n}"
                 for n in range(1, E15_ATTEMPTS + 1)}
E15_PASS_SSIM = 0.7777      # best classical bar at 5% (frozen input)
# Pre-registration: "si el mejor SSIM queda en 0,775-0,778 [...] E15 se declara
# 'estable pero no superior'". The floor is 0.775, not a rounder number.
E15_STABLE_FLOOR = 0.775
E16_ADOPT_MARGIN = 0.003    # provisional 1-sigma inter-seed threshold
GP_MEAN_MAX = 1.0           # E15 criterion 1: mean gradient penalty < 1
DIVERGENCE_WINDOW = 10      # epochs averaged for the gp / out_std readings
# E15 criterion 3 (checkerboard): nyquist_ratio "sin tendencia creciente
# sostenida por encima de ~2" -- both halves of the sentence gate, so the
# ratio must be above 2 AND still rising over the window.
NYQUIST_MAX = 2.0
TAU_MAX_CONDITIONED = 1.6   # s/iter gate for the conditioned E17 model
CONDITIONED_BATCH = 8
# A run id relaunched this many times without writing a single new epoch is
# not a transient failure: it is a crash before the first checkpoint, and no
# further resubmission will fix it.
MAX_LAUNCHES_WITHOUT_PROGRESS = 3
# /home is a 20 GB hard quota and one checkpoint pair is ~780 MB: the first
# campaign died with "OSError: Disk quota exceeded" mid-checkpoint. Below this
# headroom the chain prunes finished runs and, if that is not enough, stops
# instead of launching work that cannot save its own state.
MIN_HEADROOM_MB = 2500.0
# Stages whose best.pt is cited downstream and must survive pruning: e17 feeds
# models/weights-manifest.json and Stage 20; e15 keeps the restart material of
# the pre-registered R4 intervention ladder; e19 holds the retrained U-Net L1
# baselines that Stage 20 compares against (the e13 originals were pruned).
KEEP_BEST_PHASES = ("e15", "e17", "e19")

WORKERS = ("a", "b")
MAX_OWN_JOBS = 3  # driver + 2 GPU jobs; the 4th QoS slot stays free on purpose

# Submitting from inside a job leaks the DRIVER's allocation into the child:
# SLURM reads these from the environment and they take precedence over the
# #SBATCH lines of the submitted script, so a 4G/2-CPU driver would hand its
# GPU workers 4G and 2 CPUs and every one of them would die on the loader.
LEAKY_SLURM_VARS = (
    "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_CPUS_PER_TASK",
    "SLURM_NTASKS", "SLURM_NNODES", "SLURM_JOB_NUM_NODES", "SLURM_GRES",
    "SLURM_JOB_NAME", "SLURM_EXPORT_ENV", "SLURM_QOS",
)

# Fields whose non-finite value (the engine writes JSON null for NaN/Inf) means
# the run diverged. nyquist_ratio is deliberately NOT here: it is null whenever
# the low-frequency floor of the residual is empty, which is a degenerate
# reading, not a divergence.
FINITE_FIELDS = ("w_est", "gp", "l1", "adv", "d_real", "d_fake",
                 "grad_norm_g", "grad_norm_d", "out_std")

# override key -> command-line flag of scripts/training/train_cwgan_gp.py
OVERRIDE_FLAGS = {
    "fraction": "--fraction",
    "epochs": "--epochs",
    "batch": "--batch",
    "lambda_l1": "--lambda-l1",
    "lambda_adv": "--lambda-adv",
    "lambda_gp": "--lambda-gp",
    "lr": "--lr",
    "n_critic": "--n-critic",
    "val_every": "--val-every",
    "sample_every": "--sample-every",
    "patience": "--patience",
    "input_scaling": "--input-scaling",
    "hflip": "--hflip",
    "rethin": "--rethin",
    "generator": "--generator",
    "critic": "--critic",
}
BOOL_FLAGS = ("hflip", "rethin")


def seed_for(stage: int, config: int, rep: int = 0) -> int:
    """Protocol PA-1: seed = 20260805*10^6 + stage*10^4 + config*10^2 + rep."""
    return MASTER_SEED * 10 ** 6 + stage * 10 ** 4 + config * 10 ** 2 + rep


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================================
# Record layer (pure: filesystem in, dictionaries out)
# ============================================================================
def read_metrics(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn last line after a node death is not a failure
    return rows


def read_record(records_dir, run_id: str):
    """Return {"summary", "config", "metrics"} for one run, or None."""
    run_dir = Path(records_dir) / run_id
    if not run_dir.is_dir():
        return None
    summary = None
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = None
    config = None
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            config = None
    return {"run_id": run_id, "summary": summary, "config": config,
            "metrics": read_metrics(run_dir / "metrics.jsonl")}


def read_records(records_dir) -> dict:
    """All records under records_dir, keyed by run_id."""
    records_dir = Path(records_dir)
    if not records_dir.is_dir():
        return {}
    out = {}
    for child in sorted(records_dir.iterdir()):
        if child.is_dir():
            record = read_record(records_dir, child.name)
            if record is not None:
                out[child.name] = record
    return out


def divergence_report(metrics: list) -> dict:
    """E15 criterion 1, mechanized exactly as pre-registered.

    Three checks and no more: every logged quantity finite, mean gradient
    penalty of the last 10 epochs below 1, and val_ssim never NaN. The
    "no sustained collapse" clause has no pre-registered threshold, so
    out_std is measured and reported but never vetoes on its own.
    """
    report = {"ok": True, "reasons": [], "n_epochs": len(metrics),
              "gp_last10": None, "out_std_last10": None,
              "nyquist_last10": None}
    if not metrics:
        report["ok"] = False
        report["reasons"].append("no metrics")
        return report

    bad_fields = set()
    for row in metrics:
        for key in FINITE_FIELDS:
            if key in row and row[key] is None:
                bad_fields.add(key)
        if "val_ssim" in row and row["val_ssim"] is None:
            bad_fields.add("val_ssim")
    if bad_fields:
        report["ok"] = False
        report["reasons"].append(
            "non-finite: " + ",".join(sorted(bad_fields)))

    window = metrics[-DIVERGENCE_WINDOW:]

    def _mean(key):
        values = [row[key] for row in window
                  if isinstance(row.get(key), (int, float))]
        return (sum(values) / len(values)) if values else None

    report["gp_last10"] = _mean("gp")
    report["out_std_last10"] = _mean("out_std")
    report["nyquist_last10"] = _mean("nyquist_ratio")
    if report["gp_last10"] is not None and report["gp_last10"] >= GP_MEAN_MAX:
        report["ok"] = False
        report["reasons"].append(
            f"mean gp of last {len(window)} epochs = "
            f"{report['gp_last10']:.3f} >= {GP_MEAN_MAX}")
    return report


def _slope(values) -> float:
    """Ordinary least-squares slope of values against their index."""
    n = len(values)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    numerator = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(values))
    denominator = sum((i - mean_x) ** 2 for i in range(n))
    return numerator / denominator if denominator else 0.0


def checkerboard_report(metrics: list) -> dict:
    """E15 criterion 3: a SUSTAINED RISE of nyquist_ratio above ~2.

    Both conditions gate together. A high but flat ratio is a property of the
    residual, not a growing ConvTranspose artifact; a rising ratio well under 2
    is noise. Only "above 2 and still climbing" is the checkerboard signature
    the pre-registration asks to stop for.
    """
    values = [row["nyquist_ratio"] for row in metrics[-DIVERGENCE_WINDOW:]
              if isinstance(row.get("nyquist_ratio"), (int, float))]
    if not values:
        return {"nyquist_last10": None, "nyquist_slope": None,
                "n": 0, "flagged": False}
    mean = sum(values) / len(values)
    slope = _slope(values)
    return {"nyquist_last10": mean, "nyquist_slope": slope, "n": len(values),
            "flagged": bool(mean > NYQUIST_MAX and slope > 0.0)}


def outcome(record) -> dict:
    """One run reduced to the numbers the pre-registered rules read."""
    result = {"present": False, "complete": False, "stop_reason": None,
              "best_val_ssim": None, "best_val_psnr": None, "best_epoch": None,
              "epochs_run": None, "config_sha8": None, "diverged": False,
              "divergence": None, "checkerboard": None, "epochs_written": 0}
    if record is None:
        return result
    result["present"] = True
    summary = record.get("summary")
    metrics = record.get("metrics") or []
    report = divergence_report(metrics)
    result["divergence"] = report
    result["diverged"] = not report["ok"]
    result["checkerboard"] = checkerboard_report(metrics)
    result["epochs_written"] = len(metrics)
    if not summary:
        return result
    result["stop_reason"] = summary.get("stop_reason")
    result["config_sha8"] = summary.get("config_sha8")
    result["best_val_ssim"] = summary.get("best_val_ssim")
    result["best_epoch"] = summary.get("best_epoch")
    result["epochs_run"] = summary.get("epochs_run")
    # Completion is about the run ENDING, not about it succeeding: the engine
    # writes best_val_ssim null when every validation SSIM was NaN, and that
    # run is finished and diverged, not eternally pending. Requiring a score
    # here would leave the phase open forever and eventually halt the chain
    # over a run the divergence veto already handles.
    result["complete"] = summary.get("stop_reason") in TERMINAL_STOP_REASONS
    for row in metrics:
        if row.get("epoch") == result["best_epoch"]:
            result["best_val_psnr"] = row.get("val_psnr")
            break
    return result


def is_complete(record) -> bool:
    return outcome(record)["complete"]


# ============================================================================
# Stage 15 decision (pure)
# ============================================================================
def classify_e15(oc: dict) -> str:
    """absent | incomplete | diverged | checkerboard | below | stable | pass."""
    if not oc["present"]:
        return "absent"
    if not oc["complete"]:
        return "incomplete"
    if oc["diverged"]:
        return "diverged"
    best = oc["best_val_ssim"]
    if best is None:
        return "diverged"
    if (oc.get("checkerboard") or {}).get("flagged"):
        # Criterion 3 is an acceptance gate, not a score: a checkerboarding
        # generator is never accepted, however good its SSIM looks.
        return "checkerboard"
    if best > E15_PASS_SSIM:
        return "pass"
    if best > E15_STABLE_FLOOR:
        return "stable"
    return "below"


def launch_guard(run_id: str, launch_history, epochs_written: int) -> tuple:
    """May run_id be (re)launched, and what does its history become?

    The chain relaunches an attempt that has no terminal summary and no job in
    the queue. Without a counter that is a livelock: a crash before the first
    checkpoint leaves no record at all, so every pass sees the same state and
    resubmits forever. Progress is measured in EPOCHS WRITTEN, which is the
    only evidence a relaunch achieved anything.
    """
    prior = (launch_history or {}).get(run_id) or {}
    prior_count = int(prior.get("count", 0))
    prior_epochs = int(prior.get("epochs_at_launch", -1))
    progressed = epochs_written > prior_epochs
    entry = {"count": 1 if progressed else prior_count + 1,
             "epochs_at_launch": epochs_written,
             "progressed_since_last_launch": progressed,
             "last_launch_utc": utc_now()}
    blocked = (not progressed) and prior_count >= MAX_LAUNCHES_WITHOUT_PROGRESS
    return (not blocked), entry


def decide_e15(records: dict, in_flight_runs=frozenset(),
               launch_history=None) -> dict:
    """Pre-registered Stage 15 budget: <=3 attempts, seed reps 0/1/2.

    status is one of
      closed_pass    an attempt beat the classical bar -> proceed
      closed_stable  every launched attempt finished, at least one landed in
                     (0.770, 0.7777] -> "estable pero no superior", proceed
      wait           an attempt is still running or queued
      relaunch       an attempt has a record but no terminal summary and no job
                     in the queue: resubmit it (--resume auto continues it)
      launch         start the next attempt
      halt           three attempts finished and none is stable -> human
    """
    in_flight_runs = set(in_flight_runs)
    attempts = []
    for n, run_id in enumerate(E15_RUN_IDS, start=1):
        oc = outcome(records.get(run_id))
        attempts.append({
            "attempt": n, "run_id": run_id, "seed": seed_for(15, 0, n - 1),
            "job_name": E15_JOB_NAMES[run_id], "state": classify_e15(oc),
            "in_flight": run_id in in_flight_runs,
            "best_val_ssim": oc["best_val_ssim"], "outcome": oc,
        })

    accepted = [a for a in attempts if a["state"] in ("pass", "stable")]
    passed = [a for a in attempts if a["state"] == "pass"]
    stable = [a for a in attempts if a["state"] == "stable"]
    flagged = [a for a in attempts if a["state"] == "checkerboard"]
    active = [a for a in attempts if a["in_flight"]]
    stalled = [a for a in attempts
               if a["state"] == "incomplete" and not a["in_flight"]]
    launched = [a for a in attempts if a["state"] != "absent" or a["in_flight"]]
    c0 = max((a["best_val_ssim"] for a in accepted
              if a["best_val_ssim"] is not None), default=None)

    decision = {"phase": "e15", "attempts": attempts, "c0_result": c0,
                "accepted": [{"run_id": a["run_id"],
                              "best_val_ssim": a["best_val_ssim"]}
                             for a in accepted],
                "launch": None, "launch_entry": None,
                "status": None, "detail": ""}

    if flagged:
        # Checked BEFORE the pass branch on purpose: a checkerboarding run
        # must never close the stage, however high its SSIM.
        target = flagged[0]
        report = target["outcome"]["checkerboard"]
        decision["status"] = "halt"
        decision["detail"] = (
            f"{target['run_id']} trips E15 criterion 3 (checkerboard): "
            f"nyquist_ratio mean of the last {report['n']} epochs = "
            f"{report['nyquist_last10']:.3f} > {NYQUIST_MAX} and still rising "
            f"(slope {report['nyquist_slope']:+.4f}/epoch) with best val SSIM "
            f"{target['best_val_ssim']}. The R4 remedy (upsample+conv instead "
            "of ConvTranspose) changes the architecture and is a human "
            "decision: it must be logged in the research log.")
        return decision
    if passed:
        decision["status"] = "closed_pass"
        decision["detail"] = (
            f"{passed[0]['run_id']} best val SSIM "
            f"{passed[0]['best_val_ssim']} > {E15_PASS_SSIM}; "
            f"C0 result = {c0}")
        return decision
    # Relaunch comes BEFORE the wait branch: a dead attempt and a live one can
    # coexist, and the job budget has room for both (driver + 2 GPU jobs).
    if stalled:
        target = stalled[0]
        allowed, entry = launch_guard(target["run_id"], launch_history,
                                      target["outcome"]["epochs_written"])
        if not allowed:
            decision["status"] = "halt"
            decision["detail"] = (
                f"{target['run_id']} has been launched "
                f"{MAX_LAUNCHES_WITHOUT_PROGRESS} times without writing a "
                f"single new epoch (still at "
                f"{target['outcome']['epochs_written']} epochs). That is a "
                "crash before the first checkpoint, not a transient failure: "
                "read logs/muo-e15-a*.out, fix the cause and clear the "
                "launch_history entry in phase_c_chain_state.json to retry.")
            return decision
        decision["status"] = "relaunch"
        decision["launch"] = target
        decision["launch_entry"] = entry
        decision["detail"] = (
            f"{target['run_id']} has a record but no terminal summary and no "
            f"job in the queue: resubmitting with --resume auto "
            f"(launch {entry['count']} of {MAX_LAUNCHES_WITHOUT_PROGRESS} "
            f"without progress)" if not entry["progressed_since_last_launch"]
            else f"{target['run_id']} has a record but no terminal summary "
                 "and no job in the queue: resubmitting with --resume auto")
        return decision
    if active:
        decision["status"] = "wait"
        decision["detail"] = ("in flight: "
                              + ", ".join(a["run_id"] for a in active))
        return decision
    if stable:
        decision["status"] = "closed_stable"
        decision["detail"] = (
            "no attempt beat the classical bar; "
            + ", ".join(f"{a['run_id']}={a['best_val_ssim']}" for a in stable)
            + f" lands in ({E15_STABLE_FLOOR}, {E15_PASS_SSIM}] -> "
              f"'estable pero no superior', the chain proceeds; C0 result = {c0}")
        return decision
    pending = [a for a in attempts if a["state"] == "absent" and not a["in_flight"]]
    if len(launched) < E15_ATTEMPTS and pending:
        target = pending[0]
        allowed, entry = launch_guard(target["run_id"], launch_history,
                                      target["outcome"]["epochs_written"])
        if not allowed:
            decision["status"] = "halt"
            decision["detail"] = (
                f"{target['run_id']} has been submitted "
                f"{MAX_LAUNCHES_WITHOUT_PROGRESS} times and has never written "
                "an epoch (no record at all): the job is dying before "
                "Trainer.__init__. Read logs/muo-e15-a*.out — the usual causes "
                "are a full quota, a missing shard and a bad module load.")
            return decision
        decision["status"] = "launch"
        decision["launch"] = target
        decision["launch_entry"] = entry
        decision["detail"] = (
            f"attempt {target['attempt']} of {E15_ATTEMPTS} "
            f"(seed rep {target['attempt'] - 1}); previous: "
            + ", ".join(f"{a['run_id']}={a['state']}" for a in launched))
        return decision
    decision["status"] = "halt"
    decision["detail"] = (
        f"{len(launched)} attempts finished and none is stable ("
        + ", ".join(f"{a['run_id']}={a['state']}" for a in attempts)
        + "). The pre-registered budget is exhausted: the R4 intervention "
          "ladder (critic lr -> n_critic -> lambda_L1 -> restart from best.pt) "
          "is a human decision and must be logged in the research log.")
    return decision


# ============================================================================
# Queue-phase decision (pure) — shared by E16, E17 and E18
# ============================================================================
def decide_queue_phase(records: dict, grid: dict) -> dict:
    """Which configs of a grid are done, which are still pending."""
    done, pending, broken = [], [], []
    for entry in grid["runs"]:
        oc = outcome(records.get(entry["run_id"]))
        if oc["complete"]:
            done.append(entry["run_id"])
        else:
            pending.append(entry["run_id"])
            if oc["present"]:
                broken.append(entry["run_id"])
    return {"phase": grid.get("phase"), "total": len(grid["runs"]),
            "done": done, "pending": pending, "in_progress": broken,
            "complete": not pending}


# ============================================================================
# Stage 16 selection (pure)
# ============================================================================
BUDGET_CAVEAT = (
    "E16 arms run 80 epochs / patience 25 against a C0 result measured at "
    "300 epochs / patience 50: the comparison is biased AGAINST the arms, so "
    "a rejected axis may simply not have converged yet")


def c0_provenance(accepted) -> str:
    """How the C0 result was obtained -- it goes into every derived artifact."""
    if not accepted:
        return "no accepted Stage 15 attempt"
    parts = ", ".join(f"{item['run_id']}={item['best_val_ssim']}"
                      for item in accepted)
    return f"max over the accepted Stage 15 attempts ({parts})"


def select_e16(records: dict, grid: dict, c0_result: float,
               margin: float = E16_ADOPT_MARGIN, c0_source: str = "") -> dict:
    """Pre-registered selection: primary val SSIM, tie-break val PSNR, veto
    on divergence, adopt an axis only when it beats C0 by MORE than margin."""
    threshold = None if c0_result is None else c0_result + margin
    rows = []
    for entry in grid["runs"]:
        oc = outcome(records.get(entry["run_id"]))
        best = oc["best_val_ssim"]
        delta = None if (best is None or c0_result is None) else best - c0_result
        row = {
            "run_id": entry["run_id"], "axis": entry["axis"],
            "value": entry["value"], "best_val_ssim": best,
            "best_val_psnr": oc["best_val_psnr"], "best_epoch": oc["best_epoch"],
            "epochs_run": oc["epochs_run"], "stop_reason": oc["stop_reason"],
            "diverged": oc["diverged"],
            "divergence_reasons": "; ".join(
                (oc["divergence"] or {}).get("reasons", [])),
            "gp_last10": (oc["divergence"] or {}).get("gp_last10"),
            "out_std_last10": (oc["divergence"] or {}).get("out_std_last10"),
            "nyquist_last10": (oc["divergence"] or {}).get("nyquist_last10"),
            "config_sha8": oc["config_sha8"],
            "delta_vs_c0": delta, "threshold": threshold,
            "c0_result": c0_result, "c0_source": c0_source,
            "budget_caveat": BUDGET_CAVEAT,
            "overrides": entry.get("overrides", {}),
            "decision": None,
        }
        if not oc["complete"]:
            row["decision"] = "rejected: run not complete"
        elif oc["diverged"]:
            row["decision"] = "vetoed: divergence (E15 criterion 1)"
        elif best is None:
            # A terminal run with no score at all: complete no longer implies
            # a number, so the comparison below must never see None.
            row["decision"] = "vetoed: no best_val_ssim recorded"
        elif threshold is None:
            row["decision"] = "rejected: no accepted E15 result to compare"
        elif best > threshold:
            row["decision"] = "eligible"
        else:
            row["decision"] = (
                f"rejected: delta {delta:+.4f} <= margin {margin}")
        rows.append(row)

    adopted = {}
    for axis in sorted({row["axis"] for row in rows}):
        eligible = [row for row in rows
                    if row["axis"] == axis and row["decision"] == "eligible"]
        if not eligible:
            continue
        eligible.sort(
            key=lambda row: (row["best_val_ssim"],
                             row["best_val_psnr"] if row["best_val_psnr"]
                             is not None else float("-inf")),
            reverse=True)
        winner, losers = eligible[0], eligible[1:]
        winner["decision"] = (
            f"ADOPTED: {winner['best_val_ssim']:.4f} > threshold "
            f"{threshold:.4f} (delta {winner['delta_vs_c0']:+.4f})")
        for row in losers:
            row["decision"] = (
                f"rejected: lost the {axis} axis to {winner['run_id']} "
                f"({row['best_val_ssim']:.4f} vs {winner['best_val_ssim']:.4f})")
        adopted[axis] = winner

    # Rounded only AFTER every comparison is made: the decisions above run on
    # the raw floats, this is purely so the CSV reads -0.0200 instead of
    # -0.020000000000000018.
    for row in rows:
        for key in ("delta_vs_c0", "threshold", "gp_last10", "out_std_last10",
                    "nyquist_last10"):
            if isinstance(row.get(key), float):
                row[key] = round(row[key], 6)
    return {"c0_result": c0_result, "margin": margin, "c0_source": c0_source,
            "budget_caveat": BUDGET_CAVEAT,
            "threshold": None if threshold is None else round(threshold, 6),
            "rows": rows, "adopted": adopted}


def build_final_config(base_config: dict, selection: dict) -> dict:
    """C0 with the adopted axis substitutions, plus the full derivation."""
    config = json.loads(json.dumps(base_config))  # deep copy, stdlib only
    config.pop("_comment", None)
    config["run_id"] = "e17-final"
    applied = {}
    for axis, winner in sorted(selection["adopted"].items()):
        for key, value in winner["overrides"].items():
            config[key] = value
            applied[key] = value
    config["_comment"] = (
        "Stage 17 final configuration: C0 (scripts/training/configs/e15_reference.json) "
        "with the Stage 16 axes that beat the accepted Stage 15 result by more "
        "than the pre-registered margin. Generated by "
        "scripts/training/chain_driver.py; see _derivation for every number, and "
        "results/training/e16_results.csv for the full grid.")
    config["_derivation"] = {
        "generated_utc": utc_now(),
        "rule": ("adopt an axis iff best val SSIM > C0 + margin; best score "
                 "wins inside an axis, val PSNR breaks ties; any diverged run "
                 "is vetoed (pre-registration, Stage 16 selection rule)"),
        "c0_result": selection["c0_result"],
        "c0_source": selection.get("c0_source", ""),
        "margin": selection["margin"],
        "threshold": selection["threshold"],
        "budget_caveat": selection.get("budget_caveat", BUDGET_CAVEAT),
        "applied_overrides": applied,
        "axes": [
            {"axis": row["axis"], "run_id": row["run_id"],
             "value": row["value"], "best_val_ssim": row["best_val_ssim"],
             "best_val_psnr": row["best_val_psnr"],
             "delta_vs_c0": row["delta_vs_c0"], "diverged": row["diverged"],
             "decision": row["decision"]}
            for row in selection["rows"]
        ],
    }
    return config


# ============================================================================
# Stage 17 / Stage 18 grid construction (pure)
# ============================================================================
def measure_tau(records: dict, final_config: dict, bench: dict,
                batch: int = CONDITIONED_BATCH) -> dict:
    """Seconds per iteration for the final architecture at the given batch.

    Two independent readings, both reported: the REAL training loop
    (epoch_seconds / iterations over the records of a matching run -- the
    methodology behind the tau = 1.51 s/iter quoted in the pre-registration)
    and the isolated benchmark of results/training/bench-felix.json
    (1.93 s/iter for the same architecture: it measures the same iteration
    without the generator hoisting the real loop does). The loop reading wins
    when it exists, because that is the tau the schedule is actually paid in.
    """
    generator = final_config.get("generator")
    critic = final_config.get("critic")
    info = {"generator": generator, "critic": critic, "batch": batch,
            "tau_loop": None, "tau_loop_source": None, "tau_bench": None,
            "tau_used": None, "source": "none"}

    samples = []
    for run_id, record in sorted(records.items()):
        config = (record.get("summary") or {}).get("config") or record.get("config")
        if isinstance(config, dict) and "config" in config:
            config = config["config"]
        if not isinstance(config, dict):
            continue
        if (config.get("generator") != generator
                or config.get("critic") != critic
                or int(config.get("batch", -1)) != int(batch)):
            continue
        per_iter = [row["epoch_seconds"] / row["iterations"]
                    for row in record.get("metrics", [])
                    if isinstance(row.get("epoch_seconds"), (int, float))
                    and isinstance(row.get("iterations"), int)
                    and row["iterations"] > 0]
        if per_iter:
            per_iter.sort()
            samples.append((run_id, per_iter[len(per_iter) // 2]))
    if samples:
        run_id, tau = min(samples, key=lambda item: item[1])
        info["tau_loop"] = round(tau, 4)
        info["tau_loop_source"] = run_id

    for entry in (bench or {}).get("results", []):
        if (entry.get("generator") == generator
                and entry.get("critic") == critic
                and entry.get("batch") == batch
                and entry.get("status") == "ok"):
            info["tau_bench"] = entry.get("seconds_per_iteration")
            break

    if info["tau_loop"] is not None:
        info["tau_used"] = info["tau_loop"]
        info["source"] = f"training loop ({info['tau_loop_source']})"
    elif info["tau_bench"] is not None:
        info["tau_used"] = info["tau_bench"]
        info["source"] = "bench-felix.json (isolated benchmark)"
    return info


def build_e17_runs(final_config_path: str, tau: dict) -> dict:
    """4 levels + 2 extra seeds at 5%; the conditioned model is DEFERRED."""
    levels = [("f50", 0.5, 0), ("f10", 0.1, 1), ("f5", 0.05, 2), ("f1", 0.01, 3)]
    runs = []
    for name, fraction, index in levels:
        runs.append({
            "run_id": f"e17-final-{name}",
            "axis": "fraction", "value": fraction,
            "seed": seed_for(17, index, 0),
            "epochs": 200, "patience": 50,
            "overrides": {"fraction": fraction},
        })
    for rep in (1, 2):
        runs.append({
            "run_id": f"e17-final-f5-s{rep}",
            "axis": "seed", "value": f"rep {rep} at 5%",
            "seed": seed_for(17, 2, rep),
            "epochs": 200, "patience": 50,
            "overrides": {"fraction": 0.05},
        })

    gate_ok = (tau.get("tau_used") is not None
               and tau["tau_used"] <= TAU_MAX_CONDITIONED)
    return {
        "_comment": (
            "Stage 17 final multilevel training: one model per level plus two "
            "extra 5% seeds for the inter-seed sigma. Generated by "
            "scripts/training/chain_driver.py from the Stage 16 selection."),
        "phase": "e17",
        "base_config": final_config_path,
        "runs": runs,
        "deferred": [{
            "run_id": "e17-conditioned",
            "reason": "not implemented",
            "detail": (
                "The pre-registration trains a single conditioned model (an "
                "extra 'log f' input channel over all fractions) if and only "
                "if the current tau <= "
                f"{TAU_MAX_CONDITIONED} s/iter. The conditioning channel does "
                "not exist in MuographyPairs or in the engine (both are "
                "single-channel), so the driver CANNOT launch it without "
                "changing the frozen input contract. It is deferred: the "
                "plan's literature clause covers the omission and a dated "
                "addendum is required before it is either trained or dropped."),
            "tau_gate": {
                "limit_s_per_iter": TAU_MAX_CONDITIONED,
                "tau_used": tau.get("tau_used"),
                "tau_loop": tau.get("tau_loop"),
                "tau_bench": tau.get("tau_bench"),
                "source": tau.get("source"),
                "gate_satisfied": bool(gate_ok),
            },
        }],
    }


def build_e18_runs(final_config_path: str, final_config: dict) -> dict:
    """The three Stage 18 ablations that need a new run (A2, A4, A5)."""
    generator = final_config.get("generator", "unet_p2p_6")
    runs = [
        {"run_id": "e18-abl-noL1", "axis": "lambda_l1", "value": 0,
         "seed": seed_for(18, 0, 0), "epochs": 120, "patience": 50,
         "overrides": {"fraction": 0.05, "lambda_l1": 0}},
        {"run_id": "e18-abl-noskip", "axis": "generator",
         "value": "unet_p2p_6_noskip", "seed": seed_for(18, 1, 0),
         "epochs": 120, "patience": 50,
         "overrides": {"fraction": 0.05, "generator": "unet_p2p_6_noskip"}},
        {"run_id": "e18-abl-depth4", "axis": "generator", "value": "unet_p2p_4",
         "seed": seed_for(18, 2, 0), "epochs": 120, "patience": 50,
         "overrides": {"fraction": 0.05, "generator": "unet_p2p_4"}},
    ]
    comment = (
        "Stage 18 ablations A2 (lambda_L1 = 0), A4 (no skip connections) and "
        "A5 (depth 6 vs 4), 120 epochs at the 5% level over the Stage 17 "
        "final configuration. A1, A3 and A6 reuse existing runs (E13.3, E16 "
        "c01-c03, E16 c12) and need no new training.")
    if generator != "unet_p2p_6":
        comment += (
            f" WARNING: Stage 16 selected generator '{generator}', so the A4 "
            "and A5 arms compare unet_p2p_6_noskip / unet_p2p_4 against a "
            "final model that is NOT unet_p2p_6; report the architecture of "
            "every arm explicitly in the ablation table.")
    return {"_comment": comment, "phase": "e18",
            "base_config": final_config_path, "runs": runs}


# ============================================================================
# Command construction (pure)
# ============================================================================
def train_command(base_config: str, entry: dict, device: str = "cuda") -> list:
    """argv for one grid entry — the single place overrides become flags."""
    cmd = [sys.executable, "scripts/training/train_cwgan_gp.py",
           "--config", base_config,
           "--run-id", entry["run_id"], "--seed", str(entry["seed"])]
    overrides = dict(entry.get("overrides") or {})
    for key in ("epochs", "patience"):
        if entry.get(key) is not None:
            overrides.setdefault(key, entry[key])
    for key, value in sorted(overrides.items()):
        if key not in OVERRIDE_FLAGS:
            raise KeyError(
                f"{entry['run_id']}: override {key!r} has no command-line flag "
                f"in scripts/training/train_cwgan_gp.py (known: {sorted(OVERRIDE_FLAGS)})")
        if key in BOOL_FLAGS:
            cmd += [OVERRIDE_FLAGS[key], "1" if value else "0"]
        else:
            cmd += [OVERRIDE_FLAGS[key], str(value)]
    cmd += ["--device", device, "--resume", "auto"]
    return cmd


# ============================================================================
# Cluster layer (impure)
# ============================================================================
def child_env() -> dict:
    """os.environ minus the driver's own allocation (see LEAKY_SLURM_VARS)."""
    return {key: value for key, value in os.environ.items()
            if key not in LEAKY_SLURM_VARS}


# ---------------------------------------------------------------- disk quota
def parse_quota(text: str) -> dict:
    """Parse `quota -s` output into megabytes.

    The interesting line is the one that starts with a filesystem path; its
    first three numeric columns are space / quota / limit, each with a K, M, G
    or T suffix under -s. The hard limit is what actually kills a write, and
    when it is 0 (unset) the soft quota is the effective ceiling.
    """
    result = {"used_mb": None, "quota_mb": None, "limit_mb": None,
              "headroom_mb": None}
    scale = {"K": 1 / 1024.0, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}

    def to_mb(token: str):
        token = token.strip().rstrip("*")
        if not token:
            return None
        factor = scale.get(token[-1].upper())
        number = token[:-1] if factor else token
        try:
            return float(number) * (factor if factor else 1 / 1024.0)
        except ValueError:
            return None

    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 4 or not fields[0].startswith("/"):
            continue
        used, soft, hard = (to_mb(fields[1]), to_mb(fields[2]), to_mb(fields[3]))
        if used is None:
            continue
        ceiling = hard if hard else soft
        result.update({"used_mb": used, "quota_mb": soft, "limit_mb": hard})
        if ceiling:
            result["headroom_mb"] = ceiling - used
        return result
    return result


def headroom_mb() -> dict:
    """Free megabytes before writes start failing, and where the number came
    from. The user quota is the real ceiling; the filesystem free space is a
    fallback that can be wildly optimistic on a shared /home."""
    try:
        proc = subprocess.run(["quota", "-s"], capture_output=True, text=True,
                              timeout=60)
        if proc.returncode == 0 or proc.stdout.strip():
            parsed = parse_quota(proc.stdout)
            if parsed["headroom_mb"] is not None:
                return {"headroom_mb": parsed["headroom_mb"],
                        "used_mb": parsed["used_mb"],
                        "limit_mb": parsed["limit_mb"] or parsed["quota_mb"],
                        "source": "quota -s"}
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        usage = shutil.disk_usage(str(Path.home()))
        return {"headroom_mb": usage.free / (1024.0 * 1024.0),
                "used_mb": None, "limit_mb": None,
                "source": "shutil.disk_usage($HOME) — filesystem free space, "
                          "NOT the user quota"}
    except OSError:
        return {"headroom_mb": None, "used_mb": None, "limit_mb": None,
                "source": "unavailable"}


def phase_of_run(run_id: str) -> str:
    """e17-final-f5 -> e17. The stage prefix is the run-id convention."""
    return run_id.split("-", 1)[0]


def prune_run_checkpoints(run_dir: Path, phase: str,
                          dry_run: bool = False) -> dict:
    """Delete the checkpoints a FINISHED run no longer needs.

    last.pt only exists to resume, so a terminal run never needs it again.
    best.pt is kept for the stages whose weights are cited downstream: e17
    feeds models/weights-manifest.json and the Stage 20 predictions, and e15
    keeps its best.pt because the pre-registered R4 ladder ends in "reinicio
    desde best.pt con semilla nueva" -- deleting it would remove the material
    of a human intervention the protocol allows.

    Caller must have checked that the run is complete: pruning last.pt of a
    live or interrupted run would throw away hours of training.
    """
    freed = 0
    removed = []
    targets = ["last.pt"]
    if phase not in KEEP_BEST_PHASES:
        targets.append("best.pt")
    for name in targets:
        path = run_dir / "checkpoints" / name
        if not path.exists():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if not dry_run:
            try:
                path.unlink()
            except OSError as error:
                print(f"  could not prune {path}: {error}")
                continue
        freed += size
        removed.append(name)
    return {"run_dir": str(run_dir), "removed": removed,
            "freed_mb": round(freed / (1024.0 * 1024.0), 1)}


def largest_run_dirs(root: Path, limit: int = 5) -> list:
    """The heaviest run directories — only computed when the chain halts."""
    sizes = []
    if not root.is_dir():
        return sizes
    for child in root.iterdir():
        if not child.is_dir():
            continue
        total = 0
        for path in child.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
        sizes.append((child.name, round(total / (1024.0 * 1024.0), 1)))
    sizes.sort(key=lambda item: item[1], reverse=True)
    return sizes[:limit]


def squeue_job_names() -> set:
    """Job names of the user's queued/running jobs; empty set if unavailable."""
    user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    try:
        proc = subprocess.run(["squeue", "-h", "-u", user, "-o", "%j"],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return set()
    if proc.returncode != 0:
        return set()
    return {name for name in proc.stdout.split() if name}


def own_job_count():
    """Number of own jobs (RUNNING + PENDING), or None when squeue is absent."""
    user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    try:
        proc = subprocess.run(["squeue", "-h", "-u", user, "-o", "%i"],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return len([line for line in proc.stdout.split() if line])


class Cluster:
    """sbatch with the two guards the QoS forces: name and submission cap."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.job_names = squeue_job_names()
        self.count = own_job_count()
        self.submitted = []
        self.skipped = []

    def submit(self, job_name: str, script: str, exports: dict) -> str:
        if job_name in self.job_names:
            self.skipped.append(f"{job_name}: already in the queue")
            return "in-flight"
        if self.count is None:
            # No squeue: the in-flight guard cannot be trusted, so a real run
            # refuses to submit. --dry-run assumes an empty queue instead, so
            # the whole plan is still printable off the cluster.
            if not self.dry_run:
                self.skipped.append(
                    f"{job_name}: squeue unavailable, not submitting")
                return "no-squeue"
            self.count = 0
        projected = self.count + 1
        if projected > MAX_OWN_JOBS:
            self.skipped.append(
                f"{job_name}: {self.count} own jobs, submitting would make "
                f"{projected} > {MAX_OWN_JOBS} (driver + 2 GPU jobs + 1 free "
                "QoS slot)")
            return "over-budget"
        export = "ALL" + "".join(f",{k}={v}" for k, v in exports.items())
        cmd = ["sbatch", f"--job-name={job_name}", f"--export={export}", script]
        if self.dry_run:
            print("DRY:", " ".join(cmd))
        else:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=child_env())
            if proc.returncode != 0:
                self.skipped.append(f"{job_name}: sbatch refused (submit limit?)")
                return "refused"
        # Counted even in --dry-run, so the printed plan respects the same cap
        # the real run would hit.
        self.submitted.append(job_name)
        self.job_names.add(job_name)
        self.count += 1
        return "dry-run" if self.dry_run else "submitted"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runs_root() -> Path:
    return Path(os.environ.get("MUOGRAPHY_RUNS",
                               str(REPO_ROOT / "runs")))


# ============================================================================
# Output helpers (impure)
# ============================================================================
def write_json(path: Path, payload) -> None:
    """Atomic: a truncated e17_final.json would break every later pass."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_e16_csv(path: Path, selection: dict) -> None:
    columns = ["run_id", "axis", "value", "best_val_ssim", "best_epoch",
               "best_val_psnr", "epochs_run", "stop_reason", "diverged",
               "divergence_reasons", "gp_last10", "out_std_last10",
               "nyquist_last10", "delta_vs_c0", "c0_result", "c0_source",
               "threshold", "budget_caveat", "config_sha8", "decision"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns,
                                extrasaction="ignore")
        writer.writeheader()
        for row in selection["rows"]:
            writer.writerow(row)


def write_e17_plan_csv(path: Path, grid: dict) -> None:
    columns = ["run_id", "axis", "value", "seed", "epochs", "status", "note"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for entry in grid["runs"]:
            writer.writerow({"run_id": entry["run_id"], "axis": entry["axis"],
                             "value": entry["value"], "seed": entry["seed"],
                             "epochs": entry["epochs"], "status": "planned",
                             "note": ""})
        for entry in grid.get("deferred", []):
            gate = entry["tau_gate"]
            writer.writerow({
                "run_id": entry["run_id"], "axis": "conditioning",
                "value": "log f channel", "seed": "", "epochs": "",
                "status": f"deferred: {entry['reason']}",
                "note": (f"tau gate: tau_used={gate['tau_used']} s/iter "
                         f"(loop={gate['tau_loop']}, bench={gate['tau_bench']}, "
                         f"source={gate['source']}) vs limit "
                         f"{gate['limit_s_per_iter']} -> "
                         f"gate_satisfied={gate['gate_satisfied']}; "
                         "conditioning channel not implemented, dated addendum "
                         "required")})


def append_log(line: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ============================================================================
# Driver
# ============================================================================
class Driver:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.records = read_records(RECORDS_DIR)
        self.cluster = Cluster(dry_run=dry_run)
        self.notes = []
        self.previous_state = self.load_previous_state()
        # Carried across passes: the state file is rewritten every time, so the
        # launch counters have to be read back or the livelock guard resets.
        self.launch_history = dict(
            self.previous_state.get("launch_history") or {})
        self.state = {"generated_utc": utc_now(), "phase": None,
                      "status": "RUNNING", "detail": "", "stages": {},
                      "launch_history": self.launch_history}

    @staticmethod
    def load_previous_state() -> dict:
        if not STATE_PATH.exists():
            return {}
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    # ------------------------------------------------------------ disk quota
    def ensure_headroom(self) -> tuple:
        """Free space check, an automatic prune, then a hard stop.

        Runs BEFORE any submission: a job that dies mid-checkpoint on a full
        quota loses its epoch, scores a failure and corrupts nothing but wastes
        hours, and that is exactly how the first Stage 15 pair died.
        """
        info = headroom_mb()
        free = info["headroom_mb"]
        self.state["disk"] = dict(info, threshold_mb=MIN_HEADROOM_MB)
        if free is None:
            self.note("disk: headroom unknown (no quota, no disk_usage) — "
                      "proceeding")
            return True, "disk headroom unknown"
        self.note(f"disk: {free:.0f} MB free ({info['source']}), "
                  f"threshold {MIN_HEADROOM_MB:.0f} MB")
        if free >= MIN_HEADROOM_MB:
            return True, f"{free:.0f} MB free"

        pruned = self.prune_finished_runs()
        self.state["disk"]["pruned"] = pruned
        info = headroom_mb()
        free = info["headroom_mb"]
        self.state["disk"].update(info)
        self.note(f"disk: after pruning {pruned['runs']} finished run(s) "
                  f"({pruned['freed_mb']:.0f} MB): {free:.0f} MB free")
        if free is not None and free >= MIN_HEADROOM_MB:
            return True, f"{free:.0f} MB free after pruning"

        biggest = largest_run_dirs(runs_root())
        self.state["disk"]["largest_run_dirs"] = biggest
        return False, (
            f"only {free:.0f} MB of headroom ({info['source']}), below the "
            f"{MIN_HEADROOM_MB:.0f} MB a checkpoint pair needs, and pruning "
            f"finished runs freed only {pruned['freed_mb']:.0f} MB. Largest "
            "run directories: "
            + ", ".join(f"{name} {size:.0f} MB" for name, size in biggest)
            + ". Free space by hand (checkpoints of runs already recorded in "
              "results/training/records/ are the usual candidates) before "
              "restarting the chain.")

    def prune_finished_runs(self) -> dict:
        """Drop the checkpoints of runs that already have a terminal record."""
        root = runs_root()
        freed = 0.0
        pruned = []
        if not root.is_dir():
            return {"runs": 0, "freed_mb": 0.0, "details": []}
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            # Only FINISHED runs: last.pt is the resume point of everything
            # else, including a run this driver is about to relaunch.
            if not is_complete(read_record(RECORDS_DIR, run_id)):
                continue
            result = prune_run_checkpoints(run_dir, phase_of_run(run_id),
                                           dry_run=self.dry_run)
            if result["removed"]:
                freed += result["freed_mb"]
                pruned.append({"run_id": run_id, "removed": result["removed"],
                               "freed_mb": result["freed_mb"]})
                self.note(f"  pruned {run_id}: {', '.join(result['removed'])} "
                          f"({result['freed_mb']:.0f} MB)")
        return {"runs": len(pruned), "freed_mb": round(freed, 1),
                "details": pruned}

    # ------------------------------------------------------------- utilities
    def note(self, message: str) -> None:
        self.notes.append(message)
        print(message)

    def in_flight_runs(self, run_ids) -> set:
        """Grid runs whose worker job name is in the queue is NOT knowable
        (workers are shared), so only the named E15 jobs are mapped here."""
        return {run_id for run_id in run_ids
                if E15_JOB_NAMES.get(run_id) in self.cluster.job_names}

    def load_grid(self, path: Path):
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def ensure_workers(self, phase: str, grid_path: Path) -> None:
        relative = grid_path.relative_to(REPO_ROOT).as_posix()
        for worker in WORKERS:
            status = self.cluster.submit(
                f"muo-qw-{worker}", QUEUE_SBATCH,
                {"PHASE": phase, "GRID": relative, "WORKER": worker})
            self.note(f"  worker muo-qw-{worker}: {status}")

    def abandoned(self, run_ids) -> list:
        """Runs scripts/training/run_queue.py gave up on after repeated failures."""
        root = runs_root()
        return [run_id for run_id in run_ids
                if (root / run_id / ".claim" / "abandoned").exists()]

    def queue_stage(self, phase: str, grid: dict, grid_path: Path) -> dict:
        """One pass over a worker-loop stage: report, then submit or stop."""
        state = decide_queue_phase(self.records, grid)
        self.state["stages"][phase] = {
            "done": len(state["done"]), "total": state["total"],
            "pending": state["pending"]}
        self.note(f"{phase.upper()}: {len(state['done'])}/{state['total']} "
                  f"complete; pending {state['pending']}")
        if state["complete"]:
            return state
        dead = self.abandoned(state["pending"])
        if dead and len(dead) == len(state["pending"]):
            self.state["stages"][phase]["abandoned"] = dead
            state["halt"] = (
                f"every pending {phase} config was abandoned by the queue "
                f"worker after repeated failures: {dead}. Read the worker log "
                "(logs/muo-qw-*.out), fix the cause and delete the "
                "<run>/.claim/abandoned marker to let the chain retry.")
            return state
        if dead:
            self.note(f"  WARNING: abandoned after repeated failures: {dead}")
        self.ensure_workers(phase, grid_path)
        return state

    # ---------------------------------------------------------------- stages
    def run(self) -> dict:
        headroom_ok, headroom_detail = self.ensure_headroom()
        if not headroom_ok:
            return self.finish("disk", "HALTED", headroom_detail)

        e15 = decide_e15(self.records, self.in_flight_runs(E15_RUN_IDS),
                         self.launch_history)
        self.state["stages"]["e15"] = {
            "status": e15["status"], "detail": e15["detail"],
            "c0_result": e15["c0_result"],
            "attempts": [{"run_id": a["run_id"], "state": a["state"],
                          "best_val_ssim": a["best_val_ssim"],
                          "in_flight": a["in_flight"]} for a in e15["attempts"]],
        }
        self.note(f"E15: {e15['status']} — {e15['detail']}")
        for attempt in e15["attempts"]:
            report = attempt["outcome"]["divergence"] or {}
            self.note(f"  {attempt['run_id']}: {attempt['state']} "
                      f"best={attempt['best_val_ssim']} "
                      f"epochs={report.get('n_epochs')} "
                      f"gp10={report.get('gp_last10')} "
                      f"out_std10={report.get('out_std_last10')} "
                      f"nyquist10={report.get('nyquist_last10')}")

        if e15["status"] == "halt":
            return self.finish("e15", "HALTED", e15["detail"])
        if e15["status"] in ("launch", "relaunch"):
            target = e15["launch"]
            status = self.cluster.submit(
                target["job_name"], E15_SBATCH,
                {"RUN_ID": target["run_id"], "SEED": target["seed"]})
            self.note(f"  {target['job_name']}: {status}")
            if status == "submitted" and e15["launch_entry"]:
                # Counted only when sbatch actually accepted the job: a refusal
                # or a full queue must not consume one of the three tries.
                self.launch_history[target["run_id"]] = e15["launch_entry"]
            return self.finish("e15", "RUNNING",
                               f"{e15['status']} {target['run_id']}: {status}")
        if e15["status"] == "wait":
            return self.finish("e15", "RUNNING", e15["detail"])

        return self.run_e16(e15["c0_result"], c0_provenance(e15["accepted"]))

    def run_e16(self, c0_result, c0_source: str = "") -> dict:
        grid = self.load_grid(E16_GRID)
        if grid is None:
            return self.finish("e16", "HALTED", f"missing grid {E16_GRID}")
        state = self.queue_stage("e16", grid, E16_GRID)
        if state.get("halt"):
            return self.finish("e16", "HALTED", state["halt"])
        if not state["complete"]:
            return self.finish("e16", "RUNNING",
                               f"{len(state['done'])}/{state['total']} configs")

        base_config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
        selection = select_e16(self.records, grid, c0_result,
                               c0_source=c0_source)
        self.note(f"  C0 result {c0_result} — {c0_source}")
        self.note(f"  caveat: {BUDGET_CAVEAT}")
        for row in selection["rows"]:
            self.note(f"  {row['run_id']} [{row['axis']}={row['value']}] "
                      f"ssim={row['best_val_ssim']} "
                      f"delta={row['delta_vs_c0']} -> {row['decision']}")
        if not self.dry_run:
            # No timestamp inside: rewriting it every pass is byte-idempotent.
            write_e16_csv(TABLES_DIR / "e16_results.csv", selection)

        if E17_FINAL.exists():
            # Once written, the selected configuration is frozen: later passes
            # read it back instead of regenerating it, so a rerun can never
            # silently reshape the configuration the runs were trained with.
            final_config = json.loads(E17_FINAL.read_text(encoding="utf-8"))
            self.note(f"  reusing the frozen {E17_FINAL.name}")
        else:
            final_config = build_final_config(base_config, selection)
            self.note(f"  adopted axes: {sorted(selection['adopted'])} -> "
                      f"{final_config['_derivation']['applied_overrides']}")
            if not self.dry_run:
                write_json(E17_FINAL, final_config)
            else:
                print(f"DRY: would write {E17_FINAL}")
        self.state["stages"]["e16"]["adopted"] = {
            key: final_config.get(key)
            for key in final_config.get("_derivation", {})
            .get("applied_overrides", {})}
        return self.run_e17(final_config)

    def run_e17(self, final_config: dict) -> dict:
        grid = self.load_grid(E17_GRID)
        if grid is None:
            bench = {}
            if BENCH_PATH.exists():
                bench = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
            tau = measure_tau(self.records, final_config, bench)
            grid = build_e17_runs(
                E17_FINAL.relative_to(REPO_ROOT).as_posix(), tau)
            if not self.dry_run:
                write_json(E17_GRID, grid)
                write_e17_plan_csv(TABLES_DIR / "e17_plan.csv", grid)
            else:
                print(f"DRY: would write {E17_GRID}")

        # Printed on EVERY pass, not only the one that created the grid: the
        # deferral is the one place where the chain does less than the
        # pre-registration asks, and it has to stay visible in the job log.
        for deferred in grid.get("deferred", []):
            gate = deferred["tau_gate"]
            self.note(
                f"WARNING: {deferred['run_id']} is DEFERRED ({deferred['reason']}). "
                "The 'log f' conditioning channel does not exist in the dataset "
                "or the engine, so the driver will not launch it. tau gate: "
                f"tau_used={gate['tau_used']} s/iter (loop={gate['tau_loop']}, "
                f"bench={gate['tau_bench']}, source={gate['source']}) vs limit "
                f"{gate['limit_s_per_iter']} -> gate_satisfied="
                f"{gate['gate_satisfied']}. A dated addendum is required before "
                "it is trained or dropped.")
            self.state["stages"]["e17_conditioned"] = dict(
                gate, decision=f"deferred: {deferred['reason']}")

        state = self.queue_stage("e17", grid, E17_GRID)
        if state.get("halt"):
            return self.finish("e17", "HALTED", state["halt"])
        if not state["complete"]:
            return self.finish("e17", "RUNNING",
                               f"{len(state['done'])}/{state['total']} runs")

        manifest = self.build_manifest(grid)
        self.state["stages"]["e17"]["weights_manifest"] = {
            "path": MANIFEST_PATH.relative_to(REPO_ROOT).as_posix(),
            "hashed": len([r for r in manifest["runs"] if r["sha256"]]),
            "missing": manifest["missing"]}
        return self.run_e18(final_config)

    def build_manifest(self, grid: dict) -> dict:
        root = runs_root()
        expected = [entry["run_id"] for entry in grid["runs"]]
        if MANIFEST_PATH.exists():
            # The Stage 17 checkpoints are frozen once their runs finish, so a
            # complete manifest is not re-hashed on every 45-minute pass.
            try:
                previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous = {}
            if (not previous.get("missing")
                    and [r["run_id"] for r in previous.get("runs", [])] == expected):
                self.note(f"  weights manifest: reusing {MANIFEST_PATH.name} "
                          f"({len(expected)} checkpoints already hashed)")
                return previous
        runs, missing = [], []
        for entry in grid["runs"]:
            run_id = entry["run_id"]
            oc = outcome(self.records.get(run_id))
            checkpoint = root / run_id / "checkpoints" / "best.pt"
            digest = size = None
            if checkpoint.exists():
                digest = sha256_file(checkpoint)
                size = checkpoint.stat().st_size
            else:
                missing.append(run_id)
            runs.append({"run_id": run_id,
                         "checkpoint": str(checkpoint),
                         "sha256": digest, "bytes": size,
                         "config_sha8": oc["config_sha8"],
                         "best_val_ssim": oc["best_val_ssim"],
                         "best_epoch": oc["best_epoch"]})
        manifest = {
            "generated_utc": utc_now(),
            "stage": "17",
            "comment": ("SHA-256 of the frozen Stage 17 generators. Regenerate "
                        "with scripts/training/chain_driver.py; the external backup "
                        "required before Stage 20 is a human step."),
            "runs_root": str(root), "runs": runs, "missing": missing}
        if missing:
            self.note(f"  WARNING: no best.pt for {missing} — manifest "
                      "written with null hashes for those runs")
        if not self.dry_run:
            write_json(MANIFEST_PATH, manifest)
        else:
            print(f"DRY: would write {MANIFEST_PATH}")
        self.note(f"  weights manifest: {len(runs) - len(missing)}/{len(runs)} "
                  "checkpoints hashed")
        return manifest

    def run_e18(self, final_config: dict) -> dict:
        grid = self.load_grid(E18_GRID)
        if grid is None:
            grid = build_e18_runs(
                E17_FINAL.relative_to(REPO_ROOT).as_posix(), final_config)
            if not self.dry_run:
                write_json(E18_GRID, grid)
            else:
                print(f"DRY: would write {E18_GRID}")
        state = self.queue_stage("e18", grid, E18_GRID)
        if state.get("halt"):
            return self.finish("e18", "HALTED", state["halt"])
        if not state["complete"]:
            return self.finish("e18", "RUNNING",
                               f"{len(state['done'])}/{state['total']} runs")
        return self.finish("e18", "COMPLETE",
                           "E15-E18 closed; Stage 18 needs the committee "
                           "signature (PA-4) before OE2 is declared closed")

    def finish(self, phase: str, status: str, detail: str) -> dict:
        self.state["phase"] = phase
        self.state["status"] = status
        self.state["detail"] = detail
        self.state["submitted"] = self.cluster.submitted
        self.state["skipped"] = self.cluster.skipped
        for message in self.cluster.skipped:
            self.note(f"  skipped {message}")
        disk = self.state.get("disk") or {}
        free = disk.get("headroom_mb")
        line = (f"{utc_now()} phase={phase} status={status} "
                f"submitted={','.join(self.cluster.submitted) or '-'} "
                f"headroom={'?' if free is None else f'{free:.0f}MB'} "
                f"detail={detail}")
        print(line)
        if not self.dry_run:
            write_json(STATE_PATH, self.state)
            append_log(line)
        return self.state


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="decide and print; no sbatch, no file is written")
    args = parser.parse_args()

    state = Driver(dry_run=args.dry_run).run()
    return 0 if state["status"] != "HALTED" else 2


if __name__ == "__main__":
    sys.exit(main())
