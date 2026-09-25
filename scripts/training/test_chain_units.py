#!/usr/bin/env python3
"""Stage 15-18 chain tests — the decision layer, on synthetic records.

The chain runs unattended for days, so its rules are tested here against
records built by hand in a temporary directory: no torch, no cluster, no
network. Sixteen checks, all self-verifying and pytest-free:

   1. seeds           the PA-1 derivation for every stage of the chain
   2. e16 grid file   the frozen grid on disk is the pre-registered table
   3. divergence      non-finite fields, mean gp >= 1, NaN val_ssim
   4. stable band     the pre-registered 0.775 floor, not a rounder number
   5. nyquist gate    criterion 3 halts instead of accepting a checkerboard
   6. null ssim       a NaN-collapsed run is finished AND diverged
   7. launch livelock three launches with no new epoch stop the chain
   8. relaunch        a dead attempt is resubmitted beside a running one
   9. resources       strikes, quota parsing, pruning rules, the env strip
  10. e15 outcomes    pass / stable / below / diverged / incomplete / absent
  11. e15 flow        wait, launch, relaunch, closed_pass, closed_stable, halt
  12. e16 selection   the 0.003 margin, the tie inside an axis, the veto
  13. e16 config      C0 + adopted axes + the full derivation
  14. e17 grid        4 levels + 2 seeds, and the deferred conditioned model
  15. e18 grid        the three ablations that need a new run
  16. commands        overrides -> command line, booleans included

Run:  python scripts/training/test_chain_units.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

from chain_driver import (  # noqa: E402
    E15_PASS_SSIM, E15_STABLE_FLOOR, E16_ADOPT_MARGIN, LEAKY_SLURM_VARS,
    MAX_LAUNCHES_WITHOUT_PROGRESS, build_e17_runs, build_e18_runs,
    build_final_config, checkerboard_report, child_env, classify_e15,
    decide_e15, decide_queue_phase, divergence_report, launch_guard,
    measure_tau, outcome, parse_quota, phase_of_run, prune_run_checkpoints,
    read_records, seed_for, select_e16, train_command)
from run_queue import strike_needed  # noqa: E402

C0 = {
    "run_id": "e15-c0-f05", "generator": "unet_p2p_6",
    "critic": "patch_critic_70", "fraction": 0.05,
    "input_scaling": "ref_equivalent", "hflip": True, "lambda_gp": 10,
    "lambda_l1": 100, "lambda_adv": 1, "lr": 0.0001, "n_critic": 5,
    "batch": 8, "epochs": 300, "seed": 20260805150000,
    "early_stop": {"metric": "val_ssim", "patience": 50, "min_delta": 0.001},
}


# --------------------------------------------------------------- fixtures
def write_record(root: Path, run_id: str, *, best=None, stop_reason="early_stop",
                 gp=0.4, epochs=30, best_epoch=None, val_psnr=29.5,
                 nan_field=None, nan_val_ssim=False, summary=True,
                 config=None, epoch_seconds=271.0, iterations=180,
                 nyquist=0.39, nyquist_rise=0.0) -> None:
    """A record dir shaped exactly like engine.Trainer writes it."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    best_epoch = best_epoch if best_epoch is not None else epochs
    rows = []
    for epoch in range(1, epochs + 1):
        row = {"epoch": epoch, "global_step": epoch * iterations,
               "iterations": iterations, "w_est": 0.5, "gp": gp, "l1": 0.02,
               "adv": -0.3, "d_real": 0.1, "d_fake": -0.1,
               "grad_norm_g": 10.0, "grad_norm_d": 5.0, "out_std": 0.02,
               "nyquist_ratio": nyquist + nyquist_rise * (epoch - 1),
               "epoch_seconds": epoch_seconds}
        if best is not None:
            row["val_ssim"] = best if epoch == best_epoch else best - 0.01
            row["val_psnr"] = val_psnr if epoch == best_epoch else val_psnr - 1
            row["val_ssim_std"] = 0.01
        if nan_field and epoch == epochs:
            row[nan_field] = None       # the engine writes null for NaN/Inf
        if nan_val_ssim and epoch == epochs:
            row["val_ssim"] = None
        rows.append(row)
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({"run_id": run_id, "config": config or C0}), encoding="utf-8")
    if summary:
        (run_dir / "summary.json").write_text(json.dumps({
            "run_id": run_id, "status": "done", "stop_reason": stop_reason,
            "config_sha8": "deadbeef", "best_val_ssim": best,
            "best_epoch": best_epoch, "epochs_run": epochs,
            "last_epoch": epochs, "config": config or C0}), encoding="utf-8")


def e16_grid() -> dict:
    return {"phase": "e16", "base_config": "scripts/training/configs/e15_reference.json",
            "runs": [
                {"run_id": "e16-c01", "axis": "lambda_l1", "value": 10,
                 "seed": seed_for(16, 1), "epochs": 80, "patience": 25,
                 "overrides": {"lambda_l1": 10}},
                {"run_id": "e16-c02", "axis": "lambda_l1", "value": 50,
                 "seed": seed_for(16, 2), "epochs": 80, "patience": 25,
                 "overrides": {"lambda_l1": 50}},
                {"run_id": "e16-c03", "axis": "lambda_l1", "value": 200,
                 "seed": seed_for(16, 3), "epochs": 80, "patience": 25,
                 "overrides": {"lambda_l1": 200}},
                {"run_id": "e16-c12", "axis": "critic",
                 "value": "patch_critic_142", "seed": seed_for(16, 12),
                 "epochs": 80, "patience": 25,
                 "overrides": {"critic": "patch_critic_142"}},
            ]}


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {label}: {'ok' if condition else 'FAIL'}"
          + (f" — {detail}" if detail else ""))
    return bool(condition)


# ------------------------------------------------------------------- tests
def test_seeds() -> bool:
    cases = {
        seed_for(15, 0, 0): 20260805150000,   # E15 attempt 1
        seed_for(15, 0, 2): 20260805150002,   # E15 attempt 3
        seed_for(16, 1): 20260805160100,      # E16 c01
        seed_for(16, 12): 20260805161200,     # E16 c12
        seed_for(17, 2, 1): 20260805170201,   # E17 5% seed rep 1
        seed_for(18, 2): 20260805180200,      # E18 depth-4 ablation
    }
    return check("seed derivation", all(k == v for k, v in cases.items()),
                 f"{sorted(cases)}")


def test_divergence() -> bool:
    root = Path(tempfile.mkdtemp())
    try:
        write_record(root, "clean", best=0.78, gp=0.4)
        write_record(root, "gp-blowup", best=0.78, gp=3.0)
        write_record(root, "nonfinite", best=0.78, nan_field="w_est")
        write_record(root, "nan-ssim", best=0.78, nan_val_ssim=True)
        records = read_records(root)
        clean = divergence_report(records["clean"]["metrics"])
        gp_bad = divergence_report(records["gp-blowup"]["metrics"])
        finite_bad = divergence_report(records["nonfinite"]["metrics"])
        ssim_bad = divergence_report(records["nan-ssim"]["metrics"])
        empty = divergence_report([])
        return all([
            check("clean run passes the divergence check",
                  clean["ok"] and abs(clean["gp_last10"] - 0.4) < 1e-9),
            check("mean gp >= 1 vetoes", not gp_bad["ok"],
                  gp_bad["reasons"][0]),
            check("non-finite field vetoes", not finite_bad["ok"],
                  finite_bad["reasons"][0]),
            check("NaN val_ssim vetoes", not ssim_bad["ok"],
                  ssim_bad["reasons"][0]),
            check("no metrics vetoes", not empty["ok"]),
            check("out_std is reported, never a veto",
                  clean["out_std_last10"] is not None and clean["ok"]),
        ])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_e15_outcomes() -> bool:
    root = Path(tempfile.mkdtemp())
    try:
        write_record(root, "pass", best=0.7801)
        write_record(root, "stable", best=0.7760)
        write_record(root, "edge", best=0.7777)        # NOT > 0.7777
        write_record(root, "below", best=0.7000)
        write_record(root, "diverged", best=0.7900, gp=2.0)
        write_record(root, "interrupted", best=0.7801, stop_reason="stop_file")
        write_record(root, "running", best=None, summary=False)
        records = read_records(root)
        got = {name: classify_e15(outcome(records[name]))
               for name in records}
        expected = {"pass": "pass", "stable": "stable", "edge": "stable",
                    "below": "below", "diverged": "diverged",
                    "interrupted": "incomplete", "running": "incomplete"}
        absent = classify_e15(outcome(None))
        return all([
            check("outcome classification", got == expected, json.dumps(got)),
            check("0.7777 exactly is stable, not a pass",
                  got["edge"] == "stable"),
            check("stop_file is not a terminal stop_reason",
                  got["interrupted"] == "incomplete"),
            check("a missing record is absent", absent == "absent"),
        ])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_e15_flow() -> bool:
    root = Path(tempfile.mkdtemp())
    results = []
    try:
        # (a) nothing yet, nothing queued -> launch attempt 1
        empty = decide_e15({})
        results.append(check(
            "empty state launches attempt 1",
            empty["status"] == "launch"
            and empty["launch"]["run_id"] == "e15-c0-attempt1"
            and empty["launch"]["seed"] == 20260805150000))

        # (b) two attempts in flight -> wait
        write_record(root, "e15-c0-attempt1", best=None, summary=False)
        write_record(root, "e15-c0-attempt2", best=None, summary=False)
        records = read_records(root)
        waiting = decide_e15(records, {"e15-c0-attempt1", "e15-c0-attempt2"})
        results.append(check("attempts in flight -> wait",
                             waiting["status"] == "wait"))

        # (c) same records, nothing in the queue -> relaunch the first stall
        stalled = decide_e15(records)
        results.append(check(
            "stalled attempt is resubmitted (--resume auto)",
            stalled["status"] == "relaunch"
            and stalled["launch"]["run_id"] == "e15-c0-attempt1"))

        # (d) attempt 1 passes while attempt 2 still runs -> close immediately
        write_record(root, "e15-c0-attempt1", best=0.7801)
        records = read_records(root)
        passed = decide_e15(records, {"e15-c0-attempt2"})
        results.append(check(
            "a pass closes E15 without waiting for the other attempt",
            passed["status"] == "closed_pass"
            and abs(passed["c0_result"] - 0.7801) < 1e-9))

        # (e) both finished, none above the bar, one stable -> proceed
        write_record(root, "e15-c0-attempt1", best=0.7762)
        write_record(root, "e15-c0-attempt2", best=0.7758)
        records = read_records(root)
        stable = decide_e15(records)
        results.append(check(
            "stable but not superior -> the chain proceeds",
            stable["status"] == "closed_stable"
            and abs(stable["c0_result"] - 0.7762) < 1e-9,
            "C0 result is the best accepted attempt"))

        # (f) one stable, one diverged: the diverged one is not accepted
        write_record(root, "e15-c0-attempt2", best=0.9000, gp=5.0)
        records = read_records(root)
        mixed = decide_e15(records)
        results.append(check(
            "a diverged attempt cannot set the C0 result",
            mixed["status"] == "closed_stable"
            and abs(mixed["c0_result"] - 0.7762) < 1e-9))

        # (g) two finished and unusable, third not launched -> launch attempt 3
        write_record(root, "e15-c0-attempt1", best=0.7100)
        records = read_records(root)
        third = decide_e15(records)
        results.append(check(
            "two unusable attempts -> launch the third (seed rep 2)",
            third["status"] == "launch"
            and third["launch"]["run_id"] == "e15-c0-attempt3"
            and third["launch"]["seed"] == 20260805150002))

        # (h) three finished, none stable -> halt for a human
        write_record(root, "e15-c0-attempt3", best=0.7000)
        records = read_records(root)
        halted = decide_e15(records)
        results.append(check("three attempts, none stable -> HALT",
                             halted["status"] == "halt",
                             halted["detail"][:60] + "..."))
        return all(results)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_e16_selection() -> bool:
    root = Path(tempfile.mkdtemp())
    try:
        c0 = 0.7762
        grid = e16_grid()
        # c01 below the margin, c02 and c03 both above it (c03 wins the axis),
        # c12 above the bar but diverged (vetoed).
        write_record(root, "e16-c01", best=c0 + 0.0020, epochs=40)
        write_record(root, "e16-c02", best=c0 + 0.0050, epochs=40, val_psnr=30.1)
        write_record(root, "e16-c03", best=c0 + 0.0080, epochs=40, val_psnr=30.4)
        write_record(root, "e16-c12", best=c0 + 0.0500, epochs=40, gp=4.0)
        records = read_records(root)

        phase = decide_queue_phase(records, grid)
        selection = select_e16(records, grid, c0)
        rows = {row["run_id"]: row for row in selection["rows"]}
        adopted = selection["adopted"]

        # A tie inside an axis is broken by val PSNR.
        write_record(root, "e16-c02", best=c0 + 0.0080, epochs=40, val_psnr=31.9)
        tie = select_e16(read_records(root), grid, c0)

        incomplete = select_e16(
            {k: v for k, v in records.items() if k != "e16-c03"}, grid, c0)

        return all([
            check("all four configs count as complete", phase["complete"],
                  f"{len(phase['done'])}/{phase['total']}"),
            check("threshold is C0 + 0.003",
                  abs(selection["threshold"] - (c0 + E16_ADOPT_MARGIN)) < 1e-12),
            check("a +0.0020 gain does not clear the margin",
                  rows["e16-c01"]["decision"].startswith("rejected: delta")),
            check("the best config of the axis is adopted",
                  adopted["lambda_l1"]["run_id"] == "e16-c03"
                  and rows["e16-c03"]["decision"].startswith("ADOPTED")),
            check("the runner-up of the axis is rejected, with the reason",
                  "lost the lambda_l1 axis to e16-c03"
                  in rows["e16-c02"]["decision"]),
            check("a diverged run is vetoed however good its SSIM",
                  "critic" not in adopted
                  and rows["e16-c12"]["decision"].startswith("vetoed")),
            check("val PSNR breaks an SSIM tie",
                  tie["adopted"]["lambda_l1"]["run_id"] == "e16-c02"),
            check("an incomplete run cannot be adopted",
                  incomplete["adopted"]["lambda_l1"]["run_id"] == "e16-c02"),
        ])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_e16_final_config() -> bool:
    root = Path(tempfile.mkdtemp())
    try:
        c0 = 0.7762
        grid = e16_grid()
        write_record(root, "e16-c01", best=c0 - 0.01, epochs=40)
        write_record(root, "e16-c02", best=c0 - 0.01, epochs=40)
        write_record(root, "e16-c03", best=c0 + 0.0080, epochs=40)
        write_record(root, "e16-c12", best=c0 + 0.0090, epochs=40)
        records = read_records(root)
        selection = select_e16(records, grid, c0)
        config = build_final_config(dict(C0, _comment="base"), selection)
        derivation = config["_derivation"]
        nothing = build_final_config(
            dict(C0), select_e16(records, grid, c0 + 1.0))
        return all([
            check("adopted axes land in the config",
                  config["lambda_l1"] == 200
                  and config["critic"] == "patch_critic_142"),
            check("untouched axes keep their C0 value",
                  config["lr"] == C0["lr"] and config["batch"] == C0["batch"]
                  and config["generator"] == C0["generator"]),
            check("run_id is rewritten for Stage 17",
                  config["run_id"] == "e17-final"),
            check("every considered axis is in the derivation",
                  len(derivation["axes"]) == len(grid["runs"])
                  and derivation["c0_result"] == c0
                  and derivation["margin"] == E16_ADOPT_MARGIN),
            check("no winner means the final config IS C0",
                  nothing["_derivation"]["applied_overrides"] == {}
                  and nothing["lambda_l1"] == C0["lambda_l1"]),
        ])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_e17_grid() -> bool:
    root = Path(tempfile.mkdtemp())
    try:
        write_record(root, "e15-c0-attempt1", best=0.7762, epochs=12,
                     epoch_seconds=271.0, iterations=180)  # 1.5056 s/iter
        records = read_records(root)
        bench = {"results": [{"status": "ok", "generator": "unet_p2p_6",
                              "critic": "patch_critic_70", "batch": 8,
                              "seconds_per_iteration": 1.9281}]}
        tau = measure_tau(records, C0, bench)
        grid = build_e17_runs("scripts/training/configs/e17_final.json", tau)
        ids = [entry["run_id"] for entry in grid["runs"]]
        seeds = {entry["run_id"]: entry["seed"] for entry in grid["runs"]}
        fractions = {entry["run_id"]: entry["overrides"]["fraction"]
                     for entry in grid["runs"]}
        deferred = grid["deferred"][0]

        heavy = measure_tau({}, dict(C0, critic="patch_critic_142"),
                            {"results": [{"status": "ok",
                                          "generator": "unet_p2p_6",
                                          "critic": "patch_critic_142",
                                          "batch": 8,
                                          "seconds_per_iteration": 2.9}]})
        heavy_grid = build_e17_runs("x.json", heavy)

        return all([
            check("six runs: four levels plus two extra 5% seeds",
                  ids == ["e17-final-f50", "e17-final-f10", "e17-final-f5",
                          "e17-final-f1", "e17-final-f5-s1", "e17-final-f5-s2"]),
            check("fractions follow the levels",
                  [fractions[i] for i in ids] == [0.5, 0.1, 0.05, 0.01,
                                                  0.05, 0.05]),
            check("seeds follow the PA-1 derivation",
                  seeds["e17-final-f1"] == 20260805170300
                  and seeds["e17-final-f5-s2"] == 20260805170202),
            check("200 epochs per run",
                  all(entry["epochs"] == 200 for entry in grid["runs"])),
            check("tau prefers the real training loop over the benchmark",
                  abs(tau["tau_loop"] - 1.5056) < 1e-3
                  and tau["tau_bench"] == 1.9281
                  and tau["tau_used"] == tau["tau_loop"]),
            check("the tau gate is recorded, not acted on",
                  deferred["tau_gate"]["gate_satisfied"] is True
                  and deferred["reason"] == "not implemented"),
            check("a slower architecture fails the gate and is still deferred",
                  heavy_grid["deferred"][0]["tau_gate"]["gate_satisfied"] is False
                  and heavy["source"].startswith("bench-felix")),
            check("the conditioned model is never a launchable run",
                  "e17-conditioned" not in ids),
        ])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_e18_grid() -> bool:
    grid = build_e18_runs("scripts/training/configs/e17_final.json", C0)
    entries = {entry["run_id"]: entry for entry in grid["runs"]}
    swapped = build_e18_runs("x.json", dict(C0, generator="unet_dc_5"))
    return all([
        check("three ablations that need a new run",
              sorted(entries) == ["e18-abl-depth4", "e18-abl-noL1",
                                  "e18-abl-noskip"]),
        check("lambda_L1 = 0 for the adversarial-only arm",
              entries["e18-abl-noL1"]["overrides"]["lambda_l1"] == 0),
        check("the no-skip and depth-4 generators are the new variants",
              entries["e18-abl-noskip"]["overrides"]["generator"]
              == "unet_p2p_6_noskip"
              and entries["e18-abl-depth4"]["overrides"]["generator"]
              == "unet_p2p_4"),
        check("120 epochs at the 5% level, PA-1 seeds",
              all(entry["epochs"] == 120
                  and entry["overrides"]["fraction"] == 0.05
                  for entry in grid["runs"])
              and entries["e18-abl-noL1"]["seed"] == 20260805180000),
        check("a non-default final generator raises a warning in the grid",
              "WARNING" in swapped["_comment"]),
    ])


def test_train_command() -> bool:
    entry = {"run_id": "e16-c11", "seed": 20260805161100, "epochs": 80,
             "patience": 25, "overrides": {"hflip": True, "rethin": True}}
    command = train_command("scripts/training/configs/e15_reference.json", entry)
    joined = " ".join(command)
    generator = train_command("base.json", {
        "run_id": "e18-abl-depth4", "seed": 1, "epochs": 120,
        "overrides": {"generator": "unet_p2p_4", "fraction": 0.05}})
    try:
        train_command("base.json", {"run_id": "x", "seed": 1,
                                    "overrides": {"nonsense": 1}})
        rejected = False
    except KeyError:
        rejected = True
    return all([
        check("run id, seed and config are passed through",
              "--run-id e16-c11" in joined and "--seed 20260805161100" in joined
              and "--config scripts/training/configs/e15_reference.json" in joined),
        check("booleans become 0/1 flags", "--hflip 1" in joined
              and "--rethin 1" in joined),
        check("epochs and patience reach the command line",
              "--epochs 80" in joined and "--patience 25" in joined),
        check("every run resumes and asks for the GPU",
              "--resume auto" in joined and "--device cuda" in joined),
        check("architecture overrides are passed through",
              "--generator unet_p2p_4" in " ".join(generator)),
        check("an override with no flag is a hard error", rejected),
    ])


def test_stable_band() -> bool:
    """The pre-registered band is 0.775-0.7777, not a rounder number."""
    root = Path(tempfile.mkdtemp())
    try:
        write_record(root, "under", best=0.7740)
        write_record(root, "floor", best=0.7750)   # NOT > 0.775
        write_record(root, "inside", best=0.7760)
        records = read_records(root)
        states = {name: classify_e15(outcome(records[name])) for name in records}
        return all([
            check("floor is 0.775", E15_STABLE_FLOOR == 0.775),
            check("0.7740 is below the band", states["under"] == "below"),
            check("0.7750 exactly is below the band", states["floor"] == "below"),
            check("0.7760 is stable", states["inside"] == "stable"),
            check("the bar is unchanged", E15_PASS_SSIM == 0.7777),
        ])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_nyquist_gate() -> bool:
    """E15 criterion 3 halts the chain instead of accepting a pass."""
    root = Path(tempfile.mkdtemp())
    try:
        write_record(root, "rising", best=0.7900, nyquist=2.2, nyquist_rise=0.05)
        write_record(root, "high-flat", best=0.7900, nyquist=2.5)
        write_record(root, "low-rising", best=0.7900, nyquist=0.5,
                     nyquist_rise=0.05)
        records = read_records(root)
        reports = {name: checkerboard_report(records[name]["metrics"])
                   for name in records}
        states = {name: classify_e15(outcome(records[name])) for name in records}

        # The same record under the Stage 15 attempt name must HALT the chain.
        halt_root = Path(tempfile.mkdtemp())
        try:
            write_record(halt_root, "e15-c0-attempt1", best=0.7900,
                         nyquist=2.2, nyquist_rise=0.05)
            decision = decide_e15(read_records(halt_root))
        finally:
            shutil.rmtree(halt_root, ignore_errors=True)

        return all([
            check("above 2 and rising is flagged",
                  reports["rising"]["flagged"]
                  and states["rising"] == "checkerboard"),
            check("above 2 but flat is not flagged",
                  not reports["high-flat"]["flagged"]
                  and states["high-flat"] == "pass"),
            check("rising but under 2 is not flagged",
                  not reports["low-rising"]["flagged"]
                  and states["low-rising"] == "pass"),
            check("a checkerboarding attempt halts instead of closing E15",
                  decision["status"] == "halt"
                  and "criterion 3" in decision["detail"],
                  decision["detail"][:70] + "..."),
        ])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_null_ssim_completion() -> bool:
    """A NaN-collapsed run is FINISHED and diverged, not eternally pending."""
    root = Path(tempfile.mkdtemp())
    try:
        write_record(root, "e16-c01", best=None, stop_reason="early_stop",
                     nan_val_ssim=True, epochs=40)
        write_record(root, "e16-c02", best=0.7800, epochs=40)
        write_record(root, "e16-c03", best=0.7800, epochs=40)
        write_record(root, "e16-c12", best=0.7800, epochs=40)
        records = read_records(root)
        oc = outcome(records["e16-c01"])
        phase = decide_queue_phase(records, e16_grid())
        selection = select_e16(records, e16_grid(), 0.7762)
        row = next(r for r in selection["rows"] if r["run_id"] == "e16-c01")
        return all([
            check("terminal stop_reason with a null score is complete",
                  oc["complete"]),
            check("and diverged", oc["diverged"], oc["divergence"]["reasons"][0]),
            check("so the phase can close", phase["complete"]),
            check("and the run is vetoed, never compared",
                  row["decision"].startswith("vetoed"), row["decision"]),
            check("E15 classifies the same record as diverged",
                  classify_e15(oc) == "diverged"),
        ])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_launch_livelock() -> bool:
    """Three launches with no new epoch stop the chain instead of looping."""
    root = Path(tempfile.mkdtemp())
    results = []
    try:
        # Pure counter: no progress raises the count, progress resets it.
        history = {}
        for _ in range(MAX_LAUNCHES_WITHOUT_PROGRESS):
            allowed, entry = launch_guard("r", history, 0)
            results.append(allowed)
            history["r"] = entry
        blocked, _ = launch_guard("r", history, 0)
        progressed, entry = launch_guard("r", history, 5)

        # A job that dies before Trainer.__init__ leaves NO record at all: the
        # driver must not resubmit it forever.
        no_record = {}
        for _ in range(MAX_LAUNCHES_WITHOUT_PROGRESS):
            decision = decide_e15({}, launch_history=no_record)
            if decision["status"] == "launch":
                no_record[decision["launch"]["run_id"]] = decision["launch_entry"]
        halted = decide_e15({}, launch_history=no_record)

        # A record that exists but never grows: same guard on the relaunch path.
        write_record(root, "e15-c0-attempt1", best=None, summary=False, epochs=3)
        records = read_records(root)
        stall_history = {}
        for _ in range(MAX_LAUNCHES_WITHOUT_PROGRESS):
            decision = decide_e15(records, launch_history=stall_history)
            if decision["status"] == "relaunch":
                stall_history[decision["launch"]["run_id"]] = \
                    decision["launch_entry"]
        stalled_halt = decide_e15(records, launch_history=stall_history)
        # ... but an epoch written since the last launch clears the counter.
        write_record(root, "e15-c0-attempt1", best=None, summary=False, epochs=9)
        recovered = decide_e15(read_records(root), launch_history=stall_history)

        return all([
            check(f"the first {MAX_LAUNCHES_WITHOUT_PROGRESS} launches are "
                  "allowed", all(results)),
            check("the next one is blocked", not blocked),
            check("a written epoch resets the counter",
                  progressed and entry["count"] == 1),
            check("no record at all halts after three submissions",
                  halted["status"] == "halt"
                  and "never written an epoch" in halted["detail"]),
            check("a frozen record halts on the relaunch path",
                  stalled_halt["status"] == "halt"
                  and "without writing a single new epoch"
                  in stalled_halt["detail"]),
            check("progress since the last launch resumes relaunching",
                  recovered["status"] == "relaunch"),
        ])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_relaunch_beside_active() -> bool:
    """A dead attempt is resubmitted even while another one is running."""
    root = Path(tempfile.mkdtemp())
    try:
        write_record(root, "e15-c0-attempt1", best=None, summary=False, epochs=5)
        write_record(root, "e15-c0-attempt2", best=None, summary=False, epochs=5)
        decision = decide_e15(read_records(root), {"e15-c0-attempt2"})
        return all([
            check("the stalled attempt is relaunched, not masked by the "
                  "running one",
                  decision["status"] == "relaunch"
                  and decision["launch"]["run_id"] == "e15-c0-attempt1"),
        ])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_resource_guards() -> bool:
    """Strike accounting, quota parsing, pruning rules and the env strip."""
    root = Path(tempfile.mkdtemp())
    try:
        quota_text = (
            "Disk quotas for user researcher (uid 5001):\n"
            "     Filesystem   space   quota   limit   grace   files\n"
            "      /dev/sdb1  19000M  20000M  20000M           12345\n")
        quota = parse_quota(quota_text)
        empty = parse_quota("quota: command not found")

        for run_id in ("e16-c01", "e17-final-f5", "e15-c0-attempt1"):
            ckpt = root / run_id / "checkpoints"
            ckpt.mkdir(parents=True)
            (ckpt / "last.pt").write_bytes(b"x" * 2048)
            (ckpt / "best.pt").write_bytes(b"y" * 2048)
        pruned = {run_id: prune_run_checkpoints(root / run_id,
                                                phase_of_run(run_id))
                  for run_id in ("e16-c01", "e17-final-f5", "e15-c0-attempt1")}

        os.environ["SLURM_MEM_PER_NODE"] = "4096"
        os.environ["SLURM_CPUS_PER_TASK"] = "2"
        os.environ["MUOGRAPHY_TEST_KEEP"] = "1"
        env = child_env()
        os.environ.pop("MUOGRAPHY_TEST_KEEP", None)

        return all([
            check("a wall-clock kill after progress costs no strike",
                  not strike_needed(10, 24)),
            check("a crash with no new epoch does cost one",
                  strike_needed(10, 10) and strike_needed(0, 0)),
            check("quota -s parses to megabytes of headroom",
                  abs(quota["headroom_mb"] - 1000.0) < 1e-6
                  and abs(quota["used_mb"] - 19000.0) < 1e-6),
            check("unparseable quota output yields no number",
                  empty["headroom_mb"] is None),
            check("a finished E16 arm loses both checkpoints",
                  pruned["e16-c01"]["removed"] == ["last.pt", "best.pt"]),
            check("E17 keeps best.pt for the weights manifest",
                  pruned["e17-final-f5"]["removed"] == ["last.pt"]
                  and (root / "e17-final-f5/checkpoints/best.pt").exists()),
            check("E15 keeps best.pt for the R4 restart ladder",
                  pruned["e15-c0-attempt1"]["removed"] == ["last.pt"]),
            check("the driver's own allocation never reaches a child sbatch",
                  all(name not in env for name in LEAKY_SLURM_VARS)
                  and env.get("MUOGRAPHY_TEST_KEEP") == "1"),
        ])
    finally:
        for name in ("SLURM_MEM_PER_NODE", "SLURM_CPUS_PER_TASK"):
            os.environ.pop(name, None)
        shutil.rmtree(root, ignore_errors=True)


def test_e16_grid_file() -> bool:
    """The frozen grid on disk must still be the pre-registered table."""
    path = REPO_ROOT / "scripts" / "training" / "configs" / "e16_grid.json"
    grid = json.loads(path.read_text(encoding="utf-8"))
    runs = grid["runs"]
    expected = [
        ("e16-c01", "lambda_l1", 10), ("e16-c02", "lambda_l1", 50),
        ("e16-c03", "lambda_l1", 200), ("e16-c04", "lr", 5e-05),
        ("e16-c05", "lr", 0.0002), ("e16-c06", "n_critic", 2),
        ("e16-c07", "generator", "unet_dc_5"),
        ("e16-c08", "generator", "unet_p2p_6_base32"),
        ("e16-c09", "batch", 16), ("e16-c10", "augmentation", "none"),
        ("e16-c11", "augmentation", "hflip+rethin"),
        ("e16-c12", "critic", "patch_critic_142"),
    ]
    got = [(entry["run_id"], entry["axis"], entry["value"]) for entry in runs]
    seeds_ok = all(entry["seed"] == seed_for(16, index)
                   for index, entry in enumerate(runs, start=1))
    budget_ok = all(entry["epochs"] == 80 and entry["patience"] == 25
                    for entry in runs)
    try:
        for entry in runs:
            train_command(grid["base_config"], entry)
        commands_ok = True
    except KeyError as error:
        commands_ok = False
        print(f"    {error}")
    return all([
        check("the 12 pre-registered axes, in order", got == expected),
        check("seeds follow 20260805*10^6 + 16*10^4 + config*10^2", seeds_ok),
        check("80 epochs, patience 25 for every config", budget_ok),
        check("every override maps to a real command-line flag", commands_ok),
        check("the base config is C0",
              grid["base_config"] == "scripts/training/configs/e15_reference.json"),
    ])


def main() -> int:
    print("Stage 15-18 chain tests")
    results = {
        "seeds": test_seeds(),
        "e16_grid_file": test_e16_grid_file(),
        "divergence": test_divergence(),
        "stable_band": test_stable_band(),
        "nyquist_gate": test_nyquist_gate(),
        "null_ssim_completion": test_null_ssim_completion(),
        "launch_livelock": test_launch_livelock(),
        "relaunch_beside_active": test_relaunch_beside_active(),
        "resource_guards": test_resource_guards(),
        "e15_outcomes": test_e15_outcomes(),
        "e15_flow": test_e15_flow(),
        "e16_selection": test_e16_selection(),
        "e16_final_config": test_e16_final_config(),
        "e17_grid": test_e17_grid(),
        "e18_grid": test_e18_grid(),
        "train_command": test_train_command(),
    }
    passed = all(results.values())
    if not passed:
        print("failed checks: "
              + ", ".join(name for name, ok in results.items() if not ok))
    print("STAGE 15-18 CHAIN TESTS:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
