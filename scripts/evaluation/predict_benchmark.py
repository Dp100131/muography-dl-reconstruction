#!/usr/bin/env python3
"""Stage 19 — forward-only throughput and VRAM of the final generator.

scripts/training/benchmark_throughput.py measures the TRAINING step (n_critic
critic updates plus one generator update, with backward passes and
optimizers). Stage 20 does none of that: it runs the generator under no_grad
and nothing else, and the execution plan's "about 100 img/s on the Titan X, so
~10 500 forwards in about 2 minutes of GPU" was an estimate with no
measurement behind it. This script fills that gap.

Deliberately forward-only: no optimizer, no critic, no backward, eval() mode,
torch.no_grad(), the same FP32 the two pre-tensor-core GPUs of the project
require. Random weights are enough -- the cost of a convolution does not
depend on what is in it.

Run:  python scripts/evaluation/predict_benchmark.py --device cuda \
          --out "$MUOGRAPHY_RUNS/bench-fwd.json"
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

from models import build_generator, count_params  # noqa: E402

GENERATOR_NAME = "unet_dc_5"      # the Stage 17 final architecture
BATCH_SIZES = (1, 4, 8, 16)
IMAGE_SIZE = 256
WARMUP_ITERS = 3
TIMED_ITERS = 20
STAGE_20_FORWARDS = 10_500        # the plan's forward count for the test pass


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _peak_mib(device: torch.device):
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)


def time_batch(generator, batch: int, device: torch.device) -> dict:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        x = torch.rand(batch, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                generator(x)
            _sync(device)
            t0 = time.perf_counter()
            for _ in range(TIMED_ITERS):
                generator(x)
            _sync(device)
            elapsed = time.perf_counter() - t0
        seconds = elapsed / TIMED_ITERS
        result = {"status": "ok", "batch": batch,
                  "seconds_per_forward": round(seconds, 5),
                  "images_per_second": round(batch / seconds, 2),
                  "peak_vram_mib": _peak_mib(device),
                  "stage20_seconds_estimate":
                      round(STAGE_20_FORWARDS * seconds / batch, 1)}
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" not in message and "alloc" not in message:
            raise
        result = {"status": "OOM", "batch": batch,
                  "seconds_per_forward": None, "images_per_second": None,
                  "peak_vram_mib": _peak_mib(device),
                  "stage20_seconds_estimate": None}
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"  B={batch:>2}: {result['status']} "
          f"{result.get('images_per_second')} img/s "
          f"{result.get('peak_vram_mib')} MiB "
          f"(Stage 20 estimate {result.get('stage20_seconds_estimate')} s)",
          flush=True)
    return result


def markdown_table(rows) -> str:
    lines = ["| B | s/forward | img/s | VRAM pico [MiB] | E20 estimado [s] |",
             "|---|---|---|---|---|"]
    for row in rows:
        if row["status"] != "ok":
            lines.append(f"| {row['batch']} | {row['status']} | — | "
                         f"{row['peak_vram_mib'] or '—'} | — |")
        else:
            lines.append(
                f"| {row['batch']} | {row['seconds_per_forward']:.5f} "
                f"| {row['images_per_second']:.1f} "
                f"| {row['peak_vram_mib'] if row['peak_vram_mib'] else '—'} "
                f"| {row['stage20_seconds_estimate']:.0f} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--generator", default=GENERATOR_NAME)
    parser.add_argument("--out", default=str(REPO_ROOT / "results"
                                             / "evaluation" / "bench-fwd.json"))
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"ERROR: --device {args.device} was requested but "
                  "torch.cuda.is_available() is False -- refusing to "
                  "silently benchmark on CPU instead. Pass --device cpu "
                  "if that is what you actually want.", file=sys.stderr)
            return 1
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("note: no CUDA device — CPU timings only, VRAM columns are null")

    generator = build_generator(args.generator).to(device).eval()
    info = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": "19 (forward-only inference benchmark)",
        "generator": args.generator,
        "generator_params": count_params(generator, trainable_only=False),
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(0) if device.type == "cuda"
                        else platform.processor()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "capability": (list(torch.cuda.get_device_capability(0))
                       if device.type == "cuda" else None),
        "image_size": IMAGE_SIZE,
        "warmup_iterations": WARMUP_ITERS,
        "timed_iterations": TIMED_ITERS,
        "precision": "fp32",
        "mode": "eval + no_grad, no optimizer, no critic, no backward",
        "stage20_forward_count": STAGE_20_FORWARDS,
    }
    print(json.dumps(info, indent=2))

    rows = [time_batch(generator, batch, device) for batch in BATCH_SIZES]
    payload = {**info, "results": rows}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nwrote {out_path}\n")
    print(markdown_table(rows))
    ok = any(row["status"] == "ok" for row in rows)
    print("\nSTAGE 19 FORWARD BENCHMARK:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
