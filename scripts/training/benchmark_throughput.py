#!/usr/bin/env python3
"""Stage 14.6 — measured 256x256 throughput and VRAM, per architecture pair.

Replaces the parametric cost model of the execution plan (tau in 1.7-4.0 s/iter
at B=8) with numbers measured on the actual hardware, which then becomes the
compute budget table for Stages 15 to 18. One "iteration" is the real training
step: n_critic critic steps plus one generator step, on synthetic tensors so
the measurement is free of any data-loading effect.

Both target GPUs are pre-tensor-core (GTX 1070 sm_61, GTX Titan X sm_52), so
everything is timed in pure FP32 — AMP would not help.

Run:  python scripts/training/benchmark_throughput.py --device cuda \
          --out "$MUOGRAPHY_RUNS/bench-felix.json"
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

from models import build_critic, build_generator, count_params  # noqa: E402

from losses import critic_loss, generator_loss  # noqa: E402

GENERATOR_NAMES = ("unet_p2p_6", "unet_dc_5")
CRITIC_NAMES = ("patch_critic_70", "patch_critic_142")
BATCH_SIZES = (2, 4, 8, 16)
IMAGE_SIZE = 256
WARMUP_ITERS = 3
TIMED_ITERS = 10
N_CRITIC = 5
LAMBDA_GP = 10.0
LAMBDA_L1 = 100.0
LR = 1e-4
BETAS = (0.0, 0.9)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _peak_mib(device: torch.device):
    """Peak allocated VRAM, or None on CPU (json.dumps must not emit NaN)."""
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)


def time_configuration(generator_name: str, critic_name, batch: int,
                       device: torch.device) -> dict:
    """Time WARMUP + TIMED full iterations; return timings or an OOM marker."""
    label = f"{generator_name}+{critic_name or 'none(L1)'}@B{batch}"
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        generator = build_generator(generator_name).to(device)
        opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=BETAS)
        critic = opt_c = None
        if critic_name is not None:
            critic = build_critic(critic_name).to(device)
            opt_c = torch.optim.Adam(critic.parameters(), lr=LR, betas=BETAS)

        cond = torch.rand(batch, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
        target = torch.rand(batch, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)

        def one_iteration() -> None:
            if critic is not None:
                for _ in range(N_CRITIC):
                    with torch.no_grad():
                        fake = generator(cond)
                    loss_c, _ = critic_loss(critic, cond, target, fake, LAMBDA_GP)
                    opt_c.zero_grad(set_to_none=True)
                    loss_c.backward()
                    opt_c.step()
            fake = generator(cond)
            loss_g, _ = generator_loss(critic, cond, fake, target,
                                       lambda_l1=LAMBDA_L1,
                                       lambda_adv=0.0 if critic is None else 1.0)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

        for _ in range(WARMUP_ITERS):
            one_iteration()
        _sync(device)

        t0 = time.perf_counter()
        for _ in range(TIMED_ITERS):
            one_iteration()
        _sync(device)
        elapsed = time.perf_counter() - t0

        seconds_per_iter = elapsed / TIMED_ITERS
        result = {
            "status": "ok",
            "generator": generator_name,
            "critic": critic_name,
            "batch": batch,
            "seconds_per_iteration": round(seconds_per_iter, 4),
            "images_per_second": round(batch / seconds_per_iter, 2),
            "peak_vram_mib": _peak_mib(device),
            "generator_params": count_params(generator),
            "critic_params": count_params(critic) if critic else 0,
        }
    except RuntimeError as exc:
        # torch.cuda.OutOfMemoryError subclasses RuntimeError, but a starved
        # allocator also surfaces as CUDNN_STATUS_ALLOC_FAILED or
        # CUBLAS_STATUS_ALLOC_FAILED — all three mean "this batch does not
        # fit". Anything else is a real bug and must propagate.
        message = str(exc).lower()
        if "out of memory" not in message and "alloc" not in message:
            raise
        result = {"status": "OOM", "generator": generator_name,
                  "critic": critic_name, "batch": batch,
                  "seconds_per_iteration": None, "images_per_second": None,
                  "peak_vram_mib": _peak_mib(device)}
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"  {label}: {result['status']} "
          f"{result.get('seconds_per_iteration')} s/iter "
          f"{result.get('images_per_second')} img/s "
          f"{result.get('peak_vram_mib')} MiB", flush=True)
    return result


def markdown_table(rows) -> str:
    header = ("| generador | crítico | B | s/iter | img/s | VRAM pico [MiB] |\n"
              "|---|---|---|---|---|---|")
    lines = [header]
    for row in rows:
        critic = row["critic"] or "ninguno (solo L1)"
        vram = ("—" if row["peak_vram_mib"] is None
                else f"{row['peak_vram_mib']:.0f}")
        if row["status"] != "ok":
            lines.append(f"| {row['generator']} | {critic} | {row['batch']} "
                         f"| {row['status']} | — | {vram} |")
        else:
            lines.append(
                f"| {row['generator']} | {critic} | {row['batch']} "
                f"| {row['seconds_per_iteration']:.3f} "
                f"| {row['images_per_second']:.1f} | {vram} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(REPO_ROOT / "runs"
                                             / "benchmark_throughput.json"))
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
        print("note: no CUDA device — CPU timings only, VRAM columns are NaN")

    info = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(0)
                        if device.type == "cuda" else platform.processor()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "capability": (list(torch.cuda.get_device_capability(0))
                       if device.type == "cuda" else None),
        "image_size": IMAGE_SIZE,
        "n_critic": N_CRITIC,
        "warmup_iterations": WARMUP_ITERS,
        "timed_iterations": TIMED_ITERS,
        "precision": "fp32",
    }
    print(json.dumps(info, indent=2))

    rows = []
    for generator_name in GENERATOR_NAMES:
        for critic_name in CRITIC_NAMES:
            for batch in BATCH_SIZES:
                rows.append(time_configuration(generator_name, critic_name,
                                               batch, device))
    print("pure-L1 mode (no critic):")
    for batch in BATCH_SIZES:
        rows.append(time_configuration("unet_p2p_6", None, batch, device))

    payload = {**info, "results": rows}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nwrote {out_path}\n")
    print(markdown_table(rows))
    print("\nSTAGE 14.6 BENCHMARK: COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
