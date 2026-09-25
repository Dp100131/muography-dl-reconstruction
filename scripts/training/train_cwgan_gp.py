#!/usr/bin/env python3
"""Single training entry point for Stages 15 to 18 — cWGAN-GP.

A run is fully described by its JSON config plus the seed, so every command
line below is reproducible from the record written to
results/training/records/<run_id>/config.json.

Seed convention (protocol PA-1):
    seed = 20260805 * 10**6 + stage * 10**4 + config * 10**2 + repetition

Run:  python scripts/training/train_cwgan_gp.py --config scripts/training/configs/e15_reference.json
      python scripts/training/train_cwgan_gp.py --config ... --fraction 0.01 --epochs 200 \
             --run-id e17-f01 --seed 20260805170200
      python scripts/training/train_cwgan_gp.py --config ... --overfit-n 10 --max-iters 500 \
             --run-id e14-overfit10 --seed 20260805140000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

from engine import Trainer, run_dir_for  # noqa: E402


def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None,
                        help="JSON config; omit to use the engine defaults")
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lambda-l1", type=float, default=None)
    parser.add_argument("--lambda-adv", type=float, default=None)
    parser.add_argument("--lambda-gp", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None,
                        help="overrides lr for both the generator and "
                             "critic optimizers")
    parser.add_argument("--n-critic", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None,
                        help="overrides early_stop.patience")
    parser.add_argument("--input-scaling", default=None,
                        choices=["shared", "ref_equivalent"])
    parser.add_argument("--hflip", type=int, default=None, choices=[0, 1])
    parser.add_argument("--rethin", type=int, default=None, choices=[0, 1],
                        help="redraw the input as Binomial(ref, f) every "
                             "access (Stage 16 c11 augmentation axis); left "
                             "unset the key never enters the config, so "
                             "config_sha8 of every other run is unchanged")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-root", default=None,
                        help="default: $MUOGRAPHY_RUNS, else runs/")
    parser.add_argument("--device", default=None)
    parser.add_argument("--overfit-n", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--resume", default="auto", choices=["auto", "none"])
    parser.add_argument("--generator", default=None)
    parser.add_argument("--critic", default=None)
    return parser


def load_config(path=None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    # Underscore keys are documentation, not parameters: e15_reference.json
    # carries "_comment" and the generated e17_final.json carries a whole
    # "_derivation" block. They must never reach the Trainer, because
    # config_sha8 hashes the config dict and a note about how the config was
    # chosen would change the identity of the run it describes.
    config = {key: value for key, value in config.items()
              if not str(key).startswith("_")}
    mapping = {
        "fraction": args.fraction,
        "epochs": args.epochs,
        "batch": args.batch,
        "lambda_l1": args.lambda_l1,
        "lambda_adv": args.lambda_adv,
        "lambda_gp": args.lambda_gp,
        "lr": args.lr,
        "n_critic": args.n_critic,
        "val_every": args.val_every,
        "sample_every": args.sample_every,
        "input_scaling": args.input_scaling,
        "seed": args.seed,
        "run_id": args.run_id,
        "device": args.device,
        "overfit_n": args.overfit_n,
        "max_iters": args.max_iters,
        "generator": args.generator,
        "critic": args.critic,
    }
    for key, value in mapping.items():
        if value is not None:
            config[key] = value
    if args.hflip is not None:
        config["hflip"] = bool(args.hflip)
    if args.rethin is not None:
        config["rethin"] = bool(args.rethin)
    if args.patience is not None:
        # Partial on purpose: engine.Trainer deep-merges this onto
        # DEFAULT_CONFIG["early_stop"], so metric/min_delta survive even
        # though only patience is overridden here.
        early_stop = dict(config.get("early_stop") or {})
        early_stop["patience"] = args.patience
        config["early_stop"] = early_stop
    return config


def run(config: dict, args: argparse.Namespace) -> int:
    run_id = config.get("run_id") or "run"
    run_dir = run_dir_for(run_id, args.out_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(config, run_dir)
    summary = trainer.train(resume=args.resume)

    label = "train_ssim" if config.get("overfit_n") else "best_val_ssim"
    value = summary["best_val_ssim"]
    print(f"RUN {run_id}: DONE {label}="
          f"{'nan' if value is None else f'{value:.4f}'} "
          f"epochs={summary['epochs_run']} stop={summary['stop_reason']} "
          f"wall={summary['wall_seconds']:.0f}s")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    config = apply_overrides(load_config(args.config), args)
    return run(config, args)


if __name__ == "__main__":
    sys.exit(main())
