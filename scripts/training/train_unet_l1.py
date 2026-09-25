#!/usr/bin/env python3
"""Stage 13.3 — plain U-Net L1 baseline (no critic, no adversarial term).

Thin wrapper over scripts/training/train_cwgan_gp.py: same arguments, but
lambda_adv is forced to 0 and the critic is dropped, which makes an iteration
~14x cheaper than the full cWGAN-GP. This is the A1 row of the Stage 18
ablation table and the reference the model must beat at every level.

Run:  python scripts/training/train_unet_l1.py --fraction 0.05 --seed 20260805130200
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "training"))

from train_cwgan_gp import apply_overrides, build_parser, load_config, run  # noqa: E402

DEFAULT_EPOCHS = 150
RUN_ID_PREFIX = "e13-unetl1-f"


def main() -> int:
    args = build_parser(description=__doc__).parse_args()
    config = apply_overrides(load_config(args.config), args)

    config["lambda_adv"] = 0.0
    config["critic"] = None
    if args.epochs is None:
        config["epochs"] = DEFAULT_EPOCHS
    if args.run_id is None:
        fraction = float(config.get("fraction", 0.05))
        config["run_id"] = f"{RUN_ID_PREFIX}{round(fraction * 100):02d}"
    return run(config, args)


if __name__ == "__main__":
    sys.exit(main())
