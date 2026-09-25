"""Stage 14 model zoo — generators and critics addressed by name.

Names are the strings that appear in scripts/training/configs/*.json, so a run
is fully described by its config plus its seed. Import as a package after
putting the repo root on sys.path:  from models import build_generator, build_critic
"""

from __future__ import annotations

from .patch_critic import CRITICS, PatchCritic, build_critic
from .unet_generator import (GENERATORS, UNetDoubleConv5, UNetPix2Pix4,
                             UNetPix2Pix6, UNetPix2Pix6NoSkip, build_generator)


def count_params(module, trainable_only: bool = True) -> int:
    """Number of parameters of a module."""
    params = [p for p in module.parameters()
              if p.requires_grad or not trainable_only]
    return int(sum(p.numel() for p in params))


__all__ = [
    "CRITICS", "GENERATORS", "PatchCritic", "UNetDoubleConv5", "UNetPix2Pix4",
    "UNetPix2Pix6", "UNetPix2Pix6NoSkip", "build_critic", "build_generator",
    "count_params",
]
