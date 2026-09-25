"""Stage 14 critics — conditional PatchGAN Wasserstein critics.

The critic sees concat(condition, candidate) on the channel axis and outputs a
map of scores with NO sigmoid (Wasserstein critic, the value is unbounded).

  patch_critic_70   (D-70, default) C64s2-C128s2-C256s2-C512s1-C1s1, k=4
                    -> 30x30 map, receptive field 70 px = 16.4 deg of azimuth,
                    which is exactly the width of the multiple-scattering halo
                    measured in Stage 9. ~2.76 M parameters.
  patch_critic_142  (D-142, Stage 16 axis) one more stride-2 level
                    -> 14x14 map, receptive field 142 px. ~6.96 M parameters.

Note on D-142: the execution plan's parameter column repeats 2.76 M for both
critics, which is a transcription artifact — a 142 px receptive field needs the
extra 512->512 stride-1 stage and costs ~6.96 M. The receptive field is what the
Stage 16 axis compares, so it is the property kept exact here.

InstanceNorm is omitted on the first and last convolutions (pix2pix convention:
the first layer must keep the raw input contrast, the last is the score head).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchCritic(nn.Module):
    """Conditional PatchGAN critic with n_down stride-2 levels.

    n_down=3 -> receptive field  70 px (D-70)
    n_down=4 -> receptive field 142 px (D-142)
    """

    RECEPTIVE_FIELD = {3: 70, 4: 142}

    def __init__(self, n_down: int = 3, base: int = 64, in_channels: int = 2):
        super().__init__()
        if n_down not in self.RECEPTIVE_FIELD:
            raise ValueError(f"n_down must be one of {sorted(self.RECEPTIVE_FIELD)}")
        self.n_down = n_down
        self.receptive_field = self.RECEPTIVE_FIELD[n_down]

        layers = [nn.Conv2d(in_channels, base, 4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]  # no norm on the first layer
        mult_prev, mult = 1, 1
        for level in range(1, n_down):
            mult_prev, mult = mult, min(2 ** level, 8)
            layers += [
                nn.Conv2d(base * mult_prev, base * mult, 4, stride=2, padding=1,
                          bias=False),
                nn.InstanceNorm2d(base * mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        mult_prev, mult = mult, min(2 ** n_down, 8)
        layers += [
            nn.Conv2d(base * mult_prev, base * mult, 4, stride=1, padding=1,
                      bias=False),
            nn.InstanceNorm2d(base * mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * mult, 1, 4, stride=1, padding=1),  # no norm, no sigmoid
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, cond: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cond, img], dim=1))


CRITICS = {
    "patch_critic_70": lambda: PatchCritic(n_down=3),
    "patch_critic_142": lambda: PatchCritic(n_down=4),
}


def build_critic(name: str = "patch_critic_70") -> nn.Module:
    if name not in CRITICS:
        raise ValueError(f"unknown critic {name!r}; available: {sorted(CRITICS)}")
    return CRITICS[name]()
