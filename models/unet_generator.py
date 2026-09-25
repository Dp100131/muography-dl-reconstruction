"""Stage 14 generators — U-Net variants sized by the physics of the dataset.

The multiple-scattering halo measured in Stage 9 is 16.2 deg wide, which at
0.234 deg/px (azimuth) / 0.176 deg/px (elevation) is 69 px / 92 px. The
generator therefore needs a receptive field of at least ~92 px; the default
G-A reaches >= 190 px, so a pixel of the prediction sees the whole halo.

  unet_p2p_6        (G-A, default) 6 stride-2 levels 256 -> 4, 64..512,
                    ConvTranspose decoder with skips, ~29.2 M parameters
  unet_p2p_6_base32 same topology with base 32 (~3.4x cheaper, Stage 16 axis)
  unet_dc_5         (G-B) double 3x3 conv blocks, 5 levels 256 -> 16,
                    ~20.5 M parameters (Stage 16/18 axis)
  unet_p2p_4        same blocks, 4 stride-2 levels 256 -> 16, 64..512
                    (Stage 18 ablation A5: depth 6 vs 4)
  unet_p2p_6_noskip unet_p2p_6 with the skip concatenations removed
                    (Stage 18 ablation A4: with vs without skips)

InstanceNorm is omitted on the first encoder convolution and on the 4x4
bottleneck stages: instance statistics over a 4x4 map are degenerate and make
training noisy. Both GPUs available (GTX 1070, GTX Titan X) are pre-tensor-core,
so everything runs in pure FP32.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _down(cin: int, cout: int, norm: bool = True) -> nn.Sequential:
    """4x4 stride-2 convolution: halves the spatial size."""
    layers = [nn.Conv2d(cin, cout, 4, stride=2, padding=1, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(cout))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _up(cin: int, cout: int, norm: bool = True) -> nn.Sequential:
    """4x4 stride-2 transposed convolution: doubles the spatial size."""
    layers = [nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1,
                                 bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(cout))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNetPix2Pix6(nn.Module):
    """G-A: 6 stride-2 levels, 256 -> 4 -> 256, skip concatenations."""

    def __init__(self, base: int = 64):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        c4 = c5 = c6 = base * 8

        self.enc1 = _down(1, c1, norm=False)   # 256 -> 128   (no norm: first)
        self.enc2 = _down(c1, c2)              # 128 ->  64
        self.enc3 = _down(c2, c3)              #  64 ->  32
        self.enc4 = _down(c3, c4)              #  32 ->  16
        self.enc5 = _down(c4, c5)              #  16 ->   8
        self.enc6 = _down(c5, c6, norm=False)  #   8 ->   4   (no norm: 4x4)

        self.dec6 = _up(c6, c5, norm=False)    #   4 ->   8   (no norm: from 4x4)
        self.dec5 = _up(c5 + c5, c4)           #   8 ->  16
        self.dec4 = _up(c4 + c4, c3)           #  16 ->  32
        self.dec3 = _up(c3 + c3, c2)           #  32 ->  64
        self.dec2 = _up(c2 + c2, c1)           #  64 -> 128
        self.dec1 = _up(c1 + c1, c1)           # 128 -> 256
        self.head = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)

        d = self.dec6(e6)
        d = self.dec5(torch.cat([d, e5], dim=1))
        d = self.dec4(torch.cat([d, e4], dim=1))
        d = self.dec3(torch.cat([d, e3], dim=1))
        d = self.dec2(torch.cat([d, e2], dim=1))
        d = self.dec1(torch.cat([d, e1], dim=1))
        return torch.sigmoid(self.head(d))


class UNetPix2Pix4(nn.Module):
    """A5 depth arm: the same blocks as G-A over 4 levels, 256 -> 16 -> 256.

    The bottleneck is 16x16, so unlike the 6-level variant every stage keeps
    its InstanceNorm (the 4x4 degeneracy that motivates dropping norm in
    enc6/dec6 of UNetPix2Pix6 simply does not arise here); the first encoder
    convolution stays norm-free, as in every pix2pix encoder of this zoo.
    """

    def __init__(self, base: int = 64):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.enc1 = _down(1, c1, norm=False)   # 256 -> 128   (no norm: first)
        self.enc2 = _down(c1, c2)              # 128 ->  64
        self.enc3 = _down(c2, c3)              #  64 ->  32
        self.enc4 = _down(c3, c4)              #  32 ->  16

        self.dec4 = _up(c4, c3)                #  16 ->  32
        self.dec3 = _up(c3 + c3, c2)           #  32 ->  64
        self.dec2 = _up(c2 + c2, c1)           #  64 -> 128
        self.dec1 = _up(c1 + c1, c1)           # 128 -> 256
        self.head = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d = self.dec4(e4)
        d = self.dec3(torch.cat([d, e3], dim=1))
        d = self.dec2(torch.cat([d, e2], dim=1))
        d = self.dec1(torch.cat([d, e1], dim=1))
        return torch.sigmoid(self.head(d))


class UNetPix2Pix6NoSkip(nn.Module):
    """A4 ablation arm: UNetPix2Pix6 with the skip concatenations removed.

    Encoder and decoder keep the topology, the channel widths and the norm
    placement of UNetPix2Pix6; only the concatenations disappear, so each
    decoder stage takes half the input channels of its skipped counterpart
    (c5 instead of c5 + c5, and so on). Everything the output can use must
    therefore pass through the 4x4 bottleneck -- which is exactly the claim
    the ablation tests.
    """

    def __init__(self, base: int = 64):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        c4 = c5 = c6 = base * 8

        self.enc1 = _down(1, c1, norm=False)   # 256 -> 128   (no norm: first)
        self.enc2 = _down(c1, c2)              # 128 ->  64
        self.enc3 = _down(c2, c3)              #  64 ->  32
        self.enc4 = _down(c3, c4)              #  32 ->  16
        self.enc5 = _down(c4, c5)              #  16 ->   8
        self.enc6 = _down(c5, c6, norm=False)  #   8 ->   4   (no norm: 4x4)

        self.dec6 = _up(c6, c5, norm=False)    #   4 ->   8   (no norm: from 4x4)
        self.dec5 = _up(c5, c4)                #   8 ->  16   (no skip: c5, not 2*c5)
        self.dec4 = _up(c4, c3)                #  16 ->  32
        self.dec3 = _up(c3, c2)                #  32 ->  64
        self.dec2 = _up(c2, c1)                #  64 -> 128
        self.dec1 = _up(c1, c1)                # 128 -> 256
        self.head = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.enc1(x)
        e = self.enc2(e)
        e = self.enc3(e)
        e = self.enc4(e)
        e = self.enc5(e)
        e = self.enc6(e)

        d = self.dec6(e)
        d = self.dec5(d)
        d = self.dec4(d)
        d = self.dec3(d)
        d = self.dec2(d)
        d = self.dec1(d)
        return torch.sigmoid(self.head(d))


class UNetDoubleConv5(nn.Module):
    """G-B: the toy UNetLite double-3x3 pattern extended to 5 levels."""

    def __init__(self, base: int = 64):
        super().__init__()
        channels = [base, base * 2, base * 4, base * 8, base * 8]
        self.enc1 = self._block(1, channels[0])
        self.enc2 = self._block(channels[0], channels[1])
        self.enc3 = self._block(channels[1], channels[2])
        self.enc4 = self._block(channels[2], channels[3])
        self.enc5 = self._block(channels[3], channels[4])
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], 2, stride=2)
        self.dec4 = self._block(channels[3] + channels[3], channels[3])
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3 = self._block(channels[2] + channels[2], channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2 = self._block(channels[1] + channels[1], channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.dec1 = self._block(channels[0] + channels[0], channels[0])
        self.head = nn.Conv2d(channels[0], 1, 1)

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.InstanceNorm2d(cout),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.InstanceNorm2d(cout),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                     # 256
        e2 = self.enc2(self.pool(e1))         # 128
        e3 = self.enc3(self.pool(e2))         #  64
        e4 = self.enc4(self.pool(e3))         #  32
        e5 = self.enc5(self.pool(e4))         #  16

        d4 = self.dec4(torch.cat([self.up4(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.head(d1))


GENERATORS = {
    "unet_p2p_6": lambda: UNetPix2Pix6(base=64),
    "unet_p2p_6_base32": lambda: UNetPix2Pix6(base=32),
    "unet_p2p_6_noskip": lambda: UNetPix2Pix6NoSkip(base=64),
    "unet_p2p_4": lambda: UNetPix2Pix4(base=64),
    "unet_dc_5": lambda: UNetDoubleConv5(base=64),
}


def build_generator(name: str = "unet_p2p_6") -> nn.Module:
    if name not in GENERATORS:
        raise ValueError(
            f"unknown generator {name!r}; available: {sorted(GENERATORS)}")
    return GENERATORS[name]()
