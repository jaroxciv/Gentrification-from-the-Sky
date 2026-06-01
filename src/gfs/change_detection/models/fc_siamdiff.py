"""FC-SiamDiff — Fully-Convolutional Siamese Difference network (paper §3.3).

The notebook used ``torchgeo.models.FCSiamDiff`` (Daudt, Le Saux & Boulch 2018,
"Fully Convolutional Siamese Networks for Change Detection"). torchgeo is not a
project dependency, so the canonical FC-Siam-diff U-Net is reimplemented here as
a self-contained ``nn.Module`` with the same encoder/decoder shape: a shared
(Siamese) encoder is run on each date, the *absolute differences* of the skip
features are concatenated into the decoder, and a 1-channel change logit is
returned.

The forward signature matches the other Siamese models in this package
(``forward(x1, x2)``) so the shared training/feature code can drive it uniformly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _ConvBlock(nn.Module):
    """(conv -> BN -> ReLU) x2, the FC-Siam encoder/decoder building block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FCSiamDiff(nn.Module):
    """Fully-Convolutional Siamese Difference change-detection network.

    ``in_channels`` images go through a shared 4-stage encoder; skip features are
    differenced (abs) between the two dates and fused in a mirrored decoder. The
    final 1x1 conv emits ``classes`` change channels.
    """

    def __init__(self, in_channels: int = 1, classes: int = 1) -> None:
        super().__init__()
        # Shared (Siamese) encoder.
        self.enc1: _ConvBlock = _ConvBlock(in_channels, 16)
        self.enc2: _ConvBlock = _ConvBlock(16, 32)
        self.enc3: _ConvBlock = _ConvBlock(32, 64)
        self.enc4: _ConvBlock = _ConvBlock(64, 128)
        self.pool: nn.MaxPool2d = nn.MaxPool2d(2)

        # Decoder fuses concatenated up-features with differenced skips.
        self.up4: nn.ConvTranspose2d = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec4: _ConvBlock = _ConvBlock(128 + 64, 64)
        self.up3: nn.ConvTranspose2d = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec3: _ConvBlock = _ConvBlock(64 + 32, 32)
        self.up2: nn.ConvTranspose2d = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec2: _ConvBlock = _ConvBlock(32 + 16, 16)
        self.classify: nn.Conv2d = nn.Conv2d(16, classes, kernel_size=1)

    def _encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        return s1, s2, s3, s4

    @staticmethod
    def _fuse(up: Tensor, skip: Tensor) -> Tensor:
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([up, skip], dim=1)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        a1, a2, a3, a4 = self._encode(x1)
        b1, b2, b3, b4 = self._encode(x2)

        # Absolute differences of the Siamese skip features.
        d1, d2, d3, d4 = (
            torch.abs(a1 - b1),
            torch.abs(a2 - b2),
            torch.abs(a3 - b3),
            torch.abs(a4 - b4),
        )

        x = self.dec4(self._fuse(self.up4(d4), d3))
        x = self.dec3(self._fuse(self.up3(x), d2))
        x = self.dec2(self._fuse(self.up2(x), d1))
        return self.classify(x)
