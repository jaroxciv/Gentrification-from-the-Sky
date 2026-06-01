"""BiDateNet — bi-temporal Siamese U-Net change detector (paper §3.3).

A U-Net whose shared encoder is run on each date; the two encoder pyramids are
*multiplied* (then ReLU'd) at every skip level so the decoder fuses bi-temporal
co-activation, and a final 1x1 conv emits the change map. Based on the granular.ai
``fabric`` BiDateNet; architecture is kept verbatim from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class InConv(nn.Module):
    """Input double-conv stem."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv: DoubleConv = DoubleConv(in_ch, out_ch)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Down(nn.Module):
    """Max-pool then double-conv (encoder downsampling step)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.mpconv: nn.Sequential = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_ch, out_ch)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mpconv(x)


class Up(nn.Module):
    """Upsample, pad to the skip's size, concatenate, then double-conv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True) -> None:
        super().__init__()
        self.up: nn.Module
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv: DoubleConv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1, (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2)
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 conv head."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class BiDateNet(nn.Module):
    """Bi-temporal Siamese U-Net (shared encoder, multiplicative skip fusion)."""

    def __init__(self, n_channels: int, n_classes: int) -> None:
        super().__init__()
        self.inc: InConv = InConv(n_channels, 64)
        self.down1: Down = Down(64, 128)
        self.down2: Down = Down(128, 256)
        self.down3: Down = Down(256, 512)
        self.down4: Down = Down(512, 512)

        self.up1: Up = Up(1024, 256)
        self.up2: Up = Up(512, 128)
        self.up3: Up = Up(256, 64)
        self.up4: Up = Up(128, 64)
        self.outc: OutConv = OutConv(64, n_classes)

    def forward(self, x_d1: Tensor, x_d2: Tensor) -> Tensor:
        x1_d1 = self.inc(x_d1)
        x2_d1 = self.down1(x1_d1)
        x3_d1 = self.down2(x2_d1)
        x4_d1 = self.down3(x3_d1)
        x5_d1 = self.down4(x4_d1)

        x1_d2 = self.inc(x_d2)
        x2_d2 = self.down1(x1_d2)
        x3_d2 = self.down2(x2_d2)
        x4_d2 = self.down3(x3_d2)
        x5_d2 = self.down4(x4_d2)

        x = self.up1(torch.relu(x5_d2 * x5_d1), torch.relu(x4_d2 * x4_d1))
        x = self.up2(x, torch.relu(x3_d2 * x3_d1))
        x = self.up3(x, torch.relu(x2_d2 * x2_d1))
        x = self.up4(x, torch.relu(x1_d2 * x1_d1))
        return self.outc(x)
