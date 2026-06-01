"""CGNet — Change Guide Network with an attention Change Guide Module (paper §3.3).

A Siamese VGG16-BN encoder (first conv adapted to single-channel input) extracts
features for each date; the differences are reduced, an initial coarse change map
guides self-attention (the ``ChangeGuideModule``) at three decoder scales, and a
final head emits the refined change map. Architecture and forward pass are kept
verbatim from the notebook.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models


class BasicConv2d(nn.Module):
    """conv (no bias) -> BatchNorm -> ReLU."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(out_planes)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class ChangeGuideModule(nn.Module):
    """Change-guided self-attention: the coarse change map modulates Q/K/V."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.chanel_in: int = in_dim
        self.query_conv: nn.Conv2d = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv: nn.Conv2d = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv: nn.Conv2d = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.gamma: nn.Parameter = nn.Parameter(torch.zeros(1))
        self.softmax: nn.Softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, guiding_map0: Tensor) -> Tensor:
        m_batchsize, c, height, width = x.size()
        guiding_map0 = F.interpolate(
            guiding_map0, x.size()[2:], mode="bilinear", align_corners=True
        )
        guiding_map = torch.sigmoid(guiding_map0)

        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, c, height, width)
        return self.gamma * out + x


class CGNet(nn.Module):
    """Change Guide Network over a single-channel-adapted VGG16-BN backbone."""

    def __init__(self, weights: str | None = "DEFAULT") -> None:
        super().__init__()
        vgg16_bn = models.vgg16_bn(weights=weights)
        features = cast("nn.Sequential", vgg16_bn.features)
        # Modify the first layer to accept a single channel.
        features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.inc: nn.Module = features[:5]  # 64
        self.down1: nn.Module = features[5:12]  # 128
        self.down2: nn.Module = features[12:22]  # 256
        self.down3: nn.Module = features[22:32]  # 512
        self.down4: nn.Module = features[32:42]  # 512

        self.conv_reduce_1: BasicConv2d = BasicConv2d(128 * 2, 128, 3, 1, 1)
        self.conv_reduce_2: BasicConv2d = BasicConv2d(256 * 2, 256, 3, 1, 1)
        self.conv_reduce_3: BasicConv2d = BasicConv2d(512 * 2, 512, 3, 1, 1)
        self.conv_reduce_4: BasicConv2d = BasicConv2d(512 * 2, 512, 3, 1, 1)

        self.up_layer4: BasicConv2d = BasicConv2d(512, 512, 3, 1, 1)
        self.up_layer3: BasicConv2d = BasicConv2d(512, 512, 3, 1, 1)
        self.up_layer2: BasicConv2d = BasicConv2d(256, 256, 3, 1, 1)

        self.decoder: nn.Sequential = nn.Sequential(
            BasicConv2d(512, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1)
        )

        self.decoder_final: nn.Sequential = nn.Sequential(
            BasicConv2d(128, 64, 3, 1, 1), nn.Conv2d(64, 1, 1)
        )

        self.cgm_2: ChangeGuideModule = ChangeGuideModule(256)
        self.cgm_3: ChangeGuideModule = ChangeGuideModule(512)
        self.cgm_4: ChangeGuideModule = ChangeGuideModule(512)

        self.upsample2x: nn.UpsamplingBilinear2d = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_module4: BasicConv2d = BasicConv2d(1024, 512, 3, 1, 1)
        self.decoder_module3: BasicConv2d = BasicConv2d(768, 256, 3, 1, 1)
        self.decoder_module2: BasicConv2d = BasicConv2d(384, 128, 3, 1, 1)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        size = x1.size()[2:]
        layer1_pre = self.inc(x1)
        layer1_a = self.down1(layer1_pre)
        layer2_a = self.down2(layer1_a)
        layer3_a = self.down3(layer2_a)
        layer4_a = self.down4(layer3_a)

        layer1_pre = self.inc(x2)
        layer1_b = self.down1(layer1_pre)
        layer2_b = self.down2(layer1_b)
        layer3_b = self.down3(layer2_b)
        layer4_b = self.down4(layer3_b)

        layer1 = torch.cat((layer1_b, layer1_a), dim=1)
        layer2 = torch.cat((layer2_b, layer2_a), dim=1)
        layer3 = torch.cat((layer3_b, layer3_a), dim=1)
        layer4 = torch.cat((layer4_b, layer4_a), dim=1)

        layer1 = self.conv_reduce_1(layer1)
        layer2 = self.conv_reduce_2(layer2)
        layer3 = self.conv_reduce_3(layer3)
        layer4 = self.conv_reduce_4(layer4)

        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode="bilinear", align_corners=True)
        feature_fuse = layer4_1

        change_map = self.decoder(feature_fuse)

        layer4 = self.cgm_4(layer4, change_map)
        feature4 = self.decoder_module4(torch.cat([self.upsample2x(layer4), layer3], 1))
        layer3 = self.cgm_3(feature4, change_map)
        feature3 = self.decoder_module3(torch.cat([self.upsample2x(layer3), layer2], 1))
        layer2 = self.cgm_2(feature3, change_map)
        layer1 = self.decoder_module2(torch.cat([self.upsample2x(layer2), layer1], 1))

        final_map = self.decoder_final(layer1)
        final_map = F.interpolate(final_map, size, mode="bilinear", align_corners=True)
        return final_map
