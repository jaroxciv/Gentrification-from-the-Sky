"""Res-Net feature extractor for change detection (paper §3.3-3.4).

A ResNet-style encoder/decoder (a CycleGAN-generator-shaped network) trained
per-band as a one-shot autoencoder: it learns to reconstruct ``t2`` from ``t1``,
and the absolute difference of its features between the two composites becomes
the change signal. This is the only model trained for many epochs
(``CD_EPOCHS_RESNET`` = 100) rather than one-shot.

Architecture is preserved verbatim from the notebook's ``FeatureExtractor`` /
``NetBlock`` (a reflection-padded 7x7 stem, two stride-2 downsamplings,
``n_blocks`` residual blocks, then two transposed-conv upsamplings).
"""

from __future__ import annotations

from collections.abc import Callable

import torch.optim as optim
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR

NormLayer = Callable[[int], nn.Module]


class NetBlock(nn.Module):
    """Residual block: ``x + conv_block(x)`` with two padded 3x3 convs."""

    def __init__(
        self,
        dim: int,
        padding_type: str,
        norm_layer: NormLayer,
        use_dropout: bool,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self.conv_block: nn.Sequential = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(
        self,
        dim: int,
        padding_type: str,
        norm_layer: NormLayer,
        use_dropout: bool,
        use_bias: bool,
    ) -> nn.Sequential:
        conv_block: list[nn.Module] = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv_block(x)


class FeatureExtractor(nn.Module):
    """ResNet-style encoder/decoder used as the Res-Net change feature extractor."""

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int = 64,
        norm_layer: NormLayer = nn.BatchNorm2d,
        use_dropout: bool = False,
        n_blocks: int = 9,
        padding_type: str = "reflect",
    ) -> None:
        assert n_blocks >= 0
        super().__init__()
        self.input_nc: int = input_nc
        self.output_nc: int = output_nc
        self.ngf: int = ngf
        use_bias = norm_layer == nn.InstanceNorm2d

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [
                NetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        self.model: nn.Sequential = nn.Sequential(*model)

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)

    def configure_optimizers(
        self, max_epochs: int, lr: float
    ) -> tuple[optim.Optimizer, LambdaLR]:
        """SGD with the linear lr decay schedule the notebook used."""

        def lambda_rule(epoch: int) -> float:
            return 1.0 - epoch / float(max_epochs + 1)

        optimizer = optim.SGD(
            self.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
        return optimizer, scheduler
