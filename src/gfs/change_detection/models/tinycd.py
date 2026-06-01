"""TinyCD — a tiny change-detection network on an EfficientNet encoder (paper §3.3).

A lightweight Siamese model (Codegoni, Lombardi & Ferrari 2022, "TINYCD: A (Not So)
Deep Learning Model For Change Detection"). A pretrained EfficientNet-B0 backbone
(first conv adapted to single-channel input) encodes each date; mixing-and-masking
attention blocks fuse the two streams, grouped-conv up-masking decodes, and a
pixelwise-linear head emits the change map. Architecture is verbatim from the
notebook.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tv_models
from torch import Tensor, nn
from torch.nn import Conv2d, InstanceNorm2d, Module, PReLU, Sequential, Upsample
from torch.optim.lr_scheduler import LambdaLR


class PixelwiseLinear(Module):
    """A stack of 1x1 convs with PReLU activations (a per-pixel MLP)."""

    def __init__(
        self,
        fin: list[int],
        fout: list[int],
        last_activation: Module | None = None,
    ) -> None:
        assert len(fout) == len(fin)
        super().__init__()
        n = len(fin)
        self._linears: Sequential = Sequential(
            *[
                Sequential(
                    Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                    PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._linears(x)


class MixingBlock(Module):
    """Interleave/concatenate two feature maps and mix with a 3x3 conv."""

    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.convmix: nn.Sequential = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.InstanceNorm2d(ch_out),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        mixed = torch.cat((x, y), dim=1)
        return self.convmix(mixed)


class MixingMaskAttentionBlock(Module):
    """Use a grouped convolution to make a sort of attention."""

    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        fin: list[int],
        fout: list[int],
        generate_masked: bool = False,
    ) -> None:
        super().__init__()
        self._mixing: MixingBlock = MixingBlock(ch_in, ch_out)
        self._linear: PixelwiseLinear = PixelwiseLinear(fin, fout)
        self._final_normalization: InstanceNorm2d | None = (
            InstanceNorm2d(ch_out) if generate_masked else None
        )
        self._mixing_out: MixingBlock | None = (
            MixingBlock(ch_in, ch_out) if generate_masked else None
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        z_mix = self._mixing(x, y)
        z = self._linear(z_mix)
        z_mix_out = 0 if self._mixing_out is None else self._mixing_out(x, y)
        return (
            z
            if self._final_normalization is None
            else self._final_normalization(z_mix_out * z)
        )


class UpMask(Module):
    """Bilinear upsample, optionally masked by ``y``, then a grouped-conv block."""

    def __init__(self, scale_factor: float, nin: int, nout: int) -> None:
        super().__init__()
        self._upsample: Upsample = Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )
        self._convolution: Sequential = Sequential(
            Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin, bias=False),
            PReLU(),
            InstanceNorm2d(nin),
            Conv2d(nin, nout, kernel_size=1, stride=1),
            PReLU(),
            InstanceNorm2d(nout),
        )

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        x = self._upsample(x)
        if y is not None:
            y = F.interpolate(y, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = x * y
        return self._convolution(x)


def modify_efficientnet_for_single_channel(model: nn.Module) -> nn.Module:
    """Replace the first 3-channel conv of an EfficientNet with a 1-channel one."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(
                1,
                module.out_channels,
                kernel_size=module.kernel_size,  # pyright: ignore[reportArgumentType]
                stride=module.stride,  # pyright: ignore[reportArgumentType]
                padding=module.padding,  # pyright: ignore[reportArgumentType]
                bias=module.bias is not None,
            )
            model._modules[name] = new_conv
            break
        if isinstance(module, nn.Sequential):
            for seq_name, seq_module in module.named_children():
                if isinstance(seq_module, nn.Conv2d) and seq_module.in_channels == 3:
                    new_conv = nn.Conv2d(
                        1,
                        seq_module.out_channels,
                        kernel_size=seq_module.kernel_size,  # pyright: ignore[reportArgumentType]
                        stride=seq_module.stride,  # pyright: ignore[reportArgumentType]
                        padding=seq_module.padding,  # pyright: ignore[reportArgumentType]
                        bias=seq_module.bias is not None,
                    )
                    module._modules[seq_name] = new_conv
                    break
    return model


def _get_backbone(
    bkbn_name: str,
    weights: str | None,
    output_layer_bkbn: str,
    freeze_backbone: bool,
) -> nn.ModuleList:
    """Slice a torchvision EfficientNet up to ``output_layer_bkbn``, 1-channel input."""
    backbone_fn: Callable[..., nn.Module] = getattr(tv_models, bkbn_name)
    entire_model = cast("nn.Module", backbone_fn(weights=weights).features)
    entire_model = modify_efficientnet_for_single_channel(entire_model)

    derived_model = nn.ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break

    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model


class TinyCD(Module):
    """TinyCD Siamese change detector over a sliced EfficientNet backbone."""

    def __init__(
        self,
        bkbn_name: str = "efficientnet_b0",
        weights: str | None = "DEFAULT",
        output_layer_bkbn: str = "3",
        freeze_backbone: bool = False,
        bkbn_out_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        self._backbone: nn.ModuleList = _get_backbone(
            bkbn_name, weights, output_layer_bkbn, freeze_backbone
        )

        backbone_out_channels = {
            "efficientnet_b0": bkbn_out_channels,
            "efficientnet_b4": bkbn_out_channels,
        }[bkbn_name]
        assert backbone_out_channels is not None

        self._first_mix: MixingMaskAttentionBlock = MixingMaskAttentionBlock(
            2,
            backbone_out_channels[0],
            [backbone_out_channels[0]],
            [backbone_out_channels[0]],
        )
        self._mixing_mask: nn.ModuleList = nn.ModuleList(
            [
                MixingMaskAttentionBlock(
                    backbone_out_channels[0],
                    backbone_out_channels[1],
                    [backbone_out_channels[1]],
                    [backbone_out_channels[1]],
                ),
                MixingMaskAttentionBlock(
                    backbone_out_channels[1],
                    backbone_out_channels[2],
                    [backbone_out_channels[2]],
                    [backbone_out_channels[2]],
                ),
                MixingBlock(backbone_out_channels[2] * 2, backbone_out_channels[3]),
            ]
        )

        self._up: nn.ModuleList = nn.ModuleList(
            [
                UpMask(2, backbone_out_channels[3], backbone_out_channels[2]),
                UpMask(2, backbone_out_channels[2], backbone_out_channels[1]),
                UpMask(2, backbone_out_channels[1], backbone_out_channels[0]),
            ]
        )

        self._classify: PixelwiseLinear = PixelwiseLinear(
            [
                backbone_out_channels[0],
                int(backbone_out_channels[0] / 2),
                int(backbone_out_channels[0] / 4),
            ],
            [
                int(backbone_out_channels[0] / 2),
                int(backbone_out_channels[0] / 4),
                1,
            ],
            nn.Identity(),
        )

    def forward(self, ref: Tensor, test: Tensor) -> Tensor:
        features = self._encode(ref, test)
        latents = self._decode(features)
        return self._classify(latents)

    def _encode(self, ref: Tensor, test: Tensor) -> list[Tensor]:
        features = [self._first_mix(ref, test)]
        for num, layer in enumerate(self._backbone):
            ref, test = layer(ref), layer(test)
            if num != 0:
                features.append(self._mixing_mask[num - 1](ref, test))
        return features

    def _decode(self, features: list[Tensor]) -> Tensor:
        num_up_layers = len(self._up)
        num_features = len(features)
        upping = self._up[-1](features[-1])
        for i in range(1, min(num_up_layers, num_features)):
            upping = self._up[-i - 1](upping, features[-i - 1])
        return upping

    def configure_optimizers(
        self, max_epochs: int, lr: float
    ) -> tuple[optim.Optimizer, LambdaLR]:
        """SGD + linear lr decay (the TinyCD reference schedule)."""

        def lambda_rule(epoch: int) -> float:
            return 1.0 - epoch / float(max_epochs + 1)

        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
        return optimizer, scheduler
