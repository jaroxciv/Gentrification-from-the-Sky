"""Experimental change-detection methods explored but NOT in the final paper.

The notebook tried several extra approaches beyond the six canonical methods
(paper §3.3). The self-contained, dependency-light ones are preserved here for
reproducibility; they are *not* part of the reported pipeline and are not wired
into :data:`gfs.config.CD_METHODS`.

Included (faithful to the notebook):

- :func:`siroc` — SiROC, an unsupervised spatial-context change detector.
- :func:`pca_kmeans` — classic PCA + K-Means change detection on the diff image.
- :class:`ContextGuidedBlock`, :class:`DiffFPN`, :class:`SimpleBackbone` — the
  LSNet (LightSiamese Network) building blocks.

Omitted (depend on packages outside the project's environment, see module todos):
UNet / Siam-Nested-UNet (``segmentation-models-pytorch``) and RCF
(``torchgeo``). The architectures live in ``_staging`` if needed later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch import Tensor, nn

FloatArray = npt.NDArray[np.float64]


def _import_cv2() -> Any:
    """Lazily import OpenCV (only SiROC needs it; not a project dependency)."""
    import cv2  # pyright: ignore[reportMissingImports]

    return cv2


# --- SiROC -------------------------------------------------------------------
def _obtain_change_map(
    pre: Tensor, post: Tensor, neighborhood: int, excluded: int = 0
) -> Tensor:
    """Spatial-context change map for a pre/post pair (SiROC core, verbatim)."""
    b, c, h_pre, w_pre = pre.shape
    _, _, h_post, w_post = post.shape

    padded_pre = torch.zeros((b, c, h_pre + 2 * neighborhood, w_pre + 2 * neighborhood))
    padded_pre[:, :, neighborhood : h_pre + neighborhood, neighborhood : w_pre + neighborhood] = pre
    padded_post = torch.zeros((b, c, h_pre + 2 * neighborhood, w_pre + 2 * neighborhood))
    padded_post[:, :, neighborhood : h_post + neighborhood, neighborhood : w_post + neighborhood] = post

    pre_response = padded_pre**2
    post_response = padded_pre * padded_post
    pre_sum = torch.zeros(post.shape)
    post_sum = torch.zeros(post.shape)

    for x_patch in range(-neighborhood, neighborhood + 1):
        for y_patch in range(-neighborhood, neighborhood + 1):
            if abs(x_patch) <= excluded or abs(y_patch) <= excluded:
                continue
            pre_sum += pre_response[
                :,
                :,
                y_patch + neighborhood : h_pre + y_patch + neighborhood,
                x_patch + neighborhood : w_pre + x_patch + neighborhood,
            ]
            post_sum += post_response[
                :,
                :,
                y_patch + neighborhood : h_post + y_patch + neighborhood,
                x_patch + neighborhood : w_post + x_patch + neighborhood,
            ]

    post_pred = pre * post_sum / pre_sum
    return torch.abs(post_pred - post)


def _apply_threshold(
    change_map: Tensor, j: Tensor, threshold: str, layer: int, otsu_factor: float
) -> Tensor:
    """Otsu/Triangle thresholding of a SiROC change band (verbatim)."""
    cv2 = _import_cv2()
    if threshold == "Otsu":
        img = np.int8(np.array(j * 255).ravel())
        assert not np.isnan(img).any()
        t = cv2.threshold(
            np.array(abs(j.numpy() * 255), dtype=np.uint8),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[0]
        change_map[layer, :, :] = torch.where(
            abs(j) > (t * otsu_factor / 255), torch.tensor(1), torch.tensor(0)
        )
    elif threshold == "Triangle":
        t = cv2.threshold(
            np.array(abs(j.numpy() * 255), dtype=np.uint8),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE,
        )[0]
        change_map[layer, :, :] = torch.where(
            abs(j) > (t * 0.5 * otsu_factor / 255), torch.tensor(1), torch.tensor(0)
        )
    else:
        raise AssertionError("Thresholding not identified")
    return change_map


def siroc(
    pre_image: FloatArray,
    post_image: FloatArray,
    neighborhood: int = 3,
    excluded: int = 0,
    threshold: str = "Otsu",
    otsu_factor: float = 1.0,
) -> Tensor:
    """SiROC unsupervised change detection (experimental; needs OpenCV)."""
    pre = torch.Tensor(pre_image).unsqueeze(0)
    post = torch.Tensor(post_image).unsqueeze(0)
    change_map = _obtain_change_map(pre, post, neighborhood, excluded)
    for layer in range(change_map.shape[0]):
        j = change_map[layer, 0, :, :]
        change_map = _apply_threshold(change_map, j, threshold, layer, otsu_factor)
    return change_map[0, 0, :, :]


# --- PCA + KMeans ------------------------------------------------------------
def pca_kmeans(
    im1: FloatArray, im2: FloatArray, block_size: int = 3, rate: float = 0.9
) -> npt.NDArray[np.int_]:
    """Classic PCA + K-Means change detection on the difference image (verbatim)."""
    image_size = im1.shape
    padding_size = (
        image_size[0],
        image_size[1] + block_size - 1,
        image_size[2] + block_size - 1,
    )

    delta = np.abs(im1 - im2)

    padding_img = np.zeros(padding_size)
    lb = block_size // 2
    ub_col = lb + image_size[1] - 1
    ub_row = lb + image_size[2] - 1
    padding_img[:, lb : ub_col + 1, lb : ub_row + 1] = delta

    vk: list[FloatArray] = []
    for i in range(image_size[1]):
        for j in range(image_size[2]):
            vk_temp = padding_img[:, i : i + block_size, j : j + block_size]
            vk.append(vk_temp.flatten())
    vk_arr = np.array(vk)

    mean_val = np.mean(vk_arr, axis=0)
    std_val = np.std(vk_arr, axis=0) + 1e-12
    vk_arr = (vk_arr - mean_val) / std_val

    pca = PCA(n_components=rate)
    feature = pca.fit_transform(vk_arr)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(feature)
    labels = np.asarray(kmeans.labels_)

    return labels.reshape(image_size[1], image_size[2]).astype(np.int_)


# --- LSNet (LightSiamese Network) building blocks ----------------------------
class ContextGuidedBlock(nn.Module):
    """CGNet-style context-guided block: local + surrounding context with SE gate."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        reduction: int,
        skip_connect: bool = True,
    ) -> None:
        super().__init__()
        self.conv1x1: nn.Conv2d = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.f_loc: nn.Conv2d = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False
        )
        self.f_sur: nn.Conv2d = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=dilation,
            dilation=dilation,
            groups=out_channels,
            bias=False,
        )
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.skip_connect: bool = skip_connect
        self.global_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self.fc: nn.Sequential = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1x1(x)))
        loc = self.f_loc(x)
        sur = self.f_sur(x)
        joi_feat = self.bn2(loc + sur)
        num_batch, num_channel = joi_feat.size()[:2]
        y = self.global_pool(joi_feat).view(num_batch, num_channel)
        y = self.fc(y).view(num_batch, num_channel, 1, 1)
        y = joi_feat * y
        return residual + y if self.skip_connect else y


class Up(nn.Module):
    """Bilinear (or transposed-conv) 2x upsample."""

    def __init__(self, in_ch: int, bilinear: bool = True) -> None:
        super().__init__()
        self.up: nn.Module
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(x)


class DiffFPN(nn.Module):
    """LSNet difference feature-pyramid head over context-guided blocks."""

    def __init__(
        self,
        cur_channels: list[int],
        mid_ch: int,
        dilations: list[int],
        reductions: list[int],
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.lateral_convs: nn.ModuleList = nn.ModuleList(
            [
                ContextGuidedBlock(
                    cur_channels[i] * 2, mid_ch * 2**i, dilations[i], reductions[i]
                )
                for i in range(4)
            ]
        )
        self.top_down_convs: nn.ModuleList = nn.ModuleList(
            [
                ContextGuidedBlock(
                    mid_ch * 2**i, mid_ch * 2 ** (i - 1), dilations[i], reductions[i]
                )
                for i in range(3, 0, -1)
            ]
        )
        self.diff_convs: nn.ModuleList = nn.ModuleList(
            [
                ContextGuidedBlock(
                    mid_ch * (3 * 2**i), mid_ch * 2**i, dilations[i], reductions[i]
                )
                for i in range(3)
            ]
            + [
                ContextGuidedBlock(
                    mid_ch * (3 * 2**i), mid_ch * 2**i, dilations[i], reductions[i]
                )
                for i in range(2)
            ]
            + [ContextGuidedBlock(mid_ch * 3, mid_ch * 2, dilations[0], reductions[0])]
        )
        self.up2x: Up = Up(32, bilinear)

    def forward(self, output: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        tmp = [
            self.lateral_convs[i](torch.cat([output[i * 2], output[i * 2 + 1]], dim=1))
            for i in range(4)
        ]
        for i in range(3, 0, -1):
            tmp[i - 1] = tmp[i - 1] + self.up2x(self.top_down_convs[3 - i](tmp[i]))

        tmp = [
            self.diff_convs[i](torch.cat([tmp[i], self.up2x(tmp[i + 1])], dim=1))
            for i in (0, 1, 2)
        ]
        x0_1 = tmp[0]

        tmp = [
            self.diff_convs[i](torch.cat([tmp[i - 3], self.up2x(tmp[i - 2])], dim=1))
            for i in (3, 4)
        ]
        x0_2 = tmp[0]

        x0_3 = self.diff_convs[5](torch.cat([tmp[0], self.up2x(tmp[1])], dim=1))
        return x0_1, x0_2, x0_3


class SimpleBackbone(nn.Module):
    """Four-stage plain conv backbone feeding :class:`DiffFPN`."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2: nn.Conv2d = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.conv3: nn.Conv2d = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.conv4: nn.Conv2d = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

    def forward(self, x: Tensor) -> list[Tensor]:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return [x1, x2, x3, x4]
