"""Shared utilities for the change-detection stage (paper §3.3-3.4).

Collects the helpers reused by every model: loading and standardizing the two
Sentinel-2 composites, slicing them into ``PATCH_SIZE`` imagelets and stitching
the predicted patches back together, the loss objects (MSE / Dice) used during
one-shot Siamese training, the Otsu/adaptive thresholding strategies that turn a
continuous change map into a binary one (skimage), the ``ClearCache`` GPU helper
and device selection.

These mirror the notebook's repeated ``extract_patches`` / ``reconstruct_image``
/ ``apply_thresholding_strategy`` cells verbatim, factored into one place.
"""

from __future__ import annotations

from types import TracebackType
from typing import cast

import numpy as np
import numpy.typing as npt
import rasterio
import torch
import torch.nn.functional as F
from skimage import filters, morphology
from skimage.measure import label, regionprops
from torch import Tensor, nn

from gfs.config import PATCH_SIZE

FloatArray = npt.NDArray[np.float64]


# --- Device / cache ----------------------------------------------------------
def select_device() -> torch.device:
    """Return CUDA if available, else CPU (the notebook's device guard)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClearCache:
    """Context manager that empties the CUDA cache on enter and exit.

    Used around the (memory-hungry) one-shot training loops in the notebook to
    keep the single-GPU Colab session from OOM-ing.
    """

    def __enter__(self) -> ClearCache:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --- Image loading / standardization -----------------------------------------
def load_image(image_path: str, bands: list[int] | None = None) -> FloatArray:
    """Read a composite, fill no-data with the min, and z-standardize.

    Faithful to the notebook ``load_image``: NaNs are replaced with the array
    minimum, then the whole stack is standardized to zero mean / unit std.
    ``bands`` is a 1-based rasterio band selection (e.g. ``[4]`` for B4); when
    ``None`` every band is read.
    """
    with rasterio.open(image_path) as src:
        image = src.read(bands) if bands is not None else src.read()
    image = np.nan_to_num(image, nan=float(np.nanmin(image)))
    image = (image - np.mean(image)) / np.std(image)
    return image.astype(np.float64)


def load_dataset(
    image1_path: str, image2_path: str, bands: list[int] | None
) -> tuple[FloatArray, FloatArray]:
    """Load both composites with the same band selection (t1, t2)."""
    return load_image(image1_path, bands), load_image(image2_path, bands)


def no_data_mask_from(image: FloatArray) -> npt.NDArray[np.bool_]:
    """Pixels that are NaN or equal the minimum in the first band = no-data."""
    band0 = image[0]
    return np.isnan(band0) | (band0 == np.nanmin(band0))


# --- Patching / reconstruction -----------------------------------------------
def extract_patches(
    image: Tensor, patch_size: int = PATCH_SIZE, stride: int | None = None
) -> list[Tensor]:
    """Slide a ``patch_size`` window over a ``(C, H, W)`` tensor.

    ``stride`` defaults to ``patch_size // 2`` (50% overlap), as the notebook
    used. Returns the patches in row-major order; the same order
    :func:`reconstruct_image` assumes.
    """
    if stride is None:
        stride = patch_size // 2
    patches: list[Tensor] = []
    _, height, width = image.shape[-3:]
    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            patches.append(image[:, i : i + patch_size, j : j + patch_size])
    return patches


def reconstruct_image(
    patches: list[FloatArray] | FloatArray,
    image_shape: tuple[int, ...],
    patch_size: int = PATCH_SIZE,
    stride: int | None = None,
) -> FloatArray:
    """Stitch overlapping patches back into a ``(C, H, W)`` image.

    Overlapping regions are averaged via a per-pixel count matrix (zeros guarded
    against division-by-zero). ``image_shape`` is the original ``(B, C, H, W)``
    tensor shape, matching the notebook's ``reconstruct_image``.
    """
    if stride is None:
        stride = patch_size // 2
    channels, height, width = image_shape[1:]
    reconstructed = np.zeros((channels, height, width), dtype=np.float64)
    count_matrix = np.zeros((height, width), dtype=np.float64)
    patch_idx = 0
    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            reconstructed[:, i : i + patch_size, j : j + patch_size] += patches[patch_idx]
            count_matrix[i : i + patch_size, j : j + patch_size] += 1
            patch_idx += 1
    count_matrix[count_matrix == 0] = 1  # avoid division by zero
    reconstructed /= count_matrix
    return reconstructed


def resize_to_match(target: Tensor, reference: Tensor) -> Tensor:
    """Bilinearly resize ``target`` to ``reference``'s spatial size."""
    return F.interpolate(target, size=reference.shape[2:], mode="bilinear", align_corners=False)


def resize_features(features: FloatArray, target_shape: tuple[int, int]) -> FloatArray:
    """Resize each channel of a ``(C, H, W)`` feature stack to ``target_shape``.

    Mirrors the notebook's per-channel resize (originally ``cv2.resize`` with
    bilinear interpolation); implemented here with torch so no OpenCV dep is
    needed and the result matches ``align_corners=False`` bilinear sampling.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(features)).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=target_shape, mode="bilinear", align_corners=False)
    return resized.squeeze(0).numpy().astype(np.float64)


def normalize01(change_map: FloatArray) -> FloatArray:
    """Min-max scale a change map to ``[0, 1]`` (the notebook's normalization)."""
    lo, hi = float(np.nanmin(change_map)), float(np.nanmax(change_map))
    return (change_map - lo) / (hi - lo)


# --- Losses ------------------------------------------------------------------
class DiceLoss(nn.Module):
    """Soft Dice loss on flattened predictions/targets (notebook ``DiceLoss``)."""

    def forward(self, inputs: Tensor, targets: Tensor, smooth: float = 1.0) -> Tensor:
        flat_inputs = inputs.view(-1)
        flat_targets = targets.view(-1)
        intersection = (flat_inputs * flat_targets).sum()
        dice = (2.0 * intersection + smooth) / (flat_inputs.sum() + flat_targets.sum() + smooth)
        return 1 - dice


def mse_loss() -> nn.MSELoss:
    """The paper's training objective for one-shot Siamese change detection."""
    return nn.MSELoss()


# --- Thresholding (skimage) --------------------------------------------------
def segment_image(binary_image: npt.NDArray[np.bool_], min_size: int = 10) -> npt.NDArray[np.int_]:
    """Label connected components and zero out regions smaller than ``min_size``."""
    labeled_image = cast("npt.NDArray[np.int_]", label(binary_image))
    for region in regionprops(labeled_image):
        if region.area < min_size:
            for coordinates in region.coords:
                labeled_image[coordinates[0], coordinates[1]] = 0
    return labeled_image


def apply_thresholding_strategy(
    detected_change_map_normalized: FloatArray,
    strategy: str = "otsu",
    min_size: int = 128,
    otsu_scaling_factor: float = 1.0,
    segment: bool = False,
    segment_min_size: int = 10,
) -> npt.NDArray[np.bool_] | npt.NDArray[np.int_]:
    """Turn a normalized change map into a binary change-detection map.

    Strategies (paper §3.3, verbatim from the notebook):

    - ``adaptive``  : union over Gaussian-blurred local thresholds (sigma sweep).
    - ``otsu``      : global Otsu threshold (skimage ``threshold_otsu``).
    - ``multiotsu`` : 3-class Multi-Otsu, threshold at the *upper* class boundary.
    - ``scaledOtsu``: Otsu threshold scaled by ``otsu_scaling_factor``.

    Small objects under ``min_size`` are removed; optionally a connected-component
    ``segment`` pass drops blobs under ``segment_min_size``.
    """
    cd_map: npt.NDArray[np.bool_] = np.zeros(detected_change_map_normalized.shape, dtype=bool)

    if strategy == "adaptive":
        for sigma in range(101, 202, 50):
            adaptive_threshold = 2 * filters.gaussian(detected_change_map_normalized, sigma)
            cd_map_temp = detected_change_map_normalized > adaptive_threshold
            cd_map_temp = morphology.remove_small_objects(cd_map_temp, min_size=min_size)
            cd_map = cd_map | cd_map_temp

    elif strategy == "otsu":
        otsu_threshold = filters.threshold_otsu(detected_change_map_normalized)
        cd_map = detected_change_map_normalized > otsu_threshold
        cd_map = morphology.remove_small_objects(cd_map, min_size=min_size)

    elif strategy == "multiotsu":
        otsu_threshold = filters.threshold_multiotsu(detected_change_map_normalized)
        cd_map = detected_change_map_normalized > otsu_threshold[1]
        cd_map = morphology.remove_small_objects(cd_map, min_size=min_size)

    elif strategy == "scaledOtsu":
        otsu_threshold = filters.threshold_otsu(detected_change_map_normalized)
        cd_map = detected_change_map_normalized > (otsu_scaling_factor * otsu_threshold)
        cd_map = morphology.remove_small_objects(cd_map, min_size=min_size)

    else:
        raise ValueError("Unknown thresholding strategy")

    if segment:
        return segment_image(cd_map, segment_min_size)
    return cd_map
