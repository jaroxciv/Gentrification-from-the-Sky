"""Run a change-detection model over the two composites and write features.

This is the bridge from a trained model to the per-band binary change maps that
become the satellite feature matrix Phi (paper §3.5). For each Sentinel-2 band we
build a change map and Multi-Otsu threshold it, then save the binary raster to
``outputs/features_<model>/features_band_<n>_<model>.tiff`` — the on-disk layout
the modeling stage's zonal statistics read.

Two feature-extraction shapes are provided, mirroring the notebook:

- :func:`siamese_band_change_map` — patch the single-band composites, run a
  ``forward(x1, x2)`` model patch-by-patch, reconstruct and normalize (TinyCD,
  CGNet, BiDateNet, FC-SiamDiff).
- :func:`resnet_band_change_map` — run the trained Res-Net ``FeatureExtractor``
  on each composite, mask, standardize, and take the norm of the feature
  difference (the customized Res-Net pipeline).
- :func:`simple_diff_band_change_map` — the non-neural Simple-Diff baseline.

All paths and band names come from :mod:`gfs.config`.
"""

from __future__ import annotations

import os

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch import nn

from gfs.change_detection.common import (
    ClearCache,
    FloatArray,
    apply_thresholding_strategy,
    extract_patches,
    load_dataset,
    no_data_mask_from,
    normalize01,
    reconstruct_image,
    resize_features,
    select_device,
)
from gfs.change_detection.models.simple_diff import simple_diff_change_map
from gfs.config import FEATURES_DIR, PATCH_SIZE, SENTINEL2_BANDS


def _band_to_tensor(image: FloatArray, band_index: int, device: torch.device) -> torch.Tensor:
    """Pull a single (0-based) band out of a composite into a ``(1, 1, H, W)`` tensor."""
    band = torch.tensor(image[band_index, :, :], dtype=torch.float32)
    return band.unsqueeze(0).unsqueeze(0).to(device)


def siamese_band_change_map(
    model: nn.Module,
    im1: FloatArray,
    im2: FloatArray,
    band: int,
    *,
    patch_size: int = PATCH_SIZE,
    stride: int | None = None,
    device: torch.device | None = None,
) -> FloatArray:
    """Continuous change map for one band from a ``forward(x1, x2)`` model.

    Patches the single-band composites, runs the model on each patch, resizes the
    output back to ``patch_size``, reconstructs the full image and min-max
    normalizes it. ``band`` is 1-based (as in the notebook / rasterio).
    """
    if device is None:
        device = select_device()
    if stride is None:
        stride = patch_size // 2

    im1_band = _band_to_tensor(im1, band - 1, device)
    im2_band = _band_to_tensor(im2, band - 1, device)

    im1_patches = extract_patches(im1_band.squeeze(0), patch_size, stride)
    im2_patches = extract_patches(im2_band.squeeze(0), patch_size, stride)
    im1_stacked = torch.stack([p.clone().detach().unsqueeze(0) for p in im1_patches]).to(device)
    im2_stacked = torch.stack([p.clone().detach().unsqueeze(0) for p in im2_patches]).to(device)

    change_patches: list[FloatArray] = []
    model.eval()
    with torch.no_grad():
        for i in range(im1_stacked.size(0)):
            output = model(im1_stacked[i], im2_stacked[i])
            resized_output = F.interpolate(
                output, size=(patch_size, patch_size), mode="bilinear", align_corners=False
            )
            change_patches.append(resized_output.cpu().numpy().squeeze())

    change_map = reconstruct_image(
        np.array(change_patches), im1_band.shape, patch_size, stride
    )
    return normalize01(change_map)


def resnet_band_change_map(
    model: nn.Module,
    im1: FloatArray,
    im2: FloatArray,
    *,
    device: torch.device | None = None,
) -> FloatArray:
    """Continuous change map from the Res-Net feature extractor (verbatim logic).

    Runs the trained ``FeatureExtractor`` on each single-band composite, resizes
    features to the no-data mask, masks/standardizes them, aggregates over
    channels, and returns the min-max normalized norm of the absolute difference.
    """
    if device is None:
        device = select_device()
    no_data_mask = no_data_mask_from(im1)

    features1 = model(_to_band0(im1, device)).detach().cpu().numpy()[0]
    features2 = model(_to_band0(im2, device)).detach().cpu().numpy()[0]

    features1 = resize_features(features1, no_data_mask.shape)
    features2 = resize_features(features2, no_data_mask.shape)

    features1[:, no_data_mask] = 0
    features2[:, no_data_mask] = 0

    normalized_features1 = (features1 - np.mean(features1)) / np.std(features1)
    normalized_features2 = (features2 - np.mean(features2)) / np.std(features2)

    aggregated_features1 = np.mean(normalized_features1, axis=0, keepdims=True)
    aggregated_features2 = np.mean(normalized_features2, axis=0, keepdims=True)

    absolute_difference = np.abs(aggregated_features1 - aggregated_features2)
    detected_change_map = np.linalg.norm(absolute_difference, axis=0)
    return normalize01(detected_change_map)


def _to_band0(image: FloatArray, device: torch.device) -> torch.Tensor:
    """Wrap a single-band ``(1, H, W)`` composite as a ``(1, 1, H, W)`` tensor."""
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)


def simple_diff_band_change_map(im1: FloatArray, im2: FloatArray, band: int) -> FloatArray:
    """Continuous Simple-Diff change map for one (1-based) band."""
    return simple_diff_change_map(im1[band - 1], im2[band - 1])


def threshold_and_save_band(
    change_map: FloatArray,
    band: int,
    model_name: str,
    reference_image_path: str,
    output_dir: str,
    *,
    strategy: str = "multiotsu",
    min_size: int = 4,
) -> str:
    """Multi-Otsu threshold a change map and write it as a binary GeoTIFF.

    Reuses the CRS/transform from ``reference_image_path`` (the t2 composite) so
    the change map is georeferenced. Output path follows the notebook layout:
    ``<output_dir>/features_band_<band>_<model>.tiff``.
    """
    binary = apply_thresholding_strategy(change_map, strategy=strategy, min_size=min_size)
    binary_u8 = np.asarray(binary).astype(np.uint8).squeeze()

    with rasterio.open(reference_image_path) as src:
        transform = src.transform
        crs = src.crs

    raster_meta = {
        "driver": "GTiff",
        "height": binary_u8.shape[0],
        "width": binary_u8.shape[1],
        "count": 1,
        "dtype": binary_u8.dtype,
        "crs": crs,
        "transform": transform,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"features_band_{band}_{model_name}.tiff")
    with rasterio.open(output_path, "w", **raster_meta) as dst:
        dst.write(binary_u8, 1)
    print(f"Change map features for band {band} saved to {output_path}")
    return output_path


def features_output_dir(model_name: str) -> str:
    """``outputs/features_<model>/`` for a model (paper FEATURES_DIR layout)."""
    return str(FEATURES_DIR / f"features_{model_name}")


def all_bands() -> list[int]:
    """1-based band indices for all Sentinel-2 bands used (paper §3.2)."""
    return list(range(1, len(SENTINEL2_BANDS) + 1))


def extract_and_save_features(
    model: nn.Module,
    model_name: str,
    image1_path: str,
    image2_path: str,
    *,
    kind: str = "siamese",
    bands: list[int] | None = None,
    device: torch.device | None = None,
) -> list[str]:
    """Run ``model`` over every band and write thresholded change maps.

    ``kind`` selects the feature-extraction shape: ``"siamese"`` (forward(x1, x2))
    or ``"resnet"`` (per-band autoencoder). Returns the written file paths.
    """
    if device is None:
        device = select_device()
    if bands is None:
        bands = all_bands()

    im1, im2 = load_dataset(image1_path, image2_path, bands)
    output_dir = features_output_dir(model_name)

    written: list[str] = []
    for band in bands:
        if kind == "resnet":
            single1 = im1[band - 1 : band]
            single2 = im2[band - 1 : band]
            with ClearCache():
                change_map = resnet_band_change_map(model, single1, single2, device=device)
        else:
            with ClearCache():
                change_map = siamese_band_change_map(
                    model, im1, im2, band, device=device
                )
        written.append(
            threshold_and_save_band(
                change_map, band, model_name, image2_path, output_dir
            )
        )
    return written
