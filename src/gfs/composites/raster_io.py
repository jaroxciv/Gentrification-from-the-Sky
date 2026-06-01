"""Small shared raster read/write helpers for the composites stage.

Centralizes the rasterio open/read and profile-copy/update/write patterns that
were duplicated across clipping, merging and statistics, and the one correct
notion of a valid-pixel mask (so nodata handling is consistent — a missing
``nodata`` falls back to NaN rather than silently matching nothing).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import rasterio


def read_raster(path: str, *, band: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a raster (all bands, or one ``band``) and return ``(array, profile)``."""
    with rasterio.open(path) as src:
        data = src.read() if band is None else src.read(band)
        profile = dict(src.profile)
    return data, profile


def write_raster(path: str, array: np.ndarray, profile: dict[str, Any], **overrides: Any) -> str:
    """Write ``array`` to ``path``, deriving height/width/count from its shape.

    Starts from ``profile`` (CRS, dtype, nodata, …), applies ``overrides`` (e.g. a
    new ``transform``), and sets the spatial dims from the array. Accepts a 2-D
    (single-band) or 3-D ``(bands, H, W)`` array.
    """
    out = dict(profile)
    out.update(driver="GTiff", **overrides)
    array3d = array if array.ndim == 3 else array[np.newaxis, :, :]
    out.update(count=array3d.shape[0], height=array3d.shape[1], width=array3d.shape[2])

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with rasterio.open(path, "w", **out) as dst:
        dst.write(array3d)
    return path


def valid_pixel_mask(data: np.ndarray, nodata: float | None) -> np.ndarray:
    """Boolean mask of valid pixels (not NaN and, if set, not the ``nodata`` value)."""
    mask = ~np.isnan(data)
    if nodata is not None:
        mask &= data != nodata
    return mask
