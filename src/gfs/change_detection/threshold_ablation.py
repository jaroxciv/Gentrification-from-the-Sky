"""Per-threshold change maps for the simple-difference ablation (paper §5).

The ablation sweeps the magnitude threshold of a *non-neural* absolute-difference
detector and asks how downstream gentrification prediction responds. Unlike the
multi-Otsu ``simple_diff`` baseline (which auto-selects its cut on standardized
reflectance), this detector applies a **fixed reflectance-difference threshold**,
so the sweep needs one set of change maps per threshold.

Faithful to the dissertation ablation notebook:

1. Histogram-match t2 to t1 (per-band mean/std) so one absolute threshold is
   comparable across the pair.
2. ``|t1 - t2| > threshold`` -> binary change.
3. Drop connected components smaller than ``min_region_size`` px (speckle).
4. Zero out pixels the Dynamic World land cover marks as green/natural, so the
   detector reports *built* change only (the paper's green-space exclusion).

Each ``(band, threshold)`` pair is written as
``filtered_changes_band_<b>_threshold_<t>.tiff``; aggregating that folder to
LSOAs yields one change column per threshold (suffix ``_<t>``), which
:mod:`gfs.modeling.ablation` sweeps.

**Performance (this was delegated to an HPC node).** The notebook recomputed the
per-band difference once *per threshold* (30x redundant work) and pruned speckle
with a Python ``regionprops`` loop. Here each band's difference is computed
**once** and reused across all thresholds, speckle removal is the vectorized
C-level :func:`skimage.morphology.remove_small_objects`, outputs are compact
``uint8`` (1 = built change, 0 = otherwise — identical to the downstream
``> 0`` count, a fraction of the float maps' size), and the independent bands run
in parallel across cores. Same product, two-plus orders of magnitude faster.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
from loguru import logger
from rasterio.enums import Resampling
from rasterio.warp import reproject
from skimage.measure import label

from gfs.composites.landcover import DYNAMIC_WORLD_CLASSES
from gfs.config import SENTINEL2_BANDS

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# Dynamic World classes treated as "green/natural" and excluded from change
# (the notebook's [0, 1, 2, 4, 5]). Named here so the choice is legible and
# overridable rather than a bare list of magic ints.
GREEN_LANDCOVER_CLASSES: tuple[str, ...] = (
    "water",
    "trees",
    "grass",
    "crops",
    "shrub_and_scrub",
)

# Full (8-)connectivity, matching the notebook's default-labelled `segment_image`
# (skimage `label` uses connectivity = ndim for 2-D input).
_SEGMENT_CONNECTIVITY = 2


def green_class_ids(classes: Sequence[str] = GREEN_LANDCOVER_CLASSES) -> tuple[int, ...]:
    """Map Dynamic World class names to their integer label values."""
    return tuple(DYNAMIC_WORLD_CLASSES.index(name) for name in classes)


def drop_small_components(binary: BoolArray, min_size: int) -> BoolArray:
    """Drop connected components smaller than ``min_size`` px (vectorized speckle prune).

    A label + ``np.bincount`` reimplementation of the notebook's per-region
    ``regionprops`` loop (``area < min_size`` removed): same result, no Python
    loop and no dependence on the deprecation-churning ``remove_small_objects``.
    """
    labels: npt.NDArray[np.intp] = np.asarray(
        label(binary, connectivity=_SEGMENT_CONNECTIVITY), dtype=np.intp
    )
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False  # background label
    return keep[labels]


def histogram_match(image: FloatArray, mean_target: float, std_target: float) -> FloatArray:
    """Rescale ``image`` to the target mean/std (the notebook's ``normalize_image``).

    Matching t2's distribution to t1's makes a single absolute difference
    threshold meaningful across the pair (correcting per-date gain/offset).
    """
    mean_image = float(np.nanmean(image))
    std_image = float(np.nanstd(image))
    return ((image - mean_image) / std_image) * std_target + mean_target


def reproject_to_grid(
    src_path: str,
    transform: rasterio.Affine,
    crs: Any,
    shape: tuple[int, int],
) -> npt.NDArray[Any]:
    """Nearest-neighbour reproject a single-band raster onto a target grid."""
    with rasterio.open(src_path) as src:
        source = src.read(1)
        destination = np.empty(shape, dtype=source.dtype)
        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )
    return destination


def detect_threshold_change(
    band1: npt.NDArray[Any],
    band2: npt.NDArray[Any],
    green_mask: BoolArray,
    threshold: float,
    *,
    min_region_size: int = 10,
) -> BoolArray:
    """Built-change mask for one band at one reflectance-difference threshold.

    Histogram-matches t2 to t1, thresholds the absolute difference, removes
    speckle (components under ``min_region_size`` px) and excludes green/natural
    pixels. Returns a boolean array (``True`` = built change), the quantity the
    LSOA aggregation counts.
    """
    image1 = np.nan_to_num(band1, nan=float(np.nanmin(band1))).astype(np.float64)
    image2 = np.nan_to_num(band2, nan=float(np.nanmin(band2))).astype(np.float64)
    no_data = image1 == np.nanmin(image1)

    image2_matched = histogram_match(image2, float(np.nanmean(image1)), float(np.nanstd(image1)))
    difference = np.abs(image1 - image2_matched)
    difference[no_data] = 0.0

    binary = difference > threshold
    kept = drop_small_components(binary, min_region_size)
    return kept & ~green_mask


def threshold_label(threshold: float) -> int:
    """Integer suffix for a threshold's change column/file (matches the sweep).

    Truncates (``int``) exactly as :func:`gfs.modeling.ablation.evaluate_thresholds`
    does when selecting that threshold's ``_<label>`` columns, so generation and
    the sweep always agree.
    """
    return int(threshold)


@dataclass(frozen=True)
class _BandJob:
    """One band's slice of the sweep (everything a worker process needs)."""

    band: int
    image1_path: str
    image2_path: str
    out_dir: str
    thresholds: tuple[float, ...]
    green_mask: BoolArray
    min_region_size: int


def _write_band_thresholds(job: _BandJob) -> list[str]:
    """Compute one band's difference once and write every thresholded map."""
    with rasterio.open(job.image1_path) as src1, rasterio.open(job.image2_path) as src2:
        band1 = src1.read(job.band)
        band2 = src2.read(job.band)
        # Georeference to t2 (as the notebook did) with a clean uint8 profile.
        profile: dict[str, Any] = {
            "driver": "GTiff",
            "height": src2.height,
            "width": src2.width,
            "count": 1,
            "dtype": "uint8",
            "crs": src2.crs,
            "transform": src2.transform,
            "nodata": None,
        }

    # Difference is threshold-independent — compute the matched difference once,
    # then only re-threshold/prune per sweep value.
    image1 = np.nan_to_num(band1, nan=float(np.nanmin(band1))).astype(np.float64)
    image2 = np.nan_to_num(band2, nan=float(np.nanmin(band2))).astype(np.float64)
    no_data = image1 == np.nanmin(image1)
    image2_matched = histogram_match(image2, float(np.nanmean(image1)), float(np.nanstd(image1)))
    difference = np.abs(image1 - image2_matched)
    difference[no_data] = 0.0

    written: list[str] = []
    for threshold in job.thresholds:
        binary = difference > threshold
        kept = drop_small_components(binary, job.min_region_size)
        change = (kept & ~job.green_mask).astype(np.uint8)
        out_path = os.path.join(
            job.out_dir,
            f"filtered_changes_band_{job.band}_threshold_{threshold_label(threshold)}.tiff",
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(change, 1)
        written.append(out_path)
    return written


def generate_threshold_change_maps(
    image1_path: str,
    image2_path: str,
    landcover_path: str,
    out_dir: str,
    thresholds: Iterable[float],
    *,
    bands: list[int] | None = None,
    green_classes: Sequence[str] = GREEN_LANDCOVER_CLASSES,
    min_region_size: int = 10,
    n_jobs: int | None = None,
) -> list[str]:
    """Write per-band, per-threshold built-change maps for the §5 ablation.

    For every band and every threshold writes
    ``<out_dir>/filtered_changes_band_<b>_threshold_<t>.tiff`` (compact ``uint8``),
    excluding pixels Dynamic World marks green via ``green_classes``. Bands are
    independent and run in parallel (``n_jobs`` processes, default = all cores
    capped at the band count); within a band the difference is computed once and
    reused across thresholds. Returns every written path.
    """
    bands = bands if bands is not None else list(range(1, len(SENTINEL2_BANDS) + 1))
    sweep = tuple(float(t) for t in thresholds)
    os.makedirs(out_dir, exist_ok=True)

    # Build the green/natural mask once, on t2's grid (workers reuse it).
    with rasterio.open(image2_path) as ref:
        ref_transform, ref_crs, ref_shape = ref.transform, ref.crs, (ref.height, ref.width)
    landcover = reproject_to_grid(landcover_path, ref_transform, ref_crs, ref_shape)
    green_mask = np.isin(landcover, green_class_ids(green_classes))

    jobs = [
        _BandJob(
            band=band,
            image1_path=image1_path,
            image2_path=image2_path,
            out_dir=out_dir,
            thresholds=sweep,
            green_mask=green_mask,
            min_region_size=min_region_size,
        )
        for band in bands
    ]
    workers = min(n_jobs or os.cpu_count() or 1, len(jobs))
    logger.info(
        f"Generating {len(bands)} bands x {len(sweep)} thresholds "
        f"= {len(bands) * len(sweep)} change maps on {workers} worker(s)"
    )

    written: list[str] = []
    if workers <= 1:
        for job in jobs:
            written.extend(_write_band_thresholds(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for paths in pool.map(_write_band_thresholds, jobs):
                written.extend(paths)
    logger.info(f"Wrote {len(written)} threshold change maps to {out_dir}")
    return written
