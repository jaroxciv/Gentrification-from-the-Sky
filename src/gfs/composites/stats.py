"""Difference, missing-value masking and zonal statistics for composites (paper §3.2).

Once both per-year composites exist, this module provides the descriptive and
preparatory computations that sit between compositing and change detection:

* **Band difference** (``t2 - t1``) with NaN-aware masking, the simplest change
  signal and the baseline the deep models are compared against.
* **Missing-value masks** that flag pixels with no data in *either* year, so a
  pixel only contributes where both composites are valid.
* **Per-band statistics** (mean/median/min/max/std, quartiles, null %) used to
  audit composite quality.
* **Zonal statistics** that aggregate a raster band to the LSOA polygons,
  yielding the per-neighbourhood values the modelling stage consumes.

The original notebook computed the zonal stats with the ``rasterstats`` package
(``count min mean max median nodata``). That dependency is not part of this
package, so the equivalent reduction is implemented directly with
``rasterio.features.rasterize`` + NumPy, preserving the same statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize


@dataclass
class CompositePair:
    """A loaded pair of per-year composites plus the shared geo-metadata."""

    earlier: np.ndarray  # (bands, H, W) for YEAR_T1
    later: np.ndarray  # (bands, H, W) for YEAR_T2
    transform: Any
    crs: Any
    nodata: float | None


def load_composite(path: str) -> tuple[np.ndarray, Any, Any, float | None]:
    """Read a multi-band composite into ``(array, transform, crs, nodata)``."""
    with rasterio.open(path) as src:
        data = src.read()
        return data, src.transform, src.crs, src.nodata


def load_pair(earlier_path: str, later_path: str) -> CompositePair:
    """Load both per-year composites and verify their shapes match.

    Geo-metadata is taken from the earlier composite (the notebook assumes the
    two years are already co-registered on the same grid).
    """
    earlier, transform, crs, nodata = load_composite(earlier_path)
    later, _, _, _ = load_composite(later_path)
    if earlier.shape != later.shape:
        raise ValueError(
            f"Composite shapes differ: {earlier.shape} vs {later.shape}; "
            "merge/clip both years onto the same grid first."
        )
    return CompositePair(earlier, later, transform, crs, nodata)


def masked_difference(pair: CompositePair) -> np.ma.MaskedArray:
    """NaN-aware ``later - earlier`` band difference.

    Invalid pixels in either year are masked out so the difference is defined
    only where both composites carry data (the notebook's
    ``calculate_difference`` on ``masked_invalid`` arrays).
    """
    masked_earlier = np.ma.masked_invalid(pair.earlier)
    masked_later = np.ma.masked_invalid(pair.later)
    combined_mask = np.ma.getmaskarray(masked_earlier) | np.ma.getmaskarray(masked_later)
    return np.ma.masked_where(combined_mask, masked_later - masked_earlier)


def missing_data_mask(path: str, band: int) -> np.ndarray:
    """Binary mask (1 = missing) for ``band`` of a composite at ``path``.

    A pixel is missing where it equals the raster's ``nodata`` value, matching
    the notebook's ``create_missing_data_mask``.
    """
    with rasterio.open(path) as src:
        data = src.read(band)
        return (data == src.nodata).astype(int)


def combined_missing_mask(
    earlier_path: str, later_path: str, band: int
) -> np.ndarray:
    """Union of the two years' missing-data masks for one band.

    A pixel is flagged whenever it is missing in *either* year (the
    ``np.maximum`` of the two per-year masks), so change is only evaluated where
    both composites are valid.
    """
    mask_earlier = missing_data_mask(earlier_path, band)
    mask_later = missing_data_mask(later_path, band)
    return np.maximum(mask_earlier, mask_later)


def apply_combined_mask(image: np.ndarray, combined_mask: np.ndarray) -> np.ndarray:
    """Zero out pixels flagged by ``combined_mask`` (the notebook's masking step)."""
    return np.where(combined_mask == 0, image, 0)


def band_statistics(composite: np.ndarray) -> pd.DataFrame:
    """Per-band descriptive statistics for a ``(bands, H, W)`` composite.

    Reproduces the notebook's NaN-aware audit table: mean, median, min, max,
    standard deviation, the 25th/75th percentiles and the null percentage.
    """
    rows: list[dict[str, float]] = []
    for band_index in range(composite.shape[0]):
        band = composite[band_index, :, :]
        rows.append(
            {
                "Band": float(band_index + 1),
                "Mean": float(np.nanmean(band)),
                "Median": float(np.nanmedian(band)),
                "Min": float(np.nanmin(band)),
                "Max": float(np.nanmax(band)),
                "Std Dev": float(np.nanstd(band)),
                "25th Percentile": float(np.nanpercentile(band, 25)),
                "75th Percentile": float(np.nanpercentile(band, 75)),
                "Null Pct": float(np.sum(np.isnan(band)) / band.size * 100),
            }
        )
    return pd.DataFrame(rows)


# --- Zonal statistics (LSOA aggregation) ------------------------------------
ZONAL_STATS = ("count", "min", "mean", "max", "median")


def zonal_statistics(
    zones: gpd.GeoDataFrame,
    raster_path: str,
    *,
    band: int = 1,
    stats: tuple[str, ...] = ZONAL_STATS,
) -> pd.DataFrame:
    """Aggregate one raster band to polygon zones (the notebook's ``zonal_stats``).

    Reprojects ``zones`` to the raster CRS, rasterizes each polygon to the
    raster grid, and reduces the in-zone pixels with NaN/nodata-aware
    statistics. Returns one row per zone (in the order of ``zones``) with the
    requested ``stats`` columns. Implemented with ``rasterio.features`` so it
    does not require the external ``rasterstats`` package.
    """
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        transform = src.transform
        out_shape = (src.height, src.width)
        nodata = src.nodata
        data = src.read(band).astype("float64")

    zones_proj = cast(gpd.GeoDataFrame, zones.to_crs(raster_crs))

    # Mask nodata/NaN pixels so they are excluded from every statistic.
    valid = ~np.isnan(data)
    if nodata is not None:
        valid &= data != nodata

    records: list[dict[str, float]] = []
    geometries = list(zones_proj.geometry)
    for geom in geometries:
        zone_mask = (
            rasterize(
                [(geom, 1)],
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype="uint8",
            )
            == 1
        )
        pixels = data[zone_mask & valid]
        record: dict[str, float] = {}
        count = int(pixels.size)
        if "count" in stats:
            record["count"] = float(count)
        if count:
            if "min" in stats:
                record["min"] = float(np.min(pixels))
            if "mean" in stats:
                record["mean"] = float(np.mean(pixels))
            if "max" in stats:
                record["max"] = float(np.max(pixels))
            if "median" in stats:
                record["median"] = float(np.median(pixels))
        else:
            for stat in ("min", "mean", "max", "median"):
                if stat in stats:
                    record[stat] = float("nan")
        records.append(record)

    return pd.DataFrame(records, index=zones.index)
