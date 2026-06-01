"""Clip Sentinel-2 composites to London and merge tiles into per-year rasters (paper §3.2).

After the Earth Engine processor (or the original WASDI ``s2_average_bands``
jobs) yields one multi-band GeoTIFF per Sentinel-2 MGRS tile, two steps turn
them into a single aligned raster per study year:

1. **Clip** each tile to the Greater-London LSOA boundary (``rasterio.mask``),
   reprojecting the boundary to each tile's CRS first.
2. **Merge** the clipped tiles into one mosaic. The notebook tried plain
   ``rasterio.merge`` (last-tile-wins in overlaps) and a rioxarray
   ``sum / count`` average (overlaps averaged); both are preserved here, with
   the averaging merge being the one used for the published composites
   (``clipped_merged_<year>.tiff``).

The result is the ``(11, H, W)`` per-year composite consumed by the change-
detection models and zonal statistics.
"""

from __future__ import annotations

import glob
import os
from typing import Any, cast

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray as rxr
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from rioxarray.merge import merge_arrays

from gfs.config import COMPOSITES_DIR


def _boundary_geometries(boundary_gdf: gpd.GeoDataFrame, crs: Any) -> list[dict[str, Any]]:
    """Reproject the boundary to ``crs`` and return its GeoJSON geometries.

    Matches the notebook's ``__geo_interface__`` extraction so masking uses the
    full set of LSOA polygons rather than a dissolved hull.
    """
    reprojected = cast(gpd.GeoDataFrame, boundary_gdf.to_crs(crs))
    features = reprojected.__geo_interface__["features"]
    return [feature["geometry"] for feature in features]


def clip_raster_to_boundary(
    raster_path: str,
    boundary_gdf: gpd.GeoDataFrame,
    out_path: str,
    *,
    skip_existing: bool = True,
) -> str:
    """Clip one composite tile to the London boundary, cropping to its extent.

    Reprojects ``boundary_gdf`` to the tile CRS, masks with crop=True, copies
    the source metadata (preserving any ``nodata``), and writes ``out_path``.
    Returns ``out_path``.
    """
    if skip_existing and os.path.exists(out_path):
        return out_path

    with rasterio.open(raster_path) as src:
        geometry = _boundary_geometries(boundary_gdf, src.crs)
        out_image, out_transform = rio_mask(src, geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
        if "nodata" in src.meta:
            out_meta.update(nodata=src.meta["nodata"])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)
    return out_path


def clip_tiles(
    tile_paths: list[str],
    boundary_gdf: gpd.GeoDataFrame,
    output_dir: str,
    *,
    skip_existing: bool = True,
) -> list[str]:
    """Clip every tile in ``tile_paths`` to the boundary, writing into ``output_dir``.

    Output files are named ``clipped_<original basename>`` to mirror the
    notebook layout. Returns the list of clipped file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    clipped: list[str] = []
    for tile_path in tile_paths:
        out_path = os.path.join(output_dir, f"clipped_{os.path.basename(tile_path)}")
        clip_raster_to_boundary(tile_path, boundary_gdf, out_path, skip_existing=skip_existing)
        clipped.append(out_path)
    return clipped


def valid_data_percentage(raster_path: str, band: int = 1) -> float:
    """Percentage of valid (non-nodata, non-NaN) pixels in ``band`` of a raster.

    Sanity check used in the notebook to confirm a clipped tile still carries
    usable data before merging.
    """
    with rasterio.open(raster_path) as src:
        meta = src.meta.copy()
        data = src.read(band)
    n_total = meta["width"] * meta["height"]
    if meta.get("nodata") is not None:
        n_valid = int((data != meta["nodata"]).sum())
    else:
        n_valid = int((~np.isnan(data)).sum())
    return 100.0 * n_valid / n_total


def merge_tiles_mosaic(
    clipped_files: list[str],
    out_path: str,
    *,
    skip_existing: bool = True,
) -> str | None:
    """Merge clipped tiles with ``rasterio.merge`` (last-tile-wins in overlaps).

    Writes a single mosaic GeoTIFF to ``out_path``. Returns ``out_path``, or
    ``None`` when there are no input tiles.
    """
    if not clipped_files:
        return None
    if skip_existing and os.path.exists(out_path):
        return out_path

    src_files = [rasterio.open(fp) for fp in clipped_files]
    try:
        mosaic, out_trans = rio_merge(src_files)
        out_meta = src_files[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
            }
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(mosaic)
    finally:
        for src in src_files:
            src.close()
    return out_path


def merge_tiles_average(
    clipped_files: list[str],
    out_path: str,
    *,
    skip_existing: bool = True,
) -> str | None:
    """Merge clipped tiles by *averaging* overlaps (the published method).

    Uses rioxarray ``merge_arrays`` to form a per-pixel ``sum / count``, so
    overlapping tile edges are averaged rather than overwritten. This produces
    the study's ``clipped_merged_<year>.tiff`` composites. Returns ``out_path``,
    or ``None`` when there are no input tiles.
    """
    if not clipped_files:
        return None
    if skip_existing and os.path.exists(out_path):
        return out_path

    arrays = [rxr.open_rasterio(fp, mask_and_scale=True) for fp in clipped_files]
    raster_sum = merge_arrays(dataarrays=cast(Any, arrays), method="sum")
    raster_count = merge_arrays(dataarrays=cast(Any, arrays), method="count")
    raster_avg = raster_sum / raster_count

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cast(Any, raster_avg).rio.to_raster(out_path)
    return out_path


def clip_and_merge_year(
    tile_paths: list[str],
    boundary_gdf: gpd.GeoDataFrame,
    year: int,
    *,
    clipped_dir: str | None = None,
    out_path: str | None = None,
    average_overlaps: bool = True,
    skip_existing: bool = True,
) -> str | None:
    """Clip all tiles for ``year`` to London and merge them into one composite.

    Defaults write the per-year clipped tiles to ``data/clipped_<year>/`` and
    the merged composite to ``data/composites/clipped_merged_<year>.tiff``,
    matching the project layout. Set ``average_overlaps=False`` for the plain
    last-tile-wins mosaic. Returns the merged composite path.
    """
    clipped_dir = clipped_dir or str(COMPOSITES_DIR.parent / f"clipped_{year}")
    out_path = out_path or str(COMPOSITES_DIR / f"clipped_merged_{year}.tiff")

    clipped = clip_tiles(tile_paths, boundary_gdf, clipped_dir, skip_existing=skip_existing)
    merge = merge_tiles_average if average_overlaps else merge_tiles_mosaic
    return merge(clipped, out_path, skip_existing=skip_existing)


def find_year_tiles(tile_dir: str, year: int, suffix: str = "08-31.tif") -> list[str]:
    """List tile GeoTIFFs in ``tile_dir`` belonging to ``year``.

    Mirrors the notebook's filename filter (``*<year>-08-31.tif``) used to group
    raw tiles by study year.
    """
    pattern = os.path.join(tile_dir, f"*{year}-{suffix}")
    return sorted(glob.glob(pattern))
