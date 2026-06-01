"""LSOA-level aggregation of change-detection outputs (paper §3.5).

Turns the per-band binary change maps produced by the change-detection stage
(``outputs/features_<model>/features_band_<b>_<model>.tiff``) into the feature
matrix Phi: one row per LSOA, one column per band, holding the **percentage of
changed pixels** that fall inside the LSOA polygon.

Method (a vectorized rasterize/bincount in place of the notebook's per-pixel
polygonization + spatial join — same quantity, ~2 orders of magnitude faster):

1. **Rasterize LSOA ids once**: burn the LSOA polygons onto the change-map grid
   so every pixel carries the id of the LSOA containing its centre.
2. **Count per id**: for each band, count changed pixels (``> 0``) per LSOA id
   with ``np.bincount`` — each pixel counted once (no boundary double-counting).
3. **Normalize**: divide the count by the LSOA area and scale to a percentage,
   giving the change density per band.

The planning-layer features (per-LSOA % area covered by each London planning
layer) are also assembled here, since they share the raster->LSOA logic and are
the non-satellite half of the predictor set (paper §3.5/§5).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features

from gfs.config import GEOGRAPHY_CODE_COL

# Boundary attribute holding the neighborhood code (e.g. "LSOA11CD").
LSOA_CODE_COL = GEOGRAPHY_CODE_COL

# Loosely-typed CRS-like value handed back by rasterio/geopandas.
CRSLike = Any


@dataclass(frozen=True)
class AggregationConfig:
    """Knobs for the raster->LSOA aggregation.

    ``min_change_value`` keeps any pixel strictly greater than it, which is how
    the binary/segmented change maps encode "changed".
    """

    min_change_value: float = 0.0


def rasterize_lsoa_ids(
    lsoa_gdf: gpd.GeoDataFrame,
    transform: rasterio.Affine,
    shape: tuple[int, int],
    crs: CRSLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Burn LSOA polygons onto the raster grid as integer ids (0 = background).

    Returns ``(id_grid, codes)`` where ``id_grid[i, j]`` is ``k + 1`` for the LSOA
    at row ``k`` of ``codes`` whose polygon contains that pixel's centre, or 0.
    Computed once and reused across bands; this is what makes aggregation fast.
    """
    lsoa = lsoa_gdf.to_crs(crs) if lsoa_gdf.crs is not None and lsoa_gdf.crs != crs else lsoa_gdf
    lsoa = lsoa.reset_index(drop=True)
    shapes = ((geom, idx + 1) for idx, geom in enumerate(lsoa.geometry))
    id_grid = rasterio.features.rasterize(
        shapes, out_shape=shape, transform=transform, fill=0, dtype="int32"
    )
    return id_grid, lsoa[LSOA_CODE_COL].to_numpy()


def _count_pixels_per_id(
    changes: np.ndarray, id_grid: np.ndarray, n_ids: int, min_value: float
) -> np.ndarray:
    """Count changed pixels (``> min_value``) falling in each LSOA id (vectorized)."""
    ids = id_grid[changes > min_value]
    return np.bincount(ids, minlength=n_ids + 1)[1:]


def count_change_pixels(
    change_file: str, lsoa_gdf: gpd.GeoDataFrame, *, min_change_value: float = 0.0
) -> pd.Series:
    """Count changed pixels per LSOA for a single change-map tiff.

    Vectorized: each changed pixel is assigned to the LSOA containing its centre
    (via a rasterized id grid) and counted with ``np.bincount`` — no per-pixel
    geometry. Returns a Series indexed by LSOA code, named after the tiff stem.
    """
    with rasterio.open(change_file) as src:
        changes = src.read(1)
        id_grid, codes = rasterize_lsoa_ids(lsoa_gdf, src.transform, changes.shape, src.crs)
    counts = _count_pixels_per_id(changes, id_grid, len(codes), min_change_value)
    return pd.Series(counts, index=codes, name=os.path.basename(change_file).split(".")[0])


def _band_number(path: str) -> int:
    """Extract the band number from a ``features_band_<b>_<model>.tiff`` name."""
    match = re.search(r"band_(\d+)", os.path.basename(path))
    if match is None:
        raise ValueError(f"Cannot parse a band number from change file: {path}")
    return int(match.group(1))


def list_change_files(changes_dir: str) -> list[str]:
    """List the per-band change tiffs in a ``features_<model>`` folder, band-sorted.

    Filenames look like ``features_band_<b>_<model>.tiff``. The band number is
    parsed from the ``band_<b>`` token by regex, so model names containing
    underscores (e.g. ``fc_siamdiff``) sort correctly.
    """
    change_files = [
        os.path.join(changes_dir, x) for x in os.listdir(changes_dir) if x.endswith(".tiff")
    ]
    return sorted(change_files, key=_band_number)


def aggregate_changes_to_lsoa(
    changes_dir: str,
    lsoa_gdf: gpd.GeoDataFrame,
    config: AggregationConfig | None = None,
) -> pd.DataFrame:
    """Build the satellite feature matrix Phi: % changed pixels per LSOA per band.

    Rasterizes the LSOA polygons onto the change-map grid **once**, then for every
    band tiff counts changed pixels per LSOA with ``np.bincount`` and converts the
    count to a percentage of the LSOA's area (paper §3.5). Each changed pixel is
    assigned to the LSOA containing its centre (no boundary double-counting).
    Returns a DataFrame indexed by LSOA code with one column per band tiff.

    Assumes all band tiffs in ``changes_dir`` share one grid (they are produced
    from the same composite), so the id grid is computed a single time.
    """
    cfg = config or AggregationConfig()
    files = list_change_files(changes_dir)
    if not files:
        return pd.DataFrame(index=pd.Index(lsoa_gdf[LSOA_CODE_COL], name=LSOA_CODE_COL))

    with rasterio.open(files[0]) as src:
        id_grid, codes = rasterize_lsoa_ids(lsoa_gdf, src.transform, src.shape, src.crs)
        raster_crs = src.crs

    lsoa_changes = pd.DataFrame(index=pd.Index(codes, name=LSOA_CODE_COL))
    for change_file in files:
        with rasterio.open(change_file) as src:
            changes = src.read(1)
        counts = _count_pixels_per_id(changes, id_grid, len(codes), cfg.min_change_value)
        lsoa_changes[os.path.basename(change_file).split(".")[0]] = counts

    # Convert raw counts to a percentage of LSOA area. The pixel counts live on
    # the raster's (working) grid, so area is measured in the *same* CRS — not the
    # boundary CRS — to keep the count/area density internally consistent.
    lsoa_in_raster_crs = (
        lsoa_gdf.to_crs(raster_crs)
        if lsoa_gdf.crs is not None and lsoa_gdf.crs != raster_crs
        else lsoa_gdf
    )
    area_by_code = cast("gpd.GeoSeries", lsoa_in_raster_crs.set_index(LSOA_CODE_COL).geometry).area
    areas = area_by_code.reindex(codes).to_numpy()
    for column in lsoa_changes.columns:
        lsoa_changes[column] = (lsoa_changes[column] / areas) * 100

    return lsoa_changes


def create_percentage_features(
    lsoa_gdf: gpd.GeoDataFrame,
    planning_layers: dict[str, str],
) -> gpd.GeoDataFrame:
    """Per-LSOA % area covered by each London planning layer (paper §3.5/§5).

    For each planning ``.gpkg`` layer, computes the fraction of every LSOA
    polygon it intersects, scaled to a percentage. Adds one column per layer to a
    copy of ``lsoa_gdf``. Geometries are buffered by 0 to repair invalid rings.
    """
    out = lsoa_gdf.copy()
    out["geometry"] = out["geometry"].buffer(0)
    for layer_name, layer_path in planning_layers.items():
        layer_gdf = gpd.read_file(layer_path)
        layer_gdf["geometry"] = layer_gdf["geometry"].buffer(0)
        out[layer_name] = out.geometry.apply(
            lambda geom, layer=layer_gdf: (
                (layer.intersection(geom).area.sum() / geom.area) * 100 if geom.area > 0 else 0.0
            )
        )
    return out
