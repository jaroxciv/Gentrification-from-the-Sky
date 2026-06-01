"""LSOA-level aggregation of change-detection outputs (paper §3.5).

Turns the per-band binary change maps produced by the change-detection stage
(``outputs/features_<model>/features_band_<b>_<model>.tiff``) into the feature
matrix Phi: one row per LSOA, one column per band, holding the **percentage of
changed pixels** that fall inside the LSOA polygon.

The pipeline mirrors the notebook exactly:

1. **Raster -> vector**: every pixel with a change value ``> 0`` becomes a small
   ``box`` polygon (one cell wide) in the raster CRS.
2. **Spatial join**: those change cells are joined to the LSOA polygons
   (``predicate="intersects"``) and counted per LSOA code.
3. **Normalize**: the count is divided by the LSOA area and scaled to a
   percentage, giving the change density per band.

The planning-layer features (per-LSOA % area covered by each London planning
layer) are also assembled here, since they share the raster->LSOA logic and are
the non-satellite half of the predictor set (paper §3.5/§5).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed
from shapely.geometry import box

# Column in the LSOA boundary shapefile holding the 2011 LSOA code.
LSOA_CODE_COL = "LSOA11CD"

# Loosely-typed CRS-like value handed back by rasterio/geopandas.
CRSLike = Any


@dataclass(frozen=True)
class AggregationConfig:
    """Knobs for the raster->LSOA aggregation.

    ``n_jobs=-1`` lets joblib use all cores (the notebook capped it at the file
    count); ``min_change_value`` keeps any pixel strictly greater than 0, which
    is how the binary/segmented change maps encode "changed".
    """

    min_change_value: float = 0.0
    n_jobs: int = -1


def raster_to_change_gdf(
    changes: np.ndarray,
    transform: rasterio.Affine,
    crs: CRSLike,
    *,
    min_change_value: float = 0.0,
) -> gpd.GeoDataFrame:
    """Vectorize the changed pixels of a raster band into one box per cell.

    Each pixel with value ``> min_change_value`` becomes a one-cell ``box``
    polygon placed at its georeferenced position (paper §3.5, raster -> vector).
    """
    mask = changes > min_change_value
    rows, cols = np.where(mask)
    geometries: list[Any] = []
    for r, c in zip(rows, cols, strict=True):
        x, y = cast("tuple[float, float]", transform * (c, r))
        geometries.append(box(x, y, x + transform[0], y + transform[4]))
    return gpd.GeoDataFrame(geometry=geometries, crs=crs)


def count_change_pixels(change_file: str, lsoa_gdf: gpd.GeoDataFrame) -> pd.Series:
    """Count changed pixels per LSOA for a single change-map tiff.

    Returns a Series indexed by LSOA code whose ``name`` is the tiff's stem (so
    joining many bands gives one column per band). Mirrors the notebook's
    ``count_change_pixels`` verbatim.
    """
    with rasterio.open(change_file) as src:
        changes = src.read(1)
        changes_transform = src.transform
        changes_crs = src.crs

    changes_gdf = raster_to_change_gdf(changes, changes_transform, changes_crs)

    # Reproject LSOA to match raster CRS if needed.
    if lsoa_gdf.crs != changes_gdf.crs:
        lsoa_gdf = lsoa_gdf.to_crs(cast("CRSLike", changes_gdf.crs))

    # Fix potential geometry issues, then spatial-join and count per LSOA.
    lsoa_gdf = lsoa_gdf.copy()
    lsoa_gdf["geometry"] = lsoa_gdf.buffer(0)
    join_gdf = gpd.sjoin(changes_gdf, lsoa_gdf, how="left", predicate="intersects")
    change_counts = cast("pd.Series", join_gdf.groupby(LSOA_CODE_COL).size())
    change_counts.name = os.path.basename(change_file).split(".")[0]
    return change_counts


def list_change_files(changes_dir: str) -> list[str]:
    """List the per-band change tiffs in a ``features_<model>`` folder, band-sorted.

    Filenames look like ``features_band_<b>_<model>.tiff``; we sort by the band
    number, which is the second-to-last underscore-delimited token.
    """
    change_files = [
        os.path.join(changes_dir, x)
        for x in os.listdir(changes_dir)
        if x.endswith(".tiff")
    ]
    return sorted(change_files, key=lambda x: int(x.split("_")[-2]))


def aggregate_changes_to_lsoa(
    changes_dir: str,
    lsoa_gdf: gpd.GeoDataFrame,
    config: AggregationConfig | None = None,
) -> pd.DataFrame:
    """Build the satellite feature matrix Phi: % changed pixels per LSOA per band.

    For every change-map tiff in ``changes_dir`` counts the changed pixels inside
    each LSOA (in parallel via joblib), then converts counts to a percentage of
    the LSOA's area (paper §3.5). Returns a DataFrame indexed by LSOA code with
    one column per band tiff.
    """
    cfg = config or AggregationConfig()
    sorted_file_paths = list_change_files(changes_dir)

    lsoa_changes = cast(
        "pd.DataFrame", lsoa_gdf[[LSOA_CODE_COL]].copy()
    ).set_index(LSOA_CODE_COL)

    n_jobs = cfg.n_jobs
    if n_jobs <= 0:
        n_jobs = min(os.cpu_count() or 1, max(len(sorted_file_paths), 1))
    results = cast(
        "list[pd.Series]",
        Parallel(n_jobs=n_jobs)(
            delayed(count_change_pixels)(change_file, lsoa_gdf)
            for change_file in sorted_file_paths
        ),
    )

    for change_counts in results:
        lsoa_changes = lsoa_changes.join(change_counts, how="left")
    lsoa_changes = lsoa_changes.fillna(0)

    # Convert raw counts to a percentage of LSOA area.
    areas = cast("gpd.GeoSeries", lsoa_gdf.set_index(LSOA_CODE_COL).geometry).area
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
                layer.intersection(geom).area.sum() / geom.area
            )
            * 100
            if geom.area > 0
            else 0.0
        )
    return out
