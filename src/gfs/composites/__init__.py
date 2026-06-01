"""X — Satellite feature preparation.

The two annual Sentinel-2 composites (2016, 2021) that feed change detection
(paper §3.2). Provenance of the raw data:

- :mod:`gfs.composites.stac_source` — free, open Sentinel-2 L2A via a STAC API
  (the default; same product as below).
- :mod:`gfs.composites.wasdi_source` — the as-published composites built on the
  **WASDI** platform (``s2_average_bands``), cloud-masked summer median, 11 bands.
- :mod:`gfs.composites.landcover` — **Earth Engine** Dynamic World land cover
  (the green-areas layer used to mask change detection).
- :mod:`gfs.composites.clip_merge` / :mod:`gfs.composites.stats` /
  :mod:`gfs.composites.raster_io` — local rasterio/rioxarray clipping,
  mosaicking, statistics and shared raster I/O.

Both upstream sources (WASDI, Earth Engine) ship their outputs as data, so the
downstream pipeline runs without either; the modules exist to regenerate them.
"""

from gfs.composites.clip_merge import clip_and_merge_year
from gfs.composites.landcover import export_dynamic_world, initialize_ee
from gfs.composites.raster_io import read_raster, write_raster
from gfs.composites.stac_source import build_composite_stac
from gfs.composites.stats import (
    band_statistics,
    load_pair,
    masked_difference,
    zonal_statistics,
)
from gfs.composites.wasdi_source import build_average_composite

__all__ = [
    "build_composite_stac",
    "build_average_composite",
    "export_dynamic_world",
    "initialize_ee",
    "clip_and_merge_year",
    "load_pair",
    "masked_difference",
    "band_statistics",
    "zonal_statistics",
    "read_raster",
    "write_raster",
]
