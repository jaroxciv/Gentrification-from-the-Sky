"""Sentinel-2 composites via STAC — a free, headless alternative to WASDI.

Builds the same product as the paper's WASDI composites (summer cloud-masked
median, the 11 study bands at 10 m) but from open, cloud-native COGs discovered
through a STAC API. No licence, no account, no quota: the default endpoint is
Element84's ``earth-search`` over the AWS Open Data Sentinel-2 L2A archive, read
anonymously.

Pipeline: ``pystac-client`` searches scenes by bounding box / summer window /
cloud cover, ``odc-stac`` lazily loads the bands (+ the SCL scene-classification
mask) into an ``xarray`` cube, clouds/shadows are masked via SCL, and the
per-pixel median over time is written as a multi-band GeoTIFF.

This is the recommended default for regenerating composites; :mod:`wasdi_source`
remains as the faithful "as-published" route.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd

from gfs.config import (
    COMPOSITES_DIR,
    GEOGRAPHIC_CRS,
    WORKING_CRS,
    composite_window,
)

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"

# The 11 study bands (paper §3.2) mapped to earth-search L2A asset names.
BAND_ASSETS: dict[str, str] = {
    "B1": "coastal",
    "B2": "blue",
    "B3": "green",
    "B4": "red",
    "B5": "rededge1",
    "B6": "rededge2",
    "B7": "rededge3",
    "B8": "nir",
    "B8A": "nir08",
    "B11": "swir16",
    "B12": "swir22",
}

# Sentinel-2 Scene Classification (SCL) values to drop: no-data, saturated,
# cloud shadow, cloud medium/high probability, thin cirrus.
SCL_DROP = (0, 1, 3, 8, 9, 10)


def search_sentinel2(
    bbox: tuple[float, float, float, float],
    year: int,
    *,
    max_cloud: float = 20.0,
    stac_url: str = EARTH_SEARCH_URL,
    collection: str = S2_COLLECTION,
) -> list[Any]:
    """Find summer Sentinel-2 L2A scenes intersecting ``bbox`` for ``year``."""
    import pystac_client

    start, end = composite_window(year)
    client = pystac_client.Client.open(stac_url)
    search = client.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    return list(search.items())


def build_composite_stac(
    boundary_path: str,
    year: int,
    out_path: str | None = None,
    *,
    max_cloud: float = 20.0,
    resolution: int = 10,
    crs: str = WORKING_CRS,
) -> str:
    """Build a cloud-masked summer median composite for ``year`` and write a GeoTIFF.

    Loads the 11 study bands + SCL for the scenes over the boundary's bounding
    box, masks clouds/shadows via SCL, takes the per-pixel median over time, and
    saves a multi-band GeoTIFF. The default name matches what the change-detection
    stage reads (``clipped_merged_<year>.tiff``), so the two chain directly.
    """
    import odc.stac
    import rioxarray  # noqa: F401  (registers the .rio accessor used below)

    if out_path is None:
        out_path = str(COMPOSITES_DIR / f"clipped_merged_{year}.tiff")

    gdf = gpd.read_file(boundary_path).to_crs(GEOGRAPHIC_CRS)
    b = gdf.total_bounds
    bbox: tuple[float, float, float, float] = (
        float(b[0]),
        float(b[1]),
        float(b[2]),
        float(b[3]),
    )
    items = search_sentinel2(bbox, year, max_cloud=max_cloud)
    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes found for {year} over {bbox}")

    assets = list(BAND_ASSETS.values())
    cube: Any = odc.stac.load(
        items,
        bands=[*assets, "scl"],
        bbox=bbox,
        resolution=resolution,
        crs=crs,
        chunks={},
        groupby="solar_day",
    )

    # Mask clouds/shadows via the scene-classification layer, then median over time.
    clear = ~cube["scl"].isin(SCL_DROP)
    bands = cube[assets].where(clear)
    median = bands.to_array(dim="band").median(dim="time")
    median = median.rio.write_crs(crs)

    COMPOSITES_DIR.mkdir(parents=True, exist_ok=True)
    median.rio.to_raster(out_path)
    return out_path
