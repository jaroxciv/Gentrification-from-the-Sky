"""Dynamic World land cover via Google Earth Engine (the green-areas layer).

Earth Engine's only role in the study: fetch Google **Dynamic World V1** land
cover for London (the 9 classes including ``trees``/``grass`` → green space),
summer 2021, at 10 m, and export it as a GeoTIFF. The change-detection stage
uses this layer to build a green-areas mask (paper's green-space angle).

Authentication is programmatic and headless:
  * if a **service account** is configured (``GFS_GEE_SERVICE_ACCOUNT`` +
    ``GFS_GEE_SERVICE_ACCOUNT_KEY``), it is used — no browser, works on HPC/CI;
  * otherwise persistent user credentials (from a one-time
    ``earthengine authenticate``) are used.

The output (``data/dynamic_world_london_10m.tif``) already exists; this module is
only needed to regenerate it.
"""

from __future__ import annotations

from typing import Any

from gfs.config import (
    DATA_DIR,
    GEE_PROJECT,
    GEE_SERVICE_ACCOUNT,
    GEE_SERVICE_ACCOUNT_KEY,
    GEOGRAPHIC_CRS,
    WORKING_CRS,
    YEAR_T2,
    composite_window,
)

# Dynamic World class labels (band "label" values 0-8), trees/grass = green.
DYNAMIC_WORLD_CLASSES = (
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
)


def initialize_ee(
    project: str = GEE_PROJECT,
    *,
    service_account: str | None = GEE_SERVICE_ACCOUNT,
    key_file: str | None = GEE_SERVICE_ACCOUNT_KEY,
) -> None:
    """Initialize Earth Engine, headless via a service account when configured.

    Falls back to persistent user credentials (a prior ``earthengine
    authenticate``) when no service account is set. Never triggers an interactive
    flow itself.
    """
    import ee

    if service_account and key_file:
        credentials = ee.ServiceAccountCredentials(service_account, key_file)
        ee.Initialize(credentials, project=project)
    else:
        ee.Initialize(project=project)


def export_dynamic_world(
    boundary_path: str,
    out_path: str | None = None,
    *,
    year: int = YEAR_T2,
    scale: int = 10,
    crs: str = WORKING_CRS,
) -> str:
    """Export the Dynamic World land-cover class image for London to a GeoTIFF.

    Mirrors the notebook: build the Dynamic World ``class`` image over the year's
    summer window clipped to the London wards, and download it locally (the
    layer is small at 10 m, so a direct download is fine — no Drive needed).
    Call :func:`initialize_ee` first.
    """
    import ee
    import geemap
    import geopandas as gpd

    if out_path is None:
        out_path = str(DATA_DIR / f"dynamic_world_london_{scale}m.tif")

    # Use the boundary's bounding rectangle (4 vertices, in WGS84) as the region.
    # Passing the full London outline — thousands of vertices — would inflate the
    # Earth Engine request expression past its size limit; the bbox keeps it tiny.
    min_lng, min_lat, max_lng, max_lat = (
        float(v) for v in gpd.read_file(boundary_path).to_crs(GEOGRAPHIC_CRS).total_bounds
    )
    region: Any = ee.Geometry.Rectangle([min_lng, min_lat, max_lng, max_lat])

    start_date, end_date = composite_window(year)
    landcover: Any = geemap.dynamic_world(
        region, start_date, end_date, return_type="class", clip=True
    )
    # download_ee_image (geedim-backed) fetches the image as tiles, so it is not
    # bound by Earth Engine's single-request download cap (unlike ee_export_image).
    geemap.download_ee_image(
        landcover,
        filename=out_path,
        scale=scale,
        region=region,
        crs=crs,
    )
    return out_path
