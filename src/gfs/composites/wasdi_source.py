"""Raw Sentinel-2 composites via WASDI (paper §3.2, the X source).

The annual cloud-masked median composites were built on the **WASDI** platform
with its ``s2_average_bands`` processor (one job per Sentinel-2 MGRS tile),
summer window, 11 bands at 10 m. This module wraps that workflow programmatically
(credentials from a WASDI config file, no interactive steps).

.. warning::
   WASDI is a **licensed** platform: the rights to use it — and at finer usage
   levels — must be arranged with the WASDI team. Each
   :func:`build_average_composite` call runs metered remote compute on your WASDI
   account, so it is **guarded behind an explicit opt-in** (``confirm_paid_run=
   True``) and is never invoked automatically. The composites already exist as
   data — you only need this to regenerate them.

Downstream stages read the produced GeoTIFFs from ``COMPOSITES_DIR``; they never
touch WASDI.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd

from gfs.config import GEOGRAPHIC_CRS, WASDI_CONFIG, composite_window

# The 11 Sentinel-2 bands requested from WASDI, at their native resolutions
# (the s2_average_bands processor resamples to SPATIAL_RESOLUTION_M).
WASDI_BAND_NAMES = (
    "B01_60m",
    "B02_10m",
    "B03_10m",
    "B04_10m",
    "B05_20m",
    "B06_20m",
    "B07_20m",
    "B08_10m",
    "B8A_20m",
    "B11_20m",
    "B12_20m",
)
WASDI_PROVIDER = "CREODIAS2"
SPATIAL_RESOLUTION_M = 10


def init_wasdi(config_path: str | None = None) -> Any:
    """Initialize the WASDI client from a config file (USER/PASSWORD/WORKSPACE).

    Returns the imported ``wasdi`` module, authenticated and ready. Headless: no
    interactive prompts. ``config_path`` defaults to :data:`gfs.config.WASDI_CONFIG`.
    """
    import wasdi

    wasdi.init(str(config_path or WASDI_CONFIG))
    return wasdi


def build_average_composite(
    boundary_path: str,
    year: int,
    *,
    confirm_paid_run: bool = False,
    provider: str = WASDI_PROVIDER,
    resolution_m: int = SPATIAL_RESOLUTION_M,
) -> list[str]:
    """Run WASDI ``s2_average_bands`` for every tile of the London bounding box.

    Submits one processor job per MGRS tile in the active workspace over the
    year's summer window, waits for completion, and returns the processed tile
    ids. Faithful to the dissertation notebook.

    .. warning::
       This runs metered compute on the WASDI platform; usage rights must be
       arranged with the WASDI team. It raises unless ``confirm_paid_run=True``
       is passed explicitly.
    """
    if not confirm_paid_run:
        raise RuntimeError(
            "build_average_composite runs metered WASDI compute (licensed "
            "platform — arrange usage rights with the WASDI team). The composites "
            "already exist as data; pass confirm_paid_run=True only if you really "
            "intend to regenerate them on your WASDI account."
        )

    wasdi = init_wasdi()
    gdf = gpd.read_file(boundary_path).to_crs(GEOGRAPHIC_CRS)
    min_lng, min_lat, max_lng, max_lat = (float(v) for v in gdf.total_bounds)
    tiles = [x.split("_")[1] for x in wasdi.getProductsByActiveWorkspace() if x.endswith(".tif")]

    start_date, end_date = composite_window(year)
    jobs: list[str] = []
    for tile in tiles:
        params = {
            "BBOX": {
                "northEast": {"lat": max_lat, "lng": max_lng},
                "southWest": {"lat": min_lat, "lng": min_lng},
            },
            "PROVIDER": provider,
            "DELETE": False,
            "START_DATE": start_date,
            "END_DATE": end_date,
            "BAND_NAMES": list(WASDI_BAND_NAMES),
            "TILE_NAMES": [tile],
            "SPATIAL_RESOLUTION_M": resolution_m,
        }
        jobs.append(wasdi.executeProcessor("s2_average_bands", params))

    wasdi.waitProcesses(jobs)
    return tiles
