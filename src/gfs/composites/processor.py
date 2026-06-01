"""Build cloud-masked Sentinel-2 median composites via Google Earth Engine (paper §3.2).

For each study year the pipeline pulls every Sentinel-2 Level-2A scene that
covers Greater London during the low-cloud summer window (Jun 1 - Aug 31),
masks clouds/cirrus with the scene QA band, and reduces the collection to a
*per-pixel median* composite of all 11 bands at a uniform 10 m resolution. The
raw composite is then exported as a multi-band GeoTIFF for the downstream
clip / merge / change-detection stages.

The original notebook produced the same product through the WASDI
``s2_average_bands`` processor (one job per Sentinel-2 MGRS tile). That service
is not part of this package's dependency set, so the equivalent computation is
expressed directly against the Earth Engine ``COPERNICUS/S2_SR_HARMONIZED``
collection here, preserving the study parameters: the 2016/2021 years, the
summer date window, the 11-band selection, the 10 m target resolution and the
cloud-masked reduction.

Earth Engine is *not* initialized at import time. Call :func:`initialize_ee`
once per session (it authenticates on first use), then build composites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import ee

from gfs.config import (
    COMPOSITE_MONTH_END,
    COMPOSITE_MONTH_START,
    SENTINEL2_BANDS,
    TARGET_RESOLUTION_M,
)

# Earth Engine Sentinel-2 surface-reflectance collection (cloud-masked L2A).
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

# QA60 cloud-mask bit positions (Sentinel-2 quality band): opaque clouds and
# cirrus. Both must be clear for a pixel to enter the median composite.
_QA60_CLOUD_BIT = 1 << 10
_QA60_CIRRUS_BIT = 1 << 11

# Scenes with more than this cloud cover are dropped before compositing.
DEFAULT_MAX_CLOUD_PCT = 60


@dataclass(frozen=True)
class CompositeSpec:
    """Parameters for one annual Sentinel-2 median composite (paper §3.2).

    ``bands`` defaults to the study's 11-band selection and ``scale`` to the
    10 m target resolution; the summer window comes from :mod:`gfs.config`.
    """

    year: int
    bands: tuple[str, ...] = SENTINEL2_BANDS
    month_start: int = COMPOSITE_MONTH_START
    month_end: int = COMPOSITE_MONTH_END
    scale_m: int = TARGET_RESOLUTION_M
    max_cloud_pct: int = DEFAULT_MAX_CLOUD_PCT
    collection: str = S2_COLLECTION

    @property
    def start_date(self) -> str:
        """Inclusive composite window start, ``YYYY-06-01``."""
        return f"{self.year}-{self.month_start:02d}-01"

    @property
    def end_date(self) -> str:
        """Exclusive composite window end, ``YYYY-09-01`` (covers Aug 31)."""
        return f"{self.year}-{self.month_end + 1:02d}-01"


@dataclass
class ExportTask:
    """Handle to a queued Earth Engine export, plus its destination."""

    task: Any
    description: str
    file_prefix: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def initialize_ee(project: str | None = None) -> None:
    """Authenticate (first run only) and initialize the Earth Engine session.

    Kept out of import time so importing this module never triggers a browser
    auth flow. Call once before building composites.
    """
    try:
        ee.Initialize(project=project)
    except Exception:  # noqa: BLE001 - EE raises a generic error if not yet authed.
        ee.Authenticate()
        ee.Initialize(project=project)


def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Mask opaque-cloud and cirrus pixels using the Sentinel-2 QA60 band.

    Mirrors the canonical Earth Engine Sentinel-2 cloud mask: a pixel is kept
    only when both QA60 cloud bits are zero. Reflectance is scaled to physical
    units (divide by 10000) so the median composite is in reflectance.
    """
    qa = image.select("QA60")
    clear = (
        qa.bitwiseAnd(_QA60_CLOUD_BIT).eq(0).And(qa.bitwiseAnd(_QA60_CIRRUS_BIT).eq(0))
    )
    return image.updateMask(clear).divide(10000).copyProperties(image, ["system:time_start"])


def london_geometry(boundary_gdf: Any) -> ee.Geometry:
    """Convert a GeoPandas London-boundary frame into an Earth Engine geometry.

    The frame is reprojected to EPSG:4326 (Earth Engine's working CRS) and its
    dissolved bounding extent is returned as an ``ee.Geometry`` for filtering /
    clipping the collection.
    """
    bounds = boundary_gdf.to_crs(4326).total_bounds
    min_lng, min_lat, max_lng, max_lat = (float(v) for v in bounds)
    return ee.Geometry.Rectangle([min_lng, min_lat, max_lng, max_lat])


def build_median_composite(spec: CompositeSpec, region: ee.Geometry) -> ee.Image:
    """Build the cloud-masked median composite for one year (paper §3.2).

    Filters ``spec.collection`` to ``region`` and the summer window, drops
    cloudy scenes, masks remaining clouds, takes the per-pixel median over the
    11 study bands, and clips to ``region``.
    """
    collection = (
        ee.ImageCollection(spec.collection)
        .filterBounds(region)
        .filterDate(spec.start_date, spec.end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", spec.max_cloud_pct))
        .map(mask_s2_clouds)
    )
    median = cast(ee.Image, collection.median())
    return median.select(list(spec.bands)).clip(region)


def export_composite_to_drive(
    composite: ee.Image,
    spec: CompositeSpec,
    region: ee.Geometry,
    *,
    folder: str = "gfs_composites",
    file_name: str | None = None,
) -> ExportTask:
    """Queue a Drive export of the multi-band composite GeoTIFF and start it.

    Exports at ``spec.scale_m`` (10 m) in EPSG:4326 with no max-pixel cap
    removed beyond Earth Engine's batch limit. Returns the started task so the
    caller can poll ``task.status()``. The downstream stages download the TIFF
    and clip/merge it (see :mod:`gfs.composites.clip_merge`).
    """
    name = file_name or f"composite_{spec.year}"
    # ee.batch is available at runtime but not re-exported from the ee package.
    ee_any = cast(Any, ee)
    task = cast(
        Any,
        ee_any.batch.Export.image.toDrive(
            image=composite,
            description=name,
            folder=folder,
            fileNamePrefix=name,
            region=region,
            scale=spec.scale_m,
            crs="EPSG:4326",
            maxPixels=int(1e13),
            fileFormat="GeoTIFF",
        ),
    )
    task.start()
    return ExportTask(task=task, description=name, file_prefix=name, extra={"folder": folder})


def download_composite_local(
    composite: ee.Image,
    spec: CompositeSpec,
    region: ee.Geometry,
    out_path: str,
) -> str:
    """Download the composite straight to a local GeoTIFF via geemap.

    Convenience alternative to the Drive export for the (small) London extent;
    blocks until the file is written. Returns ``out_path``.
    """
    import geemap

    cast(Any, geemap).ee_export_image(
        composite,
        filename=out_path,
        scale=spec.scale_m,
        region=region,
        crs="EPSG:4326",
        file_per_band=False,
    )
    return out_path
