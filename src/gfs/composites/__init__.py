"""X — Satellite feature preparation.

The two annual Sentinel-2 composites (2016, 2021) that feed change detection
(paper §3.2). Provenance of the raw data:

- :mod:`gfs.composites.wasdi_source` — the composites were built on the **WASDI**
  platform (``s2_average_bands``), cloud-masked summer median, 11 bands at 10 m.
- :mod:`gfs.composites.landcover` — **Earth Engine** Dynamic World land cover
  (the green-areas layer used to mask change detection).
- :mod:`gfs.composites.clip_merge` / :mod:`gfs.composites.stats` — local
  rasterio/rioxarray clipping, mosaicking and difference/zonal statistics.

Both upstream sources (WASDI, Earth Engine) ship their outputs as data, so the
downstream pipeline runs without either; the modules exist to regenerate them.
"""
