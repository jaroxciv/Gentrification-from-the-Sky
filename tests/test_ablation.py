"""Behavioral contract for the §5 threshold ablation's change-map generation.

The point of generation is that the sweep can run end-to-end: each
``(band, threshold)`` map must aggregate to a column the sweep selects by its
``_<label>`` suffix, and green/natural pixels must be excluded from change.
These lock that contract, independent of the (fast, parallel) implementation.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from gfs.change_detection.threshold_ablation import generate_threshold_change_maps
from gfs.modeling.ablation import change_columns_for_threshold
from gfs.modeling.features import aggregate_changes_to_lsoa


def _write_raster(path: Path, array: np.ndarray, transform: rasterio.Affine) -> None:
    bands = array if array.ndim == 3 else array[np.newaxis]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=bands.shape[1],
        width=bands.shape[2],
        count=bands.shape[0],
        dtype=str(bands.dtype),
        crs="EPSG:32630",
        transform=transform,
    ) as dst:
        dst.write(bands)


def _scene(tmp_path: Path, landcover_class: int) -> tuple[str, str, str]:
    """A tiny bi-temporal pair (one band) plus a constant land-cover raster."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    transform = from_origin(0, 200, 10, 10)
    ramp = np.add.outer(np.arange(20.0), np.arange(20.0)).astype("float32")  # nonzero std
    t1 = ramp[np.newaxis].copy()
    t2 = ramp[np.newaxis].copy()
    t2[0, 4:16, 4:16] += 5000.0  # a large, well-above-speckle block of change
    landcover = np.full((20, 20), landcover_class, dtype="uint8")

    p1, p2, plc = tmp_path / "t1.tiff", tmp_path / "t2.tiff", tmp_path / "lc.tiff"
    _write_raster(p1, t1, transform)
    _write_raster(p2, t2, transform)
    _write_raster(plc, landcover, transform)
    return str(p1), str(p2), str(plc)


def test_threshold_maps_are_selectable_by_the_sweep(tmp_path: Path) -> None:
    p1, p2, plc = _scene(tmp_path, landcover_class=6)  # 6 = built (not green)
    out_dir = tmp_path / "thresholds"
    thresholds = [100.0, 200.0]

    generate_threshold_change_maps(
        p1, p2, plc, str(out_dir), thresholds, bands=[1], min_region_size=1, n_jobs=1
    )

    lsoa = gpd.GeoDataFrame({"LSOA11CD": ["A"]}, geometry=[box(0, 0, 200, 200)], crs="EPSG:32630")
    phi = aggregate_changes_to_lsoa(str(out_dir), lsoa)

    # Each threshold yields exactly one selectable change column (suffix _<int>).
    for threshold in thresholds:
        cols = change_columns_for_threshold(phi, int(threshold))
        assert len(cols) == 1, f"threshold {threshold} -> {cols}"


def test_green_pixels_excluded_from_change(tmp_path: Path) -> None:
    out_dir, lsoa = (
        tmp_path / "th",
        gpd.GeoDataFrame({"LSOA11CD": ["A"]}, geometry=[box(0, 0, 200, 200)], crs="EPSG:32630"),
    )

    # Same scene, built vs all-green land cover.
    built = _scene(tmp_path / "built", landcover_class=6)
    green = _scene(tmp_path / "green", landcover_class=1)  # 1 = trees (green)

    generate_threshold_change_maps(
        *built[:2], built[2], str(out_dir / "built"), [5.0], bands=[1], min_region_size=1, n_jobs=1
    )
    generate_threshold_change_maps(
        *green[:2], green[2], str(out_dir / "green"), [5.0], bands=[1], min_region_size=1, n_jobs=1
    )

    built_phi = aggregate_changes_to_lsoa(str(out_dir / "built"), lsoa)
    green_phi = aggregate_changes_to_lsoa(str(out_dir / "green"), lsoa)

    # Built land cover detects change; an all-green scene masks it all out.
    assert built_phi.to_numpy().sum() > 0
    assert green_phi.to_numpy().sum() == 0
