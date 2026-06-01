"""Behavioral contract for LSOA aggregation of change maps (paper §3.5).

Locks the meaning of the vectorized aggregation: each changed pixel is counted
once into the LSOA containing its centre, and the value is that count as a
percentage of the LSOA area. Independent of the (fast) implementation.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from gfs.modeling.features import aggregate_changes_to_lsoa


def test_change_pixels_counted_once_per_lsoa(tmp_path: Path) -> None:
    # 10x10 grid of 10 m pixels over a 100 m x 100 m extent (EPSG:27700).
    transform = from_origin(0, 100, 10, 10)
    changes = np.zeros((10, 10), dtype="float32")
    changes[0, 0:3] = 1.0  # 3 changed pixels in the left half (x-centres < 50)
    changes[0, 5:7] = 1.0  # 2 changed pixels in the right half

    changes_dir = tmp_path / "features_test"
    changes_dir.mkdir()
    with rasterio.open(
        changes_dir / "features_band_1_test.tiff",
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="float32",
        crs="EPSG:27700",
        transform=transform,
    ) as dst:
        dst.write(changes, 1)

    # Two LSOAs splitting the extent into left/right halves (each 50 x 100 = 5000 m^2).
    lsoa = gpd.GeoDataFrame(
        {"LSOA11CD": ["LEFT", "RIGHT"]},
        geometry=[box(0, 0, 50, 100), box(50, 0, 100, 100)],
        crs="EPSG:27700",
    )

    phi = aggregate_changes_to_lsoa(str(changes_dir), lsoa)
    col = phi.columns[0]

    # % = changed-pixel-count / area(m^2) * 100; pixels counted once each.
    assert phi.loc["LEFT", col] == 3 / 5000 * 100
    assert phi.loc["RIGHT", col] == 2 / 5000 * 100
