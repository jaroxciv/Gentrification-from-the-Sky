"""Build a tidy, shareable LSOA-level GeoPackage of the study's outputs.

Bundles, per LSOA (keyed on LSOA11CD):
  - the gentrification score + components (neighborhood indices, the four
    percentile measures) and the ``disadvantaged`` flag,
  - a derived binary ``gentrified`` label (1 if score > top quartile, 0 if below
    bottom quartile, NaN for the middle 50% — the paper's §4.1 definition),
  - the per-band satellite change-detection features for one model.

This is what we hand to external researchers requesting the data.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd

from gfs.config import OUTPUTS_DIR

# CRS of the gentrification GeoPackage (British National Grid). The original
# notebook mis-tagged it EPSG:32630, but the coordinate bounds (easting
# ~503-562k, northing ~156-201k) are unambiguously EPSG:27700.
SOURCE_EPSG = 27700

# Score columns worth sharing (drop raw census counts / admin duplicates).
SCORE_COLUMNS = [
    "LSOA11CD",
    "LSOA11NM",
    "LAD11NM",  # borough, for reference
    "t1_age_perc",
    "t3_age_perc",
    "t1_edu_perc",
    "t3_edu_perc",
    "t1_house_perc",
    "t3_house_perc",
    "t1_income_perc",
    "t3_income_perc",
    "neighborhood_index_t1",
    "neighborhood_index_t3",
    "disadvantaged",
    "gentrification_score",
]


def add_binary_label(
    gdf: gpd.GeoDataFrame, score_col: str = "gentrification_score"
) -> gpd.GeoDataFrame:
    """Add a ``gentrified`` column: 1 above Q75, 0 below Q25, NaN in between."""
    q25 = gdf[score_col].quantile(0.25)
    q75 = gdf[score_col].quantile(0.75)
    label = pd.Series(pd.NA, index=gdf.index, dtype="Int64")
    label[gdf[score_col] > q75] = 1
    label[gdf[score_col] < q25] = 0
    gdf = gdf.copy()
    gdf["gentrified"] = label
    return gdf


def export_outputs(model: str = "tinycd", *, date: str = "100824") -> Path:
    """Join scores + a model's LSOA change features into a shareable GeoPackage."""
    scores_path = OUTPUTS_DIR / "gent_merged.gpkg"
    features_path = OUTPUTS_DIR / f"lsoa_changes_{model}_{date}.csv"

    scores = gpd.read_file(scores_path)
    if scores.crs is None:
        scores = scores.set_crs(epsg=SOURCE_EPSG)

    keep = [c for c in SCORE_COLUMNS if c in scores.columns] + ["geometry"]
    scores = cast(gpd.GeoDataFrame, scores[keep])
    scores = add_binary_label(scores)

    features = pd.read_csv(features_path)
    merged = cast(gpd.GeoDataFrame, scores.merge(features, on="LSOA11CD", how="left"))
    merged = merged.set_crs(epsg=SOURCE_EPSG, allow_override=True)

    out_path = OUTPUTS_DIR / f"gentrification_outputs_{model}.gpkg"
    merged.to_file(out_path, driver="GPKG")
    return out_path
