"""Stage 5 — change-threshold ablation (paper §5).

Reproduces the ablation study: sweep the simple-difference change threshold,
aggregate each threshold's change maps to LSOA features, and refit a balanced
random forest at every threshold, recording F1 / balanced accuracy / ROC-AUC.

    uv run scripts/05_ablation.py

The thresholded change maps live under ``outputs/thresholds/`` (one tiff per
band per threshold, suffixed by the integer threshold). This script aggregates
them to LSOA level, joins the score + planning predictors, and runs the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd

from gfs.config import BOUNDARIES_DIR, OUTPUTS_DIR, PLANNING_DIR
from gfs.modeling import ablation, dataset
from gfs.modeling.features import (
    aggregate_changes_to_lsoa,
    create_percentage_features,
)


@dataclass(frozen=True)
class Inputs:
    """Input locations for the ablation stage."""

    thresholds_dir: Path = OUTPUTS_DIR / "thresholds"
    lsoa_boundaries: Path = BOUNDARIES_DIR / "LSOA_2011_London_gen_MHW.shp"
    score_csv: Path = OUTPUTS_DIR / "gentrification_score.csv"
    planning_dir: Path = PLANNING_DIR


def _planning_layer_paths(planning_dir: Path) -> dict[str, str]:
    """Map planning-layer name -> gpkg path, skipping the opportunity-areas layer."""
    layers: dict[str, str] = {}
    for path in sorted(planning_dir.glob("*.gpkg")):
        if path.name == "London_Plan_Opportunity_Areas.gpkg":
            continue
        layers[path.stem] = str(path)
    return layers


def run(inputs: Inputs) -> pd.DataFrame:
    """Aggregate thresholded change maps, sweep thresholds, return the metrics frame."""
    lsoa_gdf = cast("gpd.GeoDataFrame", gpd.read_file(inputs.lsoa_boundaries))

    # All thresholded change maps -> one column per band+threshold.
    lsoa_changes = aggregate_changes_to_lsoa(str(inputs.thresholds_dir), lsoa_gdf)
    lsoa_changes = lsoa_changes.reset_index()

    planning_layers = _planning_layer_paths(inputs.planning_dir)
    planning_gdf = create_percentage_features(lsoa_gdf, planning_layers)
    planning = cast("pd.DataFrame", planning_gdf.drop(columns="geometry"))
    planning_predictors = dataset.planning_predictors(planning)

    score = pd.read_csv(inputs.score_csv)
    merged = cast(
        "pd.DataFrame",
        score.merge(lsoa_changes, on=dataset.LSOA_CODE_COL).merge(
            planning, on=dataset.LSOA_CODE_COL
        ),
    )
    merged = dataset.binarize_score(merged)

    points = ablation.evaluate_thresholds(merged, planning_predictors)
    frame = ablation.ablation_to_frame(points)
    print(frame.to_string(index=False))

    best = ablation.best_threshold(points, metric="f1")
    print(f"Best F1 threshold: {best.threshold} (F1={best.f1:.4f})")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "ablation_threshold_sweep.csv"
    frame.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return frame


if __name__ == "__main__":
    run(Inputs())
