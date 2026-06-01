"""Stage 4 — build the modeling table and train/evaluate the classifiers.

End-to-end wiring of the modeling stage (paper §3.5, §4-5):

1. Aggregate the change-detection maps for one model to LSOA-level features Phi
   (:mod:`gfs.modeling.features`).
2. Join the gentrification score + planning-layer predictors and binarize the
   target by top/bottom quartile (:mod:`gfs.modeling.dataset`).
3. Grid-search + repeated-CV evaluate Logistic Regression, Linear SVC and
   XGBoost on balanced accuracy, weighted F1 and ROC-AUC
   (:mod:`gfs.modeling.classify`).

    uv run scripts/04_run_modeling.py

Paths are declared in ``Inputs`` below; the ~80 GB of rasters/boundaries must be
present for an actual run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd

from gfs.config import (
    BOUNDARIES_DIR,
    OUTPUTS_DIR,
    PLANNING_DIR,
)
from gfs.modeling import classify, dataset
from gfs.modeling.features import (
    aggregate_changes_to_lsoa,
    create_percentage_features,
)


@dataclass(frozen=True)
class Inputs:
    """Input locations for the modeling stage."""

    model_name: str = "tinycd"
    lsoa_boundaries: Path = BOUNDARIES_DIR / "LSOA_2011_London_gen_MHW.shp"
    score_csv: Path = OUTPUTS_DIR / "gentrification_score.csv"
    planning_dir: Path = PLANNING_DIR

    @property
    def features_dir(self) -> Path:
        """``outputs/features_<model>/`` — the per-band change maps for this model."""
        return OUTPUTS_DIR / f"features_{self.model_name}"


def _planning_layer_paths(planning_dir: Path) -> dict[str, str]:
    """Map planning-layer name -> gpkg path, skipping the opportunity-areas layer."""
    layers: dict[str, str] = {}
    for path in sorted(planning_dir.glob("*.gpkg")):
        if path.name == "London_Plan_Opportunity_Areas.gpkg":
            continue
        layers[path.stem] = str(path)
    return layers


def run(inputs: Inputs) -> pd.DataFrame:
    """Build the table, train/evaluate the classifiers, return the metrics frame."""
    lsoa_gdf = cast("gpd.GeoDataFrame", gpd.read_file(inputs.lsoa_boundaries))

    # 1. Satellite feature matrix Phi (% changed pixels per LSOA per band).
    lsoa_changes = aggregate_changes_to_lsoa(str(inputs.features_dir), lsoa_gdf)
    lsoa_changes = lsoa_changes.reset_index()

    # 2. Planning-layer predictors (% LSOA area per layer).
    planning_layers = _planning_layer_paths(inputs.planning_dir)
    planning_gdf = create_percentage_features(lsoa_gdf, planning_layers)
    planning = cast("pd.DataFrame", planning_gdf.drop(columns="geometry"))

    # 3. Gentrification score (Y) — joined on the LSOA code.
    score = pd.read_csv(inputs.score_csv)

    table = dataset.assemble_modeling_table(score, lsoa_changes, planning)
    x = cast("pd.DataFrame", table.data[table.predictors])
    y = cast("pd.Series", table.data[table.target])
    print(f"Modeling table: {table.data.shape}, {len(table.predictors)} predictors")

    results = classify.train_and_evaluate(x, y)
    metrics = classify.results_to_frame(results)
    print(metrics.to_string(index=False))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"modeling_metrics_{inputs.model_name}.csv"
    metrics.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return metrics


if __name__ == "__main__":
    run(Inputs())
