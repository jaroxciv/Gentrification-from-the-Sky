"""Modeling — Predicting gentrification.

Aggregates change-detection outputs to LSOA-level feature vectors (paper §3.5),
binarizes the gentrification score into a classification target (§4.1), and
trains/evaluates baseline and satellite-enhanced classifiers — Logistic
Regression, Linear SVC, XGBoost — plus the thresholding ablation (§4–5).
"""

from gfs.modeling.ablation import ablation_to_frame, evaluate_thresholds
from gfs.modeling.classify import (
    ClassifyConfig,
    results_to_frame,
    train_and_evaluate,
)
from gfs.modeling.dataset import (
    ModelingTable,
    assemble_modeling_table,
    binarize_score,
)
from gfs.modeling.evaluation import score_estimator
from gfs.modeling.features import aggregate_changes_to_lsoa, create_percentage_features

__all__ = [
    "assemble_modeling_table",
    "binarize_score",
    "ModelingTable",
    "aggregate_changes_to_lsoa",
    "create_percentage_features",
    "train_and_evaluate",
    "results_to_frame",
    "ClassifyConfig",
    "evaluate_thresholds",
    "ablation_to_frame",
    "score_estimator",
]
