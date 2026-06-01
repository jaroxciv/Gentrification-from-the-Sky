"""Change-threshold ablation sweep (paper §5, ablation study).

The simple-difference change detector exposes one free knob: the magnitude
threshold above which a per-band pixel difference counts as "changed". This
module reproduces the notebook's ablation: build change maps over a sweep of
thresholds, aggregate each to LSOA-level features, and refit a classifier
(``BalancedRandomForestClassifier`` on ``simple_diff`` features) at every
threshold, recording F1 / balanced-accuracy / ROC-AUC.

The thresholded change maps and their LSOA aggregation are produced upstream
(``scripts/05_ablation.py`` calls :mod:`gfs.modeling.features`); here we take a
prepared table whose change columns are suffixed by their integer threshold and
sweep the model across them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold

from gfs.config import RANDOM_STATE
from gfs.modeling.classify import build_pipeline
from gfs.modeling.dataset import TARGET_COL
from gfs.modeling.evaluation import score_estimator

# Threshold sweep used in the notebook ablation: 30 points from 100 to 800.
THRESHOLD_MIN = 100.0
THRESHOLD_MAX = 800.0
N_THRESHOLDS = 30


def default_thresholds() -> np.ndarray:
    """The 30 evenly spaced reflectance-difference thresholds swept (100..800).

    Returned as floats (the values actually applied to the difference image);
    the ``_<threshold>`` column/file suffix is the truncated integer label (see
    :func:`gfs.change_detection.threshold_ablation.threshold_label`), and both
    generation and :func:`evaluate_thresholds` truncate identically so they agree.
    """
    return np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, N_THRESHOLDS)


def change_columns_for_threshold(df: pd.DataFrame, threshold: int) -> list[str]:
    """Change-feature columns for one threshold (names ending ``_<threshold>``).

    Matches the underscore-anchored suffix so e.g. threshold 10 does not also
    capture columns ending in 110/210 (a bare ``endswith("10")`` would).
    """
    return [c for c in df.columns if c.endswith(f"_{threshold}")]


def threshold_predictors(
    df: pd.DataFrame,
    threshold: int,
    *,
    planning_predictors: list[str],
) -> list[str]:
    """That threshold's change columns plus the (threshold-invariant) planning ones."""
    return change_columns_for_threshold(df, threshold) + planning_predictors


@dataclass
class AblationPoint:
    """Metrics for one threshold in the sweep."""

    threshold: float
    f1: float
    balanced_accuracy: float
    roc_auc: float


def default_model() -> ClassifierMixin:
    """The ablation classifier: a balanced random forest (paper §5)."""
    return BalancedRandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE)


def evaluate_thresholds(
    df: pd.DataFrame,
    planning_predictors: list[str],
    *,
    model: ClassifierMixin | None = None,
    thresholds: np.ndarray | None = None,
    target: str = TARGET_COL,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> list[AblationPoint]:
    """Sweep the change threshold and score the model at each (paper §5).

    For every threshold: select that threshold's change columns + the planning
    predictors and score the model (behind a Yeo-Johnson transform, fit per fold
    to avoid leakage) with stratified cross-validation on weighted F1, balanced
    accuracy and ROC-AUC — the same scorer as the main classifier comparison, so
    the numbers are directly comparable.

    ``df`` must carry one set of change columns **per threshold**, named with the
    ``_<threshold>`` suffix (produced by aggregating the simple-diff change maps
    thresholded at each sweep value). If a threshold matches no change column the
    sweep would silently reduce to planning-only predictors, so this raises
    instead — see :func:`change_columns_for_threshold`.
    """
    base = model if model is not None else default_model()
    sweep = thresholds if thresholds is not None else default_thresholds()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    points: list[AblationPoint] = []

    for raw_threshold in sweep:
        # Truncated integer label, matching threshold_ablation.threshold_label so
        # the generated `_<label>` change columns are the ones selected here.
        threshold = int(raw_threshold)
        if not change_columns_for_threshold(df, threshold):
            raise ValueError(
                f"No change columns ending '_{threshold}' in the table — the "
                "thresholded change maps must be aggregated per threshold first "
                "(see scripts/05_ablation.py / gfs.modeling.features). Refusing to "
                "run the sweep on planning predictors alone."
            )
        predictors = threshold_predictors(df, threshold, planning_predictors=planning_predictors)
        x = cast("pd.DataFrame", df[predictors])
        y = cast("pd.Series", df[target])

        clf = cast("Any", build_pipeline(base))
        metrics = score_estimator(clf, x, y, cv=cv)
        points.append(
            AblationPoint(
                threshold=float(raw_threshold),
                f1=metrics["f1_weighted"],
                balanced_accuracy=metrics["balanced_accuracy"],
                roc_auc=metrics["roc_auc"],
            )
        )
    return points


def ablation_to_frame(points: list[AblationPoint]) -> pd.DataFrame:
    """Tidy the sweep into a DataFrame of ``threshold, f1, accuracy, auc``."""
    return pd.DataFrame(
        {
            "threshold": [p.threshold for p in points],
            "f1": [p.f1 for p in points],
            "balanced_accuracy": [p.balanced_accuracy for p in points],
            "roc_auc": [p.roc_auc for p in points],
        }
    )


def best_threshold(points: list[AblationPoint], *, metric: str = "f1") -> AblationPoint:
    """The sweep point maximizing ``metric`` (``f1``/``balanced_accuracy``/``roc_auc``)."""
    return max(points, key=lambda p: cast("float", getattr(p, metric)))
