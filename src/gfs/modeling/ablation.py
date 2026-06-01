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
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from gfs.config import RANDOM_STATE
from gfs.modeling.dataset import (
    TARGET_COL,
    TransformConfig,
    transform_predictors,
)

# Threshold sweep used in the notebook ablation: 30 points from 100 to 800.
THRESHOLD_MIN = 100.0
THRESHOLD_MAX = 800.0
N_THRESHOLDS = 30


def default_thresholds() -> np.ndarray:
    """The 30 evenly spaced change thresholds swept in the paper's ablation."""
    return np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, N_THRESHOLDS)


def threshold_predictors(
    df: pd.DataFrame,
    threshold: int,
    *,
    planning_predictors: list[str],
) -> list[str]:
    """Change columns for one threshold plus the (threshold-invariant) planning ones.

    The threshold-specific change columns end with the integer threshold (the
    notebook names them ``..._threshold_<t>``); these are combined with the
    planning-layer predictors, which do not depend on the threshold.
    """
    change_predictors = [c for c in df.columns if c.endswith(str(threshold))]
    return change_predictors + planning_predictors


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
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> list[AblationPoint]:
    """Sweep the change threshold and score the model at each (paper §5).

    For every threshold: select that threshold's change columns + the planning
    predictors, Yeo-Johnson transform them, fit ``model`` on a train split and
    score weighted F1, balanced accuracy and ROC-AUC on the held-out split.
    Mirrors the notebook's ``evaluate_model`` ablation loop.
    """
    clf = cast("Any", model if model is not None else default_model())
    sweep = thresholds if thresholds is not None else default_thresholds()
    points: list[AblationPoint] = []

    for raw_threshold in sweep:
        threshold = int(raw_threshold)
        predictors = threshold_predictors(
            df, threshold, planning_predictors=planning_predictors
        )
        data = transform_predictors(
            df, predictors, TransformConfig(pt_transform=True)
        )

        x = cast("pd.DataFrame", data[predictors])
        y = cast("pd.Series", data[target])
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_proba = clf.predict_proba(x_test)[:, 1]

        points.append(
            AblationPoint(
                threshold=float(raw_threshold),
                f1=float(f1_score(y_test, y_pred, average="weighted")),
                balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
                roc_auc=float(roc_auc_score(y_test, y_proba)),
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
