"""Shared cross-validated scoring for the modeling stage (paper §4.3).

One place that turns an estimator + (X, y) + a CV splitter into the study's three
metrics, so the main classifier comparison and the threshold ablation report
comparable numbers instead of each rolling its own evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Metrics reported for every model (paper §4.3).
SCORING = ("balanced_accuracy", "f1_weighted", "roc_auc")


def score_estimator(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: Any,
    scoring: tuple[str, ...] = SCORING,
    skip_roc_auc: bool = False,
) -> dict[str, float]:
    """Mean cross-validated score of ``estimator`` on each metric in ``scoring``.

    ``cv`` is any scikit-learn splitter (or fold count). ROC-AUC is skipped
    (returned as ``nan``) when ``skip_roc_auc`` is set, for estimators without a
    probability/decision output.
    """
    from sklearn.model_selection import cross_val_score

    metrics: dict[str, float] = {}
    for metric in scoring:
        if metric == "roc_auc" and skip_roc_auc:
            metrics[metric] = float("nan")
            continue
        scores = cross_val_score(estimator, X, y, scoring=metric, cv=cv, n_jobs=-1)
        metrics[metric] = float(np.mean(scores))
    return metrics
