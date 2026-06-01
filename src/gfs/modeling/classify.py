"""Train and evaluate the gentrification classifiers (paper §4-5).

The paper compares three classifiers on the binarized gentrification target:
Logistic Regression, Linear SVC and XGBoost (the best performer). Each is tuned
with a grid search and scored with repeated stratified cross-validation, on
three metrics: balanced accuracy, weighted F1 and ROC-AUC.

Class imbalance is handled the way the notebook does it: ``class_weight=
"balanced"`` for the linear models and ``scale_pos_weight`` for XGBoost. The
function API takes the already-assembled, already-transformed feature matrix
``X`` and target ``y`` from :mod:`gfs.modeling.dataset`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from gfs.config import RANDOM_STATE
from gfs.modeling.evaluation import score_estimator

# Pipeline step name for the estimator; grid-search params are prefixed with it.
_CLF_STEP = "clf"


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Wrap an estimator behind a Yeo-Johnson power transform.

    Putting the transform in the pipeline means ``GridSearchCV`` /
    ``cross_val_score`` refit it on each training fold only, so the transform
    never sees the held-out fold (avoids the leakage of transforming up front).
    """
    return Pipeline(
        [
            ("transform", PowerTransformer(method="yeo-johnson", standardize=True)),
            (_CLF_STEP, estimator),
        ]
    )


# Grid-search CV folds and repeated-CV design (paper §4.2): 10-fold grid search,
# 30-fold (3 repeats x 10 folds) repeated stratified CV.
GRID_CV_FOLDS = 10
REPEATED_CV_SPLITS = 10
REPEATED_CV_REPEATS = 3


@dataclass(frozen=True)
class ModelSpec:
    """A candidate classifier and its grid-search parameter grid.

    ``needs_proba`` flags estimators (LinearSVC) that have no ``predict_proba``
    and so cannot be scored on ROC-AUC directly; the repeated-CV helper skips
    that metric for them.
    """

    name: str
    estimator: ClassifierMixin
    param_grid: Mapping[str, list[Any]]
    needs_decision_function: bool = False


def _pos_weight(y: pd.Series) -> float:
    """XGBoost ``scale_pos_weight`` = (#negatives / #positives), as in the notebook."""
    counts = y.value_counts()
    n_neg = cast("float", counts.get(0, 1))
    n_pos = cast("float", counts.get(1, 1))
    return float(n_neg) / float(n_pos)


def default_model_specs(y: pd.Series) -> list[ModelSpec]:
    """The three published classifiers with their grids (paper §4).

    ``y`` is needed only to derive XGBoost's ``scale_pos_weight`` from the class
    balance. Linear models use ``class_weight="balanced"``.
    """
    return [
        ModelSpec(
            name="LogisticRegression",
            estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
            param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
        ),
        ModelSpec(
            name="LinearSVC",
            estimator=LinearSVC(class_weight="balanced", max_iter=5000),
            param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
            needs_decision_function=True,
        ),
        ModelSpec(
            name="XGBoost",
            estimator=XGBClassifier(
                n_jobs=-1,
                scale_pos_weight=_pos_weight(y),
                eval_metric="logloss",
            ),
            param_grid={
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 10],
                "learning_rate": [0.01, 0.1, 0.2],
            },
        ),
    ]


@dataclass
class TunedModel:
    """A grid-search-tuned estimator plus its best hyper-parameters."""

    name: str
    estimator: ClassifierMixin
    best_params: dict[str, Any]
    cv_best_score: float


def grid_search(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    scoring: str = "f1_weighted",
    cv_folds: int = GRID_CV_FOLDS,
) -> TunedModel:
    """Tune one model's hyper-parameters by grid search (paper §4.2).

    Returns the refit best estimator and its chosen parameters. Uses a
    stratified ``cv_folds``-fold split and the given ``scoring`` metric.
    """
    pipeline = build_pipeline(spec.estimator)
    # Grid params target the estimator step inside the pipeline.
    param_grid = {f"{_CLF_STEP}__{k}": v for k, v in spec.param_grid.items()}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    grid.fit(X, y)
    return TunedModel(
        name=spec.name,
        estimator=cast("ClassifierMixin", grid.best_estimator_),
        best_params=cast("dict[str, Any]", grid.best_params_),
        cv_best_score=float(grid.best_score_),
    )


def repeated_cv_metrics(
    spec: ModelSpec,
    estimator: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = REPEATED_CV_SPLITS,
    n_repeats: int = REPEATED_CV_REPEATS,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Mean repeated-stratified-CV score of an estimator on each metric (paper §4.3).

    Runs ``n_repeats x n_splits`` repeated stratified k-fold CV and averages each
    of :data:`SCORING`. ROC-AUC is skipped for estimators without a probability
    output (e.g. plain LinearSVC).
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return score_estimator(estimator, X, y, cv=cv, skip_roc_auc=spec.needs_decision_function)


@dataclass
class ModelResult:
    """Tuned model plus its averaged repeated-CV metrics."""

    name: str
    best_params: dict[str, Any]
    metrics: dict[str, float]


@dataclass(frozen=True)
class ClassifyConfig:
    """Cross-validation design for :func:`train_and_evaluate`."""

    grid_cv_folds: int = GRID_CV_FOLDS
    repeated_cv_splits: int = REPEATED_CV_SPLITS
    repeated_cv_repeats: int = REPEATED_CV_REPEATS
    grid_scoring: str = "f1_weighted"
    random_state: int = RANDOM_STATE
    specs: list[ModelSpec] | None = field(default=None)


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    config: ClassifyConfig | None = None,
) -> list[ModelResult]:
    """Tune and evaluate every classifier; return per-model CV metrics (paper §4-5).

    For each :class:`ModelSpec` (defaulting to the paper's three classifiers):
    grid-search the hyper-parameters, then score the refit estimator with
    repeated stratified CV on balanced accuracy, weighted F1 and ROC-AUC.
    """
    cfg = config or ClassifyConfig()
    specs = cfg.specs if cfg.specs is not None else default_model_specs(y)
    results: list[ModelResult] = []
    for spec in specs:
        tuned = grid_search(spec, X, y, scoring=cfg.grid_scoring, cv_folds=cfg.grid_cv_folds)
        metrics = repeated_cv_metrics(
            spec,
            tuned.estimator,
            X,
            y,
            n_splits=cfg.repeated_cv_splits,
            n_repeats=cfg.repeated_cv_repeats,
            random_state=cfg.random_state,
        )
        results.append(ModelResult(name=spec.name, best_params=tuned.best_params, metrics=metrics))
    return results


def results_to_frame(results: list[ModelResult]) -> pd.DataFrame:
    """Tidy the per-model metrics into a DataFrame for printing/saving."""
    rows = [{"model": r.name, **r.metrics} for r in results]
    return pd.DataFrame(rows)
