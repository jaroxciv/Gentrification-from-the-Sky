"""Assemble the modeling table and the classification target (paper §3.5, §4.1).

Joins the three ingredients of the predictor set with the gentrification score:

* **Y**: the gentrification score per LSOA (from :mod:`gfs.gentrification`).
* **Phi**: the satellite change features, one column per band
  (:mod:`gfs.modeling.features`).
* **baseline socio predictors**: planning-layer coverage and, optionally, the
  housing / population-churn / income / gini measures used by the paper's
  baseline model (paper §5).

The target is binarized by the top/bottom quartiles of the score: ``1`` if the
score is above Q75, ``0`` if below Q25, and the middle 50 % is dropped
(paper §4.1). Skewed predictors are passed through a Yeo-Johnson power transform
so the linear models see roughly Gaussian features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from gfs.config import GENT_BOTTOM_PERCENTILE, GENT_TOP_PERCENTILE

LSOA_CODE_COL = "LSOA11CD"
SCORE_COL = "gentrification_score"
TARGET_COL = "gentrification_binary"
DISADVANTAGED_COL = "disadvantaged"

# Satellite feature columns start with this stem (one per band).
SATELLITE_PREFIX = "features"

# Boundary / admin / target columns that are never predictors. Used to select
# planning-layer predictors explicitly rather than guessing from name casing.
ADMIN_COLUMNS: frozenset[str] = frozenset(
    {
        LSOA_CODE_COL,
        "lsoa_codes",
        "LSOA11NM",
        "MSOA11CD",
        "MSOA11NM",
        "LAD11CD",
        "LAD11NM",
        "RGN11CD",
        "RGN11NM",
        "USUALRES",
        "HHOLDRES",
        "COMESTRES",
        "POPDEN",
        "HHOLDS",
        "AVHHOLDSZ",
        "geometry",
        SCORE_COL,
        TARGET_COL,
        DISADVANTAGED_COL,
    }
)


@dataclass(frozen=True)
class TargetConfig:
    """How to binarize the gentrification score into the classification label.

    Defaults reproduce the paper: keep the top quartile (``1``) and the bottom
    quartile (``0``) of the score, dropping the middle half. ``top``/``bottom``
    are percentiles in [0, 100].
    """

    top: float = GENT_TOP_PERCENTILE
    bottom: float = GENT_BOTTOM_PERCENTILE
    drop_middle: bool = True


@dataclass(frozen=True)
class TransformConfig:
    """Which monotone transforms to apply to predictors before modeling.

    The published pipeline standardizes predictors with a Yeo-Johnson power
    transform (``pt_transform``). The log1p / cube-root / inverse-sine options
    are kept for faithfulness with the notebook's exploration but default off.
    """

    log_transform: bool = False
    cube_root: bool = False
    inverse_sine: bool = False
    pt_transform: bool = True


def binarize_score(
    df: pd.DataFrame,
    config: TargetConfig | None = None,
    *,
    score_col: str = SCORE_COL,
) -> pd.DataFrame:
    """Add the binary gentrification target by top/bottom quartile (paper §4.1).

    ``1`` where the score exceeds its ``top`` percentile, ``0`` where it is below
    its ``bottom`` percentile; the middle band is ``NaN`` and dropped when
    ``config.drop_middle`` is set. Mirrors the notebook's ``create_label`` with
    ``use_percentile=True``.
    """
    cfg = config or TargetConfig()
    out = df.copy()
    top_threshold = out[score_col].quantile(cfg.top / 100)
    bottom_threshold = out[score_col].quantile(cfg.bottom / 100)
    out[TARGET_COL] = np.where(
        out[score_col] > top_threshold,
        1.0,
        np.where(out[score_col] < bottom_threshold, 0.0, np.nan),
    )
    if cfg.drop_middle:
        out = cast("pd.DataFrame", out.dropna(subset=[TARGET_COL]))
    return out


def transform_predictors(
    df: pd.DataFrame,
    predictors: list[str],
    config: TransformConfig | None = None,
) -> pd.DataFrame:
    """Apply the configured monotone / power transforms to ``predictors`` in place.

    Each transform is applied column-by-column (the Yeo-Johnson transformer is
    fit per column with ``standardize=True``), reproducing the notebook's
    ``transform_predictors`` exactly.
    """
    cfg = config or TransformConfig()
    out = df.copy()
    if cfg.log_transform:
        for predictor in predictors:
            out[predictor] = np.log1p(out[predictor])
    if cfg.cube_root:
        for predictor in predictors:
            out[predictor] = np.cbrt(out[predictor])
    if cfg.inverse_sine:
        for predictor in predictors:
            out[predictor] = np.arcsinh(out[predictor])
    if cfg.pt_transform:
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        for predictor in predictors:
            out[predictor] = pt.fit_transform(out[[predictor]])
    return out


def satellite_predictors(df: pd.DataFrame, *, prefix: str = SATELLITE_PREFIX) -> list[str]:
    """Column names of the satellite change features (Phi), one per band."""
    return [c for c in df.columns if c.startswith(prefix)]


def planning_predictors(
    planning: pd.DataFrame, *, exclude: frozenset[str] = ADMIN_COLUMNS
) -> list[str]:
    """Planning-layer predictor columns.

    Every column that is not a boundary/admin/target field (``exclude``) and not
    a satellite feature is treated as a planning-layer predictor. This is
    explicit rather than inferring membership from name casing (the notebook used
    ``not col.isupper()``, which silently swept in any lower-case metadata
    column such as ``lsoa_codes``).
    """
    return [
        c
        for c in planning.columns
        if c not in exclude and not c.startswith(SATELLITE_PREFIX)
    ]


@dataclass
class ModelingTable:
    """The assembled modeling table plus its predictor/target column names."""

    data: pd.DataFrame
    predictors: list[str]
    target: str = TARGET_COL


def ensure_lsoa_key(df: pd.DataFrame, *, source_col: str = "lsoa_code") -> pd.DataFrame:
    """Guarantee the canonical join key :data:`LSOA_CODE_COL` is present.

    The gentrification-score table (``scripts/03``) keys on ``lsoa_code`` while
    the satellite/planning features key on ``LSOA11CD``. If the canonical column
    is missing but ``source_col`` is present, rename it; otherwise return as-is.
    """
    if LSOA_CODE_COL not in df.columns and source_col in df.columns:
        return df.rename(columns={source_col: LSOA_CODE_COL})
    return df


def assemble_modeling_table(
    score_gdf: pd.DataFrame,
    lsoa_changes: pd.DataFrame,
    planning: pd.DataFrame,
    *,
    target_config: TargetConfig | None = None,
    score_code_col: str = "lsoa_code",
    extra_baseline: pd.DataFrame | None = None,
    extra_baseline_predictors: list[str] | None = None,
) -> ModelingTable:
    """Join score + Phi + baseline predictors into a model-ready table.

    Steps (paper §3.5):

    1. Normalize the score table's LSOA key to :data:`LSOA_CODE_COL`, then merge
       the gentrification score, the satellite change features (``lsoa_changes``)
       and the planning-layer features (``planning``) on it; optionally merge
       ``extra_baseline`` socio predictors too.
    2. Binarize the score into the classification target (top/bottom quartile).

    Predictors are returned **untransformed**: the Yeo-Johnson power transform is
    applied inside the cross-validation pipeline (:mod:`gfs.modeling.classify`)
    so it is refit per fold and does not leak across the train/test split.

    Returns a :class:`ModelingTable` with the merged DataFrame and the predictor /
    target column names.
    """
    score_gdf = ensure_lsoa_key(score_gdf, source_col=score_code_col)
    merged = cast(
        "pd.DataFrame",
        score_gdf.merge(lsoa_changes, on=LSOA_CODE_COL).merge(planning, on=LSOA_CODE_COL),
    )
    if extra_baseline is not None:
        merged = cast("pd.DataFrame", merged.merge(extra_baseline, on=LSOA_CODE_COL))

    merged = binarize_score(merged, target_config)

    predictors = satellite_predictors(merged) + planning_predictors(planning)
    if extra_baseline_predictors:
        predictors = predictors + extra_baseline_predictors

    return ModelingTable(data=merged, predictors=predictors, target=TARGET_COL)


@dataclass(frozen=True)
class BaselineConfig:
    """Column names for the optional socio baseline predictors (paper §5)."""

    housing_cols: tuple[str, ...] = field(
        default=(
            "house_price_median_2011",
            "house_price_median_2016",
            "house_price_median_2021",
        )
    )
    churn_cols: tuple[str, ...] = field(default=("chn2011", "chn2020", "churn_change"))
    income_cols: tuple[str, ...] = ()
    gini_cols: tuple[str, ...] = ()

    def all_cols(self) -> list[str]:
        """Flatten every configured baseline predictor column into one list."""
        return [
            *self.housing_cols,
            *self.churn_cols,
            *self.income_cols,
            *self.gini_cols,
        ]
