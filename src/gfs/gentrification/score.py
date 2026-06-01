"""The gentrification score — the study's target variable Y (paper §3.1).

Given the four socioeconomic measures (age, education, housing, income) at two
time points t1 (2011) and t2 (2021) per LSOA, the score is built in three steps:

1. **Standardize** each measure to its percentile rank, so the four measures are
   comparable and robust to outliers (raw IMD values are not comparable across
   years — ONS 2015).
2. **Neighborhood Index** = weighted mean of the four standardized measures at
   each time point (equation 1, equal weights by default)::

       NI_t = w_age·age_t + w_edu·edu_t + w_house·house_t + w_income·income_t

3. **Gentrification Score** = change in the index over time (equation 2)::

       G = NI_t2 - NI_t1

A higher score means a greater influx of younger, educated, wealthier residents
or improved housing access. The study focuses on *disadvantaged* neighborhoods,
defined as the bottom 50th percentile of NI_t1.

Input is a "wide" dataframe with one row per LSOA and columns
``lsoa_code, {t1,t2}_{age,edu,house,income}``. See :mod:`gfs.gentrification.census`
for building it from the raw ONS / IMD sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gfs.config import DISADVANTAGED_PERCENTILE

# Column stems for the four measures, in index order.
MEASURES = ("age", "edu", "house", "income")
EQUAL_WEIGHTS = (0.25, 0.25, 0.25, 0.25)


@dataclass(frozen=True)
class ScoreColumns:
    """Names of the columns produced by this module."""

    index_t1: str = "neighborhood_index_t1"
    index_t2: str = "neighborhood_index_t2"
    score: str = "gentrification_score"
    disadvantaged: str = "disadvantaged"


COLS = ScoreColumns()


def standardize_percentiles(df: pd.DataFrame, measure_cols: list[str]) -> pd.DataFrame:
    """Replace each measure column with its percentile rank, scaled to 0-100.

    Uses ``Series.rank(pct=True)`` so values are comparable across measures and
    years (paper §3.1; mirrors the use of percentile-standardized IMD ranks).
    """
    out = df.copy()
    for col in measure_cols:
        out[col] = out[col].rank(pct=True) * 100
    return out


def neighborhood_index(
    df: pd.DataFrame,
    weights: tuple[float, float, float, float] = EQUAL_WEIGHTS,
) -> pd.DataFrame:
    """Add ``neighborhood_index_t1`` / ``_t2`` columns (equation 1).

    Expects standardized columns ``t1_<m>`` and ``t2_<m>`` for each measure in
    :data:`MEASURES`. ``weights`` are (age, education, housing, income).
    """
    out = df.copy()
    for tp, out_col in ((1, COLS.index_t1), (2, COLS.index_t2)):
        out[out_col] = sum(
            w * out[f"t{tp}_{m}"] for w, m in zip(weights, MEASURES, strict=True)
        )
    return out


def flag_disadvantaged(df: pd.DataFrame) -> pd.DataFrame:
    """Flag neighborhoods in the bottom percentile of NI_t1 (paper §3.1).

    Disadvantaged = NI_t1 at or below the 50th percentile (the median).
    """
    out = df.copy()
    threshold = out[COLS.index_t1].quantile(DISADVANTAGED_PERCENTILE / 100)
    out[COLS.disadvantaged] = out[COLS.index_t1] <= threshold
    return out


def gentrification_score(
    df: pd.DataFrame,
    weights: tuple[float, float, float, float] = EQUAL_WEIGHTS,
) -> pd.DataFrame:
    """Full pipeline: standardize -> index -> score -> disadvantaged flag.

    Returns the input dataframe with the neighborhood indices, the
    ``gentrification_score`` (equation 2), and the ``disadvantaged`` flag added.
    """
    measure_cols = [f"t{tp}_{m}" for tp in (1, 2) for m in MEASURES]
    out = standardize_percentiles(df, measure_cols)
    out = neighborhood_index(out, weights)
    out = flag_disadvantaged(out)
    out[COLS.score] = out[COLS.index_t2] - out[COLS.index_t1]
    return out
