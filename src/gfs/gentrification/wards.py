"""Aggregate the LSOA-level gentrification score up to electoral wards.

The paper reports results at LSOA level but also visualizes wards (a coarser
geography) for readability. Aggregation is an area-unweighted mean of the LSOA
scores within each ward, via the ONS LSOA→ward lookup.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from gfs.gentrification.score import COLS


def aggregate_to_wards(
    lsoa_scores: pd.DataFrame,
    lookup_path: str,
    *,
    lsoa_col: str = "lsoa_code",
    ward_col: str = "wd_code",
    value_cols: tuple[str, ...] = (COLS.index_t1, COLS.index_t2, COLS.score),
) -> pd.DataFrame:
    """Mean the given score columns within each ward.

    ``lookup_path`` is a CSV with at least an LSOA code column and a ward code
    column (lower-cased on load). Returns one row per ward.
    """
    lookup = pd.read_csv(lookup_path)
    lookup.columns = [c.lower() for c in lookup.columns]
    merged = lsoa_scores.merge(
        lookup[[lsoa_col, ward_col]], on=lsoa_col, how="left"
    )
    grouped = cast(pd.DataFrame, merged.groupby(ward_col)[list(value_cols)].mean())
    return grouped.reset_index()
