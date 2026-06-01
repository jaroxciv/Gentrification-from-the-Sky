"""Behavioral contracts for the modeling table + target (paper §3.5, §4.1).

Locks the binarization rule and the table-assembly guarantees (including the
cross-vintage LSOA join), independent of how they're implemented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gfs.modeling.dataset import (
    SCORE_COL,
    TARGET_COL,
    TargetConfig,
    assemble_modeling_table,
    binarize_score,
)


def test_binarize_keeps_quartile_extremes_and_drops_the_middle() -> None:
    """Contract: 1 above Q75, 0 below Q25, middle 50% dropped (paper §4.1)."""
    df = pd.DataFrame({SCORE_COL: np.arange(100.0)})
    out = binarize_score(df)

    assert set(out[TARGET_COL].unique()) == {0.0, 1.0}
    # Symmetric quartiles -> equal class sizes, ~half the rows dropped.
    assert (out[TARGET_COL] == 1).sum() == (out[TARGET_COL] == 0).sum()
    assert len(out) < len(df)
    # The label is monotonic in the score: the extremes land in the right class.
    assert out.loc[out[SCORE_COL] == 99.0, TARGET_COL].iloc[0] == 1.0
    assert out.loc[out[SCORE_COL] == 0.0, TARGET_COL].iloc[0] == 0.0


def test_assemble_joins_across_lsoa_key_and_excludes_admin_columns() -> None:
    """Contract: score (lsoa_code) joins to features (LSOA11CD); predictors are
    the satellite + planning columns only, never admin/identifier columns."""
    n = 40
    codes = [f"E{i:06d}" for i in range(n)]
    rng = np.random.default_rng(0)

    # The score table carries the disadvantaged flag; the modeling population is
    # disadvantaged-only, so include it (all True here to keep every row).
    score = pd.DataFrame({"lsoa_code": codes, SCORE_COL: rng.normal(size=n), "disadvantaged": True})
    features = pd.DataFrame(
        {"LSOA11CD": codes, **{f"features_band_{b}_tinycd": rng.random(n) for b in (1, 2)}}
    )
    planning = pd.DataFrame(
        {
            "LSOA11CD": codes,
            "lsoa_codes": codes,  # admin column that must NOT become a predictor
            "Town_Centre_Boundaries": rng.random(n),
        }
    )

    table = assemble_modeling_table(score, features, planning)

    assert len(table.data) > 0  # the cross-key join produced rows
    assert "lsoa_codes" not in table.predictors
    assert "features_band_1_tinycd" in table.predictors
    assert "Town_Centre_Boundaries" in table.predictors
    assert table.target in table.data.columns


def test_modeling_population_defaults_to_disadvantaged_only() -> None:
    """Contract: by default the table is restricted to disadvantaged neighborhoods
    (the study's population), and that is a config choice, not hardcoded."""
    n = 40
    codes = [f"E{i:06d}" for i in range(n)]
    rng = np.random.default_rng(1)
    disadvantaged = np.array([True] * 20 + [False] * 20)

    score = pd.DataFrame(
        {"lsoa_code": codes, SCORE_COL: rng.normal(size=n), "disadvantaged": disadvantaged}
    )
    features = pd.DataFrame({"LSOA11CD": codes, "features_band_1_tinycd": rng.random(n)})
    planning = pd.DataFrame({"LSOA11CD": codes})

    default = assemble_modeling_table(score, features, planning)
    all_pop = assemble_modeling_table(
        score, features, planning, target_config=TargetConfig(disadvantaged_only=False)
    )

    # Default keeps only (a quartile slice of) the 20 disadvantaged rows.
    assert len(default.data) <= 20
    assert len(all_pop.data) > len(default.data)
