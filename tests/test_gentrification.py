"""Behavioral contracts for the gentrification score (the target variable Y).

These lock the *meaning* of the score (paper eq 1-2), not the implementation:
whatever we refactor underneath, these properties must hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gfs.gentrification.score import COLS, MEASURES, gentrification_score


def _synthetic_table(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {f"t{tp}_{m}": rng.normal(size=n) for tp in (1, 2) for m in MEASURES}
    return pd.DataFrame({"lsoa_code": [f"E{i:06d}" for i in range(n)], **cols})


def test_score_is_change_in_neighborhood_index() -> None:
    """Contract: gentrification score == NI(t2) - NI(t1) (paper eq 2)."""
    out = gentrification_score(_synthetic_table())
    assert np.allclose(out[COLS.score], out[COLS.index_t2] - out[COLS.index_t1])


def test_score_is_invariant_to_monotonic_rescaling() -> None:
    """Contract: percentile standardization makes the score robust to units.

    A positive monotonic transform of any raw measure preserves its ranks, so
    the score must not change — the property that lets the method travel across
    measures with incomparable units.
    """
    df = _synthetic_table()
    baseline = gentrification_score(df)[COLS.score].to_numpy()

    rescaled = df.copy()
    rescaled["t1_age"] = rescaled["t1_age"] * 1000.0 + 7.0  # monotonic increasing

    after = gentrification_score(rescaled)[COLS.score].to_numpy()
    assert np.allclose(baseline, after)
