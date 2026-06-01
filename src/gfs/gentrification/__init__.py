"""Y — Gentrification score (the target variable).

Builds the census-based ground truth from ONS / IMD data at LSOA level
(paper §3.1): loads the four socioeconomic measures (age, education, housing,
income) for the two time points, computes the neighborhood index, and derives
the gentrification score as its change over time.
"""

from gfs.gentrification.census import assemble
from gfs.gentrification.score import (
    ScoreColumns,
    gentrification_score,
    neighborhood_index,
)
from gfs.gentrification.wards import aggregate_to_wards

__all__ = [
    "gentrification_score",
    "neighborhood_index",
    "ScoreColumns",
    "assemble",
    "aggregate_to_wards",
]
