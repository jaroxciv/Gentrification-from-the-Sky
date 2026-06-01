"""Stage 3 — build the gentrification score (Y), the modeling target.

Loads the four socioeconomic measures for both time points, assembles the wide
table, computes the neighborhood index and gentrification score (paper §3.1),
and writes the result to ``outputs/gentrification_score.csv``.

    uv run scripts/03_build_gentrification.py

File locations are declared in SOURCES below; adjust them to match your data/
layout (see data/README.md). Requires the census + IMD datasets to be present.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from gfs.config import CENSUS_DIR, OUTPUTS_DIR
from gfs.gentrification import census
from gfs.gentrification.score import gentrification_score


@dataclass(frozen=True)
class Sources:
    """Paths to the raw measure files (relative to data/census)."""

    age_2011: Path = CENSUS_DIR / "age-2011-lsoa.csv"
    age_2021: Path = CENSUS_DIR / "age-2021-lsoa.csv"
    education_2021: Path = CENSUS_DIR / "education-2021-lsoa.csv"
    education_2011_xlsx: Path = CENSUS_DIR / "qualifications-2011-2021-lsoa.xlsx"
    imd_2010: Path = CENSUS_DIR / "imd2010eng.csv"
    imd_2019: Path = CENSUS_DIR / "imd2019lsoa.csv"
    lsoa01_to_11: Path = CENSUS_DIR / "lsoa01-to-lsoa11-lookup.csv"
    lsoa11_to_21: Path = CENSUS_DIR / "lsoa11-to-lsoa21-lookup.csv"


def build(src: Sources) -> None:
    s = str  # paths -> str for the loaders
    table = census.assemble(
        # Age: % aged 25–34.
        age_t1=census.load_age(
            s(src.age_2011), code_col="Area Codes", age_cols=("25-29", "30-34")
        ),
        age_t2=census.load_age(
            s(src.age_2021),
            code_col="LSOA Code",
            age_cols=tuple(str(a) for a in range(25, 35)),
        ),
        # Education: % NVQ Level 4+.
        edu_t1=census.load_education_2011(s(src.education_2011_xlsx), s(src.lsoa11_to_21)),
        edu_t2=census.load_education_2021(s(src.education_2021)),
        # Housing: IMD Barriers to Housing & Services.
        house_t1=census.load_imd2010_column(
            s(src.imd_2010), s(src.lsoa01_to_11), column="housesb_rank"
        ),
        house_t2=census.load_imd_domain(
            s(src.imd_2019), census.IMD_HOUSING_DOMAIN, code_col="lsoa_codes"
        ),
        # Income: IMD Income Deprivation Domain.
        income_t1=census.load_imd2010_column(
            s(src.imd_2010), s(src.lsoa01_to_11), column="income_rank"
        ),
        income_t2=census.load_imd_domain(
            s(src.imd_2019), census.IMD_INCOME_DOMAIN, code_col="lsoa_codes"
        ),
    )

    scored = gentrification_score(table)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "gentrification_score.csv"
    scored.to_csv(out_path, index=False)
    n_disadv = int(cast("int", scored["disadvantaged"].sum()))
    print(f"Wrote {out_path}  ({len(scored)} LSOAs, {n_disadv} disadvantaged)")


if __name__ == "__main__":
    build(Sources())
