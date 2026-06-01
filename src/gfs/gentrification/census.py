"""Loaders for the four socioeconomic measures behind the score (paper §3.1).

Each measure is reduced to a single tidy series ``[lsoa_code, value]`` per time
point, then assembled into the wide table consumed by
:func:`gfs.gentrification.score.gentrification_score`.

| Measure   | Definition                              | Source (t1 / t2)        |
|-----------|-----------------------------------------|-------------------------|
| age       | % residents aged 25–34                  | ONS mid-year (2011/2021)|
| education | % with NVQ Level 4+ qualifications      | ONS census (2011/2021)  |
| housing   | IMD "Barriers to Housing & Services"    | IMD (2010 / 2019)       |
| income    | IMD "Income Deprivation Domain"         | IMD (2010 / 2019)       |

The IMD release lags the census by ~2 years (Table 1), so t1≈2010/2011 and
t2≈2019/2021. Raw file/column names below match the project's data files; pass
overrides if your copies differ.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

# Canonical IMD domain labels (long-format "Indices of Deprivation" column).
IMD_HOUSING_DOMAIN = "g. Barriers to Housing and Services Domain"
IMD_INCOME_DOMAIN = "b. Income Deprivation Domain"

# The 2021 census "Highest level of qualification" category counted as Level 4+.
EDUCATION_L4_CATEGORY_2021 = (
    "Level 4 qualifications or above: degree (BA, BSc), higher degree "
    "(MA, PhD, PGCE), NVQ level 4 to 5, HNC, HND, RSA Higher Diploma, "
    "BTEC Higher level, professional qualifications (for example, teaching, "
    "nursing, accountancy)"
)

LSOA_CODE = "lsoa_code"


# --- LSOA vintage reconciliation --------------------------------------------
def reconcile_lsoa(
    df: pd.DataFrame,
    lookup_path: str,
    *,
    source_code_col: str,
    lookup_from: str,
    lookup_to: str = "lsoa11cd",
) -> pd.DataFrame:
    """Map an older/newer LSOA vintage onto 2011 codes, averaging splits/merges.

    LSOA boundaries changed between the 2001, 2011 and 2021 censuses. We join the
    official ONS lookup and average numeric columns over the target code so every
    measure shares one geography (mirrors the notebook's de-duplication step).
    """
    lookup = pd.read_csv(lookup_path, encoding="latin-1")
    lookup.columns = [c.lower() for c in lookup.columns]
    merged = df.merge(lookup, left_on=source_code_col, right_on=lookup_from, how="left")
    numeric = merged.select_dtypes(include="number").columns.tolist()
    grouped = cast(pd.DataFrame, merged.groupby(lookup_to)[numeric].mean())
    out = grouped.reset_index()
    return out.rename(columns={lookup_to: LSOA_CODE})


# --- Age ---------------------------------------------------------------------
def load_age(
    csv_path: str,
    *,
    code_col: str,
    all_ages_col: str = "All Ages",
    age_cols: tuple[str, ...],
) -> pd.DataFrame:
    """% of residents aged 25–34.

    ``age_cols`` are the single-year/age-band columns summed into the 25–34 band
    (e.g. ``("25-29", "30-34")`` for 2011, or ``("25", ..., "34")`` for 2021).
    """
    df = pd.read_csv(csv_path)
    band_total = df[list(age_cols)].sum(axis=1)
    value = band_total / df[all_ages_col] * 100
    return pd.DataFrame({LSOA_CODE: df[code_col], "value": value})


# --- Education ---------------------------------------------------------------
def load_education_2021(csv_path: str) -> pd.DataFrame:
    """% with Level 4+ qualifications, from the long-format 2021 census table."""
    df = pd.read_csv(csv_path).rename(
        columns={
            "Lower layer Super Output Areas Code": LSOA_CODE,
            "Highest level of qualification (8 categories)": "level_cat",
            "Observation": "value",
        }
    )
    pivot = df.pivot_table(
        values="value", index=LSOA_CODE, columns="level_cat", fill_value=0
    ).reset_index()
    numeric = pivot.select_dtypes(include="number").columns.tolist()
    total = pivot[numeric].sum(axis=1)
    value = pivot[EDUCATION_L4_CATEGORY_2021] / total * 100
    return pd.DataFrame({LSOA_CODE: pivot[LSOA_CODE], "value": value})


def load_education_2011(
    xlsx_path: str,
    lookup_path: str,
    *,
    sheet: str = "2011",
    code_col: str = "LSOA code",
    l4_col: str = "Level 4+",
    residents_col: str = "Usual residents aged 16+",
) -> pd.DataFrame:
    """% with Level 4+ qualifications in 2011, reconciled onto 2011 LSOA codes."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df["l4_perc"] = df[l4_col] / df[residents_col] * 100
    reconciled = reconcile_lsoa(df, lookup_path, source_code_col=code_col, lookup_from="lsoa21cd")
    sub = cast(pd.DataFrame, reconciled[[LSOA_CODE, "l4_perc"]])
    return sub.rename(columns={"l4_perc": "value"})


# --- IMD domains (housing, income) -------------------------------------------
def load_imd_domain(
    csv_path: str,
    domain: str,
    *,
    measurement: str = "Rank",
    code_col: str = "FeatureCode",
) -> pd.DataFrame:
    """Extract one IMD domain from a long-format IMD release (2015/2019 layout).

    Filters to a single ``Measurement`` (Rank or Score), pivots the domains wide,
    and returns the requested ``domain``. Higher rank = less deprived.
    """
    df = pd.read_csv(csv_path)
    if code_col != "FeatureCode" and "FeatureCode" in df.columns:
        df = df.rename(columns={"FeatureCode": code_col})
    subset = df[df["Measurement"] == measurement]
    pivot = subset.pivot_table(
        values="Value", index=code_col, columns="Indices of Deprivation", fill_value=0
    ).reset_index()
    return pd.DataFrame({LSOA_CODE: pivot[code_col], "value": pivot[domain]})


def load_imd2010_column(
    csv_path: str,
    lookup_path: str,
    *,
    column: str,
    code_col: str = "lsoacode",
) -> pd.DataFrame:
    """Extract a wide-format IMD 2010 column (e.g. ``housesb_rank``, ``income_rank``).

    IMD 2010 uses 2001 LSOA codes, so we reconcile onto 2011 codes.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    reconciled = reconcile_lsoa(df, lookup_path, source_code_col=code_col, lookup_from="lsoa01cd")
    sub = cast(pd.DataFrame, reconciled[[LSOA_CODE, column.lower()]])
    return sub.rename(columns={column.lower(): "value"})


# --- Assembly ----------------------------------------------------------------
def assemble(
    *,
    age_t1: pd.DataFrame,
    age_t2: pd.DataFrame,
    edu_t1: pd.DataFrame,
    edu_t2: pd.DataFrame,
    house_t1: pd.DataFrame,
    house_t2: pd.DataFrame,
    income_t1: pd.DataFrame,
    income_t2: pd.DataFrame,
) -> pd.DataFrame:
    """Join the eight tidy measure series into the wide score-input table.

    Each input is ``[lsoa_code, value]``. Output columns:
    ``lsoa_code, t1_age, t2_age, t1_edu, t2_edu, t1_house, t2_house,
    t1_income, t2_income``.
    """
    pieces = {
        "t1_age": age_t1,
        "t2_age": age_t2,
        "t1_edu": edu_t1,
        "t2_edu": edu_t2,
        "t1_house": house_t1,
        "t2_house": house_t2,
        "t1_income": income_t1,
        "t2_income": income_t2,
    }
    out: pd.DataFrame | None = None
    for col, frame in pieces.items():
        renamed = cast(pd.DataFrame, frame.rename(columns={"value": col})[[LSOA_CODE, col]])
        out = (
            renamed
            if out is None
            else cast(pd.DataFrame, out.merge(renamed, on=LSOA_CODE, how="outer"))
        )
    assert out is not None
    return out
