"""Project-wide configuration: filesystem layout, credentials, and the study design.

The **study design** (geography, time points, bands, CRS, score parameters) is
captured in a single :class:`StudyConfig` so the pipeline is not hard-wired to
Greater London. :data:`LONDON` reproduces the paper; :data:`STUDY` is the active
study (``LONDON`` unless a TOML is given via ``GFS_STUDY_CONFIG``). The familiar
module-level constants (``YEAR_T1``, ``SENTINEL2_BANDS``, ``WORKING_CRS`` …) are
derived from ``STUDY``, so existing imports keep working and swapping ``STUDY``
re-points the whole pipeline at another city.
"""

from __future__ import annotations

import calendar
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# --- Filesystem layout -------------------------------------------------------
# Repo root = two levels up from this file (src/gfs/config.py -> repo root).
ROOT = Path(__file__).resolve().parents[2]

# Load .env so credentials/settings are picked up automatically (process
# environment still takes precedence; missing file is a no-op).
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.environ.get("GFS_DATA_DIR", ROOT / "data"))
OUTPUTS_DIR = Path(os.environ.get("GFS_OUTPUTS_DIR", ROOT / "outputs"))
FIGURES_DIR = Path(os.environ.get("GFS_FIGURES_DIR", ROOT / "figures"))

# Per-stage data subfolders (see data/README.md).
COMPOSITES_DIR = DATA_DIR / "composites"
CENSUS_DIR = DATA_DIR / "census"
BOUNDARIES_DIR = DATA_DIR / "boundaries"
PLANNING_DIR = DATA_DIR / "planning_layers"

# Change-detection features land here, one subfolder per model
# (e.g. outputs/features_tinycd/), matching the original project layout.
FEATURES_DIR = OUTPUTS_DIR

# --- External data-source credentials (optional; for regenerating raw data) --
# These power the two upstream stages whose outputs already ship as data:
#   * WASDI    -> the raw Sentinel-2 composites (licensed platform; see below)
#   * Earth Engine -> the Dynamic World land-cover layer (green areas)
# All read from the environment / .env so runs stay headless and credential
# files never enter git.
GEE_PROJECT = os.environ.get("GFS_GEE_PROJECT", "lse23-24")
GEE_SERVICE_ACCOUNT = os.environ.get("GFS_GEE_SERVICE_ACCOUNT") or None
GEE_SERVICE_ACCOUNT_KEY = os.environ.get("GFS_GEE_SERVICE_ACCOUNT_KEY") or None

# WASDI: a JSON config with USER / PASSWORD / WORKSPACE (wasdi.init reads it).
# NOTE: WASDI is a licensed platform (arrange usage rights with the WASDI team);
# the compositing processor runs only on explicit opt-in (gfs.composites.wasdi_source).
WASDI_CONFIG = Path(os.environ.get("GFS_WASDI_CONFIG", ROOT / "wasdi_config.json"))


# --- Study design (paper §3) -------------------------------------------------
@dataclass(frozen=True)
class StudyConfig:
    """Everything that is specific to a study area / run.

    Swap this (or load one from TOML) to apply the method to another city: a
    different boundary, geography, year pair, projection or score definition.
    """

    name: str
    # Geography: boundary shapefile name (under BOUNDARIES_DIR) + its code column.
    boundary_filename: str
    geography: str  # short label, e.g. "LSOA"
    geography_code_col: str  # boundary attribute holding the area code, e.g. "LSOA11CD"
    # Time points: satellite composites and census measures.
    year_t1: int
    year_t2: int
    census_t1: int
    census_t2: int
    # Sentinel-2 composite window + bands.
    composite_month_start: int
    composite_month_end: int
    sentinel2_bands: tuple[str, ...]
    target_resolution_m: int
    patch_size: int  # imagelet size for the DL feature extractors
    # Coordinate reference systems.
    working_crs: str  # rasters / composites / change maps (projected, metres)
    boundary_crs: str  # vectors / boundary / score / export (projected, metres)
    geographic_crs: str  # lon/lat used for Earth Engine & STAC search
    # Gentrification score + modeling.
    score_measures: tuple[str, ...]
    disadvantaged_percentile: int
    gent_top_percentile: int
    gent_bottom_percentile: int
    # Whether to model only the disadvantaged subset. The paper does; another
    # study may model all neighborhoods — this is a design choice, not a default.
    model_disadvantaged_only: bool
    random_state: int

    @classmethod
    def from_toml(cls, path: str | os.PathLike[str]) -> StudyConfig:
        """Load a study from a TOML file, overriding only the keys it specifies.

        Keys may sit at the top level or under a ``[study]`` table; anything not
        given falls back to :data:`LONDON`.
        """
        import tomllib

        with open(path, "rb") as handle:
            data: dict[str, Any] = tomllib.load(handle)
        overrides = data.get("study", data)
        for seq_key in ("sentinel2_bands", "score_measures"):
            if seq_key in overrides:
                overrides[seq_key] = tuple(overrides[seq_key])
        return replace(LONDON, **overrides)


# The paper's study: Greater London, Sentinel-2 2016/2021, census 2011/2021.
LONDON = StudyConfig(
    name="Greater London",
    boundary_filename="LSOA_2011_London_gen_MHW.shp",
    geography="LSOA",
    geography_code_col="LSOA11CD",
    year_t1=2016,  # Sentinel-2 starts mid-2015; 2016 is the first full composite.
    year_t2=2021,
    census_t1=2011,
    census_t2=2021,
    composite_month_start=6,  # June 1st  — summer window, fewest clouds
    composite_month_end=8,  # August 31st
    sentinel2_bands=("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"),
    target_resolution_m=10,
    patch_size=256,
    working_crs="EPSG:32630",  # UTM zone 30N (composites / change maps)
    boundary_crs="EPSG:27700",  # British National Grid (LSOA boundary / score)
    geographic_crs="EPSG:4326",  # WGS84 lon/lat (EE / STAC search)
    score_measures=("age", "edu", "house", "income"),  # age, education, housing, income
    disadvantaged_percentile=50,  # focus on bottom-half neighborhoods at t1
    gent_top_percentile=75,  # binarize score: 1 above Q75, 0 below Q25
    gent_bottom_percentile=25,
    model_disadvantaged_only=True,  # the paper models the disadvantaged subset only
    random_state=42,
)

# Active study: a TOML via GFS_STUDY_CONFIG, else London.
_study_toml = os.environ.get("GFS_STUDY_CONFIG")
STUDY: StudyConfig = StudyConfig.from_toml(_study_toml) if _study_toml else LONDON

# --- Derived module-level constants (sourced from STUDY) ---------------------
YEAR_T1 = STUDY.year_t1
YEAR_T2 = STUDY.year_t2
CENSUS_T1 = STUDY.census_t1
CENSUS_T2 = STUDY.census_t2
COMPOSITE_MONTH_START = STUDY.composite_month_start
COMPOSITE_MONTH_END = STUDY.composite_month_end
SENTINEL2_BANDS = STUDY.sentinel2_bands
TARGET_RESOLUTION_M = STUDY.target_resolution_m
PATCH_SIZE = STUDY.patch_size
GEOGRAPHY = STUDY.geography
GEOGRAPHY_CODE_COL = STUDY.geography_code_col
DEFAULT_BOUNDARY = BOUNDARIES_DIR / STUDY.boundary_filename

# Coordinate reference systems (centralized; were scattered as literals).
WORKING_CRS = STUDY.working_crs
BOUNDARY_CRS = STUDY.boundary_crs
GEOGRAPHIC_CRS = STUDY.geographic_crs


def composite_window(year: int) -> tuple[str, str]:
    """ISO ``(start, end)`` dates for the study's summer composite window in ``year``.

    The end day is the actual last day of ``COMPOSITE_MONTH_END`` (via
    ``calendar.monthrange``), so non-31-day months don't produce invalid dates.
    """
    last_day = calendar.monthrange(year, COMPOSITE_MONTH_END)[1]
    return (
        f"{year}-{COMPOSITE_MONTH_START:02d}-01",
        f"{year}-{COMPOSITE_MONTH_END:02d}-{last_day:02d}",
    )


# Gentrification score + modeling.
SCORE_MEASURES = STUDY.score_measures
DISADVANTAGED_PERCENTILE = STUDY.disadvantaged_percentile
GENT_TOP_PERCENTILE = STUDY.gent_top_percentile
GENT_BOTTOM_PERCENTILE = STUDY.gent_bottom_percentile
MODEL_DISADVANTAGED_ONLY = STUDY.model_disadvantaged_only
RANDOM_STATE = STUDY.random_state

# --- Change-detection training (paper §3.4; method config, not study-specific) ---
CD_LEARNING_RATE = 1e-3
CD_BATCH_SIZE = 8
CD_EPOCHS = 1  # one-shot Siamese learning...
CD_EPOCHS_RESNET = 100  # ...except the customized Res-Net.

# Canonical change-detection methods compared in the paper (§3.3).
CD_METHODS = (
    "simple_diff",
    "resnet",
    "fc_siamdiff",
    "cgnet",
    "bidatenet",
    "tinycd",
)
