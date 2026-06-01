"""Project-wide paths and constants.

Centralizes the magic numbers from the paper (bands, years, geographies, model
hyperparameters) and the on-disk layout so the rest of the package never
hard-codes a path or a year.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Filesystem layout -------------------------------------------------------
# Repo root = two levels up from this file (src/gfs/config.py -> repo root).
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("GFS_DATA_DIR", ROOT / "data"))
OUTPUTS_DIR = Path(os.environ.get("GFS_OUTPUTS_DIR", ROOT / "outputs"))
FIGURES_DIR = Path(os.environ.get("GFS_FIGURES_DIR", ROOT / "figures"))

# Per-stage data subfolders (see data/README.md).
COMPOSITES_DIR = DATA_DIR / "composites"
CENSUS_DIR = DATA_DIR / "census"
BOUNDARIES_DIR = DATA_DIR / "boundaries"
PLANNING_DIR = DATA_DIR / "planning_layers"

# --- External data-source credentials (optional; for regenerating raw data) --
# These power the two upstream stages whose outputs already ship as data:
#   * WASDI    -> the raw Sentinel-2 composites (paid platform; see below)
#   * Earth Engine -> the Dynamic World land-cover layer (green areas)
# All are read from the environment / .env so runs stay headless and credential
# files never enter git.
#
# Earth Engine: a project is always required; a service account makes auth fully
# headless (preferred on HPC/CI), otherwise persistent user credentials are used.
GEE_PROJECT = os.environ.get("GFS_GEE_PROJECT", "lse23-24")
GEE_SERVICE_ACCOUNT = os.environ.get("GFS_GEE_SERVICE_ACCOUNT") or None
GEE_SERVICE_ACCOUNT_KEY = os.environ.get("GFS_GEE_SERVICE_ACCOUNT_KEY") or None

# WASDI: a JSON config with USER / PASSWORD / WORKSPACE (wasdi.init reads it).
# NOTE: WASDI is a licensed platform (arrange usage rights with the WASDI team);
# the compositing processor runs only on explicit opt-in (gfs.composites.wasdi_source).
WASDI_CONFIG = Path(os.environ.get("GFS_WASDI_CONFIG", ROOT / "wasdi_config.json"))

# Change-detection features land here, one subfolder per model
# (e.g. outputs/features_tinycd/), matching the original project layout.
FEATURES_DIR = OUTPUTS_DIR

# --- Study design (paper §3) -------------------------------------------------
# Two time points the whole study compares.
YEAR_T1 = 2016  # Sentinel-2 starts mid-2015; 2016 is the first complete composite.
YEAR_T2 = 2021

# Census time points for the gentrification score (paper §3.1, Table 1).
CENSUS_T1 = 2011
CENSUS_T2 = 2021

# Sentinel-2 summer window with fewest clouds over London (paper §3.2).
COMPOSITE_MONTH_START = 6  # June 1st
COMPOSITE_MONTH_END = 8  # August 31st

# All 11 Sentinel-2 bands used, resampled to a uniform 10 m resolution.
SENTINEL2_BANDS = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
)
TARGET_RESOLUTION_M = 10

# Imagelet size fed to the deep-learning feature extractors (paper §3.3).
PATCH_SIZE = 256

# Neighborhood geography.
GEOGRAPHY = "LSOA"  # Lower Layer Super Output Area

# --- Gentrification score (paper §3.1) ---------------------------------------
# The four socioeconomic measures averaged into the neighborhood index.
SCORE_MEASURES = ("age", "education", "housing", "income")
# Study focuses on disadvantaged neighborhoods: bottom 50th percentile in t1.
DISADVANTAGED_PERCENTILE = 50

# --- Modeling (paper §4) -----------------------------------------------------
# Binarize the gentrification score by its top/bottom quartiles; drop the middle.
GENT_TOP_PERCENTILE = 75
GENT_BOTTOM_PERCENTILE = 25

RANDOM_STATE = 42

# --- Change-detection training (paper §3.4) ----------------------------------
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
