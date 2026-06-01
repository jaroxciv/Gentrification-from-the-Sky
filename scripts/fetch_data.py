"""Fetch the working datasets/outputs needed to run the pipeline.

The full Sentinel-2 imagery (~80 GB) and the raw ONS/IMD inputs come from the
open sources documented in ``data/README.md``. This script pulls the smaller
*derived* artifacts (gentrification scores, per-LSOA change features, planning
layers) that let you reproduce the modeling stage without rebuilding everything.

During development these live in the authors' Dropbox and require a read-only
token in ``.env`` (see the repo README / ``scripts/dropbox_auth.py``). External
users without the token should obtain the inputs from the open sources in
``data/README.md`` (a public Zenodo mirror is planned).

    uv run scripts/fetch_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from gfs.config import DATA_DIR, OUTPUTS_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dbx import data_root, get_client  # noqa: E402

# Derived artifacts to fetch: (path under the Dropbox data root) -> (local path).
MANIFEST: dict[str, Path] = {
    "outputs/gent_merged.gpkg": OUTPUTS_DIR / "gent_merged.gpkg",
    "outputs/lsoa_changes_tinycd_100824.csv": OUTPUTS_DIR / "lsoa_changes_tinycd_100824.csv",
    "data/LSOA_with_features.gpkg": DATA_DIR / "LSOA_with_features.gpkg",
}


def main() -> None:
    try:
        dbx = get_client()
    except SystemExit:
        print(
            "No Dropbox credentials found in .env.\n"
            "External users: obtain the datasets from the open sources listed in "
            "data/README.md (ONS, GOV.UK IMD, Copernicus/Earth Engine).\n"
            "Authors: run `uv run scripts/dropbox_auth.py` to set up the token first."
        )
        return

    root = data_root()
    for remote_rel, local_path in MANIFEST.items():
        remote = f"{root}/{remote_rel}"
        if local_path.exists():
            print(f"skip (exists): {local_path}")
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            dbx.files_download_to_file(str(local_path), remote)
            print(f"fetched: {remote} -> {local_path}")
        except Exception as exc:  # noqa: BLE001 - report and continue per file
            print(f"FAILED: {remote} ({exc})")


if __name__ == "__main__":
    main()
