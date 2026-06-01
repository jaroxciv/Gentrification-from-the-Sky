"""Stage 1 — build the Sentinel-2 median composites (X), the satellite inputs.

End-to-end driver for the composite stage (paper §3.2):

1. Load the Greater-London LSOA boundary and derive the Earth Engine region.
2. For YEAR_T1 (2016) and YEAR_T2 (2021), build a cloud-masked median composite
   of the 11 study bands at 10 m and download it as a multi-band GeoTIFF.
3. (Optional) clip the downloaded tiles to the boundary and merge them into one
   aligned per-year composite, then print a per-band quality table.

    uv run scripts/01_build_composites.py

Requires Earth Engine access (``initialize_ee`` authenticates on first use) and
the ~80 GB of imagery to actually produce output; it imports and type-checks
without them. File locations come from SOURCES below and :mod:`gfs.config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd

from gfs.composites import clip_merge, processor, stats
from gfs.composites.processor import CompositeSpec
from gfs.config import BOUNDARIES_DIR, COMPOSITES_DIR, YEAR_T1, YEAR_T2


@dataclass(frozen=True)
class Sources:
    """Paths to the inputs/outputs for the composite stage."""

    london_boundary: Path = BOUNDARIES_DIR / "LSOA_2011_London_gen_MHW.shp"
    composites_dir: Path = COMPOSITES_DIR
    ee_project: str | None = None


def build(src: Sources) -> None:
    processor.initialize_ee(project=src.ee_project)

    boundary = gpd.read_file(str(src.london_boundary))
    region = processor.london_geometry(boundary)

    src.composites_dir.mkdir(parents=True, exist_ok=True)
    merged_paths: dict[int, str] = {}

    for year in (YEAR_T1, YEAR_T2):
        spec = CompositeSpec(year=year)
        composite = processor.build_median_composite(spec, region)

        # Download the raw composite GeoTIFF for this year.
        raw_path = str(src.composites_dir / f"composite_{year}.tif")
        processor.download_composite_local(composite, spec, region, raw_path)
        print(f"Downloaded {year} composite -> {raw_path}")

        # Clip + merge into the aligned per-year composite (averaging overlaps).
        merged = clip_merge.clip_and_merge_year(
            [raw_path], boundary, year, average_overlaps=True
        )
        if merged is not None:
            merged_paths[year] = merged
            print(f"Merged {year} composite -> {merged}")

    # Quick quality audit + masked difference between the two years.
    if YEAR_T1 in merged_paths and YEAR_T2 in merged_paths:
        pair = stats.load_pair(merged_paths[YEAR_T1], merged_paths[YEAR_T2])
        print("\nPer-band statistics (later composite):")
        print(stats.band_statistics(pair.later))
        diff = stats.masked_difference(pair)
        print(f"\nDifference computed: shape {diff.shape}, "
              f"{int(diff.count())} valid pixels.")


if __name__ == "__main__":
    build(Sources())
