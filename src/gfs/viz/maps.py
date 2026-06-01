"""Key figures for the paper (geopandas + matplotlib + contextily).

Each function takes already-loaded GeoDataFrames / rasters and writes a figure
to :data:`gfs.config.FIGURES_DIR`. The figures reproduced here are:

* :func:`plot_gentrification_with_changes` — London gentrification-score
  choropleth (disadvantaged LSOAs) with detected change pixels overlaid
  (paper Fig., §4).
* :func:`plot_conservation_areas_with_changes` — conservation-area validation:
  detected changes against London's protected conservation areas (paper §5).
* :func:`plot_gentrification_probabilities` — per-borough choropleth of
  predicted gentrification probability with selected planning layers overlaid
  (paper §5, planning-policy overlay).

No display side effects: figures are saved, not shown, so the module is import-
safe and headless-friendly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.mask import mask

from gfs.config import FIGURES_DIR

# Working CRS for all London layers in the study (UTM zone 30N).
TARGET_CRS = "EPSG:32630"
# Change-pixel overlay colors (RGBA).
CHANGE_PURPLE = (0.5, 0.0, 0.5, 1.0)
CHANGE_BLACK = (0.0, 0.0, 0.0, 1.0)


def _change_overlay_rgba(
    changes: np.ndarray, color: tuple[float, float, float, float]
) -> np.ndarray:
    """Build a transparent RGBA image painting changed pixels in ``color``."""
    height, width = changes.shape
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba[changes > 0] = color
    return rgba


def _imshow_extent(
    transform: rasterio.Affine, shape: tuple[int, int]
) -> tuple[float, float, float, float]:
    """``imshow`` extent (left, right, bottom, top) for a georeferenced raster."""
    height, width = shape
    return (
        transform[2],
        transform[2] + transform[0] * width,
        transform[5] + transform[4] * height,
        transform[5],
    )


def _ensure_figures_dir() -> Path:
    """Create and return the figures output directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


@dataclass(frozen=True)
class ChoroplethStyle:
    """Styling for the gentrification choropleth (paper Fig., §4)."""

    score_column: str = "gent_score_clip"
    disadvantaged_column: str = "disadvantaged"
    cmap: str = "coolwarm"
    alpha: float = 0.6
    figsize: tuple[float, float] = (15.0, 10.0)
    dpi: int = 500


def plot_gentrification_with_changes(
    gdf: gpd.GeoDataFrame,
    boroughs: gpd.GeoDataFrame,
    changes: np.ndarray,
    changes_transform: rasterio.Affine,
    changes_crs: Any,
    *,
    out_name: str = "gentrification_with_changes.png",
    title: str = "",
    note: str = "Used ONS Census 2011-2021 and IMD data.",
    style: ChoroplethStyle | None = None,
) -> Path:
    """London gentrification-score choropleth with detected changes overlaid.

    Disadvantaged LSOAs are filled by their (clipped) gentrification score on a
    diverging colormap; non-disadvantaged LSOAs are drawn as outlines only.
    Detected change pixels are painted purple on top, with borough boundaries
    framing the map. Saves to ``FIGURES_DIR / out_name`` and returns the path.
    Mirrors the notebook's ``plot_gentrification_with_changes``.
    """
    s = style or ChoroplethStyle()
    fig, ax = plt.subplots(1, 1, figsize=s.figsize)
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    rgba = _change_overlay_rgba(changes, CHANGE_PURPLE)
    ax.imshow(
        rgba,
        interpolation="none",
        extent=_imshow_extent(changes_transform, changes.shape),
        origin="upper",
    )
    ax.set_title(title, fontsize=20)

    gdf = gdf.to_crs(changes_crs)
    flag = gdf[s.disadvantaged_column].astype(bool)
    disadvantaged = cast("gpd.GeoDataFrame", gdf[flag])
    rest = cast("gpd.GeoDataFrame", gdf[~flag])
    disadvantaged.plot(
        column=s.score_column,
        ax=ax,
        legend=True,
        cax=cax,
        cmap=s.cmap,
        alpha=s.alpha,
    )
    rest.plot(linewidth=0.5, ax=ax, edgecolor="0.7", facecolor="none")

    boroughs = boroughs.to_crs(changes_crs)
    boroughs.boundary.plot(color="black", linewidth=0.4, ax=ax)

    ax.annotate(
        note,
        xy=(0.0, -0.025),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=8,
        color="black",
    )
    ax.legend(
        handles=[
            Patch(color="gray", label="Non-disadvantaged (border)"),
            Patch(color="purple", label="Detected Change"),
        ],
        loc="lower left",
        fontsize=8,
    )

    out_path = _ensure_figures_dir() / out_name
    fig.savefig(out_path, bbox_inches="tight", transparent=True, dpi=s.dpi)
    plt.close(fig)
    return out_path


def plot_conservation_areas_with_changes(
    lsoa_gdf: gpd.GeoDataFrame,
    boroughs_gdf: gpd.GeoDataFrame,
    conservation_gdf: gpd.GeoDataFrame,
    changes_path: str,
    *,
    out_name: str = "changes-london-conservation-areas.png",
    title: str = "London Conservation Areas and Detected Changes (2016-2021)",
    figsize: tuple[float, float] = (14.0, 14.0),
    dpi: int = 500,
) -> Path:
    """Validate detected change against protected conservation areas (paper §5).

    Reprojects all layers to :data:`TARGET_CRS`, repairs/clips the conservation
    geometries, masks the change raster to the borough extent and overlays the
    changed pixels (black) on the conservation areas (red). Saves to
    ``FIGURES_DIR / out_name``. Mirrors the notebook's "Full" conservation map.
    """
    if conservation_gdf.crs is None or "unknown" in str(conservation_gdf.crs):
        conservation_gdf = conservation_gdf.set_crs("EPSG:27700")
    conservation_gdf = cast(
        "gpd.GeoDataFrame", conservation_gdf[conservation_gdf.is_valid]
    )

    lsoa_gdf = lsoa_gdf.to_crs(TARGET_CRS)
    boroughs_gdf = boroughs_gdf.to_crs(TARGET_CRS)
    conservation_gdf = conservation_gdf.to_crs(TARGET_CRS)
    conservation_gdf = cast(
        "gpd.GeoDataFrame",
        conservation_gdf[conservation_gdf.is_valid & ~conservation_gdf.is_empty],
    )

    with rasterio.open(changes_path) as src_changes:
        masked, changes_transform = mask(
            src_changes, boroughs_gdf.geometry, crop=True
        )
    changes = masked[0]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    boroughs_gdf.plot(ax=ax, color="none", edgecolor="black", linewidth=0.5)
    lsoa_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5)
    if not conservation_gdf.empty:
        conservation_gdf.plot(ax=ax, color="red", edgecolor="red", alpha=0.5)

    ax.legend(
        handles=[
            Patch(facecolor="red", edgecolor="red", label="Conservation Areas"),
            Patch(color="black", label="Detected Change"),
        ],
        loc="upper left",
        bbox_to_anchor=(-0.2, 1.1),
        fontsize="small",
        frameon=True,
    )

    rgba = _change_overlay_rgba(changes, CHANGE_BLACK)
    ax.imshow(
        rgba,
        interpolation="none",
        extent=_imshow_extent(changes_transform, changes.shape),
        origin="upper",
    )
    ax.set_title(title)
    ax.axis("off")

    out_path = _ensure_figures_dir() / out_name
    fig.savefig(out_path, format="png", dpi=dpi, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return out_path


@dataclass(frozen=True)
class ProbabilityMapConfig:
    """Config for the per-borough gentrification-probability planning overlay."""

    probability_column: str = "probs"
    borough_name_column: str = "LAD11NM"
    layers_to_visualize: tuple[str, ...] = field(
        default=("Town_Centre_Boundaries", "Conservation_Areas")
    )
    cmap: str = "Reds"
    figsize: tuple[float, float] = (10.0, 8.0)
    dpi: int = 500


def plot_gentrification_probabilities(
    lsoa_probs_gdf: gpd.GeoDataFrame,
    planning_layers: Mapping[str, gpd.GeoDataFrame],
    boroughs_to_analyze: list[str],
    *,
    config: ProbabilityMapConfig | None = None,
    out_prefix: str = "gent_probabilities",
) -> list[Path]:
    """Per-borough choropleth of predicted gentrification probability + planning layers.

    For each borough: plot its LSOAs colored by predicted probability, then clip
    and overlay the selected planning layers (paper §5, planning-policy overlay).
    One figure per borough is written to ``FIGURES_DIR``; returns the paths.
    Mirrors the notebook's "Probabilities" planning-policy maps.
    """
    cfg = config or ProbabilityMapConfig()
    out_dir = _ensure_figures_dir()
    paths: list[Path] = []

    for borough_name in boroughs_to_analyze:
        borough_lsoas = cast(
            "gpd.GeoDataFrame",
            lsoa_probs_gdf[lsoa_probs_gdf[cfg.borough_name_column] == borough_name],
        )
        fig, ax = plt.subplots(1, 1, figsize=cfg.figsize)
        borough_lsoas.boundary.plot(ax=ax, linewidth=1, color="black")
        borough_lsoas.plot(
            column=cfg.probability_column, cmap=cfg.cmap, ax=ax, legend=True
        )

        layer_patches: list[Patch] = []
        for layer_name in cfg.layers_to_visualize:
            if layer_name not in planning_layers:
                continue
            borough_layer = gpd.clip(planning_layers[layer_name], borough_lsoas)
            borough_layer.plot(ax=ax, alpha=0.5)
            layer_patches.append(
                Patch(
                    facecolor="none",
                    edgecolor="black",
                    label=layer_name.replace("_", " "),
                )
            )
        if layer_patches:
            ax.legend(handles=layer_patches, loc="lower left")

        ax.set_title(f"Gentrification Probabilities in {borough_name} - LSOA")
        ax.axis("off")
        fig.tight_layout()

        slug = borough_name.replace(" ", "_").lower()
        out_path = out_dir / f"{out_prefix}_{slug}.png"
        fig.savefig(out_path, format="png", dpi=cfg.dpi, transparent=True, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)
    return paths
