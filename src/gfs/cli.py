"""``gfs`` — the Gentrification from the Sky command-line interface.

A single entry point over the whole pipeline, in dependency order::

    gfs composites        # X: build the Sentinel-2 composites (2016 & 2021)
    gfs change-detect     # models: detect change -> per-band feature maps
    gfs gentrification    # Y: build the census gentrification score
    gfs model             # predict gentrification (LogReg / SVC / XGBoost)
    gfs ablation          # threshold ablation sweep
    gfs export            # bundle a shareable LSOA GeoPackage

The orchestration lives here (the ``run_*`` functions); the numbered
``scripts/NN_*.py`` are thin shims that call them, and the Typer commands below
are thin wrappers. Heavy dependencies (torch, geopandas, Earth Engine) are
imported lazily inside each ``run_*`` so ``gfs --help`` stays instant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from gfs.config import (
    BOUNDARIES_DIR,
    CENSUS_DIR,
    COMPOSITES_DIR,
    OUTPUTS_DIR,
    PLANNING_DIR,
    YEAR_T1,
    YEAR_T2,
)

console = Console()
app = typer.Typer(
    help="Gentrification from the Sky — remote-sensing + ML pipeline.",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_BOUNDARY = BOUNDARIES_DIR / "LSOA_2011_London_gen_MHW.shp"


def _stage(title: str, subtitle: str) -> None:
    console.print(Panel(title, subtitle=subtitle, expand=False, style="bold cyan"))


def _planning_layer_paths(planning_dir: Path) -> dict[str, str]:
    """Map planning-layer name -> gpkg path, skipping the opportunity-areas layer."""
    return {
        p.stem: str(p)
        for p in sorted(planning_dir.glob("*.gpkg"))
        if p.name != "London_Plan_Opportunity_Areas.gpkg"
    }


# --- X: composites (WASDI) --------------------------------------------------
def run_composites(*, boundary: Path = DEFAULT_BOUNDARY, confirm_paid_run: bool = False) -> None:
    """Regenerate the Sentinel-2 composites on WASDI for both years (paper §3.2).

    The composites already ship as data; this only rebuilds them. WASDI is a
    licensed platform (arrange usage rights with the WASDI team) and runs metered
    remote compute, so it is gated behind ``confirm_paid_run`` / ``--confirm``.
    """
    from gfs.composites import wasdi_source

    _stage("Composites (X) — WASDI", f"Sentinel-2 {YEAR_T1} & {YEAR_T2}")
    for year in (YEAR_T1, YEAR_T2):
        tiles = wasdi_source.build_average_composite(
            str(boundary), year, confirm_paid_run=confirm_paid_run
        )
        console.print(f"  [green]✓[/] {year}: processed {len(tiles)} tile(s) on WASDI")
    console.print("  Download the processed tiles, then clip/merge with gfs.composites.clip_merge.")


# --- land cover (Google Earth Engine) ---------------------------------------
def run_landcover(*, boundary: Path = DEFAULT_BOUNDARY) -> Path:
    """Export the Dynamic World land-cover (green areas) layer via Earth Engine."""
    from gfs.composites import landcover

    _stage("Land cover — Earth Engine", "Dynamic World (green areas)")
    landcover.initialize_ee()
    out = landcover.export_dynamic_world(str(boundary))
    console.print(f"  [green]✓[/] wrote {out}")
    return Path(out)


# --- models: change detection -----------------------------------------------
def run_change_detection() -> None:
    """Run every canonical change-detection model and save per-band features."""
    import torch

    from gfs.change_detection.common import load_dataset, select_device
    from gfs.change_detection.features import (
        all_bands,
        extract_and_save_features,
        features_output_dir,
        resnet_band_change_map,
        threshold_and_save_band,
    )
    from gfs.change_detection.models.bidatenet import BiDateNet
    from gfs.change_detection.models.cgnet import CGNet
    from gfs.change_detection.models.fc_siamdiff import FCSiamDiff
    from gfs.change_detection.models.simple_diff import simple_diff_change_map
    from gfs.change_detection.models.tinycd import TinyCD
    from gfs.change_detection.train import train_resnet_band, train_siamese
    from gfs.config import CD_EPOCHS, CD_METHODS
    from gfs.seed import seed_everything

    seed_everything()  # deterministic weight init + data shuffling
    t1 = str(COMPOSITES_DIR / f"clipped_merged_{YEAR_T1}.tiff")
    t2 = str(COMPOSITES_DIR / f"clipped_merged_{YEAR_T2}.tiff")
    device = select_device()
    _stage("Change detection (models)", f"device: {device}")

    def build_siamese(method: str) -> torch.nn.Module:
        if method == "tinycd":
            return TinyCD(output_layer_bkbn="1", bkbn_out_channels=[32, 32, 32, 1]).to(device)
        if method == "cgnet":
            return CGNet(weights="DEFAULT").to(device)
        if method == "bidatenet":
            return BiDateNet(n_channels=1, n_classes=1).to(device)
        if method == "fc_siamdiff":
            return FCSiamDiff(in_channels=1, classes=1).to(device)
        raise ValueError(f"Not a Siamese method: {method}")

    for method in CD_METHODS:
        console.print(f"  [cyan]{method}[/]")
        if method == "simple_diff":
            bands = all_bands()
            im1, im2 = load_dataset(t1, t2, bands)
            out_dir = features_output_dir("simple_diff")
            for band in bands:
                cm = simple_diff_change_map(im1[band - 1], im2[band - 1])
                threshold_and_save_band(cm, band, "simple_diff", t2, out_dir)
        elif method == "resnet":
            bands = all_bands()
            im1, im2 = load_dataset(t1, t2, bands)
            out_dir = features_output_dir("resnet")
            for band in bands:
                s1, s2 = im1[band - 1 : band], im2[band - 1 : band]
                im1_t = torch.tensor(s1, dtype=torch.float32).unsqueeze(0).to(device)
                im2_t = torch.tensor(s2, dtype=torch.float32).unsqueeze(0).to(device)
                model = train_resnet_band(im1_t, im2_t, ngf=1, n_blocks=4, device=device)
                cm = resnet_band_change_map(model, s1, s2, device=device)
                threshold_and_save_band(cm, band, "resnet", t2, out_dir)
        else:
            im1, im2 = load_dataset(t1, t2, [4])
            im1_t = torch.tensor(im1, dtype=torch.float32).unsqueeze(0).to(device)
            im2_t = torch.tensor(im2, dtype=torch.float32).unsqueeze(0).to(device)
            model = train_siamese(
                build_siamese(method), im1_t, im2_t, n_epochs=CD_EPOCHS, device=device
            )
            extract_and_save_features(model, method, t1, t2, kind="siamese", device=device)


# --- Y: gentrification score -------------------------------------------------
def run_gentrification() -> Path:
    """Build the census-based gentrification score (paper §3.1)."""
    import geopandas as gpd
    import pandas as pd

    from gfs.gentrification import census
    from gfs.gentrification.score import gentrification_score

    _stage("Gentrification score (Y)", "ONS census + IMD")
    cd = CENSUS_DIR
    s = str
    table = census.assemble(
        age_t1=census.load_age(
            s(cd / "age-2011-lsoa.csv"), code_col="Area Codes", age_cols=("25-29", "30-34")
        ),
        # Later period uses the LSOA mid-year file the published score was built
        # from (the 2021 census age export is MSOA-level long-format, not usable).
        age_t2=census.load_age(
            s(cd / "age-2020-lsoa.csv"),
            code_col="LSOA Code",
            age_cols=tuple(str(a) for a in range(25, 35)),
        ),
        edu_t1=census.load_education_2011(
            s(cd / "qualifications-2011-2021-lsoa.xlsx"), s(cd / "lsoa11-to-lsoa21-lookup.csv")
        ),
        edu_t2=census.load_education_2021(s(cd / "education-2021-lsoa.csv")),
        house_t1=census.load_imd2010_column(
            s(cd / "imd2010eng.csv"), s(cd / "lsoa01-to-lsoa11-lookup.csv"), column="housesb_rank"
        ),
        house_t2=census.load_imd_domain(
            s(cd / "imd2019lsoa.csv"), census.IMD_HOUSING_DOMAIN, code_col="lsoa_codes"
        ),
        income_t1=census.load_imd2010_column(
            s(cd / "imd2010eng.csv"), s(cd / "lsoa01-to-lsoa11-lookup.csv"), column="income_rank"
        ),
        income_t2=census.load_imd_domain(
            s(cd / "imd2019lsoa.csv"), census.IMD_INCOME_DOMAIN, code_col="lsoa_codes"
        ),
    )
    # Restrict to Greater London and rank percentiles within it (the published
    # score is London-relative); the raw census/IMD inputs are England-wide.
    london = list(gpd.read_file(str(DEFAULT_BOUNDARY))["LSOA11CD"])
    table = cast(pd.DataFrame, table[table["lsoa_code"].isin(london)])

    scored = gentrification_score(table)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "gentrification_score.csv"
    scored.to_csv(out_path, index=False)
    n_disadv = int(cast("int", scored["disadvantaged"].sum()))
    console.print(f"  [green]✓[/] {len(scored)} LSOAs ({n_disadv} disadvantaged) -> {out_path}")
    return out_path


# --- modeling ----------------------------------------------------------------
def _metrics_table(frame: object) -> Table:
    import pandas as pd

    assert isinstance(frame, pd.DataFrame)
    table = Table(show_header=True, header_style="bold magenta")
    for col in frame.columns:
        table.add_column(str(col))
    for _, row in frame.iterrows():
        table.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row])
    return table


def run_modeling(model: str = "tinycd") -> None:
    """Build the modeling table and evaluate the classifiers (paper §4-5)."""
    from typing import cast

    import geopandas as gpd
    import pandas as pd

    from gfs.modeling import classify, dataset
    from gfs.modeling.features import aggregate_changes_to_lsoa, create_percentage_features

    _stage("Modeling", f"change-detection features: {model}")
    lsoa_gdf = cast(gpd.GeoDataFrame, gpd.read_file(DEFAULT_BOUNDARY))
    features_dir = OUTPUTS_DIR / f"features_{model}"
    lsoa_changes = aggregate_changes_to_lsoa(str(features_dir), lsoa_gdf).reset_index()
    planning_gdf = create_percentage_features(lsoa_gdf, _planning_layer_paths(PLANNING_DIR))
    planning = cast(pd.DataFrame, planning_gdf.drop(columns="geometry"))
    score = pd.read_csv(OUTPUTS_DIR / "gentrification_score.csv")

    table = dataset.assemble_modeling_table(score, lsoa_changes, planning)
    x = cast(pd.DataFrame, table.data[table.predictors])
    y = cast(pd.Series, table.data[table.target])
    results = classify.train_and_evaluate(x, y)
    frame = classify.results_to_frame(results)
    console.print(_metrics_table(frame))
    frame.to_csv(OUTPUTS_DIR / f"modeling_metrics_{model}.csv", index=False)


def run_ablation() -> None:
    """Run the change-threshold ablation sweep (paper §5)."""
    from typing import cast

    import geopandas as gpd
    import pandas as pd

    from gfs.modeling import ablation, dataset
    from gfs.modeling.features import aggregate_changes_to_lsoa, create_percentage_features

    _stage("Ablation", "threshold sweep (simple-diff)")
    lsoa_gdf = cast(gpd.GeoDataFrame, gpd.read_file(DEFAULT_BOUNDARY))
    lsoa_changes = aggregate_changes_to_lsoa(
        str(OUTPUTS_DIR / "thresholds"), lsoa_gdf
    ).reset_index()
    planning_gdf = create_percentage_features(lsoa_gdf, _planning_layer_paths(PLANNING_DIR))
    planning = cast(pd.DataFrame, planning_gdf.drop(columns="geometry"))
    planning_cols = dataset.planning_predictors(planning)

    score = dataset.ensure_lsoa_key(pd.read_csv(OUTPUTS_DIR / "gentrification_score.csv"))
    merged = cast(
        pd.DataFrame,
        score.merge(lsoa_changes, on=dataset.LSOA_CODE_COL).merge(
            planning, on=dataset.LSOA_CODE_COL
        ),
    )
    merged = dataset.binarize_score(merged)
    points = ablation.evaluate_thresholds(merged, planning_cols)
    frame = ablation.ablation_to_frame(points)
    best = ablation.best_threshold(points, metric="f1")
    console.print(_metrics_table(frame))
    console.print(f"  best F1 threshold: [bold]{best.threshold}[/] (F1={best.f1:.4f})")
    frame.to_csv(OUTPUTS_DIR / "ablation_threshold_sweep.csv", index=False)


# --- export ------------------------------------------------------------------
def run_export(model: str = "tinycd") -> Path:
    """Build the shareable LSOA GeoPackage of scores + a model's features."""
    from gfs.export import export_outputs

    _stage("Export", f"shareable GeoPackage ({model})")
    out_path = export_outputs(model)
    console.print(f"  [green]✓[/] wrote {out_path}")
    return out_path


# --- Typer command wrappers --------------------------------------------------
@app.command()
def composites(
    boundary: Annotated[Path, typer.Option(help="LSOA boundary shapefile.")] = DEFAULT_BOUNDARY,
    confirm: Annotated[
        bool, typer.Option("--confirm", help="Confirm metered WASDI compute.")
    ] = False,
) -> None:
    """X — regenerate the Sentinel-2 composites on WASDI (already provided as data)."""
    run_composites(boundary=boundary, confirm_paid_run=confirm)


@app.command()
def landcover(
    boundary: Annotated[Path, typer.Option(help="LSOA boundary shapefile.")] = DEFAULT_BOUNDARY,
) -> None:
    """Export the Dynamic World land-cover (green areas) layer via Earth Engine."""
    run_landcover(boundary=boundary)


@app.command("change-detect")
def change_detect() -> None:
    """Models — run every change-detection method."""
    run_change_detection()


@app.command()
def gentrification() -> None:
    """Y — build the census gentrification score."""
    run_gentrification()


@app.command()
def model(
    name: Annotated[str, typer.Option("--model", "-m", help="Change-detection model.")] = "tinycd",
) -> None:
    """Predict gentrification from satellite + planning features."""
    run_modeling(name)


@app.command()
def ablation() -> None:
    """Run the change-threshold ablation sweep."""
    run_ablation()


@app.command()
def export(
    name: Annotated[str, typer.Option("--model", "-m", help="Change-detection model.")] = "tinycd",
) -> None:
    """Bundle a shareable LSOA GeoPackage of the outputs."""
    run_export(name)


if __name__ == "__main__":
    app()
