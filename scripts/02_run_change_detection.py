"""Stage 2 — deep-learning change detection (paper §3.3-3.4).

For each canonical method in :data:`gfs.config.CD_METHODS`, build/train the model
under one-shot Siamese learning, run it over the two Sentinel-2 composites, Otsu-
threshold the per-band change maps, and write them to
``outputs/features_<model>/`` (the satellite feature matrix Phi for stage 3).

    uv run scripts/02_run_change_detection.py

Requires the two clipped/merged composites (t1=2016, t2=2021) under COMPOSITES_DIR.
It can't run without the ~80GB rasters, but imports and type-checks cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

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
from gfs.config import (
    CD_EPOCHS,
    CD_METHODS,
    COMPOSITES_DIR,
    YEAR_T1,
    YEAR_T2,
)

# Single-channel TinyCD output channels (notebook: [32, 32, 32, 1], layer "1").
TINYCD_OUT_CHANNELS = [32, 32, 32, 1]


@dataclass(frozen=True)
class Composites:
    """Paths to the two clipped/merged annual composites."""

    t1: Path = COMPOSITES_DIR / f"clipped_merged_{YEAR_T1}.tiff"
    t2: Path = COMPOSITES_DIR / f"clipped_merged_{YEAR_T2}.tiff"


def build_siamese_model(method: str, device: torch.device) -> nn.Module:
    """Instantiate one of the canonical Siamese models on the given device."""
    if method == "tinycd":
        return TinyCD(
            bkbn_name="efficientnet_b0",
            weights="DEFAULT",
            output_layer_bkbn="1",
            freeze_backbone=False,
            bkbn_out_channels=TINYCD_OUT_CHANNELS,
        ).to(device)
    if method == "cgnet":
        return CGNet(weights="DEFAULT").to(device)
    if method == "bidatenet":
        return BiDateNet(n_channels=1, n_classes=1).to(device)
    if method == "fc_siamdiff":
        return FCSiamDiff(in_channels=1, classes=1).to(device)
    raise ValueError(f"Not a Siamese method: {method}")


def run_simple_diff(comps: Composites) -> None:
    """Simple-Diff baseline: abs diff + Multi-Otsu per band (no training)."""
    bands = all_bands()
    im1, im2 = load_dataset(str(comps.t1), str(comps.t2), bands)
    output_dir = features_output_dir("simple_diff")
    for band in bands:
        change_map = simple_diff_change_map(im1[band - 1], im2[band - 1])
        threshold_and_save_band(
            change_map, band, "simple_diff", str(comps.t2), output_dir
        )


def run_resnet(comps: Composites, device: torch.device) -> None:
    """Customized Res-Net: per-band autoencoder, trained for CD_EPOCHS_RESNET."""
    bands = all_bands()
    im1, im2 = load_dataset(str(comps.t1), str(comps.t2), bands)
    output_dir = features_output_dir("resnet")
    for band in bands:
        single1 = im1[band - 1 : band]
        single2 = im2[band - 1 : band]
        im1_t = torch.tensor(single1, dtype=torch.float32).unsqueeze(0).to(device)
        im2_t = torch.tensor(single2, dtype=torch.float32).unsqueeze(0).to(device)
        model = train_resnet_band(im1_t, im2_t, ngf=1, n_blocks=4, device=device)
        change_map = resnet_band_change_map(model, single1, single2, device=device)
        threshold_and_save_band(change_map, band, "resnet", str(comps.t2), output_dir)


def run_siamese(method: str, comps: Composites, device: torch.device) -> None:
    """Train a Siamese model one-shot, then extract + save its per-band features."""
    im1, im2 = load_dataset(str(comps.t1), str(comps.t2), [4])  # warm-up single band
    im1_t = torch.tensor(im1, dtype=torch.float32).unsqueeze(0).to(device)
    im2_t = torch.tensor(im2, dtype=torch.float32).unsqueeze(0).to(device)

    model = build_siamese_model(method, device)
    model = train_siamese(model, im1_t, im2_t, n_epochs=CD_EPOCHS, device=device)
    extract_and_save_features(
        model, method, str(comps.t1), str(comps.t2), kind="siamese", device=device
    )


def main() -> None:
    comps = Composites()
    device = select_device()
    print(f"Using device: {device}")

    for method in CD_METHODS:
        print(f"\n=== {method} ===")
        if method == "simple_diff":
            run_simple_diff(comps)
        elif method == "resnet":
            run_resnet(comps, device)
        else:
            run_siamese(method, comps, device)


if __name__ == "__main__":
    main()
