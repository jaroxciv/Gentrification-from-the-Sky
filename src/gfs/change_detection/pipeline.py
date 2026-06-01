"""Run change detection for the canonical methods, via a model registry.

Replaces ad-hoc ``if method == ...`` dispatch with a single :data:`REGISTRY`
mapping each name in :data:`gfs.config.CD_METHODS` to how it is built and run.
Adding a seventh method is then one registry entry — no edits to the runner.

Three execution *kinds* share the per-band feature workflow:

- ``simple_diff`` — non-neural absolute-difference + Multi-Otsu baseline.
- ``resnet`` — per-band autoencoder trained for many epochs.
- ``siamese`` — a ``forward(x1, x2)`` model trained one-shot (TinyCD, CGNet,
  BiDateNet, FC-SiamDiff).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
from gfs.change_detection.train import train_resnet_band, train_siamese
from gfs.config import CD_EPOCHS

# TinyCD single-channel output config (notebook: layer "1", [32, 32, 32, 1]).
_TINYCD_OUT_CHANNELS = [32, 32, 32, 1]


def _build_fc_siamdiff(device: torch.device) -> nn.Module:
    from gfs.change_detection.models.fc_siamdiff import FCSiamDiff

    return FCSiamDiff(in_channels=1, classes=1).to(device)


def _build_cgnet(device: torch.device) -> nn.Module:
    from gfs.change_detection.models.cgnet import CGNet

    return CGNet(weights="DEFAULT").to(device)


def _build_bidatenet(device: torch.device) -> nn.Module:
    from gfs.change_detection.models.bidatenet import BiDateNet

    return BiDateNet(n_channels=1, n_classes=1).to(device)


def _build_tinycd(device: torch.device) -> nn.Module:
    from gfs.change_detection.models.tinycd import TinyCD

    return TinyCD(output_layer_bkbn="1", bkbn_out_channels=_TINYCD_OUT_CHANNELS).to(device)


@dataclass(frozen=True)
class CDMethodSpec:
    """How to build and run one change-detection method.

    ``build`` is ``None`` for the non-Siamese kinds (simple_diff, resnet), which
    have their own per-band routines.
    """

    name: str
    kind: str  # "simple_diff" | "resnet" | "siamese"
    build: Callable[[torch.device], nn.Module] | None = None


REGISTRY: dict[str, CDMethodSpec] = {
    "simple_diff": CDMethodSpec("simple_diff", "simple_diff"),
    "resnet": CDMethodSpec("resnet", "resnet"),
    "fc_siamdiff": CDMethodSpec("fc_siamdiff", "siamese", _build_fc_siamdiff),
    "cgnet": CDMethodSpec("cgnet", "siamese", _build_cgnet),
    "bidatenet": CDMethodSpec("bidatenet", "siamese", _build_bidatenet),
    "tinycd": CDMethodSpec("tinycd", "siamese", _build_tinycd),
}


def get_method(name: str) -> CDMethodSpec:
    """Look up a change-detection method spec by name."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown change-detection method {name!r}; known: {sorted(REGISTRY)}"
        ) from None


def _run_simple_diff(t1: str, t2: str) -> None:
    from gfs.change_detection.models.simple_diff import simple_diff_change_map

    bands = all_bands()
    im1, im2 = load_dataset(t1, t2, bands)
    out_dir = features_output_dir("simple_diff")
    for band in bands:
        change_map = simple_diff_change_map(im1[band - 1], im2[band - 1])
        threshold_and_save_band(change_map, band, "simple_diff", t2, out_dir)


def _run_resnet(t1: str, t2: str, device: torch.device) -> None:
    bands = all_bands()
    im1, im2 = load_dataset(t1, t2, bands)
    out_dir = features_output_dir("resnet")
    for band in bands:
        single1, single2 = im1[band - 1 : band], im2[band - 1 : band]
        im1_t = torch.tensor(single1, dtype=torch.float32).unsqueeze(0).to(device)
        im2_t = torch.tensor(single2, dtype=torch.float32).unsqueeze(0).to(device)
        model = train_resnet_band(im1_t, im2_t, ngf=1, n_blocks=4, device=device)
        change_map = resnet_band_change_map(model, single1, single2, device=device)
        threshold_and_save_band(change_map, band, "resnet", t2, out_dir)


def _run_siamese(spec: CDMethodSpec, t1: str, t2: str, device: torch.device) -> None:
    assert spec.build is not None
    im1, im2 = load_dataset(t1, t2, [4])  # warm-up single band
    im1_t = torch.tensor(im1, dtype=torch.float32).unsqueeze(0).to(device)
    im2_t = torch.tensor(im2, dtype=torch.float32).unsqueeze(0).to(device)
    model = train_siamese(spec.build(device), im1_t, im2_t, n_epochs=CD_EPOCHS, device=device)
    extract_and_save_features(model, spec.name, t1, t2, kind="siamese", device=device)


def extract_method(name: str, t1: str, t2: str, device: torch.device | None = None) -> None:
    """Build, train (if needed) and save per-band change features for one method."""
    spec = get_method(name)
    dev = device or select_device()
    if spec.kind == "simple_diff":
        _run_simple_diff(t1, t2)
    elif spec.kind == "resnet":
        _run_resnet(t1, t2, dev)
    else:
        _run_siamese(spec, t1, t2, dev)
