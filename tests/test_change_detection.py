"""Behavioral contract for the change-detection models.

The package promises a model that maps a bi-temporal image pair to a change map
of the same spatial extent. We check that contract on a self-contained model
(no pretrained backbone download), which is enough to catch a broken forward
pass after any refactor.
"""

from __future__ import annotations

import torch

from gfs.change_detection.models.fc_siamdiff import FCSiamDiff


def test_change_map_preserves_spatial_extent() -> None:
    model = FCSiamDiff(in_channels=1, classes=1).eval()
    image_t1 = torch.zeros(1, 1, 64, 64)
    image_t2 = torch.ones(1, 1, 64, 64)

    with torch.no_grad():
        change_map = model(image_t1, image_t2)

    assert change_map.shape[-2:] == (64, 64)
    assert torch.isfinite(change_map).all()
