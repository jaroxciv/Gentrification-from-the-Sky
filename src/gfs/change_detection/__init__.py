"""Models — Deep-learning change detection.

Detects physical urban change between the two annual composites. Compares the
six methodologies from the paper (§3.3): Simple-Diff, Res-Net, FC-SiamDiff,
CGNet, Bi-Temporal Siamese (BiDateNet), and TinyCD. Each model produces a
per-band change map; Otsu thresholding turns features into binary change.

Methods are dispatched through :data:`REGISTRY`; :func:`extract_method` runs one
end-to-end. Importing this subpackage pulls in PyTorch.
"""

from gfs.change_detection.features import extract_and_save_features
from gfs.change_detection.pipeline import (
    REGISTRY,
    CDMethodSpec,
    extract_method,
    get_method,
)
from gfs.change_detection.train import train_resnet_band, train_siamese

__all__ = [
    "REGISTRY",
    "CDMethodSpec",
    "get_method",
    "extract_method",
    "train_siamese",
    "train_resnet_band",
    "extract_and_save_features",
]
