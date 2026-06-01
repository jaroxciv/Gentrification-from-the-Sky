"""Change-detection model architectures, one module per method.

Each module exposes a model class (and where relevant a feature-extraction
entry point) for one of the methodologies compared in the paper. They share the
training loop, patching and thresholding utilities in ``gfs.change_detection``.
"""

from gfs.change_detection.models.bidatenet import BiDateNet
from gfs.change_detection.models.cgnet import CGNet
from gfs.change_detection.models.fc_siamdiff import FCSiamDiff
from gfs.change_detection.models.resnet import FeatureExtractor
from gfs.change_detection.models.simple_diff import simple_diff_change_map
from gfs.change_detection.models.tinycd import TinyCD

__all__ = [
    "BiDateNet",
    "CGNet",
    "FCSiamDiff",
    "FeatureExtractor",
    "TinyCD",
    "simple_diff_change_map",
]
