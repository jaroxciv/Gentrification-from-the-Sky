"""Simple-Diff baseline change detector (paper §3.3).

The non-neural reference method the deep models are compared against: standardize
both composites, take the per-pixel absolute difference, normalize, and threshold
with Multi-Otsu. No training, no patches — a direct image-algebra baseline.

The notebook computes its change maps from the z-standardized composites already
produced by :func:`gfs.change_detection.common.load_image`, so the difference is
taken on standardized reflectance.
"""

from __future__ import annotations

import numpy as np

from gfs.change_detection.common import FloatArray, normalize01


def simple_diff_change_map(im1: FloatArray, im2: FloatArray) -> FloatArray:
    """Absolute-difference change map between two standardized single-band images.

    ``im1`` / ``im2`` are ``(H, W)`` standardized arrays for one band. Returns the
    min-max normalized absolute difference in ``[0, 1]`` (the continuous change
    map fed to :func:`apply_thresholding_strategy` with ``multiotsu``).
    """
    absolute_difference = np.abs(im2 - im1)
    return normalize01(absolute_difference)
