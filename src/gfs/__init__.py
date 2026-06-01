"""gfs — Gentrification from the Sky.

Measuring urban gentrification in Greater London from open Sentinel-2 satellite
imagery using deep-learning change detection and machine learning.

Pipeline (read top to bottom):

    composites        ── X: build Sentinel-2 annual composites (2016, 2021)
    change_detection  ── models: detect physical change between the two years
    gentrification    ── Y: build the census-based gentrification score (target)
    modeling          ── predict gentrification from satellite + baseline features
    viz               ── maps and figures

See the paper: "Gentrification from the Sky: Using Remote Sensing and Machine
Learning for Urban Change Detection" (CUPUM 2025).

Subpackages are imported on demand (``from gfs.modeling import ...``) so the
heavy geospatial / deep-learning dependencies load only when used; the top level
stays light and exposes just the study configuration.
"""

from __future__ import annotations

from gfs.config import LONDON, STUDY, StudyConfig

__version__ = "0.1.0"
__all__ = ["LONDON", "STUDY", "StudyConfig", "__version__"]
