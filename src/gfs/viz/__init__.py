"""Visualization — maps and figures.

Reproduces the paper's figures: the London gentrification-score map, detected
changes overlaid on the score, conservation-area validation, and the
planning-policy overlays.
"""

from gfs.viz.maps import (
    plot_conservation_areas_with_changes,
    plot_gentrification_probabilities,
    plot_gentrification_with_changes,
)

__all__ = [
    "plot_gentrification_with_changes",
    "plot_conservation_areas_with_changes",
    "plot_gentrification_probabilities",
]
