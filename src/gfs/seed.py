"""Deterministic seeding for reproducible runs.

The modeling stage already threads :data:`gfs.config.RANDOM_STATE` through
scikit-learn, but the deep-learning change-detection stage (weight init,
DataLoader shuffling) is otherwise non-deterministic. :func:`seed_everything`
seeds Python, NumPy and (lazily) PyTorch so a run can be reproduced.
"""

from __future__ import annotations

import os
import random

from gfs.config import RANDOM_STATE


def seed_everything(seed: int = RANDOM_STATE) -> None:
    """Seed Python / NumPy / PyTorch RNGs for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # torch is optional for the non-DL stages
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
