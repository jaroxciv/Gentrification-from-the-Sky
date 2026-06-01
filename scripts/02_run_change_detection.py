"""Stage 2 — deep-learning change detection. See `gfs change-detect`.

Thin shim around :func:`gfs.cli.run_change_detection` (paper §3.3-3.4). Prefer:

    gfs change-detect        # or: uv run scripts/02_run_change_detection.py
"""

from gfs.cli import run_change_detection

if __name__ == "__main__":
    run_change_detection()
