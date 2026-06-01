"""Stage 4 — modeling. See `gfs model`.

Thin shim around :func:`gfs.cli.run_modeling` (paper §3.5, §4-5). Prefer:

    gfs model --model tinycd     # or: uv run scripts/04_run_modeling.py
"""

from gfs.cli import run_modeling

if __name__ == "__main__":
    run_modeling()
