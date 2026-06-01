"""Stage 5 — change-threshold ablation. See `gfs ablation`.

Thin shim around :func:`gfs.cli.run_ablation` (paper §5). Prefer:

    gfs ablation             # or: uv run scripts/05_ablation.py
"""

from gfs.cli import run_ablation

if __name__ == "__main__":
    run_ablation()
