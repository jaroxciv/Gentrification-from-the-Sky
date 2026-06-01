"""Stage 3 — build the gentrification score (Y). See `gfs gentrification`.

Thin shim around :func:`gfs.cli.run_gentrification` (paper §3.1). Prefer:

    gfs gentrification       # or: uv run scripts/03_build_gentrification.py
"""

from gfs.cli import run_gentrification

if __name__ == "__main__":
    run_gentrification()
