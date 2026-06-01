"""Stage 1 — build the Sentinel-2 composites (X). See `gfs composites`.

Thin shim around :func:`gfs.cli.run_composites` (paper §3.2). Prefer the CLI:

    gfs composites           # or: uv run scripts/01_build_composites.py
"""

from gfs.cli import run_composites

if __name__ == "__main__":
    run_composites()
