"""Export a shareable LSOA GeoPackage of the outputs. See `gfs export`.

Thin shim around :func:`gfs.cli.run_export` (logic lives in :mod:`gfs.export`):

    gfs export --model tinycd        # or: uv run scripts/export_lsoa_outputs.py [model]
"""

import sys

from gfs.cli import run_export

if __name__ == "__main__":
    run_export(sys.argv[1] if len(sys.argv) > 1 else "tinycd")
