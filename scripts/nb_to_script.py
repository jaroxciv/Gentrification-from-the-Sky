"""Convert a Jupyter notebook into a clean, readable .py reference script.

This is a *migration aid*, not the final code: it gives a faithful, output-free
linear dump of a notebook so the logic can be reorganized into the gfs package.

It does three things:
  - markdown cells  -> comment blocks (headers become ``# === Header ===``)
  - code cells      -> kept verbatim, with a ``# [cell N]`` marker
  - cruft stripped  -> IPython magics (``%``), shell lines (``!``),
                       ``get_ipython(...)``, and Google Colab boilerplate
                       (``drive.mount`` / ``from google.colab import ...``)
                       are commented out, not deleted, so nothing is lost.

Usage:
    uv run scripts/nb_to_script.py input.ipynb [output.py]
    # default output: alongside input with a .py extension
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# Lines that only make sense inside a notebook / Colab; comment them out.
_CRUFT = re.compile(
    r"""^\s*(
        [%!]                       # line magics / shell escapes
        | get_ipython\(            # magic under the hood
        | drive\.mount\(           # Colab drive
        | from\s+google\.colab     # Colab imports
        | import\s+google\.colab
    )""",
    re.VERBOSE,
)


def _src(cell: dict[str, Any]) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def _md_to_comment(text: str) -> list[str]:
    out: list[str] = [""]
    for line in text.splitlines():
        stripped = line.lstrip("#").strip()
        if line.lstrip().startswith("#") and stripped:
            out.append(f"# === {stripped} ===")
        elif line.strip():
            out.append(f"# {line.rstrip()}")
        else:
            out.append("#")
    return out


def _clean_code(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        if _CRUFT.match(line):
            out.append(f"# [stripped] {line.rstrip()}")
        else:
            out.append(line.rstrip())
    return out


def convert(nb_path: Path, out_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    lines: list[str] = [
        '"""Auto-extracted from ' + nb_path.name + '.',
        "",
        "Linear dump of the original notebook (outputs removed, Colab/IPython",
        "cruft commented out). Reorganize from here into the gfs package.",
        '"""',
        "",
    ]
    for i, cell in enumerate(nb.get("cells", [])):
        text = _src(cell).strip("\n")
        if not text.strip():
            continue
        kind = cell.get("cell_type")
        if kind == "markdown":
            lines += _md_to_comment(text)
        elif kind == "code":
            lines.append(f"\n# [cell {i}] " + "-" * 40)
            lines += _clean_code(text)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_cells = len(nb.get("cells", []))
    print(f"{nb_path.name}: {n_cells} cells -> {out_path}  ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    nb_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else nb_path.with_suffix(".py")
    convert(nb_path, out_path)


if __name__ == "__main__":
    main()
