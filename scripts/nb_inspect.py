"""Print the structure of a notebook: markdown headers + code-cell summaries.

Ignores cell outputs entirely (so it stays fast on output-bloated notebooks).

Usage:
    uv run scripts/nb_inspect.py path/to/notebook.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def src(cell: dict[str, Any]) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else str(s)


def main() -> None:
    nb_path = Path(sys.argv[1])
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    print(f"# {nb_path.name}  ({len(cells)} cells)\n")
    for i, cell in enumerate(cells):
        kind = cell.get("cell_type")
        text = src(cell).strip()
        if not text:
            continue
        if kind == "markdown":
            for line in text.splitlines():
                if line.lstrip().startswith("#"):
                    print(f"  [md cell {i}] {line.strip()}")
        elif kind == "code":
            lines = text.splitlines()
            first = lines[0][:90]
            print(f"  [code {i:>3}] ({len(lines):>3} lines) {first}")


if __name__ == "__main__":
    main()
