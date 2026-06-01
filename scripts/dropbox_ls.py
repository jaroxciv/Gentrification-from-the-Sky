"""Inventory the Dropbox data root: recursive listing with file sizes & a summary.

Prunes heavy, irrelevant directories (virtualenvs, .git, caches) so the walk
doesn't drown in thousands of tiny files.

Usage:
    uv run scripts/dropbox_ls.py                 # list DROPBOX_DATA_ROOT
    uv run scripts/dropbox_ls.py /Master/LSE     # list a specific path
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import cast

from dropbox.files import (
    FileMetadata,
    FolderMetadata,
    ListFolderResult,
    Metadata,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dbx import data_root, get_client  # noqa: E402

# Directory names to skip entirely (never descend into them).
IGNORE_DIRS = {
    ".venv",
    "venv",
    "env",
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "node_modules",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _list_one(path: str) -> list[Metadata]:
    """List a single folder (non-recursive), handling pagination."""
    dbx = get_client()
    res = cast(ListFolderResult, dbx.files_list_folder(path))
    out: list[Metadata] = list(res.entries)
    while res.has_more:
        res = cast(ListFolderResult, dbx.files_list_folder_continue(res.cursor))
        out.extend(res.entries)
    return out


def collect(path: str) -> list[Metadata]:
    """Recursively list entries under `path`, pruning IGNORE_DIRS."""
    entries: list[Metadata] = []
    stack = [path]
    while stack:
        current = stack.pop()
        for e in _list_one(current):
            entries.append(e)
            if isinstance(e, FolderMetadata) and e.name not in IGNORE_DIRS:
                stack.append(e.path_lower or f"{current}/{e.name}")
    return entries


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else data_root()
    print(f"Listing: {path}\n")

    entries = collect(path)

    total_bytes = 0
    n_files = n_dirs = 0
    by_ext: dict[str, list[int]] = defaultdict(list)

    for e in sorted(entries, key=lambda x: (x.path_display or "").lower()):
        rel = (e.path_display or "")[len(path) :].lstrip("/")
        if isinstance(e, FolderMetadata):
            n_dirs += 1
            print(f"  📁 {rel}/")
        elif isinstance(e, FileMetadata):
            n_files += 1
            total_bytes += e.size
            ext = Path(e.name).suffix.lower() or "(none)"
            by_ext[ext].append(e.size)
            print(f"     {rel}  ({human(e.size)})")

    print("\n" + "=" * 60)
    print(f"Folders: {n_dirs}   Files: {n_files}   Total size: {human(total_bytes)}")
    print("\nBy extension:")
    for ext, sizes in sorted(by_ext.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {ext:12s} {len(sizes):5d} files   {human(sum(sizes))}")


if __name__ == "__main__":
    main()
