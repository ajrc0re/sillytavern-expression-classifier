"""Utility script to clear the sorted output directory."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SORTED_DIR = ROOT / "data" / "sorted"


def clean_sorted_dir(target: Path = SORTED_DIR) -> None:
    """Remove every file and subdirectory within the target folder."""
    target.mkdir(parents=True, exist_ok=True)

    for item in target.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


if __name__ == "__main__":
    clean_sorted_dir()
    print(f"Cleared contents of {SORTED_DIR}")
