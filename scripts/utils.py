"""Utility helpers for image loading and output organization."""
from pathlib import Path
from typing import Tuple

from PIL import Image


def load_image(path: Path, size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Open an image, convert to RGB, and resize to the desired shape."""
    img = Image.open(path).convert("RGB")
    return img.resize(size)


def ensure_sorted_path(output_root: Path, label: str) -> Path:
    """Ensure the output directory for the label exists and return the destination path."""
    target_dir = output_root / label
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def copy_to_label(image_path: Path, label: str, output_root: Path) -> Path:
    """Copy an image into the label subdirectory under the output root."""
    from shutil import copy2

    destination_dir = ensure_sorted_path(output_root, label)
    destination = destination_dir / image_path.name
    copy2(image_path, destination)
    return destination
