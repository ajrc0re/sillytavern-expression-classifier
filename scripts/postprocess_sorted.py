"""Standalone workflow to rename, combine, zip, and clean sorted files."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    """Load the YAML configuration file."""
    if not config_path.exists():
        msg = (
            f"Configuration file '{config_path}' not found. "
            "Create it manually or copy config.default.yaml > config.yaml."
        )
        raise FileNotFoundError(msg)
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_path(path_value: str) -> Path:
    """Resolve a relative path against the project root."""
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def slugify_label(label: str) -> str:
    """Create a filesystem-friendly slug from the label."""
    import re

    slug = re.sub(r"\s+", "_", label.strip())
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", slug)
    return slug or "label"


def iter_candidate_dirs(
    output_dir: Path, classes: Sequence[str]
) -> Iterator[tuple[str, Path]]:
    """Yield the label and directory pairs to post-process."""
    seen: set[Path] = set()
    for label in classes:
        dir_path = output_dir / label
        if dir_path.exists() and dir_path.is_dir():
            seen.add(dir_path)
            yield label, dir_path
    for dir_path in sorted(
        p for p in output_dir.iterdir() if p.is_dir() and p not in seen
    ):
        yield dir_path.name, dir_path


def rename_label_files(label_dir: Path, label: str) -> list[Path]:
    """Rename files within a label directory using slug and numbering."""
    files = sorted(p for p in label_dir.iterdir() if p.is_file())
    renamed: list[Path] = []
    slug = slugify_label(label)
    for idx, file_path in enumerate(files, 1):
        suffix = file_path.suffix
        new_name = f"{slug}.{idx:03d}{suffix}"
        destination = label_dir / new_name
        if destination != file_path:
            if destination.exists():
                counter = 1
                stem = f"{slug}.{idx:03d}"
                while True:
                    alt_name = f"{stem}-{counter}{suffix}"
                    alt_path = label_dir / alt_name
                    if not alt_path.exists():
                        destination = alt_path
                        break
                    counter += 1
            file_path = file_path.rename(destination)
        renamed.append(file_path)
    return renamed


def combine_sorted(
    output_dir: Path, classes: Sequence[str]
) -> tuple[list[Path], dict[str, int]]:
    """Rename and collect files from the sorted directory."""
    renamed_outputs: list[Path] = []
    per_label_counts: dict[str, int] = {}

    for label, label_dir in iter_candidate_dirs(output_dir, classes):
        renamed = rename_label_files(label_dir, label)
        per_label_counts[label] = len(renamed)
        renamed_outputs.extend(renamed)

    return renamed_outputs, per_label_counts


def copy_to_finished(
    renamed_outputs: Sequence[Path], run_dt: datetime
) -> Path:
    """Copy renamed files into a consolidated finished directory."""
    finished_root = ROOT / "data" / "finished"
    finished_root.mkdir(parents=True, exist_ok=True)

    run_local = run_dt.astimezone()
    date_part = run_local.strftime("%Y%m%d")
    time_part = run_local.strftime("%H%M%S")
    base_name = f"{len(renamed_outputs)}-{date_part}-{time_part}"
    final_dir = finished_root / base_name
    counter = 1
    while final_dir.exists():
        final_dir = finished_root / f"{base_name}-{counter}"
        counter += 1
    final_dir.mkdir(parents=True, exist_ok=True)

    for src_path in sorted(renamed_outputs):
        shutil.copy2(src_path, final_dir / src_path.name)

    return final_dir


def create_zip_archive(source_dir: Path, zip_name: str) -> Path:
    """Create a zip archive for the finalized directory."""
    output_dir = ROOT / "output_zips"
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_base = output_dir / zip_name
    archive_path = archive_base.with_suffix(".zip")
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for file_path in sorted(source_dir.iterdir()):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)
    return archive_path


def prompt_for_zip_name() -> str:
    """Prompt the user for a zip filename (without extension)."""
    while True:
        user_input = input(
            "Enter a name for the zip file (without extension): "
        ).strip()
        if user_input:
            return user_input
        print("Zip name cannot be empty. Please try again.")


def prompt_yes_no(message: str, default: bool = False) -> bool:
    """Prompt the user with a yes/no question."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    answer = input(f"{message}{suffix}").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def clean_sorted_directory(output_dir: Path) -> None:
    """Remove sorted directories while keeping the root intact."""
    for path in output_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename, combine, zip, and optionally clean sorted results."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory from config.yaml (defaults to 'data/sorted').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    output_dir_value = args.output_dir or config.get("output_dir")
    if not output_dir_value:
        msg = (
            "output_dir must be set either in config.yaml or via --output-dir."
        )
        raise ValueError(msg)
    output_dir = resolve_path(output_dir_value)
    if not output_dir.exists():
        msg = f"Output directory '{output_dir}' does not exist."
        raise FileNotFoundError(msg)

    classes_value: Iterable[str] | None = config.get("emotion_classes")
    classes = list(classes_value) if classes_value else []

    renamed_outputs, _per_label_counts = combine_sorted(output_dir, classes)
    total_files = len(renamed_outputs)
    if total_files == 0:
        print(
            f"No supported image files found in {output_dir}. Nothing to do."
        )
        return

    run_dt = datetime.now(timezone.utc)
    final_dir = copy_to_finished(renamed_outputs, run_dt)
    print(f"Copied {total_files} files into '{final_dir}'.")

    while True:
        zip_name = prompt_for_zip_name()
        tentative_path = (ROOT / "output_zips" / zip_name).with_suffix(".zip")
        if tentative_path.exists():
            if prompt_yes_no(
                f"'{tentative_path}' already exists. Overwrite?", default=False
            ):
                tentative_path.unlink()
                break
            print("Please enter a different zip name.")
        else:
            break

    archive_path = create_zip_archive(final_dir, zip_name)
    print(f"Created archive at '{archive_path}'.")

    if prompt_yes_no(
        "Would you like to delete the remaining sorted files in data/sorted? "
        "Reminder: renamed and combined data is available in data/finished.",
        default=False,
    ):
        clean_sorted_directory(output_dir)
        print(f"Removed contents of '{output_dir}'.")
    else:
        print("Sorted files preserved.")

    print("Post-processing complete.")


if __name__ == "__main__":
    main()
