"""Simple preprocessing pipeline for anime expression images."""
from pathlib import Path

from utils import load_image

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"


def preprocess_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in INPUT_DIR.glob("*"):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue

        image = load_image(image_path)
        image.save(OUTPUT_DIR / image_path.name)


if __name__ == "__main__":
    preprocess_all()
    print("Preprocessing done.")
