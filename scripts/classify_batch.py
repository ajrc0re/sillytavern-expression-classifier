"""Batch classify anime character expressions and sort images by predicted label."""
import argparse
import json
from pathlib import Path
from typing import List

import torch
import yaml
from PIL import Image
from torchvision import transforms

from utils import copy_to_label

ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_model(model_path: Path) -> torch.nn.Module:
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model


def collect_images(input_dir: Path) -> List[Path]:
    return [p for p in input_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify images and sort by emotion label.")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--input-dir", help="Override input directory from config")
    parser.add_argument("--processed-dir", help="Override processed directory from config")
    parser.add_argument("--output-dir", help="Override output directory from config")
    parser.add_argument("--model-path", help="Override model path from config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    model_path = resolve_path(args.model_path) if args.model_path else resolve_path(config["model_path"])
    input_dir = resolve_path(args.input_dir) if args.input_dir else resolve_path(config["input_dir"])
    processed_dir_config = args.processed_dir or config.get("processed_dir")
    processed_dir = resolve_path(processed_dir_config) if processed_dir_config else None
    output_dir = resolve_path(args.output_dir) if args.output_dir else resolve_path(config["output_dir"])
    classes: List[str] = config["emotion_classes"]
    threshold: float = float(config.get("confidence_threshold", 0.5))

    if processed_dir and processed_dir.exists():
        search_dir = processed_dir
    else:
        search_dir = input_dir

    model = load_model(model_path)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    log = []

    for image_path in collect_images(search_dir):
        image = Image.open(image_path).convert("RGB")
        batch = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, class_idx = torch.max(probabilities, dim=0)

        label = classes[int(class_idx)]
        assigned_label = label if float(confidence) >= threshold else "uncertain"
        destination = copy_to_label(image_path, assigned_label, output_dir)

        log.append(
            {
                "file": str(image_path.relative_to(search_dir)),
                "predicted_label": label,
                "assigned_label": assigned_label,
                "confidence": float(confidence),
                "copied_to": str(destination.relative_to(output_dir)),
            }
        )

    log_path = output_dir / "classification_log.json"
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)

    print(f"Processed {len(log)} images. Log saved to {log_path}.")


if __name__ == "__main__":
    main()
