"""Batch classify anime character expressions and sort images by predicted label."""
import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
import yaml
from PIL import Image

from utils import copy_to_label

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def collect_images(input_dir: Path) -> List[Path]:
    files = [p for p in input_dir.glob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(files)


def encode_image_base64(image_path: Path) -> str:
    with image_path.open("rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def build_labels_prompt(labels: Iterable[str]) -> str:
    label_list = ", ".join(labels)
    return (
        "Classify the facial expression of the anime-style character in the attached image. "
        f"Choose exactly one label from the following list: {label_list}. "
        'Respond with JSON in the exact format {"label": "<label>", "confidence": <float between 0 and 1>}. '
        'If none apply, respond with {"label": "uncertain", "confidence": <float between 0 and 0.5>} with no additional text.'
    )


def extract_text_from_choice(choice: Dict) -> str:
    content = choice.get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def parse_json_block(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    snippet = match.group(0)
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return {}


def call_mistral_classifier(
    api_key: str,
    endpoint: str,
    model_name: str,
    image_b64: str,
    labels: List[str],
) -> Tuple[str, float, str]:
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_labels_prompt(labels)},
                    {"type": "image", "image_base64": image_b64},
                ],
            }
        ],
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    if not data.get("choices"):
        raise ValueError("No choices returned by Mistral API.")
    text = extract_text_from_choice(data["choices"][0])
    parsed = parse_json_block(text)
    label = str(parsed.get("label", "")).strip()
    confidence_raw = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    return label, confidence, text


def load_model(model_path: Path):
    import torch

    model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model


def run_local_classification(
    images: List[Path],
    search_dir: Path,
    output_dir: Path,
    classes: List[str],
    threshold: float,
    model_path: Path,
) -> List[Dict]:
    import torch
    from torchvision import transforms

    model = load_model(model_path)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    log_entries: List[Dict] = []
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        batch = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence_tensor, class_idx = torch.max(probabilities, dim=0)

        confidence = float(confidence_tensor)
        label = classes[int(class_idx)]
        assigned_label = label if confidence >= threshold else "uncertain"
        destination = copy_to_label(image_path, assigned_label, output_dir)

        log_entries.append(
            {
                "classifier": "local",
                "file": str(image_path.relative_to(search_dir)),
                "model_label": label,
                "predicted_label": label,
                "assigned_label": assigned_label,
                "confidence": confidence,
                "copied_to": str(destination.relative_to(output_dir)),
            }
        )

    return log_entries


def run_api_classification(
    images: List[Path],
    search_dir: Path,
    output_dir: Path,
    classes: List[str],
    threshold: float,
    config: Dict,
) -> List[Dict]:
    api_key = (config.get("mistral_api_key") or os.getenv("MISTRAL_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing Mistral API key. Set 'mistral_api_key' in config.yaml or the MISTRAL_API_KEY environment variable."
        )

    model_name = config.get("mistral_model", "pixtral-12b-latest")
    endpoint = config.get("mistral_endpoint", "https://api.mistral.ai/v1/chat/completions")

    log_entries: List[Dict] = []
    for image_path in images:
        raw_label = "uncertain"
        confidence = 0.0
        raw_response = ""
        try:
            encoded = encode_image_base64(image_path)
            raw_label, confidence, raw_response = call_mistral_classifier(
                api_key=api_key,
                endpoint=endpoint,
                model_name=model_name,
                image_b64=encoded,
                labels=classes,
            )
        except Exception as exc:
            print(f"Warning: API classification failed for {image_path.name}: {exc}")

        normalized_label = raw_label.strip() if raw_label else ""
        if normalized_label not in classes:
            normalized_label = "uncertain"

        assigned_label = normalized_label if normalized_label != "uncertain" and confidence >= threshold else "uncertain"
        destination = copy_to_label(image_path, assigned_label, output_dir)

        log_entries.append(
            {
                "classifier": "api",
                "file": str(image_path.relative_to(search_dir)),
                "model_label": raw_label or "uncertain",
                "predicted_label": normalized_label or "uncertain",
                "assigned_label": assigned_label,
                "confidence": float(confidence),
                "copied_to": str(destination.relative_to(output_dir)),
                "model_response": raw_response,
            }
        )

    return log_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify images and sort by emotion label.")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--input-dir", help="Override input directory from config")
    parser.add_argument("--processed-dir", help="Override processed directory from config")
    parser.add_argument("--output-dir", help="Override output directory from config")
    parser.add_argument("--model-path", help="Override model path from config")
    parser.add_argument("--mode", choices=["api", "local"], help="Override classification_mode from config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    mode = (args.mode or config.get("classification_mode", "api")).lower()
    if mode not in {"api", "local"}:
        raise ValueError(f"Unsupported classification_mode '{mode}'. Expected 'api' or 'local'.")

    input_dir = resolve_path(args.input_dir) if args.input_dir else resolve_path(config["input_dir"])
    processed_dir_config = args.processed_dir or config.get("processed_dir")
    processed_dir = resolve_path(processed_dir_config) if processed_dir_config else None
    output_dir = resolve_path(args.output_dir) if args.output_dir else resolve_path(config["output_dir"])
    classes: List[str] = config["emotion_classes"]
    threshold: float = float(config.get("confidence_threshold", 0.5))

    search_dir = processed_dir if processed_dir and processed_dir.exists() else input_dir
    if not search_dir.exists():
        raise FileNotFoundError(f"Input directory {search_dir} does not exist.")

    images = collect_images(search_dir)
    if not images:
        print(f"No supported images found in {search_dir}. Nothing to do.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "local":
        model_path_value = args.model_path or config.get("model_path")
        if not model_path_value:
            raise ValueError("model_path must be provided in config.yaml or via --model-path when using local mode.")
        model_path = resolve_path(model_path_value)
        log_entries = run_local_classification(images, search_dir, output_dir, classes, threshold, model_path)
    else:
        log_entries = run_api_classification(images, search_dir, output_dir, classes, threshold, config)

    log_path = output_dir / "classification_log.json"
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(log_entries, fh, indent=2)

    print(f"Processed {len(log_entries)} images using {mode} classifier. Log saved to {log_path}.")


if __name__ == "__main__":
    main()
