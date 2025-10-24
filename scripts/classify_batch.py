"""Batch classify anime character expressions and sort images by label."""

import argparse
import base64
import json
import os
import re
import shutil
from collections.abc import Iterable
from datetime import datetime, timezone
from mimetypes import guess_type
from pathlib import Path

import requests
import torch
import yaml
from PIL import Image
from torchvision import transforms
from utils import copy_to_label

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_PROMPT_TEMPLATE = (
    "Classify the facial expression of the anime-style character in the "
    "attached image. Choose exactly one label from the provided list: "
    "{labels}. Respond only with JSON in the format "
    '{"label": "<label>", "confidence": <float between 0 and 1>}. '
    "Always pick the closest matching label from the list."
)


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        if config_path.name == "config.yaml":
            template_path = config_path.with_name("config.default.yaml")
            if template_path.exists():
                shutil.copy(template_path, config_path)
                print(
                    f"Created {config_path.name} from template. "
                    f"Update it with your API key and settings."
                )
        if not config_path.exists():
            msg = (
                f"Configuration file '{config_path}' not found. "
                "Create it manually or copy config.default.yaml > config.yaml."
            )
            raise FileNotFoundError(msg)
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def collect_images(input_dir: Path) -> list[Path]:
    files = [
        p
        for p in input_dir.glob("*")
        if p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files)


def encode_image_base64(image_path: Path) -> str:
    with image_path.open("rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def detect_mime_type(image_path: Path) -> str:
    mime_type, _ = guess_type(str(image_path))
    if not mime_type:
        extension = image_path.suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(extension, "application/octet-stream")
    return mime_type


def image_to_data_url(image_path: Path) -> str:
    mime_type = detect_mime_type(image_path)
    return f"data:{mime_type};base64,{encode_image_base64(image_path)}"


def slugify_label(label: str) -> str:
    slug = re.sub(r"\s+", "_", label.strip())
    slug = re.sub(r"[^A-Za-z0-9_-]", "_", slug)
    return slug or "label"


def build_labels_prompt(labels: Iterable[str], template: str | None) -> str:
    label_list = ", ".join(labels)
    prompt_template = template if template else DEFAULT_PROMPT_TEMPLATE
    if "{labels}" in prompt_template:
        return prompt_template.replace("{labels}", label_list)
    return f"{prompt_template.rstrip()}\nLabels: {label_list}"


def extract_text_from_choice(choice: dict) -> str:
    content = choice.get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def parse_json_block(text: str) -> dict:
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
    image_path: Path,
    prompt_text: str,
) -> tuple[str, float, str]:
    image_payload = {
        "type": "image_url",
        "image_url": image_to_data_url(image_path),
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    image_payload,
                ],
            }
        ],
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        endpoint, headers=headers, json=payload, timeout=60
    )
    response.raise_for_status()
    data = response.json()
    if not data.get("choices"):
        msg = "No choices returned by Mistral API."
        raise ValueError(msg)
    text = extract_text_from_choice(data["choices"][0])
    parsed = parse_json_block(text)
    label = str(parsed.get("label", "")).strip()
    confidence_raw = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    return label, confidence, text


def call_openai_classifier(
    api_key: str,
    endpoint: str,
    model: str,
    image_path: Path,
    prompt_text: str,
) -> tuple[str, float, str]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_path),
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 100,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        endpoint, headers=headers, json=payload, timeout=60
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices")
    if not choices:
        msg = "No choices returned by OpenAI API."
        raise ValueError(msg)
    text = extract_text_from_choice(choices[0])
    parsed = parse_json_block(text)
    label = str(parsed.get("label", "")).strip()
    confidence_raw = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    return label, confidence, text


def call_xai_classifier(
    api_key: str,
    endpoint: str,
    model: str,
    image_path: Path,
    prompt_text: str,
) -> tuple[str, float, str]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {
                        "type": "input_image",
                        "image_base64": encode_image_base64(image_path),
                        "mime_type": detect_mime_type(image_path),
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 100,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        endpoint, headers=headers, json=payload, timeout=60
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices")
    if not choices:
        msg = "No choices returned by xAI Grok API."
        raise ValueError(msg)
    text = extract_text_from_choice(choices[0])
    parsed = parse_json_block(text)
    label = str(parsed.get("label", "")).strip()
    confidence_raw = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    return label, confidence, text


def call_google_classifier(
    api_key: str,
    endpoint: str,
    image_path: Path,
    prompt_text: str,
) -> tuple[str, float, str]:
    mime_type = detect_mime_type(image_path)
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": encode_image_base64(image_path),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {"temperature": 0.0},
    }
    params = {"key": api_key}
    response = requests.post(endpoint, params=params, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        msg = "No candidates returned by Google AI API."
        raise ValueError(msg)
    content = candidates[0].get("content", {}) or {}
    parts = content.get("parts") or []
    text = ""
    if parts:
        text = "".join(
            part.get("text", "") for part in parts if isinstance(part, dict)
        )
    if not text:
        text = candidates[0].get("output", "")
    if not text and isinstance(content, dict):
        text = content.get("text", "")
    parsed = parse_json_block(text)
    label = str(parsed.get("label", "")).strip()
    confidence_raw = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    return label, confidence, text


def load_model(model_path: Path) -> torch.nn.Module:
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model


def run_local_classification(
    images: list[Path],
    search_dir: Path,
    output_dir: Path,
    classes: list[str],
    threshold: float,
    model_path: Path,
) -> tuple[list[dict], dict]:
    model = load_model(model_path)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    log_entries: list[dict] = []
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        batch = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence_tensor, class_idx = torch.max(probabilities, dim=0)

        confidence = float(confidence_tensor)
        label = classes[int(class_idx)]
        assigned_label = label
        below_threshold = confidence < threshold
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
                "below_threshold": below_threshold,
            }
        )

    metadata = {
        "mode": "local",
        "provider": "local",
        "model_path": str(model_path),
        "labels": classes,
        "confidence_threshold": threshold,
    }
    return log_entries, metadata


def run_remote_classification(
    provider: str,
    images: list[Path],
    search_dir: Path,
    output_dir: Path,
    classes: list[str],
    threshold: float,
    config: dict,
) -> tuple[list[dict], dict]:
    prompt_template = config.get("api_prompt_template")
    bias_template = config.get("bias_prompt_template", "").strip()
    prompt_text = build_labels_prompt(classes, prompt_template)
    counts: dict[str, int] = dict.fromkeys(classes, 0)
    label_lookup = {label.lower(): label for label in classes}

    provider_normalized = provider.lower()
    if provider_normalized in {"mistral", "api"}:
        api_key = (
            config.get("mistral_api_key") or os.getenv("MISTRAL_API_KEY") or ""
        ).strip()
        if not api_key:
            msg = (
                "Missing Mistral API key. Set 'mistral_api_key' in config.yaml"
                " or the MISTRAL_API_KEY environment variable."
            )
            raise RuntimeError(msg)
        model_name = config.get("mistral_model", "pixtral-12b-latest")
        endpoint = config.get(
            "mistral_endpoint", "https://api.mistral.ai/v1/chat/completions"
        )

        def classify(image_path: Path, prompt: str) -> tuple[str, float, str]:
            return call_mistral_classifier(
                api_key=api_key,
                endpoint=endpoint,
                model_name=model_name,
                image_path=image_path,
                prompt_text=prompt,
            )

        metadata_base = {
            "provider": "mistral",
            "model": model_name,
            "endpoint": endpoint,
        }
    elif provider_normalized == "openai":
        api_key = (
            config.get("openai_api_key") or os.getenv("OPENAI_API_KEY") or ""
        ).strip()
        if not api_key:
            msg = (
                "Missing OpenAI API key. Set 'openai_api_key' in config.yaml"
                " or the OPENAI_API_KEY environment variable."
            )
            raise RuntimeError(msg)
        model_name = config.get("openai_model", "gpt-4o-mini")
        endpoint = config.get(
            "openai_endpoint", "https://api.openai.com/v1/chat/completions"
        )

        def classify(image_path: Path, prompt: str) -> tuple[str, float, str]:
            return call_openai_classifier(
                api_key=api_key,
                endpoint=endpoint,
                model=model_name,
                image_path=image_path,
                prompt_text=prompt,
            )

        metadata_base = {
            "provider": "openai",
            "model": model_name,
            "endpoint": endpoint,
        }
    elif provider_normalized == "google":
        api_key = (
            config.get("google_api_key") or os.getenv("GOOGLE_API_KEY") or ""
        ).strip()
        if not api_key:
            msg = (
                "Missing Google AI API key. Set 'google_api_key' in "
                "config.yaml or the GOOGLE_API_KEY environment variable."
            )
            raise RuntimeError(msg)
        model_name = config.get("google_model", "gemini-1.5-flash")
        endpoint_template = config.get(
            "google_endpoint",
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        )
        endpoint = endpoint_template.format(model=model_name)

        def classify(image_path: Path, prompt: str) -> tuple[str, float, str]:
            return call_google_classifier(
                api_key=api_key,
                endpoint=endpoint,
                image_path=image_path,
                prompt_text=prompt,
            )

        metadata_base = {
            "provider": "google",
            "model": model_name,
            "endpoint": endpoint,
        }
    elif provider_normalized in {"grok", "xai"}:
        api_key = (
            config.get("xai_api_key") or os.getenv("XAI_API_KEY") or ""
        ).strip()
        if not api_key:
            msg = (
                "Missing xAI API key. Set 'xai_api_key' in config.yaml "
                "or the XAI_API_KEY environment variable."
            )
            raise RuntimeError(msg)
        model_name = config.get("xai_model", "grok-4-fast-reasoning")
        endpoint = config.get(
            "xai_endpoint", "https://api.x.ai/v1/chat/completions"
        )

        def classify(image_path: Path, prompt: str) -> tuple[str, float, str]:
            return call_xai_classifier(
                api_key=api_key,
                endpoint=endpoint,
                model=model_name,
                image_path=image_path,
                prompt_text=prompt,
            )

        metadata_base = {
            "provider": "grok",
            "model": model_name,
            "endpoint": endpoint,
        }
    else:
        msg = f"Unsupported remote classification provider '{provider}'."
        raise ValueError(msg)

    log_entries: list[dict] = []
    for image_path in images:
        raw_label = ""
        confidence = 0.0
        raw_response = ""
        fallback_reason = None
        counts_snapshot = dict(counts)
        counts_json = json.dumps(counts_snapshot, ensure_ascii=False)
        prompt_with_counts = prompt_text
        if bias_template:
            bias_snippet = bias_template.replace("{counts_json}", counts_json)
            prompt_with_counts = f"{prompt_text}\n\n{bias_snippet}"
        try:
            raw_label, confidence, raw_response = classify(
                image_path, prompt_with_counts
            )
        except requests.HTTPError as exc:
            fallback_reason = "api_error"
            detail = ""
            if exc.response is not None:
                try:
                    detail = exc.response.text
                except requests.RequestException:
                    detail = ""
            print(
                "Warning: API classification failed for "
                f"{image_path.name}: {exc}. "
                f"{detail}"
            )
        except requests.RequestException as exc:
            fallback_reason = "api_error"
            print(
                "Warning: API classification failed for "
                f"{image_path.name}: {exc}"
            )

        normalized_label = ""
        if raw_label:
            normalized_label = label_lookup.get(raw_label.strip().lower(), "")
            if not normalized_label:
                fallback_reason = fallback_reason or "unrecognized_label"
        if not normalized_label:
            normalized_label = min(
                counts.items(), key=lambda item: (item[1], item[0])
            )[0]

        assigned_label = normalized_label
        below_threshold = confidence < threshold
        destination = copy_to_label(image_path, assigned_label, output_dir)

        # Update counts after assignment to drive the next prompt
        counts[assigned_label] = counts.get(assigned_label, 0) + 1

        log_entries.append(
            {
                "classifier": metadata_base["provider"],
                "file": str(image_path.relative_to(search_dir)),
                "model_label": raw_label or normalized_label,
                "predicted_label": normalized_label,
                "assigned_label": assigned_label,
                "confidence": float(confidence),
                "copied_to": str(destination.relative_to(output_dir)),
                "model_response": raw_response,
                "below_threshold": below_threshold,
                "fallback_reason": fallback_reason,
                "counts_json": counts_json,
                "counts_snapshot": counts_snapshot,
                "prompt_with_counts": prompt_with_counts,
            }
        )

    metadata = {
        **metadata_base,
        "labels": classes,
        "confidence_threshold": threshold,
        "prompt_template": prompt_template,
        "bias_prompt_template": bias_template,
        "final_counts": dict(counts),
    }
    return log_entries, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify images and sort by emotion label."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--input-dir", help="Override input directory from config"
    )
    parser.add_argument(
        "--processed-dir", help="Override processed directory from config"
    )
    parser.add_argument(
        "--output-dir", help="Override output directory from config"
    )
    parser.add_argument("--model-path", help="Override model path from config")
    parser.add_argument(
        "--mode",
        choices=["mistral", "openai", "google", "grok", "local", "api"],
        help="Override classification_mode from config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    mode_value = (
        args.mode or config.get("classification_mode", "mistral")
    ).lower()
    if mode_value == "api":
        mode_value = "mistral"
    if mode_value not in {"local", "mistral", "openai", "google", "grok"}:
        msg = (
            f"Unsupported classification_mode '{mode_value}'. "
            "Expected one of: local, mistral, openai, google, grok."
        )
        raise ValueError(msg)

    input_dir = (
        resolve_path(args.input_dir)
        if args.input_dir
        else resolve_path(config["input_dir"])
    )
    processed_dir_config = args.processed_dir or config.get("processed_dir")
    processed_dir = (
        resolve_path(processed_dir_config) if processed_dir_config else None
    )
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else resolve_path(config["output_dir"])
    )
    classes: list[str] = config["emotion_classes"]
    threshold: float = float(config.get("confidence_threshold", 0.5))

    search_dir = (
        processed_dir
        if processed_dir and processed_dir.exists()
        else input_dir
    )
    if not search_dir.exists():
        msg = f"Input directory {search_dir} does not exist."
        raise FileNotFoundError(msg)

    run_datetime = datetime.now(timezone.utc)

    images = collect_images(search_dir)
    if not images:
        print(f"No supported images found in {search_dir}. Nothing to do.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if mode_value == "local":
        model_path_value = args.model_path or config.get("model_path")
        if not model_path_value:
            msg = (
                "model_path must be provided in config.yaml or via "
                "--model-path when using local mode."
            )
            raise ValueError(msg)
        model_path = resolve_path(model_path_value)
        log_entries, metadata = run_local_classification(
            images, search_dir, output_dir, classes, threshold, model_path
        )
    else:
        log_entries, metadata = run_remote_classification(
            mode_value,
            images,
            search_dir,
            output_dir,
            classes,
            threshold,
            config,
        )

    if not isinstance(metadata, dict):
        metadata = {"details": metadata}
    metadata["post_process"] = {
        "status": "pending",
        "message": (
            "Run scripts/postprocess_sorted.py to rename/combine results."
        ),
        "sorted_dir": str(output_dir),
        "finished_dir": str(ROOT / "data" / "finished"),
        "recorded_at": run_datetime.isoformat(),
    }

    timestamp = run_datetime.isoformat()
    log_payload = {
        "run_id": timestamp,
        "timestamp_utc": timestamp,
        "mode": mode_value,
        "provider": metadata.get("provider")
        if isinstance(metadata, dict)
        else None,
        "config_path": str(Path(args.config)),
        "input_dir": str(search_dir),
        "processed_dir": str(processed_dir) if processed_dir else None,
        "output_dir": str(output_dir),
        "total_entries": len(log_entries),
        "metadata": metadata,
        "entries": log_entries,
    }

    log_path = output_dir / "classification_log.json"
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump(log_payload, fh, indent=2)

    print(
        f"Processed {len(log_entries)} images using {mode_value} classifier. "
        f"Log saved to {log_path}."
    )
    print(
        "When ready, run 'python scripts/postprocess_sorted.py' "
        "to rename, combine, and package results."
    )


if __name__ == "__main__":
    main()
