#!/usr/bin/env python
"""Evaluate an ONNX ImageFolder classifier without importing Torch."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from local_runtime import configure_project_cache


configure_project_cache()


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--classes-json", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_json(path: Path) -> Any:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def ort_providers_without_tensorrt() -> list[str]:
    providers = [provider for provider in ort.get_available_providers() if provider != "TensorrtExecutionProvider"]
    return providers or ["CPUExecutionProvider"]


def class_dirs(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        raise SystemExit(f"split directory does not exist: {repo_rel(split_dir)}")
    return sorted(path for path in split_dir.iterdir() if path.is_dir())


def samples(split_dir: Path) -> tuple[list[tuple[Path, str]], list[str]]:
    dirs = class_dirs(split_dir)
    classes = [path.name for path in dirs]
    rows: list[tuple[Path, str]] = []
    for class_dir in dirs:
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append((image_path, class_dir.name))
    if not rows:
        raise SystemExit(f"no images found under {repo_rel(split_dir)}")
    return rows, classes


def preprocess(path: Path, image_size: int) -> np.ndarray:
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    with Image.open(path) as raw_image:
        image = ImageOps.exif_transpose(raw_image).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    return np.transpose(array, (2, 0, 1)).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> int:
    args = parse_args()
    model_path = resolve(args.model)
    data_dir = resolve(args.data)
    out_dir = resolve(args.out_dir)
    class_names = [str(name) for name in load_json(args.classes_json)]
    split_dir = data_dir / args.split
    sample_rows, data_classes = samples(split_dir)
    if data_classes != class_names:
        raise SystemExit(f"checkpoint/data classes differ: {class_names} != {data_classes}")

    session = ort.InferenceSession(str(model_path), providers=ort_providers_without_tensorrt())
    input_name = session.get_inputs()[0].name
    batch_size = max(1, int(args.batch))
    predictions: list[dict[str, Any]] = []
    confusion: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()

    for start in range(0, len(sample_rows), batch_size):
        batch_rows = sample_rows[start : start + batch_size]
        batch = np.stack([preprocess(path, args.image_size) for path, _class_name in batch_rows], axis=0)
        logits = session.run(None, {input_name: batch})[0]
        probabilities = softmax(logits)
        for (image_path, true_class), probs in zip(batch_rows, probabilities, strict=False):
            pred_index = int(probs.argmax())
            pred_class = class_names[pred_index]
            pred_conf = float(probs[pred_index])
            is_correct = pred_class == true_class
            totals[true_class] += 1
            correct[true_class] += int(is_correct)
            if not is_correct:
                confusion[(true_class, pred_class)] += 1
            predictions.append(
                {
                    "image_path": repo_rel(image_path),
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "pred_confidence": round(pred_conf, 6),
                    "correct": is_correct,
                }
            )

    per_class = []
    for class_name in class_names:
        total = int(totals[class_name])
        ok = int(correct[class_name])
        per_class.append(
            {
                "class_name": class_name,
                "total": total,
                "correct": ok,
                "accuracy": ok / total if total else None,
            }
        )
    confusion_rows = [
        {"true_class": true_name, "pred_class": pred_name, "count": count}
        for (true_name, pred_name), count in confusion.most_common()
    ]
    summary = {
        "schema": "cashsnap_imagefolder_onnx_classifier_eval_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": repo_rel(model_path),
        "classes_json": repo_rel(resolve(args.classes_json)),
        "data": repo_rel(data_dir),
        "split": args.split,
        "images": len(sample_rows),
        "accuracy": sum(correct.values()) / max(1, sum(totals.values())),
        "per_class": per_class,
        "top_confusions": confusion_rows[:40],
        "providers": session.get_providers(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(
        out_dir / "predictions.csv",
        predictions,
        ["image_path", "true_class", "pred_class", "pred_confidence", "correct"],
    )
    write_csv(out_dir / "confusion.csv", confusion_rows, ["true_class", "pred_class", "count"])
    print(f"eval_onnx_imagefolder={repo_rel(out_dir)} accuracy={summary['accuracy']:.4f} images={len(sample_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
