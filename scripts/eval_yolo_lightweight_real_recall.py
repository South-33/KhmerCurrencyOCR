#!/usr/bin/env python
"""Lightweight streaming YOLO eval for bounded real-transfer probes."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from local_runtime import configure_project_cache

configure_project_cache()

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", required=True, type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"YOLO data YAML must be a mapping: {repo_rel(path)}")
    return config


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    root = Path(str(config.get("path", "."))).expanduser()
    return root if root.is_absolute() else (config_path.parent / root).resolve()


def split_root(root: Path, split_path: str) -> Path:
    path = Path(split_path)
    return path if path.is_absolute() else root / path


def read_split_list(root: Path, split_path: str) -> list[Path]:
    list_path = split_root(root, split_path)
    images: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        image = Path(line)
        images.append(image if image.is_absolute() else root / image)
    return images


def split_images(config_path: Path, config: dict[str, Any], split: str) -> list[Path]:
    root = data_root(config_path, config)
    split_value = config.get(split)
    if split_value is None:
        raise SystemExit(f"{repo_rel(config_path)} has no split {split!r}")
    values = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for value in values:
        resolved = split_root(root, str(value))
        if resolved.suffix.lower() == ".txt":
            images.extend(read_split_list(root, str(value)))
        else:
            images.extend(sorted(path for path in resolved.glob("*") if path.suffix.lower() in IMAGE_EXTS))
    return images


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image_path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(label_path: Path, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    width, height = image_size
    labels: list[dict[str, Any]] = []
    if not label_path.exists():
        return labels
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} expected 5 YOLO fields")
        class_id = int(parts[0])
        cx, cy, bw, bh = [float(value) for value in parts[1:]]
        labels.append(
            {
                "class_id": class_id,
                "xyxy": [
                    (cx - bw / 2.0) * width,
                    (cy - bh / 2.0) * height,
                    (cx + bw / 2.0) * width,
                    (cy + bh / 2.0) * height,
                ],
            }
        )
    return labels


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def batched(items: list[Path], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def fmt_metric(value: float | None) -> str:
    return "none" if value is None else f"{value:.4f}"


def match_predictions(
    labels: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[int, int, list[dict[str, Any]]]:
    matched_labels: set[int] = set()
    false_predictions: list[dict[str, Any]] = []
    true_positive = 0
    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        best_index = -1
        best_iou = 0.0
        for index, label in enumerate(labels):
            if index in matched_labels or int(label["class_id"]) != int(prediction["class_id"]):
                continue
            score = box_iou(prediction["xyxy"], label["xyxy"])
            if score > best_iou:
                best_iou = score
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched_labels.add(best_index)
            true_positive += 1
        else:
            false_predictions.append({**prediction, "best_iou": best_iou})
    return true_positive, len(labels) - len(matched_labels), false_predictions


def main() -> None:
    args = parse_args()
    data_path = resolve(args.data)
    config = load_config(data_path)
    names = {int(key): str(value) for key, value in (config.get("names") or {}).items()}
    images = split_images(data_path, config, args.split)
    if args.max_images > 0:
        rng = random.Random(args.seed)
        images = rng.sample(images, min(args.max_images, len(images)))
    if not images:
        raise SystemExit("No images selected")

    model = YOLO(str(resolve(args.model)))
    gt_by_class: Counter[int] = Counter()
    tp_by_class: Counter[int] = Counter()
    fp_by_class: Counter[int] = Counter()
    fn_by_class: Counter[int] = Counter()
    background_images = 0
    background_images_with_fp = 0
    images_with_fp = 0
    total_predictions = 0
    examples_with_fp: list[dict[str, Any]] = []

    for batch in batched(images, max(1, args.batch)):
        results = model.predict(
            source=[str(path) for path in batch],
            imgsz=args.imgsz,
            conf=args.conf,
            batch=len(batch),
            device=args.device,
            verbose=False,
        )
        for image_path, result in zip(batch, results):
            from PIL import Image

            with Image.open(image_path) as image:
                labels = read_labels(label_path_for_image(image_path), image.size)
            for label in labels:
                gt_by_class[int(label["class_id"])] += 1
            if not labels:
                background_images += 1
            predictions: list[dict[str, Any]] = []
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy()
                conf = result.boxes.conf.cpu().numpy()
                for box, class_id, score in zip(xyxy, cls, conf):
                    predictions.append(
                        {
                            "class_id": int(class_id),
                            "confidence": float(score),
                            "xyxy": [float(value) for value in box.tolist()],
                        }
                    )
            total_predictions += len(predictions)
            tp, fn, false_predictions = match_predictions(labels, predictions, args.iou)
            for prediction in false_predictions:
                fp_by_class[int(prediction["class_id"])] += 1
            matched_by_class = Counter()
            for label in labels:
                matched_by_class[int(label["class_id"])] += 1
            # Recompute TP/FN by class greedily for transparent per-class stats.
            per_class_tp, per_class_fn = Counter(), Counter()
            matched_label_indices: set[int] = set()
            for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
                best_index = -1
                best_iou = 0.0
                for index, label in enumerate(labels):
                    if index in matched_label_indices or int(label["class_id"]) != int(prediction["class_id"]):
                        continue
                    score = box_iou(prediction["xyxy"], label["xyxy"])
                    if score > best_iou:
                        best_iou = score
                        best_index = index
                if best_index >= 0 and best_iou >= args.iou:
                    matched_label_indices.add(best_index)
                    per_class_tp[int(labels[best_index]["class_id"])] += 1
            for index, label in enumerate(labels):
                if index not in matched_label_indices:
                    per_class_fn[int(label["class_id"])] += 1
            tp_by_class.update(per_class_tp)
            fn_by_class.update(per_class_fn)
            if false_predictions:
                images_with_fp += 1
                if not labels:
                    background_images_with_fp += 1
                if len(examples_with_fp) < 30:
                    examples_with_fp.append(
                        {
                            "image": repo_rel(image_path),
                            "labels": labels,
                            "false_predictions": false_predictions[:5],
                        }
                    )

    per_class = {}
    for class_id in sorted(set(gt_by_class) | set(tp_by_class) | set(fp_by_class) | set(fn_by_class)):
        gt = int(gt_by_class[class_id])
        tp = int(tp_by_class[class_id])
        fp = int(fp_by_class[class_id])
        fn = int(fn_by_class[class_id])
        per_class[names.get(class_id, str(class_id))] = {
            "gt": gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": tp / gt if gt else None,
            "precision": tp / (tp + fp) if (tp + fp) else None,
        }
    total_gt = sum(gt_by_class.values())
    total_tp = sum(tp_by_class.values())
    total_fp = sum(fp_by_class.values())
    total_fn = sum(fn_by_class.values())
    summary = {
        "schema": "cashsnap_yolo_lightweight_recall_eval_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": repo_rel(resolve(args.model)),
        "data": repo_rel(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "images": len(images),
        "background_images": background_images,
        "background_images_with_fp": background_images_with_fp,
        "images_with_fp": images_with_fp,
        "total_predictions": total_predictions,
        "gt": int(total_gt),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "recall": total_tp / total_gt if total_gt else None,
        "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) else None,
        "per_class": per_class,
        "fp_examples": examples_with_fp,
    }
    out_path = resolve(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"light_eval={repo_rel(out_path)} images={len(images)} "
        f"recall={fmt_metric(summary['recall'])} precision={fmt_metric(summary['precision'])} "
        f"bg_fp={background_images_with_fp}/{background_images}",
        flush=True,
    )


if __name__ == "__main__":
    main()
