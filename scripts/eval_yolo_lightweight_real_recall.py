#!/usr/bin/env python
"""Lightweight streaming YOLO eval for bounded real-transfer probes."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
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
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=None,
        help="Optional YOLO prediction NMS IoU. Matching still uses --iou.",
    )
    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic YOLO NMS before matching predictions.",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--per-image-csv-out", default=None, type=Path)
    parser.add_argument(
        "--false-pred-jsonl-out",
        default=None,
        type=Path,
        help="Optional JSONL export of every false-positive prediction after filtering/NMS.",
    )
    parser.add_argument(
        "--ignore-pred-class",
        action="append",
        default=[],
        help="Prediction class id or name to drop before matching. Repeat or comma-separate for product-filtered views.",
    )
    parser.add_argument(
        "--class-min-conf",
        action="append",
        default=[],
        help="Class-specific prediction confidence floor, e.g. KHR_50000=0.30. Repeat or comma-separate.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"YOLO data YAML must be a mapping: {repo_rel(path)}")
    return config


def ignored_prediction_class_ids(raw_values: list[str], names: dict[int, str]) -> set[int]:
    ignored: set[int] = set()
    name_to_id = {name: class_id for class_id, name in names.items()}
    for raw_value in raw_values:
        for token in raw_value.replace(",", " ").split():
            if token in name_to_id:
                ignored.add(name_to_id[token])
                continue
            try:
                ignored.add(int(token))
            except ValueError as exc:
                raise SystemExit(f"unknown --ignore-pred-class value: {token}") from exc
    return ignored


def class_min_conf_by_id(raw_values: list[str], names: dict[int, str]) -> dict[int, float]:
    thresholds: dict[int, float] = {}
    name_to_id = {name: class_id for class_id, name in names.items()}
    for raw_value in raw_values:
        for token in raw_value.replace(",", " ").split():
            if "=" not in token:
                raise SystemExit(f"--class-min-conf must look like CLASS=THRESHOLD, got {token!r}")
            raw_class, raw_threshold = token.split("=", 1)
            if raw_class in name_to_id:
                class_id = name_to_id[raw_class]
            else:
                try:
                    class_id = int(raw_class)
                except ValueError as exc:
                    raise SystemExit(f"unknown --class-min-conf class: {raw_class}") from exc
            try:
                threshold = float(raw_threshold)
            except ValueError as exc:
                raise SystemExit(f"invalid --class-min-conf threshold for {raw_class}: {raw_threshold}") from exc
            thresholds[class_id] = threshold
    return thresholds


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


def source_group_for_image(image_path: Path) -> str:
    name = image_path.name.lower()
    prefixes = {
        "asian_currency_": "asian_currency",
        "billsbank_": "billsbank",
        "cambodia_currency_project_": "cambodia_currency_project",
        "cashcountingxl_": "cashcountingxl",
        "khmer_us_currency_": "khmer_us_currency",
        "usd_total_": "usd_total",
    }
    for prefix, group in prefixes.items():
        if name.startswith(prefix):
            return group
    return name.split("_", 1)[0] if "_" in name else "unknown"


def box_area_ratio(box: list[float], image_size: tuple[int, int]) -> float:
    width, height = image_size
    x1, y1, x2, y2 = box
    x1 = min(max(0.0, x1), float(width))
    x2 = min(max(0.0, x2), float(width))
    y1 = min(max(0.0, y1), float(height))
    y2 = min(max(0.0, y2), float(height))
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    total = max(1.0, float(width * height))
    return area / total


def update_area_stats(stats: dict[str, float | int], area_ratio: float) -> None:
    stats["count"] = int(stats["count"]) + 1
    stats["sum"] = float(stats["sum"]) + area_ratio
    stats["max"] = max(float(stats["max"]), area_ratio)
    if area_ratio >= 0.50:
        stats["large_ge_50pct"] = int(stats["large_ge_50pct"]) + 1
    if area_ratio >= 0.90:
        stats["full_ge_90pct"] = int(stats["full_ge_90pct"]) + 1


def finalize_area_stats(stats: dict[str, float | int]) -> dict[str, float | int]:
    count = int(stats["count"])
    return {
        "count": count,
        "mean_area_ratio": (float(stats["sum"]) / count) if count else None,
        "max_area_ratio": float(stats["max"]) if count else None,
        "large_ge_50pct": int(stats["large_ge_50pct"]),
        "full_ge_90pct": int(stats["full_ge_90pct"]),
    }


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
    ignored_pred_class_ids = ignored_prediction_class_ids(args.ignore_pred_class, names)
    class_conf_thresholds = class_min_conf_by_id(args.class_min_conf, names)
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
    raw_total_predictions = 0
    ignored_predictions = 0
    ignored_predictions_by_class: Counter[int] = Counter()
    ignored_predictions_by_source: Counter[str] = Counter()
    ignored_background_predictions_by_source: Counter[str] = Counter()
    ignored_predictions_by_source_class: dict[str, Counter[int]] = defaultdict(Counter)
    thresholded_predictions = 0
    thresholded_predictions_by_class: Counter[int] = Counter()
    thresholded_predictions_by_source: Counter[str] = Counter()
    thresholded_background_predictions_by_source: Counter[str] = Counter()
    thresholded_predictions_by_source_class: dict[str, Counter[int]] = defaultdict(Counter)
    examples_with_fp: list[dict[str, Any]] = []
    examples_with_fn: list[dict[str, Any]] = []
    examples_with_large_fp: list[dict[str, Any]] = []
    fp_examples_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    fn_examples_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    large_fp_examples_by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    false_prediction_rows: list[dict[str, Any]] = []
    prediction_area_stats: dict[str, float | int] = {
        "count": 0,
        "sum": 0.0,
        "max": 0.0,
        "large_ge_50pct": 0,
        "full_ge_90pct": 0,
    }
    fp_area_stats: dict[str, float | int] = {
        "count": 0,
        "sum": 0.0,
        "max": 0.0,
        "large_ge_50pct": 0,
        "full_ge_90pct": 0,
    }
    ignored_prediction_area_stats: dict[str, float | int] = {
        "count": 0,
        "sum": 0.0,
        "max": 0.0,
        "large_ge_50pct": 0,
        "full_ge_90pct": 0,
    }
    source_images: Counter[str] = Counter()
    source_background_images: Counter[str] = Counter()
    source_background_fp_images: Counter[str] = Counter()
    source_images_with_fp: Counter[str] = Counter()
    source_gt: Counter[str] = Counter()
    source_tp: Counter[str] = Counter()
    source_fp: Counter[str] = Counter()
    source_fn: Counter[str] = Counter()
    per_image_rows: list[dict[str, Any]] = []

    for batch in batched(images, max(1, args.batch)):
        results = model.predict(
            source=[str(path) for path in batch],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou if args.nms_iou is not None else 0.7,
            agnostic_nms=args.agnostic_nms,
            batch=len(batch),
            device=args.device,
            verbose=False,
        )
        for image_path, result in zip(batch, results):
            from PIL import Image

            with Image.open(image_path) as image:
                image_size = image.size
                labels = read_labels(label_path_for_image(image_path), image_size)
            source_group = source_group_for_image(image_path)
            source_images[source_group] += 1
            for label in labels:
                gt_by_class[int(label["class_id"])] += 1
                source_gt[source_group] += 1
            if not labels:
                background_images += 1
                source_background_images[source_group] += 1
            predictions: list[dict[str, Any]] = []
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy()
                conf = result.boxes.conf.cpu().numpy()
                for box, class_id, score in zip(xyxy, cls, conf):
                    raw_total_predictions += 1
                    class_id = int(class_id)
                    xyxy_box = [float(value) for value in box.tolist()]
                    area_ratio = box_area_ratio(xyxy_box, image_size)
                    if class_id in ignored_pred_class_ids:
                        ignored_predictions += 1
                        ignored_predictions_by_class[class_id] += 1
                        ignored_predictions_by_source[source_group] += 1
                        ignored_predictions_by_source_class[source_group][class_id] += 1
                        if not labels:
                            ignored_background_predictions_by_source[source_group] += 1
                        update_area_stats(ignored_prediction_area_stats, area_ratio)
                        continue
                    if class_id in class_conf_thresholds and float(score) < class_conf_thresholds[class_id]:
                        thresholded_predictions += 1
                        thresholded_predictions_by_class[class_id] += 1
                        thresholded_predictions_by_source[source_group] += 1
                        thresholded_predictions_by_source_class[source_group][class_id] += 1
                        if not labels:
                            thresholded_background_predictions_by_source[source_group] += 1
                        continue
                    update_area_stats(prediction_area_stats, area_ratio)
                    predictions.append(
                        {
                            "class_id": class_id,
                            "confidence": float(score),
                            "xyxy": xyxy_box,
                            "area_ratio": area_ratio,
                        }
                    )
            total_predictions += len(predictions)
            tp, fn, false_predictions = match_predictions(labels, predictions, args.iou)
            for prediction in false_predictions:
                fp_class_id = int(prediction["class_id"])
                fp_by_class[fp_class_id] += 1
                source_fp[source_group] += 1
                update_area_stats(fp_area_stats, float(prediction.get("area_ratio", 0.0)))
                false_prediction_rows.append(
                    {
                        "image": repo_rel(image_path),
                        "source_group": source_group,
                        "class_id": fp_class_id,
                        "class_name": names.get(fp_class_id, str(fp_class_id)),
                        "confidence": float(prediction.get("confidence", 0.0)),
                        "xyxy": prediction.get("xyxy", []),
                        "area_ratio": float(prediction.get("area_ratio", 0.0)),
                        "labels": labels,
                    }
                )
                if len(fp_examples_by_class[fp_class_id]) < 12:
                    fp_examples_by_class[fp_class_id].append(
                        {
                            "image": repo_rel(image_path),
                            "source_group": source_group,
                            "labels": labels,
                            "false_predictions": [prediction],
                        }
                    )
                if float(prediction.get("area_ratio", 0.0)) >= 0.50 and len(examples_with_large_fp) < 30:
                    examples_with_large_fp.append(
                        {
                            "image": repo_rel(image_path),
                            "source_group": source_group,
                            "labels": labels,
                            "false_prediction": prediction,
                        }
                    )
                if float(prediction.get("area_ratio", 0.0)) >= 0.50 and len(large_fp_examples_by_class[fp_class_id]) < 12:
                    large_fp_examples_by_class[fp_class_id].append(
                        {
                            "image": repo_rel(image_path),
                            "source_group": source_group,
                            "labels": labels,
                            "false_prediction": prediction,
                        }
                    )
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
            source_tp[source_group] += sum(per_class_tp.values())
            source_fn[source_group] += sum(per_class_fn.values())
            per_image_rows.append(
                {
                    "image": repo_rel(image_path),
                    "source_group": source_group,
                    "gt": len(labels),
                    "tp": sum(per_class_tp.values()),
                    "fn": sum(per_class_fn.values()),
                    "fp": len(false_predictions),
                    "total_predictions": len(predictions),
                    "label_classes": ";".join(
                        names.get(int(label["class_id"]), str(label["class_id"])) for label in labels
                    ),
                    "pred_classes": ";".join(
                        names.get(int(prediction["class_id"]), str(prediction["class_id"])) for prediction in predictions
                    ),
                    "false_pred_classes": ";".join(
                        names.get(int(prediction["class_id"]), str(prediction["class_id"]))
                        for prediction in false_predictions
                    ),
                    "max_prediction_confidence": max(
                        (float(prediction["confidence"]) for prediction in predictions),
                        default=0.0,
                    ),
                }
            )
            if false_predictions:
                images_with_fp += 1
                source_images_with_fp[source_group] += 1
                if not labels:
                    background_images_with_fp += 1
                    source_background_fp_images[source_group] += 1
                if len(examples_with_fp) < 30:
                    examples_with_fp.append(
                        {
                            "image": repo_rel(image_path),
                            "source_group": source_group,
                            "labels": labels,
                            "false_predictions": false_predictions[:5],
                        }
                    )
            if fn and len(examples_with_fn) < 30:
                missed_labels = [
                    label
                    for index, label in enumerate(labels)
                    if index not in matched_label_indices
                ]
                examples_with_fn.append(
                    {
                        "image": repo_rel(image_path),
                        "source_group": source_group,
                        "missed_labels": missed_labels,
                        "predictions": predictions[:5],
                    }
                )
            if fn:
                missed_labels = [
                    label
                    for index, label in enumerate(labels)
                    if index not in matched_label_indices
                ]
                for label in missed_labels:
                    fn_class_id = int(label["class_id"])
                    if len(fn_examples_by_class[fn_class_id]) >= 12:
                        continue
                    fn_examples_by_class[fn_class_id].append(
                        {
                            "image": repo_rel(image_path),
                            "source_group": source_group,
                            "missed_labels": [label],
                            "predictions": predictions[:5],
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
    per_source = {}
    for source_group in sorted(set(source_images) | set(source_gt) | set(source_fp) | set(source_fn)):
        gt = int(source_gt[source_group])
        tp = int(source_tp[source_group])
        fp = int(source_fp[source_group])
        fn = int(source_fn[source_group])
        per_source[source_group] = {
            "images": int(source_images[source_group]),
            "background_images": int(source_background_images[source_group]),
            "background_images_with_fp": int(source_background_fp_images[source_group]),
            "images_with_fp": int(source_images_with_fp[source_group]),
            "gt": gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": tp / gt if gt else None,
            "precision": tp / (tp + fp) if (tp + fp) else None,
        }
    summary = {
        "schema": "cashsnap_yolo_lightweight_recall_eval_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": repo_rel(resolve(args.model)),
        "data": repo_rel(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "nms_iou": args.nms_iou if args.nms_iou is not None else 0.7,
        "agnostic_nms": bool(args.agnostic_nms),
        "ignored_pred_class_ids": sorted(ignored_pred_class_ids),
        "ignored_pred_class_names": [names.get(class_id, str(class_id)) for class_id in sorted(ignored_pred_class_ids)],
        "class_min_conf": {
            names.get(class_id, str(class_id)): threshold for class_id, threshold in sorted(class_conf_thresholds.items())
        },
        "images": len(images),
        "background_images": background_images,
        "background_images_with_fp": background_images_with_fp,
        "images_with_fp": images_with_fp,
        "raw_total_predictions": raw_total_predictions,
        "ignored_predictions": ignored_predictions,
        "ignored_predictions_by_class": {
            names.get(class_id, str(class_id)): count for class_id, count in sorted(ignored_predictions_by_class.items())
        },
        "ignored_predictions_by_source": dict(sorted(ignored_predictions_by_source.items())),
        "ignored_background_predictions_by_source": dict(sorted(ignored_background_predictions_by_source.items())),
        "ignored_predictions_by_source_class": {
            source_group: {
                names.get(class_id, str(class_id)): count
                for class_id, count in sorted(class_counts.items())
            }
            for source_group, class_counts in sorted(ignored_predictions_by_source_class.items())
        },
        "thresholded_predictions": thresholded_predictions,
        "thresholded_predictions_by_class": {
            names.get(class_id, str(class_id)): count for class_id, count in sorted(thresholded_predictions_by_class.items())
        },
        "thresholded_predictions_by_source": dict(sorted(thresholded_predictions_by_source.items())),
        "thresholded_background_predictions_by_source": dict(sorted(thresholded_background_predictions_by_source.items())),
        "thresholded_predictions_by_source_class": {
            source_group: {
                names.get(class_id, str(class_id)): count
                for class_id, count in sorted(class_counts.items())
            }
            for source_group, class_counts in sorted(thresholded_predictions_by_source_class.items())
        },
        "total_predictions": total_predictions,
        "gt": int(total_gt),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "recall": total_tp / total_gt if total_gt else None,
        "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) else None,
        "per_class": per_class,
        "per_source": per_source,
        "prediction_area_stats": finalize_area_stats(prediction_area_stats),
        "ignored_prediction_area_stats": finalize_area_stats(ignored_prediction_area_stats),
        "fp_area_stats": finalize_area_stats(fp_area_stats),
        "fp_examples": examples_with_fp,
        "fn_examples": examples_with_fn,
        "large_fp_examples": examples_with_large_fp,
        "fp_examples_by_class": {
            names.get(class_id, str(class_id)): rows for class_id, rows in sorted(fp_examples_by_class.items())
        },
        "fn_examples_by_class": {
            names.get(class_id, str(class_id)): rows for class_id, rows in sorted(fn_examples_by_class.items())
        },
        "large_fp_examples_by_class": {
            names.get(class_id, str(class_id)): rows for class_id, rows in sorted(large_fp_examples_by_class.items())
        },
    }
    out_path = resolve(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.per_image_csv_out:
        per_image_csv = resolve(args.per_image_csv_out)
        per_image_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "image",
            "source_group",
            "gt",
            "tp",
            "fn",
            "fp",
            "total_predictions",
            "label_classes",
            "pred_classes",
            "false_pred_classes",
            "max_prediction_confidence",
        ]
        with per_image_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_image_rows)
    if args.false_pred_jsonl_out:
        false_pred_jsonl = resolve(args.false_pred_jsonl_out)
        false_pred_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with false_pred_jsonl.open("w", encoding="utf-8") as handle:
            for row in false_prediction_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        f"light_eval={repo_rel(out_path)} images={len(images)} "
        f"recall={fmt_metric(summary['recall'])} precision={fmt_metric(summary['precision'])} "
        f"bg_fp={background_images_with_fp}/{background_images}",
        flush=True,
    )


if __name__ == "__main__":
    main()
