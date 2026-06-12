#!/usr/bin/env python
"""Evaluate a crop-level proposal gate on YOLO detections.

This is a diagnostic harness for the "detector proposals + banknote/background
gate" branch. The gate only rejects proposals; kept proposals retain the
detector denomination class.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from local_runtime import configure_project_cache

configure_project_cache()

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_VALUES = {
    "USD_1": ("USD", 1.0),
    "USD_5": ("USD", 5.0),
    "USD_10": ("USD", 10.0),
    "USD_20": ("USD", 20.0),
    "USD_50": ("USD", 50.0),
    "USD_100": ("USD", 100.0),
    "KHR_500": ("KHR", 500.0),
    "KHR_1000": ("KHR", 1000.0),
    "KHR_2000": ("KHR", 2000.0),
    "KHR_5000": ("KHR", 5000.0),
    "KHR_10000": ("KHR", 10000.0),
    "KHR_20000": ("KHR", 20000.0),
    "KHR_50000": ("KHR", 50000.0),
}


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
    parser.add_argument("--detector", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--det-iou", type=float, default=0.50)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument(
        "--detector-class-min-conf",
        action="append",
        default=[],
        help="Detector class-specific confidence floor as CLASS=THRESHOLD. Repeat or comma-separate.",
    )
    parser.add_argument(
        "--detector-class-mode",
        choices=["data", "banknote"],
        default="data",
        help=(
            "Use 'data' when detector class IDs are data-label class IDs. Use 'banknote' for "
            "class-agnostic note detectors whose classes should be ignored before reclassification."
        ),
    )
    parser.add_argument("--crop-padding", type=float, default=0.06)
    parser.add_argument(
        "--reject-class",
        default="background",
        help="Gate class name that rejects detector proposals when it is the top class.",
    )
    parser.add_argument(
        "--reject-min-conf",
        type=float,
        default=0.0,
        help="Minimum top-class probability for --reject-class to reject a proposal.",
    )
    parser.add_argument(
        "--reject-max-det-conf",
        type=float,
        default=None,
        help=(
            "Only allow --reject-class to reject detector proposals whose detector confidence is "
            "at or below this value. Omit to allow rejection at any detector confidence."
        ),
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument(
        "--per-image-json-out",
        type=Path,
        default=None,
        help="Optional compact per-image stage rows for count/value disagreement analysis.",
    )
    parser.add_argument(
        "--proposal-json-out",
        type=Path,
        default=None,
        help="Optional detector proposal rows with gate scores for offline reject-threshold sweeps.",
    )
    parser.add_argument(
        "--reclassifier",
        type=Path,
        default=None,
        help="Optional MobileNetV3 checkpoint that reassigns kept proposal denomination classes.",
    )
    parser.add_argument(
        "--reclassifier-min-conf",
        type=float,
        default=0.0,
        help="Only override kept proposal classes when the reclassifier top probability is at least this value.",
    )
    parser.add_argument(
        "--reclassifier-block-class",
        action="append",
        default=[],
        help="Class name that the reclassifier is not allowed to override into. Repeatable or comma-separate.",
    )
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value != "auto":
        if value.isdigit():
            return torch.device(f"cuda:{value}")
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"YOLO data YAML must be a mapping: {repo_rel(path)}")
    return config


def parse_names(config: dict[str, Any]) -> dict[int, str]:
    raw_names = config.get("names") or {}
    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    raise SystemExit("YOLO data names must be a list or mapping")


def parse_name_set(raw_values: list[str]) -> set[str]:
    names: set[str] = set()
    for raw_value in raw_values:
        names.update(token for token in raw_value.replace(",", " ").split() if token)
    return names


def parse_class_min_conf(raw_values: list[str], names: dict[int, str]) -> dict[int, float]:
    name_to_id = {name: class_id for class_id, name in names.items()}
    thresholds: dict[int, float] = {}
    for raw_value in raw_values:
        for token in raw_value.replace(",", " ").split():
            if not token:
                continue
            if "=" not in token:
                raise SystemExit(f"--detector-class-min-conf expected CLASS=THRESHOLD, got {token!r}")
            raw_name, raw_threshold = token.split("=", 1)
            class_name = raw_name.strip()
            if class_name not in name_to_id:
                raise SystemExit(f"unknown detector class for --detector-class-min-conf: {class_name}")
            try:
                threshold = float(raw_threshold)
            except ValueError as exc:
                raise SystemExit(f"invalid threshold for {class_name}: {raw_threshold!r}") from exc
            if threshold < 0.0 or threshold > 1.0:
                raise SystemExit(f"threshold for {class_name} must be between 0 and 1: {threshold}")
            thresholds[name_to_id[class_name]] = threshold
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


def batched(items: list[Path], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


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


def box_area_ratio(box: list[float], image_size: tuple[int, int]) -> float:
    width, height = image_size
    x1, y1, x2, y2 = box
    x1 = min(max(0.0, x1), float(width))
    x2 = min(max(0.0, x2), float(width))
    y1 = min(max(0.0, y1), float(height))
    y2 = min(max(0.0, y2), float(height))
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / max(1.0, float(width * height))


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def best_iou_prediction(label: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any] | None:
    best_prediction: dict[str, Any] | None = None
    best_iou = 0.0
    for prediction in predictions:
        score = box_iou(prediction["xyxy"], label["xyxy"])
        if score > best_iou:
            best_iou = score
            best_prediction = prediction
    if best_prediction is None:
        return None
    label_area = box_area(label["xyxy"])
    prediction_area = box_area(best_prediction["xyxy"])
    return {
        **best_prediction,
        "best_iou": best_iou,
        "gt_to_prediction_area_ratio": (label_area / prediction_area) if prediction_area > 0 else None,
        "prediction_to_gt_area_ratio": (prediction_area / label_area) if label_area > 0 else None,
    }


def miss_iou_bucket(best_prediction: dict[str, Any] | None, match_iou: float) -> str:
    if best_prediction is None:
        return "no_prediction"
    best_iou = float(best_prediction.get("best_iou", 0.0))
    if best_iou >= match_iou:
        return f"iou_ge_{match_iou:.2f}"
    if best_iou >= 0.25:
        return "iou_0.25_to_match"
    if best_iou >= 0.10:
        return "iou_0.10_to_0.25"
    if best_iou > 0.0:
        return "iou_lt_0.10"
    return "no_overlap"


def no_box_cause(
    *,
    stage_best: dict[str, Any] | None,
    reference_best: dict[str, Any] | None,
    match_iou: float,
) -> str:
    if reference_best is not None and float(reference_best.get("best_iou", 0.0)) >= match_iou:
        return "reference_had_box_stage_lost"
    return "reference_" + miss_iou_bucket(reference_best, match_iou)


def crop_with_padding(image: Image.Image, box: list[float], padding: float) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return image.crop(
        (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(width, int(x2 + pad_x)),
            min(height, int(y2 + pad_y)),
        )
    ).copy()


def image_from_yolo_result(result: Any, fallback_path: Path) -> Image.Image:
    orig_img = getattr(result, "orig_img", None)
    if orig_img is not None and getattr(orig_img, "ndim", 0) == 3 and orig_img.shape[2] >= 3:
        return Image.fromarray(orig_img[:, :, :3][:, :, ::-1].copy())
    with Image.open(fallback_path) as image:
        return image.convert("RGB")


def build_gate_model(class_count: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, class_count)
    return model


def load_gate(path: Path, device: torch.device) -> tuple[nn.Module, list[str], int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    class_names = [str(name) for name in checkpoint["classes"]]
    image_size = int(checkpoint.get("image_size", 224))
    model = build_gate_model(len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names, image_size


def gate_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def classify_crops(
    gate: nn.Module,
    transform: transforms.Compose,
    crops: list[Image.Image],
    class_names: list[str],
    device: torch.device,
    *,
    class_key: str = "gate_class",
    conf_key: str = "gate_conf",
    probs_key: str = "gate_probs",
) -> list[dict[str, Any]]:
    if not crops:
        return []
    batch = torch.stack([transform(crop) for crop in crops]).to(device)
    with torch.no_grad():
        probs = torch.softmax(gate(batch), dim=1).detach().cpu()
    rows: list[dict[str, Any]] = []
    for prob in probs:
        best_prob, best_index = prob.max(dim=0)
        rows.append(
            {
                class_key: class_names[int(best_index)],
                conf_key: float(best_prob),
                probs_key: {class_names[index]: float(prob[index]) for index in range(len(class_names))},
            }
        )
    return rows


def match_predictions(
    labels: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
    *,
    match_classes: bool = True,
) -> tuple[set[int], list[dict[str, Any]]]:
    matched_labels: set[int] = set()
    false_predictions: list[dict[str, Any]] = []
    for prediction in sorted(predictions, key=lambda item: float(item["confidence"]), reverse=True):
        best_index = -1
        best_iou = 0.0
        for index, label in enumerate(labels):
            if index in matched_labels:
                continue
            if match_classes and int(label["class_id"]) != int(prediction["class_id"]):
                continue
            score = box_iou(prediction["xyxy"], label["xyxy"])
            if score > best_iou:
                best_iou = score
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched_labels.add(best_index)
        else:
            false_predictions.append({**prediction, "best_iou": best_iou})
    return matched_labels, false_predictions


def match_predictions_any_class(
    labels: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[int, dict[str, Any]]:
    matched_labels: dict[int, dict[str, Any]] = {}
    for prediction in sorted(predictions, key=lambda item: float(item["confidence"]), reverse=True):
        best_index = -1
        best_iou = 0.0
        for index, label in enumerate(labels):
            if index in matched_labels:
                continue
            score = box_iou(prediction["xyxy"], label["xyxy"])
            if score > best_iou:
                best_iou = score
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched_labels[best_index] = {**prediction, "best_iou": best_iou}
    return matched_labels


def best_same_class_iou(labels: list[dict[str, Any]], prediction: dict[str, Any]) -> float:
    return max(
        (
            box_iou(prediction["xyxy"], label["xyxy"])
            for label in labels
            if int(label["class_id"]) == int(prediction["class_id"])
        ),
        default=0.0,
    )


def best_match_iou(labels: list[dict[str, Any]], prediction: dict[str, Any], *, match_classes: bool) -> float:
    return max(
        (
            box_iou(prediction["xyxy"], label["xyxy"])
            for label in labels
            if not match_classes or int(label["class_id"]) == int(prediction["class_id"])
        ),
        default=0.0,
    )


def init_area_stats() -> dict[str, float | int]:
    return {"count": 0, "sum": 0.0, "max": 0.0, "large_ge_50pct": 0, "full_ge_90pct": 0}


def update_area_stats(stats: dict[str, float | int], area_ratio: float) -> None:
    stats["count"] = int(stats["count"]) + 1
    stats["sum"] = float(stats["sum"]) + area_ratio
    stats["max"] = max(float(stats["max"]), area_ratio)
    if area_ratio >= 0.50:
        stats["large_ge_50pct"] = int(stats["large_ge_50pct"]) + 1
    if area_ratio >= 0.90:
        stats["full_ge_90pct"] = int(stats["full_ge_90pct"]) + 1


def finalize_area_stats(stats: dict[str, float | int]) -> dict[str, float | int | None]:
    count = int(stats["count"])
    return {
        "count": count,
        "mean_area_ratio": (float(stats["sum"]) / count) if count else None,
        "max_area_ratio": float(stats["max"]) if count else None,
        "large_ge_50pct": int(stats["large_ge_50pct"]),
        "full_ge_90pct": int(stats["full_ge_90pct"]),
    }


class Metrics:
    def __init__(
        self,
        names: dict[int, str],
        match_iou: float,
        *,
        match_classes: bool = True,
        value_errors: bool = True,
    ) -> None:
        self.names = names
        self.match_iou = match_iou
        self.match_classes = match_classes
        self.value_errors = value_errors
        self.gt_by_class: Counter[int] = Counter()
        self.tp_by_class: Counter[int] = Counter()
        self.fp_by_class: Counter[int] = Counter()
        self.fn_by_class: Counter[int] = Counter()
        self.background_images = 0
        self.background_images_with_fp = 0
        self.images_with_fp = 0
        self.total_predictions = 0
        self.source_images: Counter[str] = Counter()
        self.source_background_images: Counter[str] = Counter()
        self.source_background_fp_images: Counter[str] = Counter()
        self.source_images_with_fp: Counter[str] = Counter()
        self.source_gt: Counter[str] = Counter()
        self.source_tp: Counter[str] = Counter()
        self.source_fp: Counter[str] = Counter()
        self.source_fn: Counter[str] = Counter()
        self.localization_tp_by_class: Counter[int] = Counter()
        self.wrong_class_fn_by_class: Counter[int] = Counter()
        self.missed_no_box_fn_by_class: Counter[int] = Counter()
        self.same_class_conflict_fn_by_class: Counter[int] = Counter()
        self.source_localization_tp: Counter[str] = Counter()
        self.source_wrong_class_fn: Counter[str] = Counter()
        self.source_missed_no_box_fn: Counter[str] = Counter()
        self.source_same_class_conflict_fn: Counter[str] = Counter()
        self.source_class_gt: Counter[str] = Counter()
        self.source_class_tp: Counter[str] = Counter()
        self.source_class_fn: Counter[str] = Counter()
        self.source_class_localization_tp: Counter[str] = Counter()
        self.source_class_wrong_class_fn: Counter[str] = Counter()
        self.source_class_missed_no_box_fn: Counter[str] = Counter()
        self.source_class_same_class_conflict_fn: Counter[str] = Counter()
        self.wrong_class_pairs: Counter[str] = Counter()
        self.no_box_cause_by_class: Counter[str] = Counter()
        self.no_box_cause_by_source: Counter[str] = Counter()
        self.no_box_cause_by_source_class: Counter[str] = Counter()
        self.miss_best_stage_iou_buckets: Counter[str] = Counter()
        self.miss_best_reference_iou_buckets: Counter[str] = Counter()
        self.prediction_area_stats = init_area_stats()
        self.fp_area_stats = init_area_stats()
        self.examples_with_fp: list[dict[str, Any]] = []
        self.examples_with_fn: list[dict[str, Any]] = []
        self.examples_with_wrong_class_fn: list[dict[str, Any]] = []
        self.examples_with_no_box_fn: list[dict[str, Any]] = []
        self.count_abs_error_sum = 0.0
        self.usd_abs_error_sum = 0.0
        self.khr_abs_error_sum = 0.0
        self.exact_count_images = 0
        self.exact_value_images = 0
        self.per_image_rows: list[dict[str, Any]] = []

    def value_totals(self, rows: list[dict[str, Any]]) -> tuple[float, float]:
        usd_total = 0.0
        khr_total = 0.0
        for row in rows:
            class_name = self.names.get(int(row["class_id"]), str(row["class_id"]))
            value = CLASS_VALUES.get(class_name)
            if value is None:
                continue
            currency, amount = value
            if currency == "USD":
                usd_total += amount
            elif currency == "KHR":
                khr_total += amount
        return usd_total, khr_total

    def add_image(
        self,
        image_path: Path,
        source_group: str,
        labels: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
        reference_predictions: list[dict[str, Any]] | None = None,
    ) -> None:
        if reference_predictions is None:
            reference_predictions = predictions
        self.source_images[source_group] += 1
        if not labels:
            self.background_images += 1
            self.source_background_images[source_group] += 1
        for label in labels:
            class_id = int(label["class_id"])
            class_name = self.names.get(class_id, str(class_id))
            source_class_key = f"{source_group}|{class_name}"
            self.gt_by_class[class_id] += 1
            self.source_gt[source_group] += 1
            self.source_class_gt[source_class_key] += 1
        for prediction in predictions:
            update_area_stats(self.prediction_area_stats, float(prediction.get("area_ratio", 0.0)))
        self.total_predictions += len(predictions)
        count_error = len(predictions) - len(labels)
        self.count_abs_error_sum += abs(float(count_error))
        if count_error == 0:
            self.exact_count_images += 1
        gt_usd: float | None = None
        gt_khr: float | None = None
        pred_usd: float | None = None
        pred_khr: float | None = None
        usd_error: float | None = None
        khr_error: float | None = None
        exact_value = None
        if self.value_errors:
            gt_usd, gt_khr = self.value_totals(labels)
            pred_usd, pred_khr = self.value_totals(predictions)
            usd_error = pred_usd - gt_usd
            khr_error = pred_khr - gt_khr
            self.usd_abs_error_sum += abs(usd_error)
            self.khr_abs_error_sum += abs(khr_error)
            exact_value = usd_error == 0 and khr_error == 0
            if exact_value:
                self.exact_value_images += 1

        matched_label_indices, false_predictions = match_predictions(
            labels,
            predictions,
            self.match_iou,
            match_classes=self.match_classes,
        )
        any_class_matches = match_predictions_any_class(labels, predictions, self.match_iou)
        per_class_tp: Counter[int] = Counter()
        per_class_fn: Counter[int] = Counter()
        per_class_localization_tp: Counter[int] = Counter()
        per_class_wrong_class_fn: Counter[int] = Counter()
        per_class_missed_no_box_fn: Counter[int] = Counter()
        per_class_same_class_conflict_fn: Counter[int] = Counter()
        for index, label in enumerate(labels):
            class_id = int(label["class_id"])
            class_name = self.names.get(class_id, str(class_id))
            source_class_key = f"{source_group}|{class_name}"
            any_class_prediction = any_class_matches.get(index)
            if any_class_prediction is not None:
                per_class_localization_tp[class_id] += 1
                self.source_class_localization_tp[source_class_key] += 1
            if index in matched_label_indices:
                per_class_tp[class_id] += 1
                self.source_class_tp[source_class_key] += 1
            else:
                per_class_fn[class_id] += 1
                self.source_class_fn[source_class_key] += 1
                stage_best = best_iou_prediction(label, predictions)
                reference_best = best_iou_prediction(label, reference_predictions)
                self.miss_best_stage_iou_buckets[miss_iou_bucket(stage_best, self.match_iou)] += 1
                self.miss_best_reference_iou_buckets[miss_iou_bucket(reference_best, self.match_iou)] += 1
                if any_class_prediction is None:
                    per_class_missed_no_box_fn[class_id] += 1
                    self.source_class_missed_no_box_fn[source_class_key] += 1
                    cause = no_box_cause(
                        stage_best=stage_best,
                        reference_best=reference_best,
                        match_iou=self.match_iou,
                    )
                    self.no_box_cause_by_class[f"{class_name}|{cause}"] += 1
                    self.no_box_cause_by_source[f"{source_group}|{cause}"] += 1
                    self.no_box_cause_by_source_class[f"{source_group}|{class_name}|{cause}"] += 1
                    if len(self.examples_with_no_box_fn) < 30:
                        self.examples_with_no_box_fn.append(
                            {
                                "image": repo_rel(image_path),
                                "source_group": source_group,
                                "label": label,
                                "stage_best_prediction": stage_best,
                                "reference_best_prediction": reference_best,
                                "cause": cause,
                                "predictions": predictions[:5],
                            }
                        )
                elif int(any_class_prediction["class_id"]) == class_id:
                    per_class_same_class_conflict_fn[class_id] += 1
                    self.source_class_same_class_conflict_fn[source_class_key] += 1
                else:
                    per_class_wrong_class_fn[class_id] += 1
                    self.source_class_wrong_class_fn[source_class_key] += 1
                    gt_name = self.names.get(class_id, str(class_id))
                    pred_class_id = int(any_class_prediction["class_id"])
                    pred_name = self.names.get(pred_class_id, str(pred_class_id))
                    self.wrong_class_pairs[f"{gt_name}->{pred_name}"] += 1
                    if len(self.examples_with_wrong_class_fn) < 30:
                        self.examples_with_wrong_class_fn.append(
                            {
                                "image": repo_rel(image_path),
                                "source_group": source_group,
                                "label": label,
                                "matched_prediction": any_class_prediction,
                            }
                        )
        self.tp_by_class.update(per_class_tp)
        self.fn_by_class.update(per_class_fn)
        self.localization_tp_by_class.update(per_class_localization_tp)
        self.wrong_class_fn_by_class.update(per_class_wrong_class_fn)
        self.missed_no_box_fn_by_class.update(per_class_missed_no_box_fn)
        self.same_class_conflict_fn_by_class.update(per_class_same_class_conflict_fn)
        self.source_tp[source_group] += sum(per_class_tp.values())
        self.source_fn[source_group] += sum(per_class_fn.values())
        self.source_localization_tp[source_group] += sum(per_class_localization_tp.values())
        self.source_wrong_class_fn[source_group] += sum(per_class_wrong_class_fn.values())
        self.source_missed_no_box_fn[source_group] += sum(per_class_missed_no_box_fn.values())
        self.source_same_class_conflict_fn[source_group] += sum(per_class_same_class_conflict_fn.values())

        if false_predictions:
            self.images_with_fp += 1
            self.source_images_with_fp[source_group] += 1
            if not labels:
                self.background_images_with_fp += 1
                self.source_background_fp_images[source_group] += 1
            if len(self.examples_with_fp) < 30:
                self.examples_with_fp.append(
                    {
                        "image": repo_rel(image_path),
                        "source_group": source_group,
                        "labels": labels,
                        "false_predictions": false_predictions[:5],
                    }
                )
        for prediction in false_predictions:
            class_id = int(prediction["class_id"])
            self.fp_by_class[class_id] += 1
            self.source_fp[source_group] += 1
            update_area_stats(self.fp_area_stats, float(prediction.get("area_ratio", 0.0)))

        if len(labels) - len(matched_label_indices) and len(self.examples_with_fn) < 30:
            self.examples_with_fn.append(
                {
                    "image": repo_rel(image_path),
                    "source_group": source_group,
                    "missed_labels": [
                        label for index, label in enumerate(labels) if index not in matched_label_indices
                    ],
                    "predictions": predictions[:5],
                }
            )
        self.per_image_rows.append(
            {
                "image": repo_rel(image_path),
                "source_group": source_group,
                "labels": len(labels),
                "predictions": len(predictions),
                "tp": len(matched_label_indices),
                "fp": len(false_predictions),
                "fn": len(labels) - len(matched_label_indices),
                "count_error": count_error,
                "exact_count": count_error == 0,
                "gt_usd": gt_usd,
                "pred_usd": pred_usd,
                "usd_error": usd_error,
                "gt_khr": gt_khr,
                "pred_khr": pred_khr,
                "khr_error": khr_error,
                "exact_value": exact_value,
            }
        )

    def summary(self) -> dict[str, Any]:
        per_class = {}
        for class_id in sorted(set(self.gt_by_class) | set(self.tp_by_class) | set(self.fp_by_class) | set(self.fn_by_class)):
            gt = int(self.gt_by_class[class_id])
            tp = int(self.tp_by_class[class_id])
            fp = int(self.fp_by_class[class_id])
            fn = int(self.fn_by_class[class_id])
            localization_tp = int(self.localization_tp_by_class[class_id])
            per_class[self.names.get(class_id, str(class_id))] = {
                "gt": gt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "recall": tp / gt if gt else None,
                "precision": tp / (tp + fp) if (tp + fp) else None,
                "localization_tp": localization_tp,
                "localization_recall": localization_tp / gt if gt else None,
                "wrong_class_fn": int(self.wrong_class_fn_by_class[class_id]),
                "missed_no_box_fn": int(self.missed_no_box_fn_by_class[class_id]),
                "same_class_conflict_fn": int(self.same_class_conflict_fn_by_class[class_id]),
            }
        per_source = {}
        for source_group in sorted(set(self.source_images) | set(self.source_gt) | set(self.source_fp) | set(self.source_fn)):
            gt = int(self.source_gt[source_group])
            tp = int(self.source_tp[source_group])
            fp = int(self.source_fp[source_group])
            fn = int(self.source_fn[source_group])
            localization_tp = int(self.source_localization_tp[source_group])
            per_source[source_group] = {
                "images": int(self.source_images[source_group]),
                "background_images": int(self.source_background_images[source_group]),
                "background_images_with_fp": int(self.source_background_fp_images[source_group]),
                "images_with_fp": int(self.source_images_with_fp[source_group]),
                "gt": gt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "recall": tp / gt if gt else None,
                "precision": tp / (tp + fp) if (tp + fp) else None,
                "localization_tp": localization_tp,
                "localization_recall": localization_tp / gt if gt else None,
                "wrong_class_fn": int(self.source_wrong_class_fn[source_group]),
                "missed_no_box_fn": int(self.source_missed_no_box_fn[source_group]),
                "same_class_conflict_fn": int(self.source_same_class_conflict_fn[source_group]),
            }
        per_source_class = {}
        for key in sorted(
            set(self.source_class_gt)
            | set(self.source_class_tp)
            | set(self.source_class_fn)
            | set(self.source_class_localization_tp)
            | set(self.source_class_wrong_class_fn)
            | set(self.source_class_missed_no_box_fn)
            | set(self.source_class_same_class_conflict_fn)
        ):
            source_group, class_name = key.split("|", 1)
            gt = int(self.source_class_gt[key])
            tp = int(self.source_class_tp[key])
            fn = int(self.source_class_fn[key])
            localization_tp = int(self.source_class_localization_tp[key])
            per_source_class[key] = {
                "source_group": source_group,
                "class_name": class_name,
                "gt": gt,
                "tp": tp,
                "fn": fn,
                "recall": tp / gt if gt else None,
                "localization_tp": localization_tp,
                "localization_recall": localization_tp / gt if gt else None,
                "wrong_class_fn": int(self.source_class_wrong_class_fn[key]),
                "missed_no_box_fn": int(self.source_class_missed_no_box_fn[key]),
                "same_class_conflict_fn": int(self.source_class_same_class_conflict_fn[key]),
            }
        total_gt = sum(self.gt_by_class.values())
        total_tp = sum(self.tp_by_class.values())
        total_fp = sum(self.fp_by_class.values())
        total_fn = sum(self.fn_by_class.values())
        total_localization_tp = sum(self.localization_tp_by_class.values())
        total_wrong_class_fn = sum(self.wrong_class_fn_by_class.values())
        total_missed_no_box_fn = sum(self.missed_no_box_fn_by_class.values())
        total_same_class_conflict_fn = sum(self.same_class_conflict_fn_by_class.values())
        return {
            "images_with_fp": self.images_with_fp,
            "background_images": self.background_images,
            "background_images_with_fp": self.background_images_with_fp,
            "total_predictions": self.total_predictions,
            "match_classes": self.match_classes,
            "gt": int(total_gt),
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "recall": total_tp / total_gt if total_gt else None,
            "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) else None,
            "localization_tp": int(total_localization_tp),
            "localization_recall": total_localization_tp / total_gt if total_gt else None,
            "wrong_class_fn": int(total_wrong_class_fn),
            "missed_no_box_fn": int(total_missed_no_box_fn),
            "same_class_conflict_fn": int(total_same_class_conflict_fn),
            "wrong_class_pairs": dict(self.wrong_class_pairs.most_common(30)),
            "no_box_cause_by_class": dict(self.no_box_cause_by_class.most_common(60)),
            "no_box_cause_by_source": dict(self.no_box_cause_by_source.most_common(60)),
            "no_box_cause_by_source_class": dict(self.no_box_cause_by_source_class.most_common(80)),
            "miss_best_stage_iou_buckets": dict(self.miss_best_stage_iou_buckets.most_common()),
            "miss_best_reference_iou_buckets": dict(self.miss_best_reference_iou_buckets.most_common()),
            "per_class": per_class,
            "per_source": per_source,
            "per_source_class": per_source_class,
            "prediction_area_stats": finalize_area_stats(self.prediction_area_stats),
            "fp_area_stats": finalize_area_stats(self.fp_area_stats),
            "count_value_errors": {
                "mean_abs_count_error": self.count_abs_error_sum / max(1, self.source_image_count()),
                "mean_abs_usd_total_error": (
                    self.usd_abs_error_sum / max(1, self.source_image_count()) if self.value_errors else None
                ),
                "mean_abs_khr_total_error": (
                    self.khr_abs_error_sum / max(1, self.source_image_count()) if self.value_errors else None
                ),
                "exact_count_images": self.exact_count_images,
                "exact_value_images": self.exact_value_images if self.value_errors else None,
            },
            "fp_examples": self.examples_with_fp,
            "fn_examples": self.examples_with_fn,
            "wrong_class_fn_examples": self.examples_with_wrong_class_fn,
            "no_box_fn_examples": self.examples_with_no_box_fn,
        }

    def source_image_count(self) -> int:
        return sum(self.source_images.values())


def fmt_metric(value: float | None) -> str:
    return "none" if value is None else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    data_path = resolve(args.data)
    config = load_config(data_path)
    names = parse_names(config)
    images = split_images(data_path, config, args.split)
    if args.max_images > 0:
        rng = random.Random(args.seed)
        images = rng.sample(images, min(args.max_images, len(images)))
    if not images:
        raise SystemExit("No images selected")

    device = choose_device(args.device)
    detector = YOLO(str(resolve(args.detector)))
    gate, gate_class_names, gate_image_size = load_gate(resolve(args.gate), device)
    if args.reject_class not in gate_class_names:
        raise SystemExit(f"reject class {args.reject_class!r} not in gate classes: {gate_class_names}")
    transform = gate_transform(gate_image_size)
    detector_match_classes = args.detector_class_mode == "data"
    name_to_id = {name: class_id for class_id, name in names.items()}
    detector_class_min_conf = parse_class_min_conf(args.detector_class_min_conf, names)
    reclassifier_block_classes = parse_name_set(args.reclassifier_block_class)
    unknown_block_classes = sorted(name for name in reclassifier_block_classes if name not in name_to_id)
    if unknown_block_classes:
        raise SystemExit("unknown --reclassifier-block-class values: " + ", ".join(unknown_block_classes))
    reclassifier = None
    reclassifier_class_names: list[str] = []
    reclassifier_transform: transforms.Compose | None = None
    if args.reclassifier is not None:
        reclassifier, reclassifier_class_names, reclassifier_image_size = load_gate(resolve(args.reclassifier), device)
        missing_reclassifier_classes = [name for name in reclassifier_class_names if name not in name_to_id]
        if missing_reclassifier_classes:
            raise SystemExit(
                "reclassifier classes missing from YOLO data names: "
                + ", ".join(missing_reclassifier_classes)
            )
        reclassifier_transform = gate_transform(reclassifier_image_size)
    pre_gate = Metrics(
        names,
        args.match_iou,
        match_classes=detector_match_classes,
        value_errors=detector_match_classes,
    )
    post_gate = Metrics(
        names,
        args.match_iou,
        match_classes=detector_match_classes,
        value_errors=detector_match_classes,
    )
    post_reclassifier = Metrics(names, args.match_iou) if reclassifier is not None else None

    rejected_predictions = 0
    rejected_by_gate_class: Counter[str] = Counter()
    rejected_by_source: Counter[str] = Counter()
    rejected_background_images_by_source: Counter[str] = Counter()
    rejected_true_positive_like = 0
    rejected_false_positive_like = 0
    rejected_area_stats = init_area_stats()
    rejected_examples: list[dict[str, Any]] = []
    rejected_true_positive_like_examples: list[dict[str, Any]] = []
    rejected_false_positive_like_examples: list[dict[str, Any]] = []
    kept_by_gate_class: Counter[str] = Counter()
    proposal_rows: list[dict[str, Any]] = []

    for batch in batched(images, max(1, args.batch)):
        results = detector.predict(
            source=[str(path) for path in batch],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.det_iou,
            agnostic_nms=args.agnostic_nms,
            batch=len(batch),
            device=args.device,
            verbose=False,
        )
        batch_contexts: list[dict[str, Any]] = []
        flat_crops: list[Image.Image] = []
        flat_context_indexes: list[int] = []
        for context_index, (image_path, result) in enumerate(zip(batch, results)):
            image = image_from_yolo_result(result, image_path)
            try:
                image_size = image.size
                labels = read_labels(label_path_for_image(image_path), image_size)
                predictions: list[dict[str, Any]] = []
                if result.boxes is not None:
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    cls = result.boxes.cls.cpu().numpy()
                    conf = result.boxes.conf.cpu().numpy()
                    for box, class_id, score in zip(xyxy, cls, conf):
                        class_index = int(class_id)
                        if float(score) < detector_class_min_conf.get(class_index, args.conf):
                            continue
                        xyxy_box = [float(value) for value in box.tolist()]
                        predictions.append(
                            {
                                "class_id": class_index,
                                "confidence": float(score),
                                "xyxy": xyxy_box,
                                "area_ratio": box_area_ratio(xyxy_box, image_size),
                            }
                        )
                        flat_crops.append(crop_with_padding(image, xyxy_box, args.crop_padding))
                        flat_context_indexes.append(context_index)
                batch_contexts.append(
                    {
                        "image_path": image_path,
                        "labels": labels,
                        "predictions": predictions,
                        "source_group": source_group_for_image(image_path),
                    }
                )
            finally:
                image.close()

        gate_rows_flat = classify_crops(gate, transform, flat_crops, gate_class_names, device)
        reclassifier_rows_flat = (
            classify_crops(
                reclassifier,
                reclassifier_transform,
                flat_crops,
                reclassifier_class_names,
                device,
                class_key="reclassifier_class",
                conf_key="reclassifier_conf",
                probs_key="reclassifier_probs",
            )
            if reclassifier is not None and reclassifier_transform is not None
            else []
        )
        gate_rows_by_context: list[list[dict[str, Any]]] = [[] for _ in batch_contexts]
        reclassifier_rows_by_context: list[list[dict[str, Any]]] = [[] for _ in batch_contexts]
        for context_index, gate_row in zip(flat_context_indexes, gate_rows_flat, strict=False):
            gate_rows_by_context[context_index].append(gate_row)
        for context_index, reclassifier_row in zip(flat_context_indexes, reclassifier_rows_flat, strict=False):
            reclassifier_rows_by_context[context_index].append(reclassifier_row)

        for context_index, context in enumerate(batch_contexts):
            image_path = context["image_path"]
            labels = context["labels"]
            predictions = context["predictions"]
            source_group = context["source_group"]
            gate_rows = gate_rows_by_context[context_index]
            reclassifier_rows = reclassifier_rows_by_context[context_index]
            gated_predictions: list[dict[str, Any]] = []
            reclassified_predictions: list[dict[str, Any]] = []
            for index, (prediction, gate_row) in enumerate(zip(predictions, gate_rows, strict=False)):
                prediction.update(gate_row)
                rejected = (
                    gate_row["gate_class"] == args.reject_class
                    and float(gate_row["gate_conf"]) >= args.reject_min_conf
                    and (
                        args.reject_max_det_conf is None
                        or float(prediction["confidence"]) <= args.reject_max_det_conf
                    )
                )
                if rejected:
                    match_iou_score = best_match_iou(
                        labels,
                        prediction,
                        match_classes=detector_match_classes,
                    )
                    rejected_predictions += 1
                    rejected_by_gate_class[str(gate_row["gate_class"])] += 1
                    rejected_by_source[source_group] += 1
                    if not labels:
                        rejected_background_images_by_source[source_group] += 1
                    update_area_stats(rejected_area_stats, float(prediction.get("area_ratio", 0.0)))
                    example = {
                        "image": repo_rel(image_path),
                        "source_group": source_group,
                        "labels": labels,
                        "prediction": {
                            **prediction,
                            "match_iou": match_iou_score,
                            "same_class_iou": best_same_class_iou(labels, prediction),
                        },
                    }
                    if len(rejected_examples) < 30:
                        rejected_examples.append(example)
                    if match_iou_score >= args.match_iou:
                        rejected_true_positive_like += 1
                        if len(rejected_true_positive_like_examples) < 30:
                            rejected_true_positive_like_examples.append(example)
                    else:
                        rejected_false_positive_like += 1
                        if len(rejected_false_positive_like_examples) < 30:
                            rejected_false_positive_like_examples.append(example)
                else:
                    kept_by_gate_class[str(gate_row["gate_class"])] += 1
                    gated_predictions.append(prediction)
                    if reclassifier is not None:
                        reclassified = {
                            **prediction,
                            "detector_class_id": int(prediction["class_id"]),
                            "detector_class_name": (
                                "BANKNOTE"
                                if args.detector_class_mode == "banknote"
                                else names.get(
                                    int(prediction["class_id"]),
                                    str(prediction["class_id"]),
                                )
                            ),
                        }
                        if index < len(reclassifier_rows):
                            reclassifier_row = reclassifier_rows[index]
                            reclassified.update(reclassifier_row)
                            reclassifier_class = str(reclassifier_row["reclassifier_class"])
                            reclassifier_conf = float(reclassifier_row["reclassifier_conf"])
                            if (
                                reclassifier_conf >= args.reclassifier_min_conf
                                and reclassifier_class not in reclassifier_block_classes
                            ):
                                reclassified["class_id"] = int(name_to_id[reclassifier_class])
                        reclassified_predictions.append(reclassified)

            if args.proposal_json_out is not None:
                proposal_rows.append(
                    {
                        "image": repo_rel(image_path),
                        "source_group": source_group,
                        "labels": labels,
                        "predictions": predictions,
                    }
                )

            pre_gate.add_image(image_path, source_group, labels, predictions)
            post_gate.add_image(
                image_path,
                source_group,
                labels,
                gated_predictions,
                reference_predictions=predictions,
            )
            if post_reclassifier is not None:
                post_reclassifier.add_image(
                    image_path,
                    source_group,
                    labels,
                    reclassified_predictions,
                    reference_predictions=predictions,
                )

    pre_summary = pre_gate.summary()
    post_summary = post_gate.summary()
    reclassifier_summary = post_reclassifier.summary() if post_reclassifier is not None else None
    summary = {
        "schema": "cashsnap_yolo_proposal_gate_eval_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "detector": repo_rel(resolve(args.detector)),
        "gate": repo_rel(resolve(args.gate)),
        "reclassifier": repo_rel(resolve(args.reclassifier)) if args.reclassifier is not None else None,
        "reclassifier_min_conf": args.reclassifier_min_conf if args.reclassifier is not None else None,
        "reclassifier_block_classes": sorted(reclassifier_block_classes) if args.reclassifier is not None else [],
        "data": repo_rel(data_path),
        "split": args.split,
        "images": len(images),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "detector_class_min_conf": {
            names[class_id]: threshold for class_id, threshold in sorted(detector_class_min_conf.items())
        },
        "det_iou": args.det_iou,
        "match_iou": args.match_iou,
        "agnostic_nms": bool(args.agnostic_nms),
        "detector_class_mode": args.detector_class_mode,
        "crop_padding": args.crop_padding,
        "gate_classes": gate_class_names,
        "reject_class": args.reject_class,
        "reject_min_conf": args.reject_min_conf,
        "reject_max_det_conf": args.reject_max_det_conf,
        "pre_gate": pre_summary,
        "post_gate": post_summary,
        "post_reclassifier": reclassifier_summary,
        "delta": {
            "recall": (
                post_summary["recall"] - pre_summary["recall"]
                if post_summary["recall"] is not None and pre_summary["recall"] is not None
                else None
            ),
            "precision": (
                post_summary["precision"] - pre_summary["precision"]
                if post_summary["precision"] is not None and pre_summary["precision"] is not None
                else None
            ),
            "background_images_with_fp": post_summary["background_images_with_fp"]
            - pre_summary["background_images_with_fp"],
            "total_predictions": post_summary["total_predictions"] - pre_summary["total_predictions"],
        },
        "reclassifier_delta": (
            {
                "recall": (
                    reclassifier_summary["recall"] - post_summary["recall"]
                    if reclassifier_summary["recall"] is not None and post_summary["recall"] is not None
                    else None
                ),
                "precision": (
                    reclassifier_summary["precision"] - post_summary["precision"]
                    if reclassifier_summary["precision"] is not None and post_summary["precision"] is not None
                    else None
                ),
                "exact_value_images": (
                    reclassifier_summary["count_value_errors"]["exact_value_images"]
                    - post_summary["count_value_errors"]["exact_value_images"]
                    if reclassifier_summary["count_value_errors"]["exact_value_images"] is not None
                    and post_summary["count_value_errors"]["exact_value_images"] is not None
                    else None
                ),
            }
            if reclassifier_summary is not None
            else None
        ),
        "gate_rejections": {
            "rejected_predictions": rejected_predictions,
            "rejected_by_gate_class": dict(sorted(rejected_by_gate_class.items())),
            "rejected_by_source": dict(sorted(rejected_by_source.items())),
            "rejected_background_predictions_by_source": dict(sorted(rejected_background_images_by_source.items())),
            "rejected_true_positive_like": rejected_true_positive_like,
            "rejected_false_positive_like": rejected_false_positive_like,
            "rejected_area_stats": finalize_area_stats(rejected_area_stats),
            "rejected_examples": rejected_examples,
            "rejected_true_positive_like_examples": rejected_true_positive_like_examples,
            "rejected_false_positive_like_examples": rejected_false_positive_like_examples,
            "kept_by_gate_class": dict(sorted(kept_by_gate_class.items())),
        },
    }
    out_path = resolve(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.per_image_json_out is not None:
        per_image_out_path = resolve(args.per_image_json_out)
        per_image_out_path.parent.mkdir(parents=True, exist_ok=True)
        per_image_payload = {
            "schema": "cashsnap_yolo_proposal_gate_per_image_v1",
            "created_utc": summary["created_utc"],
            "summary": repo_rel(out_path),
            "detector": summary["detector"],
            "gate": summary["gate"],
            "reclassifier": summary["reclassifier"],
            "data": summary["data"],
            "split": summary["split"],
            "images": summary["images"],
            "imgsz": summary["imgsz"],
            "conf": summary["conf"],
            "detector_class_min_conf": summary["detector_class_min_conf"],
            "det_iou": summary["det_iou"],
            "match_iou": summary["match_iou"],
            "reject_class": summary["reject_class"],
            "reject_min_conf": summary["reject_min_conf"],
            "stages": {
                "pre_gate": pre_gate.per_image_rows,
                "post_gate": post_gate.per_image_rows,
                "post_reclassifier": (
                    post_reclassifier.per_image_rows if post_reclassifier is not None else []
                ),
            },
        }
        per_image_out_path.write_text(
            json.dumps(per_image_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.proposal_json_out is not None:
        proposal_out_path = resolve(args.proposal_json_out)
        proposal_out_path.parent.mkdir(parents=True, exist_ok=True)
        proposal_payload = {
            "schema": "cashsnap_yolo_proposal_gate_proposals_v1",
            "created_utc": summary["created_utc"],
            "summary": repo_rel(out_path),
            "detector": summary["detector"],
            "gate": summary["gate"],
            "data": summary["data"],
            "split": summary["split"],
            "images": summary["images"],
            "imgsz": summary["imgsz"],
            "conf": summary["conf"],
            "detector_class_min_conf": summary["detector_class_min_conf"],
            "det_iou": summary["det_iou"],
            "match_iou": summary["match_iou"],
            "detector_class_mode": summary["detector_class_mode"],
            "detector_match_classes": detector_match_classes,
            "reject_class": summary["reject_class"],
            "reject_min_conf": summary["reject_min_conf"],
            "reject_max_det_conf": summary["reject_max_det_conf"],
            "names": {str(class_id): name for class_id, name in names.items()},
            "rows": proposal_rows,
        }
        proposal_out_path.write_text(
            json.dumps(proposal_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"proposal_gate={repo_rel(out_path)} images={len(images)} "
        f"pre_recall={fmt_metric(pre_summary['recall'])} post_recall={fmt_metric(post_summary['recall'])} "
        f"pre_precision={fmt_metric(pre_summary['precision'])} post_precision={fmt_metric(post_summary['precision'])} "
        f"bg_fp={pre_summary['background_images_with_fp']}->{post_summary['background_images_with_fp']}/"
        f"{post_summary['background_images']}",
        flush=True,
    )
    if reclassifier_summary is not None:
        print(
            f"post_reclassifier_recall={fmt_metric(reclassifier_summary['recall'])} "
            f"post_reclassifier_precision={fmt_metric(reclassifier_summary['precision'])} "
            f"exact_value={reclassifier_summary['count_value_errors']['exact_value_images']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
