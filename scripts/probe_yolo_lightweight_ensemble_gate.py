#!/usr/bin/env python
"""Probe lightweight two-model YOLO gates on real splits.

This is a diagnostic harness, not a promotion evaluator. It asks whether a
high-precision/background-safe model can be used as the base while a second
high-recall model contributes class-limited rescue boxes.
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
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--rescue-model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--base-conf", type=float, default=0.05)
    parser.add_argument(
        "--rescue-conf",
        action="append",
        type=float,
        default=[],
        help="Rescue confidence threshold. Repeat for a grid.",
    )
    parser.add_argument(
        "--rescue-class",
        action="append",
        default=[],
        help="Class name or id allowed from the rescue model. Repeat to add classes. Empty means all classes.",
    )
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", required=True, type=Path)
    return parser.parse_args()


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


def box_area_ratio(box: list[float], image_size: tuple[int, int]) -> float:
    width, height = image_size
    x1, y1, x2, y2 = box
    x1 = min(max(0.0, x1), float(width))
    x2 = min(max(0.0, x2), float(width))
    y1 = min(max(0.0, y1), float(height))
    y2 = min(max(0.0, y2), float(height))
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / max(1.0, float(width * height))


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


def predictions_from_result(result: Any, image_size: tuple[int, int], source_name: str) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    if result.boxes is None:
        return predictions
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    for box, class_id, score in zip(xyxy, cls, conf):
        xyxy_box = [float(value) for value in box.tolist()]
        predictions.append(
            {
                "class_id": int(class_id),
                "confidence": float(score),
                "xyxy": xyxy_box,
                "area_ratio": box_area_ratio(xyxy_box, image_size),
                "source": source_name,
            }
        )
    return predictions


def class_nms(predictions: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_class[int(prediction["class_id"])].append(prediction)
    for rows in by_class.values():
        for prediction in sorted(rows, key=lambda item: float(item["confidence"]), reverse=True):
            if any(box_iou(prediction["xyxy"], kept_row["xyxy"]) >= iou_threshold for kept_row in kept):
                continue
            kept.append(prediction)
    return sorted(kept, key=lambda item: float(item["confidence"]), reverse=True)


def match_predictions(
    labels: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[Counter[int], Counter[int], Counter[int], list[dict[str, Any]]]:
    matched_labels: set[int] = set()
    tp_by_class: Counter[int] = Counter()
    fp_by_class: Counter[int] = Counter()
    false_predictions: list[dict[str, Any]] = []
    for prediction in sorted(predictions, key=lambda item: float(item["confidence"]), reverse=True):
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
            tp_by_class[int(labels[best_index]["class_id"])] += 1
        else:
            fp_by_class[int(prediction["class_id"])] += 1
            false_predictions.append({**prediction, "best_iou": best_iou})
    fn_by_class: Counter[int] = Counter()
    for index, label in enumerate(labels):
        if index not in matched_labels:
            fn_by_class[int(label["class_id"])] += 1
    return tp_by_class, fp_by_class, fn_by_class, false_predictions


def class_filter_ids(raw_values: list[str], names: dict[int, str]) -> set[int] | None:
    if not raw_values:
        return None
    by_name = {name: class_id for class_id, name in names.items()}
    selected: set[int] = set()
    for raw_value in raw_values:
        value = raw_value.strip()
        if not value:
            continue
        if value.isdigit():
            selected.add(int(value))
        elif value in by_name:
            selected.add(by_name[value])
        else:
            raise SystemExit(f"Unknown class filter {value!r}")
    return selected


def evaluate_rows(
    rows: list[dict[str, Any]],
    names: dict[int, str],
    rescue_conf: float | None,
    rescue_class_ids: set[int] | None,
    match_iou: float,
    nms_iou: float,
) -> dict[str, Any]:
    gt_by_class: Counter[int] = Counter()
    tp_by_class: Counter[int] = Counter()
    fp_by_class: Counter[int] = Counter()
    fn_by_class: Counter[int] = Counter()
    background_images = 0
    background_images_with_fp = 0
    total_predictions = 0
    rescue_predictions = 0
    rescue_kept = 0
    source_images: Counter[str] = Counter()
    source_background_images: Counter[str] = Counter()
    source_background_fp_images: Counter[str] = Counter()
    source_gt: Counter[str] = Counter()
    source_tp: Counter[str] = Counter()
    source_fp: Counter[str] = Counter()
    source_fn: Counter[str] = Counter()

    for row in rows:
        labels = row["labels"]
        source_group = row["source_group"]
        source_images[source_group] += 1
        if not labels:
            background_images += 1
            source_background_images[source_group] += 1
        for label in labels:
            gt_by_class[int(label["class_id"])] += 1
            source_gt[source_group] += 1

        predictions = list(row["base_predictions"])
        rescue_candidates = []
        if rescue_conf is not None:
            for prediction in row["rescue_predictions"]:
                class_id = int(prediction["class_id"])
                if rescue_class_ids is not None and class_id not in rescue_class_ids:
                    continue
                if float(prediction["confidence"]) < rescue_conf:
                    continue
                rescue_candidates.append(prediction)
        rescue_predictions += len(rescue_candidates)
        combined = class_nms(predictions + rescue_candidates, nms_iou)
        rescue_kept += sum(1 for prediction in combined if prediction.get("source") == "rescue")
        total_predictions += len(combined)

        tp, fp, fn, false_predictions = match_predictions(labels, combined, match_iou)
        tp_by_class.update(tp)
        fp_by_class.update(fp)
        fn_by_class.update(fn)
        source_tp[source_group] += sum(tp.values())
        source_fp[source_group] += sum(fp.values())
        source_fn[source_group] += sum(fn.values())
        if false_predictions and not labels:
            background_images_with_fp += 1
            source_background_fp_images[source_group] += 1

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
    return {
        "rescue_conf": rescue_conf,
        "images": len(rows),
        "background_images": background_images,
        "background_images_with_fp": background_images_with_fp,
        "total_predictions": total_predictions,
        "rescue_candidates": rescue_predictions,
        "rescue_kept_after_nms": rescue_kept,
        "gt": int(total_gt),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "recall": total_tp / total_gt if total_gt else None,
        "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) else None,
        "per_class": per_class,
        "per_source": per_source,
    }


def main() -> None:
    args = parse_args()
    data_path = resolve(args.data)
    config = load_config(data_path)
    names = parse_names(config)
    rescue_class_ids = class_filter_ids(args.rescue_class, names)
    images = split_images(data_path, config, args.split)
    if args.max_images > 0:
        rng = random.Random(args.seed)
        images = rng.sample(images, min(args.max_images, len(images)))
    if not images:
        raise SystemExit("No images selected")

    thresholds = sorted(set(args.rescue_conf or [0.05, 0.10, 0.20, 0.30, 0.50]))
    min_rescue_conf = min(thresholds)
    base_model = YOLO(str(resolve(args.base_model)))
    rescue_model = YOLO(str(resolve(args.rescue_model)))
    rows: list[dict[str, Any]] = []

    from PIL import Image

    for batch in batched(images, max(1, args.batch)):
        batch_paths = [str(path) for path in batch]
        base_results = base_model.predict(
            source=batch_paths,
            imgsz=args.imgsz,
            conf=args.base_conf,
            batch=len(batch),
            device=args.device,
            verbose=False,
        )
        rescue_results = rescue_model.predict(
            source=batch_paths,
            imgsz=args.imgsz,
            conf=min_rescue_conf,
            batch=len(batch),
            device=args.device,
            verbose=False,
        )
        for image_path, base_result, rescue_result in zip(batch, base_results, rescue_results):
            with Image.open(image_path) as image:
                image_size = image.size
                labels = read_labels(label_path_for_image(image_path), image_size)
            rows.append(
                {
                    "image": repo_rel(image_path),
                    "source_group": source_group_for_image(image_path),
                    "labels": labels,
                    "base_predictions": predictions_from_result(base_result, image_size, "base"),
                    "rescue_predictions": predictions_from_result(rescue_result, image_size, "rescue"),
                }
            )

    evaluations = [
        evaluate_rows(rows, names, None, rescue_class_ids, args.match_iou, args.nms_iou),
        *[
            evaluate_rows(rows, names, threshold, rescue_class_ids, args.match_iou, args.nms_iou)
            for threshold in thresholds
        ],
    ]
    summary = {
        "schema": "cashsnap_yolo_lightweight_ensemble_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": repo_rel(resolve(args.base_model)),
        "rescue_model": repo_rel(resolve(args.rescue_model)),
        "data": repo_rel(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "base_conf": args.base_conf,
        "rescue_conf_grid": thresholds,
        "rescue_classes": None
        if rescue_class_ids is None
        else [names.get(class_id, str(class_id)) for class_id in sorted(rescue_class_ids)],
        "match_iou": args.match_iou,
        "nms_iou": args.nms_iou,
        "evaluations": evaluations,
    }
    out_path = resolve(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"ensemble_gate={repo_rel(out_path)} split={args.split} images={len(rows)}")
    for row in evaluations:
        label = "base_only" if row["rescue_conf"] is None else f"rescue_conf={row['rescue_conf']:.3f}"
        print(
            f"{label} recall={row['recall']:.4f} precision={row['precision']:.4f} "
            f"bg_fp={row['background_images_with_fp']}/{row['background_images']} "
            f"fp={row['fp']} fn={row['fn']} rescue_kept={row['rescue_kept_after_nms']}"
        )


if __name__ == "__main__":
    main()
