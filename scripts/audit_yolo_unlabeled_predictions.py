#!/usr/bin/env python
"""Audit YOLO images for detector predictions that do not match labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageOps

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
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--match-iou", type=float, default=0.10)
    parser.add_argument(
        "--min-prediction-coverage",
        type=float,
        default=0.0,
        help=(
            "Require this fraction of the prediction area to be covered by a "
            "declared label. Default 0 preserves the legacy IoU-only audit."
        ),
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--sheet-out", type=Path, default=None)
    parser.add_argument("--sheet-items", type=int, default=24)
    return parser.parse_args()


def load_data_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"YOLO data YAML must be a mapping: {repo_rel(path)}")
    return config


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    root = Path(str(config.get("path", ".")))
    return root if root.is_absolute() else (config_path.parent / root).resolve()


def split_images(config_path: Path, split: str) -> list[Path]:
    config = load_data_config(config_path)
    root = data_root(config_path, config)
    split_value = config.get(split)
    if split_value is None:
        raise SystemExit(f"{repo_rel(config_path)} has no split {split!r}")
    values = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for raw in values:
        path = Path(str(raw))
        path = path if path.is_absolute() else root / path
        if path.suffix.lower() == ".txt":
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line and not line.startswith("#"):
                    image = Path(line)
                    images.append(image if image.is_absolute() else root / image)
        else:
            images.extend(sorted(item for item in path.glob("*") if item.suffix.lower() in IMAGE_EXTS))
    return images


def unique_images(images: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for image in images:
        resolved = image.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(image)
    return unique


def manifest_images(path: Path) -> list[Path]:
    images: list[Path] = []
    if path.suffix.lower() == ".jsonl":
        for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            image = payload.get("image")
            if not image:
                raise SystemExit(f"{repo_rel(path)}:{line_no} missing image field")
            images.append(resolve(image))
    elif path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "image" not in (reader.fieldnames or []):
                raise SystemExit(f"{repo_rel(path)} must include an image column")
            for row in reader:
                if row.get("image"):
                    images.append(resolve(row["image"]))
    else:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                images.append(resolve(line))
    return unique_images(images)


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
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        labels.append({"class_id": class_id, "xyxy": [x1, y1, x2, y2]})
    return labels


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def overlap_metrics(prediction: list[float], label: list[float]) -> dict[str, float]:
    inter = intersection_area(prediction, label)
    prediction_area = box_area(prediction)
    label_area = box_area(label)
    union = prediction_area + label_area - inter
    return {
        "iou": inter / union if union > 0 else 0.0,
        "prediction_coverage": inter / prediction_area if prediction_area > 0 else 0.0,
        "label_coverage": inter / label_area if label_area > 0 else 0.0,
    }


def best_overlap(prediction: list[float], labels: list[dict[str, Any]]) -> dict[str, float]:
    best = {"iou": 0.0, "prediction_coverage": 0.0, "label_coverage": 0.0}
    for label in labels:
        metrics = overlap_metrics(prediction, label["xyxy"])
        if (metrics["iou"], metrics["prediction_coverage"]) > (best["iou"], best["prediction_coverage"]):
            best = metrics
    return best


def iou(a: list[float], b: list[float]) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def batched(items: list[Path], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def draw_sheet(records: list[dict[str, Any]], sheet_out: Path, count: int) -> None:
    if count <= 0 or not records:
        return
    chosen = records[:count]
    thumb_w, thumb_h = 260, 220
    cols = min(4, len(chosen))
    rows = (len(chosen) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (238, 238, 238))
    for idx, record in enumerate(chosen):
        image_path = resolve(record["image"])
        with Image.open(image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for label in record["labels"]:
                draw.rectangle(label["xyxy"], outline=(40, 180, 70), width=3)
            for pred in record["unmatched_predictions"]:
                draw.rectangle(pred["xyxy"], outline=(220, 30, 30), width=3)
            thumb = ImageOps.contain(image, (thumb_w, thumb_h - 34), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb_w
        y = (idx // cols) * thumb_h
        sheet.paste(thumb, (x + (thumb_w - thumb.width) // 2, y))
        ImageDraw.Draw(sheet).text((x + 6, y + thumb_h - 30), f"unmatched={record['unmatched_count']}", fill=(0, 0, 0))
    sheet_out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_out, quality=92)


def main() -> None:
    args = parse_args()
    if (args.data is None) == (args.manifest is None):
        raise SystemExit("Provide exactly one of --data or --manifest")
    if args.data is not None:
        source_path = resolve(args.data)
        source_kind = "data_yaml"
        images = split_images(source_path, args.split)
    else:
        source_path = resolve(args.manifest)
        source_kind = "manifest"
        images = manifest_images(source_path)
    if args.max_images > 0:
        rng = random.Random(args.seed)
        images = rng.sample(images, min(args.max_images, len(images)))
    if not images:
        raise SystemExit("No images selected")

    model = YOLO(str(resolve(args.model)))
    suspect_records: list[dict[str, Any]] = []
    unmatched_by_class: Counter[int] = Counter()
    total_predictions = 0
    total_unmatched = 0
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
            with Image.open(image_path) as image:
                labels = read_labels(label_path_for_image(image_path), image.size)
            predictions = []
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
            unmatched = []
            for prediction in predictions:
                best = best_overlap(prediction["xyxy"], labels)
                reasons = []
                if best["iou"] < args.match_iou:
                    reasons.append("low_iou")
                if best["prediction_coverage"] < args.min_prediction_coverage:
                    reasons.append("low_prediction_coverage")
                if reasons:
                    unmatched.append(
                        {
                            **prediction,
                            "best_label_iou": best["iou"],
                            "best_prediction_coverage": best["prediction_coverage"],
                            "best_label_coverage": best["label_coverage"],
                            "unmatched_reasons": reasons,
                        }
                    )
                    unmatched_by_class[prediction["class_id"]] += 1
            total_predictions += len(predictions)
            total_unmatched += len(unmatched)
            if unmatched:
                suspect_records.append(
                    {
                        "image": repo_rel(image_path),
                        "label": repo_rel(label_path_for_image(image_path)),
                        "labels": labels,
                        "predictions": predictions,
                        "unmatched_predictions": unmatched,
                        "unmatched_count": len(unmatched),
                    }
                )

    suspect_records.sort(key=lambda item: item["unmatched_count"], reverse=True)
    summary = {
        "schema": "cashsnap_yolo_unlabeled_prediction_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": repo_rel(resolve(args.model)),
        "data": repo_rel(source_path),
        "source": repo_rel(source_path),
        "source_kind": source_kind,
        "split": args.split,
        "images": len(images),
        "conf": args.conf,
        "match_iou": args.match_iou,
        "min_prediction_coverage": args.min_prediction_coverage,
        "total_predictions": total_predictions,
        "total_unmatched_predictions": total_unmatched,
        "images_with_unmatched_predictions": len(suspect_records),
        "unmatched_by_class_id": dict(sorted(unmatched_by_class.items())),
        "suspect_records": suspect_records,
    }
    json_out = resolve(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv_out is not None:
        csv_out = resolve(args.csv_out)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "unmatched_count"])
            writer.writeheader()
            for record in suspect_records:
                writer.writerow({"image": record["image"], "unmatched_count": record["unmatched_count"]})
    if args.sheet_out is not None:
        draw_sheet(suspect_records, resolve(args.sheet_out), args.sheet_items)
    print(
        f"unlabeled_audit={repo_rel(json_out)} images={len(images)} "
        f"suspect_images={len(suspect_records)} unmatched={total_unmatched}",
        flush=True,
    )


if __name__ == "__main__":
    main()
