from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from local_runtime import configure_project_cache

configure_project_cache()

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def parse_csv_numbers(value: str, caster):
    return [caster(item.strip()) for item in value.split(",") if item.strip()]


def summarize_detection(result, names: dict[int, str]) -> dict[str, str | int | float]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        boxes = getattr(result, "obb", None)
    if boxes is None or len(boxes) == 0:
        return {
            "detections": 0,
            "classes": "",
            "top_conf": "",
            "median_box_area_pct": "",
            "small_box_count": 0,
        }

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    image_area = float(result.orig_shape[0] * result.orig_shape[1])
    areas = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0]) * np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])
    area_pct = areas / image_area * 100.0

    class_counts: dict[str, int] = {}
    for class_id in cls:
        class_counts[names.get(int(class_id), str(class_id))] = class_counts.get(
            names.get(int(class_id), str(class_id)), 0
        ) + 1
    class_summary = ";".join(f"{name}:{count}" for name, count in sorted(class_counts.items()))
    top = sorted(
        (
            f"{names.get(int(class_id), str(class_id))}:{score:.3f}"
            for class_id, score in zip(cls, conf, strict=False)
        ),
        key=lambda item: float(item.rsplit(":", 1)[1]),
        reverse=True,
    )[:8]

    return {
        "detections": int(len(boxes)),
        "classes": class_summary,
        "top_conf": ";".join(top),
        "median_box_area_pct": f"{float(np.median(area_pct)):.3f}",
        "small_box_count": int((area_pct < 1.0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run small, repeatable diagnostics on hard fan/overlap images."
    )
    parser.add_argument("--image", action="append", required=True, help="Image path; repeatable.")
    parser.add_argument("--model", action="append", required=True, help="YOLO checkpoint path; repeatable.")
    parser.add_argument("--imgsz", default="416,640,960", help="Comma-separated image sizes.")
    parser.add_argument("--conf", default="0.01,0.05,0.25", help="Comma-separated confidence thresholds.")
    parser.add_argument("--iou", default="0.50,0.70,0.90", help="Comma-separated NMS IoU thresholds.")
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    images = [resolve_path(value) for value in args.image]
    models = [resolve_path(value) for value in args.model]
    image_sizes = parse_csv_numbers(args.imgsz, int)
    confs = parse_csv_numbers(args.conf, float)
    ious = parse_csv_numbers(args.iou, float)

    rows: list[dict[str, str | int | float]] = []
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
        for model_path in models:
            model = YOLO(str(model_path))
            names = {int(key): value for key, value in model.names.items()}
            for image_size in image_sizes:
                for conf in confs:
                    for iou in ious:
                        result = model.predict(
                            source=str(image_path),
                            imgsz=image_size,
                            conf=conf,
                            iou=iou,
                            verbose=False,
                        )[0]
                        row = {
                            "image": image_path.relative_to(ROOT).as_posix(),
                            "image_width": width,
                            "image_height": height,
                            "model": model_path.relative_to(ROOT).as_posix(),
                            "imgsz": image_size,
                            "conf": conf,
                            "iou": iou,
                        }
                        row.update(summarize_detection(result, names))
                        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    if args.out is not None:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
