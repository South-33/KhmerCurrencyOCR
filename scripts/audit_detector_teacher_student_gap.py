#!/usr/bin/env python
"""Audit GT boxes where a teacher detector localizes what a student misses.

This is a data-design diagnostic. It compares two YOLO detectors on the same
YOLO split and writes source/class summaries plus GT-level rows for cases where
the teacher has usable evidence that the student lacks.
"""

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
    parser.add_argument("--student", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--det-iou", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument(
        "--csv-rows",
        choices=["teacher_wins", "all"],
        default="teacher_wins",
        help="Rows to write to --csv-out.",
    )
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


def predictions_from_result(result: Any) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    confs = boxes.conf.detach().cpu().tolist()
    predictions: list[dict[str, Any]] = []
    for box, class_id, confidence in zip(xyxy, classes, confs):
        predictions.append(
            {
                "class_id": int(class_id),
                "confidence": float(confidence),
                "xyxy": [float(value) for value in box],
            }
        )
    return predictions


def best_prediction(
    label: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    same_class: bool,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_iou = 0.0
    for prediction in predictions:
        if same_class and int(prediction["class_id"]) != int(label["class_id"]):
            continue
        score = box_iou(prediction["xyxy"], label["xyxy"])
        if score > best_iou:
            best_iou = score
            best = prediction
    if best is None:
        return None
    return {**best, "best_iou": best_iou}


def iou_bucket(best: dict[str, Any] | None, match_iou: float) -> str:
    if best is None:
        return "no_prediction"
    best_iou = float(best.get("best_iou", 0.0))
    if best_iou >= match_iou:
        return f"iou_ge_{match_iou:.2f}"
    if best_iou >= 0.25:
        return "iou_0.25_to_match"
    if best_iou >= 0.10:
        return "iou_0.10_to_0.25"
    if best_iou > 0.0:
        return "iou_lt_0.10"
    return "no_overlap"


def best_fields(prefix: str, best: dict[str, Any] | None, names: dict[int, str], match_iou: float) -> dict[str, Any]:
    if best is None:
        return {
            f"{prefix}_best_iou": 0.0,
            f"{prefix}_best_class": "",
            f"{prefix}_best_conf": None,
            f"{prefix}_bucket": "no_prediction",
        }
    class_id = int(best["class_id"])
    return {
        f"{prefix}_best_iou": float(best["best_iou"]),
        f"{prefix}_best_class": names.get(class_id, str(class_id)),
        f"{prefix}_best_conf": float(best["confidence"]),
        f"{prefix}_bucket": iou_bucket(best, match_iou),
    }


def increment(counter: Counter[str], *parts: str) -> None:
    counter["|".join(parts)] += 1


def summarize_counter(counter: Counter[str], limit: int = 80) -> dict[str, int]:
    return dict(counter.most_common(limit))


def main() -> None:
    args = parse_args()
    data_path = resolve(args.data)
    config = load_config(data_path)
    names = parse_names(config)
    images = split_images(data_path, config, args.split)
    if args.max_images:
        rng = random.Random(args.seed)
        rng.shuffle(images)
        images = images[: args.max_images]

    student = YOLO(str(resolve(args.student)))
    teacher = YOLO(str(resolve(args.teacher)))

    totals = Counter()
    by_source = Counter()
    by_class = Counter()
    by_source_class = Counter()
    student_bucket = Counter()
    teacher_bucket = Counter()
    csv_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for batch in batched(images, args.batch):
        sources = [str(path) for path in batch]
        student_results = student.predict(
            source=sources,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.det_iou,
            device=args.device,
            verbose=False,
            save=False,
        )
        teacher_results = teacher.predict(
            source=sources,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.det_iou,
            device=args.device,
            verbose=False,
            save=False,
        )
        for image_path, student_result, teacher_result in zip(batch, student_results, teacher_results):
            height, width = tuple(int(value) for value in student_result.orig_shape[:2])
            image_size = (width, height)
            source = source_group_for_image(image_path)
            labels = read_labels(label_path_for_image(image_path), image_size)
            student_predictions = predictions_from_result(student_result)
            teacher_predictions = predictions_from_result(teacher_result)
            if labels:
                increment(by_source, source, "gt")
            else:
                increment(by_source, source, "background_image")
            for index, label in enumerate(labels):
                class_id = int(label["class_id"])
                class_name = names.get(class_id, str(class_id))
                source_class = f"{source}|{class_name}"
                totals["gt"] += 1
                increment(by_source, source, "gt_box")
                increment(by_class, class_name, "gt")
                increment(by_source_class, source_class, "gt")

                student_any = best_prediction(label, student_predictions, same_class=False)
                student_same = best_prediction(label, student_predictions, same_class=True)
                teacher_any = best_prediction(label, teacher_predictions, same_class=False)
                teacher_same = best_prediction(label, teacher_predictions, same_class=True)

                student_localized = student_any is not None and float(student_any["best_iou"]) >= args.match_iou
                student_same_hit = student_same is not None and float(student_same["best_iou"]) >= args.match_iou
                teacher_localized = teacher_any is not None and float(teacher_any["best_iou"]) >= args.match_iou
                teacher_same_hit = teacher_same is not None and float(teacher_same["best_iou"]) >= args.match_iou

                if student_localized:
                    totals["student_localized"] += 1
                    increment(by_source, source, "student_localized")
                    increment(by_class, class_name, "student_localized")
                    increment(by_source_class, source_class, "student_localized")
                if teacher_localized:
                    totals["teacher_localized"] += 1
                    increment(by_source, source, "teacher_localized")
                    increment(by_class, class_name, "teacher_localized")
                    increment(by_source_class, source_class, "teacher_localized")
                if student_same_hit:
                    totals["student_same_class"] += 1
                    increment(by_source, source, "student_same_class")
                    increment(by_class, class_name, "student_same_class")
                    increment(by_source_class, source_class, "student_same_class")
                if teacher_same_hit:
                    totals["teacher_same_class"] += 1
                    increment(by_source, source, "teacher_same_class")
                    increment(by_class, class_name, "teacher_same_class")
                    increment(by_source_class, source_class, "teacher_same_class")

                teacher_localizes_student_misses = teacher_localized and not student_localized
                teacher_same_student_not_same = teacher_same_hit and not student_same_hit
                if teacher_localizes_student_misses:
                    totals["teacher_localizes_student_misses"] += 1
                    increment(by_source, source, "teacher_localizes_student_misses")
                    increment(by_class, class_name, "teacher_localizes_student_misses")
                    increment(by_source_class, source_class, "teacher_localizes_student_misses")
                if teacher_same_student_not_same:
                    totals["teacher_same_student_not_same"] += 1
                    increment(by_source, source, "teacher_same_student_not_same")
                    increment(by_class, class_name, "teacher_same_student_not_same")
                    increment(by_source_class, source_class, "teacher_same_student_not_same")

                student_any_bucket = iou_bucket(student_any, args.match_iou)
                teacher_any_bucket = iou_bucket(teacher_any, args.match_iou)
                increment(student_bucket, class_name, student_any_bucket)
                increment(teacher_bucket, class_name, teacher_any_bucket)

                row = {
                    "image": repo_rel(image_path),
                    "label": repo_rel(label_path_for_image(image_path)),
                    "label_index": index,
                    "source_group": source,
                    "class_name": class_name,
                    "gt_area_ratio": box_area_ratio(label["xyxy"], image_size),
                    "teacher_localizes_student_misses": teacher_localizes_student_misses,
                    "teacher_same_student_not_same": teacher_same_student_not_same,
                    "student_localized": student_localized,
                    "teacher_localized": teacher_localized,
                    "student_same_class": student_same_hit,
                    "teacher_same_class": teacher_same_hit,
                    **best_fields("student_any", student_any, names, args.match_iou),
                    **best_fields("teacher_any", teacher_any, names, args.match_iou),
                    **best_fields("student_same", student_same, names, args.match_iou),
                    **best_fields("teacher_same", teacher_same, names, args.match_iou),
                }
                if args.csv_rows == "all" or teacher_localizes_student_misses or teacher_same_student_not_same:
                    csv_rows.append(row)
                if (teacher_localizes_student_misses or teacher_same_student_not_same) and len(examples) < 80:
                    examples.append(row)

    def metric(numerator: str, denominator: str = "gt") -> float | None:
        total = totals[denominator]
        return (totals[numerator] / total) if total else None

    summary = {
        "schema": "cashsnap_detector_teacher_student_gap_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "student": repo_rel(resolve(args.student)),
        "teacher": repo_rel(resolve(args.teacher)),
        "data": repo_rel(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "det_iou": args.det_iou,
        "match_iou": args.match_iou,
        "images": len(images),
        "totals": dict(totals),
        "rates": {
            "student_localization_recall": metric("student_localized"),
            "teacher_localization_recall": metric("teacher_localized"),
            "student_same_class_recall": metric("student_same_class"),
            "teacher_same_class_recall": metric("teacher_same_class"),
            "teacher_localizes_student_misses_rate": metric("teacher_localizes_student_misses"),
            "teacher_same_student_not_same_rate": metric("teacher_same_student_not_same"),
        },
        "by_source": summarize_counter(by_source, 160),
        "by_class": summarize_counter(by_class, 160),
        "by_source_class": summarize_counter(by_source_class, 220),
        "student_any_iou_buckets_by_class": summarize_counter(student_bucket, 160),
        "teacher_any_iou_buckets_by_class": summarize_counter(teacher_bucket, 160),
        "csv_rows": len(csv_rows),
        "examples": examples,
    }

    json_out = resolve(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.csv_out:
        csv_out = resolve(args.csv_out)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(csv_rows[0].keys()) if csv_rows else ["image"]
        with csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    print(
        "teacher_student_gap="
        f"{repo_rel(json_out)} images={len(images)} gt={totals['gt']} "
        f"student_loc={metric('student_localized'):.4f} teacher_loc={metric('teacher_localized'):.4f} "
        f"teacher_loc_student_miss={totals['teacher_localizes_student_misses']} "
        f"teacher_same_student_not_same={totals['teacher_same_student_not_same']} csv_rows={len(csv_rows)}"
    )


if __name__ == "__main__":
    main()
