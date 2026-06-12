#!/usr/bin/env python
"""Build a target/non-target/empty bridge for YOLO empty-label rows."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
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
TARGET_SOURCE_GROUPS = {
    "billsbank",
    "cambodia_currency_project",
    "khmer_scan",
    "khmer_us_currency",
    "usd_total",
}
MIXED_CURRENCY_SOURCE_GROUPS = {"asian_currency", "cashcountingxl"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-conf", type=float, default=0.25)
    parser.add_argument("--high-conf", type=float, default=0.50)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sheet-items", type=int, default=32)
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


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(path)} must be a YAML mapping")
    return data


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw = Path(str(config.get("path", "."))).expanduser()
    return raw if raw.is_absolute() else (config_path.parent / raw).resolve()


def split_images(config_path: Path, config: dict[str, Any], split: str) -> list[Path]:
    root = data_root(config_path, config)
    split_value = config.get(split)
    if split_value is None:
        raise SystemExit(f"{repo_rel(config_path)} has no split {split!r}")
    values = split_value if isinstance(split_value, list) else [split_value]
    rows: list[Path] = []
    for raw in values:
        path = Path(str(raw))
        resolved = path if path.is_absolute() else root / path
        if resolved.suffix.lower() == ".txt":
            for raw_line in resolved.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line and not line.startswith("#"):
                    image = Path(line)
                    rows.append(image if image.is_absolute() else root / image)
        else:
            rows.extend(sorted(item for item in resolved.glob("*") if item.suffix.lower() in IMAGE_EXTS))
    return rows


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image_path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def is_empty_label(image_path: Path) -> bool:
    label = label_path_for_image(image_path)
    if not label.exists():
        return True
    return not label.read_text(encoding="utf-8").strip()


def source_group(image_path: Path) -> str:
    name = image_path.name.lower()
    for group in sorted(TARGET_SOURCE_GROUPS | MIXED_CURRENCY_SOURCE_GROUPS, key=len, reverse=True):
        if name.startswith(group):
            return group
    match = re.match(r"([a-z]+(?:_[a-z]+)?)_", name)
    return match.group(1) if match else image_path.stem.split("_")[0].lower()


def batched(items: list[Path], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def class_names(model: YOLO) -> dict[int, str]:
    names = getattr(model.model, "names", None)
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def run_model(
    *,
    model_path: Path,
    images: list[Path],
    args: argparse.Namespace,
) -> tuple[dict[str, list[dict[str, Any]]], dict[int, str]]:
    model = YOLO(str(model_path))
    names = class_names(model)
    by_image: dict[str, list[dict[str, Any]]] = {}
    for batch in batched(images, max(1, args.batch)):
        results = model.predict(
            source=[str(path) for path in batch],
            imgsz=args.imgsz,
            conf=args.min_conf,
            batch=len(batch),
            device=args.device,
            verbose=False,
        )
        for image_path, result in zip(batch, results, strict=True):
            rows: list[dict[str, Any]] = []
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy()
                conf = result.boxes.conf.cpu().numpy()
                for box, class_id, score in zip(xyxy, cls, conf, strict=True):
                    class_id_int = int(class_id)
                    rows.append(
                        {
                            "class_id": class_id_int,
                            "class_name": names.get(class_id_int, f"class_{class_id_int}"),
                            "confidence": float(score),
                            "xyxy": [float(value) for value in box.tolist()],
                        }
                    )
            rows.sort(key=lambda item: float(item["confidence"]), reverse=True)
            by_image[repo_rel(image_path)] = rows
    return by_image, names


def top_detection(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    return detections[0] if detections else None


def max_conf(detections: list[dict[str, Any]]) -> float:
    top = top_detection(detections)
    return float(top["confidence"]) if top else 0.0


def bucket_for(
    *,
    group: str,
    teacher: list[dict[str, Any]],
    student: list[dict[str, Any]],
    high_conf: float,
    min_conf: float,
) -> str:
    teacher_high = max_conf(teacher) >= high_conf
    student_high = max_conf(student) >= high_conf
    teacher_mid = max_conf(teacher) >= min_conf
    student_mid = max_conf(student) >= min_conf
    if teacher_high and group in TARGET_SOURCE_GROUPS:
        return "suspect_unlabeled_target"
    if teacher_high and group in MIXED_CURRENCY_SOURCE_GROUPS:
        return "currency_review"
    if teacher_high:
        return "teacher_high_review"
    if student_high and not teacher_mid:
        return "student_overfire_review"
    if teacher_mid or student_mid:
        return "model_review"
    return "likely_true_empty"


def summarize_detections(detections: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    top = top_detection(detections)
    if not top:
        return {
            f"{prefix}_detections": 0,
            f"{prefix}_top_class": "",
            f"{prefix}_top_conf": 0.0,
        }
    return {
        f"{prefix}_detections": len(detections),
        f"{prefix}_top_class": top["class_name"],
        f"{prefix}_top_conf": round(float(top["confidence"]), 6),
    }


def draw_sheet(
    *,
    records: list[dict[str, Any]],
    out_path: Path,
    count: int,
    title: str,
) -> None:
    chosen = records[:count]
    if not chosen:
        return
    thumb_w, thumb_h = 320, 260
    caption_h = 48
    cols = min(4, len(chosen))
    rows = (len(chosen) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + caption_h) + 28), (238, 238, 238))
    draw_sheet_ctx = ImageDraw.Draw(sheet)
    draw_sheet_ctx.text((8, 6), title[:120], fill=(0, 0, 0))
    for index, record in enumerate(chosen):
        image_path = resolve(record["image"])
        with Image.open(image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for det in record.get("teacher_detections_full", [])[:3]:
                draw.rectangle(det["xyxy"], outline=(40, 180, 70), width=3)
            for det in record.get("student_detections_full", [])[:3]:
                draw.rectangle(det["xyxy"], outline=(230, 80, 30), width=2)
            thumb = ImageOps.contain(image, (thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (index % cols) * thumb_w
        y = 28 + (index // cols) * (thumb_h + caption_h)
        sheet.paste(thumb, (x + (thumb_w - thumb.width) // 2, y))
        caption = (
            f"{record['bucket']} {record['source_group']} "
            f"T:{record['teacher_top_class']} {record['teacher_top_conf']:.2f} "
            f"S:{record['student_top_class']} {record['student_top_conf']:.2f}"
        )
        draw_sheet_ctx.text((x + 5, y + thumb_h + 4), caption[:58], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image",
        "label",
        "bucket",
        "source_group",
        "teacher_detections",
        "teacher_top_class",
        "teacher_top_conf",
        "student_detections",
        "student_top_class",
        "student_top_conf",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def main() -> int:
    args = parse_args()
    data_path = resolve(args.data)
    config = load_yaml(data_path)
    images = [image for image in split_images(data_path, config, args.split) if is_empty_label(image)]
    if args.max_images > 0:
        rng = random.Random(args.seed)
        images = rng.sample(images, min(args.max_images, len(images)))
    if not images:
        raise SystemExit("No empty-label images selected")

    teacher_rows, names = run_model(model_path=resolve(args.teacher_model), images=images, args=args)
    if args.student_model is not None:
        student_rows, _ = run_model(model_path=resolve(args.student_model), images=images, args=args)
    else:
        student_rows = {repo_rel(image): [] for image in images}

    records: list[dict[str, Any]] = []
    for image in images:
        image_rel = repo_rel(image)
        group = source_group(image)
        teacher = teacher_rows.get(image_rel, [])
        student = student_rows.get(image_rel, [])
        bucket = bucket_for(
            group=group,
            teacher=teacher,
            student=student,
            high_conf=args.high_conf,
            min_conf=args.min_conf,
        )
        record = {
            "image": image_rel,
            "label": repo_rel(label_path_for_image(image)),
            "source_group": group,
            "bucket": bucket,
            "teacher_detections_full": teacher,
            "student_detections_full": student,
            "teacher_detections": len(teacher),
            "student_detections": len(student),
            **summarize_detections(teacher, "teacher"),
            **summarize_detections(student, "student"),
        }
        records.append(record)

    records.sort(
        key=lambda item: (
            item["bucket"],
            -float(item["teacher_top_conf"]),
            -float(item["student_top_conf"]),
            item["image"],
        )
    )
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "manifest.csv", records)

    bucket_counts = Counter(record["bucket"] for record in records)
    source_counts = Counter(record["source_group"] for record in records)
    bucket_source_counts = Counter((record["bucket"], record["source_group"]) for record in records)
    teacher_class_counts = Counter(
        record["teacher_top_class"]
        for record in records
        if float(record["teacher_top_conf"]) >= args.min_conf
    )
    student_class_counts = Counter(
        record["student_top_class"]
        for record in records
        if float(record["student_top_conf"]) >= args.min_conf
    )
    summary = {
        "schema": "cashsnap_empty_label_semantic_bridge_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": repo_rel(data_path),
        "split": args.split,
        "images": len(images),
        "teacher_model": repo_rel(resolve(args.teacher_model)),
        "student_model": repo_rel(resolve(args.student_model)) if args.student_model else "",
        "imgsz": args.imgsz,
        "min_conf": args.min_conf,
        "high_conf": args.high_conf,
        "names": names,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "bucket_source_counts": {
            f"{bucket}|{source}": count
            for (bucket, source), count in sorted(bucket_source_counts.items())
        },
        "teacher_top_class_counts": dict(sorted(teacher_class_counts.items())),
        "student_top_class_counts": dict(sorted(student_class_counts.items())),
        "manifest": "manifest.csv",
        "sheets": {},
    }

    for bucket in [
        "suspect_unlabeled_target",
        "currency_review",
        "teacher_high_review",
        "student_overfire_review",
        "model_review",
    ]:
        bucket_rows = [record for record in records if record["bucket"] == bucket]
        sheet_path = out_dir / "sheets" / f"{bucket}.jpg"
        draw_sheet(records=bucket_rows, out_path=sheet_path, count=args.sheet_items, title=bucket)
        if sheet_path.exists():
            summary["sheets"][bucket] = repo_rel(sheet_path)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"semantic_bridge={repo_rel(summary_path)} images={len(images)} "
        f"buckets={dict(sorted(bucket_counts.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
