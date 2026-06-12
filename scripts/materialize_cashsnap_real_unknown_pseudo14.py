#!/usr/bin/env python
"""Materialize real unknown-money target FPs as a YOLO UNKNOWN class."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from local_runtime import configure_project_cache


configure_project_cache()

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-list", type=Path, required=True)
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--unknown-class-id", type=int, default=13)
    parser.add_argument("--unknown-class-name", default="UNKNOWN_FOREIGN_NOTE")
    parser.add_argument("--source-pred-class-ids", default="0-12")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-boxes-per-image", type=int, default=2)
    parser.add_argument("--min-box-area", type=float, default=0.02)
    parser.add_argument("--max-box-area", type=float, default=0.90)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def short_hash(value: str, *, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def slug(value: str, *, max_length: int = 72) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    if not cleaned:
        cleaned = "image"
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("_")
    return cleaned


def read_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(resolved)} must be a YAML mapping")
    return data


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def class_names(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names", {})
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    raise SystemExit("base config names must be a mapping or list")


def parse_class_ids(value: str) -> set[int]:
    class_ids: set[int] = set()
    for token in re.split(r"[,\s]+", value.strip()):
        if not token:
            continue
        if "-" in token:
            raw_start, raw_end = token.split("-", 1)
            start = int(raw_start)
            end = int(raw_end)
            if end < start:
                raise SystemExit(f"invalid class range: {token}")
            class_ids.update(range(start, end + 1))
        else:
            class_ids.add(int(token))
    return class_ids


def read_image_list(path: Path) -> list[Path]:
    resolved = resolve(path)
    rows: list[Path] = []
    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        image = resolve(line)
        if image.suffix.lower() not in IMAGE_EXTS:
            raise SystemExit(f"unsupported image extension in {repo_rel(resolved)}: {repo_rel(image)}")
        if not image.exists():
            raise SystemExit(f"missing image from list {repo_rel(resolved)}: {repo_rel(image)}")
        rows.append(image)
    if not rows:
        raise SystemExit(f"empty image list: {repo_rel(resolved)}")
    return rows


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def label_lines(image: Path) -> list[str]:
    label_path = label_path_for_image(image)
    if not label_path.exists():
        return []
    return [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_zero_label_sources(images: list[Path]) -> None:
    offenders = [image for image in images if label_lines(image)]
    if offenders:
        formatted = ", ".join(repo_rel(image) for image in offenders[:8])
        raise SystemExit(f"source images must be zero-label rows; found labels for: {formatted}")


def batched(items: list[Path], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def output_image_path(out_root: Path, image: Path) -> Path:
    rel = repo_rel(image)
    stem = f"{slug(Path(rel).with_suffix('').as_posix())}_{short_hash(rel)}"
    return out_root / "images" / "train" / f"{stem}{image.suffix.lower()}"


def xyxy_to_yolo_line(xyxy: list[float], width: int, height: int, class_id: int) -> str:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid clipped box: {xyxy}")
    cx = ((x1 + x2) / 2.0) / float(width)
    cy = ((y1 + y2) / 2.0) / float(height)
    bw = (x2 - x1) / float(width)
    bh = (y2 - y1) / float(height)
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def clean_out_root(out_root: Path) -> None:
    if not out_root.exists():
        return
    allowed = (ROOT / "data" / "processed").resolve()
    try:
        out_root.resolve().relative_to(allowed)
    except ValueError as exc:
        raise SystemExit(f"--clean target must stay under data/processed: {repo_rel(out_root)}") from exc
    shutil.rmtree(out_root)


def prediction_rows(
    result: Any,
    *,
    source_class_ids: set[int],
    unknown_class_id: int,
    min_box_area: float,
    max_box_area: float,
    max_boxes_per_image: int,
) -> tuple[list[str], list[dict[str, Any]], Counter[int], int]:
    height, width = [int(value) for value in result.orig_shape]
    candidates: list[dict[str, Any]] = []
    raw_count = 0
    raw_class_counts: Counter[int] = Counter()
    if result.boxes is not None:
        xyxys = result.boxes.xyxy.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        for xyxy, conf, raw_class in zip(xyxys, confs, classes):
            raw_count += 1
            source_class_id = int(raw_class)
            raw_class_counts[source_class_id] += 1
            if source_class_ids and source_class_id not in source_class_ids:
                continue
            x1, y1, x2, y2 = [float(value) for value in xyxy]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(width * height)
            if area < min_box_area or area > max_box_area:
                continue
            candidates.append(
                {
                    "source_class_id": source_class_id,
                    "confidence": float(conf),
                    "xyxy": [x1, y1, x2, y2],
                    "area": area,
                }
            )
    candidates.sort(key=lambda item: (float(item["confidence"]), float(item["area"])), reverse=True)
    kept = candidates[:max_boxes_per_image]
    label_lines_out = [
        xyxy_to_yolo_line(item["xyxy"], width, height, unknown_class_id)
        for item in kept
    ]
    return label_lines_out, kept, raw_class_counts, raw_count


def main() -> int:
    args = parse_args()
    if args.unknown_class_id < 0:
        raise SystemExit("--unknown-class-id must be non-negative")
    if args.max_boxes_per_image < 1:
        raise SystemExit("--max-boxes-per-image must be at least 1")
    if args.min_box_area < 0 or args.max_box_area <= 0 or args.min_box_area > args.max_box_area:
        raise SystemExit("invalid box-area bounds")

    model_path = resolve(args.model)
    if not model_path.exists():
        raise SystemExit(f"missing model: {repo_rel(model_path)}")
    out_root = resolve(args.out_root)
    out_list = resolve(args.out_list)
    out_config = resolve(args.out_config)
    images = read_image_list(args.image_list)
    if args.max_images:
        images = images[: args.max_images]
    ensure_zero_label_sources(images)

    source_class_ids = parse_class_ids(args.source_pred_class_ids)
    names = class_names(read_yaml(args.base_config))
    if args.unknown_class_id in names and names[args.unknown_class_id] != args.unknown_class_name:
        raise SystemExit(
            f"class {args.unknown_class_id} already exists as {names[args.unknown_class_id]!r}; "
            f"cannot write {args.unknown_class_name!r}"
        )
    names[args.unknown_class_id] = args.unknown_class_name

    if args.clean and not args.dry_run:
        clean_out_root(out_root)
    if not args.dry_run:
        (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (out_root / "metadata").mkdir(parents=True, exist_ok=True)
        out_list.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    rows: list[str] = []
    records: list[dict[str, Any]] = []
    raw_class_counts: Counter[int] = Counter()
    kept_source_class_counts: Counter[int] = Counter()
    skipped_no_raw_prediction = 0
    skipped_no_kept_prediction = 0
    raw_prediction_count = 0
    kept_box_count = 0

    for batch in batched(images, args.batch):
        results = model.predict(
            [str(path) for path in batch],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_det=args.max_det,
            batch=args.batch,
            verbose=False,
        )
        for image, result in zip(batch, results):
            label_lines_out, kept, batch_class_counts, raw_count = prediction_rows(
                result,
                source_class_ids=source_class_ids,
                unknown_class_id=args.unknown_class_id,
                min_box_area=args.min_box_area,
                max_box_area=args.max_box_area,
                max_boxes_per_image=args.max_boxes_per_image,
            )
            raw_prediction_count += raw_count
            raw_class_counts.update(batch_class_counts)
            if raw_count == 0:
                skipped_no_raw_prediction += 1
                continue
            if not kept:
                skipped_no_kept_prediction += 1
                continue

            out_image = output_image_path(out_root, image)
            out_label = label_path_for_image(out_image)
            row = repo_rel(out_image)
            rows.append(row)
            kept_box_count += len(kept)
            for item in kept:
                kept_source_class_counts[int(item["source_class_id"])] += 1
            record = {
                "image": row,
                "label": repo_rel(out_label),
                "source_image": repo_rel(image),
                "source_label": repo_rel(label_path_for_image(image)),
                "unknown_class_id": args.unknown_class_id,
                "unknown_class_name": args.unknown_class_name,
                "source_predictions_relabelled_unknown": kept,
            }
            records.append(record)
            if not args.dry_run:
                shutil.copy2(image, out_image)
                out_label.write_text("\n".join(label_lines_out) + "\n", encoding="utf-8")

    if not rows:
        raise SystemExit("no pseudo-UNKNOWN rows were materialized")

    summary = {
        "schema": "cashsnap_real_unknown_pseudo14_v1",
        "model": repo_rel(model_path),
        "image_list": repo_rel(resolve(args.image_list)),
        "out_root": repo_rel(out_root),
        "out_list": repo_rel(out_list),
        "out_config": repo_rel(out_config),
        "unknown_class_id": args.unknown_class_id,
        "unknown_class_name": args.unknown_class_name,
        "source_pred_class_ids": sorted(source_class_ids),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "max_det": args.max_det,
        "max_boxes_per_image": args.max_boxes_per_image,
        "min_box_area": args.min_box_area,
        "max_box_area": args.max_box_area,
        "input_images": len(images),
        "materialized_images": len(rows),
        "unknown_boxes": kept_box_count,
        "skipped_no_raw_prediction": skipped_no_raw_prediction,
        "skipped_no_kept_prediction": skipped_no_kept_prediction,
        "raw_prediction_count": raw_prediction_count,
        "raw_source_class_counts": {str(key): raw_class_counts[key] for key in sorted(raw_class_counts)},
        "kept_source_class_counts": {str(key): kept_source_class_counts[key] for key in sorted(kept_source_class_counts)},
        "dry_run": bool(args.dry_run),
    }

    if not args.dry_run:
        out_list.write_text("\n".join(rows) + "\n", encoding="utf-8")
        (out_root / "metadata" / "train.jsonl").write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
            encoding="utf-8",
        )
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        dataset_yaml = {
            "path": ".",
            "train": "images/train",
            "val": "images/train",
            "test": "images/train",
            "names": names,
            "cashsnap_policy": {
                "intended_use": "Train-split real unknown-money pseudo-UNKNOWN diagnostic bridge only.",
                "promotion_rule": "Never treat this train-split pseudo set as held-out proof; promotion must pass full real, partial, source-excluded, held-out unknown-money, and true-empty guards after filtering UNKNOWN predictions.",
            },
        }
        write_yaml(out_root / "data.yaml", dataset_yaml)
        write_yaml(
            out_config,
            {
                "path": rel_between(out_config.parent, ROOT),
                "train": repo_rel(out_list),
                "val": repo_rel(out_list),
                "test": repo_rel(out_list),
                "names": names,
                "cashsnap_policy": dataset_yaml["cashsnap_policy"],
                "cashsnap_sources": {
                    "pseudo_unknown_source_list": repo_rel(resolve(args.image_list)),
                    "pseudo_unknown_teacher_model": repo_rel(model_path),
                    "pseudo_unknown_root": repo_rel(out_root),
                },
                "cashsnap_real_unknown_pseudo14": summary,
            },
        )

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
