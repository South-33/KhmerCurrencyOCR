#!/usr/bin/env python
"""Probe YOLO false positives on known zero-label/background image roots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from local_runtime import configure_project_cache


configure_project_cache()

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
REVIEW_ARTIFACT_STEMS = {"contact_sheet", "suspect_contact"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="Model path or label=path. Repeatable.")
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        type=Path,
        help="Directory of zero-label images. Repeatable.",
    )
    parser.add_argument(
        "--data",
        action="append",
        default=[],
        type=Path,
        help="YOLO dataset YAML to probe via empty-label split rows. Repeatable.",
    )
    parser.add_argument(
        "--image-list",
        action="append",
        default=[],
        type=Path,
        help="Text file of image paths to probe as a named zero-label source. Repeatable.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help="Dataset split key to probe from each --data YAML; only empty/missing-label rows are used.",
    )
    parser.add_argument("--conf", default="0.05,0.18,0.25", help="Comma-separated confidence thresholds.")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help=(
            "Bounded inference batch size. Images are fed through a path-list file to avoid RAM preloading. "
            "Use batch=1 for promotion/guardrail parity; larger batches are for fast visual review."
        ),
    )
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--json-top-k", type=int, default=10, help="Top detections to retain in JSON per row.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--review-out-dir",
        type=Path,
        default=None,
        help="Optional directory for top false-positive crop/overlay review artifacts.",
    )
    parser.add_argument(
        "--review-top-k",
        type=int,
        default=0,
        help="Top detections to export per model/source/conf row. Disabled by default.",
    )
    parser.add_argument(
        "--review-classes",
        default="",
        help="Optional comma/space-separated class names or ids to include in review artifacts.",
    )
    parser.add_argument(
        "--review-crop-pad",
        type=float,
        default=0.08,
        help="Fractional bbox padding for exported review crops.",
    )
    parser.add_argument(
        "--ignore-pred-class",
        action="append",
        default=[],
        help="Prediction class id or name to drop before counting false positives. Repeat or comma-separate.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def short_hash(value: str, *, length: int = 8) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def slug(value: str, *, max_length: int = 72) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    if not cleaned:
        cleaned = "item"
    if len(cleaned) > max_length:
        cleaned = f"{cleaned[: max_length - 9].rstrip('_')}_{short_hash(value)}"
    return cleaned


def conf_slug(conf: float) -> str:
    return f"{conf:g}".replace("-", "m").replace(".", "p")


def parse_review_classes(value: str) -> set[str]:
    return {item.strip() for item in re.split(r"[,\s]+", value) if item.strip()}


def ignored_prediction_class_ids(raw_values: list[str], names: dict[int, str]) -> set[int]:
    ignored: set[int] = set()
    name_to_id = {name: class_id for class_id, name in names.items()}
    for raw_value in raw_values:
        for token in re.split(r"[,\s]+", raw_value):
            if not token:
                continue
            if token in name_to_id:
                ignored.add(name_to_id[token])
                continue
            try:
                ignored.add(int(token))
            except ValueError as exc:
                raise SystemExit(f"unknown --ignore-pred-class value: {token}") from exc
    return ignored


def review_class_matches(args: argparse.Namespace, class_id: int, class_name: str) -> bool:
    tokens: set[str] = getattr(args, "review_class_tokens", set())
    if not tokens:
        return True
    lower_tokens: set[str] = getattr(args, "review_class_tokens_lower", set())
    return str(class_id) in tokens or class_name in tokens or class_name.lower() in lower_tokens


def source_short_label(source_label: str) -> str:
    match = re.search(r"#([^:#]+):", source_label)
    if match:
        return match.group(1)
    path = Path(source_label.replace("\\", "/"))
    return path.name or source_label


def parse_model(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
    else:
        path = Path(value.strip())
        label = path.parent.parent.name if path.name == "best.pt" else path.stem
    if not label:
        raise SystemExit(f"empty model label: {value!r}")
    resolved = resolve(path)
    if not resolved.exists():
        raise SystemExit(f"missing model: {resolved}")
    return label, resolved


def parse_confs(value: str) -> list[float]:
    confs = [float(item.strip()) for item in re.split(r"[,\s]+", value) if item.strip()]
    if not confs:
        raise SystemExit("--conf must include at least one threshold")
    return confs


def image_rows(root: Path) -> list[Path]:
    resolved = resolve(root)
    if not resolved.exists():
        raise SystemExit(f"missing image root: {resolved}")
    rows = [
        path
        for path in sorted(resolved.glob("*"))
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
        and path.stem not in REVIEW_ARTIFACT_STEMS
    ]
    if not rows:
        raise SystemExit(f"image root has no images: {resolved}")
    return rows


def image_list_rows(path: Path) -> list[Path]:
    resolved = resolve(path)
    if not resolved.exists():
        raise SystemExit(f"missing image list: {resolved}")
    rows: list[Path] = []
    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        image = Path(line)
        rows.append(image if image.is_absolute() else ROOT / image)
    if not rows:
        raise SystemExit(f"image list has no rows: {resolved}")
    return rows


def dataset_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw_root = Path(str(config.get("path", ".")))
    return raw_root if raw_root.is_absolute() else (config_path.parent / raw_root).resolve()


def split_root(config_path: Path, config: dict[str, Any], split_path: str) -> Path:
    path = Path(split_path)
    if path.is_absolute():
        return path
    return dataset_root(config_path, config) / path


def read_split_list(config_path: Path, config: dict[str, Any], split_path: str) -> list[Path]:
    list_path = split_root(config_path, config, split_path)
    images: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        images.append(path if path.is_absolute() else dataset_root(config_path, config) / path)
    return images


def split_images(config_path: Path, config: dict[str, Any], split_value: str | list[str]) -> list[Path]:
    split_paths = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for split_path in split_paths:
        resolved = split_root(config_path, config, str(split_path))
        if resolved.suffix.lower() == ".txt":
            images.extend(read_split_list(config_path, config, str(split_path)))
        else:
            images.extend(
                path
                for path in sorted(resolved.glob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            )
    return images


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def is_zero_label_image(image: Path) -> tuple[bool, bool]:
    label = label_path_for_image(image)
    if not label.exists():
        return True, True
    return not label.read_text(encoding="utf-8").strip(), False


def dataset_empty_label_source(config_path: Path, split: str) -> dict[str, Any]:
    resolved_config = resolve(config_path)
    if not resolved_config.exists():
        raise SystemExit(f"missing dataset YAML: {resolved_config}")
    config = yaml.safe_load(resolved_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"dataset YAML must be a mapping: {resolved_config}")
    if split not in config:
        raise SystemExit(f"dataset YAML has no split {split!r}: {resolved_config}")
    candidates = split_images(resolved_config, config, config[split])
    rows: list[Path] = []
    missing_labels = 0
    for image in candidates:
        is_zero, missing = is_zero_label_image(image)
        if missing:
            missing_labels += 1
        if is_zero:
            rows.append(image)
    if not rows:
        raise SystemExit(f"dataset split has no empty-label images: {repo_rel(resolved_config)}#{split}")
    return {
        "label": f"{repo_rel(resolved_config)}#{split}:empty-label",
        "images": rows,
        "candidate_images": len(candidates),
        "missing_labels": missing_labels,
    }


def class_names(model: YOLO) -> dict[int, str]:
    names = getattr(model.model, "names", None)
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def clipped_box(box: list[float], width: int, height: int, pad_fraction: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad = max(0.0, pad_fraction) * max(box_width, box_height)
    left = min(max(0, int(round(x1 - pad))), max(0, width - 1))
    top = min(max(0, int(round(y1 - pad))), max(0, height - 1))
    right = min(width, max(left + 1, int(round(x2 + pad))))
    bottom = min(height, max(top + 1, int(round(y2 + pad))))
    return left, top, right, bottom


def draw_label(draw: Any, text: str, x: int, y: int) -> None:
    try:
        bbox = draw.textbbox((x, y), text)
    except AttributeError:
        text_width, text_height = draw.textsize(text)
        bbox = (x, y, x + text_width, y + text_height)
    margin = 3
    draw.rectangle(
        (bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin),
        fill=(15, 15, 15),
    )
    draw.text((x, y), text, fill=(255, 255, 255))


def write_review_artifacts(
    *,
    args: argparse.Namespace,
    model_label: str,
    source_label: str,
    conf: float,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if args.review_out_dir is None or args.review_top_k <= 0:
        return {}
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit("Pillow is required for --review-out-dir artifacts") from exc

    selected = sorted(candidates, key=lambda row: float(row["confidence"]), reverse=True)[: args.review_top_k]
    source_label_short = source_short_label(source_label)
    artifact_dir = (
        resolve(args.review_out_dir)
        / slug(model_label)
        / f"{slug(source_label_short)}_{short_hash(source_label)}"
        / f"conf_{conf_slug(conf)}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        image_path = Path(row["image_path"])
        class_name = str(row["class"])
        class_dir = artifact_dir / slug(class_name)
        class_dir.mkdir(parents=True, exist_ok=True)
        image_key = f"{repo_rel(image_path)}:{row['bbox_xyxy']}:{class_name}:{row['confidence']:.6f}"
        filename_base = (
            f"rank{rank:03d}_{slug(class_name, max_length=32)}_"
            f"{float(row['confidence']):.3f}_{slug(image_path.stem, max_length=32)}_{short_hash(image_key)}"
        )
        crop_path = class_dir / f"{filename_base}_crop.jpg"
        overlay_path = class_dir / f"{filename_base}_overlay.jpg"

        with Image.open(image_path) as image_raw:
            image = image_raw.convert("RGB")
        width, height = image.size
        crop_box = clipped_box(row["bbox_xyxy"], width, height, args.review_crop_pad)
        crop = image.crop(crop_box)

        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = clipped_box(row["bbox_xyxy"], width, height, 0.0)
        line_width = max(2, min(width, height) // 180)
        for offset in range(line_width):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=(255, 40, 40))
        draw_label(draw, f"{class_name} {float(row['confidence']):.3f}", max(4, x1), max(4, y1 - 18))

        crop.save(crop_path, quality=92)
        overlay.save(overlay_path, quality=92)

        manifest_rows.append(
            {
                "rank": rank,
                "model_label": model_label,
                "source": source_label,
                "conf": conf,
                "image": repo_rel(image_path),
                "class": class_name,
                "class_id": int(row["class_id"]),
                "confidence": round(float(row["confidence"]), 6),
                "bbox_xyxy": [round(float(value), 2) for value in row["bbox_xyxy"]],
                "crop": repo_rel(crop_path),
                "overlay": repo_rel(overlay_path),
            }
        )

    manifest_path = artifact_dir / "review_manifest.json"
    csv_path = artifact_dir / "review.csv"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "model_label",
                "source",
                "conf",
                "image",
                "class",
                "class_id",
                "confidence",
                "bbox_xyxy",
                "crop",
                "overlay",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "artifact_dir": repo_rel(artifact_dir),
        "manifest": repo_rel(manifest_path),
        "csv": repo_rel(csv_path),
        "top_k": args.review_top_k,
        "classes": sorted(getattr(args, "review_class_tokens", set())),
        "count": len(manifest_rows),
    }


def probe_root(
    *,
    model: YOLO,
    names: dict[int, str],
    label: str,
    model_path: Path,
    source_label: str,
    images: list[Path],
    conf: float,
    ignored_class_ids: set[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    detections = 0
    raw_detections = 0
    ignored_detections = 0
    images_with_fp = 0
    by_class: Counter[str] = Counter()
    ignored_by_class: Counter[str] = Counter()
    top: list[dict[str, Any]] = []
    review_candidates: list[dict[str, Any]] = []
    review_enabled = args.review_out_dir is not None and args.review_top_k > 0
    with tempfile.TemporaryDirectory(prefix="cashsnap_fp_probe_") as tmp_dir:
        source_file = Path(tmp_dir) / "images.txt"
        source_file.write_text(
            "\n".join(path.resolve().as_posix() for path in images) + "\n",
            encoding="utf-8",
        )
        results = model.predict(
            source=str(source_file),
            imgsz=args.imgsz,
            batch=args.batch,
            conf=conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
            save=False,
            stream=True,
        )
        for image_path, result in zip(images, results, strict=True):
            boxes = result.boxes
            raw_detections += int(len(boxes))
            kept_count = 0
            if not len(boxes):
                continue
            classes = boxes.cls.cpu().numpy().astype(int).tolist()
            scores = boxes.conf.cpu().numpy().tolist()
            xyxy = boxes.xyxy.cpu().numpy().tolist()
            for class_id, score, box in zip(classes, scores, xyxy, strict=True):
                class_id = int(class_id)
                class_name = names.get(class_id, f"class_{class_id}")
                if class_id in ignored_class_ids:
                    ignored_detections += 1
                    ignored_by_class[class_name] += 1
                    continue
                kept_count += 1
                detections += 1
                by_class[class_name] += 1
                top.append(
                    {
                        "image": repo_rel(image_path),
                        "class": class_name,
                        "confidence": round(float(score), 6),
                    }
                )
                if review_enabled and review_class_matches(args, class_id, class_name):
                    review_candidates.append(
                        {
                            "image_path": image_path,
                            "class": class_name,
                            "class_id": class_id,
                            "confidence": float(score),
                            "bbox_xyxy": [float(value) for value in box],
                        }
                    )
            if kept_count:
                images_with_fp += 1
    top.sort(key=lambda row: float(row["confidence"]), reverse=True)
    row = {
        "model_label": label,
        "model": repo_rel(model_path),
        "image_root": source_label,
        "conf": conf,
        "images": len(images),
        "images_with_fp": images_with_fp,
        "detections": detections,
        "raw_detections": raw_detections,
        "ignored_detections": ignored_detections,
        "ignored_pred_class_ids": sorted(ignored_class_ids),
        "ignored_pred_class_names": [names.get(class_id, str(class_id)) for class_id in sorted(ignored_class_ids)],
        "fp_per_image": detections / len(images),
        "by_class": dict(sorted(by_class.items())),
        "ignored_by_class": dict(sorted(ignored_by_class.items())),
        "top": top[: args.json_top_k],
    }
    review = write_review_artifacts(
        args=args,
        model_label=label,
        source_label=source_label,
        conf=conf,
        candidates=review_candidates,
    )
    if review:
        row["review"] = review
    return row


def main() -> int:
    args = parse_args()
    if args.review_top_k < 0:
        raise SystemExit("--review-top-k must be >= 0")
    if args.json_top_k < 1:
        raise SystemExit("--json-top-k must be >= 1")
    if args.review_out_dir is not None and args.review_top_k <= 0:
        raise SystemExit("--review-out-dir requires --review-top-k > 0")
    args.review_class_tokens = parse_review_classes(args.review_classes)
    args.review_class_tokens_lower = {value.lower() for value in args.review_class_tokens}
    if not args.image_root and not args.image_list and not args.data:
        raise SystemExit("provide at least one --image-root, --image-list, or --data/--split source")
    if args.data and not args.split:
        raise SystemExit("--data requires at least one --split")
    models = [parse_model(value) for value in args.model]
    confs = parse_confs(args.conf)
    sources: list[dict[str, Any]] = []
    for root in args.image_root:
        rows = image_rows(root)
        sources.append(
            {
                "label": repo_rel(resolve(root)),
                "images": rows,
                "candidate_images": len(rows),
                "missing_labels": None,
            }
        )
    for image_list in args.image_list:
        rows = image_list_rows(image_list)
        sources.append(
            {
                "label": repo_rel(resolve(image_list)),
                "images": rows,
                "candidate_images": len(rows),
                "missing_labels": None,
            }
        )
    for data_path in args.data:
        for split in args.split:
            sources.append(dataset_empty_label_source(data_path, split))
    rows: list[dict[str, Any]] = []
    for model_label, model_path in models:
        model = YOLO(str(model_path))
        names = class_names(model)
        ignored_class_ids = ignored_prediction_class_ids(args.ignore_pred_class, names)
        for source in sources:
            images = source["images"]
            for conf in confs:
                row = probe_root(
                    model=model,
                    names=names,
                    label=model_label,
                    model_path=model_path,
                    source_label=source["label"],
                    images=images,
                    conf=conf,
                    ignored_class_ids=ignored_class_ids,
                    args=args,
                )
                rows.append(row)
                print(
                    f"{row['model_label']} conf={conf:g} root={row['image_root']} "
                    f"images_with_fp={row['images_with_fp']}/{row['images']} "
                    f"detections={row['detections']} fp_per_image={row['fp_per_image']:.3f} "
                    f"classes={row['by_class']}"
                )
                if "review" in row:
                    review = row["review"]
                    print(
                        f"wrote_review={review['artifact_dir']} "
                        f"count={review['count']} classes={review['classes']}"
                    )
    payload = {
        "schema": "cashsnap_yolo_background_false_positive_probe_v1",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "iou": args.iou,
        "max_det": args.max_det,
        "device": args.device,
        "sources": [
            {
                "label": source["label"],
                "images": len(source["images"]),
                "candidate_images": source.get("candidate_images"),
                "missing_labels": source.get("missing_labels"),
            }
            for source in sources
        ],
        "rows": rows,
    }
    if args.json_out is not None:
        json_out = resolve(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote_json={repo_rel(json_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
