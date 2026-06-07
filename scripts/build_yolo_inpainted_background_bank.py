#!/usr/bin/env python
"""Build train-only inpainted background canvases from YOLO-labeled images."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None, help="JSONL/CSV/list with image rows.")
    parser.add_argument("--data", type=Path, default=None, help="YOLO data YAML used when --manifest is absent.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--suffix", default="train", help="Output filename suffix for background split filtering.")
    parser.add_argument("--pad-fraction", type=float, default=0.10)
    parser.add_argument("--inpaint-radius", type=float, default=7.0)
    parser.add_argument("--mask-dilate-px", type=int, default=5)
    parser.add_argument(
        "--max-mask-fraction",
        type=float,
        default=0.55,
        help="Skip images whose erased YOLO boxes cover more than this fraction; <=0 disables.",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
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


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed = (ROOT / "data" / "backgrounds").resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise SystemExit(f"Refusing to clean outside {allowed}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_manifest_images(path: Path) -> list[Path]:
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


def read_yaml_images(path: Path, split: str) -> list[Path]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"YOLO data YAML must be a mapping: {repo_rel(path)}")
    root = Path(str(config.get("path", ".")))
    root = root if root.is_absolute() else path.parent / root
    split_value = config.get(split)
    if split_value is None:
        raise SystemExit(f"{repo_rel(path)} has no split {split!r}")
    split_values = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for raw in split_values:
        split_path = Path(str(raw))
        split_path = split_path if split_path.is_absolute() else root / split_path
        if split_path.suffix.lower() == ".txt":
            for raw_line in split_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line and not line.startswith("#"):
                    image = Path(line)
                    images.append(image if image.is_absolute() else root / image)
        else:
            images.extend(
                sorted(path for path in split_path.glob("*") if path.suffix.lower() in IMAGE_EXTS)
            )
    return unique_images(images)


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


def read_yolo_boxes(label_path: Path, image_width: int, image_height: int) -> list[tuple[int, int, int, int]]:
    if not label_path.exists():
        return []
    boxes = []
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} expected 5 YOLO fields")
        _cls, cx, cy, width, height = parts
        cx_f, cy_f, w_f, h_f = (float(cx), float(cy), float(width), float(height))
        x1 = int(round((cx_f - w_f / 2.0) * image_width))
        y1 = int(round((cy_f - h_f / 2.0) * image_height))
        x2 = int(round((cx_f + w_f / 2.0) * image_width))
        y2 = int(round((cy_f + h_f / 2.0) * image_height))
        boxes.append((x1, y1, x2, y2))
    return boxes


def mask_from_boxes(
    boxes: list[tuple[int, int, int, int]],
    shape: tuple[int, int],
    pad_fraction: float,
    dilate_px: int,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = int(round(box_w * pad_fraction))
        pad_y = int(round(box_h * pad_fraction))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)
        mask[y1:y2, x1:x2] = 255
    if dilate_px > 0 and mask.any():
        kernel_size = dilate_px * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def main() -> None:
    args = parse_args()
    out_root = resolve(args.out_root)
    if args.clean:
        safe_clean(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.manifest is not None:
        images = read_manifest_images(resolve(args.manifest))
        source = repo_rel(resolve(args.manifest))
    elif args.data is not None:
        images = read_yaml_images(resolve(args.data), args.split)
        source = f"{repo_rel(resolve(args.data))}#{args.split}"
    else:
        raise SystemExit("Provide --manifest or --data")
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise SystemExit("No input images")

    records: list[dict[str, Any]] = []
    skipped = Counter()
    for index, image_path in enumerate(images):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped["unreadable"] += 1
            continue
        height, width = image.shape[:2]
        label_path = label_path_for_image(image_path)
        boxes = read_yolo_boxes(label_path, width, height)
        if not boxes:
            skipped["no_boxes"] += 1
            continue
        mask = mask_from_boxes(boxes, (height, width), args.pad_fraction, args.mask_dilate_px)
        mask_fraction = float((mask > 0).mean())
        if args.max_mask_fraction > 0 and mask_fraction > args.max_mask_fraction:
            skipped["mask_too_large"] += 1
            continue
        inpainted = cv2.inpaint(image, mask, args.inpaint_radius, cv2.INPAINT_TELEA)
        stem = f"{image_path.stem}_inpaint_{args.suffix}"
        out_image = out_root / f"{stem}.jpg"
        out_label = out_root / f"{stem}.txt"
        cv2.imwrite(str(out_image), inpainted, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        out_label.write_text("", encoding="utf-8")
        records.append(
            {
                "image": repo_rel(out_image),
                "label": repo_rel(out_label),
                "source_image": repo_rel(image_path),
                "source_label": repo_rel(label_path),
                "boxes_erased": len(boxes),
                "mask_fraction": mask_fraction,
                "width": width,
                "height": height,
            }
        )
        if (index + 1) % 100 == 0:
            print(f"processed {index + 1}/{len(images)}", flush=True)

    summary = {
        "schema": "cashsnap_yolo_inpainted_background_bank_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "out_root": repo_rel(out_root),
        "images": len(records),
        "input_images": len(images),
        "skipped": dict(skipped),
        "suffix": args.suffix,
        "pad_fraction": args.pad_fraction,
        "inpaint_radius": args.inpaint_radius,
        "mask_dilate_px": args.mask_dilate_px,
        "max_mask_fraction": args.max_mask_fraction,
        "records": records,
    }
    (out_root / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"ok: inpainted_backgrounds={len(records)} skipped={dict(skipped)} "
        f"out={repo_rel(out_root)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
