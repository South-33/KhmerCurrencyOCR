#!/usr/bin/env python
"""Compare real and synthetic image/label statistics inside a YOLO dataset split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="YOLO data YAML.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--image-csv-out", type=Path, default=None)
    parser.add_argument("--box-csv-out", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return document


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    root_value = Path(str(config.get("path", "."))).expanduser()
    return root_value if root_value.is_absolute() else (config_path.parent / root_value).resolve()


def split_root(dataset_root: Path, split_path: str) -> Path:
    path = Path(split_path)
    return path if path.is_absolute() else dataset_root / path


def read_split_list(dataset_root: Path, split_path: str) -> list[Path]:
    list_path = split_root(dataset_root, split_path)
    images: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        images.append(path if path.is_absolute() else dataset_root / path)
    return images


def iter_split_images(dataset_root: Path, split_value: str | list[str]) -> list[Path]:
    split_paths = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for split_path in split_paths:
        resolved = split_root(dataset_root, str(split_path))
        if resolved.suffix.lower() == ".txt":
            images.extend(read_split_list(dataset_root, str(split_path)))
        else:
            images.extend(
                sorted(path for path in resolved.glob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
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


def source_group(image: Path) -> str:
    rel = repo_rel(image)
    if rel.startswith("data/synthetic/"):
        parts = rel.split("/")
        return "synthetic:" + parts[2] if len(parts) > 2 else "synthetic"
    if rel.startswith("data/cashsnap_v1/"):
        return "real"
    return "other"


def source_family(group: str) -> str:
    return "synthetic" if group.startswith("synthetic:") else group


def class_name(names: dict[Any, Any] | list[Any], class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, list) and class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def image_stats(image: Path) -> dict[str, Any]:
    with Image.open(image) as handle:
        rgb = handle.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0

    height, width = array.shape[:2]
    luma = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    max_rgb = array.max(axis=2)
    min_rgb = array.min(axis=2)
    saturation = np.divide(max_rgb - min_rgb, np.maximum(max_rgb, 1e-6))
    dx = np.diff(luma, axis=1)
    dy = np.diff(luma, axis=0)
    sharpness = float(dx.var() + dy.var())
    return {
        "width": width,
        "height": height,
        "aspect": width / height,
        "luma_mean": float(luma.mean()),
        "luma_std": float(luma.std()),
        "luma_p05": float(np.quantile(luma, 0.05)),
        "luma_p95": float(np.quantile(luma, 0.95)),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "sharpness_grad_var": sharpness,
    }


def label_rows(image: Path, names: dict[Any, Any] | list[Any]) -> list[dict[str, Any]]:
    label = label_path_for_image(image)
    if not label.exists():
        raise FileNotFoundError(f"missing label for {image}: {label}")
    rows = []
    for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label}:{line_no} expected 5 YOLO fields, found {len(parts)}")
        class_id = int(parts[0])
        width = float(parts[3])
        height = float(parts[4])
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name(names, class_id),
                "x_center": float(parts[1]),
                "y_center": float(parts[2]),
                "box_width": width,
                "box_height": height,
                "box_area": width * height,
                "box_aspect": width / height if height else None,
            }
        )
    return rows


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    average = sum(values) / len(values)
    return float(math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1)))


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def summarize_numeric(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        summary[key] = {
            "mean": mean(values),
            "stdev": stdev(values),
            "p05": quantile(values, 0.05),
            "p50": quantile(values, 0.50),
            "p95": quantile(values, 0.95),
        }
    return summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(image_rows: list[dict[str, Any]], box_rows: list[dict[str, Any]]) -> dict[str, Any]:
    image_keys = [
        "width",
        "height",
        "aspect",
        "luma_mean",
        "luma_std",
        "luma_p05",
        "luma_p95",
        "saturation_mean",
        "saturation_std",
        "sharpness_grad_var",
    ]
    box_keys = ["box_width", "box_height", "box_area", "box_aspect"]
    groups = sorted({row["source_group"] for row in image_rows})
    families = sorted({row["source_family"] for row in image_rows})

    def group_summary(group_key: str, group_value: str) -> dict[str, Any]:
        group_images = [row for row in image_rows if row[group_key] == group_value]
        group_boxes = [row for row in box_rows if row[group_key] == group_value]
        class_counts = Counter(row["class_name"] for row in group_boxes)
        return {
            "images": len(group_images),
            "backgrounds": sum(1 for row in group_images if int(row["box_count"]) == 0),
            "boxes": len(group_boxes),
            "class_counts": dict(sorted(class_counts.items())),
            "image_stats": summarize_numeric(group_images, image_keys),
            "box_stats": summarize_numeric(group_boxes, box_keys),
        }

    by_group = {group: group_summary("source_group", group) for group in groups}
    by_family = {family: group_summary("source_family", family) for family in families}
    deltas: dict[str, Any] = {}
    if "real" in by_family and "synthetic" in by_family:
        deltas["synthetic_minus_real"] = {}
        for section in ["image_stats", "box_stats"]:
            deltas["synthetic_minus_real"][section] = {}
            for metric, real_stats in by_family["real"][section].items():
                synth_stats = by_family["synthetic"][section].get(metric, {})
                real_mean = real_stats.get("mean")
                synth_mean = synth_stats.get("mean")
                deltas["synthetic_minus_real"][section][metric] = (
                    None if real_mean is None or synth_mean is None else synth_mean - real_mean
                )
    return {
        "by_family": by_family,
        "by_group": by_group,
        "deltas": deltas,
    }


def main() -> int:
    args = parse_args()
    data_path = resolve(args.data)
    config = read_yaml(data_path)
    root = data_root(data_path, config)
    if args.split not in config:
        raise SystemExit(f"split {args.split!r} missing from {data_path}")
    names = config.get("names", {})
    images = iter_split_images(root, config[args.split])
    if args.max_images is not None:
        images = images[: args.max_images]

    image_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    for image in images:
        group = source_group(image)
        family = source_family(group)
        boxes = label_rows(image, names)
        row = {
            "image": repo_rel(image),
            "source_group": group,
            "source_family": family,
            "box_count": len(boxes),
            **image_stats(image),
        }
        image_rows.append(row)
        for box in boxes:
            box_rows.append(
                {
                    "image": repo_rel(image),
                    "source_group": group,
                    "source_family": family,
                    **box,
                }
            )

    payload = {
        "data": repo_rel(data_path),
        "split": args.split,
        "images": len(image_rows),
        "boxes": len(box_rows),
        **summarize(image_rows, box_rows),
    }
    if args.json_out:
        write_json(resolve(args.json_out), payload)
    if args.image_csv_out:
        write_csv(resolve(args.image_csv_out), image_rows)
    if args.box_csv_out:
        write_csv(resolve(args.box_csv_out), box_rows)

    print(
        f"images={payload['images']} boxes={payload['boxes']} "
        f"families={','.join(sorted(payload['by_family']))}",
        flush=True,
    )
    if args.json_out:
        print(f"wrote_json={repo_rel(resolve(args.json_out))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
