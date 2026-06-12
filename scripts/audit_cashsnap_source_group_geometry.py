#!/usr/bin/env python
"""Summarize CashSnap split geometry by source group and class.

This is a planning audit for source-context synthetic rebuilds. It does not run
models; it inventories the real YOLO splits so replacement or unknown-aware
generators can choose source groups/classes with explicit geometry and label
density instead of relying on filename hunches.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SOURCE_PREFIXES = {
    "asian_currency_": "asian_currency",
    "billsbank_": "billsbank",
    "cambodia_currency_project_": "cambodia_currency_project",
    "cashcountingxl_": "cashcountingxl",
    "khmer_us_currency_": "khmer_us_currency",
    "usd_total_": "usd_total",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("configs/cashsnap_v1.yaml"))
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help="YOLO split to audit. Repeatable; defaults to train, val, test.",
    )
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--examples-per-bucket", type=int, default=5)
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Open every image with PIL to count unreadable files. Slower; off by default.",
    )
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


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"YOLO data config must be a mapping: {repo_rel(path)}")
    return payload


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


def source_group_for_image(image_path: Path) -> str:
    name = image_path.name.lower()
    for prefix, group in SOURCE_PREFIXES.items():
        if name.startswith(prefix):
            return group
    return name.split("_", 1)[0] if "_" in name else "unknown"


def verify_image(path: Path) -> None:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()


def read_labels(label_path: Path, names: dict[int, str]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    if not label_path.exists():
        return labels
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no}: expected 5 YOLO fields")
        class_id = int(float(parts[0]))
        if class_id not in names:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no}: class id {class_id} outside schema")
        cx, cy, width, height = [float(value) for value in parts[1:]]
        if width <= 0.0 or height <= 0.0:
            continue
        labels.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
                "area_ratio": width * height,
            }
        )
    return labels


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


class Bucket:
    def __init__(self, examples_per_bucket: int) -> None:
        self.examples_per_bucket = examples_per_bucket
        self.images: set[str] = set()
        self.labeled_images: set[str] = set()
        self.boxes = 0
        self.area_ratios: list[float] = []
        self.short_at_imgsz: list[float] = []
        self.boxes_per_image: Counter[int] = Counter()
        self.class_count_per_image: Counter[int] = Counter()
        self.examples: list[str] = []

    def add_image(self, image_path: Path, labels: list[dict[str, Any]], imgsz: int, *, class_name: str | None = None) -> None:
        rel = repo_rel(image_path)
        selected = [label for label in labels if class_name is None or label["class_name"] == class_name]
        if not selected and class_name is not None:
            return
        self.images.add(rel)
        if labels:
            self.labeled_images.add(rel)
        self.boxes_per_image[len(labels)] += 1
        self.class_count_per_image[len({label["class_name"] for label in labels})] += 1
        if selected and len(self.examples) < self.examples_per_bucket:
            self.examples.append(rel)
        for label in selected:
            self.boxes += 1
            self.area_ratios.append(float(label["area_ratio"]))
            self.short_at_imgsz.append(min(float(label["width"]), float(label["height"])) * float(imgsz))

    def summary(self) -> dict[str, Any]:
        empty_images = len(self.images) - len(self.labeled_images)
        area = self.area_ratios
        short = self.short_at_imgsz
        return {
            "images": len(self.images),
            "labeled_images": len(self.labeled_images),
            "empty_images": empty_images,
            "boxes": self.boxes,
            "box_density": self.boxes / max(1, len(self.images)),
            "boxes_per_image_hist": dict(sorted(self.boxes_per_image.items())),
            "class_count_per_image_hist": dict(sorted(self.class_count_per_image.items())),
            "area_mean": (sum(area) / len(area)) if area else None,
            "area_p10": percentile(area, 0.10),
            "area_p50": percentile(area, 0.50),
            "area_p90": percentile(area, 0.90),
            "area_ge50": sum(1 for value in area if value >= 0.50),
            "area_ge90": sum(1 for value in area if value >= 0.90),
            "short_at_imgsz_p10": percentile(short, 0.10),
            "short_at_imgsz_p50": percentile(short, 0.50),
            "short_at_imgsz_p90": percentile(short, 0.90),
            "short_at_imgsz_lt80": sum(1 for value in short if value < 80.0),
            "examples": self.examples,
        }


def flatten_summary(level: str, split: str, source_group: str, class_name: str, row: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "level": level,
        "split": split,
        "source_group": source_group,
        "class_name": class_name,
    }
    for key in [
        "images",
        "labeled_images",
        "empty_images",
        "boxes",
        "box_density",
        "area_mean",
        "area_p10",
        "area_p50",
        "area_p90",
        "area_ge50",
        "area_ge90",
        "short_at_imgsz_p10",
        "short_at_imgsz_p50",
        "short_at_imgsz_p90",
        "short_at_imgsz_lt80",
    ]:
        flat[key] = row.get(key)
    flat["examples"] = "|".join(row.get("examples") or [])
    flat["boxes_per_image_hist"] = json.dumps(row.get("boxes_per_image_hist") or {}, sort_keys=True)
    flat["class_count_per_image_hist"] = json.dumps(row.get("class_count_per_image_hist") or {}, sort_keys=True)
    return flat


def main() -> int:
    args = parse_args()
    if args.imgsz <= 0:
        raise SystemExit("--imgsz must be positive")
    if args.examples_per_bucket < 0:
        raise SystemExit("--examples-per-bucket must be >= 0")

    data_path = resolve(args.data)
    config = load_config(data_path)
    names = parse_names(config)
    splits = args.split or ["train", "val", "test"]

    by_split_source: dict[str, dict[str, Bucket]] = defaultdict(dict)
    by_split_source_class: dict[str, dict[str, Bucket]] = defaultdict(dict)
    split_totals: dict[str, dict[str, Any]] = {}

    for split in splits:
        images = split_images(data_path, config, split)
        missing_images = 0
        unreadable_images = 0
        total_boxes = 0
        empty_images = 0
        label_count_hist: Counter[int] = Counter()
        source_counts: Counter[str] = Counter()
        for image_path in images:
            if not image_path.exists():
                missing_images += 1
                continue
            if args.verify_images:
                try:
                    verify_image(image_path)
                except Exception:
                    unreadable_images += 1
                    continue
            labels = read_labels(label_path_for_image(image_path), names)
            source_group = source_group_for_image(image_path)
            source_counts[source_group] += 1
            label_count_hist[len(labels)] += 1
            total_boxes += len(labels)
            if not labels:
                empty_images += 1

            source_bucket = by_split_source[split].setdefault(source_group, Bucket(args.examples_per_bucket))
            source_bucket.add_image(image_path, labels, args.imgsz)
            for class_name in sorted({label["class_name"] for label in labels}):
                key = f"{source_group}|{class_name}"
                bucket = by_split_source_class[split].setdefault(key, Bucket(args.examples_per_bucket))
                bucket.add_image(image_path, labels, args.imgsz, class_name=class_name)

        split_totals[split] = {
            "images": len(images),
            "readable_images": len(images) - missing_images - unreadable_images,
            "missing_images": missing_images,
            "unreadable_images": unreadable_images,
            "empty_images": empty_images,
            "boxes": total_boxes,
            "label_count_hist": dict(sorted(label_count_hist.items())),
            "source_image_counts": dict(source_counts.most_common()),
        }

    source_payload: dict[str, dict[str, Any]] = {}
    source_class_payload: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    for split, buckets in by_split_source.items():
        source_payload[split] = {}
        for source_group, bucket in sorted(buckets.items()):
            row = bucket.summary()
            source_payload[split][source_group] = row
            csv_rows.append(flatten_summary("source", split, source_group, "", row))
    for split, buckets in by_split_source_class.items():
        source_class_payload[split] = {}
        for key, bucket in sorted(buckets.items()):
            source_group, class_name = key.split("|", 1)
            row = bucket.summary()
            source_class_payload[split][key] = {"source_group": source_group, "class_name": class_name, **row}
            csv_rows.append(flatten_summary("source_class", split, source_group, class_name, row))

    payload = {
        "schema": "cashsnap_source_group_geometry_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": repo_rel(data_path),
        "args": {
            "splits": splits,
            "imgsz": args.imgsz,
            "examples_per_bucket": args.examples_per_bucket,
            "verify_images": args.verify_images,
        },
        "names": names,
        "split_totals": split_totals,
        "by_split_source": source_payload,
        "by_split_source_class": source_class_payload,
    }

    json_out = resolve(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote_json={repo_rel(json_out)}")
    for split, totals in split_totals.items():
        print(
            "source_group_geometry "
            f"split={split} images={totals['images']} empty={totals['empty_images']} boxes={totals['boxes']} "
            f"sources={len(totals['source_image_counts'])}"
        )

    if args.csv_out:
        csv_out = resolve(args.csv_out)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "level",
            "split",
            "source_group",
            "class_name",
            "images",
            "labeled_images",
            "empty_images",
            "boxes",
            "box_density",
            "area_mean",
            "area_p10",
            "area_p50",
            "area_p90",
            "area_ge50",
            "area_ge90",
            "short_at_imgsz_p10",
            "short_at_imgsz_p50",
            "short_at_imgsz_p90",
            "short_at_imgsz_lt80",
            "boxes_per_image_hist",
            "class_count_per_image_hist",
            "examples",
        ]
        with csv_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"wrote_csv={repo_rel(csv_out)} rows={len(csv_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
