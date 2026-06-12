#!/usr/bin/env python
"""Build an ImageFolder crop dataset from YOLO labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from local_runtime import configure_project_cache


configure_project_cache()


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--source",
        nargs=4,
        action="append",
        metavar=("DATA_YAML", "SOURCE_SPLIT", "OUT_SPLIT", "MAX_CROPS_PER_CLASS"),
        required=True,
        help="Add YOLO labels from DATA_YAML SOURCE_SPLIT into ImageFolder OUT_SPLIT.",
    )
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--min-side", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--crop-variants",
        choices=["full", "full_and_fragments_v1"],
        default="full",
        help="Whether to emit only full label crops or also deterministic partial/edge fragments.",
    )
    parser.add_argument(
        "--fragment-out-split",
        action="append",
        default=[],
        help="ImageFolder split that should receive fragment variants when --crop-variants enables them.",
    )
    parser.add_argument("--fragments-per-label", type=int, default=2)
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
    allowed_root = (ROOT / "data").resolve()
    if not (resolved == allowed_root or allowed_root in resolved.parents):
        raise SystemExit(f"Refusing to clean outside data/: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"YOLO data YAML must be a mapping: {repo_rel(path)}")
    return payload


def parse_names(config: dict[str, Any]) -> dict[int, str]:
    raw_names = config.get("names")
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
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


def yolo_to_xyxy(values: list[float], width: int, height: int) -> list[float]:
    cx, cy, bw, bh = values
    return [
        (cx - bw / 2.0) * width,
        (cy - bh / 2.0) * height,
        (cx + bw / 2.0) * width,
        (cy + bh / 2.0) * height,
    ]


def read_labels(label_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not label_path.exists():
        return rows
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} expected 5 YOLO fields")
        class_id = int(parts[0])
        rows.append(
            {
                "class_id": class_id,
                "yolo": [float(value) for value in parts[1:]],
            }
        )
    return rows


def padded_crop_box(box: list[float], image_size: tuple[int, int], padding: float) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * max(0.0, padding)
    pad_y = (y2 - y1) * max(0.0, padding)
    return (
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(width, int(round(x2 + pad_x))),
        min(height, int(round(y2 + pad_y))),
    )


def fragment_boxes_v1(box: list[float]) -> list[tuple[str, list[float]]]:
    x1, y1, x2, y2 = box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return [
        ("left_strip", [x1, y1, x1 + width * 0.45, y2]),
        ("right_strip", [x2 - width * 0.45, y1, x2, y2]),
        ("top_strip", [x1, y1, x2, y1 + height * 0.45]),
        ("bottom_strip", [x1, y2 - height * 0.45, x2, y2]),
        ("center_crop", [x1 + width * 0.15, y1 + height * 0.15, x2 - width * 0.15, y2 - height * 0.15]),
        ("top_left", [x1, y1, x1 + width * 0.60, y1 + height * 0.60]),
        ("top_right", [x2 - width * 0.60, y1, x2, y1 + height * 0.60]),
        ("bottom_left", [x1, y2 - height * 0.60, x1 + width * 0.60, y2]),
        ("bottom_right", [x2 - width * 0.60, y2 - height * 0.60, x2, y2]),
    ]


def crop_variants_for_label(
    label: dict[str, Any],
    *,
    out_split: str,
    crop_variants: str,
    fragment_out_splits: set[str],
    fragments_per_label: int,
    seed: int,
    image_path: Path,
    label_index: int,
) -> list[tuple[str, list[float]]]:
    variants = [("full", label["xyxy"])]
    if crop_variants == "full_and_fragments_v1" and out_split in fragment_out_splits and fragments_per_label > 0:
        fragments = fragment_boxes_v1(label["xyxy"])
        fragments.sort(key=lambda item: stable_score(seed, image_path, label_index, item[0]))
        variants.extend(fragments[:fragments_per_label])
    return variants


def stable_score(seed: int, image: Path, label_index: int, variant: str = "full") -> str:
    key = f"{seed}|{repo_rel(image)}|{label_index}|{variant}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def select_labels(
    rows: list[tuple[str, Path, int, dict[str, Any]]],
    max_per_class: int,
    seed: int,
) -> list[tuple[str, Path, int, dict[str, Any]]]:
    if max_per_class <= 0:
        return rows
    selected: list[tuple[str, Path, int, dict[str, Any]]] = []
    grouped: dict[str, list[tuple[str, Path, int, dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row[0], []).append(row)
    for class_name, class_rows in sorted(grouped.items()):
        class_rows.sort(key=lambda item: stable_score(seed, item[1], item[2]))
        selected.extend(class_rows[:max_per_class])
    selected.sort(key=lambda item: (item[0], repo_rel(item[1]), item[2]))
    return selected


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_source_rows(
    *,
    data_path: Path,
    source_split: str,
    out_split: str,
    max_per_class: int,
    seed: int,
) -> tuple[list[tuple[str, Path, int, dict[str, Any]]], dict[int, str]]:
    config = load_config(data_path)
    names = parse_names(config)
    candidates: list[tuple[str, Path, int, dict[str, Any]]] = []
    for image_path in split_images(data_path, config, source_split):
        labels = read_labels(label_path_for_image(image_path))
        for label_index, label in enumerate(labels):
            class_name = names.get(int(label["class_id"]), str(label["class_id"]))
            candidates.append((class_name, image_path, label_index, label))
    return select_labels(candidates, max_per_class, seed), names


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out)
    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    summary_counts: Counter[str] = Counter()
    class_names: set[str] = set()
    source_specs: list[dict[str, Any]] = []
    fragment_out_splits = {str(value) for value in args.fragment_out_split}

    for raw_data, source_split, out_split, raw_max in args.source:
        data_path = resolve(raw_data)
        max_per_class = int(raw_max)
        rows, names = collect_source_rows(
            data_path=data_path,
            source_split=source_split,
            out_split=out_split,
            max_per_class=max_per_class,
            seed=args.seed,
        )
        source_specs.append(
            {
                "data": repo_rel(data_path),
                "source_split": source_split,
                "out_split": out_split,
                "max_crops_per_class": max_per_class,
                "selected_crops": len(rows),
            }
        )
        for class_name, image_path, label_index, label in rows:
            with Image.open(image_path).convert("RGB") as image:
                label_xyxy = yolo_to_xyxy(label["yolo"], *image.size)
                crop_label = {**label, "xyxy": label_xyxy}
                variants = crop_variants_for_label(
                    crop_label,
                    out_split=out_split,
                    crop_variants=args.crop_variants,
                    fragment_out_splits=fragment_out_splits,
                    fragments_per_label=args.fragments_per_label,
                    seed=args.seed,
                    image_path=image_path,
                    label_index=label_index,
                )
                for variant_name, variant_box in variants:
                    crop_box = padded_crop_box(variant_box, image.size, args.padding)
                    if crop_box[2] - crop_box[0] < args.min_side or crop_box[3] - crop_box[1] < args.min_side:
                        continue
                    crop = image.crop(crop_box).copy()
                    class_names.add(class_name)
                    split_class_dir = out_dir / out_split / class_name
                    split_class_dir.mkdir(parents=True, exist_ok=True)
                    stem = (
                        f"{Path(image_path).stem}_l{label_index:02d}_{variant_name}_"
                        f"{stable_score(args.seed, image_path, label_index, variant_name)[:10]}"
                    )
                    crop_path = split_class_dir / f"{stem}.jpg"
                    crop.save(crop_path, quality=92)
                    summary_counts[f"{out_split}/{class_name}"] += 1
                    manifest_rows.append(
                        {
                            "split": out_split,
                            "class_name": class_name,
                            "class_id": int(label["class_id"]),
                            "crop_path": repo_rel(crop_path),
                            "image_path": repo_rel(image_path),
                            "label_index": label_index,
                            "crop_variant": variant_name,
                            "crop_box_xyxy": json.dumps(list(crop_box)),
                            "source_data": repo_rel(data_path),
                            "source_split": source_split,
                        }
                    )

    for split in sorted({str(spec["out_split"]) for spec in source_specs}):
        for class_name in sorted(class_names):
            (out_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    write_manifest(out_dir / "manifest.csv", manifest_rows)
    summary = {
        "schema": "cashsnap_yolo_crop_imagefolder_dataset_v1",
        "out": repo_rel(out_dir),
        "padding": args.padding,
        "min_side": args.min_side,
        "seed": args.seed,
        "crop_variants": args.crop_variants,
        "fragment_out_splits": sorted(fragment_out_splits),
        "fragments_per_label": args.fragments_per_label,
        "sources": source_specs,
        "classes": sorted(class_names),
        "counts": dict(sorted(summary_counts.items())),
        "total_crops": len(manifest_rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"crop_imagefolder={repo_rel(out_dir)} crops={len(manifest_rows)} classes={len(class_names)}")


if __name__ == "__main__":
    main()
