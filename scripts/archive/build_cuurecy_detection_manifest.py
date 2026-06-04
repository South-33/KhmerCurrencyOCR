from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CORE_CLASSES = {
    "USD_1",
    "USD_5",
    "USD_10",
    "USD_20",
    "USD_50",
    "USD_100",
    "KHR_500",
    "KHR_1000",
    "KHR_2000",
    "KHR_5000",
    "KHR_10000",
    "KHR_20000",
    "KHR_50000",
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
SPLITS = ("train", "valid", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact object manifest for the Roboflow cuurecy-detection-is export."
    )
    parser.add_argument(
        "--data",
        default="data/raw_datasets/roboflow_cuurecy_detection_is/data.yaml",
        help="Path to the Roboflow data.yaml.",
    )
    parser.add_argument(
        "--out-csv",
        default="data/processed/roboflow_cuurecy_detection_is/manifest.csv",
    )
    parser.add_argument(
        "--out-json",
        default="data/processed/roboflow_cuurecy_detection_is/manifest_summary.json",
    )
    parser.add_argument("--edge-margin", type=float, default=0.01)
    parser.add_argument("--tiny-bbox-area", type=float, default=0.0025)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_config(data_yaml: Path) -> tuple[Path, dict[int, str]]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = (data_yaml.parent / config["path"]).resolve() if "path" in config else data_yaml.parent.resolve()
    raw_names = config.get("names", {})
    if isinstance(raw_names, dict):
        names = {int(key): str(value) for key, value in raw_names.items()}
    else:
        names = {index: str(value) for index, value in enumerate(raw_names)}
    return root, names


def split_dirs(root: Path, split: str) -> tuple[str, Path, Path] | None:
    normalized_split = "valid" if split == "val" else split
    candidates = [
        (root / split / "labels", root / split / "images"),
        (root / "labels" / split, root / "images" / split),
    ]
    for label_dir, image_dir in candidates:
        if label_dir.exists() and image_dir.exists():
            return normalized_split, label_dir, image_dir
    return None


def find_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def parse_raw_class(raw_name: str) -> dict[str, str]:
    parts = raw_name.lower().split("-")
    side_code = parts[-1] if parts and parts[-1] in {"f", "b"} else ""
    side = {"f": "front", "b": "back"}.get(side_code, "unknown")
    amount = next((part for part in parts if part.isdigit()), "")
    if "us" in parts:
        currency = "USD"
    elif "riel" in parts:
        currency = "KHR"
    else:
        currency = "unknown"
    canonical_name = f"{currency}_{amount}" if currency != "unknown" and amount else ""
    return {
        "currency": currency,
        "denomination": amount,
        "side": side,
        "canonical_name": canonical_name,
        "cashsnap_core": str(canonical_name in CORE_CLASSES).lower(),
    }


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def shape_stats(coords: list[float], edge_margin: float, tiny_bbox_area: float) -> dict[str, Any]:
    points = list(zip(coords[0::2], coords[1::2], strict=False))
    xs = coords[0::2]
    ys = coords[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    bbox_area = bbox_w * bbox_h
    edge_touch = x1 <= edge_margin or y1 <= edge_margin or x2 >= 1.0 - edge_margin or y2 >= 1.0 - edge_margin
    return {
        "points": len(points),
        "bbox_xc": (x1 + x2) / 2,
        "bbox_yc": (y1 + y2) / 2,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "bbox_area": bbox_area,
        "polygon_area": polygon_area(points),
        "edge_touch": str(edge_touch).lower(),
        "tiny_bbox": str(bbox_area < tiny_bbox_area).lower(),
    }


def build_rows(data_yaml: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root, names = load_config(data_yaml)
    rows: list[dict[str, Any]] = []
    image_paths: set[str] = set()
    skipped = Counter()

    for split in SPLITS:
        dirs = split_dirs(root, split)
        if dirs is None:
            continue
        normalized_split, label_dir, image_dir = dirs
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = find_image(image_dir, label_path.stem)
            if image_path is None:
                skipped["missing_image"] += 1
                continue
            image_paths.add(rel(image_path))
            for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                parts = raw_line.split()
                if not parts:
                    continue
                try:
                    raw_class_id = int(float(parts[0]))
                    coords = [float(value) for value in parts[1:]]
                except ValueError:
                    skipped["parse_error"] += 1
                    continue
                if len(coords) < 6 or len(coords) % 2 != 0:
                    skipped["not_segmentation"] += 1
                    continue

                raw_name = names.get(raw_class_id, str(raw_class_id))
                parsed = parse_raw_class(raw_name)
                rows.append(
                    {
                        "source_dataset": "roboflow_cuurecy_detection_is",
                        "split": normalized_split,
                        "image_path": rel(image_path),
                        "label_path": rel(label_path),
                        "line_number": line_number,
                        "raw_class_id": raw_class_id,
                        "raw_class_name": raw_name,
                        **parsed,
                        **shape_stats(coords, args.edge_margin, args.tiny_bbox_area),
                    }
                )

    summary = {
        "dataset": rel(root),
        "objects": len(rows),
        "images": len(image_paths),
        "skipped": dict(skipped),
        "by_split": dict(Counter(row["split"] for row in rows)),
        "by_raw_class": dict(Counter(row["raw_class_name"] for row in rows)),
        "by_canonical": dict(Counter(row["canonical_name"] for row in rows)),
        "by_side": dict(Counter(row["side"] for row in rows)),
        "cashsnap_core_objects": sum(1 for row in rows if row["cashsnap_core"] == "true"),
        "non_core_objects": sum(1 for row in rows if row["cashsnap_core"] == "false"),
        "edge_touch_objects": sum(1 for row in rows if row["edge_touch"] == "true"),
        "tiny_bbox_objects": sum(1 for row in rows if row["tiny_bbox"] == "true"),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_dataset",
        "split",
        "image_path",
        "label_path",
        "line_number",
        "raw_class_id",
        "raw_class_name",
        "currency",
        "denomination",
        "side",
        "canonical_name",
        "cashsnap_core",
        "points",
        "bbox_xc",
        "bbox_yc",
        "bbox_w",
        "bbox_h",
        "bbox_area",
        "polygon_area",
        "edge_touch",
        "tiny_bbox",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_yaml = resolve(args.data)
    out_csv = resolve(args.out_csv)
    out_json = resolve(args.out_json)
    rows, summary = build_rows(data_yaml, args)
    write_csv(out_csv, rows)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Objects: {summary['objects']} from {summary['images']} images")
    print(f"Core CashSnap objects: {summary['cashsnap_core_objects']}")
    print(f"Non-core objects: {summary['non_core_objects']}")
    print(f"Edge-touch objects: {summary['edge_touch_objects']}")
    print(f"Tiny bbox objects: {summary['tiny_bbox_objects']}")
    print(f"Reports saved to: {out_csv} and {out_json}")


if __name__ == "__main__":
    main()
