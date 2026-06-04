from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit YOLO segmentation label geometry.")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--out-json", default="data/audit/yolo_segmentation_geometry.json")
    parser.add_argument("--out-csv", default="data/audit/yolo_segmentation_geometry_by_class.csv")
    parser.add_argument("--edge-margin", type=float, default=0.01)
    parser.add_argument("--tiny-bbox-area", type=float, default=0.0025)
    parser.add_argument("--huge-bbox-area", type=float, default=0.90)
    parser.add_argument("--max-examples", type=int, default=50)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def load_config(data_yaml: Path) -> tuple[Path, dict[int, str]]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = (data_yaml.parent / config["path"]).resolve() if "path" in config else data_yaml.parent.resolve()
    raw_names = config.get("names", {})
    if isinstance(raw_names, dict):
        names = {int(key): str(value) for key, value in raw_names.items()}
    else:
        names = {index: str(value) for index, value in enumerate(raw_names)}
    return root, names


def label_dirs(root: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for split in SPLITS:
        normalized_split = "valid" if split == "val" else split
        candidates = [root / split / "labels", root / "labels" / split]
        for candidate in candidates:
            if candidate.exists():
                found.append((normalized_split, candidate))
                break
    return found


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def add_example(examples: list[dict[str, Any]], limit: int, **item: Any) -> None:
    if len(examples) < limit:
        examples.append(item)


def audit(data_yaml: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root, names = load_config(data_yaml)
    per_class: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "bbox_areas": [],
            "polygon_areas": [],
            "points": [],
            "edge_touch_count": 0,
            "tiny_bbox_count": 0,
            "huge_bbox_count": 0,
        }
    )
    split_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    label_file_count = 0
    row_count = 0

    for split, label_dir in label_dirs(root):
        label_files = sorted(label_dir.glob("*.txt"))
        label_file_count += len(label_files)
        for label_file in label_files:
            for line_number, raw_line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), start=1):
                parts = raw_line.split()
                if not parts:
                    continue
                row_count += 1
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    issue_counts["invalid_class_id"] += 1
                    add_example(examples, args.max_examples, issue="invalid_class_id", file=str(label_file), line=line_number)
                    continue

                coords_text = parts[1:]
                if len(coords_text) < 6 or len(coords_text) % 2 != 0:
                    issue_counts["not_segmentation_polygon"] += 1
                    add_example(
                        examples,
                        args.max_examples,
                        issue="not_segmentation_polygon",
                        file=str(label_file.relative_to(ROOT)),
                        line=line_number,
                        class_id=class_id,
                        value_count=len(coords_text),
                    )
                    continue

                try:
                    coords = [float(value) for value in coords_text]
                except ValueError:
                    issue_counts["invalid_coordinate"] += 1
                    add_example(examples, args.max_examples, issue="invalid_coordinate", file=str(label_file), line=line_number)
                    continue

                points = list(zip(coords[0::2], coords[1::2], strict=False))
                xs = coords[0::2]
                ys = coords[1::2]
                out_of_bounds = any(value < 0.0 or value > 1.0 for value in coords)
                if out_of_bounds:
                    issue_counts["out_of_bounds_coordinate"] += 1
                    add_example(
                        examples,
                        args.max_examples,
                        issue="out_of_bounds_coordinate",
                        file=str(label_file.relative_to(ROOT)),
                        line=line_number,
                        class_id=class_id,
                    )

                bbox_w = max(xs) - min(xs)
                bbox_h = max(ys) - min(ys)
                bbox_area = bbox_w * bbox_h
                mask_area = polygon_area(points)
                if bbox_area <= 0 or mask_area <= 0:
                    issue_counts["zero_area_polygon"] += 1
                    add_example(
                        examples,
                        args.max_examples,
                        issue="zero_area_polygon",
                        file=str(label_file.relative_to(ROOT)),
                        line=line_number,
                        class_id=class_id,
                    )

                touches_edge = (
                    min(xs) <= args.edge_margin
                    or min(ys) <= args.edge_margin
                    or max(xs) >= 1.0 - args.edge_margin
                    or max(ys) >= 1.0 - args.edge_margin
                )
                stats = per_class[class_id]
                stats["count"] += 1
                stats["bbox_areas"].append(bbox_area)
                stats["polygon_areas"].append(mask_area)
                stats["points"].append(float(len(points)))
                stats["edge_touch_count"] += int(touches_edge)
                stats["tiny_bbox_count"] += int(bbox_area < args.tiny_bbox_area)
                stats["huge_bbox_count"] += int(bbox_area > args.huge_bbox_area)
                split_counts[split] += 1

    rows = []
    for class_id, stats in sorted(per_class.items()):
        count = stats["count"]
        rows.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "count": count,
                "median_points": median(stats["points"]),
                "min_points": min(stats["points"]) if stats["points"] else 0,
                "max_points": max(stats["points"]) if stats["points"] else 0,
                "median_bbox_area": median(stats["bbox_areas"]),
                "min_bbox_area": min(stats["bbox_areas"]) if stats["bbox_areas"] else 0,
                "max_bbox_area": max(stats["bbox_areas"]) if stats["bbox_areas"] else 0,
                "median_polygon_area": median(stats["polygon_areas"]),
                "edge_touch_count": stats["edge_touch_count"],
                "edge_touch_percent": round(stats["edge_touch_count"] / count * 100, 2) if count else 0.0,
                "tiny_bbox_count": stats["tiny_bbox_count"],
                "huge_bbox_count": stats["huge_bbox_count"],
            }
        )

    summary = {
        "dataset": str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root),
        "label_files": label_file_count,
        "labels": row_count,
        "classes": len(rows),
        "split_counts": dict(split_counts),
        "issue_counts": dict(issue_counts),
        "examples": examples,
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class_id",
        "class_name",
        "count",
        "median_points",
        "min_points",
        "max_points",
        "median_bbox_area",
        "min_bbox_area",
        "max_bbox_area",
        "median_polygon_area",
        "edge_touch_count",
        "edge_touch_percent",
        "tiny_bbox_count",
        "huge_bbox_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_yaml = resolve(args.data)
    out_json = resolve(args.out_json)
    out_csv = resolve(args.out_csv)
    summary, rows = audit(data_yaml, args)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"summary": summary, "by_class": rows}, indent=2), encoding="utf-8")
    write_csv(out_csv, rows)

    print(f"Dataset: {summary['dataset']}")
    print(f"Labels: {summary['labels']} rows from {summary['label_files']} files")
    print(f"Classes: {summary['classes']}")
    print(f"Splits: {summary['split_counts']}")
    print(f"Issues: {summary['issue_counts'] or 'none'}")
    print(f"Reports saved to: {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
