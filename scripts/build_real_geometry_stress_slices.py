#!/usr/bin/env python
"""Build real validation slices from YOLO label geometry."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    ROOT
    / "configs"
    / "webgl_ablation"
    / "cashsnap_target_anchor_transplant_poisson_contact_geoscale205_minshort190_bal96_scaled_foreignhardneg6_probe_puresynth_realval_v1.yaml"
)
DEFAULT_OUT_DIR = ROOT / "runs" / "cashsnap" / "real_geometry_stress_slices_v1"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
PROTECTED_RIEL = {"KHR_20000", "KHR_50000"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", action="append", default=["val", "test"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-area", type=float, default=0.16)
    parser.add_argument("--max-area", type=float, default=0.86)
    parser.add_argument("--edge-margin", type=float, default=0.03)
    parser.add_argument("--min-short", type=float, default=0.28)
    parser.add_argument("--min-aspect", type=float, default=0.65)
    parser.add_argument("--max-aspect", type=float, default=2.8)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"{repo_rel(resolve(path))}: expected YAML mapping")
    return document


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw = Path(str(config.get("path", "."))).expanduser()
    return raw if raw.is_absolute() else (config_path.parent / raw).resolve()


def split_path(config_path: Path, config: dict[str, Any], split: str) -> Path:
    raw = config.get(split)
    if not isinstance(raw, str):
        raise SystemExit(f"{repo_rel(config_path)}: split {split!r} must be a path string")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else data_root(config_path, config) / path


def image_rows(path: Path) -> list[str]:
    if path.suffix.lower() == ".txt":
        return [
            line.strip().replace("\\", "/")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    if path.is_dir():
        return [
            repo_rel(image)
            for image in sorted(path.iterdir())
            if image.is_file() and image.suffix.lower() in IMAGE_EXTS
        ]
    raise SystemExit(f"unsupported split path: {repo_rel(path)}")


def label_path(image: str) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(image: str, names: dict[int, str]) -> list[dict[str, Any]]:
    label = resolve(label_path(image))
    if not label.exists():
        return []
    rows = []
    for raw_line in label.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        x, y, width, height = map(float, parts[1:])
        x1 = x - width / 2
        y1 = y - height / 2
        x2 = x + width / 2
        y2 = y + height / 2
        aspect = width / max(height, 1e-9)
        rows.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
                "area": width * height,
                "short": min(width, height),
                "aspect": aspect,
            }
        )
    return rows


def touches_edge(row: dict[str, Any], edge_margin: float) -> bool:
    return (
        float(row["x1"]) <= edge_margin
        or float(row["y1"]) <= edge_margin
        or float(row["x2"]) >= 1.0 - edge_margin
        or float(row["y2"]) >= 1.0 - edge_margin
    )


def row_is_partial_like(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        touches_edge(row, args.edge_margin)
        or float(row["area"]) < args.min_area
        or float(row["area"]) > args.max_area
        or float(row["short"]) < args.min_short
        or float(row["aspect"]) < args.min_aspect
        or float(row["aspect"]) > args.max_aspect
    )


def clean_visible_single(labels: list[dict[str, Any]], args: argparse.Namespace) -> bool:
    return len(labels) == 1 and not row_is_partial_like(labels[0], args)


def names_by_id(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names", {})
    if not isinstance(names, dict):
        raise SystemExit("data config names must be a mapping")
    return {int(class_id): str(class_name) for class_id, class_name in names.items()}


def class_counts(images: list[str], labels_by_image: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for image in images:
        for row in labels_by_image[image]:
            counts[str(row["class_name"])] += 1
    return dict(counts)


def write_list(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_data_yaml(
    *,
    path: Path,
    source_config: dict[str, Any],
    source_config_path: Path,
    split_lists: dict[str, Path],
) -> None:
    payload = {
        "path": Path(os.path.relpath(ROOT, path.parent)).as_posix(),
        "train": source_config.get("train", ""),
        "val": repo_rel(split_lists.get("val", split_lists[next(iter(split_lists))])),
        "test": repo_rel(split_lists.get("test", split_lists[next(iter(split_lists))])),
        "names": source_config.get("names", {}),
        "cashsnap_diagnostic": {
            "purpose": "Label-geometry real stress slice for synthetic transfer guardrails",
            "source_data": repo_rel(source_config_path),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    data_path = resolve(args.data)
    config = read_yaml(data_path)
    names = names_by_id(config)
    out_dir = resolve(args.out_dir)
    criteria = {
        "min_area": args.min_area,
        "max_area": args.max_area,
        "edge_margin": args.edge_margin,
        "min_short": args.min_short,
        "min_aspect": args.min_aspect,
        "max_aspect": args.max_aspect,
        "protected_riel": sorted(PROTECTED_RIEL),
    }
    slices = {
        "labeled_all": defaultdict(list),
        "geometry_stress": defaultdict(list),
        "multi_note": defaultdict(list),
        "partial_edge": defaultdict(list),
        "protected_riel": defaultdict(list),
    }
    split_summaries: dict[str, Any] = {}
    for split in args.split:
        rows = image_rows(split_path(data_path, config, split))
        labels_by_image = {image: read_labels(image, names) for image in rows}
        labeled = [image for image in rows if labels_by_image[image]]
        for image in labeled:
            labels = labels_by_image[image]
            slices["labeled_all"][split].append(image)
            if not clean_visible_single(labels, args):
                slices["geometry_stress"][split].append(image)
            if len(labels) >= 2:
                slices["multi_note"][split].append(image)
            if any(row_is_partial_like(row, args) for row in labels):
                slices["partial_edge"][split].append(image)
            if any(str(row["class_name"]) in PROTECTED_RIEL for row in labels):
                slices["protected_riel"][split].append(image)
        split_summaries[split] = {
            "total_images": len(rows),
            "labeled_images": len(labeled),
            "background_images": len(rows) - len(labeled),
        }

    slice_outputs: dict[str, Any] = {}
    for slice_name, by_split in slices.items():
        split_lists: dict[str, Path] = {}
        split_payload: dict[str, Any] = {}
        for split in args.split:
            images = list(by_split.get(split, []))
            list_path = out_dir / f"{slice_name}_{split}.txt"
            write_list(list_path, images)
            split_lists[split] = list_path
            labels_by_image = {image: read_labels(image, names) for image in images}
            split_payload[split] = {
                "images": len(images),
                "boxes": sum(len(labels) for labels in labels_by_image.values()),
                "class_counts": class_counts(images, labels_by_image),
                "list": repo_rel(list_path),
            }
        data_yaml = out_dir / f"{slice_name}_data.yaml"
        write_data_yaml(
            path=data_yaml,
            source_config=config,
            source_config_path=data_path,
            split_lists=split_lists,
        )
        slice_outputs[slice_name] = {
            "data_yaml": repo_rel(data_yaml),
            "splits": split_payload,
        }

    summary = {
        "schema": "cashsnap_real_geometry_stress_slices_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_data": repo_rel(data_path),
        "criteria": criteria,
        "source_splits": split_summaries,
        "slices": slice_outputs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote_summary={repo_rel(summary_path)}")
    for slice_name, payload in slice_outputs.items():
        split_bits = ", ".join(
            f"{split}={stats['images']} images/{stats['boxes']} boxes"
            for split, stats in payload["splits"].items()
        )
        print(f"{slice_name}: {split_bits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
