#!/usr/bin/env python
"""Build a compact real-train bridge config from activation-microscope analogs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", required=True, type=Path)
    parser.add_argument("--real-data", required=True, type=Path)
    parser.add_argument("--analog-csv", required=True, action="append", type=Path)
    parser.add_argument("--out-config", required=True, type=Path)
    parser.add_argument("--out-train-list", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    parser.add_argument("--positive-repeat", type=int, default=6)
    parser.add_argument("--empty-backgrounds", type=int, default=128)
    parser.add_argument("--background-repeat", type=int, default=2)
    parser.add_argument(
        "--class-floor-per-class",
        type=int,
        default=0,
        help="Optional real-train positive image floor per class to prevent bridge starvation.",
    )
    parser.add_argument("--class-floor-repeat", type=int, default=1)
    parser.add_argument(
        "--class-floor-classes",
        default="",
        help="Optional comma/space-separated class-name subset for the floor. Default uses all classes.",
    )
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument(
        "--selection-note",
        default="activation-microscope selected train bridge; diagnostic until rebuilt from a non-test split",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    root = Path(str(config.get("path", "."))).expanduser()
    if root.is_absolute():
        return root
    return (config_path.parent / root).resolve()


def split_path(config_path: Path, config: dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return data_root(config_path, config) / path


def split_rows(config_path: Path, config: dict[str, Any], split_name: str) -> list[Path]:
    split = config.get(split_name)
    if not isinstance(split, (str, list)):
        raise ValueError(f"{config_path} {split_name!r} must be a string or list")
    values = split if isinstance(split, list) else [split]
    rows: list[Path] = []
    for value in values:
        path = split_path(config_path, config, str(value))
        if path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(resolve(Path(line)))
        elif path.is_dir():
            rows.extend(
                sorted(
                    item.resolve()
                    for item in path.iterdir()
                    if item.is_file() and item.suffix.lower() in IMAGE_EXTS
                )
            )
        else:
            raise FileNotFoundError(f"cannot resolve {split_name} row source {value!r} to {path}")
    return rows


def row_fingerprint(rows: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.replace("\\", "/").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        idx = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"image path does not contain an images/ segment: {image}") from exc
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def names_by_id(config: dict[str, Any]) -> dict[int, str]:
    raw_names = config.get("names")
    if not isinstance(raw_names, dict):
        raise ValueError("data YAML must contain names mapping")
    names: dict[int, str] = {}
    for key, value in raw_names.items():
        names[int(key)] = str(value)
    return names


def parse_class_set(raw: str) -> set[str]:
    return {item.strip() for item in raw.replace(",", " ").split() if item.strip()}


def label_class_ids(label: Path) -> set[int]:
    class_ids: set[int] = set()
    if not label.exists():
        return class_ids
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts:
            class_ids.add(int(float(parts[0])))
    return class_ids


def class_floor_rows(
    real_data: Path,
    *,
    per_class: int,
    repeat_seed: int,
    class_filter: set[str],
) -> tuple[list[str], dict[str, int]]:
    if per_class <= 0:
        return [], {}
    config_path = resolve(real_data)
    config = read_yaml(real_data)
    names = names_by_id(config)
    target_classes = sorted(class_filter or set(names.values()))
    by_class: dict[str, list[str]] = {class_name: [] for class_name in target_classes}
    for image in split_rows(config_path, config, "train"):
        class_names = {names[class_id] for class_id in label_class_ids(label_path_for_image(image)) if class_id in names}
        for class_name in sorted(class_names & set(target_classes)):
            by_class[class_name].append(repo_rel(image))

    rng = random.Random(repeat_seed)
    selected: list[str] = []
    selected_counts: dict[str, int] = {}
    for class_name in target_classes:
        candidates = sorted(set(by_class.get(class_name, [])))
        rng.shuffle(candidates)
        chosen = sorted(candidates[:per_class])
        selected.extend(chosen)
        selected_counts[class_name] = len(chosen)
    return sorted(set(selected)), selected_counts


def mined_positive_rows(paths: list[Path]) -> tuple[list[str], Counter[str], dict[str, float]]:
    by_image: dict[str, tuple[str, float]] = {}
    for path in paths:
        with resolve(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                image = repo_rel(resolve(Path(row["candidate_image"])))
                class_name = row.get("candidate_class") or row.get("query_class") or "unknown"
                distance = float(row["distance_l2"])
                old = by_image.get(image)
                if old is None or distance < old[1]:
                    by_image[image] = (class_name, distance)
    rows = sorted(by_image)
    class_counts = Counter(by_image[row][0] for row in rows)
    nearest_distance_by_image = {row: by_image[row][1] for row in rows}
    return rows, class_counts, nearest_distance_by_image


def empty_background_rows(real_data: Path, count: int, seed: int) -> list[str]:
    config_path = resolve(real_data)
    config = read_yaml(real_data)
    candidates = []
    for image in split_rows(config_path, config, "train"):
        label = label_path_for_image(image)
        if label.exists() and not label.read_text(encoding="utf-8").strip():
            candidates.append(repo_rel(image))
    if len(candidates) < count:
        raise ValueError(f"requested {count} empty backgrounds, only found {len(candidates)}")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return sorted(candidates[:count])


def relative_split_value(out_config: Path, row_path: Path) -> str:
    try:
        return row_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return row_path.resolve().as_posix()


def main() -> int:
    args = parse_args()
    if args.positive_repeat < 1:
        raise SystemExit("--positive-repeat must be >= 1")
    if args.background_repeat < 0:
        raise SystemExit("--background-repeat must be >= 0")
    if args.empty_backgrounds < 0:
        raise SystemExit("--empty-backgrounds must be >= 0")
    if args.class_floor_per_class < 0:
        raise SystemExit("--class-floor-per-class must be >= 0")
    if args.class_floor_repeat < 0:
        raise SystemExit("--class-floor-repeat must be >= 0")

    source_config_path = resolve(args.source_data)
    source_config = read_yaml(args.source_data)
    source_rows = [repo_rel(row) for row in split_rows(source_config_path, source_config, "train")]
    positive_rows, positive_counts, nearest_distances = mined_positive_rows(args.analog_csv)
    floor_rows, floor_counts = class_floor_rows(
        args.real_data,
        per_class=args.class_floor_per_class,
        repeat_seed=args.seed + 17,
        class_filter=parse_class_set(args.class_floor_classes),
    )
    background_rows = empty_background_rows(args.real_data, args.empty_backgrounds, args.seed)

    train_rows = (
        source_rows
        + positive_rows * args.positive_repeat
        + floor_rows * args.class_floor_repeat
        + background_rows * args.background_repeat
    )

    out_train_list = resolve(args.out_train_list)
    out_config = resolve(args.out_config)
    out_report = resolve(args.out_report)
    out_train_list.parent.mkdir(parents=True, exist_ok=True)
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_train_list.write_text("\n".join(train_rows) + "\n", encoding="utf-8")

    output_config = {
        "path": "../..",
        "train": relative_split_value(out_config, out_train_list),
        "val": source_config.get("val"),
        "test": source_config.get("test"),
        "names": source_config.get("names"),
        "cashsnap_policy": {
            "schema": "cashsnap_activation_failure_bridge_v1",
            "selection_note": args.selection_note,
            "source_data": repo_rel(source_config_path),
            "real_data": repo_rel(resolve(args.real_data)),
            "analog_csvs": [repo_rel(resolve(path)) for path in args.analog_csv],
            "source_train_rows": len(source_rows),
            "positive_unique_images": len(positive_rows),
            "positive_repeat": args.positive_repeat,
            "positive_rows_added": len(positive_rows) * args.positive_repeat,
            "positive_unique_by_class": dict(sorted(positive_counts.items())),
            "class_floor_per_class": args.class_floor_per_class,
            "class_floor_repeat": args.class_floor_repeat,
            "class_floor_unique_images": len(floor_rows),
            "class_floor_rows_added": len(floor_rows) * args.class_floor_repeat,
            "class_floor_selected_by_class": dict(sorted(floor_counts.items())),
            "empty_background_unique_images": len(background_rows),
            "background_repeat": args.background_repeat,
            "background_rows_added": len(background_rows) * args.background_repeat,
            "total_train_rows": len(train_rows),
            "unique_train_rows": len(set(train_rows)),
            "duplicate_rows": len(train_rows) - len(set(train_rows)),
            "seed": args.seed,
            "train_list_fingerprint": row_fingerprint(train_rows),
        },
    }
    out_config.write_text(yaml.safe_dump(output_config, sort_keys=False), encoding="utf-8")

    report = {
        "schema": "cashsnap_activation_failure_bridge_report_v1",
        "config": repo_rel(out_config),
        "train_list": repo_rel(out_train_list),
        "source_data": repo_rel(source_config_path),
        "real_data": repo_rel(resolve(args.real_data)),
        "analog_csvs": [repo_rel(resolve(path)) for path in args.analog_csv],
        "source_train_rows": len(source_rows),
        "positive_unique_images": len(positive_rows),
        "positive_repeat": args.positive_repeat,
        "positive_rows_added": len(positive_rows) * args.positive_repeat,
        "positive_unique_by_class": dict(sorted(positive_counts.items())),
        "class_floor_per_class": args.class_floor_per_class,
        "class_floor_repeat": args.class_floor_repeat,
        "class_floor_unique_images": len(floor_rows),
        "class_floor_rows_added": len(floor_rows) * args.class_floor_repeat,
        "class_floor_selected_by_class": dict(sorted(floor_counts.items())),
        "empty_background_unique_images": len(background_rows),
        "background_repeat": args.background_repeat,
        "background_rows_added": len(background_rows) * args.background_repeat,
        "total_train_rows": len(train_rows),
        "unique_train_rows": len(set(train_rows)),
        "duplicate_rows": len(train_rows) - len(set(train_rows)),
        "train_list_fingerprint": row_fingerprint(train_rows),
        "positive_nearest_distance_by_image": nearest_distances,
        "class_floor_rows": floor_rows,
        "empty_background_rows": background_rows,
    }
    out_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "wrote_bridge "
        f"config={repo_rel(out_config)} "
        f"rows={len(train_rows)} "
        f"positives={len(positive_rows)}x{args.positive_repeat} "
        f"floor={len(floor_rows)}x{args.class_floor_repeat} "
        f"backgrounds={len(background_rows)}x{args.background_repeat}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
