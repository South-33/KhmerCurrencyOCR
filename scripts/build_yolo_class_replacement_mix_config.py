#!/usr/bin/env python
"""Build a YOLO config by replacing selected class rows with support rows."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--replace-class", action="append", default=[])
    parser.add_argument("--base-keep-per-class", type=int, required=True)
    parser.add_argument("--support-keep-per-class", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--out-list", type=Path, required=True)
    parser.add_argument("--tag", default="class_replacement_mix")
    parser.add_argument("--intended-use", default="")
    parser.add_argument("--promotion-rule", default="")
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


def read_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(resolved)}: expected YAML mapping")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw = Path(str(config.get("path", "."))).expanduser()
    return raw if raw.is_absolute() else (config_path.parent / raw).resolve()


def split_root(dataset_root: Path, split_path: str) -> Path:
    path = Path(split_path)
    return path if path.is_absolute() else dataset_root / path


def read_image_list(path: Path) -> list[str]:
    rows: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            rows.append(repo_rel(resolve(line)))
    return rows


def image_rows(root: Path) -> list[str]:
    image_dir = resolve(root)
    if not image_dir.exists():
        raise SystemExit(f"missing image dir: {repo_rel(image_dir)}")
    rows = [
        repo_rel(path)
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not rows:
        raise SystemExit(f"no images found under: {repo_rel(image_dir)}")
    return rows


def train_rows(config_path: Path, config: dict[str, Any]) -> tuple[list[str], list[str]]:
    root = data_root(config_path, config)
    train = config.get("train")
    if isinstance(train, str):
        train_items = [train]
    elif isinstance(train, list) and all(isinstance(item, str) for item in train):
        train_items = [str(item) for item in train]
    else:
        raise SystemExit(f"{repo_rel(config_path)} train split must be a string or list of strings")

    rows: list[str] = []
    sources: list[str] = []
    for item in train_items:
        path = split_root(root, item)
        if path.suffix.lower() == ".txt":
            rows.extend(read_image_list(path))
        elif path.is_dir():
            rows.extend(image_rows(path))
        else:
            raise SystemExit(f"{repo_rel(config_path)} train item must point to a .txt list or image directory: {item}")
        sources.append(repo_rel(path))
    return ordered_unique(rows)[0], sources


def class_names(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names", {})
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def label_path_for_image(image: str) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def label_lines(image: str) -> list[str]:
    label_path = resolve(label_path_for_image(image))
    if not label_path.exists():
        return []
    return [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_label(line: str, image: str) -> tuple[int, float]:
    parts = line.split()
    if len(parts) != 5:
        raise SystemExit(f"{repo_rel(resolve(label_path_for_image(image)))} expected 5 YOLO fields: {line}")
    try:
        class_id = int(float(parts[0]))
        width = float(parts[3])
        height = float(parts[4])
    except ValueError as exc:
        raise SystemExit(f"{repo_rel(resolve(label_path_for_image(image)))} has malformed YOLO row: {line}") from exc
    return class_id, width * height


def row_classes_and_area(row: str) -> tuple[list[int], float]:
    class_ids: list[int] = []
    area = 0.0
    for line in label_lines(row):
        class_id, row_area = parse_label(line, row)
        class_ids.append(class_id)
        area += row_area
    return class_ids, area


def ordered_unique(rows: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    unique: list[str] = []
    duplicates = 0
    for row in rows:
        normalized = row.replace("\\", "/")
        if normalized in seen:
            duplicates += 1
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique, duplicates


def sample_exact(rows: list[str], count: int, seed: int, label: str) -> list[str]:
    if len(rows) < count:
        raise SystemExit(f"{label} has only {len(rows)} rows, need {count}")
    if count < 0:
        raise SystemExit("sample counts must be non-negative")
    if len(rows) == count:
        return list(rows)
    rng = random.Random(seed)
    selected_indexes = set(rng.sample(range(len(rows)), count))
    return [row for index, row in enumerate(rows) if index in selected_indexes]


def bucket_rows(rows: list[str], names: dict[int, str]) -> tuple[dict[str, list[str]], list[str], list[str]]:
    by_class: dict[str, list[str]] = defaultdict(list)
    empty_rows: list[str] = []
    multi_rows: list[str] = []
    for row in rows:
        class_ids, _area = row_classes_and_area(row)
        if not class_ids:
            empty_rows.append(row)
        elif len(class_ids) == 1:
            by_class[names.get(class_ids[0], f"class_{class_ids[0]}")].append(row)
        else:
            multi_rows.append(row)
    return by_class, empty_rows, multi_rows


def label_class_counts(rows: list[str], names: dict[int, str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        class_ids, _area = row_classes_and_area(row)
        for class_id in class_ids:
            counts[names.get(class_id, f"class_{class_id}")] += 1
    return dict(sorted(counts.items()))


def area_stats(rows: list[str], names: dict[int, str]) -> dict[str, dict[str, float | int]]:
    values_by_class: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        class_ids, area = row_classes_and_area(row)
        if len(class_ids) == 1:
            values_by_class[names.get(class_ids[0], f"class_{class_ids[0]}")].append(area)

    stats: dict[str, dict[str, float | int]] = {}
    for class_name, values in sorted(values_by_class.items()):
        ordered = sorted(values)
        if not ordered:
            continue
        def quantile(fraction: float) -> float:
            index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
            return ordered[index]

        stats[class_name] = {
            "count": len(ordered),
            "mean": round(sum(ordered) / len(ordered), 6),
            "p50": round(quantile(0.50), 6),
            "p90": round(quantile(0.90), 6),
            "ge50": sum(value >= 0.50 for value in ordered),
            "ge90": sum(value >= 0.90 for value in ordered),
        }
    return stats


def write_image_list(path: Path, rows: list[str]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.base_keep_per_class < 0 or args.support_keep_per_class < 0:
        raise SystemExit("--base-keep-per-class and --support-keep-per-class must be >= 0")
    replace_classes = sorted({item.strip() for value in args.replace_class for item in value.split(",") if item.strip()})
    if not replace_classes:
        raise SystemExit("provide at least one --replace-class")

    base_path = resolve(args.base)
    support_path = resolve(args.support)
    base_config = read_yaml(base_path)
    support_config = read_yaml(support_path)
    names = class_names(base_config)
    support_names = class_names(support_config)
    if names != support_names:
        raise SystemExit("base and support configs must use identical class names")
    known_classes = set(names.values())
    unknown = sorted(set(replace_classes) - known_classes)
    if unknown:
        raise SystemExit(f"unknown --replace-class values: {', '.join(unknown)}")

    base_rows, base_sources = train_rows(base_path, base_config)
    support_rows, support_sources = train_rows(support_path, support_config)
    base_by_class, base_empty_rows, base_multi_rows = bucket_rows(base_rows, names)
    support_by_class, _support_empty_rows, support_multi_rows = bucket_rows(support_rows, names)

    blocked_multi: list[str] = []
    replace_class_ids = {class_id for class_id, name in names.items() if name in replace_classes}
    for row in base_multi_rows + support_multi_rows:
        class_ids, _area = row_classes_and_area(row)
        if replace_class_ids.intersection(class_ids):
            blocked_multi.append(row)
    if blocked_multi:
        raise SystemExit(f"replacement classes appear in multi-label rows: {blocked_multi[:5]}")

    selected_base_by_class: dict[str, list[str]] = {}
    selected_support_by_class: dict[str, list[str]] = {}
    dropped_base_by_class: dict[str, int] = {}
    for offset, class_name in enumerate(replace_classes):
        base_pool = base_by_class.get(class_name, [])
        support_pool = support_by_class.get(class_name, [])
        selected_base = sample_exact(
            base_pool,
            args.base_keep_per_class,
            args.seed + 10_003 + offset,
            f"base {class_name}",
        )
        selected_support = sample_exact(
            support_pool,
            args.support_keep_per_class,
            args.seed + 20_003 + offset,
            f"support {class_name}",
        )
        selected_base_by_class[class_name] = selected_base
        selected_support_by_class[class_name] = selected_support
        dropped_base_by_class[class_name] = len(base_pool) - len(selected_base)

    selected_base_set = {row for rows in selected_base_by_class.values() for row in rows}
    replace_set = set(replace_classes)
    kept_base_rows: list[str] = []
    for row in base_rows:
        class_ids, _area = row_classes_and_area(row)
        if not class_ids or len(class_ids) > 1:
            kept_base_rows.append(row)
            continue
        class_name = names.get(class_ids[0], f"class_{class_ids[0]}")
        if class_name not in replace_set or row in selected_base_set:
            kept_base_rows.append(row)

    support_selected_rows = [row for class_name in replace_classes for row in selected_support_by_class[class_name]]
    combined_rows, combined_duplicates = ordered_unique(kept_base_rows + support_selected_rows)
    report = {
        "schema": "cashsnap_yolo_class_replacement_mix_v1",
        "tag": args.tag,
        "base_config": repo_rel(base_path),
        "support_config": repo_rel(support_path),
        "base_train_sources": base_sources,
        "support_train_sources": support_sources,
        "replace_classes": replace_classes,
        "base_keep_per_class": args.base_keep_per_class,
        "support_keep_per_class": args.support_keep_per_class,
        "seed": args.seed,
        "base_images": len(base_rows),
        "support_images": len(support_rows),
        "base_empty_images_kept": len(base_empty_rows),
        "base_multi_images_kept": len(base_multi_rows),
        "base_available_by_class": {class_name: len(base_by_class.get(class_name, [])) for class_name in replace_classes},
        "support_available_by_class": {
            class_name: len(support_by_class.get(class_name, [])) for class_name in replace_classes
        },
        "base_dropped_by_class": dropped_base_by_class,
        "base_kept_images": len(kept_base_rows),
        "support_selected_images": len(support_selected_rows),
        "combined_images": len(combined_rows),
        "combined_duplicate_images_removed": combined_duplicates,
        "class_counts": label_class_counts(combined_rows, names),
        "area_stats": area_stats(combined_rows, names),
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    write_image_list(args.out_list, combined_rows)
    out_config = resolve(args.out_config)
    config = copy.deepcopy(base_config)
    config["path"] = rel_between(out_config.parent, ROOT)
    config["train"] = repo_rel(resolve(args.out_list))
    config["names"] = {index: names[index] for index in sorted(names)}

    sources = copy.deepcopy(config.get("cashsnap_sources", {}))
    if not isinstance(sources, dict):
        sources = {}
    sources["class_replacement_mix_base_config"] = repo_rel(base_path)
    sources["class_replacement_mix_support_config"] = repo_rel(support_path)
    config["cashsnap_sources"] = sources
    config["cashsnap_class_replacement_mix"] = report

    policy = copy.deepcopy(config.get("cashsnap_policy", {}))
    if not isinstance(policy, dict):
        policy = {}
    if args.intended_use:
        policy["intended_use"] = args.intended_use
    if args.promotion_rule:
        policy["promotion_rule"] = args.promotion_rule
    config["cashsnap_policy"] = policy

    write_yaml(out_config, config)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote_list={repo_rel(resolve(args.out_list))}")
    print(f"wrote_config={repo_rel(out_config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
