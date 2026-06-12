#!/usr/bin/env python
"""Build a list-backed YOLO config by appending extra train image rows."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extra-image-dir", action="append", type=Path, default=[])
    parser.add_argument("--extra-image-list", action="append", type=Path, default=[])
    parser.add_argument(
        "--extra-max-images",
        type=int,
        default=None,
        help="Deterministically sample at most this many unique extra image rows before appending.",
    )
    parser.add_argument(
        "--extra-max-per-class",
        type=int,
        default=None,
        help=(
            "Deterministically sample at most this many extra image rows for each labeled class. "
            "Rows with multiple labels are kept only while all represented classes remain under cap."
        ),
    )
    parser.add_argument(
        "--extra-max-empty",
        type=int,
        default=None,
        help="Deterministically sample at most this many empty-label extra image rows.",
    )
    parser.add_argument("--extra-sample-seed", type=int, default=0)
    parser.add_argument(
        "--extra-label-policy",
        choices=("any", "empty", "non-empty"),
        default="any",
        help="Label policy enforced only for extra rows.",
    )
    parser.add_argument(
        "--extra-row-require-regex",
        default="",
        help="Keep only extra rows whose repo-relative path matches this regex.",
    )
    parser.add_argument(
        "--extra-row-block-regex",
        default="",
        help="Drop extra rows whose repo-relative path matches this regex.",
    )
    parser.add_argument(
        "--extra-min-label-area",
        type=float,
        default=None,
        help="Keep only extra rows with at least one label at or above this normalized bbox area.",
    )
    parser.add_argument(
        "--extra-max-label-area",
        type=float,
        default=None,
        help="Keep only extra rows with at least one label at or below this normalized bbox area.",
    )
    parser.add_argument(
        "--extra-edge-margin",
        type=float,
        default=None,
        help="Keep only extra rows with at least one label touching an image edge within this normalized margin.",
    )
    parser.add_argument(
        "--extra-single-label-only",
        action="store_true",
        help="Keep only extra rows with exactly one non-empty YOLO label.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan --extra-image-dir roots.")
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--out-list", type=Path, required=True)
    parser.add_argument(
        "--out-extra-list",
        type=Path,
        default=None,
        help="Optional path to write the final post-filter extra rows before merging with the base list.",
    )
    parser.add_argument("--tag", default="train_mix")
    parser.add_argument("--intended-use", default="")
    parser.add_argument("--promotion-rule", default="")
    parser.add_argument(
        "--extra-name",
        action="append",
        default=[],
        metavar="INDEX:NAME",
        help="Append/override a YOLO class name in the output config, e.g. 13:UNKNOWN_FOREIGN_NOTE.",
    )
    parser.add_argument(
        "--preserve-duplicate-exposures",
        action="store_true",
        help="Keep duplicate rows from the base and extra lists instead of collapsing to unique images.",
    )
    parser.add_argument(
        "--extra-repeat",
        type=int,
        default=1,
        help="Repeat the final filtered extra rows this many times before merging.",
    )
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
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(resolved)}: expected YAML mapping")
    return data


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


def image_rows(root: Path, *, recursive: bool) -> list[str]:
    image_dir = resolve(root)
    if not image_dir.exists():
        raise SystemExit(f"missing image dir: {repo_rel(image_dir)}")
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    rows = [
        repo_rel(path)
        for path in sorted(iterator)
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
            rows.extend(image_rows(path, recursive=False))
        else:
            raise SystemExit(f"{repo_rel(config_path)} train item must point to a .txt list or image directory: {item}")
        sources.append(repo_rel(path))
    return rows, sources


def repo_relative_split_value(config_path: Path, config: dict[str, Any], value: Any) -> Any:
    root = data_root(config_path, config)

    def convert(item: str) -> str:
        return repo_rel(split_root(root, item))

    if isinstance(value, str):
        return convert(value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [convert(str(item)) for item in value]
    return value


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


def label_class_ids(image: str) -> set[int]:
    class_ids: set[int] = set()
    for line in label_lines(image):
        parts = line.split()
        if not parts:
            continue
        try:
            class_ids.add(int(float(parts[0])))
        except ValueError:
            continue
    return class_ids


def label_geometries(image: str) -> list[dict[str, float | int]]:
    geometries: list[dict[str, float | int]] = []
    for line in label_lines(image):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue
        x1 = cx - width / 2.0
        y1 = cy - height / 2.0
        x2 = cx + width / 2.0
        y2 = cy + height / 2.0
        geometries.append(
            {
                "class_id": class_id,
                "area": width * height,
                "edge_margin": min(x1, y1, 1.0 - x2, 1.0 - y2),
            }
        )
    return geometries


def enforce_label_policy(rows: list[str], policy: str) -> None:
    if policy == "any":
        return
    offenders: list[str] = []
    for row in rows:
        has_label = bool(label_lines(row))
        if policy == "empty" and has_label:
            offenders.append(row)
        elif policy == "non-empty" and not has_label:
            offenders.append(row)
    if offenders:
        raise SystemExit(f"extra rows violate --extra-label-policy {policy}: {offenders[:5]}")


def filter_rows_by_label_geometry(
    rows: list[str],
    *,
    min_area: float | None,
    max_area: float | None,
    edge_margin: float | None,
    single_label_only: bool,
) -> tuple[list[str], int, dict[str, Any]]:
    if min_area is None and max_area is None and edge_margin is None and not single_label_only:
        return rows, 0, {}
    if min_area is not None and min_area < 0:
        raise SystemExit("--extra-min-label-area must be >= 0 when set")
    if max_area is not None and max_area < 0:
        raise SystemExit("--extra-max-label-area must be >= 0 when set")
    if min_area is not None and max_area is not None and min_area > max_area:
        raise SystemExit("--extra-min-label-area cannot be greater than --extra-max-label-area")
    if edge_margin is not None and edge_margin < 0:
        raise SystemExit("--extra-edge-margin must be >= 0 when set")

    kept: list[str] = []
    removed = 0
    kept_areas: list[float] = []
    kept_edge_margins: list[float] = []
    for row in rows:
        geometries = label_geometries(row)
        if single_label_only and len(geometries) != 1:
            removed += 1
            continue
        row_matches = False
        for geometry in geometries:
            area = float(geometry["area"])
            margin = float(geometry["edge_margin"])
            if min_area is not None and area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
            if edge_margin is not None and margin > edge_margin:
                continue
            row_matches = True
            kept_areas.append(area)
            kept_edge_margins.append(margin)
            break
        if row_matches:
            kept.append(row)
        else:
            removed += 1
    if not kept:
        raise SystemExit("extra label-geometry filters removed all candidate rows")

    def rounded_range(values: list[float]) -> list[float]:
        return [round(min(values), 6), round(max(values), 6)] if values else []

    report = {
        "min_label_area": min_area,
        "max_label_area": max_area,
        "edge_margin": edge_margin,
        "single_label_only": single_label_only,
        "kept_label_area_range": rounded_range(kept_areas),
        "kept_edge_margin_range": rounded_range(kept_edge_margins),
    }
    return kept, removed, report


def class_names(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names", {})
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def parse_extra_name(value: str) -> tuple[int, str]:
    if ":" not in value:
        raise SystemExit(f"--extra-name must use INDEX:NAME format: {value}")
    raw_index, name = value.split(":", 1)
    name = name.strip()
    if not name:
        raise SystemExit(f"--extra-name has empty class name: {value}")
    try:
        index = int(raw_index)
    except ValueError as exc:
        raise SystemExit(f"--extra-name has non-integer index: {value}") from exc
    if index < 0:
        raise SystemExit(f"--extra-name index must be non-negative: {value}")
    return index, name


def label_class_counts(rows: list[str], names: dict[int, str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for line in label_lines(row):
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            counts[names.get(class_id, f"class_{class_id}")] += 1
    return dict(sorted(counts.items()))


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


def filter_rows_by_regex(rows: list[str], require_regex: str, block_regex: str) -> tuple[list[str], int, int]:
    require = re.compile(require_regex) if require_regex else None
    block = re.compile(block_regex) if block_regex else None
    kept: list[str] = []
    require_filtered = 0
    block_filtered = 0
    for row in rows:
        if require and not require.search(row):
            require_filtered += 1
            continue
        if block and block.search(row):
            block_filtered += 1
            continue
        kept.append(row)
    if not kept:
        raise SystemExit("extra row regex filters removed all candidate rows")
    return kept, require_filtered, block_filtered


def sample_rows(rows: list[str], max_rows: int | None, seed: int) -> tuple[list[str], int]:
    if max_rows is None or len(rows) <= max_rows:
        return rows, 0
    if max_rows < 1:
        raise SystemExit("--extra-max-images must be at least 1 when set")
    rng = random.Random(seed)
    selected_indexes = set(rng.sample(range(len(rows)), max_rows))
    sampled = [row for index, row in enumerate(rows) if index in selected_indexes]
    return sampled, len(rows) - len(sampled)


def cap_rows_by_class(
    rows: list[str],
    *,
    max_per_class: int | None,
    max_empty: int | None,
    seed: int,
) -> tuple[list[str], int, dict[str, int], int]:
    if max_per_class is None and max_empty is None:
        return rows, 0, {}, 0
    if max_per_class is not None and max_per_class < 0:
        raise SystemExit("--extra-max-per-class must be >= 0 when set")
    if max_empty is not None and max_empty < 0:
        raise SystemExit("--extra-max-empty must be >= 0 when set")

    rng = random.Random(seed)
    priority = {row: (rng.random(), index) for index, row in enumerate(rows)}
    counts: Counter[int] = Counter()
    empty_count = 0
    selected: set[str] = set()

    for row in sorted(rows, key=lambda item: priority[item]):
        class_ids = label_class_ids(row)
        if not class_ids:
            if max_empty is not None and empty_count >= max_empty:
                continue
            selected.add(row)
            empty_count += 1
            continue
        if max_per_class is not None and any(counts[class_id] >= max_per_class for class_id in class_ids):
            continue
        selected.add(row)
        for class_id in class_ids:
            counts[class_id] += 1

    capped_rows = [row for row in rows if row in selected]
    return capped_rows, len(rows) - len(capped_rows), {str(key): counts[key] for key in sorted(counts)}, empty_count


def write_image_list(path: Path, rows: list[str]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.extra_image_dir and not args.extra_image_list:
        raise SystemExit("provide at least one --extra-image-dir or --extra-image-list")
    if args.extra_repeat < 1:
        raise SystemExit("--extra-repeat must be at least 1")
    if args.extra_repeat > 1 and not args.preserve_duplicate_exposures:
        raise SystemExit("--extra-repeat > 1 requires --preserve-duplicate-exposures")

    base_path = resolve(args.base)
    base_config = read_yaml(base_path)
    base_rows, base_sources = train_rows(base_path, base_config)

    extra_rows: list[str] = []
    extra_sources: list[str] = []
    for image_dir in args.extra_image_dir:
        rows = image_rows(image_dir, recursive=args.recursive)
        extra_rows.extend(rows)
        extra_sources.append(repo_rel(resolve(image_dir)))
    for image_list in args.extra_image_list:
        resolved_list = resolve(image_list)
        rows = read_image_list(resolved_list)
        if not rows:
            raise SystemExit(f"empty extra image list: {repo_rel(resolved_list)}")
        extra_rows.extend(rows)
        extra_sources.append(repo_rel(resolved_list))

    extra_rows, extra_duplicate_rows = ordered_unique(extra_rows)
    extra_images_before_regex_filter = len(extra_rows)
    extra_rows, extra_regex_require_filtered_out_images, extra_regex_block_filtered_out_images = filter_rows_by_regex(
        extra_rows,
        args.extra_row_require_regex,
        args.extra_row_block_regex,
    )
    enforce_label_policy(extra_rows, args.extra_label_policy)
    extra_images_before_geometry_filter = len(extra_rows)
    extra_rows, extra_geometry_filtered_out_images, extra_geometry_filter = filter_rows_by_label_geometry(
        extra_rows,
        min_area=args.extra_min_label_area,
        max_area=args.extra_max_label_area,
        edge_margin=args.extra_edge_margin,
        single_label_only=bool(args.extra_single_label_only),
    )
    extra_images_before_class_cap = len(extra_rows)
    extra_rows, extra_class_cap_sampled_out_images, extra_class_cap_counts, extra_class_cap_empty_count = (
        cap_rows_by_class(
            extra_rows,
            max_per_class=args.extra_max_per_class,
            max_empty=args.extra_max_empty,
            seed=args.extra_sample_seed,
        )
    )
    extra_images_before_sampling = len(extra_rows)
    extra_rows, extra_sampled_out_images = sample_rows(extra_rows, args.extra_max_images, args.extra_sample_seed)
    extra_unique_rows = list(extra_rows)
    if args.extra_repeat > 1:
        extra_rows = extra_rows * args.extra_repeat
    if args.preserve_duplicate_exposures:
        combined_rows = base_rows + extra_rows
        combined_unique_rows, combined_duplicate_rows = ordered_unique(combined_rows)
        base_unique_rows = ordered_unique(base_rows)[0]
        base_unique_row_set = set(base_unique_rows)
        added_rows = len([row for row in extra_rows if row not in base_unique_row_set])
        combined_duplicate_images_removed = 0
    else:
        combined_rows, combined_duplicate_rows = ordered_unique(base_rows + extra_rows)
        combined_unique_rows = combined_rows
        base_unique_rows = ordered_unique(base_rows)[0]
        added_rows = len(combined_rows) - len(base_unique_rows)
        combined_duplicate_images_removed = combined_duplicate_rows
    if added_rows < 1:
        raise SystemExit("extra rows did not add any new train images")

    base_names = class_names(base_config)
    names = dict(base_names)
    for extra_name in args.extra_name:
        index, name = parse_extra_name(extra_name)
        names[index] = name
    report = {
        "schema": "cashsnap_yolo_train_mix_v1",
        "tag": args.tag,
        "base_config": repo_rel(base_path),
        "base_train_sources": base_sources,
        "extra_sources": extra_sources,
        "extra_label_policy": args.extra_label_policy,
        "extra_row_require_regex": args.extra_row_require_regex,
        "extra_row_block_regex": args.extra_row_block_regex,
        "extra_min_label_area": args.extra_min_label_area,
        "extra_max_label_area": args.extra_max_label_area,
        "extra_edge_margin": args.extra_edge_margin,
        "extra_single_label_only": bool(args.extra_single_label_only),
        "extra_max_per_class": args.extra_max_per_class,
        "extra_max_empty": args.extra_max_empty,
        "extra_max_images": args.extra_max_images,
        "extra_sample_seed": args.extra_sample_seed,
        "out_extra_list": repo_rel(resolve(args.out_extra_list)) if args.out_extra_list else "",
        "extra_names": {str(index): name for index, name in sorted(names.items()) if base_names.get(index) != name},
        "recursive": bool(args.recursive),
        "preserve_duplicate_exposures": bool(args.preserve_duplicate_exposures),
        "extra_repeat": args.extra_repeat,
        "base_images": len(base_rows),
        "base_unique_images": len(base_unique_rows),
        "extra_images_before_regex_filter": extra_images_before_regex_filter,
        "extra_regex_require_filtered_out_images": extra_regex_require_filtered_out_images,
        "extra_regex_block_filtered_out_images": extra_regex_block_filtered_out_images,
        "extra_images_before_geometry_filter": extra_images_before_geometry_filter,
        "extra_geometry_filtered_out_images": extra_geometry_filtered_out_images,
        "extra_geometry_filter": extra_geometry_filter,
        "extra_images_before_class_cap": extra_images_before_class_cap,
        "extra_class_cap_sampled_out_images": extra_class_cap_sampled_out_images,
        "extra_class_cap_counts": extra_class_cap_counts,
        "extra_class_cap_empty_count": extra_class_cap_empty_count,
        "extra_images_before_sampling": extra_images_before_sampling,
        "extra_images": len(extra_rows),
        "extra_unique_images": len(extra_unique_rows),
        "extra_duplicate_images": extra_duplicate_rows,
        "extra_sampled_out_images": extra_sampled_out_images,
        "combined_images": len(combined_rows),
        "combined_unique_images": len(combined_unique_rows),
        "combined_duplicate_exposures": combined_duplicate_rows,
        "combined_duplicate_images_removed": combined_duplicate_images_removed,
        "added_images": added_rows,
        "extra_class_counts": label_class_counts(extra_rows, names),
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0

    write_image_list(args.out_list, combined_rows)
    if args.out_extra_list is not None:
        write_image_list(args.out_extra_list, extra_rows)
    out_config = resolve(args.out_config)
    config = copy.deepcopy(base_config)
    config["path"] = rel_between(out_config.parent, ROOT)
    config["train"] = repo_rel(resolve(args.out_list))
    for split in ("val", "test"):
        if split in config:
            config[split] = repo_relative_split_value(base_path, base_config, config[split])
    config["names"] = {index: names[index] for index in sorted(names)}

    sources = copy.deepcopy(config.get("cashsnap_sources", {}))
    if not isinstance(sources, dict):
        sources = {}
    sources["train_mix_base_config"] = repo_rel(base_path)
    sources["train_mix_extra_sources"] = extra_sources
    config["cashsnap_sources"] = sources
    config["cashsnap_train_mix"] = report

    policy = copy.deepcopy(config.get("cashsnap_policy", {}))
    if not isinstance(policy, dict):
        policy = {}
    if args.intended_use:
        policy["intended_use"] = args.intended_use
    if args.promotion_rule:
        policy["promotion_rule"] = args.promotion_rule
    config["cashsnap_policy"] = policy

    write_yaml(out_config, config)
    print(json.dumps(report, indent=2))
    print(f"wrote_list={repo_rel(resolve(args.out_list))}")
    print(f"wrote_config={repo_rel(out_config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
