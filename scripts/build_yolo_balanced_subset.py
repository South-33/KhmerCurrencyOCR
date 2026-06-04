from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, list-backed YOLO subset with class balance guards."
    )
    parser.add_argument("--data", required=True, help="Source YOLO data YAML.")
    parser.add_argument("--out", required=True, help="Output YOLO data YAML.")
    parser.add_argument("--train-list", default=None, help="Output train image list. Defaults beside --out.")
    parser.add_argument("--per-class", type=int, default=24, help="Target real labeled images per class.")
    parser.add_argument("--backgrounds", type=int, default=24, help="Target empty-label real background images.")
    parser.add_argument(
        "--include-all-target-images",
        action="store_true",
        help="Include every non-excluded target image instead of stopping at per-class/background targets.",
    )
    parser.add_argument(
        "--always-include-prefix",
        action="append",
        default=None,
        help=(
            "Repo-relative path prefix to include even after class targets are met. "
            "Repeatable. Defaults to data/synthetic/ when omitted."
        ),
    )
    parser.add_argument(
        "--no-always-include",
        action="store_true",
        help="Do not force-include any prefix; useful for real-only balanced baselines.",
    )
    parser.add_argument(
        "--always-max-per-class",
        type=int,
        default=None,
        help=(
            "Optional cap on labeled always-included images per class. "
            "Useful for preventing synthetic recipes from dominating rare real classes."
        ),
    )
    parser.add_argument(
        "--oversample-min-per-class",
        type=int,
        default=None,
        help=(
            "Repeat selected labeled images until each represented class has at least this many "
            "train-list appearances. Duplicates are appended after capped selection, so this can "
            "raise rare-class exposure without admitting more backgrounds/common-class images."
        ),
    )
    parser.add_argument(
        "--target-exclude-prefix",
        action="append",
        default=None,
        help=(
            "Repo-relative path prefix to skip for target real/background filling unless it is always-included. "
            "Repeatable. Defaults to data/synthetic/."
        ),
    )
    parser.add_argument(
        "--no-target-exclude",
        action="store_true",
        help="Allow non-always-included images from any prefix to fill target class/background slots.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_from_root(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_yaml(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return config


def data_root(config_path: Path, config: dict) -> Path:
    root_value = config.get("path", ".")
    root = Path(str(root_value)).expanduser()
    return root if root.is_absolute() else (config_path.parent / root).resolve()


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


def iter_split_images(dataset_root: Path, split_paths: str | list[str]) -> list[Path]:
    paths = split_paths if isinstance(split_paths, list) else [split_paths]
    images: list[Path] = []
    for split_path in paths:
        resolved = split_root(dataset_root, split_path)
        if resolved.suffix.lower() == ".txt":
            images.extend(read_split_list(dataset_root, split_path))
        else:
            images.extend(
                sorted(p for p in resolved.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
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


def label_classes(image: Path) -> list[int]:
    label = label_path_for_image(image)
    if not label.exists():
        raise FileNotFoundError(f"missing label for {image}: {label}")
    classes: list[int] = []
    for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label}:{line_no} expected 5 YOLO fields, found {len(parts)}")
        classes.append(int(parts[0]))
    return classes


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def rel_between(base: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), base.resolve()).replace("\\", "/")


def normalize_prefix(prefix: str) -> str:
    normalized = prefix.replace("\\", "/").strip("/")
    return f"{normalized}/" if normalized else normalized


def split_to_repo_relative(config_path: Path, config: dict, split_value: str | list[str]) -> str | list[str]:
    root = data_root(config_path, config)

    def convert(value: str) -> str:
        return repo_rel(split_root(root, value))

    if isinstance(split_value, list):
        return [convert(value) for value in split_value]
    return convert(split_value)


def should_always_include(image: Path, prefixes: list[str]) -> bool:
    rel = repo_rel(image)
    return any(rel.startswith(prefix) for prefix in prefixes)


def should_exclude_from_target(image: Path, prefixes: list[str]) -> bool:
    rel = repo_rel(image)
    return any(rel.startswith(prefix) for prefix in prefixes)


def build_subset(
    images: list[Path],
    class_count: int,
    per_class: int,
    backgrounds: int,
    always_prefixes: list[str],
    target_exclude_prefixes: list[str],
    include_all_targets: bool,
    always_max_per_class: int | None,
) -> tuple[list[Path], Counter[int], Counter[int], Counter[int], int, int, int, int, int]:
    selected: list[Path] = []
    selected_set: set[Path] = set()
    target_class_counts: Counter[int] = Counter()
    total_class_counts: Counter[int] = Counter()
    always_class_counts: Counter[int] = Counter()
    target_background_count = 0
    total_background_count = 0
    always_count = 0
    target_excluded_count = 0
    always_skipped_by_class_cap = 0

    for image in images:
        classes = label_classes(image)
        unique_classes = {cls for cls in set(classes) if 0 <= cls < class_count}
        always = should_always_include(image, always_prefixes)
        if always:
            over_class_cap = (
                always_max_per_class is not None
                and bool(unique_classes)
                and any(always_class_counts[cls] >= always_max_per_class for cls in unique_classes)
            )
            if over_class_cap:
                always_skipped_by_class_cap += 1
                include = False
            else:
                include = True
        elif should_exclude_from_target(image, target_exclude_prefixes):
            target_excluded_count += 1
            include = False
        elif include_all_targets:
            include = True
        elif not classes:
            include = target_background_count < backgrounds
        else:
            include = any(target_class_counts[cls] < per_class for cls in set(classes))

        if not include or image in selected_set:
            continue

        selected.append(image)
        selected_set.add(image)
        if always:
            always_count += 1
            for cls in unique_classes:
                always_class_counts[cls] += 1
        if not classes:
            total_background_count += 1
            if not always:
                target_background_count += 1
        for cls in unique_classes:
            total_class_counts[cls] += 1
            if not always:
                target_class_counts[cls] += 1

    return (
        selected,
        total_class_counts,
        target_class_counts,
        always_class_counts,
        target_background_count,
        total_background_count,
        always_count,
        always_skipped_by_class_cap,
        target_excluded_count,
    )


def oversample_underrepresented(
    selected: list[Path],
    class_count: int,
    min_per_class: int | None,
) -> tuple[list[Path], Counter[int], Counter[int], int]:
    if min_per_class is None:
        appearances = Counter()
        for image in selected:
            for cls in {cls for cls in set(label_classes(image)) if 0 <= cls < class_count}:
                appearances[cls] += 1
        return selected, appearances, Counter(), 0
    if min_per_class < 1:
        raise SystemExit("--oversample-min-per-class must be positive when provided")

    class_to_pool: dict[int, list[Path]] = {class_id: [] for class_id in range(class_count)}
    image_classes: dict[Path, set[int]] = {}
    appearances: Counter[int] = Counter()
    oversampled_by_trigger: Counter[int] = Counter()
    expanded = list(selected)

    for image in selected:
        classes = {cls for cls in set(label_classes(image)) if 0 <= cls < class_count}
        image_classes[image] = classes
        for cls in classes:
            class_to_pool[cls].append(image)
            appearances[cls] += 1

    added = 0
    for class_id in range(class_count):
        pool = class_to_pool[class_id]
        if not pool:
            continue
        pool_index = 0
        while appearances[class_id] < min_per_class:
            image = pool[pool_index % len(pool)]
            expanded.append(image)
            oversampled_by_trigger[class_id] += 1
            added += 1
            for cls in image_classes[image]:
                appearances[cls] += 1
            pool_index += 1

    return expanded, appearances, oversampled_by_trigger, added


def main() -> None:
    args = parse_args()
    if args.no_always_include and args.always_include_prefix:
        raise SystemExit("--no-always-include cannot be combined with --always-include-prefix")
    if args.always_max_per_class is not None and args.always_max_per_class < 1:
        raise SystemExit("--always-max-per-class must be positive when provided")
    if args.oversample_min_per_class is not None and args.oversample_min_per_class < 1:
        raise SystemExit("--oversample-min-per-class must be positive when provided")

    data_path = resolve_from_root(args.data)
    out_path = resolve_from_root(args.out)
    train_list = resolve_from_root(args.train_list) if args.train_list else out_path.with_name(f"{out_path.stem}_train.txt")

    config = read_yaml(data_path)
    root = data_root(data_path, config)
    names = config["names"]
    class_count = len(names)
    raw_prefixes = [] if args.no_always_include else (args.always_include_prefix or ["data/synthetic/"])
    prefixes = [prefix for prefix in (normalize_prefix(prefix) for prefix in raw_prefixes) if prefix]
    raw_target_exclude_prefixes = [] if args.no_target_exclude else (args.target_exclude_prefix or ["data/synthetic/"])
    target_exclude_prefixes = [
        prefix for prefix in (normalize_prefix(prefix) for prefix in raw_target_exclude_prefixes) if prefix
    ]
    images = iter_split_images(root, config["train"])
    (
        selected,
        counts,
        target_counts,
        always_counts,
        target_background_count,
        total_background_count,
        always_count,
        always_skipped_by_class_cap,
        target_excluded_count,
    ) = build_subset(
        images,
        class_count=class_count,
        per_class=args.per_class,
        backgrounds=args.backgrounds,
        always_prefixes=prefixes,
        target_exclude_prefixes=target_exclude_prefixes,
        include_all_targets=args.include_all_target_images,
        always_max_per_class=args.always_max_per_class,
    )
    selected_unique_count = len(selected)
    selected, appearance_counts, oversampled_by_trigger, oversample_added_count = oversample_underrepresented(
        selected,
        class_count=class_count,
        min_per_class=args.oversample_min_per_class,
    )

    missing = [str(names[index]) for index in range(class_count) if counts[index] == 0]
    if missing:
        raise SystemExit(f"selected subset has no labels for class(es): {', '.join(missing)}")
    if not selected:
        raise SystemExit("selected subset is empty")

    out_config = {
        "path": rel_between(out_path.parent, ROOT),
        "train": repo_rel(train_list),
        "val": split_to_repo_relative(data_path, config, config["val"]),
        "test": split_to_repo_relative(data_path, config, config["test"]) if config.get("test") else None,
        "names": names,
        "cashsnap_sources": {
            "source_data": repo_rel(data_path),
            "always_include_prefixes": prefixes,
            "target_exclude_prefixes": target_exclude_prefixes,
        },
        "cashsnap_subset_policy": {
            "per_class_real_target": args.per_class,
            "background_target": args.backgrounds,
            "include_all_target_images": args.include_all_target_images,
            "always_max_per_class": args.always_max_per_class,
            "oversample_min_per_class": args.oversample_min_per_class,
            "oversample_added_images": oversample_added_count,
            "oversample_trigger_class_images": {
                str(names[index]): oversampled_by_trigger[index] for index in range(class_count)
            },
            "always_included_images": always_count,
            "always_skipped_by_class_cap": always_skipped_by_class_cap,
            "target_excluded_images": target_excluded_count,
            "selected_unique_images": selected_unique_count,
            "selected_images": len(selected),
            "selected_backgrounds": total_background_count,
            "selected_target_backgrounds": target_background_count,
            "selected_class_images": {str(names[index]): appearance_counts[index] for index in range(class_count)},
            "selected_target_class_images": {str(names[index]): target_counts[index] for index in range(class_count)},
            "selected_always_class_images": {
                str(names[index]): always_counts[index] for index in range(class_count)
            },
        },
    }
    if out_config["test"] is None:
        out_config.pop("test")

    print(
        f"selected {len(selected)} train images, target_backgrounds={target_background_count}, "
        f"total_backgrounds={total_background_count}, always_included={always_count}, "
        f"always_skipped_by_class_cap={always_skipped_by_class_cap}, "
        f"oversample_added={oversample_added_count}, "
        f"target_excluded={target_excluded_count}",
        flush=True,
    )
    for class_id, class_name in names.items():
        class_index = int(class_id)
        print(
            f"  {class_name}: appearances={appearance_counts[class_index]}, "
            f"target_unique={target_counts[class_index]}, always_unique={always_counts[class_index]}, "
            f"oversample_trigger={oversampled_by_trigger[class_index]}",
            flush=True,
        )

    if args.dry_run:
        return

    train_list.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    train_list.write_text(
        "\n".join(repo_rel(image) for image in selected) + "\n",
        encoding="utf-8",
    )
    out_path.write_text(yaml.safe_dump(out_config, sort_keys=False), encoding="utf-8")
    print(f"wrote {repo_rel(out_path)}")
    print(f"wrote {repo_rel(train_list)}")


if __name__ == "__main__":
    main()
