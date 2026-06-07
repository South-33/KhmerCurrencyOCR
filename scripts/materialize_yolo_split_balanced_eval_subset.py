#!/usr/bin/env python
"""Write a YOLO config with a balanced eval subset for one split."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def rel_between(base: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), base.resolve()).replace("\\", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--source-split", default="test")
    parser.add_argument("--target-split", default="test", choices=["val", "test"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--list-out", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--backgrounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"YOLO data YAML must be a mapping: {repo_rel(path)}")
    return config


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


def image_classes(image_path: Path) -> set[int]:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        raise SystemExit(f"Missing label: {repo_rel(label_path)}")
    classes: set[int] = set()
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} expected 5 YOLO fields")
        classes.add(int(parts[0]))
    return classes


def select_subset(
    images: list[Path],
    class_count: int,
    per_class: int,
    backgrounds: int,
    seed: int,
) -> tuple[list[Path], Counter[int], int]:
    rng = random.Random(seed)
    pool = list(images)
    rng.shuffle(pool)
    selected: list[Path] = []
    selected_set: set[Path] = set()
    class_counts: Counter[int] = Counter()
    background_count = 0
    for image in pool:
        classes = {class_id for class_id in image_classes(image) if 0 <= class_id < class_count}
        if not classes:
            include = background_count < backgrounds
        else:
            include = any(class_counts[class_id] < per_class for class_id in classes)
        if not include or image.resolve() in selected_set:
            continue
        selected.append(image)
        selected_set.add(image.resolve())
        if not classes:
            background_count += 1
        for class_id in classes:
            class_counts[class_id] += 1
        if background_count >= backgrounds and all(class_counts[index] >= per_class for index in range(class_count)):
            break
    missing = [str(index) for index in range(class_count) if class_counts[index] < per_class]
    if missing:
        raise SystemExit(f"subset underfilled class ids: {', '.join(missing)}")
    return selected, class_counts, background_count


def main() -> None:
    args = parse_args()
    if args.per_class < 1:
        raise SystemExit("--per-class must be >= 1")
    if args.backgrounds < 0:
        raise SystemExit("--backgrounds must be >= 0")
    data_path = resolve(args.data)
    out_path = resolve(args.out)
    list_out = resolve(args.list_out)
    config = load_config(data_path)
    names = config.get("names") or {}
    class_count = len(names)
    images = split_images(data_path, config, args.source_split)
    selected, class_counts, background_count = select_subset(
        images,
        class_count=class_count,
        per_class=args.per_class,
        backgrounds=args.backgrounds,
        seed=args.seed,
    )
    list_out.parent.mkdir(parents=True, exist_ok=True)
    list_out.write_text("\n".join(repo_rel(image) for image in selected) + "\n", encoding="utf-8")

    out_config = dict(config)
    out_config["path"] = rel_between(out_path.parent, ROOT)
    out_config[args.target_split] = repo_rel(list_out)
    policy = dict(out_config.get("cashsnap_eval_subset_policy") or {})
    policy.update(
        {
            "schema": "cashsnap_yolo_balanced_eval_subset_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_data": repo_rel(data_path),
            "source_split": args.source_split,
            "target_split": args.target_split,
            "list": repo_rel(list_out),
            "seed": args.seed,
            "per_class": args.per_class,
            "backgrounds": args.backgrounds,
            "selected_images": len(selected),
            "selected_backgrounds": background_count,
            "selected_class_images": {str(names[index]): int(class_counts[index]) for index in range(class_count)},
        }
    )
    out_config["cashsnap_eval_subset_policy"] = policy
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out_config, sort_keys=False), encoding="utf-8")
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote={repo_rel(out_path)} list={repo_rel(list_out)} "
        f"selected={len(selected)} backgrounds={background_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
