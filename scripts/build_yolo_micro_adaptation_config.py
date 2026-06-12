#!/usr/bin/env python
"""Build tiny class-balanced YOLO adaptation configs with a matched control."""

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
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--extra-image-dir", required=True, type=Path)
    parser.add_argument("--base-per-class", type=int, default=4)
    parser.add_argument("--extra-per-class", type=int, default=4)
    parser.add_argument("--empty-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--extra-classes-only",
        action="store_true",
        help="Only append candidate/control extra rows for classes present under --extra-image-dir.",
    )
    parser.add_argument(
        "--allow-missing-base-classes",
        action="store_true",
        help="Skip base rehearsal for official classes with no base rows instead of failing.",
    )
    parser.add_argument(
        "--missing-extra-control",
        choices=("error", "empty", "skip"),
        default="error",
        help="Control-row policy when an extra class has no matching base rows.",
    )
    parser.add_argument("--val", default="")
    parser.add_argument("--test", default="")
    parser.add_argument("--candidate-out-config", required=True, type=Path)
    parser.add_argument("--candidate-out-list", required=True, type=Path)
    parser.add_argument("--control-out-config", required=True, type=Path)
    parser.add_argument("--control-out-list", required=True, type=Path)
    parser.add_argument("--tag", default="micro_adaptation")
    parser.add_argument("--intended-use", default="")
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
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(path)} must be a YAML mapping")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def names_by_id(payload: dict[str, Any]) -> dict[int, str]:
    raw = payload.get("names")
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    raise SystemExit("base config must contain names as a mapping or list")


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw = Path(str(config.get("path", "."))).expanduser()
    return raw if raw.is_absolute() else (config_path.parent / raw).resolve()


def split_root(root: Path, split_value: str) -> Path:
    path = Path(split_value).expanduser()
    return path if path.is_absolute() else root / path


def read_image_list(root: Path, split_value: str) -> list[str]:
    path = split_root(root, split_value)
    rows: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        image = Path(line).expanduser()
        rows.append(repo_rel(image if image.is_absolute() else root / image))
    return rows


def train_rows(config_path: Path, config: dict[str, Any]) -> list[str]:
    root = data_root(config_path, config)
    train = config.get("train")
    values = train if isinstance(train, list) else [train]
    rows: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise SystemExit(f"{repo_rel(config_path)} train split must be string/list of strings")
        path = split_root(root, value)
        if path.suffix.lower() == ".txt":
            rows.extend(read_image_list(root, value))
        elif path.is_dir():
            rows.extend(repo_rel(item) for item in sorted(path.iterdir()) if item.suffix.lower() in IMAGE_EXTS)
        else:
            raise SystemExit(f"cannot read train split: {repo_rel(path)}")
    return rows


def image_rows(root: Path) -> list[str]:
    image_dir = resolve(root)
    rows = [repo_rel(path) for path in sorted(image_dir.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    if not rows:
        raise SystemExit(f"no images under {repo_rel(image_dir)}")
    return rows


def label_path_for_image(image: str) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def label_class_ids(image: str) -> list[int]:
    label_path = resolve(label_path_for_image(image))
    if not label_path.exists():
        return []
    class_ids: list[int] = []
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} expected YOLO label fields")
        class_ids.append(int(float(parts[0])))
    return class_ids


def pools_by_class(rows: list[str]) -> tuple[dict[int, list[str]], list[str]]:
    by_class: dict[int, list[str]] = defaultdict(list)
    empty: list[str] = []
    for row in rows:
        class_ids = label_class_ids(row)
        if not class_ids:
            empty.append(row)
        elif len(class_ids) == 1:
            by_class[class_ids[0]].append(row)
    return by_class, empty


def take(pool: list[str], count: int, rng: random.Random, label: str) -> list[str]:
    if count <= 0:
        return []
    if not pool:
        raise SystemExit(f"empty pool for {label}")
    ranked = list(pool)
    rng.shuffle(ranked)
    selected: list[str] = []
    while len(selected) < count:
        selected.extend(ranked[: count - len(selected)])
    return selected


def write_list(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def class_counts(rows: list[str], names: dict[int, str]) -> dict[str, int]:
    counts: Counter[int] = Counter()
    empty = 0
    for row in rows:
        class_ids = label_class_ids(row)
        if not class_ids:
            empty += 1
        for class_id in class_ids:
            counts[class_id] += 1
    result = {names.get(class_id, f"class_{class_id}"): counts[class_id] for class_id in sorted(counts)}
    if empty:
        result["__empty__"] = empty
    return result


def data_config(
    base_config: dict[str, Any],
    base_path: Path,
    out_config: Path,
    out_list: Path,
    names: dict[int, str],
    *,
    val: str,
    test: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["path"] = rel_between(out_config.parent, ROOT)
    config["train"] = repo_rel(out_list)
    if val:
        config["val"] = val
    if test:
        config["test"] = test
    config["names"] = {index: names[index] for index in sorted(names)}
    sources = copy.deepcopy(config.get("cashsnap_sources", {}))
    if not isinstance(sources, dict):
        sources = {}
    sources["micro_adaptation_base_config"] = repo_rel(base_path)
    sources["micro_adaptation_train_list"] = repo_rel(out_list)
    config["cashsnap_sources"] = sources
    config["cashsnap_micro_adaptation"] = meta
    return config


def main() -> int:
    args = parse_args()
    if args.base_per_class < 0 or args.extra_per_class < 0 or args.empty_count < 0:
        raise SystemExit("row caps must be non-negative")
    base_path = resolve(args.base)
    base_config = read_yaml(base_path)
    names = names_by_id(base_config)
    rng = random.Random(args.seed)

    base_rows = train_rows(base_path, base_config)
    extra_rows = image_rows(args.extra_image_dir)
    base_by_class, base_empty = pools_by_class(base_rows)
    extra_by_class, _ = pools_by_class(extra_rows)

    rehearsal_rows: list[str] = []
    control_extra_rows: list[str] = []
    candidate_extra_rows: list[str] = []
    skipped_base_classes: list[str] = []
    skipped_extra_classes: list[str] = []
    empty_control_classes: list[str] = []
    for class_id in sorted(names):
        class_name = names[class_id]
        base_pool = base_by_class.get(class_id, [])
        extra_pool = extra_by_class.get(class_id, [])
        if base_pool:
            rehearsal_rows.extend(take(base_pool, args.base_per_class, rng, f"base rehearsal {class_name}"))
        elif args.base_per_class:
            if not args.allow_missing_base_classes:
                raise SystemExit(f"empty pool for base rehearsal {class_name}")
            skipped_base_classes.append(class_name)

        if args.extra_classes_only and not extra_pool:
            skipped_extra_classes.append(class_name)
            continue

        candidate_extra_rows.extend(take(extra_pool, args.extra_per_class, rng, f"candidate extra {class_name}"))
        if base_pool:
            control_extra_rows.extend(take(base_pool, args.extra_per_class, rng, f"control extra {class_name}"))
        elif args.extra_per_class:
            if args.missing_extra_control == "error":
                raise SystemExit(f"empty pool for control extra {class_name}")
            if args.missing_extra_control == "empty":
                control_extra_rows.extend(take(base_empty, args.extra_per_class, rng, f"control extra empty for {class_name}"))
                empty_control_classes.append(class_name)

    empty_rows = take(base_empty, args.empty_count, rng, "empty rehearsal")
    candidate_rows = rehearsal_rows + empty_rows + candidate_extra_rows
    control_rows = rehearsal_rows + empty_rows + control_extra_rows

    candidate_list = resolve(args.candidate_out_list)
    control_list = resolve(args.control_out_list)
    candidate_config_path = resolve(args.candidate_out_config)
    control_config_path = resolve(args.control_out_config)
    write_list(candidate_list, candidate_rows)
    write_list(control_list, control_rows)

    common_meta = {
        "schema": "cashsnap_yolo_micro_adaptation_v1",
        "tag": args.tag,
        "base_config": repo_rel(base_path),
        "extra_image_dir": repo_rel(resolve(args.extra_image_dir)),
        "seed": args.seed,
        "base_per_class": args.base_per_class,
        "extra_per_class": args.extra_per_class,
        "empty_count": args.empty_count,
        "extra_classes_only": args.extra_classes_only,
        "allow_missing_base_classes": args.allow_missing_base_classes,
        "missing_extra_control": args.missing_extra_control,
        "skipped_base_classes": skipped_base_classes,
        "skipped_extra_classes": skipped_extra_classes,
        "empty_control_classes": empty_control_classes,
        "val_override": args.val,
        "test_override": args.test,
        "intended_use": args.intended_use,
    }
    candidate_meta = {
        **common_meta,
        "role": "candidate",
        "rows": len(candidate_rows),
        "class_counts": class_counts(candidate_rows, names),
    }
    control_meta = {
        **common_meta,
        "role": "control",
        "rows": len(control_rows),
        "class_counts": class_counts(control_rows, names),
    }
    write_yaml(
        candidate_config_path,
        data_config(
            base_config,
            base_path,
            candidate_config_path,
            candidate_list,
            names,
            val=args.val,
            test=args.test,
            meta=candidate_meta,
        ),
    )
    write_yaml(
        control_config_path,
        data_config(
            base_config,
            base_path,
            control_config_path,
            control_list,
            names,
            val=args.val,
            test=args.test,
            meta=control_meta,
        ),
    )
    summary = {
        "candidate_config": repo_rel(candidate_config_path),
        "control_config": repo_rel(control_config_path),
        "candidate_rows": len(candidate_rows),
        "control_rows": len(control_rows),
        "candidate_counts": candidate_meta["class_counts"],
        "control_counts": control_meta["class_counts"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
