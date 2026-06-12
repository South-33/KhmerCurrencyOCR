#!/usr/bin/env python
"""Build a YOLO config that matches another config's class exposure by duplication."""

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
    parser.add_argument("--base", type=Path, required=True, help="Base YOLO config to duplicate from.")
    parser.add_argument("--target", type=Path, required=True, help="YOLO config whose train exposure should be matched.")
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--out-list", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="exposure_control")
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
    return rows, sources


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
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} expected 5 YOLO fields")
        try:
            class_ids.append(int(float(parts[0])))
        except ValueError as exc:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} has malformed class id: {parts[0]}") from exc
    return class_ids


def exposure_counts(rows: list[str]) -> tuple[Counter[int], int]:
    counts: Counter[int] = Counter()
    empty = 0
    for row in rows:
        class_ids = label_class_ids(row)
        if not class_ids:
            empty += 1
        for class_id in class_ids:
            counts[class_id] += 1
    return counts, empty


def pools_by_single_class(rows: list[str]) -> tuple[dict[int, list[str]], list[str], list[str]]:
    by_class: dict[int, list[str]] = defaultdict(list)
    empty: list[str] = []
    multi: list[str] = []
    for row in rows:
        class_ids = label_class_ids(row)
        if not class_ids:
            empty.append(row)
        elif len(set(class_ids)) == 1 and len(class_ids) == 1:
            by_class[class_ids[0]].append(row)
        else:
            multi.append(row)
    return by_class, empty, multi


def duplicate_from_pool(pool: list[str], count: int, rng: random.Random, label: str) -> list[str]:
    if count <= 0:
        return []
    if not pool:
        raise SystemExit(f"cannot duplicate {label}: empty source pool")
    selected: list[str] = []
    shuffled = list(pool)
    while len(selected) < count:
        rng.shuffle(shuffled)
        selected.extend(shuffled[: count - len(selected)])
    return selected


def named_counts(counts: Counter[int], names: dict[int, str]) -> dict[str, int]:
    return {names.get(class_id, f"class_{class_id}"): counts[class_id] for class_id in sorted(counts)}


def write_image_list(path: Path, rows: list[str]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_path = resolve(args.base)
    target_path = resolve(args.target)
    base_config = read_yaml(base_path)
    target_config = read_yaml(target_path)
    names = class_names(base_config)
    if names != class_names(target_config):
        raise SystemExit("base and target configs must use identical class names")

    base_rows, base_sources = train_rows(base_path, base_config)
    target_rows, target_sources = train_rows(target_path, target_config)
    base_counts, base_empty = exposure_counts(base_rows)
    target_counts, target_empty = exposure_counts(target_rows)
    class_pools, empty_pool, multi_pool = pools_by_single_class(base_rows)
    rng = random.Random(args.seed)

    duplicate_rows: list[str] = []
    duplicated_by_class: Counter[int] = Counter()
    for class_id in sorted(set(target_counts) | set(base_counts)):
        deficit = target_counts[class_id] - base_counts[class_id]
        if deficit < 0:
            raise SystemExit(
                f"target has lower exposure than base for {names.get(class_id, class_id)}: "
                f"{target_counts[class_id]} < {base_counts[class_id]}"
            )
        selected = duplicate_from_pool(class_pools.get(class_id, []), deficit, rng, names.get(class_id, str(class_id)))
        duplicate_rows.extend(selected)
        duplicated_by_class[class_id] += len(selected)

    empty_deficit = target_empty - base_empty
    if empty_deficit < 0:
        raise SystemExit(f"target has lower empty exposure than base: {target_empty} < {base_empty}")
    duplicate_empty_rows = duplicate_from_pool(empty_pool, empty_deficit, rng, "empty-label rows")
    duplicate_rows.extend(duplicate_empty_rows)

    combined_rows = base_rows + duplicate_rows
    combined_counts, combined_empty = exposure_counts(combined_rows)
    report = {
        "schema": "cashsnap_yolo_exposure_control_v1",
        "tag": args.tag,
        "base_config": repo_rel(base_path),
        "target_config": repo_rel(target_path),
        "base_train_sources": base_sources,
        "target_train_sources": target_sources,
        "seed": args.seed,
        "base_images": len(base_rows),
        "target_images": len(target_rows),
        "combined_images": len(combined_rows),
        "duplicated_images": len(duplicate_rows),
        "base_empty": base_empty,
        "target_empty": target_empty,
        "combined_empty": combined_empty,
        "duplicated_empty": len(duplicate_empty_rows),
        "base_class_counts": named_counts(base_counts, names),
        "target_class_counts": named_counts(target_counts, names),
        "combined_class_counts": named_counts(combined_counts, names),
        "duplicated_by_class": named_counts(duplicated_by_class, names),
        "base_multi_label_images": len(multi_pool),
        "class_mix_exact": combined_counts == target_counts and combined_empty == target_empty,
    }
    if not report["class_mix_exact"]:
        raise SystemExit("internal error: exposure control did not match target counts")
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
    sources["exposure_control_base_config"] = repo_rel(base_path)
    sources["exposure_control_target_config"] = repo_rel(target_path)
    config["cashsnap_sources"] = sources
    config["cashsnap_exposure_control"] = report

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
