#!/usr/bin/env python
"""Build a YOLO config by appending repeated rows for selected classes."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument(
        "--repeat-rule",
        action="append",
        required=True,
        help=(
            "Repeat rule as CLASS:REPEAT[:REQUIRE_REGEX[:BLOCK_REGEX]]. "
            "REPEAT is extra copies per selected row."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-config", required=True, type=Path)
    parser.add_argument("--out-list", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
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
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(path)} must be a YAML mapping")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def read_image_list(path: Path) -> list[str]:
    rows: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        image = Path(line).expanduser()
        rows.append(repo_rel(image if image.is_absolute() else resolve(image)))
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


def label_path_for_image(image: str) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return resolve(path.with_suffix(".txt"))
    parts[index] = "labels"
    return resolve(Path(*parts).with_suffix(".txt"))


def label_class_ids(image: str) -> list[int]:
    label_path = label_path_for_image(image)
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


def parse_repeat_rule(value: str, name_to_id: dict[str, int]) -> dict[str, Any]:
    parts = value.split(":", 3)
    if len(parts) < 2:
        raise SystemExit(f"--repeat-rule must be CLASS:REPEAT[:REQUIRE_REGEX[:BLOCK_REGEX]]: {value}")
    class_name = parts[0]
    if class_name not in name_to_id:
        raise SystemExit(f"unknown class in --repeat-rule: {class_name}")
    try:
        repeat = int(parts[1])
    except ValueError as exc:
        raise SystemExit(f"repeat count must be an integer in --repeat-rule: {value}") from exc
    if repeat < 1:
        raise SystemExit(f"repeat count must be at least 1 in --repeat-rule: {value}")
    require_regex = parts[2] if len(parts) >= 3 else ""
    block_regex = parts[3] if len(parts) >= 4 else ""
    return {
        "class_name": class_name,
        "class_id": name_to_id[class_name],
        "repeat": repeat,
        "require_regex": require_regex,
        "block_regex": block_regex,
    }


def row_matches_rule(row: str, class_ids: set[int], rule: dict[str, Any]) -> bool:
    if rule["class_id"] not in class_ids:
        return False
    if rule["require_regex"] and re.search(str(rule["require_regex"]), row) is None:
        return False
    if rule["block_regex"] and re.search(str(rule["block_regex"]), row) is not None:
        return False
    return True


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


def named_counts(counts: Counter[int], names: dict[int, str]) -> dict[str, int]:
    return {names.get(class_id, f"class_{class_id}"): int(counts[class_id]) for class_id in sorted(counts)}


def write_image_list(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_path = resolve(args.base_config)
    base_config = read_yaml(base_path)
    names = names_by_id(base_config)
    name_to_id = {name: class_id for class_id, name in names.items()}
    rules = [parse_repeat_rule(value, name_to_id) for value in args.repeat_rule]

    base_rows, base_sources = train_rows(base_path, base_config)
    classes_by_row = {row: set(label_class_ids(row)) for row in set(base_rows)}
    rng = random.Random(args.seed)

    appended_rows: list[str] = []
    rule_summaries: list[dict[str, Any]] = []
    for rule_index, rule in enumerate(rules):
        candidates = [row for row in base_rows if row_matches_rule(row, classes_by_row.get(row, set()), rule)]
        if not candidates:
            raise SystemExit(f"repeat rule {rule_index} selected no rows: {rule}")
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        repeated_once = shuffled * int(rule["repeat"])
        appended_rows.extend(repeated_once)
        rule_summaries.append(
            {
                "class": rule["class_name"],
                "class_id": rule["class_id"],
                "repeat": rule["repeat"],
                "require_regex": rule["require_regex"],
                "block_regex": rule["block_regex"],
                "selected_rows": len(candidates),
                "selected_unique_rows": len(set(candidates)),
                "appended_rows": len(repeated_once),
                "selected_sample": sorted(candidates)[:20],
            }
        )

    combined_rows = base_rows + appended_rows
    base_counts, base_empty = exposure_counts(base_rows)
    appended_counts, appended_empty = exposure_counts(appended_rows)
    combined_counts, combined_empty = exposure_counts(combined_rows)
    summary = {
        "schema": "cashsnap_yolo_class_repeat_config_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "base_config": repo_rel(base_path),
        "base_train_sources": base_sources,
        "out_config": repo_rel(resolve(args.out_config)),
        "out_list": repo_rel(resolve(args.out_list)),
        "seed": args.seed,
        "rules": rule_summaries,
        "base_rows": len(base_rows),
        "base_unique_rows": len(set(base_rows)),
        "appended_rows": len(appended_rows),
        "appended_unique_rows": len(set(appended_rows)),
        "combined_rows": len(combined_rows),
        "combined_unique_rows": len(set(combined_rows)),
        "combined_duplicate_rows": len(combined_rows) - len(set(combined_rows)),
        "base_empty_rows": base_empty,
        "appended_empty_rows": appended_empty,
        "combined_empty_rows": combined_empty,
        "base_class_counts": named_counts(base_counts, names),
        "appended_class_counts": named_counts(appended_counts, names),
        "combined_class_counts": named_counts(combined_counts, names),
        "intended_use": args.intended_use,
        "promotion_rule": args.promotion_rule,
    }

    output_config = copy.deepcopy(base_config)
    output_config["path"] = rel_between(resolve(args.out_config).parent, ROOT)
    output_config["train"] = repo_rel(resolve(args.out_list))
    output_config["cashsnap_class_repeat"] = summary
    if args.intended_use:
        output_config.setdefault("cashsnap_policy", {})["intended_use"] = args.intended_use
    if args.promotion_rule:
        output_config.setdefault("cashsnap_policy", {})["promotion_rule"] = args.promotion_rule

    if not args.dry_run:
        write_image_list(resolve(args.out_list), combined_rows)
        write_yaml(resolve(args.out_config), output_config)
        write_json(resolve(args.summary_json), summary)

    print(
        "class_repeat "
        f"base_rows={len(base_rows)} appended_rows={len(appended_rows)} "
        f"combined_rows={len(combined_rows)} duplicate_rows={len(combined_rows) - len(set(combined_rows))}"
    )
    for rule in rule_summaries:
        print(
            "rule "
            f"class={rule['class']} repeat={rule['repeat']} "
            f"selected={rule['selected_rows']} appended={rule['appended_rows']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
