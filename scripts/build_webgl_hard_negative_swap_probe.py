#!/usr/bin/env python
"""Build paired accepted-blend configs for a hard-negative root swap probe."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    "USD_1",
    "USD_5",
    "USD_10",
    "USD_20",
    "USD_50",
    "USD_100",
    "KHR_500",
    "KHR_1000",
    "KHR_2000",
    "KHR_5000",
    "KHR_10000",
    "KHR_20000",
    "KHR_50000",
]
DEFAULT_BASE_CONFIG = ROOT / "configs" / "cashsnap_v1_plus_webgl_accepted_nowarmup_probe.yaml"
DEFAULT_OLD_ROOT = "data/synthetic/cashsnap_webgl_hard_negative_candidate_v1/images/train"
DEFAULT_NEW_ROOT = "data/synthetic/cashsnap_webgl_hard_negative_diversity_catalog_gate_v1/images/train"
DEFAULT_OLD_OUT = ROOT / "configs" / "cashsnap_v1_plus_webgl_accepted_hardneg_old8_probe.yaml"
DEFAULT_NEW_OUT = ROOT / "configs" / "cashsnap_v1_plus_webgl_accepted_hardneg_diversity8_probe.yaml"
DEFAULT_OLD_LIST = ROOT / "configs" / "generated_lists" / "cashsnap_v1_plus_webgl_accepted_hardneg_old8_probe_train.txt"
DEFAULT_NEW_LIST = ROOT / "configs" / "generated_lists" / "cashsnap_v1_plus_webgl_accepted_hardneg_diversity8_probe_train.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--old-root", default=DEFAULT_OLD_ROOT)
    parser.add_argument("--new-root", default=DEFAULT_NEW_ROOT)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--old-out", type=Path, default=DEFAULT_OLD_OUT)
    parser.add_argument("--new-out", type=Path, default=DEFAULT_NEW_OUT)
    parser.add_argument("--old-train-list", type=Path, default=DEFAULT_OLD_LIST)
    parser.add_argument("--new-train-list", type=Path, default=DEFAULT_NEW_LIST)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def normalize_prefix(value: str) -> str:
    return value.replace("\\", "/").strip("/")


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return data


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    root = Path(str(config.get("path", ".")))
    return root if root.is_absolute() else (config_path.parent / root).resolve()


def split_path(config_path: Path, config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root(config_path, config) / path


def read_train_rows(config_path: Path, config: dict[str, Any]) -> list[str]:
    train = config.get("train")
    if not isinstance(train, str):
        raise ValueError(f"{config_path}: train must be a train-list path")
    train_path = split_path(config_path, config, train)
    return [line.strip().replace("\\", "/") for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_rows(root_prefix: str, count: int) -> list[str]:
    root = ROOT / normalize_prefix(root_prefix)
    if not root.exists():
        raise FileNotFoundError(f"missing image root: {root}")
    rows = [repo_rel(path) for path in sorted(root.glob("*")) if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if len(rows) < count:
        raise ValueError(f"{root_prefix}: expected at least {count} image(s), got {len(rows)}")
    return rows[:count]


def label_path_for_row(row: str) -> Path:
    path = ROOT / row
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def classes_for_row(row: str) -> set[int]:
    label = label_path_for_row(row)
    if not label.exists():
        raise FileNotFoundError(f"missing label for {row}: {label}")
    classes: set[int] = set()
    for line_number, raw_line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label}:{line_number}: expected 5 YOLO fields, got {len(parts)}")
        classes.add(int(parts[0]))
    return classes


def row_summary(rows: list[str]) -> dict[str, Any]:
    class_rows: Counter[int] = Counter()
    backgrounds = 0
    for row in rows:
        classes = classes_for_row(row)
        if not classes:
            backgrounds += 1
        for class_id in classes:
            if 0 <= class_id < len(CLASS_NAMES):
                class_rows[class_id] += 1
    return {
        "rows": len(rows),
        "unique_rows": len(set(rows)),
        "backgrounds": backgrounds,
        "class_image_rows": {CLASS_NAMES[index]: class_rows[index] for index in sorted(class_rows)},
    }


def replace_hard_negative_rows(base_rows: list[str], old_root: str, replacement_rows: list[str]) -> tuple[list[str], int, int]:
    old_prefix = normalize_prefix(old_root) + "/"
    old_indices = [index for index, row in enumerate(base_rows) if row.startswith(old_prefix)]
    if not old_indices:
        raise ValueError(f"base train list has no rows under {old_root}")
    insert_at = old_indices[0]
    without_old = [row for row in base_rows if not row.startswith(old_prefix)]
    adjusted_insert_at = sum(1 for index, row in enumerate(base_rows) if index < insert_at and not row.startswith(old_prefix))
    return (
        without_old[:adjusted_insert_at] + replacement_rows + without_old[adjusted_insert_at:],
        len(old_indices),
        insert_at,
    )


def write_train_list(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_config(
    *,
    base_config_path: Path,
    base_config: dict[str, Any],
    out_path: Path,
    train_list_path: Path,
    rows: list[str],
    mode: str,
    old_root: str,
    replacement_root: str,
    removed_old_rows: int,
    inserted_rows: int,
    insertion_index: int,
) -> None:
    config = copy.deepcopy(base_config)
    config.pop("cashsnap_domain_gap_gate", None)
    config.pop("cashsnap_webgl_blend_gate", None)
    config["train"] = repo_rel(train_list_path)
    sources = dict(config.get("cashsnap_sources", {}))
    prefixes = [
        prefix
        for prefix in sources.get("always_include_prefixes", [])
        if normalize_prefix(str(prefix)).rstrip("/") != normalize_prefix(old_root).removesuffix("/images/train")
    ]
    replacement_dataset_root = normalize_prefix(replacement_root).removesuffix("/images/train")
    prefixes.append(f"{replacement_dataset_root}/")
    sources["always_include_prefixes"] = prefixes
    config["cashsnap_sources"] = sources
    acceptance = dict(config.get("cashsnap_webgl_acceptance", {}))
    if acceptance:
        acceptance["selected_prefixes"] = prefixes
        acceptance["hard_negative_swap_probe"] = mode
        config["cashsnap_webgl_acceptance"] = acceptance
    subset_policy = dict(config.get("cashsnap_subset_policy", {}))
    summary = row_summary(rows)
    subset_policy["selected_images"] = summary["rows"]
    subset_policy["selected_unique_images"] = summary["unique_rows"]
    subset_policy["selected_backgrounds"] = summary["backgrounds"]
    subset_policy["selected_class_images"] = summary["class_image_rows"]
    config["cashsnap_subset_policy"] = subset_policy
    config["cashsnap_hard_negative_swap_probe"] = {
        "base_config": repo_rel(base_config_path),
        "base_train_list": str(base_config.get("train", "")),
        "mode": mode,
        "old_root": old_root,
        "replacement_root": replacement_root,
        "removed_old_rows": removed_old_rows,
        "inserted_rows": inserted_rows,
        "insertion_index": insertion_index,
        "row_summary": summary,
        "purpose": "isolate hard-negative prop diversity from row-count effects before fixed-step model probing",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    base_config_path = resolve(args.base_config)
    old_out = resolve(args.old_out)
    new_out = resolve(args.new_out)
    old_train_list = resolve(args.old_train_list)
    new_train_list = resolve(args.new_train_list)

    base_config = read_yaml(base_config_path)
    base_rows = read_train_rows(base_config_path, base_config)
    old_rows = image_rows(args.old_root, args.count)
    new_rows = image_rows(args.new_root, args.count)
    old_probe_rows, removed_old_rows, insertion_index = replace_hard_negative_rows(base_rows, args.old_root, old_rows)
    new_probe_rows, removed_new_old_rows, new_insertion_index = replace_hard_negative_rows(base_rows, args.old_root, new_rows)
    if len(old_probe_rows) != len(new_probe_rows):
        raise SystemExit("old/new hard-negative probe rows must match")
    if removed_old_rows != removed_new_old_rows or insertion_index != new_insertion_index:
        raise SystemExit("old/new hard-negative replacement metadata mismatch")

    write_train_list(old_train_list, old_probe_rows)
    write_train_list(new_train_list, new_probe_rows)
    write_config(
        base_config_path=base_config_path,
        base_config=base_config,
        out_path=old_out,
        train_list_path=old_train_list,
        rows=old_probe_rows,
        mode="old_hard_negative_first_n",
        old_root=args.old_root,
        replacement_root=args.old_root,
        removed_old_rows=removed_old_rows,
        inserted_rows=len(old_rows),
        insertion_index=insertion_index,
    )
    write_config(
        base_config_path=base_config_path,
        base_config=base_config,
        out_path=new_out,
        train_list_path=new_train_list,
        rows=new_probe_rows,
        mode="prop_diverse_hard_negative_swap",
        old_root=args.old_root,
        replacement_root=args.new_root,
        removed_old_rows=removed_old_rows,
        inserted_rows=len(new_rows),
        insertion_index=insertion_index,
    )
    print(
        "wrote hard-negative swap probes "
        f"rows={len(old_probe_rows)} removed_old={removed_old_rows} inserted={args.count} "
        f"old={repo_rel(old_out)} new={repo_rel(new_out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
