#!/usr/bin/env python
"""Build a low-churn WebGL positive-support config."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_BASE = (
    ROOT
    / "configs"
    / "webgl_ablation"
    / "cashsnap_target_anchor_transplant_poisson_contact_geoscale205_minshort190_bal96_scaled_foreignhardneg6_probe_puresynth_realval_v1.yaml"
)
DEFAULT_CANDIDATE_ROOTS = [
    ROOT / "data" / "synthetic" / "cashsnap_webgl_clean_base_topdown_768x640_handled_probe_v2",
    ROOT / "data" / "synthetic" / "cashsnap_webgl_clean_base_square_classdiverse_postproc_geometry_selected96_v2",
    ROOT / "data" / "synthetic" / "cashsnap_webgl_clean_base_square_phoneauto_mixedcondition_postproc_pool_v1",
]
DEFAULT_TARGET_CLASSES = "KHR_500,KHR_2000,USD_5,USD_20,KHR_50000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--candidate-root", action="append", type=Path, default=[])
    parser.add_argument("--target-classes", default=DEFAULT_TARGET_CLASSES)
    parser.add_argument("--rows-per-class", type=int, default=2)
    parser.add_argument("--selection", choices=("spread", "random"), default="spread")
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--out-list", type=Path, required=True)
    parser.add_argument(
        "--intended-use",
        default="pure-synth TSTR probe adding a capped WebGL positive-support dose",
    )
    parser.add_argument(
        "--promotion-rule",
        default=(
            "reject unless fixed-step self-eval preserves the reference before "
            "any bounded real-transfer run"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", ",").replace(" ", ",").split(",") if item.strip()]


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(resolve(path))}: expected YAML mapping")
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
            rows.append(line.replace("\\", "/"))
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
            item_rows = read_image_list(path)
            rows.extend(item_rows)
            sources.append(repo_rel(path))
            continue
        image_root = split_root(root, item)
        if not image_root.exists():
            raise SystemExit(f"missing train source: {repo_rel(image_root)}")
        item_rows = [
            repo_rel(path)
            for path in sorted(image_root.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]
        rows.extend(item_rows)
        sources.append(repo_rel(image_root))
    return rows, sources


def write_image_list(path: Path, rows: list[str]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def names_by_id(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names", {})
    if not isinstance(names, dict):
        raise SystemExit("base config names must be a mapping")
    return {int(class_id): str(class_name) for class_id, class_name in names.items()}


def label_path_for_image(image: str) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def image_rows(root: Path) -> list[str]:
    image_dir = resolve(root) / "images" / "train"
    if not image_dir.exists():
        raise SystemExit(f"missing image dir: {repo_rel(image_dir)}")
    return [
        repo_rel(path)
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def single_label_class(image: str, names: dict[int, str]) -> str | None:
    label = resolve(label_path_for_image(image))
    if not label.exists():
        return None
    lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    parts = lines[0].split()
    if len(parts) != 5:
        return None
    return names.get(int(parts[0]), str(parts[0]))


def select_rows(rows: list[str], count: int, selection: str, rng: random.Random) -> list[str]:
    if count < 1:
        raise SystemExit("--rows-per-class must be at least 1")
    rows = sorted(dict.fromkeys(rows))
    if len(rows) <= count:
        return rows
    if selection == "random":
        return sorted(rng.sample(rows, count))
    if count == 1:
        return [rows[len(rows) // 2]]
    indexes = sorted({round(index * (len(rows) - 1) / (count - 1)) for index in range(count)})
    selected = [rows[index] for index in indexes]
    cursor = 0
    while len(selected) < count and cursor < len(rows):
        row = rows[cursor]
        if row not in selected:
            selected.append(row)
        cursor += 1
    return sorted(selected)


def main() -> int:
    args = parse_args()
    base_path = resolve(args.base)
    base_config = read_yaml(base_path)
    base_rows, base_sources = train_rows(base_path, base_config)
    base_set = set(base_rows)
    class_names = names_by_id(base_config)
    target_classes = parse_csv(args.target_classes)
    candidate_roots = [resolve(path) for path in (args.candidate_root or DEFAULT_CANDIDATE_ROOTS)]
    rng = random.Random(args.seed)

    candidates_by_class: dict[str, list[str]] = defaultdict(list)
    for root in candidate_roots:
        for image in image_rows(root):
            if image in base_set:
                continue
            class_name = single_label_class(image, class_names)
            if class_name in target_classes:
                candidates_by_class[class_name].append(image)

    support_by_class: dict[str, list[str]] = {}
    for class_name in target_classes:
        rows = select_rows(candidates_by_class[class_name], args.rows_per_class, args.selection, rng)
        if len(rows) < args.rows_per_class:
            raise SystemExit(
                f"not enough support rows for {class_name}: "
                f"{len(rows)} < {args.rows_per_class}"
            )
        support_by_class[class_name] = rows

    support_rows = [row for class_name in target_classes for row in support_by_class[class_name]]
    combined_rows = list(dict.fromkeys([*base_rows, *support_rows]))
    if len(combined_rows) != len(base_rows) + len(support_rows):
        raise SystemExit("support selection introduced duplicate train rows")

    report = {
        "schema": "cashsnap_webgl_targeted_support_v1",
        "base_config": repo_rel(base_path),
        "base_train_sources": base_sources,
        "candidate_roots": [repo_rel(path) for path in candidate_roots],
        "target_classes": target_classes,
        "rows_per_class": int(args.rows_per_class),
        "selection": str(args.selection),
        "seed": int(args.seed),
        "base_images": len(base_rows),
        "support_images": len(support_rows),
        "combined_images": len(combined_rows),
        "available_candidates": {
            class_name: len(candidates_by_class[class_name]) for class_name in target_classes
        },
        "support_image_rows": support_by_class,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0

    write_image_list(args.out_list, combined_rows)
    out_config = resolve(args.out_config)
    config = copy.deepcopy(base_config)
    config["path"] = rel_between(out_config.parent, ROOT)
    config["train"] = repo_rel(resolve(args.out_list))
    sources = copy.deepcopy(config.get("cashsnap_sources", {}))
    if not isinstance(sources, dict):
        sources = {}
    sources["targeted_support_base_config"] = repo_rel(base_path)
    sources["targeted_support_candidate_roots"] = [repo_rel(path) for path in candidate_roots]
    config["cashsnap_sources"] = sources
    config["cashsnap_webgl_targeted_support"] = report
    policy = copy.deepcopy(config.get("cashsnap_policy", {}))
    if not isinstance(policy, dict):
        policy = {}
    policy["intended_use"] = args.intended_use
    policy["promotion_rule"] = args.promotion_rule
    config["cashsnap_policy"] = policy
    write_yaml(out_config, config)
    print(json.dumps(report, indent=2))
    print(f"wrote_list={repo_rel(resolve(args.out_list))}")
    print(f"wrote_config={repo_rel(out_config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
