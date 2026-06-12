#!/usr/bin/env python
"""Build a synth+real YOLO mix that protects high-value USD with real duplicates."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_MARKER = "/images/"


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--target-mix-config", required=True, type=Path)
    parser.add_argument(
        "--protected-class",
        action="append",
        default=None,
        help="Class to replace synthetic support for; defaults to USD_50 and USD_100 when omitted.",
    )
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-config", required=True, type=Path)
    parser.add_argument("--out-list", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(resolve(path).read_text(encoding="utf-8"))


def train_list_path(config: dict[str, Any]) -> Path:
    train = config.get("train")
    if not isinstance(train, str):
        raise SystemExit("config has no string train entry")
    return resolve(train)


def read_list(path: Path) -> list[str]:
    return [line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_path_for_image(image: str) -> Path:
    normalized = image.replace("\\", "/")
    if IMAGE_MARKER not in normalized:
        raise SystemExit(f"image path has no {IMAGE_MARKER!r}: {image}")
    label = normalized.replace(IMAGE_MARKER, "/labels/", 1)
    return resolve(Path(label).with_suffix(".txt"))


def image_classes(image: str) -> set[int]:
    label_path = label_path_for_image(image)
    if not label_path.exists():
        return set()
    classes: set[int] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            classes.add(int(float(parts[0])))
    return classes


def class_names(config: dict[str, Any]) -> dict[int, str]:
    raw_names = config.get("names")
    if not isinstance(raw_names, dict):
        raise SystemExit("config has no names mapping")
    return {int(key): str(value) for key, value in raw_names.items()}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    out_path = resolve(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    out_path = resolve(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    protected_class_names = args.protected_class or ["USD_50", "USD_100"]
    base_config = read_yaml(args.base_config)
    target_config = read_yaml(args.target_mix_config)
    names = class_names(target_config)
    name_to_id = {name: class_id for class_id, name in names.items()}
    protected_ids = {name_to_id[name] for name in protected_class_names if name in name_to_id}
    missing = sorted(set(protected_class_names) - set(name_to_id))
    if missing:
        raise SystemExit("unknown protected classes: " + ", ".join(missing))

    base_rows = read_list(train_list_path(base_config))
    target_rows = read_list(train_list_path(target_config))
    if target_rows[: len(base_rows)] != base_rows:
        raise SystemExit("target mix does not start with the base rows; refusing to infer extra rows")
    extra_rows = target_rows[len(base_rows) :]

    classes_by_image = {image: image_classes(image) for image in set(base_rows + extra_rows)}
    kept_extra_rows = [
        image for image in extra_rows if not (classes_by_image.get(image, set()) & protected_ids)
    ]
    excluded_extra_rows = [
        image for image in extra_rows if classes_by_image.get(image, set()) & protected_ids
    ]
    excluded_protected_counts: Counter[int] = Counter()
    for image in excluded_extra_rows:
        for class_id in classes_by_image.get(image, set()) & protected_ids:
            excluded_protected_counts[class_id] += 1

    rng = random.Random(args.seed)
    duplicate_rows: list[str] = []
    duplicate_protected_counts: Counter[int] = Counter()
    candidates_by_class: dict[int, list[str]] = {}
    for class_id in protected_ids:
        candidates = [image for image in base_rows if class_id in classes_by_image.get(image, set())]
        if not candidates:
            raise SystemExit(f"base train list has no rows for {names[class_id]}")
        rng.shuffle(candidates)
        candidates_by_class[class_id] = candidates

    for class_id in sorted(protected_ids):
        candidates = candidates_by_class[class_id]
        index = 0
        while duplicate_protected_counts[class_id] < excluded_protected_counts[class_id]:
            image = candidates[index % len(candidates)]
            duplicate_rows.append(image)
            for present_id in classes_by_image.get(image, set()) & protected_ids:
                duplicate_protected_counts[present_id] += 1
            index += 1

    while len(duplicate_rows) < len(excluded_extra_rows):
        for class_id in sorted(protected_ids):
            if len(duplicate_rows) >= len(excluded_extra_rows):
                break
            duplicate_rows.append(candidates_by_class[class_id][len(duplicate_rows) % len(candidates_by_class[class_id])])

    combined_rows = base_rows + kept_extra_rows + duplicate_rows
    out_list = resolve(args.out_list)
    out_list.parent.mkdir(parents=True, exist_ok=True)
    out_list.write_text("\n".join(combined_rows) + "\n", encoding="utf-8")

    output_config = dict(target_config)
    output_config["train"] = repo_rel(out_list)
    output_config["cashsnap_usd_highvalue_realdup_mix"] = {
        "schema": "cashsnap_usd_highvalue_realdup_mix_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "base_config": repo_rel(resolve(args.base_config)),
        "target_mix_config": repo_rel(resolve(args.target_mix_config)),
        "protected_classes": [names[class_id] for class_id in sorted(protected_ids)],
        "seed": args.seed,
        "base_rows": len(base_rows),
        "target_rows": len(target_rows),
        "target_extra_rows": len(extra_rows),
        "kept_extra_rows": len(kept_extra_rows),
        "excluded_extra_rows": len(excluded_extra_rows),
        "duplicate_rows": len(duplicate_rows),
        "combined_rows": len(combined_rows),
        "excluded_protected_counts": {
            names[class_id]: excluded_protected_counts[class_id] for class_id in sorted(protected_ids)
        },
        "duplicate_protected_counts": {
            names[class_id]: duplicate_protected_counts[class_id] for class_id in sorted(protected_ids)
        },
    }
    write_yaml(args.out_config, output_config)

    class_counts: Counter[int] = Counter()
    duplicate_class_counts: Counter[int] = Counter()
    for image in combined_rows:
        for class_id in classes_by_image.get(image, set()):
            class_counts[class_id] += 1
    for image in duplicate_rows:
        for class_id in classes_by_image.get(image, set()):
            duplicate_class_counts[class_id] += 1
    summary = {
        "schema": "cashsnap_usd_highvalue_realdup_mix_summary_v1",
        "config": repo_rel(resolve(args.out_config)),
        "train_list": repo_rel(out_list),
        "rows": len(combined_rows),
        "unique_rows": len(set(combined_rows)),
        "duplicate_rows": len(combined_rows) - len(set(combined_rows)),
        "base_rows": len(base_rows),
        "kept_extra_rows": len(kept_extra_rows),
        "excluded_extra_rows": len(excluded_extra_rows),
        "real_duplicate_rows": len(duplicate_rows),
        "class_image_counts": {names[class_id]: class_counts[class_id] for class_id in sorted(class_counts)},
        "real_duplicate_class_counts": {
            names[class_id]: duplicate_class_counts[class_id] for class_id in sorted(duplicate_class_counts)
        },
    }
    write_json(args.summary_json, summary)
    print(
        f"config={repo_rel(resolve(args.out_config))} rows={len(combined_rows)} "
        f"excluded_synth={len(excluded_extra_rows)} real_dupes={len(duplicate_rows)}"
    )


if __name__ == "__main__":
    main()
