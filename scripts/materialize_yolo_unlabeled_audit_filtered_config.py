#!/usr/bin/env python
"""Write a YOLO data YAML with audit-suspect images excluded from a split."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
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


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Source YOLO data YAML.")
    parser.add_argument("--split", default="train", help="Split to filter.")
    parser.add_argument("--audit", required=True, action="append", type=Path, help="Unlabeled prediction audit JSON.")
    parser.add_argument(
        "--min-unmatched-count",
        type=int,
        default=1,
        help="Exclude audit records with at least this many unmatched predictions.",
    )
    parser.add_argument("--out-config", required=True, type=Path, help="Output YOLO data YAML.")
    parser.add_argument("--list-out", required=True, type=Path, help="Output image list for the filtered split.")
    parser.add_argument(
        "--path-mode",
        choices=["preserve", "repo-root"],
        default="preserve",
        help="Keep the source YAML path or rewrite path so repo-relative list/val/test rows resolve from repo root.",
    )
    parser.add_argument("--val", default="", help="Optional val split override, usually data/cashsnap_v1/images/val.")
    parser.add_argument("--test", default="", help="Optional test split override, usually data/cashsnap_v1/images/test.")
    parser.add_argument("--fail-if-empty", action="store_true", help="Fail if filtering removes every image.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
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
        return set()
    classes: set[int] = set()
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            classes.add(int(line.split()[0]))
    return classes


def audit_exclusions(audit_paths: list[Path], min_unmatched_count: int) -> tuple[set[Path], list[dict[str, Any]]]:
    excluded: set[Path] = set()
    sources: list[dict[str, Any]] = []
    for audit_path in audit_paths:
        resolved = resolve(audit_path)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        records = payload.get("suspect_records", [])
        if not isinstance(records, list):
            raise SystemExit(f"Audit JSON has no suspect_records list: {repo_rel(resolved)}")
        before = len(excluded)
        for record in records:
            if int(record.get("unmatched_count", 0)) >= min_unmatched_count:
                excluded.add(resolve(record["image"]))
        sources.append(
            {
                "path": repo_rel(resolved),
                "schema": payload.get("schema"),
                "conf": payload.get("conf"),
                "match_iou": payload.get("match_iou"),
                "images": payload.get("images"),
                "suspect_records": len(records),
                "excluded_records": len(excluded) - before,
            }
        )
    return excluded, sources


def class_counts(images: list[Path]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for image in images:
        counts.update(image_classes(image))
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    if args.min_unmatched_count < 1:
        raise SystemExit("--min-unmatched-count must be >= 1")

    data_path = resolve(args.data)
    config = load_yaml(data_path)
    images = split_images(data_path, config, args.split)
    excluded, audit_sources = audit_exclusions(args.audit, args.min_unmatched_count)
    excluded_resolved = {path.resolve() for path in excluded}
    kept = [image for image in images if image.resolve() not in excluded_resolved]
    removed_images = [image for image in images if image.resolve() in excluded_resolved]
    removed = len(images) - len(kept)
    if args.fail_if_empty and not kept:
        raise SystemExit("Filtering removed every image")

    list_out = resolve(args.list_out)
    list_out.parent.mkdir(parents=True, exist_ok=True)
    list_out.write_text("".join(f"{repo_rel(image)}\n" for image in kept), encoding="utf-8")

    out_config = dict(config)
    out_config[args.split] = repo_rel(list_out)
    out_path = resolve(args.out_config)
    if args.path_mode == "repo-root":
        out_config["path"] = rel_between(out_path.parent, ROOT)
    if args.val:
        out_config["val"] = args.val
    if args.test:
        out_config["test"] = args.test
    policy = dict(out_config.get("cashsnap_policy") or {})
    policy["unlabeled_audit_filter"] = {
        "source_data": repo_rel(data_path),
        "split": args.split,
        "list": repo_rel(list_out),
        "path_mode": args.path_mode,
        "val_override": args.val,
        "test_override": args.test,
        "min_unmatched_count": args.min_unmatched_count,
        "input_images": len(images),
        "kept_images": len(kept),
        "removed_images": removed,
        "removed_class_image_counts": class_counts(removed_images),
        "kept_class_image_counts": class_counts(kept),
        "audits": audit_sources,
    }
    out_config["cashsnap_policy"] = policy

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out_config, sort_keys=False), encoding="utf-8")
    print(
        f"wrote={repo_rel(out_path)} list={repo_rel(list_out)} "
        f"input={len(images)} kept={len(kept)} removed={removed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
