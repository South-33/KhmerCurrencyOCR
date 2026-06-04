#!/usr/bin/env python
"""Gate WebGL count-stress properties such as same-class repeats."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Packaged WebGL dataset root.")
    parser.add_argument("--min-images", type=int, default=1, help="Minimum packaged image count.")
    parser.add_argument(
        "--min-repeat-images",
        type=int,
        default=0,
        help="Minimum images where at least one class appears two or more times.",
    )
    parser.add_argument(
        "--min-max-same-class",
        type=int,
        default=1,
        help="Minimum maximum same-class physical count in any single image.",
    )
    parser.add_argument(
        "--min-kept-split-parent-count",
        type=int,
        default=0,
        help="Minimum parents split into multiple kept fragments across the package.",
    )
    parser.add_argument(
        "--min-all-split-parent-count",
        type=int,
        default=0,
        help="Minimum parents split into multiple kept+ignored fragments across the package.",
    )
    parser.add_argument(
        "--min-naive-kept-fragment-overcount",
        type=int,
        default=0,
        help="Minimum kept-fragment overcount versus physical visible instances.",
    )
    parser.add_argument(
        "--min-naive-all-fragment-overcount",
        type=int,
        default=0,
        help="Minimum kept+ignored fragment overcount versus physical visible instances.",
    )
    parser.add_argument(
        "--require-parent-fusion-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require parent_fused_all_fragments to match physical_visible_instances.",
    )
    parser.add_argument("--per-image", action="store_true", help="Print per-image repeat summaries.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing JSON file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{line_number}: expected JSON object")
        rows.append(row)
    return rows


def int_at(document: dict[str, Any], *keys: str, default: int = 0) -> int:
    value: Any = document
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def physical_counts(row: dict[str, Any]) -> dict[str, int]:
    block = row.get("physical_visible_instances", {})
    by_class = block.get("by_class", {}) if isinstance(block, dict) else {}
    if not isinstance(by_class, dict):
        raise SystemExit(f"variant {row.get('variant', '?')}: physical_visible_instances.by_class must be an object")
    counts: dict[str, int] = {}
    for class_name, raw_count in by_class.items():
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"variant {row.get('variant', '?')}: invalid count for {class_name!r}") from exc
        if count < 0:
            raise SystemExit(f"variant {row.get('variant', '?')}: negative count for {class_name!r}")
        counts[str(class_name)] = count
    return counts


def fail_if_errors(errors: list[str]) -> None:
    if errors:
        detail = "\n  - ".join(errors)
        raise SystemExit(f"count stress audit failed:\n  - {detail}")


def main() -> int:
    args = parse_args()
    dataset_root = resolve_path(args.root)
    summary_path = dataset_root / "counts" / "summary.json"
    targets_path = dataset_root / "counts" / "targets.jsonl"
    summary = read_json(summary_path)
    rows = read_jsonl(targets_path)

    repeat_images = 0
    max_same_class = 0
    max_same_hist: Counter[int] = Counter()
    repeat_examples: list[str] = []

    for row in rows:
        counts = physical_counts(row)
        max_for_image = max(counts.values()) if counts else 0
        max_same_class = max(max_same_class, max_for_image)
        max_same_hist[max_for_image] += 1
        repeated = {class_name: count for class_name, count in counts.items() if count >= 2}
        if repeated:
            repeat_images += 1
            if len(repeat_examples) < 8:
                repeated_text = ",".join(f"{name}:{count}" for name, count in sorted(repeated.items()))
                repeat_examples.append(f"{row.get('variant', '?')}({repeated_text})")
        if args.per_image:
            print(
                f"variant={row.get('variant', '?')} total={sum(counts.values())} "
                f"max_same_class={max_for_image} repeats={repeated or {}}"
            )

    images = int_at(summary, "images")
    errors: list[str] = []
    if images != len(rows):
        errors.append(f"summary images={images} does not match targets rows={len(rows)}")
    if images < args.min_images:
        errors.append(f"expected at least {args.min_images} images, got {images}")
    if args.require_parent_fusion_match and not summary.get("parent_fused_all_matches_physical", False):
        errors.append("parent_fused_all_matches_physical must be true")
    if repeat_images < args.min_repeat_images:
        errors.append(f"expected at least {args.min_repeat_images} same-class repeat images, got {repeat_images}")
    if max_same_class < args.min_max_same_class:
        errors.append(f"expected max same-class count >= {args.min_max_same_class}, got {max_same_class}")

    kept_split_parent_count = int_at(summary, "kept_split_parent_count")
    all_split_parent_count = int_at(summary, "all_split_parent_count")
    naive_kept_overcount = int_at(summary, "naive_kept_fragment_overcount")
    naive_all_overcount = int_at(summary, "naive_all_fragment_overcount")
    if kept_split_parent_count < args.min_kept_split_parent_count:
        errors.append(
            f"expected kept_split_parent_count >= {args.min_kept_split_parent_count}, got {kept_split_parent_count}"
        )
    if all_split_parent_count < args.min_all_split_parent_count:
        errors.append(f"expected all_split_parent_count >= {args.min_all_split_parent_count}, got {all_split_parent_count}")
    if naive_kept_overcount < args.min_naive_kept_fragment_overcount:
        errors.append(
            f"expected naive_kept_fragment_overcount >= {args.min_naive_kept_fragment_overcount}, got {naive_kept_overcount}"
        )
    if naive_all_overcount < args.min_naive_all_fragment_overcount:
        errors.append(
            f"expected naive_all_fragment_overcount >= {args.min_naive_all_fragment_overcount}, got {naive_all_overcount}"
        )

    fail_if_errors(errors)

    hist_text = ",".join(f"{key}:{max_same_hist[key]}" for key in sorted(max_same_hist))
    examples_text = ";".join(repeat_examples) if repeat_examples else "none"
    print(
        "ok: "
        f"{display_path(dataset_root)} images={images}, "
        f"same_class_repeat_images={repeat_images}, max_same_class_per_image={max_same_class}, "
        f"max_same_hist={hist_text}, kept_split_parent_count={kept_split_parent_count}, "
        f"all_split_parent_count={all_split_parent_count}, "
        f"naive_kept_fragment_overcount={naive_kept_overcount}, "
        f"naive_all_fragment_overcount={naive_all_overcount}, repeat_examples={examples_text}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
