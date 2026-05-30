#!/usr/bin/env python
"""Compare physical-note counts with visible-fragment counts in a WebGL package."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Packaged WebGL dataset root.")
    parser.add_argument("--per-image", action="store_true", help="Print per-image count rows.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> object:
    if not path.exists():
        raise SystemExit(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def class_name(class_index: int) -> str:
    if 0 <= class_index < len(CLASS_NAMES):
        return CLASS_NAMES[class_index]
    return f"class_{class_index}"


def counter_text(counter: Counter[str]) -> str:
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter)) or "none"


def main() -> int:
    args = parse_args()
    dataset_root = resolve_path(args.root)
    manifest = read_json(dataset_root / "manifest.json")
    if not isinstance(manifest, list) or not manifest:
        raise SystemExit("manifest.json must be a non-empty list")

    total_physical: Counter[str] = Counter()
    total_fragments: Counter[str] = Counter()
    total_parent_fused: Counter[str] = Counter()
    split_parent_count = 0

    for row in manifest:
        boxes_doc = read_json(dataset_root / row["visible_boxes"])
        visible_boxes = boxes_doc.get("boxes", [])
        fragment_metadata = read_json(dataset_root / row["fragment_metadata"])

        physical = Counter(str(box["className"]) for box in visible_boxes)
        fragments = Counter(str(fragment["className"]) for fragment in fragment_metadata)
        parent_keys = {
            (int(fragment["parentVisibleIndex"]), str(fragment["className"]))
            for fragment in fragment_metadata
        }
        parent_fused = Counter(class_name for _parent_index, class_name in parent_keys)
        parent_fragment_counts: Counter[tuple[int, str]] = Counter(
            (int(fragment["parentVisibleIndex"]), str(fragment["className"]))
            for fragment in fragment_metadata
        )
        split_parent_count += sum(1 for count in parent_fragment_counts.values() if count > 1)

        total_physical.update(physical)
        total_fragments.update(fragments)
        total_parent_fused.update(parent_fused)

        if args.per_image:
            print(
                f"{row['image']}: physical={sum(physical.values())} "
                f"fragments={sum(fragments.values())} parent_fused={sum(parent_fused.values())} "
                f"physical_by_class=({counter_text(physical)}) "
                f"fragments_by_class=({counter_text(fragments)})"
            )

    physical_total = sum(total_physical.values())
    fragment_total = sum(total_fragments.values())
    parent_fused_total = sum(total_parent_fused.values())
    print(f"images: {len(manifest)}")
    print(f"physical visible instances: {physical_total} ({counter_text(total_physical)})")
    print(f"visible fragments: {fragment_total} ({counter_text(total_fragments)})")
    print(f"parent-fused fragments: {parent_fused_total} ({counter_text(total_parent_fused)})")
    print(f"naive fragment overcount: {fragment_total - physical_total}")
    print(f"parents split into multiple fragments: {split_parent_count}")
    if total_parent_fused != total_physical:
        raise SystemExit("parent-fused counts do not match physical counts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
