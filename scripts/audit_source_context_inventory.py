#!/usr/bin/env python
"""Summarize real train source scenes available for source-preserving replacement."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_cashsnap_multi_instance_replacement import (
    CLASS_NAMES,
    iter_train_images,
    label_path_for_image,
    min_train_short_px,
    read_boxes,
    repo_rel,
    resolve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cashsnap-root", type=Path, default=Path("data/cashsnap_v1"))
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--min-source-boxes", type=int, default=1)
    parser.add_argument("--max-source-boxes", type=int, default=0)
    parser.add_argument("--min-source-box-short-at-imgsz", type=float, default=0.0)
    parser.add_argument("--source-name-require-regex", default="")
    parser.add_argument("--source-name-block-regex", default="")
    parser.add_argument("--json-out", required=True, type=Path)
    return parser.parse_args()


def name_allowed(image_path: Path, args: argparse.Namespace) -> bool:
    name = image_path.name
    if args.source_name_require_regex and not re.search(args.source_name_require_regex, name, flags=re.IGNORECASE):
        return False
    if args.source_name_block_regex and re.search(args.source_name_block_regex, name, flags=re.IGNORECASE):
        return False
    return True


def box_counts_by_class(boxes: list[Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for box in boxes:
        counts[box.class_name] += 1
    return counts


def main() -> int:
    args = parse_args()
    if args.min_source_boxes < 1:
        raise SystemExit("--min-source-boxes must be >= 1")
    if args.max_source_boxes < 0:
        raise SystemExit("--max-source-boxes must be >= 0")
    if args.min_source_box_short_at_imgsz < 0:
        raise SystemExit("--min-source-box-short-at-imgsz must be >= 0")

    cashsnap_root = resolve(args.cashsnap_root)
    total_images = 0
    labeled_images = 0
    selected_images = 0
    selected_boxes = 0
    reject_reasons: Counter[str] = Counter()
    selected_image_counts_by_class: Counter[str] = Counter()
    selected_box_counts_by_class: Counter[str] = Counter()
    selected_multiclass_image_counts: Counter[str] = Counter()
    examples_by_class: dict[str, list[str]] = defaultdict(list)
    short_side_by_class: dict[str, list[float]] = defaultdict(list)

    for image_path in iter_train_images(cashsnap_root):
        total_images += 1
        boxes = read_boxes(label_path_for_image(image_path, cashsnap_root))
        if boxes:
            labeled_images += 1

        reasons: list[str] = []
        if not name_allowed(image_path, args):
            reasons.append("name_filter")
        if len(boxes) < args.min_source_boxes:
            reasons.append("too_few_boxes")
        if args.max_source_boxes > 0 and len(boxes) > args.max_source_boxes:
            reasons.append("too_many_boxes")
        if (
            args.min_source_box_short_at_imgsz > 0
            and min_train_short_px(boxes, args.imgsz) < args.min_source_box_short_at_imgsz
        ):
            reasons.append("too_small_box")
        if reasons:
            reject_reasons.update(reasons)
            continue

        selected_images += 1
        selected_boxes += len(boxes)
        counts = box_counts_by_class(boxes)
        selected_multiclass_image_counts[str(len(counts))] += 1
        for class_name, count in counts.items():
            selected_image_counts_by_class[class_name] += 1
            selected_box_counts_by_class[class_name] += count
            if len(examples_by_class[class_name]) < 5:
                examples_by_class[class_name].append(repo_rel(image_path))
        for box in boxes:
            short_side_by_class[box.class_name].append(min(box.width, box.height) * args.imgsz)

    class_rows: dict[str, dict[str, Any]] = {}
    for class_name in CLASS_NAMES:
        shorts = sorted(short_side_by_class.get(class_name, []))
        p50 = shorts[len(shorts) // 2] if shorts else 0.0
        p10 = shorts[max(0, int(len(shorts) * 0.10) - 1)] if shorts else 0.0
        class_rows[class_name] = {
            "selected_images": selected_image_counts_by_class.get(class_name, 0),
            "selected_boxes": selected_box_counts_by_class.get(class_name, 0),
            "short_side_p10_at_imgsz": round(float(p10), 3),
            "short_side_p50_at_imgsz": round(float(p50), 3),
            "examples": examples_by_class.get(class_name, []),
        }

    missing_classes = [name for name, row in class_rows.items() if row["selected_boxes"] == 0]
    scarce_classes = [name for name, row in class_rows.items() if 0 < row["selected_boxes"] < 8]
    payload = {
        "schema": "cashsnap_source_context_inventory_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cashsnap_root": repo_rel(cashsnap_root),
        "args": {
            "imgsz": args.imgsz,
            "min_source_boxes": args.min_source_boxes,
            "max_source_boxes": args.max_source_boxes,
            "min_source_box_short_at_imgsz": args.min_source_box_short_at_imgsz,
            "source_name_require_regex": args.source_name_require_regex,
            "source_name_block_regex": args.source_name_block_regex,
        },
        "total_train_images": total_images,
        "labeled_train_images": labeled_images,
        "selected_images": selected_images,
        "selected_boxes": selected_boxes,
        "selected_multiclass_image_counts": dict(sorted(selected_multiclass_image_counts.items())),
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "missing_classes": missing_classes,
        "scarce_classes_lt8_boxes": scarce_classes,
        "classes": class_rows,
    }
    out = resolve(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "source_context_inventory "
        f"selected_images={selected_images} selected_boxes={selected_boxes} "
        f"missing={missing_classes} scarce_lt8={scarce_classes}"
    )
    print(f"wrote_json={repo_rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
