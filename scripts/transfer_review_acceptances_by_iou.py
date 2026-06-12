#!/usr/bin/env python
"""Transfer prior accepted review decisions into a refreshed proposal queue.

Rows are matched by image path and accepted only when the target box overlaps the
prior accepted box above ``--min-iou``. This is meant for regenerating review
queues from newer detector/gate proposals without silently accepting changed
boxes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BOX_FIELDS = ("x1", "y1", "x2", "y2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-csv", required=True, type=Path)
    parser.add_argument("--accepted-csv", required=True, action="append", type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--min-iou", type=float, default=0.95)
    parser.add_argument("--accepted-decision", default="accepted_box")
    return parser.parse_args()


def normalized(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def parse_box(row: dict[str, str]) -> tuple[float, float, float, float]:
    return tuple(float(row[field]) for field in BOX_FIELDS)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def require_fields(path: Path, fieldnames: list[str], required: set[str]) -> None:
    missing = sorted(required - set(fieldnames))
    if missing:
        raise SystemExit(f"{path} missing required fields: {missing}")


def load_acceptances(paths: list[Path], accepted_decision: str) -> dict[str, list[dict[str, Any]]]:
    accepted_norm = normalized(accepted_decision)
    by_image: dict[str, list[dict[str, Any]]] = {}
    required = {"image", "review_decision", "proposed_new_class", "review_id", "review_notes", *BOX_FIELDS}
    for path in paths:
        rows, fieldnames = read_csv(path)
        require_fields(path, fieldnames, required)
        for row in rows:
            if normalized(row.get("review_decision", "")) != accepted_norm:
                continue
            image = row.get("image", "").strip()
            if not image:
                continue
            by_image.setdefault(image, []).append(
                {
                    "row": row,
                    "box": parse_box(row),
                    "source_csv": path.as_posix(),
                }
            )
    return by_image


def main() -> None:
    args = parse_args()
    target_rows, target_fields = read_csv(args.target_csv)
    require_fields(
        args.target_csv,
        target_fields,
        {"image", "review_decision", "proposed_new_class", "review_id", "review_notes", *BOX_FIELDS},
    )
    acceptances = load_acceptances(args.accepted_csv, args.accepted_decision)
    accepted_norm = normalized(args.accepted_decision)

    transfers: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in target_rows:
        image = row.get("image", "").strip()
        candidates = acceptances.get(image, [])
        if not candidates:
            continue
        target_box = parse_box(row)
        best = max(candidates, key=lambda item: box_iou(target_box, item["box"]))
        score = box_iou(target_box, best["box"])
        accepted_row = best["row"]
        accepted_class = accepted_row.get("proposed_new_class", "").strip()
        target_class = row.get("proposed_new_class", "").strip()
        transfer = {
            "image": image,
            "target_review_id": row.get("review_id", ""),
            "accepted_review_id": accepted_row.get("review_id", ""),
            "accepted_class": accepted_class,
            "iou": score,
            "source_csv": best["source_csv"],
        }
        if score < args.min_iou:
            skipped.append({**transfer, "reason": "iou_below_min"})
            continue
        if target_class and target_class != accepted_class:
            skipped.append({**transfer, "reason": "target_class_conflict", "target_class": target_class})
            continue
        row["proposed_new_class"] = accepted_class
        row["review_decision"] = accepted_norm
        prior_notes = accepted_row.get("review_notes", "").strip()
        row["review_notes"] = (
            f"{prior_notes} Prior visual acceptance transferred from "
            f"{accepted_row.get('review_id', 'prior')}; box IoU={score:.4f}."
        ).strip()
        transfers.append({**transfer, "transferred": True})

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=target_fields)
        writer.writeheader()
        writer.writerows(target_rows)

    summary = {
        "schema": "cashsnap_review_acceptance_iou_transfer_v1",
        "target_csv": args.target_csv.as_posix(),
        "accepted_csvs": [path.as_posix() for path in args.accepted_csv],
        "out_csv": args.out_csv.as_posix(),
        "rows": len(target_rows),
        "accepted_source_images": len(acceptances),
        "transferred_acceptances": len(transfers),
        "min_iou": args.min_iou,
        "min_transfer_iou": min((item["iou"] for item in transfers), default=None),
        "skipped": skipped,
        "transfers": transfers,
    }
    summary_path = args.summary_json or args.out_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(transfers)} transferred acceptances to {args.out_csv}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
