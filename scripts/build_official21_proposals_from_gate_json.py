#!/usr/bin/env python
"""Convert proposal-gate detections into official21 review proposal CSV rows.

This is a review-queue helper only. It selects detector boxes from
``probe_yolo_proposal_gate.py --proposal-json-out`` output and writes
materializer-compatible proposal columns with blank ``review_decision`` values.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "image",
    "model",
    "current_pred_class",
    "current_pred_class_id",
    "confidence",
    "area_ratio",
    "x1",
    "y1",
    "x2",
    "y2",
    "proposed_new_class",
    "review_decision",
    "review_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument(
        "--proposed-class",
        default="",
        help="Optional proposed official21 class. Leave blank for generic denomination triage.",
    )
    parser.add_argument(
        "--select",
        choices=["top_conf_per_image"],
        default="top_conf_per_image",
        help="Selection policy. Currently writes at most one proposal per image.",
    )
    parser.add_argument(
        "--sort",
        choices=["confidence_desc", "input_order"],
        default="confidence_desc",
    )
    parser.add_argument(
        "--note",
        default=(
            "Candidate from detector proposal-gate audit; proposed class is a "
            "review target only and review_decision is intentionally blank."
        ),
    )
    return parser.parse_args()


def class_name(names: dict[str, str] | dict[int, str], class_id: int) -> str:
    return str(names.get(str(class_id), names.get(class_id, class_id)))


def selected_predictions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = payload.get("names", {})
    detector = str(payload.get("detector", ""))

    for source_row in payload.get("rows", []):
        image = str(source_row.get("image", ""))
        predictions = source_row.get("predictions") or []
        if not image or not predictions:
            continue
        pred = max(predictions, key=lambda item: float(item.get("confidence", 0.0)))
        xyxy = pred.get("xyxy") or []
        if len(xyxy) != 4:
            continue
        class_id = int(pred["class_id"])
        gate_class = pred.get("gate_class", "")
        gate_conf = pred.get("gate_conf", "")
        rows.append(
            {
                "image": image,
                "model": detector,
                "current_pred_class": class_name(names, class_id),
                "current_pred_class_id": class_id,
                "confidence": float(pred.get("confidence", 0.0)),
                "area_ratio": float(pred.get("area_ratio", 0.0)),
                "x1": float(xyxy[0]),
                "y1": float(xyxy[1]),
                "x2": float(xyxy[2]),
                "y2": float(xyxy[3]),
                "gate_class": gate_class,
                "gate_conf": gate_conf,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    payload = json.loads(args.proposal_json.read_text(encoding="utf-8"))
    rows = selected_predictions(payload)
    if args.sort == "confidence_desc":
        rows.sort(key=lambda row: row["confidence"], reverse=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            note = (
                f"{args.note} gate={row.pop('gate_class', '')}:"
                f"{float(row.pop('gate_conf', 0.0) or 0.0):.3f}"
            )
            writer.writerow(
                {
                    **row,
                    "confidence": f"{row['confidence']:.9f}",
                    "area_ratio": f"{row['area_ratio']:.9f}",
                    "x1": f"{row['x1']:.6f}",
                    "y1": f"{row['y1']:.6f}",
                    "x2": f"{row['x2']:.6f}",
                    "y2": f"{row['y2']:.6f}",
                    "proposed_new_class": args.proposed_class,
                    "review_decision": "",
                    "review_notes": note,
                }
            )

    print(f"wrote {len(rows)} proposals to {args.out_csv}")


if __name__ == "__main__":
    main()
