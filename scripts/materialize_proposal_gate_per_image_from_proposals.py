#!/usr/bin/env python
"""Materialize per-image proposal-gate rows from saved proposal+gate scores."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sweep_proposal_gate_thresholds import (
    box_iou,
    match_predictions,
    parse_class_thresholds,
    reject_threshold_for_prediction,
    repo_rel,
    resolve,
    summarize,
    value_totals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-json", required=True, type=Path)
    parser.add_argument("--reject-min-conf", required=True, type=float)
    parser.add_argument(
        "--class-threshold",
        action="append",
        default=[],
        help="Optional detector-class-specific reject threshold as CLASS=THRESHOLD. Repeat as needed.",
    )
    parser.add_argument(
        "--class-threshold-min-det-conf",
        type=float,
        default=None,
        help="Only apply --class-threshold overrides when detector confidence is at least this value.",
    )
    parser.add_argument("--stage-name", default="post_gate")
    parser.add_argument(
        "--final-nms-iou",
        type=float,
        default=None,
        help=(
            "Optional browser-style final class-agnostic NMS IoU applied after "
            "gate rejection. Omit to preserve raw post-gate rows."
        ),
    )
    parser.add_argument("--json-out", required=True, type=Path)
    return parser.parse_args()


def class_agnostic_nms(predictions: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for prediction in sorted(predictions, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        if all(box_iou(prediction["xyxy"], kept_prediction["xyxy"]) < iou_threshold for kept_prediction in kept):
            kept.append(prediction)
    return kept


def per_image_row(
    payload: dict[str, Any],
    source_row: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    names = {int(key): str(value) for key, value in payload["names"].items()}
    labels = source_row["labels"]
    matched_label_indices, false_predictions = match_predictions(
        labels,
        predictions,
        float(payload["match_iou"]),
        match_classes=bool(payload.get("detector_match_classes", True)),
    )
    count_error = len(predictions) - len(labels)
    gt_usd = gt_khr = pred_usd = pred_khr = usd_error = khr_error = exact_value = None
    if bool(payload.get("detector_match_classes", True)):
        gt_usd, gt_khr = value_totals(labels, names)
        pred_usd, pred_khr = value_totals(predictions, names)
        usd_error = pred_usd - gt_usd
        khr_error = pred_khr - gt_khr
        exact_value = usd_error == 0 and khr_error == 0
    return {
        "image": source_row["image"],
        "source_group": source_row["source_group"],
        "labels": len(labels),
        "predictions": len(predictions),
        "tp": len(matched_label_indices),
        "fp": len(false_predictions),
        "fn": len(labels) - len(matched_label_indices),
        "count_error": count_error,
        "exact_count": count_error == 0,
        "gt_usd": gt_usd,
        "pred_usd": pred_usd,
        "usd_error": usd_error,
        "gt_khr": gt_khr,
        "pred_khr": pred_khr,
        "khr_error": khr_error,
        "exact_value": exact_value,
    }


def summary_from_rows(payload: dict[str, Any], rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    gt = sum(int(row["labels"]) for row in rows)
    predictions = sum(int(row["predictions"]) for row in rows)
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    background_images = sum(1 for row in rows if int(row["labels"]) == 0)
    background_images_with_fp = sum(1 for row in rows if int(row["labels"]) == 0 and int(row["fp"]) > 0)
    exact_value_images = sum(1 for row in rows if bool(row.get("exact_value")))
    return {
        "threshold": threshold,
        "images": len(rows),
        "background_images": background_images,
        "background_images_with_fp": background_images_with_fp,
        "gt": gt,
        "total_predictions": predictions,
        "tp": tp,
        "fp": fp,
        "fn": gt - tp,
        "recall": tp / gt if gt else 0.0,
        "precision": tp / predictions if predictions else 0.0,
        "count_value_errors": {
            "exact_value_images": exact_value_images if bool(payload.get("detector_match_classes", True)) else None,
            "exact_count_images": sum(1 for row in rows if bool(row.get("exact_count"))),
            "count_abs_error_sum": sum(abs(float(row["count_error"])) for row in rows),
            "usd_abs_error_sum": sum(
                abs(float(row["usd_error"])) for row in rows if row.get("usd_error") is not None
            ),
            "khr_abs_error_sum": sum(
                abs(float(row["khr_error"])) for row in rows if row.get("khr_error") is not None
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.final_nms_iou is not None and args.final_nms_iou <= 0:
        raise SystemExit("--final-nms-iou must be > 0 when set")
    proposal_path = resolve(args.proposal_json)
    payload = json.loads(proposal_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "cashsnap_yolo_proposal_gate_proposals_v1":
        raise SystemExit(f"{repo_rel(proposal_path)} is not proposal schema v1")
    reject_class = str(payload.get("reject_class", "background"))
    names = {int(key): str(value) for key, value in payload["names"].items()}
    class_thresholds = parse_class_thresholds(args.class_threshold)
    pre_rows = []
    post_rows = []
    for source_row in payload["rows"]:
        predictions = list(source_row["predictions"])
        gated_predictions = []
        for prediction in predictions:
            reject_threshold = reject_threshold_for_prediction(
                prediction,
                names,
                base_threshold=args.reject_min_conf,
                class_thresholds=class_thresholds,
                class_threshold_min_det_conf=args.class_threshold_min_det_conf,
            )
            rejected = (
                str(prediction.get("gate_class")) == reject_class
                and float(prediction.get("gate_conf", 0.0)) >= reject_threshold
            )
            if not rejected:
                gated_predictions.append(prediction)
        if args.final_nms_iou is not None:
            gated_predictions = class_agnostic_nms(gated_predictions, args.final_nms_iou)
        pre_rows.append(per_image_row(payload, source_row, predictions))
        post_rows.append(per_image_row(payload, source_row, gated_predictions))

    if args.final_nms_iou is None:
        summary_metrics = summarize(
            payload,
            threshold=args.reject_min_conf,
            bgfp_costs=[1.0, 2.0, 5.0],
            class_thresholds=class_thresholds,
            class_threshold_min_det_conf=args.class_threshold_min_det_conf,
        )
    else:
        summary_metrics = summary_from_rows(payload, post_rows, threshold=args.reject_min_conf)
    out_path = resolve(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema": "cashsnap_yolo_proposal_gate_per_image_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "summary": repo_rel(proposal_path),
        "detector": payload["detector"],
        "gate": payload["gate"],
        "reclassifier": None,
        "data": payload["data"],
        "split": payload["split"],
        "images": payload["images"],
        "imgsz": payload["imgsz"],
        "conf": payload["conf"],
        "det_iou": payload["det_iou"],
        "match_iou": payload["match_iou"],
        "reject_class": payload["reject_class"],
        "reject_min_conf": args.reject_min_conf,
        "class_thresholds": class_thresholds,
        "class_threshold_min_det_conf": args.class_threshold_min_det_conf,
        "final_nms_iou": args.final_nms_iou,
        "summary_metrics": summary_metrics,
        "stages": {
            "pre_gate": pre_rows,
            args.stage_name: post_rows,
        },
    }
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"per_image={repo_rel(out_path)} threshold={args.reject_min_conf:g} "
        f"exact={summary_metrics['count_value_errors']['exact_value_images']} "
        f"bg_fp={summary_metrics['background_images_with_fp']}"
    )


if __name__ == "__main__":
    main()
