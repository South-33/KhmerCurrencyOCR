from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_real_draft_labels import greedy_match, read_yolo_detect_labels
from evaluate_two_stage_csv import CLASS_TO_ID
from fuse_two_stage_csv import (
    infer_supported_classes,
    nms,
    parse_classes,
    prefer_supported_duplicate_classes,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep detector/classifier fusion thresholds against draft labels.")
    parser.add_argument("--csv", required=True, help="Raw two-stage proposal CSV.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--det-thresholds", default="0.10,0.12,0.15,0.16,0.17,0.175,0.18,0.20,0.25")
    parser.add_argument("--nms-ious", default="0.75,0.80,0.85,0.90")
    parser.add_argument("--nms-score-column", default="detector_conf")
    parser.add_argument("--det-class-column", default="detector_class")
    parser.add_argument("--det-score-column", default="detector_conf")
    parser.add_argument("--out-class-column", default="fragment_class")
    parser.add_argument("--out-score-column", default="fragment_conf")
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--prefer-supported-detector-duplicates", action="store_true")
    parser.add_argument("--duplicate-iou", type=float, default=0.95)
    parser.add_argument("--supported-detector-classes", default="")
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in re.split(r"[,\s]+", text) if part.strip()]


def fused_rows(
    rows: list[dict[str, str]],
    det_threshold: float,
    nms_iou: float,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        detector_score = float(item.get(args.det_score_column, "0") or 0)
        if detector_score >= det_threshold:
            item[args.out_class_column] = item[args.det_class_column]
            item[args.out_score_column] = item[args.det_score_column]
        output.append(item)
    return nms(output, args.nms_score_column or args.out_score_column, nms_iou)


def evaluate(
    rows: list[dict[str, str]],
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, str]:
    pred_boxes: list[list[float]] = []
    pred_classes: list[int] = []
    pred_scores: list[float] = []
    counts: dict[str, int] = {}
    for row in rows:
        class_name = row.get(args.out_class_column, "")
        if class_name not in CLASS_TO_ID:
            continue
        pred_boxes.append([float(row[key]) for key in ["x1", "y1", "x2", "y2"]])
        pred_classes.append(CLASS_TO_ID[class_name])
        pred_scores.append(float(row.get(args.out_score_column, "0") or 0))
        counts[class_name] = counts.get(class_name, 0) + 1
    boxes = np.asarray(pred_boxes, dtype=float)
    classes = np.asarray(pred_classes, dtype=int)
    scores = np.asarray(pred_scores, dtype=float)
    same = greedy_match(boxes, classes, scores, gt_boxes, gt_classes, args.match_iou, require_same_class=True)
    any_class = greedy_match(boxes, classes, scores, gt_boxes, gt_classes, args.match_iou, require_same_class=False)
    gt_count = len(gt_boxes)
    pred_count = len(boxes)
    return {
        "pred": str(pred_count),
        "count_error": str(pred_count - gt_count),
        "matched_same_class": str(same),
        "recall_same_class": f"{same / max(1, gt_count):.3f}",
        "matched_any_class": str(any_class),
        "recall_any_class": f"{any_class / max(1, gt_count):.3f}",
        "pred_classes": ";".join(f"{key}:{value}" for key, value in sorted(counts.items())),
    }


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(resolve(args.csv).open("r", newline="", encoding="utf-8")))
    duplicate_relabels = 0
    if args.prefer_supported_detector_duplicates:
        rows = [dict(row) for row in rows]
        supported_classes = parse_classes(args.supported_detector_classes) or infer_supported_classes(rows)
        duplicate_relabels = prefer_supported_duplicate_classes(
            rows,
            args.det_class_column,
            args.det_score_column,
            supported_classes,
            args.duplicate_iou,
        )
    with Image.open(resolve(args.image)) as image:
        gt_boxes, gt_classes = read_yolo_detect_labels(resolve(args.labels), image.size)
    results: list[dict[str, str]] = []
    for det_threshold in parse_float_list(args.det_thresholds):
        for nms_iou in parse_float_list(args.nms_ious):
            kept = fused_rows(rows, det_threshold, nms_iou, args)
            result = {
                "det_threshold": f"{det_threshold:g}",
                "nms_iou": f"{nms_iou:g}",
                "gt": str(len(gt_boxes)),
            }
            result.update(evaluate(kept, gt_boxes, gt_classes, args))
            results.append(result)

    fieldnames = list(results[0].keys())
    if args.out:
        out_path = resolve(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(
            f"wrote {len(results)} sweep rows to {out_path.relative_to(ROOT)}; "
            f"duplicate_relabels={duplicate_relabels}"
        )
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)


if __name__ == "__main__":
    main()
