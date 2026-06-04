from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_real_draft_labels import box_iou, greedy_match, read_yolo_detect_labels


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
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate two-stage proposal CSV classes against draft YOLO labels.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--class-column", default="fragment_class")
    parser.add_argument("--score-column", default="fragment_conf")
    parser.add_argument("--match-iou", type=float, default=0.50)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    image_path = resolve(args.image)
    labels_path = resolve(args.labels)
    csv_path = resolve(args.csv)
    with Image.open(image_path) as image:
        gt_boxes, gt_classes = read_yolo_detect_labels(labels_path, image.size)
    pred_boxes: list[list[float]] = []
    pred_classes: list[int] = []
    pred_scores: list[float] = []
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    for row in rows:
        class_name = row[args.class_column]
        if class_name not in CLASS_TO_ID:
            continue
        pred_boxes.append([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])])
        pred_classes.append(CLASS_TO_ID[class_name])
        pred_scores.append(float(row[args.score_column]))
    boxes = np.asarray(pred_boxes, dtype=float)
    classes = np.asarray(pred_classes, dtype=int)
    scores = np.asarray(pred_scores, dtype=float)
    same = greedy_match(boxes, classes, scores, gt_boxes, gt_classes, args.match_iou, require_same_class=True)
    any_class = greedy_match(boxes, classes, scores, gt_boxes, gt_classes, args.match_iou, require_same_class=False)
    counts: dict[str, int] = {}
    for class_id in classes:
        name = CLASS_NAMES[int(class_id)]
        counts[name] = counts.get(name, 0) + 1
    print(f"csv={csv_path.relative_to(ROOT)}")
    print(f"gt={len(gt_boxes)} pred={len(boxes)} count_error={len(boxes) - len(gt_boxes)}")
    print(f"matched_same_class={same} recall_same_class={same / max(1, len(gt_boxes)):.3f}")
    print(f"matched_any_class={any_class} recall_any_class={any_class / max(1, len(gt_boxes)):.3f}")
    print("pred_classes=" + ";".join(f"{key}:{value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()
