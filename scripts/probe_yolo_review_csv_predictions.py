#!/usr/bin/env python
"""Probe YOLO predictions against reviewed CSV boxes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from hardware_profile import recommended_device
from local_runtime import configure_project_cache


configure_project_cache()

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path, help="YOLO data YAML for class names.")
    parser.add_argument("--review-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--target-class", required=True)
    parser.add_argument("--accepted-decision", default="accepted_box")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--det-iou", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(path)} must be a YAML mapping")
    return payload


def names_by_id(config: dict[str, Any]) -> dict[int, str]:
    raw = config.get("names")
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    raise SystemExit("data config has no names mapping")


def read_review_rows(path: Path, accepted_decision: str, target_class: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("review_decision") or "").strip() != accepted_decision:
                continue
            if (row.get("proposed_new_class") or "").strip() != target_class:
                continue
            try:
                box = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"malformed box in {path}: {row}") from exc
            rows.append(
                {
                    "review_id": row.get("review_id", ""),
                    "image": repo_rel(resolve(row["image"])),
                    "target_class": target_class,
                    "xyxy": normalize_box(box),
                    "source_row": row,
                }
            )
    if not rows:
        raise SystemExit(f"no accepted rows for {target_class} in {repo_rel(path)}")
    return rows


def normalize_box(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def predictions_from_result(result: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    if result.boxes is None:
        return predictions
    xyxy = result.boxes.xyxy.cpu().numpy().tolist()
    confs = result.boxes.conf.cpu().numpy().tolist()
    classes = result.boxes.cls.cpu().numpy().tolist()
    for box, conf, class_id_raw in zip(xyxy, confs, classes, strict=False):
        class_id = int(class_id_raw)
        predictions.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, f"class_{class_id}"),
                "confidence": float(conf),
                "xyxy": [float(value) for value in box],
            }
        )
    return predictions


def best_prediction(label: dict[str, Any], predictions: list[dict[str, Any]], target_class_id: int | None) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_iou = 0.0
    for prediction in predictions:
        if target_class_id is not None and int(prediction["class_id"]) != target_class_id:
            continue
        score = box_iou(label["xyxy"], prediction["xyxy"])
        if score > best_iou:
            best_iou = score
            best = prediction
    if best is None:
        return None
    return {**best, "iou": best_iou}


def prediction_fields(prefix: str, prediction: dict[str, Any] | None) -> dict[str, Any]:
    if prediction is None:
        return {
            f"{prefix}_class": "",
            f"{prefix}_class_id": "",
            f"{prefix}_confidence": "",
            f"{prefix}_iou": 0.0,
        }
    return {
        f"{prefix}_class": prediction["class_name"],
        f"{prefix}_class_id": prediction["class_id"],
        f"{prefix}_confidence": round(float(prediction["confidence"]), 6),
        f"{prefix}_iou": round(float(prediction["iou"]), 6),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_id",
        "image",
        "target_class",
        "predictions",
        "target_predictions",
        "target_conf_max",
        "target_iou_max",
        "target_iou_ge_match",
        "any_iou_ge_match",
        "best_target_class",
        "best_target_class_id",
        "best_target_confidence",
        "best_target_iou",
        "best_any_class",
        "best_any_class_id",
        "best_any_confidence",
        "best_any_iou",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    data_path = resolve(args.data)
    names = names_by_id(read_yaml(data_path))
    name_to_id = {name: class_id for class_id, name in names.items()}
    if args.target_class not in name_to_id:
        raise SystemExit(f"unknown target class {args.target_class!r} in {repo_rel(data_path)}")
    target_class_id = name_to_id[args.target_class]
    review_rows = read_review_rows(resolve(args.review_csv), args.accepted_decision, args.target_class)
    image_paths = [str(resolve(row["image"])) for row in review_rows]

    model = YOLO(str(resolve(args.model)))
    results = model.predict(
        source=image_paths,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.det_iou,
        device=recommended_device(args.device),
        verbose=False,
    )

    output_rows: list[dict[str, Any]] = []
    target_prediction_counts: Counter[str] = Counter()
    any_prediction_counts: Counter[str] = Counter()
    target_hits = 0
    any_hits = 0
    with_target_prediction = 0
    for review_row, result in zip(review_rows, results, strict=True):
        predictions = predictions_from_result(result, names)
        target_predictions = [prediction for prediction in predictions if int(prediction["class_id"]) == target_class_id]
        best_target = best_prediction(review_row, predictions, target_class_id)
        best_any = best_prediction(review_row, predictions, None)
        target_iou = float(best_target["iou"]) if best_target else 0.0
        any_iou = float(best_any["iou"]) if best_any else 0.0
        target_hit = target_iou >= args.match_iou
        any_hit = any_iou >= args.match_iou
        target_hits += int(target_hit)
        any_hits += int(any_hit)
        with_target_prediction += int(bool(target_predictions))
        for prediction in target_predictions:
            target_prediction_counts[prediction["class_name"]] += 1
        for prediction in predictions:
            any_prediction_counts[prediction["class_name"]] += 1
        output_rows.append(
            {
                "review_id": review_row["review_id"],
                "image": review_row["image"],
                "target_class": args.target_class,
                "predictions": len(predictions),
                "target_predictions": len(target_predictions),
                "target_conf_max": round(max((float(p["confidence"]) for p in target_predictions), default=0.0), 6),
                "target_iou_max": round(target_iou, 6),
                "target_iou_ge_match": target_hit,
                "any_iou_ge_match": any_hit,
                **prediction_fields("best_target", best_target),
                **prediction_fields("best_any", best_any),
            }
        )

    summary = {
        "schema": "cashsnap_yolo_review_csv_prediction_probe_v1",
        "model": repo_rel(resolve(args.model)),
        "data": repo_rel(data_path),
        "review_csv": repo_rel(resolve(args.review_csv)),
        "target_class": args.target_class,
        "target_class_id": target_class_id,
        "accepted_decision": args.accepted_decision,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "det_iou": args.det_iou,
        "match_iou": args.match_iou,
        "rows": len(review_rows),
        "rows_with_target_prediction": with_target_prediction,
        "target_iou_hits": target_hits,
        "any_iou_hits": any_hits,
        "target_prediction_class_counts": dict(target_prediction_counts.most_common()),
        "all_prediction_class_counts": dict(any_prediction_counts.most_common()),
        "out_csv": repo_rel(resolve(args.out_csv)),
    }
    resolve(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    resolve(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(resolve(args.out_csv), output_rows)
    print(
        f"probe rows={len(review_rows)} target_predictions={with_target_prediction} "
        f"target_iou_hits={target_hits} any_iou_hits={any_hits}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
