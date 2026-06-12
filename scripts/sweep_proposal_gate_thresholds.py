#!/usr/bin/env python
"""Sweep proposal-gate reject thresholds from saved detector+gate proposal rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CLASS_VALUES = {
    "USD_1": ("USD", 1.0),
    "USD_5": ("USD", 5.0),
    "USD_10": ("USD", 10.0),
    "USD_20": ("USD", 20.0),
    "USD_50": ("USD", 50.0),
    "USD_100": ("USD", 100.0),
    "KHR_500": ("KHR", 500.0),
    "KHR_1000": ("KHR", 1000.0),
    "KHR_2000": ("KHR", 2000.0),
    "KHR_5000": ("KHR", 5000.0),
    "KHR_10000": ("KHR", 10000.0),
    "KHR_20000": ("KHR", 20000.0),
    "KHR_50000": ("KHR", 50000.0),
}


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proposal",
        action="append",
        required=True,
        help="Repeated LABEL=proposal_json path produced by probe_yolo_proposal_gate.py.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=[],
        help="Reject-min-conf value to evaluate. Repeat for a grid.",
    )
    parser.add_argument(
        "--bgfp-cost",
        action="append",
        type=float,
        default=[],
        help="Utility cost per background image with a false positive. Repeat for multiple product stances.",
    )
    parser.add_argument(
        "--bgfp-cap",
        action="append",
        type=int,
        default=[],
        help="Optional background-FP image cap used for guarded exact-value ranking.",
    )
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
    parser.add_argument(
        "--reject-max-det-conf",
        action="append",
        type=float,
        default=[],
        help=(
            "Only allow gate rejection when detector confidence is at or below this value. "
            "Repeat to sweep several caps; omit to allow rejection at any detector confidence."
        ),
    )
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--csv-out", type=Path, default=None)
    return parser.parse_args()


def box_iou(a: list[float], b: list[float]) -> float:
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


def match_predictions(
    labels: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
    *,
    match_classes: bool,
) -> tuple[set[int], list[dict[str, Any]]]:
    matched_labels: set[int] = set()
    false_predictions: list[dict[str, Any]] = []
    for prediction in sorted(predictions, key=lambda item: float(item["confidence"]), reverse=True):
        best_index = -1
        best_iou = 0.0
        for index, label in enumerate(labels):
            if index in matched_labels:
                continue
            if match_classes and int(label["class_id"]) != int(prediction["class_id"]):
                continue
            score = box_iou(prediction["xyxy"], label["xyxy"])
            if score > best_iou:
                best_iou = score
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched_labels.add(best_index)
        else:
            false_predictions.append({**prediction, "best_iou": best_iou})
    return matched_labels, false_predictions


def value_totals(rows: list[dict[str, Any]], names: dict[int, str]) -> tuple[float, float]:
    usd_total = 0.0
    khr_total = 0.0
    for row in rows:
        class_name = names.get(int(row["class_id"]), str(row["class_id"]))
        value = CLASS_VALUES.get(class_name)
        if value is None:
            continue
        currency, amount = value
        if currency == "USD":
            usd_total += amount
        elif currency == "KHR":
            khr_total += amount
    return usd_total, khr_total


def reject_threshold_for_prediction(
    prediction: dict[str, Any],
    names: dict[int, str],
    *,
    base_threshold: float,
    class_thresholds: dict[str, float],
    class_threshold_min_det_conf: float | None,
) -> float:
    class_name = names.get(int(prediction["class_id"]), str(prediction["class_id"]))
    if class_name not in class_thresholds:
        return base_threshold
    if (
        class_threshold_min_det_conf is not None
        and float(prediction.get("confidence", 0.0)) < class_threshold_min_det_conf
    ):
        return base_threshold
    return class_thresholds[class_name]


def summarize(
    payload: dict[str, Any],
    *,
    threshold: float,
    bgfp_costs: list[float],
    reject_max_det_conf: float | None = None,
    class_thresholds: dict[str, float] | None = None,
    class_threshold_min_det_conf: float | None = None,
) -> dict[str, Any]:
    names = {int(key): str(value) for key, value in payload["names"].items()}
    match_iou = float(payload["match_iou"])
    match_classes = bool(payload.get("detector_match_classes", True))
    reject_class = str(payload.get("reject_class", "background"))
    class_thresholds = class_thresholds or {}

    images = 0
    background_images = 0
    background_images_with_fp = 0
    images_with_fp = 0
    total_gt = 0
    total_predictions = 0
    total_tp = 0
    total_fp = 0
    count_abs_error_sum = 0.0
    usd_abs_error_sum = 0.0
    khr_abs_error_sum = 0.0
    exact_count_images = 0
    exact_value_images = 0
    source_images: Counter[str] = Counter()
    source_bgfp: Counter[str] = Counter()
    source_exact_value: Counter[str] = Counter()
    class_gt: Counter[int] = Counter()
    class_tp: Counter[int] = Counter()
    class_fp: Counter[int] = Counter()

    for row in payload["rows"]:
        images += 1
        source = str(row["source_group"])
        source_images[source] += 1
        labels = row["labels"]
        predictions = []
        for prediction in row["predictions"]:
            reject_threshold = reject_threshold_for_prediction(
                prediction,
                names,
                base_threshold=threshold,
                class_thresholds=class_thresholds,
                class_threshold_min_det_conf=class_threshold_min_det_conf,
            )
            rejected = (
                str(prediction.get("gate_class")) == reject_class
                and float(prediction.get("gate_conf", 0.0)) >= reject_threshold
                and (
                    reject_max_det_conf is None
                    or float(prediction.get("confidence", 0.0)) <= reject_max_det_conf
                )
            )
            if not rejected:
                predictions.append(prediction)
        if not labels:
            background_images += 1
        for label in labels:
            class_gt[int(label["class_id"])] += 1
        matched_labels, false_predictions = match_predictions(
            labels,
            predictions,
            match_iou,
            match_classes=match_classes,
        )
        total_gt += len(labels)
        total_predictions += len(predictions)
        total_tp += len(matched_labels)
        total_fp += len(false_predictions)
        if false_predictions:
            images_with_fp += 1
            if not labels:
                background_images_with_fp += 1
                source_bgfp[source] += 1
        count_error = len(predictions) - len(labels)
        count_abs_error_sum += abs(float(count_error))
        if count_error == 0:
            exact_count_images += 1
        if match_classes:
            gt_usd, gt_khr = value_totals(labels, names)
            pred_usd, pred_khr = value_totals(predictions, names)
            usd_error = pred_usd - gt_usd
            khr_error = pred_khr - gt_khr
            usd_abs_error_sum += abs(usd_error)
            khr_abs_error_sum += abs(khr_error)
            if usd_error == 0 and khr_error == 0:
                exact_value_images += 1
                source_exact_value[source] += 1
        for index, label in enumerate(labels):
            if index in matched_labels:
                class_tp[int(label["class_id"])] += 1
        for prediction in false_predictions:
            class_fp[int(prediction["class_id"])] += 1

    total_fn = total_gt - total_tp
    per_class = {}
    for class_id in sorted(set(class_gt) | set(class_tp) | set(class_fp)):
        gt = int(class_gt[class_id])
        tp = int(class_tp[class_id])
        fp = int(class_fp[class_id])
        fn = gt - tp
        per_class[names.get(class_id, str(class_id))] = {
            "gt": gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": tp / gt if gt else None,
        }
    utilities = {
        f"exact_value_minus_{cost:g}x_bgfp": exact_value_images - cost * background_images_with_fp
        for cost in bgfp_costs
    }
    return {
        "threshold": threshold,
        "reject_max_det_conf": reject_max_det_conf,
        "images": images,
        "background_images": background_images,
        "labeled_images": images - background_images,
        "gt": total_gt,
        "total_predictions": total_predictions,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "recall": total_tp / total_gt if total_gt else None,
        "precision": total_tp / total_predictions if total_predictions else None,
        "images_with_fp": images_with_fp,
        "background_images_with_fp": background_images_with_fp,
        "count_value_errors": {
            "exact_count_images": exact_count_images,
            "exact_value_images": exact_value_images if match_classes else None,
            "mean_abs_count_error": count_abs_error_sum / max(1, images),
            "mean_abs_usd_total_error": usd_abs_error_sum / max(1, images) if match_classes else None,
            "mean_abs_khr_total_error": khr_abs_error_sum / max(1, images) if match_classes else None,
        },
        "utilities": utilities,
        "source_background_images_with_fp": dict(sorted(source_bgfp.items())),
        "source_exact_value_images": dict(sorted(source_exact_value.items())),
        "source_images": dict(sorted(source_images.items())),
        "per_class": per_class,
    }


def parse_proposal_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--proposal must be LABEL=path, got {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"--proposal label is empty in {spec!r}")
    return label, resolve(path.strip())


def parse_class_thresholds(raw_values: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for raw_value in raw_values:
        if "=" not in raw_value:
            raise SystemExit(f"--class-threshold must be CLASS=THRESHOLD, got {raw_value!r}")
        class_name, threshold_text = raw_value.split("=", 1)
        class_name = class_name.strip()
        if not class_name:
            raise SystemExit(f"--class-threshold class is empty in {raw_value!r}")
        try:
            threshold = float(threshold_text)
        except ValueError as exc:
            raise SystemExit(f"invalid threshold in {raw_value!r}") from exc
        thresholds[class_name] = threshold
    return thresholds


def best_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    best_value = max(float(row[key]) for row in rows if row[key] is not None)
    return [row for row in rows if row[key] == best_value]


def main() -> None:
    args = parse_args()
    thresholds = sorted(set(args.threshold or [0.0, 0.50, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.1]))
    reject_max_det_confs = [None] if not args.reject_max_det_conf else sorted(set(args.reject_max_det_conf))
    bgfp_costs = sorted(set(args.bgfp_cost or [1.0, 2.0, 5.0]))
    class_thresholds = parse_class_thresholds(args.class_threshold)

    rows: list[dict[str, Any]] = []
    inputs = {}
    for spec in args.proposal:
        label, path = parse_proposal_spec(spec)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "cashsnap_yolo_proposal_gate_proposals_v1":
            raise SystemExit(f"{repo_rel(path)} is not proposal schema v1")
        inputs[label] = repo_rel(path)
        for threshold in thresholds:
            for reject_max_det_conf in reject_max_det_confs:
                summary = summarize(
                    payload,
                    threshold=threshold,
                    bgfp_costs=bgfp_costs,
                    reject_max_det_conf=reject_max_det_conf,
                    class_thresholds=class_thresholds,
                    class_threshold_min_det_conf=args.class_threshold_min_det_conf,
                )
                exact_value_images = summary["count_value_errors"]["exact_value_images"]
                row = {
                    "label": label,
                    "proposal": repo_rel(path),
                    "threshold": threshold,
                    "reject_max_det_conf": reject_max_det_conf,
                    "images": summary["images"],
                    "recall": summary["recall"],
                    "precision": summary["precision"],
                    "background_images_with_fp": summary["background_images_with_fp"],
                    "images_with_fp": summary["images_with_fp"],
                    "total_predictions": summary["total_predictions"],
                    "exact_count_images": summary["count_value_errors"]["exact_count_images"],
                    "exact_value_images": exact_value_images,
                    "mean_abs_count_error": summary["count_value_errors"]["mean_abs_count_error"],
                    "mean_abs_usd_total_error": summary["count_value_errors"]["mean_abs_usd_total_error"],
                    "mean_abs_khr_total_error": summary["count_value_errors"]["mean_abs_khr_total_error"],
                    "utilities": summary["utilities"],
                    "summary": summary,
                }
                for utility_key, utility_value in summary["utilities"].items():
                    row[utility_key] = utility_value
                rows.append(row)

    flat_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"summary", "utilities"}
        }
        for row in rows
    ]
    rankings: dict[str, Any] = {
        "best_by_exact_value_images": best_rows(flat_rows, "exact_value_images"),
        "best_by_exact_count_images": best_rows(flat_rows, "exact_count_images"),
    }
    for cost in bgfp_costs:
        key = f"exact_value_minus_{cost:g}x_bgfp"
        rankings[f"best_by_{key}"] = best_rows(flat_rows, key)
    for cap in sorted(set(args.bgfp_cap)):
        capped = [row for row in flat_rows if int(row["background_images_with_fp"]) <= cap]
        rankings[f"best_exact_value_with_bgfp_lte_{cap}"] = best_rows(capped, "exact_value_images")

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flat_rows:
        by_label[str(row["label"])].append(row)
    payload = {
        "schema": "cashsnap_proposal_gate_threshold_sweep_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "thresholds": thresholds,
        "reject_max_det_confs": reject_max_det_confs,
        "class_thresholds": class_thresholds,
        "class_threshold_min_det_conf": args.class_threshold_min_det_conf,
        "bgfp_costs": bgfp_costs,
        "bgfp_caps": sorted(set(args.bgfp_cap)),
        "rows": flat_rows,
        "rows_by_label": {
            label: sorted(
                label_rows,
                key=lambda item: (
                    item["threshold"],
                    -1.0 if item["reject_max_det_conf"] is None else item["reject_max_det_conf"],
                ),
            )
            for label, label_rows in sorted(by_label.items())
        },
        "rankings": rankings,
    }
    out_path = resolve(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.csv_out is not None:
        csv_path = resolve(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "label",
            "threshold",
            "reject_max_det_conf",
            "recall",
            "precision",
            "background_images_with_fp",
            "images_with_fp",
            "total_predictions",
            "exact_count_images",
            "exact_value_images",
            "mean_abs_count_error",
            "mean_abs_usd_total_error",
            "mean_abs_khr_total_error",
            *[f"exact_value_minus_{cost:g}x_bgfp" for cost in bgfp_costs],
            "proposal",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in flat_rows:
                writer.writerow({field: row.get(field) for field in fieldnames})

    print(
        f"wrote {repo_rel(out_path)} rows={len(flat_rows)} "
        f"labels={','.join(sorted(inputs))}"
    )


if __name__ == "__main__":
    main()
