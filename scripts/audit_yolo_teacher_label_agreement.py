#!/usr/bin/env python
"""Audit labeled YOLO rows against a teacher detector's class-aware predictions."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from local_runtime import configure_project_cache

configure_project_cache()

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SOURCE_PREFIXES = {
    "asian_currency_": "asian_currency",
    "billsbank_": "billsbank",
    "cambodia_currency_project_": "cambodia_currency_project",
    "cashcountingxl_": "cashcountingxl",
    "khmer_us_currency_": "khmer_us_currency",
    "usd_total_": "usd_total",
}


def resolve(path: str | Path) -> Path:
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
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--match-iou", type=float, default=0.10)
    parser.add_argument("--min-label-coverage", type=float, default=0.50)
    parser.add_argument("--min-prediction-coverage", type=float, default=0.25)
    parser.add_argument("--max-extra-predictions", type=int, default=0)
    parser.add_argument(
        "--extra-prediction-policy",
        choices=["count_all", "ignore_label_overlap"],
        default="count_all",
        help="Whether overlapping duplicate teacher detections count as extra predictions for acceptance.",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--source-group-include",
        default="",
        help="Comma-separated inferred source groups to include, e.g. usd_total,billsbank.",
    )
    parser.add_argument(
        "--source-group-exclude",
        default="",
        help="Comma-separated inferred source groups to exclude.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--accepted-list-out", type=Path, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"YOLO data config must be a mapping: {repo_rel(path)}")
    return payload


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw_root = Path(str(config.get("path", ".")))
    return raw_root if raw_root.is_absolute() else (config_path.parent / raw_root).resolve()


def class_names_from_config(config_path: Path | None) -> dict[int, str]:
    if config_path is None:
        return {}
    config = load_yaml(config_path)
    names = config.get("names", {})
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        out: dict[int, str] = {}
        for raw_key, raw_name in names.items():
            try:
                out[int(raw_key)] = str(raw_name)
            except (TypeError, ValueError):
                continue
        return out
    return {}


def split_images(config_path: Path, split: str) -> list[Path]:
    config = load_yaml(config_path)
    root = data_root(config_path, config)
    split_value = config.get(split)
    if split_value is None and split == "val":
        split_value = config.get("valid")
    if split_value is None:
        raise SystemExit(f"{repo_rel(config_path)} has no split {split!r}")
    values = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for raw in values:
        split_path = Path(str(raw))
        split_path = split_path if split_path.is_absolute() else root / split_path
        if split_path.suffix.lower() == ".txt":
            for raw_line in split_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line and not line.startswith("#"):
                    image = Path(line)
                    images.append(image if image.is_absolute() else root / image)
        else:
            images.extend(sorted(path for path in split_path.glob("*") if path.suffix.lower() in IMAGE_EXTS))
    return unique_images(images)


def manifest_images(path: Path) -> list[Path]:
    images: list[Path] = []
    if path.suffix.lower() == ".jsonl":
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            image = payload.get("image")
            if image:
                images.append(resolve(image))
    elif path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "image" not in (reader.fieldnames or []):
                raise SystemExit(f"{repo_rel(path)} must include an image column")
            images.extend(resolve(row["image"]) for row in reader if row.get("image"))
    else:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                images.append(resolve(line))
    return unique_images(images)


def unique_images(images: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for image in images:
        key = image.resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append(image)
    return out


def source_group_for_image(image_path: Path) -> str:
    name = image_path.name.lower()
    for prefix, group in SOURCE_PREFIXES.items():
        if name.startswith(prefix) or f"_{prefix}" in name:
            return group
    return name.split("_", 1)[0] if "_" in name else "unknown"


def parse_source_groups(value: str) -> set[str]:
    return {item.strip() for item in value.replace(";", ",").split(",") if item.strip()}


def source_group_allowed(image_path: Path, args: argparse.Namespace) -> bool:
    source_group = source_group_for_image(image_path)
    include = parse_source_groups(args.source_group_include)
    exclude = parse_source_groups(args.source_group_exclude)
    if include and source_group not in include:
        return False
    if exclude and source_group in exclude:
        return False
    return True


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def label_box_xyxy(cx: float, cy: float, bw: float, bh: float, width: int, height: int) -> list[float]:
    return [
        (cx - bw / 2.0) * width,
        (cy - bh / 2.0) * height,
        (cx + bw / 2.0) * width,
        (cy + bh / 2.0) * height,
    ]


def load_labels(image_path: Path) -> list[dict[str, Any]]:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return []
    with Image.open(image_path) as opened:
        width, height = opened.size
    labels: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"bad YOLO label row {repo_rel(label_path)}:{line_no}: {line}")
        cls, cx, cy, bw, bh = parts
        labels.append(
            {
                "class_id": int(float(cls)),
                "box": label_box_xyxy(float(cx), float(cy), float(bw), float(bh), width, height),
                "line": line,
            }
        )
    return labels


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_metrics(a: list[float], b: list[float]) -> dict[str, float]:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = area(a)
    area_b = area(b)
    union = area_a + area_b - inter
    return {
        "iou": inter / union if union > 0 else 0.0,
        "label_coverage": inter / area_a if area_a > 0 else 0.0,
        "prediction_coverage": inter / area_b if area_b > 0 else 0.0,
    }


def prediction_rows(result: Any) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    rows: list[dict[str, Any]] = []
    for box, cls, conf in zip(xyxy, classes, confidences, strict=True):
        rows.append({"class_id": int(cls), "box": [float(value) for value in box], "conf": float(conf)})
    return rows


def best_prediction_for_label(
    label: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> tuple[int | None, dict[str, float]]:
    best_index: int | None = None
    best = {"iou": 0.0, "label_coverage": 0.0, "prediction_coverage": 0.0}
    for index, prediction in enumerate(predictions):
        if prediction["class_id"] != label["class_id"]:
            continue
        metrics = box_metrics(label["box"], prediction["box"])
        score = (metrics["iou"], metrics["label_coverage"], metrics["prediction_coverage"])
        best_score = (best["iou"], best["label_coverage"], best["prediction_coverage"])
        if score > best_score:
            best_index = index
            best = metrics
    return best_index, best


def label_matched(metrics: dict[str, float], args: argparse.Namespace) -> bool:
    return (
        metrics["iou"] >= args.match_iou
        or (
            metrics["label_coverage"] >= args.min_label_coverage
            and metrics["prediction_coverage"] >= args.min_prediction_coverage
        )
    )


def class_count_payload(counter: Counter[int], names: dict[int, str]) -> dict[str, int]:
    payload: dict[str, int] = {}
    for class_id, count in sorted(counter.items()):
        class_name = names.get(class_id)
        key = f"{class_id}:{class_name}" if class_name else str(class_id)
        payload[key] = count
    return payload


def record_label_class_counts(records: list[dict[str, Any]], *, matched: bool | None = None) -> Counter[int]:
    counts: Counter[int] = Counter()
    for record in records:
        for label_record in record["label_records"]:
            if matched is not None and bool(label_record["matched"]) != matched:
                continue
            counts[int(label_record["class_id"])] += 1
    return counts


def record_extra_prediction_class_counts(
    records: list[dict[str, Any]],
    *,
    counted_only: bool = False,
) -> Counter[int]:
    counts: Counter[int] = Counter()
    for record in records:
        for prediction_record in record["extra_prediction_records"]:
            if counted_only and not prediction_record.get("counted_for_acceptance", True):
                continue
            counts[int(prediction_record["class_id"])] += 1
    return counts


def record_slice_summary(records: list[dict[str, Any]], names: dict[int, str]) -> dict[str, Any]:
    labels = sum(int(record["labels"]) for record in records)
    matched_labels = sum(int(record["matched_labels"]) for record in records)
    extra_predictions = sum(int(record["extra_predictions"]) for record in records)
    extra_predictions_for_acceptance = sum(int(record["extra_predictions_for_acceptance"]) for record in records)
    return {
        "images": len(records),
        "labels": labels,
        "matched_labels": matched_labels,
        "extra_predictions": extra_predictions,
        "extra_predictions_for_acceptance": extra_predictions_for_acceptance,
        "label_class_counts": class_count_payload(record_label_class_counts(records), names),
        "matched_label_class_counts": class_count_payload(record_label_class_counts(records, matched=True), names),
        "missing_label_class_counts": class_count_payload(record_label_class_counts(records, matched=False), names),
        "extra_prediction_class_counts": class_count_payload(record_extra_prediction_class_counts(records), names),
        "counted_extra_prediction_class_counts": class_count_payload(
            record_extra_prediction_class_counts(records, counted_only=True),
            names,
        ),
        "images_list": [record["image"] for record in records],
    }


def audit_image(
    image: Path,
    labels: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    used_predictions: set[int] = set()
    label_records: list[dict[str, Any]] = []
    missing_labels: list[dict[str, Any]] = []
    for label_index, label in enumerate(labels):
        prediction_index, metrics = best_prediction_for_label(label, predictions)
        matched = prediction_index is not None and label_matched(metrics, args)
        if matched and prediction_index is not None:
            used_predictions.add(prediction_index)
        record = {
            "label_index": label_index,
            "class_id": label["class_id"],
            "matched": matched,
            "best_prediction_index": prediction_index,
            **{key: round(float(value), 6) for key, value in metrics.items()},
        }
        label_records.append(record)
        if not matched:
            missing_labels.append(record)

    extra_predictions: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        if index in used_predictions:
            continue
        best_any = {"iou": 0.0, "label_coverage": 0.0, "prediction_coverage": 0.0}
        best_label_index: int | None = None
        best_label_class_id: int | None = None
        for label_index, label in enumerate(labels):
            metrics = box_metrics(label["box"], prediction["box"])
            if (metrics["iou"], metrics["prediction_coverage"]) > (
                best_any["iou"],
                best_any["prediction_coverage"],
            ):
                best_any = metrics
                best_label_index = label_index
                best_label_class_id = int(label["class_id"])
        overlaps_label = label_matched(best_any, args)
        counted_for_acceptance = args.extra_prediction_policy == "count_all" or not overlaps_label
        extra_predictions.append(
            {
                "prediction_index": index,
                "class_id": prediction["class_id"],
                "conf": round(float(prediction["conf"]), 6),
                "best_label_index": best_label_index,
                "best_label_class_id": best_label_class_id,
                "overlaps_label": overlaps_label,
                "counted_for_acceptance": counted_for_acceptance,
                **{key: round(float(value), 6) for key, value in best_any.items()},
            }
        )
    extra_predictions_for_acceptance = [
        row for row in extra_predictions if row["counted_for_acceptance"]
    ]

    reasons: list[str] = []
    if not labels:
        reasons.append("no_labels")
    if missing_labels:
        reasons.append("missing_or_wrong_class_labels")
    if len(extra_predictions_for_acceptance) > args.max_extra_predictions:
        reasons.append("extra_teacher_predictions")
    accepted = not reasons
    return {
        "image": repo_rel(image),
        "source_group": source_group_for_image(image),
        "label": repo_rel(label_path_for_image(image)),
        "labels": len(labels),
        "predictions": len(predictions),
        "matched_labels": len(labels) - len(missing_labels),
        "extra_predictions": len(extra_predictions),
        "extra_predictions_for_acceptance": len(extra_predictions_for_acceptance),
        "overlapping_extra_predictions": len(extra_predictions) - len(extra_predictions_for_acceptance),
        "accepted": accepted,
        "reasons": reasons,
        "label_records": label_records,
        "extra_prediction_records": extra_predictions,
    }


def main() -> int:
    args = parse_args()
    if (args.data is None) == (args.manifest is None):
        raise SystemExit("provide exactly one of --data or --manifest")
    if args.max_extra_predictions < 0:
        raise SystemExit("--max-extra-predictions must be >= 0")
    if args.data is not None:
        source = resolve(args.data)
        images = split_images(source, args.split)
        source_kind = "data_yaml"
        class_names = class_names_from_config(source)
    else:
        source = resolve(args.manifest)
        images = manifest_images(source)
        source_kind = "manifest"
        class_names = {}
    images = [image for image in images if source_group_allowed(image, args)]
    if args.max_images > 0 and len(images) > args.max_images:
        rng = random.Random(args.seed)
        images = sorted(rng.sample(images, args.max_images), key=lambda path: path.as_posix())

    model = YOLO(str(resolve(args.model)))
    records: list[dict[str, Any]] = []
    for start in range(0, len(images), args.batch):
        batch = images[start : start + args.batch]
        results = model.predict(
            source=[str(image) for image in batch],
            imgsz=args.imgsz,
            conf=args.conf,
            batch=len(batch),
            device=args.device,
            verbose=False,
        )
        for image, result in zip(batch, results, strict=True):
            records.append(audit_image(image, load_labels(image), prediction_rows(result), args))

    reason_counts: Counter[str] = Counter()
    class_missing_counts: Counter[int] = Counter()
    for record in records:
        reason_counts.update(record["reasons"])
        for label_record in record["label_records"]:
            if not label_record["matched"]:
                class_missing_counts[int(label_record["class_id"])] += 1

    accepted = [record for record in records if record["accepted"]]
    all_labels_matched = [
        record for record in records if record["labels"] > 0 and record["matched_labels"] == record["labels"]
    ]
    no_extra = [record for record in records if record["extra_predictions_for_acceptance"] <= args.max_extra_predictions]
    payload = {
        "schema": "cashsnap_yolo_teacher_label_agreement_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": repo_rel(resolve(args.model)),
        "source": repo_rel(source),
        "source_kind": source_kind,
        "class_names": {str(key): value for key, value in sorted(class_names.items())},
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "match_iou": args.match_iou,
        "min_label_coverage": args.min_label_coverage,
        "min_prediction_coverage": args.min_prediction_coverage,
        "max_extra_predictions": args.max_extra_predictions,
        "extra_prediction_policy": args.extra_prediction_policy,
        "source_group_include": args.source_group_include,
        "source_group_exclude": args.source_group_exclude,
        "images": len(records),
        "accepted_images": len(accepted),
        "rejected_images": len(records) - len(accepted),
        "reason_counts": dict(sorted(reason_counts.items())),
        "missing_label_class_counts": {str(key): value for key, value in sorted(class_missing_counts.items())},
        "missing_label_class_counts_named": class_count_payload(class_missing_counts, class_names),
        "slices": {
            "accepted": record_slice_summary(accepted, class_names),
            "all_labels_matched": record_slice_summary(all_labels_matched, class_names),
            "no_counted_extra_predictions": record_slice_summary(no_extra, class_names),
            "no_raw_extra_predictions": record_slice_summary(
                [record for record in records if record["extra_predictions"] <= args.max_extra_predictions],
                class_names,
            ),
        },
        "accepted_images_list": [record["image"] for record in accepted],
        "all_labels_matched_images_list": [record["image"] for record in all_labels_matched],
        "no_counted_extra_predictions_images_list": [record["image"] for record in no_extra],
        "records": records,
    }
    json_out = resolve(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.csv_out is not None:
        csv_out = resolve(args.csv_out)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "image",
                    "source_group",
                    "accepted",
                    "labels",
                    "matched_labels",
                    "predictions",
                    "extra_predictions",
                    "extra_predictions_for_acceptance",
                    "reasons",
                ],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "image": record["image"],
                        "source_group": record["source_group"],
                        "accepted": int(record["accepted"]),
                        "labels": record["labels"],
                        "matched_labels": record["matched_labels"],
                        "predictions": record["predictions"],
                        "extra_predictions": record["extra_predictions"],
                        "extra_predictions_for_acceptance": record["extra_predictions_for_acceptance"],
                        "reasons": "|".join(record["reasons"]),
                    }
                )

    if args.accepted_list_out is not None:
        list_out = resolve(args.accepted_list_out)
        list_out.parent.mkdir(parents=True, exist_ok=True)
        list_out.write_text("".join(path + "\n" for path in payload["accepted_images_list"]), encoding="utf-8")

    print(
        "teacher_label_agreement "
        f"images={payload['images']} accepted={payload['accepted_images']} "
        f"rejected={payload['rejected_images']} reasons={payload['reason_counts']}"
    )
    print(f"wrote_json={repo_rel(json_out)}")
    if args.accepted_list_out is not None:
        print(f"wrote_accepted_list={repo_rel(resolve(args.accepted_list_out))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
