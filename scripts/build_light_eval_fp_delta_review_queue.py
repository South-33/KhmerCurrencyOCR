#!/usr/bin/env python
"""Build a detector FP-delta review queue from lightweight eval JSONs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--slice-name", default="")
    parser.add_argument(
        "--only-class",
        action="append",
        default=[],
        help="Optional class name filter. Repeat or comma-separate.",
    )
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--csv-out", required=True, type=Path)
    parser.add_argument("--sheet-out", type=Path, default=None)
    parser.add_argument("--sheet-items", type=int, default=80)
    parser.add_argument("--thumb-width", type=int, default=240)
    parser.add_argument("--cols", type=int, default=5)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {repo_rel(resolved)}")
    return payload


def class_names_from_data(data_path: str) -> dict[int, str]:
    resolved = resolve(data_path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    raw_names = payload.get("names", {}) if isinstance(payload, dict) else {}
    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        names: dict[int, str] = {}
        for raw_id, raw_name in raw_names.items():
            try:
                names[int(raw_id)] = str(raw_name)
            except (TypeError, ValueError):
                continue
        return names
    return {}


def parse_class_filter(raw_values: list[str]) -> set[str]:
    values: set[str] = set()
    for raw in raw_values:
        values.update(token.strip() for token in raw.split(",") if token.strip())
    return values


def metric_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    name: str,
    metric: str,
) -> float:
    return float(candidate.get(name, {}).get(metric, 0) or 0) - float(baseline.get(name, {}).get(metric, 0) or 0)


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


def fp_kind(prediction: dict[str, Any], labels: list[dict[str, Any]]) -> tuple[str, float, float]:
    if not labels:
        return "background_empty", 0.0, 0.0
    pred_box = prediction.get("xyxy", [])
    pred_class = int(prediction.get("class_id", -1))
    max_any = 0.0
    max_same = 0.0
    for label in labels:
        score = box_iou(pred_box, label.get("xyxy", []))
        max_any = max(max_any, score)
        if int(label.get("class_id", -2)) == pred_class:
            max_same = max(max_same, score)
    if max_same >= 0.50:
        return "duplicate_same_class_overlap", max_any, max_same
    if max_any >= 0.50:
        return "wrong_class_overlap", max_any, max_same
    if max_any >= 0.10:
        return "fragment_or_loose_overlap", max_any, max_same
    return "off_target_on_labeled_image", max_any, max_same


def build_class_deltas(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    base = baseline.get("per_class", {})
    cand = candidate.get("per_class", {})
    rows: list[dict[str, Any]] = []
    for name in sorted(set(base) | set(cand)):
        base_row = base.get(name, {})
        cand_row = cand.get(name, {})
        rows.append(
            {
                "class_name": name,
                "baseline_gt": int(base_row.get("gt", 0) or 0),
                "candidate_gt": int(cand_row.get("gt", 0) or 0),
                "baseline_tp": int(base_row.get("tp", 0) or 0),
                "candidate_tp": int(cand_row.get("tp", 0) or 0),
                "baseline_fn": int(base_row.get("fn", 0) or 0),
                "candidate_fn": int(cand_row.get("fn", 0) or 0),
                "baseline_fp": int(base_row.get("fp", 0) or 0),
                "candidate_fp": int(cand_row.get("fp", 0) or 0),
                "fp_delta": metric_delta(base, cand, name, "fp"),
                "recall_delta": metric_delta(base, cand, name, "recall"),
                "precision_delta": metric_delta(base, cand, name, "precision"),
            }
        )
    return sorted(rows, key=lambda row: (row["fp_delta"], -row["recall_delta"]), reverse=True)


def build_source_deltas(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    base = baseline.get("per_source", {})
    cand = candidate.get("per_source", {})
    rows: list[dict[str, Any]] = []
    for name in sorted(set(base) | set(cand)):
        base_row = base.get(name, {})
        cand_row = cand.get(name, {})
        rows.append(
            {
                "source_group": name,
                "baseline_images": int(base_row.get("images", 0) or 0),
                "candidate_images": int(cand_row.get("images", 0) or 0),
                "baseline_fp": int(base_row.get("fp", 0) or 0),
                "candidate_fp": int(cand_row.get("fp", 0) or 0),
                "fp_delta": metric_delta(base, cand, name, "fp"),
                "background_fp_image_delta": metric_delta(base, cand, name, "background_images_with_fp"),
                "recall_delta": metric_delta(base, cand, name, "recall"),
                "precision_delta": metric_delta(base, cand, name, "precision"),
            }
        )
    return sorted(rows, key=lambda row: (row["fp_delta"], row["background_fp_image_delta"]), reverse=True)


def flatten_candidate_examples(
    candidate: dict[str, Any],
    class_deltas: list[dict[str, Any]],
    names: dict[int, str],
    only_classes: set[str],
) -> list[dict[str, Any]]:
    delta_by_class = {row["class_name"]: row for row in class_deltas}
    examples_by_class = candidate.get("fp_examples_by_class", {})
    rows: list[dict[str, Any]] = []
    for class_name, examples in examples_by_class.items():
        if only_classes and class_name not in only_classes:
            continue
        delta = delta_by_class.get(class_name, {})
        if not only_classes and float(delta.get("fp_delta", 0) or 0) <= 0:
            continue
        for example in examples:
            labels = example.get("labels", [])
            label_classes = [
                names.get(int(label.get("class_id", -1)), str(label.get("class_id", "")))
                for label in labels
            ]
            for prediction in example.get("false_predictions", []):
                class_id = int(prediction.get("class_id", -1))
                kind, max_iou_any_label, max_iou_same_class = fp_kind(prediction, labels)
                rows.append(
                    {
                        "image": str(example.get("image", "")),
                        "source_group": str(example.get("source_group", "")),
                        "class_name": class_name,
                        "class_id": class_id,
                        "class_from_id": names.get(class_id, str(class_id)),
                        "confidence": float(prediction.get("confidence", 0.0) or 0.0),
                        "area_ratio": float(prediction.get("area_ratio", 0.0) or 0.0),
                        "best_iou": float(prediction.get("best_iou", 0.0) or 0.0),
                        "max_iou_any_label": max_iou_any_label,
                        "max_iou_same_class": max_iou_same_class,
                        "fp_kind": kind,
                        "label_count": len(labels),
                        "label_classes": ",".join(label_classes),
                        "fp_delta_for_class": float(delta.get("fp_delta", 0) or 0),
                        "candidate_fp_for_class": int(delta.get("candidate_fp", 0) or 0),
                        "baseline_fp_for_class": int(delta.get("baseline_fp", 0) or 0),
                        "prediction": prediction,
                        "labels": labels,
                    }
                )
    rows.sort(
        key=lambda row: (
            row["fp_delta_for_class"],
            row["confidence"],
            row["area_ratio"],
        ),
        reverse=True,
    )
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    resolved = resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image",
        "source_group",
        "class_name",
        "class_id",
        "class_from_id",
        "confidence",
        "area_ratio",
        "best_iou",
        "max_iou_any_label",
        "max_iou_same_class",
        "fp_kind",
        "label_count",
        "label_classes",
        "fp_delta_for_class",
        "candidate_fp_for_class",
        "baseline_fp_for_class",
    ]
    with resolved.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def scale_box(box: list[float], source_size: tuple[int, int], target_size: tuple[int, int]) -> tuple[int, int, int, int]:
    source_w, source_h = source_size
    target_w, target_h = target_size
    x_scale = target_w / max(1, source_w)
    y_scale = target_h / max(1, source_h)
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * x_scale)),
        int(round(y1 * y_scale)),
        int(round(x2 * x_scale)),
        int(round(y2 * y_scale)),
    )


def draw_sheet(path: Path, rows: list[dict[str, Any]], names: dict[int, str], *, items: int, thumb_width: int, cols: int) -> None:
    selected = rows[:items]
    if not selected:
        return
    thumb_h = thumb_width
    caption_h = 72
    row_count = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_width, row_count * (thumb_h + caption_h)), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(selected):
        x = (index % cols) * thumb_width
        y = (index // cols) * (thumb_h + caption_h)
        image_path = resolve(str(row["image"]))
        try:
            with Image.open(image_path) as loaded:
                original = ImageOps.exif_transpose(loaded.convert("RGB"))
            source_size = original.size
            original.thumbnail((thumb_width, thumb_h), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (thumb_width, thumb_h), "white")
            paste_xy = ((thumb_width - original.width) // 2, (thumb_h - original.height) // 2)
            tile.paste(original, paste_xy)
            tile_draw = ImageDraw.Draw(tile)
            for label in row.get("labels", []):
                box = scale_box(label["xyxy"], source_size, original.size)
                box = tuple(value + offset for value, offset in zip(box, (*paste_xy, *paste_xy), strict=True))
                tile_draw.rectangle(box, outline=(30, 160, 60), width=3)
                label_name = names.get(int(label.get("class_id", -1)), str(label.get("class_id", "")))
                tile_draw.text((box[0] + 3, max(2, box[1] - 12)), label_name, fill=(30, 160, 60), font=font)
            pred = row.get("prediction", {})
            if pred.get("xyxy"):
                box = scale_box(pred["xyxy"], source_size, original.size)
                box = tuple(value + offset for value, offset in zip(box, (*paste_xy, *paste_xy), strict=True))
                tile_draw.rectangle(box, outline=(220, 35, 35), width=4)
        except Exception:
            tile = Image.new("RGB", (thumb_width, thumb_h), (80, 80, 80))
            ImageDraw.Draw(tile).text((8, 8), "missing", fill=(255, 255, 255), font=font)
        sheet.paste(tile, (x, y))
        caption = (
            f"{row['class_name']} fp+{row['fp_delta_for_class']:.0f} "
            f"conf {row['confidence']:.2f} area {row['area_ratio']:.2f}\n"
            f"{row['source_group']}\n"
            f"labels: {row['label_classes'] or 'empty'}"
        )
        draw.multiline_text((x + 4, y + thumb_h + 4), caption[:180], fill=(10, 10, 10), font=font, spacing=2)
    resolved = resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(resolved, quality=92)


def main() -> None:
    args = parse_args()
    baseline_path = resolve(args.baseline)
    candidate_path = resolve(args.candidate)
    baseline = load_json(baseline_path)
    candidate = load_json(candidate_path)
    names = class_names_from_data(str(candidate.get("data", "")))
    only_classes = parse_class_filter(args.only_class)
    class_deltas = build_class_deltas(baseline, candidate)
    source_deltas = build_source_deltas(baseline, candidate)
    review_rows = flatten_candidate_examples(candidate, class_deltas, names, only_classes)
    fp_kind_counts = Counter(str(row.get("fp_kind", "")) for row in review_rows)
    if args.sheet_out:
        draw_sheet(args.sheet_out, review_rows, names, items=args.sheet_items, thumb_width=args.thumb_width, cols=args.cols)
    write_csv(args.csv_out, review_rows)
    class_fp_delta = Counter({row["class_name"]: int(row["fp_delta"]) for row in class_deltas})
    source_fp_delta = Counter({row["source_group"]: int(row["fp_delta"]) for row in source_deltas})
    top_positive_class_fp_deltas = [
        [name, delta] for name, delta in class_fp_delta.most_common() if delta > 0
    ][:8]
    top_positive_source_fp_deltas = [
        [name, delta] for name, delta in source_fp_delta.most_common() if delta > 0
    ][:8]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "slice_name": args.slice_name,
        "baseline": repo_rel(baseline_path),
        "candidate": repo_rel(candidate_path),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "overall": {
            "baseline_fp": int(baseline.get("fp", 0) or 0),
            "candidate_fp": int(candidate.get("fp", 0) or 0),
            "fp_delta": int(candidate.get("fp", 0) or 0) - int(baseline.get("fp", 0) or 0),
            "baseline_background_images_with_fp": int(baseline.get("background_images_with_fp", 0) or 0),
            "candidate_background_images_with_fp": int(candidate.get("background_images_with_fp", 0) or 0),
            "background_fp_image_delta": int(candidate.get("background_images_with_fp", 0) or 0)
            - int(baseline.get("background_images_with_fp", 0) or 0),
            "baseline_recall": float(baseline.get("recall", 0.0) or 0.0),
            "candidate_recall": float(candidate.get("recall", 0.0) or 0.0),
            "recall_delta": float(candidate.get("recall", 0.0) or 0.0) - float(baseline.get("recall", 0.0) or 0.0),
            "baseline_precision": float(baseline.get("precision", 0.0) or 0.0),
            "candidate_precision": float(candidate.get("precision", 0.0) or 0.0),
            "precision_delta": float(candidate.get("precision", 0.0) or 0.0)
            - float(baseline.get("precision", 0.0) or 0.0),
        },
        "class_deltas": class_deltas,
        "source_deltas": source_deltas,
        "top_positive_class_fp_deltas": top_positive_class_fp_deltas,
        "top_positive_source_fp_deltas": top_positive_source_fp_deltas,
        "sampled_fp_kind_counts": fp_kind_counts.most_common(),
        "review_rows": [
            {key: value for key, value in row.items() if key not in {"prediction", "labels"}}
            for row in review_rows
        ],
        "csv_out": repo_rel(resolve(args.csv_out)),
        "sheet_out": repo_rel(resolve(args.sheet_out)) if args.sheet_out else "",
    }
    write_json(args.json_out, payload)
    print(
        f"wrote {len(review_rows)} review rows; overall fp_delta={payload['overall']['fp_delta']} "
        f"recall_delta={payload['overall']['recall_delta']:.6f}"
    )


if __name__ == "__main__":
    main()
