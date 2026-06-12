#!/usr/bin/env python
"""Build a review queue of candidate FPs not matched by baseline FPs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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
    parser.add_argument("--baseline-json", required=True, type=Path)
    parser.add_argument("--candidate-json", required=True, type=Path)
    parser.add_argument("--baseline-fp-jsonl", required=True, type=Path)
    parser.add_argument("--candidate-fp-jsonl", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--csv-out", required=True, type=Path)
    parser.add_argument("--sheet-out", type=Path, default=None)
    parser.add_argument("--sheet-items", type=int, default=80)
    parser.add_argument("--thumb-width", type=int, default=280)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--fullsize-dir", type=Path, default=None)
    parser.add_argument("--fullsize-items", type=int, default=24)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with resolve(path).open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def class_names(data_path: Path) -> dict[int, str]:
    payload = yaml.safe_load(resolve(data_path).read_text(encoding="utf-8"))
    raw_names = payload.get("names", {}) if isinstance(payload, dict) else {}
    if isinstance(raw_names, list):
        return {idx: str(name) for idx, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        return {int(idx): str(name) for idx, name in raw_names.items()}
    return {}


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


def class_delta_rows(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    base = baseline.get("per_class", {})
    cand = candidate.get("per_class", {})
    for class_name in sorted(set(base) | set(cand)):
        base_row = base.get(class_name, {})
        cand_row = cand.get(class_name, {})
        rows[class_name] = {
            "baseline_fp": int(base_row.get("fp", 0) or 0),
            "candidate_fp": int(cand_row.get("fp", 0) or 0),
            "fp_delta": int(cand_row.get("fp", 0) or 0) - int(base_row.get("fp", 0) or 0),
            "baseline_tp": int(base_row.get("tp", 0) or 0),
            "candidate_tp": int(cand_row.get("tp", 0) or 0),
            "tp_delta": int(cand_row.get("tp", 0) or 0) - int(base_row.get("tp", 0) or 0),
        }
    return rows


def find_candidate_only(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    match_iou: float,
) -> list[dict[str, Any]]:
    baseline_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in baseline_rows:
        key = (str(row.get("image", "")), int(row.get("class_id", -1)))
        baseline_by_key.setdefault(key, []).append(row)

    used_by_key: dict[tuple[str, int], set[int]] = {}
    extras: list[dict[str, Any]] = []
    for candidate in sorted(candidate_rows, key=lambda row: float(row.get("confidence", 0.0) or 0.0), reverse=True):
        key = (str(candidate.get("image", "")), int(candidate.get("class_id", -1)))
        used = used_by_key.setdefault(key, set())
        best_index = -1
        best_iou = 0.0
        for index, baseline in enumerate(baseline_by_key.get(key, [])):
            if index in used:
                continue
            score = box_iou(candidate.get("xyxy", []), baseline.get("xyxy", []))
            if score > best_iou:
                best_iou = score
                best_index = index
        if best_index >= 0 and best_iou >= match_iou:
            used.add(best_index)
            continue
        enriched = dict(candidate)
        enriched["baseline_match_iou"] = best_iou
        extras.append(enriched)
    return extras


def draw_overlay(
    image_path: Path,
    row: dict[str, Any],
    names: dict[int, str],
    out_path: Path,
    thumb_width: int | None = None,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for label in row.get("labels", []):
        box = [float(value) for value in label.get("xyxy", [])]
        draw.rectangle(box, outline=(0, 210, 0), width=4)
        class_name = names.get(int(label.get("class_id", -1)), str(label.get("class_id", "")))
        draw.text((box[0] + 4, max(0, box[1] - 20)), f"GT {class_name}", fill=(0, 160, 0), font=font)

    box = [float(value) for value in row.get("xyxy", [])]
    draw.rectangle(box, outline=(255, 0, 0), width=4)
    draw.text(
        (box[0] + 4, min(image.height - 20, box[1] + 4)),
        f"EXTRA FP {row.get('class_name')} {float(row.get('confidence', 0.0)):.3f}",
        fill=(255, 0, 0),
        font=font,
    )

    caption = (
        f"{row.get('source_group')} {row.get('class_name')} {row.get('fp_kind')} "
        f"conf={float(row.get('confidence', 0.0)):.3f}"
    )
    band = Image.new("RGB", (image.width, image.height + 34), (255, 255, 255))
    band.paste(image, (0, 34))
    band_draw = ImageDraw.Draw(band)
    band_draw.text((6, 8), caption[:140], fill=(0, 0, 0), font=font)

    if thumb_width:
        band = ImageOps.contain(band, (thumb_width, thumb_width + 80))
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        band.save(out_path, quality=95)
    return band


def write_sheet(rows: list[dict[str, Any]], names: dict[int, str], out_path: Path, limit: int, thumb_width: int, cols: int) -> None:
    thumbs: list[Image.Image] = []
    for row in rows[:limit]:
        image_path = resolve(str(row["image"]))
        thumbs.append(draw_overlay(image_path, row, names, out_path, thumb_width=thumb_width))
    if not thumbs:
        return
    cell_w = max(thumb.width for thumb in thumbs)
    cell_h = max(thumb.height for thumb in thumbs)
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows_count * cell_h), (245, 245, 245))
    for index, thumb in enumerate(thumbs):
        x = (index % cols) * cell_w
        y = (index // cols) * cell_h
        sheet.paste(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=95)


def counter_rows(counter: Counter[tuple[str, str]], key_names: tuple[str, str]) -> list[dict[str, Any]]:
    first_key, second_key = key_names
    return [
        {first_key: first, second_key: second, "count": count}
        for (first, second), count in counter.most_common()
    ]


def main() -> None:
    args = parse_args()
    names = class_names(args.data)
    baseline = load_json(args.baseline_json)
    candidate = load_json(args.candidate_json)
    deltas = class_delta_rows(baseline, candidate)
    extras = find_candidate_only(
        load_jsonl(args.baseline_fp_jsonl),
        load_jsonl(args.candidate_fp_jsonl),
        args.match_iou,
    )

    rows: list[dict[str, Any]] = []
    for row in extras:
        labels = row.get("labels", [])
        kind, max_iou_any, max_iou_same = fp_kind(row, labels)
        class_name = str(row.get("class_name", names.get(int(row.get("class_id", -1)), row.get("class_id", ""))))
        delta = deltas.get(class_name, {})
        rows.append(
            {
                "image": row.get("image", ""),
                "source_group": row.get("source_group", ""),
                "class_name": class_name,
                "class_id": int(row.get("class_id", -1)),
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "area_ratio": float(row.get("area_ratio", 0.0) or 0.0),
                "xyxy": row.get("xyxy", []),
                "fp_kind": kind,
                "max_iou_any_label": max_iou_any,
                "max_iou_same_class": max_iou_same,
                "baseline_match_iou": float(row.get("baseline_match_iou", 0.0) or 0.0),
                "label_count": len(labels),
                "label_classes": ",".join(names.get(int(label.get("class_id", -1)), str(label.get("class_id", ""))) for label in labels),
                "labels": labels,
                "fp_delta_for_class": int(delta.get("fp_delta", 0) or 0),
                "tp_delta_for_class": int(delta.get("tp_delta", 0) or 0),
                "candidate_fp_for_class": int(delta.get("candidate_fp", 0) or 0),
                "baseline_fp_for_class": int(delta.get("baseline_fp", 0) or 0),
            }
        )

    rows.sort(key=lambda row: (row["fp_delta_for_class"], row["confidence"]), reverse=True)

    summary = {
        "baseline_json": repo_rel(resolve(args.baseline_json)),
        "candidate_json": repo_rel(resolve(args.candidate_json)),
        "match_iou": args.match_iou,
        "candidate_only_fp_count": len(rows),
        "by_source_kind": counter_rows(Counter((row["source_group"], row["fp_kind"]) for row in rows), ("source_group", "fp_kind")),
        "by_class_kind": counter_rows(Counter((row["class_name"], row["fp_kind"]) for row in rows), ("class_name", "fp_kind")),
        "by_source_class": counter_rows(Counter((row["source_group"], row["class_name"]) for row in rows), ("source_group", "class_name")),
        "rows": rows,
    }

    json_out = resolve(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_fields = [
        "image",
        "source_group",
        "class_name",
        "class_id",
        "confidence",
        "area_ratio",
        "fp_kind",
        "max_iou_any_label",
        "max_iou_same_class",
        "baseline_match_iou",
        "label_count",
        "label_classes",
        "fp_delta_for_class",
        "tp_delta_for_class",
        "candidate_fp_for_class",
        "baseline_fp_for_class",
    ]
    csv_out = resolve(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in csv_fields})

    if args.fullsize_dir:
        fullsize_dir = resolve(args.fullsize_dir)
        fullsize_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows[: args.fullsize_items], start=1):
            safe_name = Path(str(row["image"])).stem
            out_path = fullsize_dir / f"{index:02d}_{row['source_group']}_{row['class_name']}_{safe_name}.jpg"
            draw_overlay(resolve(str(row["image"])), row, names, out_path)

    if args.sheet_out:
        write_sheet(rows, names, resolve(args.sheet_out), args.sheet_items, args.thumb_width, args.cols)

    print(
        f"candidate_only_fp={len(rows)} json={repo_rel(json_out)} csv={repo_rel(csv_out)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
