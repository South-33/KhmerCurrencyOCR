#!/usr/bin/env python
"""Build a review queue for proposal-gate exact count/value failures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
USD_VALUES = {
    "USD_1": 1.0,
    "USD_5": 5.0,
    "USD_10": 10.0,
    "USD_20": 20.0,
    "USD_50": 50.0,
    "USD_100": 100.0,
}
KHR_VALUES = {
    "KHR_500": 500.0,
    "KHR_1000": 1000.0,
    "KHR_2000": 2000.0,
    "KHR_5000": 5000.0,
    "KHR_10000": 10000.0,
    "KHR_20000": 20000.0,
    "KHR_50000": 50000.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-image-json", type=Path, required=True)
    parser.add_argument("--proposal-json", type=Path, required=True)
    parser.add_argument("--stage", default="post_gate")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--sheet-out", type=Path, default=None)
    parser.add_argument("--sheet-items", type=int, default=48)
    parser.add_argument("--thumb-width", type=int, default=260)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--fullsize-dir", type=Path, default=None)
    parser.add_argument("--fullsize-items", type=int, default=24)
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return data


def names_from_proposals(data: dict[str, Any]) -> dict[int, str]:
    names = data.get("names") or {}
    return {int(key): str(value) for key, value in names.items()}


def class_counts(items: list[dict[str, Any]], names: dict[int, str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in items:
        counts[names.get(int(item["class_id"]), str(item["class_id"]))] += 1
    return counts


def value_totals(counts: Counter[str]) -> tuple[float, float]:
    usd = sum(USD_VALUES.get(name, 0.0) * count for name, count in counts.items())
    khr = sum(KHR_VALUES.get(name, 0.0) * count for name, count in counts.items())
    return usd, khr


def compact_counts(counts: Counter[str]) -> str:
    return ",".join(f"{name}x{count}" for name, count in sorted(counts.items())) or "-"


def failure_kind(row: dict[str, Any]) -> str:
    labels = int(row["labels"])
    predictions = int(row["predictions"])
    fn = int(row["fn"])
    fp = int(row["fp"])
    if labels == 0 and predictions > 0:
        return "false_positive_on_empty"
    if labels > 0 and predictions == 0:
        return "missed_all"
    if fn > 0 and fp > 0:
        return "miss_plus_extra_or_wrong"
    if fn > 0:
        return "missed_value"
    if fp > 0:
        return "extra_value"
    return "value_mismatch"


def text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(255, 255, 255))
    draw.text((x, y), text, fill=fill, font=font)


def draw_overlay(
    image_path: Path,
    proposal_row: dict[str, Any],
    queue_row: dict[str, Any],
    names: dict[int, str],
    *,
    thumb_width: int | None = None,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if thumb_width and image.width > thumb_width:
        ratio = thumb_width / image.width
        image = image.resize((thumb_width, max(1, int(round(image.height * ratio)))), Image.Resampling.LANCZOS)
    scale_x = image.width / Image.open(image_path).width
    scale_y = image.height / Image.open(image_path).height
    draw = ImageDraw.Draw(image)

    def scaled(box: list[float]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y)),
        )

    for label in proposal_row.get("labels", []):
        box = scaled(label["xyxy"])
        name = names.get(int(label["class_id"]), str(label["class_id"]))
        draw.rectangle(box, outline=(0, 170, 70), width=3)
        text_box(draw, (box[0] + 2, max(0, box[1] + 2)), f"GT {name}", (0, 120, 40))

    for pred in proposal_row.get("predictions", []):
        box = scaled(pred["xyxy"])
        name = names.get(int(pred["class_id"]), str(pred["class_id"]))
        conf = float(pred.get("confidence", 0.0))
        gate = pred.get("gate_class", "")
        gate_conf = float(pred.get("gate_conf", 0.0))
        draw.rectangle(box, outline=(220, 40, 40), width=3)
        text_box(
            draw,
            (box[0] + 2, min(image.height - 12, max(0, box[1] + 14))),
            f"P {name} {conf:.2f} {gate}:{gate_conf:.2f}",
            (170, 30, 30),
        )

    title = (
        f"{queue_row['failure_kind']} | gt ${queue_row['gt_usd']:.0f}/KHR {queue_row['gt_khr']:.0f} "
        f"pred ${queue_row['pred_usd']:.0f}/KHR {queue_row['pred_khr']:.0f}"
    )
    text_box(draw, (4, 4), title, (20, 20, 20))
    return image


def safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:120]


def make_sheet(images: list[Image.Image], out_path: Path, cols: int) -> None:
    if not images:
        return
    cols = max(1, cols)
    rows = math.ceil(len(images) / cols)
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (245, 245, 245))
    for index, image in enumerate(images):
        x = (index % cols) * cell_w
        y = (index // cols) * cell_h
        sheet.paste(image, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> int:
    args = parse_args()
    per_image = read_json(args.per_image_json)
    proposals = read_json(args.proposal_json)
    names = names_from_proposals(proposals)

    stage_rows = per_image.get("stages", {}).get(args.stage)
    if not isinstance(stage_rows, list):
        raise SystemExit(f"{args.per_image_json}: missing stages.{args.stage}")
    proposal_by_image = {str(row["image"]): row for row in proposals.get("rows", [])}

    queue_rows: list[dict[str, Any]] = []
    for row in stage_rows:
        if row.get("exact_value") is True:
            continue
        image = str(row["image"])
        proposal_row = proposal_by_image.get(image)
        if proposal_row is None:
            raise SystemExit(f"missing proposal row for {image}")
        label_counts = class_counts(proposal_row.get("labels", []), names)
        pred_counts = class_counts(proposal_row.get("predictions", []), names)
        gt_usd, gt_khr = value_totals(label_counts)
        pred_usd, pred_khr = value_totals(pred_counts)
        out_row = {
            "image": image,
            "source_group": row.get("source_group", proposal_row.get("source_group", "")),
            "failure_kind": failure_kind(row),
            "labels": int(row["labels"]),
            "predictions": int(row["predictions"]),
            "tp": int(row["tp"]),
            "fp": int(row["fp"]),
            "fn": int(row["fn"]),
            "count_error": int(row["count_error"]),
            "gt_usd": gt_usd,
            "pred_usd": pred_usd,
            "usd_error": pred_usd - gt_usd,
            "gt_khr": gt_khr,
            "pred_khr": pred_khr,
            "khr_error": pred_khr - gt_khr,
            "label_classes": compact_counts(label_counts),
            "pred_classes": compact_counts(pred_counts),
        }
        queue_rows.append(out_row)

    queue_rows.sort(
        key=lambda row: (
            abs(float(row["usd_error"])) * 4000.0 + abs(float(row["khr_error"])),
            abs(int(row["count_error"])),
            row["source_group"],
        ),
        reverse=True,
    )

    json_out = resolve(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "cashsnap_proposal_gate_exact_error_queue_v1",
        "per_image_json": repo_rel(resolve(args.per_image_json)),
        "proposal_json": repo_rel(resolve(args.proposal_json)),
        "stage": args.stage,
        "rows": queue_rows,
        "summary": {
            "error_images": len(queue_rows),
            "by_failure_kind": dict(Counter(row["failure_kind"] for row in queue_rows).most_common()),
            "by_source_group": dict(Counter(row["source_group"] for row in queue_rows).most_common()),
            "by_label_classes": dict(Counter(row["label_classes"] for row in queue_rows).most_common()),
            "by_pred_classes": dict(Counter(row["pred_classes"] for row in queue_rows).most_common()),
        },
    }
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_out = resolve(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(queue_rows[0].keys()) if queue_rows else [
            "image",
            "source_group",
            "failure_kind",
            "labels",
            "predictions",
            "tp",
            "fp",
            "fn",
            "count_error",
            "gt_usd",
            "pred_usd",
            "usd_error",
            "gt_khr",
            "pred_khr",
            "khr_error",
            "label_classes",
            "pred_classes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(queue_rows)

    if args.fullsize_dir:
        fullsize_dir = resolve(args.fullsize_dir)
        fullsize_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(queue_rows[: max(0, args.fullsize_items)], start=1):
            proposal_row = proposal_by_image[str(row["image"])]
            out_path = fullsize_dir / f"{index:02d}_{row['failure_kind']}_{safe_stem(Path(row['image']).stem)}.jpg"
            draw_overlay(resolve(row["image"]), proposal_row, row, names).save(out_path, quality=92)

    if args.sheet_out:
        thumbs: list[Image.Image] = []
        for row in queue_rows[: max(0, args.sheet_items)]:
            proposal_row = proposal_by_image[str(row["image"])]
            thumbs.append(draw_overlay(resolve(row["image"]), proposal_row, row, names, thumb_width=args.thumb_width))
        make_sheet(thumbs, resolve(args.sheet_out), args.cols)

    print(f"exact_error_queue={len(queue_rows)} json={repo_rel(json_out)} csv={repo_rel(csv_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
