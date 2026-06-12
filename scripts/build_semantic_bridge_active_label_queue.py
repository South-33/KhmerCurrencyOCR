#!/usr/bin/env python
"""Build a class-balanced review queue from semantic-bridge empty-label rows."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKETS = "suspect_unlabeled_target,currency_review"
DEFAULT_WEAK_CLASSES = "KHR_50000,USD_100,USD_50,USD_5,USD_20,KHR_20000,KHR_10000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bucket", default=DEFAULT_BUCKETS, help="Comma-separated buckets to include.")
    parser.add_argument("--weak-class", default=DEFAULT_WEAK_CLASSES, help="Comma-separated classes to prioritize.")
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--max-total", type=int, default=160)
    parser.add_argument("--max-per-source-group", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--sheet-items", type=int, default=80)
    parser.add_argument("--thumb-width", type=int, default=280)
    parser.add_argument("--cols", type=int, default=4)
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_csv_tokens(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    resolved = resolve(path)
    with resolved.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row = dict(row)
            row["manifest"] = repo_rel(resolved)
            rows.append(row)
    return rows


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def candidate_class(row: dict[str, Any]) -> str:
    return str(row.get("teacher_top_class") or row.get("student_top_class") or "unknown")


def score_row(row: dict[str, Any], weak_classes: set[str]) -> float:
    teacher_class = str(row.get("teacher_top_class", ""))
    student_class = str(row.get("student_top_class", ""))
    teacher_conf = as_float(row.get("teacher_top_conf"))
    student_conf = as_float(row.get("student_top_conf"))
    score = teacher_conf
    if teacher_class in weak_classes:
        score += 0.35
    if student_class and teacher_class and student_class != teacher_class:
        score += 0.18
    if str(row.get("bucket", "")) == "suspect_unlabeled_target":
        score += 0.12
    score += min(student_conf, 1.0) * 0.04
    return score


def choose_rows(
    rows: list[dict[str, Any]],
    *,
    per_class: int,
    max_total: int,
    max_per_source_group: int,
    weak_classes: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row["candidate_class"] = candidate_class(row)
        row["priority_score"] = round(score_row(row, weak_classes), 6)
        by_class[row["candidate_class"]].append(row)

    class_order = sorted(
        by_class,
        key=lambda name: (0 if name in weak_classes else 1, name),
    )
    chosen: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    seen_images: set[str] = set()
    for class_name in class_order:
        pool = list(by_class[class_name])
        rng.shuffle(pool)
        pool.sort(key=lambda row: float(row["priority_score"]), reverse=True)
        taken = 0
        for row in pool:
            image = str(row.get("image", ""))
            source_group = str(row.get("source_group", ""))
            if image in seen_images:
                continue
            if max_per_source_group > 0 and source_counts[source_group] >= max_per_source_group:
                continue
            chosen.append(row)
            seen_images.add(image)
            source_counts[source_group] += 1
            taken += 1
            if taken >= per_class or len(chosen) >= max_total:
                break
        if len(chosen) >= max_total:
            break

    if len(chosen) < max_total:
        remainder = [row for row in rows if str(row.get("image", "")) not in seen_images]
        rng.shuffle(remainder)
        remainder.sort(key=lambda row: float(row.get("priority_score", 0.0)), reverse=True)
        for row in remainder:
            source_group = str(row.get("source_group", ""))
            if max_per_source_group > 0 and source_counts[source_group] >= max_per_source_group:
                continue
            chosen.append(row)
            seen_images.add(str(row.get("image", "")))
            source_counts[source_group] += 1
            if len(chosen) >= max_total:
                break

    for index, row in enumerate(chosen, start=1):
        row["queue_rank"] = index
    return chosen


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "queue_rank",
        "image",
        "label",
        "bucket",
        "source_group",
        "candidate_class",
        "priority_score",
        "teacher_detections",
        "teacher_top_class",
        "teacher_top_conf",
        "student_detections",
        "student_top_class",
        "student_top_conf",
        "manifest",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_review_template(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "queue_rank",
        "image",
        "bucket",
        "candidate_class",
        "review_decision",
        "true_class",
        "needs_box",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "queue_rank": row["queue_rank"],
                    "image": row["image"],
                    "bucket": row["bucket"],
                    "candidate_class": row["candidate_class"],
                    "review_decision": "",
                    "true_class": "",
                    "needs_box": "",
                    "notes": "",
                }
            )


def draw_sheet(rows: list[dict[str, Any]], out_path: Path, *, items: int, thumb_width: int, cols: int) -> None:
    selected = rows[:items]
    if not selected:
        return
    thumb_h = int(thumb_width * 0.78)
    caption_h = 54
    cols = max(1, min(cols, len(selected)))
    sheet_rows = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_width, sheet_rows * (thumb_h + caption_h)), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        image_path = resolve(str(row["image"]))
        with Image.open(image_path).convert("RGB") as image:
            thumb = ImageOps.contain(image, (thumb_width, thumb_h), Image.Resampling.LANCZOS)
        x = (index % cols) * thumb_width
        y = (index // cols) * (thumb_h + caption_h)
        sheet.paste(thumb, (x + (thumb_width - thumb.width) // 2, y))
        caption = (
            f"#{row['queue_rank']} {row['bucket']} {row['candidate_class']} "
            f"T:{row.get('teacher_top_class', '')}@{row.get('teacher_top_conf', '')} "
            f"S:{row.get('student_top_class', '')}@{row.get('student_top_conf', '')}"
        )
        draw.text((x + 5, y + thumb_h + 5), caption[:64], fill=(0, 0, 0))
        draw.text((x + 5, y + thumb_h + 24), Path(str(row["image"])).name[:54], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> int:
    args = parse_args()
    buckets = set(parse_csv_tokens(args.bucket))
    weak_classes = set(parse_csv_tokens(args.weak_class))
    rows: list[dict[str, Any]] = []
    for manifest in args.manifest:
        rows.extend(read_manifest(manifest))
    rows = [row for row in rows if str(row.get("bucket", "")) in buckets]
    if not rows:
        raise SystemExit("no rows selected for active label queue")

    chosen = choose_rows(
        rows,
        per_class=max(1, args.per_class),
        max_total=max(1, args.max_total),
        max_per_source_group=args.max_per_source_group,
        weak_classes=weak_classes,
        seed=args.seed,
    )
    out_dir = resolve(args.out_dir)
    queue_csv = out_dir / "queue.csv"
    review_template_csv = out_dir / "review_template.csv"
    sheet = out_dir / "queue.jpg"
    summary_path = out_dir / "summary.json"
    write_csv(queue_csv, chosen)
    write_review_template(review_template_csv, chosen)
    draw_sheet(chosen, sheet, items=args.sheet_items, thumb_width=args.thumb_width, cols=args.cols)
    summary = {
        "schema": "cashsnap_semantic_bridge_active_label_queue_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifests": [repo_rel(resolve(path)) for path in args.manifest],
        "buckets": sorted(buckets),
        "weak_classes": sorted(weak_classes),
        "available_rows": len(rows),
        "selected_rows": len(chosen),
        "per_class": args.per_class,
        "max_total": args.max_total,
        "max_per_source_group": args.max_per_source_group,
        "queue_csv": repo_rel(queue_csv),
        "review_template_csv": repo_rel(review_template_csv),
        "sheet": repo_rel(sheet),
        "bucket_counts": dict(sorted(Counter(str(row.get("bucket", "")) for row in chosen).items())),
        "candidate_class_counts": dict(sorted(Counter(str(row.get("candidate_class", "")) for row in chosen).items())),
        "source_group_counts": dict(sorted(Counter(str(row.get("source_group", "")) for row in chosen).items())),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
