#!/usr/bin/env python
"""Join activation microscope gaps to positive-error review failures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microscope-dir", required=True, type=Path)
    parser.add_argument(
        "--error-review-dir",
        type=Path,
        default=None,
        help="Optional existing positive_error_review directory to reuse.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def class_error_counts(errors: list[dict[str, str]], model: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = defaultdict(lambda: {"missed_gt": 0, "wrong_class": 0, "unmatched_fp": 0, "wrong_pairs": Counter(), "fp_pred": Counter()})
    for row in errors:
        if row["model"] != model:
            continue
        error_type = row["error_type"]
        if error_type in {"missed_gt", "wrong_class"}:
            cls = row["gt_class"]
            if not cls:
                continue
            out[cls][error_type] += 1
            if error_type == "wrong_class":
                out[cls]["wrong_pairs"][f"{cls}->{row['pred_class']}"] += 1
        elif error_type == "unmatched_fp":
            cls = row["pred_class"]
            if not cls:
                continue
            out[cls]["unmatched_fp"] += 1
            out[cls]["fp_pred"][cls] += 1
    return out


def image_error_links(errors: list[dict[str, str]], model: str, image: str, class_name: str) -> dict[str, Any]:
    matched = [row for row in errors if row["model"] == model and row["image"] == image]
    relevant = [
        row
        for row in matched
        if row["gt_class"] == class_name or row["pred_class"] == class_name or row["error_type"] == "unmatched_fp"
    ]
    rows = relevant or matched
    return {
        "error_count": len(rows),
        "error_types": ";".join(f"{key}:{value}" for key, value in Counter(row["error_type"] for row in rows).most_common()),
        "gt_classes": ";".join(f"{key}:{value}" for key, value in Counter(row["gt_class"] for row in rows if row["gt_class"]).most_common()),
        "pred_classes": ";".join(f"{key}:{value}" for key, value in Counter(row["pred_class"] for row in rows if row["pred_class"]).most_common()),
        "max_confidence": max((fnum(row["confidence"]) for row in rows), default=0.0),
        "max_review_score": max((fnum(row["review_score"]) for row in rows), default=0.0),
        "review_overlays": ";".join(row["overlay"] for row in rows if row.get("overlay")),
    }


def top_counter_text(counter: Counter[str], n: int = 4) -> str:
    return ";".join(f"{key}:{value}" for key, value in counter.most_common(n))


def main() -> None:
    args = parse_args()
    microscope_dir = resolve(args.microscope_dir)
    out_dir = resolve(args.out_dir) if args.out_dir else microscope_dir / "failure_links"
    out_dir.mkdir(parents=True, exist_ok=True)

    microscope = json.loads((microscope_dir / "summary.json").read_text(encoding="utf-8"))
    error_dir = resolve(args.error_review_dir) if args.error_review_dir else microscope_dir / "positive_error_review"
    error_summary = json.loads((error_dir / "summary.json").read_text(encoding="utf-8"))
    errors = read_csv(error_dir / "errors.csv")

    class_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    for model in microscope["models"]:
        label = model["label"]
        split_key = next((key for key in error_summary["summaries"] if key.startswith(f"{label}/")), None)
        if split_key is None:
            continue
        by_class = error_summary["summaries"][split_key]["by_class"]
        error_counts = class_error_counts(errors, label)
        top_nearest_by_class: dict[str, Counter[str]] = defaultdict(Counter)
        top_gap_by_class: dict[str, float] = defaultdict(float)
        for row in model["top_uncovered_real"]:
            class_name = row["class_name"]
            top_nearest_by_class[class_name][row["nearest_synthetic_class"]] += 1
            top_gap_by_class[class_name] = max(top_gap_by_class[class_name], fnum(row["nearest_l2"]))
            link = image_error_links(errors, label, row["image"], class_name)
            image_rows.append(
                {
                    "model": label,
                    "image": row["image"],
                    "class_name": class_name,
                    "nearest_synthetic_class": row["nearest_synthetic_class"],
                    "nearest_l2": f"{fnum(row['nearest_l2']):.6f}",
                    "error_count": link["error_count"],
                    "error_types": link["error_types"],
                    "gt_classes": link["gt_classes"],
                    "pred_classes": link["pred_classes"],
                    "max_confidence": f"{link['max_confidence']:.6f}",
                    "max_review_score": f"{link['max_review_score']:.6f}",
                    "review_overlays": link["review_overlays"],
                }
            )

        gap_by_class = {row["class_name"]: row for row in model["per_class_gaps"]}
        for class_name, stats in by_class.items():
            gap = gap_by_class.get(class_name, {})
            recall = fnum(stats.get("recall_at_iou"))
            rep_gap = fnum(gap.get("real_to_synthetic_nearest_l2_mean"))
            missed = int(error_counts[class_name]["missed_gt"])
            wrong = int(error_counts[class_name]["wrong_class"])
            fp = int(error_counts[class_name]["unmatched_fp"])
            priority = rep_gap * (1.0 - recall) + 0.08 * missed + 0.15 * wrong + 0.02 * fp
            class_rows.append(
                {
                    "model": label,
                    "class_name": class_name,
                    "priority_score": f"{priority:.6f}",
                    "recall_at_iou": f"{recall:.6f}",
                    "gt": stats.get("gt", 0),
                    "tp": stats.get("tp", 0),
                    "predictions": stats.get("predictions", 0),
                    "missed_gt": missed,
                    "wrong_class": wrong,
                    "unmatched_fp_pred_class": fp,
                    "rep_gap_mean": f"{rep_gap:.6f}",
                    "rep_gap_top_uncovered_max": f"{top_gap_by_class[class_name]:.6f}",
                    "centroid_l2": f"{fnum(gap.get('centroid_l2')):.6f}",
                    "top_wrong_pairs": top_counter_text(error_counts[class_name]["wrong_pairs"]),
                    "top_nearest_synth_classes": top_counter_text(top_nearest_by_class[class_name]),
                }
            )

    class_rows.sort(key=lambda row: fnum(row["priority_score"]), reverse=True)
    image_rows.sort(key=lambda row: (int(row["error_count"]), fnum(row["nearest_l2"])), reverse=True)
    write_csv(out_dir / "class_priority.csv", class_rows)
    write_csv(out_dir / "image_failure_links.csv", image_rows)

    lines = [
        "# Activation Failure Links",
        "",
        "Priority blends representation gap, low recall, misses, wrong classes, and FP pressure. It is a triage score, not a metric.",
        "",
        "## Top Class Priorities",
        "",
        "| Model | Class | Priority | Recall | Missed | Wrong | Rep gap | Top nearest synth |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in class_rows[:24]:
        lines.append(
            f"| {row['model']} | {row['class_name']} | {float(row['priority_score']):.2f} | "
            f"{float(row['recall_at_iou']):.3f} | {row['missed_gt']} | {row['wrong_class']} | "
            f"{float(row['rep_gap_mean']):.2f} | {row['top_nearest_synth_classes']} |"
        )
    lines.extend(["", "## Top Image Links", "", "| Model | Class | Error types | Nearest synth | Gap | Image |", "| --- | --- | --- | --- | ---: | --- |"])
    for row in image_rows[:20]:
        lines.append(
            f"| {row['model']} | {row['class_name']} | {row['error_types']} | "
            f"{row['nearest_synthetic_class']} | {float(row['nearest_l2']):.2f} | `{row['image']}` |"
        )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote={repo_rel(out_dir / 'summary.md')}")


if __name__ == "__main__":
    main()
