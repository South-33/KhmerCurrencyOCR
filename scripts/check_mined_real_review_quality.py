#!/usr/bin/env python
"""Validate mined real-review quality rows and materialize scoreable labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "runs" / "cashsnap" / "mined_real_benchmark_review_sources_latest.csv"
DEFAULT_QUALITY = ROOT / "runs" / "cashsnap" / "mined_real_benchmark_review_quality_template_latest.csv"
DEFAULT_DRAFT_LABEL_DIR = ROOT / "data" / "real_fan_benchmark" / "mined_cashsnap_v1" / "drafts"
DEFAULT_SCOREABLE_LABEL_DIR = ROOT / "data" / "real_fan_benchmark" / "mined_cashsnap_v1" / "scoreable"
DEFAULT_JSON_OUT = ROOT / "runs" / "cashsnap" / "mined_real_benchmark_review_quality_summary_latest.json"
DEFAULT_REPORT_CSV = ROOT / "runs" / "cashsnap" / "mined_real_benchmark_review_quality_summary_latest.csv"

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
SCOREABLE_QUALITIES = {"clear", "partial_clear"}
VALID_QUALITIES = SCOREABLE_QUALITIES | {"reject", "needs_review"}
TRUE_VALUES = {"1", "true", "yes", "y", "score", "keep"}
FALSE_VALUES = {"0", "false", "no", "n", "reject", "drop", ""}
REVIEW_VALUES = {"review", "needs_review"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--draft-label-dir", type=Path, default=DEFAULT_DRAFT_LABEL_DIR)
    parser.add_argument("--scoreable-label-dir", type=Path, default=DEFAULT_SCOREABLE_LABEL_DIR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--write-scoreable-labels", action="store_true")
    parser.add_argument("--min-ready-images", type=int, default=0)
    parser.add_argument("--min-ready-stress-images", type=int, default=0)
    parser.add_argument("--min-scoreable-boxes", type=int, default=0)
    parser.add_argument("--strict-reviewed", action="store_true", help="Fail if any mined label row is still needs_review.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return resolve(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolve(path))


def read_csv(path: Path) -> list[dict[str, str]]:
    with resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def count_for_score_state(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return "score"
    if normalized in REVIEW_VALUES:
        return "review"
    if normalized in FALSE_VALUES:
        return "drop"
    return "invalid"


def read_label_file(path: Path) -> tuple[list[str], list[int], list[str]]:
    lines: list[str] = []
    classes: list[int] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(resolve(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{repo_path(path)}:{line_number}: expected 5 YOLO fields, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{repo_path(path)}:{line_number}: non-numeric YOLO value")
            continue
        if not 0 <= class_id < len(CLASS_NAMES):
            errors.append(f"{repo_path(path)}:{line_number}: class {class_id} outside 0..{len(CLASS_NAMES) - 1}")
        if not all(0.0 <= value <= 1.0 for value in values):
            errors.append(f"{repo_path(path)}:{line_number}: normalized values must be in 0..1")
        if values[2] <= 0.0 or values[3] <= 0.0:
            errors.append(f"{repo_path(path)}:{line_number}: width/height must be positive")
        lines.append(line)
        classes.append(class_id)
    return lines, classes, errors


def role_is_stress(role: str) -> bool:
    return role in {"fan_stress", "dense_overlap_stress", "hand_occlusion_stress", "thin_edge_stress", "weak_class_stress"}


def join_counts(counter: Counter[str]) -> str:
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter))


def main() -> int:
    args = parse_args()
    sources = read_csv(args.sources)
    quality_rows = read_csv(args.quality)
    draft_dir = resolve(args.draft_label_dir)
    quality_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    all_quality_keys: set[tuple[str, str]] = set()
    errors: list[str] = []

    for row in quality_rows:
        image_id = row.get("image_id", "")
        label_path = row.get("label_path", "").replace("\\", "/")
        key = (image_id, label_path)
        all_quality_keys.add(key)
        quality_by_key[key].append(row)

    report_rows: list[dict[str, str]] = []
    scoreable_outputs: list[tuple[Path, list[str]]] = []
    status_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    count_state_counts: Counter[str] = Counter()
    by_role: dict[str, Counter[str]] = defaultdict(Counter)

    for source in sources:
        image_id = source.get("image_id", "")
        role = source.get("benchmark_role", "")
        draft_path = draft_dir / f"{image_id}.txt"
        expected_label_key = repo_path(draft_path)
        lines: list[str] = []
        class_ids: list[int] = []
        if not draft_path.exists():
            errors.append(f"{image_id}: missing draft labels {repo_path(draft_path)}")
        else:
            lines, class_ids, label_errors = read_label_file(draft_path)
            errors.extend(label_errors)

        key = (image_id, expected_label_key)
        rows = quality_by_key.get(key, [])
        seen_indices: set[int] = set()
        scoreable_indices: set[int] = set()
        image_quality_counts: Counter[str] = Counter()
        image_count_states: Counter[str] = Counter()
        image_has_undecided = False

        if not rows:
            errors.append(f"{image_id}: no quality rows for {expected_label_key}")

        for row in rows:
            try:
                label_index = int(row.get("label_index", ""))
            except ValueError:
                errors.append(f"{repo_path(args.quality)}: invalid label_index {row.get('label_index')!r} for {image_id}")
                continue
            if label_index in seen_indices:
                errors.append(f"{repo_path(args.quality)}: duplicate quality row for {image_id} index {label_index}")
                continue
            seen_indices.add(label_index)
            if not 0 <= label_index < len(class_ids):
                errors.append(f"{repo_path(args.quality)}: {image_id} index {label_index} outside 0..{len(class_ids) - 1}")
                continue

            expected_class = CLASS_NAMES[class_ids[label_index]]
            actual_class = row.get("class_name", "").strip()
            if actual_class and actual_class != expected_class:
                errors.append(f"{repo_path(args.quality)}: {image_id} index {label_index} class {actual_class} != {expected_class}")

            quality = row.get("quality", "").strip()
            count_state = count_for_score_state(row.get("count_for_score", ""))
            image_quality_counts[quality] += 1
            image_count_states[count_state] += 1
            quality_counts[quality] += 1
            count_state_counts[count_state] += 1

            if quality not in VALID_QUALITIES:
                errors.append(f"{repo_path(args.quality)}: {image_id} index {label_index} invalid quality {quality!r}")
                continue
            if count_state == "invalid":
                errors.append(f"{repo_path(args.quality)}: {image_id} index {label_index} invalid count_for_score {row.get('count_for_score')!r}")
            if quality in SCOREABLE_QUALITIES and count_state != "score":
                errors.append(f"{repo_path(args.quality)}: {image_id} index {label_index} scoreable quality requires count_for_score=score/keep/yes")
            if quality == "reject" and count_state == "score":
                errors.append(f"{repo_path(args.quality)}: {image_id} index {label_index} reject row cannot count_for_score")
            if quality == "needs_review":
                image_has_undecided = True
            if count_state == "review":
                image_has_undecided = True
            if quality in SCOREABLE_QUALITIES and count_state == "score":
                scoreable_indices.add(label_index)

        missing_indices = set(range(len(class_ids))) - seen_indices
        if missing_indices:
            image_has_undecided = True
            missing = ",".join(str(index) for index in sorted(missing_indices))
            errors.append(f"{image_id}: missing quality rows for label indices {missing}")

        all_decided = bool(lines) and not image_has_undecided and not missing_indices
        ready_scoreable = all_decided and bool(scoreable_indices)
        fully_scoreable = all_decided and len(scoreable_indices) == len(lines)
        rejected_only = all_decided and not scoreable_indices
        if ready_scoreable:
            status = "ready_scoreable"
        elif rejected_only:
            status = "decided_rejected_only"
        elif image_has_undecided or missing_indices:
            status = "needs_review"
        else:
            status = "invalid_or_empty"

        status_counts[status] += 1
        by_role[role][status] += 1
        by_role[role]["images"] += 1
        by_role[role]["draft_boxes"] += len(lines)
        by_role[role]["scoreable_boxes"] += len(scoreable_indices)

        scoreable_path = resolve(args.scoreable_label_dir) / f"{image_id}.scoreable.txt"
        if ready_scoreable:
            kept_lines = [line for index, line in enumerate(lines) if index in scoreable_indices]
            scoreable_outputs.append((scoreable_path, kept_lines))

        report_rows.append(
            {
                "image_id": image_id,
                "benchmark_role": role,
                "status": status,
                "draft_label_path": expected_label_key,
                "draft_boxes": str(len(lines)),
                "quality_rows": str(len(rows)),
                "scoreable_boxes": str(len(scoreable_indices)),
                "fully_scoreable": str(fully_scoreable).lower(),
                "ready_scoreable": str(ready_scoreable).lower(),
                "rejected_only": str(rejected_only).lower(),
                "quality_counts": join_counts(image_quality_counts),
                "count_for_score_states": join_counts(image_count_states),
                "scoreable_label_path": repo_path(scoreable_path) if ready_scoreable else "",
            }
        )

    source_keys = {
        (row.get("image_id", ""), repo_path(draft_dir / f"{row.get('image_id', '')}.txt"))
        for row in sources
    }
    for image_id, label_path in sorted(all_quality_keys - source_keys):
        errors.append(f"{repo_path(args.quality)}: quality row has no source/draft match: {image_id} {label_path}")

    ready_images = [row["image_id"] for row in report_rows if row["status"] == "ready_scoreable"]
    ready_stress_images = [
        row["image_id"]
        for row in report_rows
        if row["status"] == "ready_scoreable" and role_is_stress(row["benchmark_role"])
    ]
    scoreable_boxes = sum(int(row["scoreable_boxes"]) for row in report_rows)
    summary = {
        "sources": repo_path(args.sources),
        "quality": repo_path(args.quality),
        "draft_label_dir": repo_path(args.draft_label_dir),
        "scoreable_label_dir": repo_path(args.scoreable_label_dir),
        "images": len(sources),
        "draft_boxes": sum(int(row["draft_boxes"]) for row in report_rows),
        "quality_rows": len(quality_rows),
        "scoreable_boxes": scoreable_boxes,
        "ready_scoreable_images": len(ready_images),
        "ready_scoreable_image_ids": ready_images,
        "ready_stress_images": len(ready_stress_images),
        "ready_stress_image_ids": ready_stress_images,
        "status_counts": dict(sorted(status_counts.items())),
        "quality_counts": dict(sorted(quality_counts.items())),
        "count_for_score_states": dict(sorted(count_state_counts.items())),
        "by_role": {role: dict(sorted(counts.items())) for role, counts in sorted(by_role.items())},
        "write_scoreable_labels": bool(args.write_scoreable_labels),
        "scoreable_labels_written": 0,
        "errors": errors,
    }

    if args.strict_reviewed and status_counts.get("needs_review", 0):
        errors.append(f"{status_counts['needs_review']} image(s) still need review")
    if len(ready_images) < args.min_ready_images:
        errors.append(f"ready_scoreable_images {len(ready_images)} < required {args.min_ready_images}")
    if len(ready_stress_images) < args.min_ready_stress_images:
        errors.append(f"ready_stress_images {len(ready_stress_images)} < required {args.min_ready_stress_images}")
    if scoreable_boxes < args.min_scoreable_boxes:
        errors.append(f"scoreable_boxes {scoreable_boxes} < required {args.min_scoreable_boxes}")

    if not errors and args.write_scoreable_labels:
        for output, lines in scoreable_outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary["scoreable_labels_written"] = len(scoreable_outputs)

    write_csv(args.report_csv, report_rows)
    write_json(args.json_out, summary)
    print(
        "mined_real_review_quality "
        f"images={len(sources)} ready={len(ready_images)} "
        f"ready_stress={len(ready_stress_images)} scoreable_boxes={scoreable_boxes} "
        f"errors={len(errors)}"
    )
    print(f"wrote_report={repo_path(args.report_csv)}")
    print(f"wrote_json={repo_path(args.json_out)}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
