#!/usr/bin/env python
"""Promote a reviewed real benchmark YOLO label file into the scoreable val set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


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
DEFAULT_SOURCES = ROOT / "manifests" / "real_fan_benchmark_sources.csv"
DEFAULT_TASKS = ROOT / "manifests" / "real_fan_benchmark_label_tasks.csv"
DEFAULT_QUALITY = ROOT / "manifests" / "real_fan_benchmark_label_quality.csv"
DEFAULT_LABEL_DIR = ROOT / "data" / "real_fan_benchmark" / "labels" / "val"
DEFAULT_PROMOTION_LOG = ROOT / "manifests" / "real_fan_benchmark_label_promotions.csv"
SCOREABLE_QUALITIES = {"clear", "partial_clear"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", required=True, help="Image id from manifests/real_fan_benchmark_sources.csv.")
    parser.add_argument("--source-labels", type=Path, default=None, help="Reviewed source label file. Defaults to the matching draft label.")
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR, help="Destination promoted label directory.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES, help="Benchmark source manifest to update.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS, help="Benchmark labeling task manifest to update.")
    parser.add_argument("--quality-manifest", type=Path, default=DEFAULT_QUALITY, help="Per-box quality manifest.")
    parser.add_argument("--promotion-log", type=Path, default=DEFAULT_PROMOTION_LOG, help="Append-only promotion audit CSV.")
    parser.add_argument("--preview-out", type=Path, default=None, help="Optional promoted-label preview image path.")
    parser.add_argument("--reviewed-by", default="", help="Reviewer name/id required with --confirm-reviewed.")
    parser.add_argument("--review-notes", default="", help="Short audit note for the promotion log.")
    parser.add_argument("--allow-missing-quality", action="store_true", help="Allow promotion without full scoreable quality rows.")
    parser.add_argument("--skip-preview", action="store_true", help="Do not render a promoted-label preview.")
    parser.add_argument("--confirm-reviewed", action="store_true", help="Actually copy labels and update manifests.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return resolve(path).relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolve(path))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with resolve(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output = resolve(path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "score", "keep"}


def default_source_labels(image_id: str) -> Path:
    return ROOT / "data" / "real_fan_benchmark" / "drafts" / f"{image_id}.txt"


def default_preview_path(image_id: str) -> Path:
    return ROOT / "data" / "real_fan_benchmark" / "previews" / "promoted_labels" / f"{image_id}.jpg"


def find_row(rows: list[dict[str, str]], image_id: str, manifest_name: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("image_id") == image_id]
    if not matches:
        raise SystemExit(f"{manifest_name}: no row for image_id {image_id}")
    if len(matches) > 1:
        raise SystemExit(f"{manifest_name}: duplicate rows for image_id {image_id}")
    return matches[0]


def read_label_classes(path: Path) -> list[int]:
    classes: list[int] = []
    for line_number, raw_line in enumerate(resolve(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_path(path)}:{line_number}: expected 5 YOLO detect fields, got {len(parts)}")
        try:
            class_id = int(parts[0])
            cx, cy, width, height = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise SystemExit(f"{repo_path(path)}:{line_number}: non-numeric YOLO value") from exc
        if not 0 <= class_id < len(CLASS_NAMES):
            raise SystemExit(f"{repo_path(path)}:{line_number}: class {class_id} outside 0..{len(CLASS_NAMES) - 1}")
        if not all(0.0 <= value <= 1.0 for value in [cx, cy, width, height]):
            raise SystemExit(f"{repo_path(path)}:{line_number}: normalized values must be in 0..1")
        if width <= 0.0 or height <= 0.0:
            raise SystemExit(f"{repo_path(path)}:{line_number}: width/height must be positive")
        classes.append(class_id)
    if not classes:
        raise SystemExit(f"{repo_path(path)}: no non-empty YOLO labels to promote")
    return classes


def validate_quality_rows(
    *,
    image_id: str,
    label_path: Path,
    class_ids: list[int],
    quality_manifest: Path,
    allow_missing_quality: bool,
) -> None:
    _fields, quality_rows = read_csv(quality_manifest)
    label_key = repo_path(label_path)
    matching = [
        row
        for row in quality_rows
        if row.get("image_id") == image_id and row.get("label_path", "").replace("\\", "/") == label_key
    ]
    if not matching:
        if allow_missing_quality:
            print(f"warning: no quality rows found for {label_key}")
            return
        raise SystemExit(f"No quality rows found for {label_key}; use --allow-missing-quality only after separate review.")

    by_index: dict[int, dict[str, str]] = {}
    for row in matching:
        try:
            index = int(row.get("label_index", ""))
        except ValueError as exc:
            raise SystemExit(f"{repo_path(quality_manifest)}: invalid label_index {row.get('label_index')!r}") from exc
        if index in by_index:
            raise SystemExit(f"{repo_path(quality_manifest)}: duplicate quality row for {label_key} index {index}")
        if not 0 <= index < len(class_ids):
            raise SystemExit(f"{repo_path(quality_manifest)}: quality index {index} outside 0..{len(class_ids) - 1}")
        expected_name = CLASS_NAMES[class_ids[index]]
        actual_name = row.get("class_name", "").strip()
        if actual_name and actual_name != expected_name:
            raise SystemExit(f"{repo_path(quality_manifest)}: index {index} class {actual_name} != label class {expected_name}")
        quality = row.get("quality", "").strip()
        if quality not in SCOREABLE_QUALITIES:
            raise SystemExit(f"{repo_path(quality_manifest)}: index {index} quality {quality!r} is not scoreable")
        if not truthy(row.get("count_for_score", "")):
            raise SystemExit(f"{repo_path(quality_manifest)}: index {index} is not marked count_for_score")
        by_index[index] = row

    missing = [index for index in range(len(class_ids)) if index not in by_index]
    if missing and not allow_missing_quality:
        missing_text = ", ".join(str(index) for index in missing)
        raise SystemExit(f"{repo_path(quality_manifest)}: missing quality rows for {label_key} indices {missing_text}")
    if missing:
        print(f"warning: missing quality rows for indices {', '.join(str(index) for index in missing)}")


def append_promotion_log(
    *,
    path: Path,
    image_id: str,
    source_labels: Path,
    target_labels: Path,
    label_count: int,
    reviewed_by: str,
    review_notes: str,
) -> None:
    output = resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "promoted_at_utc",
        "image_id",
        "source_label_path",
        "target_label_path",
        "source_sha256",
        "target_sha256",
        "label_count",
        "reviewed_by",
        "review_notes",
    ]
    exists = output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "promoted_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "image_id": image_id,
                "source_label_path": repo_path(source_labels),
                "target_label_path": repo_path(target_labels),
                "source_sha256": sha256(source_labels),
                "target_sha256": sha256(target_labels),
                "label_count": str(label_count),
                "reviewed_by": reviewed_by,
                "review_notes": review_notes,
            }
        )


def render_preview(image_path: Path, labels: Path, out: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/render_yolo_label_preview.py",
            "--image",
            repo_path(image_path),
            "--labels",
            repo_path(labels),
            "--out",
            repo_path(out),
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> int:
    args = parse_args()
    if args.confirm_reviewed and not args.reviewed_by.strip():
        raise SystemExit("--reviewed-by is required with --confirm-reviewed")

    source_labels = resolve(args.source_labels) if args.source_labels is not None else default_source_labels(args.image_id)
    label_dir = resolve(args.label_dir)
    target_labels = label_dir / f"{args.image_id}.txt"
    preview_out = resolve(args.preview_out) if args.preview_out is not None else default_preview_path(args.image_id)

    if not source_labels.exists():
        raise SystemExit(f"missing source labels: {repo_path(source_labels)}")
    source_fields, source_rows = read_csv(args.sources)
    task_fields, task_rows = read_csv(args.tasks)
    source_row = find_row(source_rows, args.image_id, repo_path(args.sources))
    find_row(task_rows, args.image_id, repo_path(args.tasks))
    image_path = ROOT / source_row["local_path"]
    if not image_path.exists():
        raise SystemExit(f"missing image: {repo_path(image_path)}")

    class_ids = read_label_classes(source_labels)
    validate_quality_rows(
        image_id=args.image_id,
        label_path=source_labels,
        class_ids=class_ids,
        quality_manifest=args.quality_manifest,
        allow_missing_quality=args.allow_missing_quality,
    )

    print(f"image_id: {args.image_id}")
    print(f"source_labels: {repo_path(source_labels)}")
    print(f"target_labels: {repo_path(target_labels)}")
    print(f"label_count: {len(class_ids)}")
    print(f"class_counts: {', '.join(f'{CLASS_NAMES[class_id]}={class_ids.count(class_id)}' for class_id in sorted(set(class_ids)))}")
    if not args.confirm_reviewed:
        print("dry-run: add --confirm-reviewed --reviewed-by NAME after human review to update benchmark manifests")
        return 0

    label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_labels, target_labels)
    for row in source_rows:
        if row.get("image_id") == args.image_id:
            row["label_status"] = "labeled"
    for row in task_rows:
        if row.get("image_id") == args.image_id:
            row["label_status"] = "labeled"
    write_csv(args.sources, source_fields, source_rows)
    write_csv(args.tasks, task_fields, task_rows)
    append_promotion_log(
        path=args.promotion_log,
        image_id=args.image_id,
        source_labels=source_labels,
        target_labels=target_labels,
        label_count=len(class_ids),
        reviewed_by=args.reviewed_by.strip(),
        review_notes=args.review_notes,
    )
    if not args.skip_preview:
        render_preview(image_path, target_labels, preview_out)
    subprocess.run(
        [
            sys.executable,
            "scripts/check_real_fan_benchmark.py",
            "--sources",
            str(resolve(args.sources)),
            "--tasks",
            str(resolve(args.tasks)),
            "--label-dir",
            str(label_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    print(f"promoted: {repo_path(target_labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
