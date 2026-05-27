from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CLASS_COUNT = 13
DEFAULT_SOURCES = ROOT / "manifests" / "real_fan_benchmark_sources.csv"
DEFAULT_TASKS = ROOT / "manifests" / "real_fan_benchmark_label_tasks.csv"
DEFAULT_LABEL_DIR = ROOT / "data" / "real_fan_benchmark" / "labels" / "val"
DEFAULT_DRAFT_LABEL_DIR = ROOT / "data" / "real_fan_benchmark" / "drafts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check real fan benchmark candidates and visible-region labels.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--draft-label-dir", type=Path, default=DEFAULT_DRAFT_LABEL_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def check_label_file(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    count = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}: line {line_number} has {len(parts)} fields; expected YOLO detect format")
            continue
        try:
            class_id = int(parts[0])
            cx, cy, width, height = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{path}: line {line_number} contains a non-numeric value")
            continue
        if not 0 <= class_id < CLASS_COUNT:
            errors.append(f"{path}: line {line_number} class {class_id} outside 0..{CLASS_COUNT - 1}")
        if not all(0.0 <= value <= 1.0 for value in [cx, cy, width, height]):
            errors.append(f"{path}: line {line_number} normalized values must be in 0..1")
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{path}: line {line_number} width/height must be positive")
        count += 1
    return count, errors


def main() -> None:
    args = parse_args()
    sources_path = (ROOT / args.sources).resolve() if not args.sources.is_absolute() else args.sources.resolve()
    tasks_path = (ROOT / args.tasks).resolve() if not args.tasks.is_absolute() else args.tasks.resolve()
    label_dir = (ROOT / args.label_dir).resolve() if not args.label_dir.is_absolute() else args.label_dir.resolve()
    draft_label_dir = (ROOT / args.draft_label_dir).resolve() if not args.draft_label_dir.is_absolute() else args.draft_label_dir.resolve()

    sources = read_csv(sources_path)
    tasks = read_csv(tasks_path)
    task_by_id = {row["image_id"]: row for row in tasks}
    errors: list[str] = []
    labeled = 0
    boxes = 0
    draft_labeled = 0
    draft_boxes = 0

    for row in sources:
        image_id = row["image_id"]
        local_path = ROOT / row["local_path"]
        if image_id not in task_by_id:
            errors.append(f"{image_id}: missing row in {tasks_path.relative_to(ROOT)}")
        if not local_path.exists():
            errors.append(f"{image_id}: missing image {local_path.relative_to(ROOT)}")
            continue
        try:
            width, height = image_size(local_path)
        except Exception as exc:
            errors.append(f"{image_id}: unreadable image: {exc}")
            continue
        status = row.get("label_status", "")
        task_status = task_by_id.get(image_id, {}).get("label_status", "")
        label_path = label_dir / f"{image_id}.txt"
        draft_label_path = draft_label_dir / f"{image_id}.txt"
        draft_status = ""
        if status == "labeled" or task_status == "labeled":
            labeled += 1
            if not label_path.exists():
                errors.append(f"{image_id}: marked labeled but missing {label_path.relative_to(ROOT)}")
            else:
                count, label_errors = check_label_file(label_path)
                boxes += count
                errors.extend(label_errors)
        elif label_path.exists() and label_path.stat().st_size > 0:
            errors.append(f"{image_id}: label file exists but manifest status is not labeled")
        if draft_label_path.exists() and draft_label_path.stat().st_size > 0:
            count, label_errors = check_label_file(draft_label_path)
            draft_labeled += 1
            draft_boxes += count
            draft_status = f", draft_boxes={count}"
            errors.extend(label_errors)
        print(f"{image_id}: {width}x{height}, source={row.get('benchmark_status', '')}, labels={status or task_status}{draft_status}")

    print(f"sources: {len(sources)}")
    print(f"labeled_images: {labeled}")
    print(f"labeled_boxes: {boxes}")
    print(f"draft_labeled_images: {draft_labeled}")
    print(f"draft_boxes: {draft_boxes}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("benchmark check passed")


if __name__ == "__main__":
    main()
