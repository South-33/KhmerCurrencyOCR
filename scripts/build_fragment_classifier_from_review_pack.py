from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ImageFolder classifier dataset from a reviewed crop pack.")
    parser.add_argument("--manifest", nargs="+", required=True, help="One or more review-pack manifest.csv files.")
    parser.add_argument("--out", required=True, help="Output ImageFolder dataset under data/.")
    parser.add_argument(
        "--include-values",
        default="1,true,yes,y,keep",
        help="Comma-separated review_include values treated as selected.",
    )
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Use all rows when review_include is blank. Useful only for diagnostics.",
    )
    parser.add_argument("--classes", default="", help="Optional comma-separated review classes to keep.")
    parser.add_argument("--ensure-classes", default="", help="Optional comma-separated class folders to create in every split.")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def split_list(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,\s]+", value) if item.strip()]


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (ROOT / "data").resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise SystemExit(f"Refusing to clean outside {allowed_root}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def selected(row: dict[str, str], include_values: set[str], include_unreviewed: bool) -> bool:
    value = row.get("review_include", "").strip().lower()
    if not value:
        return include_unreviewed
    return value in include_values


def row_class_name(row: dict[str, str]) -> str:
    for key in ["review_class", "class_name", "fragment_class", "detector_class"]:
        value = row.get(key, "").strip()
        if value:
            return value
    return ""


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out)
    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    include_values = {value.lower() for value in split_list(args.include_values)}
    keep_classes = set(split_list(args.classes))
    ensure_classes = set(split_list(args.ensure_classes))

    rows: list[dict[str, str]] = []
    for manifest_text in args.manifest:
        manifest = resolve(manifest_text)
        for row in csv.DictReader(manifest.open("r", newline="", encoding="utf-8")):
            row["source_manifest"] = str(manifest.relative_to(ROOT))
            rows.append(row)
    written_rows: list[dict[str, str]] = []
    counters: dict[tuple[str, str], int] = {}
    skipped_missing = 0
    skipped_unselected = 0
    skipped_no_class = 0
    for row in rows:
        if not selected(row, include_values, args.include_unreviewed):
            skipped_unselected += 1
            continue
        source = resolve(row["crop_path"])
        if source.suffix.lower() not in IMAGE_SUFFIXES or not source.exists():
            skipped_missing += 1
            continue
        split = row.get("split", "train").strip() or "train"
        class_name = row_class_name(row)
        if not class_name:
            skipped_no_class += 1
            continue
        if keep_classes and class_name not in keep_classes:
            continue
        key = (split, class_name)
        index = counters.get(key, 0)
        counters[key] = index + 1
        target = out_dir / split / class_name / f"{class_name}_{index:05d}_{source.stem}{source.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        written = {
            "split": split,
            "class_name": class_name,
            "source_manifest": row.get("source_manifest", ""),
            "source_crop": str(source.relative_to(ROOT)),
            "image_path": str(target.relative_to(ROOT)),
            "review_notes": row.get("review_notes", ""),
        }
        written_rows.append(written)

    for split in ["train", "val", "test"]:
        for class_name in sorted(ensure_classes | keep_classes):
            (out_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    if written_rows:
        with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(written_rows[0].keys()))
            writer.writeheader()
            writer.writerows(written_rows)
    print(f"wrote {len(written_rows)} reviewed crops to {out_dir.relative_to(ROOT)}")
    print(
        f"skipped_unselected={skipped_unselected} "
        f"skipped_missing_or_invalid={skipped_missing} skipped_no_class={skipped_no_class}"
    )
    for (split, class_name), count in sorted(counters.items()):
        print(f"{split} {class_name}: {count}")


if __name__ == "__main__":
    main()
