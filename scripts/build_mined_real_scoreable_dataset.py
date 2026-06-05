#!/usr/bin/env python
"""Build an ignored YOLO dataset from reviewed mined-real diagnostic labels."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "runs" / "cashsnap" / "mined_real_benchmark_review_sources_latest.csv"
DEFAULT_QUALITY = ROOT / "manifests" / "mined_real_benchmark_review_quality.csv"
DEFAULT_DRAFT_LABEL_DIR = ROOT / "data" / "real_fan_benchmark" / "mined_cashsnap_v1" / "drafts"
DEFAULT_OUT_ROOT = ROOT / "runs" / "cashsnap" / "mined_real_scoreable_dataset_latest"

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
DECIDED_QUALITIES = SCOREABLE_QUALITIES | {"reject"}
TRUE_VALUES = {"1", "true", "yes", "y", "score", "keep"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--draft-label-dir", type=Path, default=DEFAULT_DRAFT_LABEL_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--include-source-splits", default="train,val,test")
    parser.add_argument("--min-images", type=int, default=0)
    parser.add_argument("--min-boxes", type=int, default=0)
    parser.add_argument("--min-stress-images", type=int, default=0)
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


def source_split(path_text: str) -> str:
    parts = Path(path_text.replace("\\", "/")).parts
    try:
        images_index = parts.index("images")
    except ValueError:
        return ""
    if images_index + 1 >= len(parts):
        return ""
    return parts[images_index + 1]


def truthy(value: str) -> bool:
    return value.strip().lower() in TRUE_VALUES


def read_label_lines(path: Path) -> list[str]:
    return [line.strip() for line in resolve(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def reset_output_dir(out_root: Path) -> Path:
    resolved = resolve(out_root).resolve()
    allowed_root = (ROOT / "runs" / "cashsnap").resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise SystemExit(f"refusing to clear output outside {repo_path(allowed_root)}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    (resolved / "images" / "val").mkdir(parents=True, exist_ok=True)
    (resolved / "labels" / "val").mkdir(parents=True, exist_ok=True)
    return resolved


def scoreable_lines(
    *,
    image_id: str,
    draft_path: Path,
    quality_rows: list[dict[str, str]],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    source_lines = read_label_lines(draft_path)
    if not source_lines:
        return [], [f"{image_id}: empty draft label file {repo_path(draft_path)}"]
    if len(quality_rows) != len(source_lines):
        return [], [f"{image_id}: quality rows {len(quality_rows)} != draft labels {len(source_lines)}"]

    kept: list[tuple[int, str]] = []
    seen: set[int] = set()
    for row in quality_rows:
        try:
            index = int(row.get("label_index", ""))
        except ValueError:
            errors.append(f"{image_id}: invalid label_index {row.get('label_index')!r}")
            continue
        if index in seen:
            errors.append(f"{image_id}: duplicate quality row for index {index}")
            continue
        seen.add(index)
        if not 0 <= index < len(source_lines):
            errors.append(f"{image_id}: label index {index} outside 0..{len(source_lines) - 1}")
            continue

        quality = row.get("quality", "").strip()
        if quality not in DECIDED_QUALITIES:
            return [], []
        if quality in SCOREABLE_QUALITIES and truthy(row.get("count_for_score", "")):
            kept.append((index, source_lines[index]))

    if errors:
        return [], errors
    return [line for _, line in sorted(kept)], []


def write_dataset_yaml(out_root: Path) -> None:
    payload: dict[str, Any] = {
        "path": out_root.as_posix(),
        "train": "images/val",
        "val": "images/val",
        "test": "images/val",
        "nc": len(CLASS_NAMES),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        "cashsnap_mined_real_scoreboard": {
            "promotion_status": "diagnostic_review_only",
            "reason": "Mined cashsnap_v1 labels are reviewed diagnostic anchors, not protected real-transfer proof.",
        },
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_manifest(out_root: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(out_root: Path, payload: dict[str, Any]) -> None:
    (out_root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    sources = read_csv(args.sources)
    quality_rows = read_csv(args.quality)
    include_splits = {value.strip() for value in args.include_source_splits.replace(";", ",").split(",") if value.strip()}
    quality_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in quality_rows:
        key = (row.get("image_id", ""), row.get("label_path", "").replace("\\", "/"))
        quality_by_key.setdefault(key, []).append(row)

    out_root = reset_output_dir(args.out_root)
    image_rows: list[dict[str, str]] = []
    errors: list[str] = []
    role_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    skipped_by_split = 0
    skipped_unready = 0

    for source in sources:
        image_id = source.get("image_id", "")
        split = source_split(source.get("local_path", ""))
        if include_splits and split not in include_splits:
            skipped_by_split += 1
            continue

        draft_path = resolve(args.draft_label_dir) / f"{image_id}.txt"
        key = (image_id, repo_path(draft_path))
        kept_lines, row_errors = scoreable_lines(image_id=image_id, draft_path=draft_path, quality_rows=quality_by_key.get(key, []))
        errors.extend(row_errors)
        if not kept_lines:
            if not row_errors:
                skipped_unready += 1
            continue

        source_image = resolve(Path(source.get("local_path", "")))
        if not source_image.exists():
            errors.append(f"{image_id}: missing source image {repo_path(source_image)}")
            continue
        image_out = out_root / "images" / "val" / f"{image_id}{source_image.suffix.lower()}"
        label_out = out_root / "labels" / "val" / f"{image_id}.txt"
        shutil.copy2(source_image, image_out)
        label_out.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")

        role = source.get("benchmark_role", "")
        role_counts[role] += 1
        split_counts[split] += 1
        for line in kept_lines:
            class_id = int(line.split()[0])
            class_counts[CLASS_NAMES[class_id]] += 1
        image_rows.append(
            {
                "image_id": image_id,
                "benchmark_role": role,
                "source_split": split,
                "source_image": repo_path(source_image),
                "image_path": repo_path(image_out),
                "label_path": repo_path(label_out),
                "boxes": str(len(kept_lines)),
            }
        )

    write_dataset_yaml(out_root)
    write_manifest(out_root, image_rows)

    box_count = sum(int(row["boxes"]) for row in image_rows)
    summary = {
        "data_yaml": repo_path(out_root / "data.yaml"),
        "images": len(image_rows),
        "boxes": box_count,
        "roles": dict(sorted(role_counts.items())),
        "source_splits": dict(sorted(split_counts.items())),
        "classes": dict(sorted(class_counts.items())),
        "skipped_by_split": skipped_by_split,
        "skipped_unready": skipped_unready,
        "errors": errors,
    }
    write_summary(out_root, summary)

    if len(image_rows) < args.min_images:
        errors.append(f"images {len(image_rows)} < required {args.min_images}")
    if box_count < args.min_boxes:
        errors.append(f"boxes {box_count} < required {args.min_boxes}")
    stress_images = sum(count for role, count in role_counts.items() if role.endswith("_stress"))
    if stress_images < args.min_stress_images:
        errors.append(f"stress_images {stress_images} < required {args.min_stress_images}")

    print(
        "mined_real_scoreable_dataset "
        f"images={len(image_rows)} boxes={box_count} stress_images={stress_images} "
        f"skipped_unready={skipped_unready} errors={len(errors)}"
    )
    print(f"data_yaml={repo_path(out_root / 'data.yaml')}")
    print(f"summary={repo_path(out_root / 'summary.json')}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
