#!/usr/bin/env python
"""Build a balanced real visible-evidence review packet from overlap clusters."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from build_real_overlap_review_queue import draw_sheet, names_by_id, read_labels


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DATA = Path("data/cashsnap_v1/data.yaml")
DEFAULT_REVIEW_CLUSTERS = Path("runs/cashsnap/real_overlap_review_queue_v1/review_clusters.csv")
DEFAULT_OUT_DIR = Path("runs/cashsnap/real_visible_evidence_review_packet_v1")
EXCLUDED_TRAINLIKE_SOURCES = {"billsbank", "khmer_us_currency"}


BucketPredicate = Callable[[dict[str, str], set[str]], bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE_DATA)
    parser.add_argument("--review-clusters", type=Path, default=DEFAULT_REVIEW_CLUSTERS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-heldout-overlap", type=int, default=24)
    parser.add_argument("--max-heldout-protected-khr", type=int, default=16)
    parser.add_argument("--max-heldout-partial", type=int, default=24)
    parser.add_argument("--max-train-overlap", type=int, default=24)
    parser.add_argument("--max-train-partial", type=int, default=16)
    parser.add_argument("--max-source-policy", type=int, default=16)
    parser.add_argument("--max-preview-side", type=int, default=1800)
    parser.add_argument("--thumb-width", type=int, default=260)
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


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def read_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(resolved)} must be a YAML mapping")
    return payload


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    resolved = resolve(path)
    with resolved.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def float_value(value: str) -> float:
    try:
        return float(str(value).strip() or "0")
    except ValueError:
        return 0.0


def int_value(value: str) -> int:
    try:
        return int(float(str(value).strip() or "0"))
    except ValueError:
        return 0


def row_tags(row: dict[str, str]) -> set[str]:
    return {tag for tag in str(row.get("tags", "")).split(",") if tag}


def row_classes(row: dict[str, str]) -> list[str]:
    return [name for name in str(row.get("classes", "")).split(",") if name]


def row_source(row: dict[str, str]) -> str:
    return str(row.get("source_group", "")).strip()


def row_split(row: dict[str, str]) -> str:
    return str(row.get("split", "")).strip()


def realish_source(row: dict[str, str]) -> bool:
    return row_source(row) not in EXCLUDED_TRAINLIKE_SOURCES


def heldout(row: dict[str, str]) -> bool:
    return row_split(row) in {"val", "test"}


def train(row: dict[str, str]) -> bool:
    return row_split(row) == "train"


def has_any(tags: set[str], *wanted: str) -> bool:
    return any(tag in tags for tag in wanted)


def sort_key(row: dict[str, str]) -> tuple[float, float, int, float, str]:
    return (
        -float_value(row.get("priority", "")),
        -float_value(row.get("max_intersection_small_ratio", "")),
        -int_value(row.get("boxes", "")),
        float_value(row.get("min_gap", "")),
        row.get("image", ""),
    )


def selected_rows(
    rows: list[dict[str, str]],
    *,
    existing_keys: set[str],
    selection_bucket: str,
    cap: int,
    predicate: BucketPredicate,
    suggested_usable_as: str,
    selection_reason: str,
    source_cap: int,
    class_cap: int,
) -> list[dict[str, str]]:
    candidates = [row for row in rows if predicate(row, row_tags(row))]
    candidates.sort(key=sort_key)
    selected: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    for row in candidates:
        if len(selected) >= cap:
            break
        key = str(row.get("canonical_key", "")).strip() or str(row.get("image", "")).strip()
        source = row_source(row)
        classes = row_classes(row)
        if key in existing_keys:
            continue
        if source_counts[source] >= source_cap:
            continue
        if classes and any(class_counts[name] >= class_cap for name in set(classes)):
            continue
        copy = dict(row)
        copy["visible_review_id"] = ""
        copy["selection_bucket"] = selection_bucket
        copy["selection_reason"] = selection_reason
        copy["suggested_usable_as"] = suggested_usable_as
        copy["suggested_final_route"] = suggested_usable_as
        copy["full_preview"] = ""
        selected.append(copy)
        existing_keys.add(key)
        source_counts[source] += 1
        for name in set(classes):
            class_counts[name] += 1
    return selected


def build_packet(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen_keys: set[str] = set()

    def add_bucket(**kwargs: Any) -> None:
        selected.extend(selected_rows(rows, existing_keys=seen_keys, **kwargs))

    add_bucket(
        selection_bucket="heldout_overlap_counting_eval_review",
        cap=args.max_heldout_overlap,
        predicate=lambda row, tags: heldout(row)
        and realish_source(row)
        and has_any(tags, "bbox_overlap", "tight_pair", "multi_note"),
        suggested_usable_as="trusted_overlap_eval",
        selection_reason="held-out real/photo-like overlap, fan, tight-pair, or multi-note rows for counting eval",
        source_cap=8,
        class_cap=8,
    )
    add_bucket(
        selection_bucket="heldout_protected_khr_partial_eval_review",
        cap=args.max_heldout_protected_khr,
        predicate=lambda row, tags: heldout(row)
        and realish_source(row)
        and "protected_riel" in tags
        and "partial_edge" in tags,
        suggested_usable_as="trusted_overlap_eval",
        selection_reason="held-out KHR_20000/KHR_50000 partial-edge rows protect high-value Riel behavior",
        source_cap=10,
        class_cap=12,
    )
    add_bucket(
        selection_bucket="heldout_partial_visible_eval_review",
        cap=args.max_heldout_partial,
        predicate=lambda row, tags: heldout(row) and realish_source(row) and "partial_edge" in tags,
        suggested_usable_as="trusted_overlap_eval",
        selection_reason="held-out partial-edge rows test visible-evidence localization without becoming train anchors",
        source_cap=10,
        class_cap=7,
    )
    add_bucket(
        selection_bucket="train_overlap_anchor_review",
        cap=args.max_train_overlap,
        predicate=lambda row, tags: train(row)
        and realish_source(row)
        and has_any(tags, "bbox_overlap", "tight_pair", "multi_note"),
        suggested_usable_as="train_anchor_candidate",
        selection_reason="train-side real/photo-like overlap and tight-pair anchors that may teach countable visible evidence",
        source_cap=8,
        class_cap=8,
    )
    add_bucket(
        selection_bucket="train_partial_visible_anchor_review",
        cap=args.max_train_partial,
        predicate=lambda row, tags: train(row) and realish_source(row) and "partial_edge" in tags,
        suggested_usable_as="train_anchor_candidate",
        selection_reason="train-side real/photo-like partial-edge anchors, kept separate from overlap anchors",
        source_cap=8,
        class_cap=6,
    )
    add_bucket(
        selection_bucket="source_policy_review",
        cap=args.max_source_policy,
        predicate=lambda row, tags: row_source(row) in EXCLUDED_TRAINLIKE_SOURCES and bool(tags),
        suggested_usable_as="exclude_duplicate_or_flat",
        selection_reason="flat/catalog-prone sources need explicit policy review before any reuse",
        source_cap=10,
        class_cap=10,
    )

    bucket_order = {
        "heldout_overlap_counting_eval_review": 0,
        "heldout_protected_khr_partial_eval_review": 1,
        "heldout_partial_visible_eval_review": 2,
        "train_overlap_anchor_review": 3,
        "train_partial_visible_anchor_review": 4,
        "source_policy_review": 5,
    }
    selected.sort(key=lambda row: (bucket_order.get(row.get("selection_bucket", ""), 99), sort_key(row)))
    for index, row in enumerate(selected, start=1):
        row["visible_review_id"] = f"VE-{index:03d}"
    return selected


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_full_preview(
    *,
    image_path: Path,
    labels: list[dict[str, Any]],
    row: dict[str, str],
    out_path: Path,
    max_side: int,
) -> None:
    with Image.open(image_path) as loaded:
        image = ImageOps.exif_transpose(loaded.convert("RGB"))
    source_width, source_height = image.size
    scale = min(1.0, max_side / max(source_width, source_height))
    if scale < 1.0:
        image = image.resize((round(source_width * scale), round(source_height * scale)), Image.Resampling.LANCZOS)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    font = load_font(max(18, round(max(width, height) * 0.018)))
    label_font = load_font(max(16, round(max(width, height) * 0.014)))
    for label in labels:
        x1 = float(label["x1"]) * width
        y1 = float(label["y1"]) * height
        x2 = float(label["x2"]) * width
        y2 = float(label["y2"]) * height
        name = str(label["class_name"])
        color = (35, 145, 60) if name.startswith("KHR_") else (35, 95, 210)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(3, round(max(width, height) * 0.003)))
        left, top, right, bottom = draw.textbbox((0, 0), name, font=label_font)
        pad = 5
        label_y1 = max(0, y1 - (bottom - top) - pad * 2)
        draw.rectangle(
            (x1, label_y1, x1 + (right - left) + pad * 2, label_y1 + (bottom - top) + pad * 2),
            fill=color,
        )
        draw.text((x1 + pad, label_y1 + pad), name, fill=(255, 255, 255), font=label_font)

    caption_lines = [
        f"{row.get('visible_review_id', '')} {row.get('selection_bucket', '')}",
        f"{row.get('split', '')} {row.get('source_group', '')} boxes={row.get('boxes', '')} classes={row.get('classes', '')}",
        f"tags={row.get('tags', '')}",
    ]
    line_height = max(22, round(max(width, height) * 0.024))
    pad = 10
    banner_height = pad * 2 + line_height * len(caption_lines)
    banner = Image.new("RGB", (width, banner_height), (20, 24, 29))
    banner_draw = ImageDraw.Draw(banner)
    for line_index, text in enumerate(caption_lines):
        banner_draw.text((pad, pad + line_index * line_height), text[:170], fill=(245, 247, 250), font=font)
    preview = Image.new("RGB", (width, height + banner_height), (20, 24, 29))
    preview.paste(banner, (0, 0))
    preview.paste(image, (0, banner_height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(out_path, quality=92)


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image_path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def safe_stem(value: str) -> str:
    stem = Path(value).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return stem[:90]


def attach_labels_and_previews(
    rows: list[dict[str, str]],
    *,
    names: dict[int, str],
    out_dir: Path,
    max_side: int,
) -> list[dict[str, Any]]:
    sheet_rows: list[dict[str, Any]] = []
    preview_dir = out_dir / "full_label_previews"
    for row in rows:
        image_path = resolve(row["image"])
        labels = read_labels(image_path, names)
        preview_path = preview_dir / f"{row['visible_review_id']}_{safe_stem(row['image'])}.jpg"
        draw_full_preview(
            image_path=image_path,
            labels=labels,
            row=row,
            out_path=preview_path,
            max_side=max_side,
        )
        row["full_preview"] = repo_rel(preview_path)
        sheet_row = dict(row)
        sheet_row["labels"] = labels
        sheet_rows.append(sheet_row)
    return sheet_rows


def write_csv(path: Path, rows: list[dict[str, str]], input_fields: list[str]) -> None:
    added_fields = [
        "visible_review_id",
        "selection_bucket",
        "selection_reason",
        "suggested_usable_as",
        "suggested_final_route",
        "full_preview",
    ]
    fields = added_fields + [field for field in input_fields if field not in added_fields]
    for required in ["usable_as", "review_decision", "final_route", "review_notes"]:
        if required not in fields:
            fields.append(required)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_list(path: Path, rows: list[dict[str, str]]) -> None:
    images = list(dict.fromkeys(row.get("image", "").replace("\\", "/") for row in rows if row.get("image")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{image}\n" for image in images), encoding="utf-8")


def write_data_yaml(path: Path, image_list: Path, base_config: dict[str, Any], review_csv: Path) -> None:
    payload = {
        "path": rel_between(path.parent, ROOT),
        "train": repo_rel(image_list),
        "val": repo_rel(image_list),
        "test": repo_rel(image_list),
        "names": base_config.get("names", {}),
        "cashsnap_diagnostic": {
            "purpose": "Balanced real visible-evidence review packet; not accepted training or promotion data.",
            "review_csv": repo_rel(review_csv),
            "source_review_packet": repo_rel(image_list),
            "not_a_promotion_config": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_bucket_views(
    *,
    out_dir: Path,
    rows: list[dict[str, str]],
    base_config: dict[str, Any],
    review_csv: Path,
) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for bucket in sorted({row.get("selection_bucket", "") for row in rows if row.get("selection_bucket")}):
        bucket_rows = [row for row in rows if row.get("selection_bucket") == bucket]
        list_path = out_dir / f"{bucket}_images.txt"
        data_yaml = out_dir / f"{bucket}_data.yaml"
        write_list(list_path, bucket_rows)
        write_data_yaml(data_yaml, list_path, base_config, review_csv)
        views.append(
            {
                "selection_bucket": bucket,
                "rows": len(bucket_rows),
                "images": len({row.get("image", "") for row in bucket_rows}),
                "images_txt": repo_rel(list_path),
                "data_yaml": repo_rel(data_yaml),
            }
        )
    return views


def write_summary(
    *,
    path: Path,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    review_csv: Path,
    image_list: Path,
    data_yaml: Path,
    sheet: Path,
    views: list[dict[str, Any]],
) -> None:
    summary = {
        "schema": "cashsnap_real_visible_evidence_review_packet_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "review_clusters": repo_rel(resolve(args.review_clusters)),
        "base_data": repo_rel(resolve(args.base_data)),
        "out_dir": repo_rel(resolve(args.out_dir)),
        "review_csv": repo_rel(review_csv),
        "images": repo_rel(image_list),
        "data_yaml": repo_rel(data_yaml),
        "sheet": repo_rel(sheet),
        "bucket_views": views,
        "rows": len(rows),
        "unique_images": len({row.get("image", "") for row in rows}),
        "selection_bucket_counts": dict(Counter(row.get("selection_bucket", "") for row in rows).most_common()),
        "source_counts": dict(Counter(row.get("source_group", "") for row in rows).most_common()),
        "split_counts": dict(Counter(row.get("split", "") for row in rows).most_common()),
        "class_counts": dict(Counter(name for row in rows for name in row_classes(row)).most_common()),
        "tag_counts": dict(Counter(tag for row in rows for tag in row_tags(row)).most_common()),
        "suggested_route_counts": dict(Counter(row.get("suggested_usable_as", "") for row in rows).most_common()),
        "requires_explicit_review_before_materialization": True,
        "not_training_data": True,
        "not_a_promotion_config": True,
        "policy": (
            "Rows are selected for visual review only. Accept only human-countable single-note "
            "fragments or clearly countable multi-note scenes; exclude flat/catalog, duplicate, "
            "ambiguous, or unlabeled-evidence layouts before materialization."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_config = read_yaml(args.base_data)
    names = names_by_id(base_config)
    rows, fields = read_csv(args.review_clusters)
    out_dir = resolve(args.out_dir)
    selected = build_packet(rows, args)
    if not selected:
        raise SystemExit("no visible-evidence rows selected")

    sheet_rows = attach_labels_and_previews(
        selected,
        names=names,
        out_dir=out_dir,
        max_side=args.max_preview_side,
    )
    review_csv = out_dir / "visible_evidence_review_packet_v1.csv"
    image_list = out_dir / "visible_evidence_review_packet_v1_images.txt"
    data_yaml = out_dir / "visible_evidence_review_packet_v1_data.yaml"
    sheet = out_dir / "visible_evidence_review_packet_v1_sheet.jpg"
    write_csv(review_csv, selected, fields)
    write_list(image_list, selected)
    write_data_yaml(data_yaml, image_list, base_config, review_csv)
    draw_sheet(sheet, sheet_rows, items=len(sheet_rows), thumb_width=args.thumb_width, cols=args.cols)
    views = write_bucket_views(out_dir=out_dir, rows=selected, base_config=base_config, review_csv=review_csv)
    write_summary(
        path=out_dir / "summary.json",
        rows=selected,
        args=args,
        review_csv=review_csv,
        image_list=image_list,
        data_yaml=data_yaml,
        sheet=sheet,
        views=views,
    )
    print(
        "visible_evidence_review_packet="
        f"{repo_rel(review_csv)} rows={len(selected)} images={len({row['image'] for row in selected})}"
    )
    print(f"sheet={repo_rel(sheet)}")
    print(f"full_previews={repo_rel(out_dir / 'full_label_previews')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
