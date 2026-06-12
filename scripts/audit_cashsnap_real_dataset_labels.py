#!/usr/bin/env python
"""Audit CashSnap real YOLO labels and produce ranked visual QA queues."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from cashsnap_classes import CLASS_NAMES, ID_TO_CLASS


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

SOURCE_PREFIXES = {
    "asian_currency_": "asian_currency",
    "billsbank_": "billsbank",
    "cambodia_currency_project_": "cambodia_currency_project",
    "cashcountingxl_": "cashcountingxl",
    "khmer_scan_": "khmer",
    "khmer_us_currency_": "khmer_us_currency",
    "usd_total_": "usd_total",
}

SOURCE_EXPECTED_CURRENCY = {
    "asian_currency": "KHR_OR_FOREIGN",
    "billsbank": "USD",
    "cambodia_currency_project": "KHR",
    "cashcountingxl": "USD",
    "khmer": "KHR",
    "khmer_us_currency": "MIXED",
    "usd_total": "USD",
}

SEVERITY = {
    "unreadable_image": 100,
    "missing_label_file": 98,
    "bad_label_format": 96,
    "invalid_class_id": 95,
    "invalid_box_geometry": 92,
    "box_out_of_bounds": 88,
    "exact_pixel_duplicate_cross_split": 88,
    "filename_class_clue_mismatch": 84,
    "source_currency_conflict": 82,
    "empty_label_with_target_filename_clue": 78,
    "canonical_base_cross_split": 70,
    "extreme_box_area": 62,
    "tiny_box_short_at_imgsz": 54,
    "multi_class_image": 38,
    "multi_box_image": 30,
}


@dataclass
class LabelRow:
    class_id: int | None
    class_name: str
    cx: float
    cy: float
    bw: float
    bh: float
    xyxy: tuple[float, float, float, float]
    area_norm: float
    short_at_imgsz: float
    line: str
    line_no: int


@dataclass
class Issue:
    image: str
    label: str
    split: str
    source_group: str
    reason: str
    severity: int
    detail: str
    class_names: str = ""
    clue_class: str = ""
    canonical_base: str = ""


@dataclass
class ImageRecord:
    image_path: Path
    label_path: Path
    split: str
    source_group: str
    canonical_base: str
    width: int = 0
    height: int = 0
    readable: bool = False
    labels: list[LabelRow] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    clue_class: str = ""
    sha256: str = ""

    @property
    def image(self) -> str:
        return repo_rel(self.image_path)

    @property
    def label(self) -> str:
        return repo_rel(self.label_path)

    @property
    def class_names(self) -> list[str]:
        return [row.class_name for row in self.labels if row.class_name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("configs/cashsnap_v1.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--sheet-items", type=int, default=80)
    parser.add_argument("--thumb-width", type=int, default=260)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--compute-hash", action="store_true")
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


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"YOLO data config must be a mapping: {repo_rel(path)}")
    return payload


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw = Path(str(config.get("path", ".")))
    return raw if raw.is_absolute() else (config_path.parent / raw).resolve()


def split_images(config_path: Path, config: dict[str, Any], split: str) -> list[Path]:
    root = data_root(config_path, config)
    split_value = config.get(split)
    if split_value is None:
        raise SystemExit(f"{repo_rel(config_path)} has no split {split!r}")
    values = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for raw_value in values:
        value = Path(str(raw_value))
        resolved = value if value.is_absolute() else root / value
        if resolved.suffix.lower() == ".txt":
            for raw_line in resolved.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                image = Path(line)
                images.append(image if image.is_absolute() else root / image)
        else:
            images.extend(sorted(path for path in resolved.glob("*") if path.suffix.lower() in IMAGE_EXTS))
    return images


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def source_group_for_image(image_path: Path) -> str:
    name = image_path.name.lower()
    for prefix, group in SOURCE_PREFIXES.items():
        if name.startswith(prefix) or f"_{prefix}" in name:
            return group
    return name.split("_", 1)[0] if "_" in name else "unknown"


def canonical_base_for_image(image_path: Path) -> str:
    stem = image_path.stem.lower()
    return re.sub(r"\.rf\.[0-9a-f]+$", "", stem)


def currency_of_class(class_name: str) -> str:
    if class_name.startswith("USD_"):
        return "USD"
    if class_name.startswith("KHR_"):
        return "KHR"
    return "UNKNOWN"


def class_clue_from_filename(image_path: Path) -> str:
    name = image_path.stem.lower()
    patterns = [
        (r"(?<!\d)(1|5|10|20|50|100)\s*dollar", "USD"),
        (r"(?<!\d)(1|5|10|20|50|100)\s*usd", "USD"),
        (r"(?<!\d)(1|5|10|20|50|100)\s*us(?![a-z])", "USD"),
        (r"(?<!\d)(500|1000|2000|5000|10000|20000|50000)\s*riel", "KHR"),
        (r"(?<!\d)(500|1000|2000|5000|10000|20000|50000)\s*riels", "KHR"),
    ]
    cleaned = re.sub(r"[_-]+", " ", name)
    for pattern, currency in patterns:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        return f"{currency}_{match.group(1)}"
    return ""


def add_issue(record: ImageRecord, reason: str, detail: str) -> None:
    record.issues.append(
        Issue(
            image=record.image,
            label=record.label,
            split=record.split,
            source_group=record.source_group,
            reason=reason,
            severity=SEVERITY[reason],
            detail=detail,
            class_names=";".join(record.class_names),
            clue_class=record.clue_class,
            canonical_base=record.canonical_base,
        )
    )


def read_labels(record: ImageRecord, imgsz: int) -> None:
    if not record.label_path.exists():
        add_issue(record, "missing_label_file", "image has no paired YOLO label file")
        return
    text = record.label_path.read_text(encoding="utf-8", errors="replace")
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            add_issue(record, "bad_label_format", f"line {line_no} has {len(parts)} fields")
            continue
        try:
            raw_class, cx, cy, bw, bh = parts
            class_id = int(float(raw_class))
            cx_f = float(cx)
            cy_f = float(cy)
            bw_f = float(bw)
            bh_f = float(bh)
        except ValueError:
            add_issue(record, "bad_label_format", f"line {line_no} has non-numeric fields")
            continue
        if class_id not in ID_TO_CLASS:
            class_name = f"class_{class_id}"
            add_issue(record, "invalid_class_id", f"line {line_no} class_id={class_id}")
        else:
            class_name = ID_TO_CLASS[class_id]
        if not all(math.isfinite(value) for value in [cx_f, cy_f, bw_f, bh_f]) or bw_f <= 0 or bh_f <= 0:
            add_issue(record, "invalid_box_geometry", f"line {line_no} has non-finite/non-positive box")
            continue
        x1_norm = cx_f - bw_f / 2.0
        y1_norm = cy_f - bh_f / 2.0
        x2_norm = cx_f + bw_f / 2.0
        y2_norm = cy_f + bh_f / 2.0
        if min(x1_norm, y1_norm) < -0.001 or max(x2_norm, y2_norm) > 1.001:
            add_issue(record, "box_out_of_bounds", f"line {line_no} extends outside normalized image bounds")
        xyxy = (
            x1_norm * record.width,
            y1_norm * record.height,
            x2_norm * record.width,
            y2_norm * record.height,
        )
        scale = imgsz / max(record.width, record.height)
        short_at_imgsz = min(bw_f * record.width * scale, bh_f * record.height * scale)
        row = LabelRow(
            class_id=class_id if class_id in ID_TO_CLASS else None,
            class_name=class_name,
            cx=cx_f,
            cy=cy_f,
            bw=bw_f,
            bh=bh_f,
            xyxy=xyxy,
            area_norm=bw_f * bh_f,
            short_at_imgsz=short_at_imgsz,
            line=line,
            line_no=line_no,
        )
        record.labels.append(row)
        if row.area_norm < 0.002 or row.area_norm > 0.98:
            add_issue(record, "extreme_box_area", f"line {line_no} area_norm={row.area_norm:.5f}")
        if row.short_at_imgsz < 32.0:
            add_issue(record, "tiny_box_short_at_imgsz", f"line {line_no} short_at_{imgsz}={row.short_at_imgsz:.1f}px")


def audit_record(record: ImageRecord, imgsz: int, compute_hash: bool) -> None:
    record.clue_class = class_clue_from_filename(record.image_path)
    try:
        with Image.open(record.image_path) as image:
            record.width, record.height = image.size
            image.verify()
        record.readable = True
    except Exception as exc:  # noqa: BLE001 - this is an audit; capture all image open failures.
        add_issue(record, "unreadable_image", f"{type(exc).__name__}: {exc}")
        return
    if compute_hash:
        record.sha256 = hashlib.sha256(record.image_path.read_bytes()).hexdigest()
    read_labels(record, imgsz)

    currencies = {currency_of_class(name) for name in record.class_names}
    expected = SOURCE_EXPECTED_CURRENCY.get(record.source_group, "UNKNOWN")
    if expected in {"USD", "KHR"} and any(currency != expected for currency in currencies):
        add_issue(record, "source_currency_conflict", f"source expected {expected}; labels={record.class_names}")
    if expected == "KHR_OR_FOREIGN" and "USD" in currencies:
        add_issue(record, "source_currency_conflict", f"asian_currency source has USD target label(s): {record.class_names}")

    if record.clue_class:
        if not record.labels:
            add_issue(record, "empty_label_with_target_filename_clue", f"filename implies {record.clue_class}")
        elif record.clue_class not in record.class_names:
            add_issue(record, "filename_class_clue_mismatch", f"filename implies {record.clue_class}; labels={record.class_names}")

    if len(record.labels) > 1:
        add_issue(record, "multi_box_image", f"{len(record.labels)} target boxes")
    if len(set(record.class_names)) > 1:
        add_issue(record, "multi_class_image", f"classes={record.class_names}")


def row_for_inventory(record: ImageRecord) -> dict[str, Any]:
    return {
        "image": record.image,
        "label": record.label,
        "split": record.split,
        "source_group": record.source_group,
        "canonical_base": record.canonical_base,
        "width": record.width,
        "height": record.height,
        "readable": int(record.readable),
        "label_count": len(record.labels),
        "class_names": ";".join(record.class_names),
        "clue_class": record.clue_class,
        "sha256": record.sha256,
        "issue_count": len(record.issues),
        "max_issue_severity": max((issue.severity for issue in record.issues), default=0),
        "issue_reasons": ";".join(sorted({issue.reason for issue in record.issues})),
    }


def summarize_records(records: list[ImageRecord]) -> dict[str, Any]:
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_class_images: dict[str, Counter[str]] = defaultdict(Counter)
    split_class_boxes: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_class_images: dict[str, Counter[str]] = defaultdict(Counter)
    source_class_boxes: dict[str, Counter[str]] = defaultdict(Counter)
    reason_counts = Counter()
    for record in records:
        split_counts[record.split]["images"] += 1
        split_counts[record.split]["readable"] += int(record.readable)
        split_counts[record.split]["empty_images"] += int(len(record.labels) == 0)
        split_counts[record.split]["boxes"] += len(record.labels)
        source_counts[f"{record.split}|{record.source_group}"]["images"] += 1
        source_counts[f"{record.split}|{record.source_group}"]["empty_images"] += int(len(record.labels) == 0)
        source_counts[f"{record.split}|{record.source_group}"]["boxes"] += len(record.labels)
        for class_name in sorted(set(record.class_names)):
            split_class_images[record.split][class_name] += 1
            source_class_images[f"{record.split}|{record.source_group}"][class_name] += 1
        for class_name in record.class_names:
            split_class_boxes[record.split][class_name] += 1
            source_class_boxes[f"{record.split}|{record.source_group}"][class_name] += 1
        for issue in record.issues:
            reason_counts[issue.reason] += 1
    return {
        "by_split": {split: dict(counter) for split, counter in sorted(split_counts.items())},
        "class_images_by_split": {split: dict(counter) for split, counter in sorted(split_class_images.items())},
        "class_boxes_by_split": {split: dict(counter) for split, counter in sorted(split_class_boxes.items())},
        "source_by_split": {key: dict(counter) for key, counter in sorted(source_counts.items())},
        "source_class_images": {key: dict(counter) for key, counter in sorted(source_class_images.items())},
        "source_class_boxes": {key: dict(counter) for key, counter in sorted(source_class_boxes.items())},
        "issue_reason_counts": dict(reason_counts.most_common()),
    }


def add_cross_split_issues(records: list[ImageRecord]) -> list[dict[str, Any]]:
    leakage_rows: list[dict[str, Any]] = []
    by_base: dict[str, list[ImageRecord]] = defaultdict(list)
    by_sha: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_base[record.canonical_base].append(record)
        if record.sha256:
            by_sha[record.sha256].append(record)

    for base, group in sorted(by_base.items()):
        splits = sorted({record.split for record in group})
        if len(splits) < 2:
            continue
        examples = [record.image for record in group[:8]]
        detail = f"canonical base appears in splits={splits} count={len(group)}"
        leakage_rows.append(
            {
                "kind": "canonical_base",
                "key": base,
                "splits": ";".join(splits),
                "count": len(group),
                "classes": ";".join(sorted({name for record in group for name in record.class_names})),
                "sources": ";".join(sorted({record.source_group for record in group})),
                "examples": ";".join(examples),
            }
        )
        for record in group:
            add_issue(record, "canonical_base_cross_split", detail)

    for digest, group in sorted(by_sha.items()):
        splits = sorted({record.split for record in group})
        if len(splits) < 2:
            continue
        examples = [record.image for record in group[:8]]
        detail = f"exact image hash appears in splits={splits} count={len(group)}"
        leakage_rows.append(
            {
                "kind": "sha256",
                "key": digest,
                "splits": ";".join(splits),
                "count": len(group),
                "classes": ";".join(sorted({name for record in group for name in record.class_names})),
                "sources": ";".join(sorted({record.source_group for record in group})),
                "examples": ";".join(examples),
            }
        )
        for record in group:
            add_issue(record, "exact_pixel_duplicate_cross_split", detail)
    return leakage_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def issue_rows(records: list[ImageRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for issue in record.issues:
            rows.append(issue.__dict__)
    rows.sort(key=lambda row: (-int(row["severity"]), row["split"], row["source_group"], row["image"], row["reason"]))
    return rows


def representative_issue_records(records: list[ImageRecord]) -> list[ImageRecord]:
    return sorted(
        [record for record in records if record.issues],
        key=lambda record: (
            -max(issue.severity for issue in record.issues),
            -len(record.issues),
            record.split,
            record.source_group,
            record.image,
        ),
    )


def sample_records(records: list[ImageRecord], count: int, seed: int) -> list[ImageRecord]:
    pool = list(records)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return sorted(pool[:count], key=lambda record: (record.split, record.source_group, record.image))


def draw_sheet(records: list[ImageRecord], out_path: Path, names: list[str], thumb_width: int, cols: int, title: str) -> None:
    if not records:
        return
    thumb_h = int(thumb_width * 0.82)
    caption_h = 58
    cols = max(1, min(cols, len(records)))
    rows = (len(records) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_width, rows * (thumb_h + caption_h)), (238, 238, 238))
    font = ImageFont.load_default()
    for index, record in enumerate(records):
        x = (index % cols) * thumb_width
        y = (index // cols) * (thumb_h + caption_h)
        try:
            with Image.open(record.image_path).convert("RGB") as image:
                draw = ImageDraw.Draw(image)
                for label in record.labels:
                    color = (30, 170, 70)
                    if label.class_id is not None:
                        color = palette(label.class_id)
                    draw.rectangle(label.xyxy, outline=color, width=4)
                    if label.class_id is not None:
                        draw.text((label.xyxy[0] + 4, max(0, label.xyxy[1] - 14)), names[label.class_id], fill=color, font=font)
                thumb = ImageOps.contain(image, (thumb_width, thumb_h), Image.Resampling.LANCZOS)
        except Exception:
            thumb = Image.new("RGB", (thumb_width, thumb_h), (80, 80, 80))
        sheet.paste(thumb, (x + (thumb_width - thumb.width) // 2, y))
        reasons = ",".join(issue.reason for issue in sorted(record.issues, key=lambda issue: -issue.severity)[:2])
        if not reasons:
            reasons = "sample"
        classes = ",".join(record.class_names) or "background"
        caption_1 = f"{record.split} {record.source_group} {classes}"
        caption_2 = f"{reasons}"
        caption_3 = record.image_path.name
        caption_draw = ImageDraw.Draw(sheet)
        caption_draw.text((x + 5, y + thumb_h + 4), caption_1[:44], fill=(0, 0, 0), font=font)
        caption_draw.text((x + 5, y + thumb_h + 22), caption_2[:44], fill=(120, 0, 0), font=font)
        caption_draw.text((x + 5, y + thumb_h + 40), caption_3[:44], fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    print(f"sheet={repo_rel(out_path)} title={title} records={len(records)}")


def palette(class_id: int) -> tuple[int, int, int]:
    colors = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (0, 128, 128),
        (240, 50, 230),
        (170, 110, 40),
        (0, 0, 128),
        (128, 128, 0),
        (128, 0, 0),
        (70, 70, 70),
    ]
    return colors[class_id % len(colors)]


def write_manifest(path: Path, records: list[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{record.image}\n" for record in records), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_path = resolve(args.data)
    config = load_yaml(data_path)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = args.split or ["train", "val", "test"]

    records: list[ImageRecord] = []
    seen_images: set[Path] = set()
    for split in splits:
        for image_path in split_images(data_path, config, split):
            resolved_image = image_path.resolve()
            if resolved_image in seen_images:
                continue
            seen_images.add(resolved_image)
            record = ImageRecord(
                image_path=resolved_image,
                label_path=label_path_for_image(resolved_image),
                split=split,
                source_group=source_group_for_image(resolved_image),
                canonical_base=canonical_base_for_image(resolved_image),
            )
            audit_record(record, args.imgsz, args.compute_hash)
            records.append(record)

    leakage_rows = add_cross_split_issues(records)
    issues = issue_rows(records)
    inventory = [row_for_inventory(record) for record in records]
    summary = summarize_records(records)
    summary.update(
        {
            "schema": "cashsnap_real_dataset_label_audit_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "data": repo_rel(data_path),
            "splits": splits,
            "records": len(records),
            "issues": len(issues),
            "compute_hash": bool(args.compute_hash),
            "outputs": {
                "inventory_csv": repo_rel(out_dir / "inventory.csv"),
                "issues_csv": repo_rel(out_dir / "issues.csv"),
                "leakage_csv": repo_rel(out_dir / "cross_split_leakage.csv"),
            },
            "leakage_groups": len(leakage_rows),
        }
    )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(
        out_dir / "inventory.csv",
        inventory,
        [
            "image",
            "label",
            "split",
            "source_group",
            "canonical_base",
            "width",
            "height",
            "readable",
            "label_count",
            "class_names",
            "clue_class",
            "sha256",
            "issue_count",
            "max_issue_severity",
            "issue_reasons",
        ],
    )
    write_csv(
        out_dir / "issues.csv",
        issues,
        ["severity", "reason", "split", "source_group", "class_names", "clue_class", "canonical_base", "image", "label", "detail"],
    )
    write_csv(
        out_dir / "cross_split_leakage.csv",
        leakage_rows,
        ["kind", "key", "splits", "count", "classes", "sources", "examples"],
    )

    ranked_records = representative_issue_records(records)
    buckets = {
        "top_issues": ranked_records,
        "mixed_currency_risk": [
            record
            for record in ranked_records
            if record.source_group == "khmer_us_currency"
            or any(issue.reason in {"source_currency_conflict", "filename_class_clue_mismatch"} for issue in record.issues)
        ],
        "empty_target_clues": [
            record for record in ranked_records if any(issue.reason == "empty_label_with_target_filename_clue" for issue in record.issues)
        ],
        "geometry": [
            record
            for record in ranked_records
            if any(issue.reason in {"invalid_box_geometry", "box_out_of_bounds", "tiny_box_short_at_imgsz", "extreme_box_area"} for issue in record.issues)
        ],
        "cross_split_leakage": [
            record
            for record in ranked_records
            if any(issue.reason in {"canonical_base_cross_split", "exact_pixel_duplicate_cross_split"} for issue in record.issues)
        ],
        "rare_khr": [
            record
            for record in records
            if any(class_name in {"KHR_20000", "KHR_50000"} for class_name in record.class_names)
        ],
        "khmer_us_currency_sample": sample_records(
            [record for record in records if record.source_group == "khmer_us_currency"],
            args.sheet_items,
            args.sample_seed,
        ),
        "asian_currency_empty_sample": sample_records(
            [record for record in records if record.source_group == "asian_currency" and not record.labels],
            args.sheet_items,
            args.sample_seed + 1,
        ),
        "cashcountingxl_empty_sample": sample_records(
            [record for record in records if record.source_group == "cashcountingxl" and not record.labels],
            args.sheet_items,
            args.sample_seed + 2,
        ),
    }
    for name, bucket_records in buckets.items():
        limited = bucket_records[: args.sheet_items]
        write_manifest(out_dir / f"{name}.txt", limited)
        draw_sheet(
            limited,
            out_dir / f"{name}.jpg",
            CLASS_NAMES,
            args.thumb_width,
            args.cols,
            name,
        )

    print(
        "real_dataset_audit "
        f"records={len(records)} issues={len(issues)} leakage_groups={len(leakage_rows)} "
        f"out={repo_rel(out_dir)}"
    )


if __name__ == "__main__":
    main()
