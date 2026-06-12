#!/usr/bin/env python
"""Materialize a reviewed CashSnap official21 YOLO bridge dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_BASE_DATA = Path("data/cashsnap_v1/data.yaml")
DEFAULT_SCHEMA = Path("configs/taxonomy/cashsnap_official21_schema_draft_v1.yaml")
ACCEPTED_BOX_DECISION = "accepted_box"


@dataclass(frozen=True)
class ProposalBox:
    source_row: int
    image: Path
    new_class_name: str
    new_class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]
    model: str
    current_pred_class: str
    review_notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE_DATA)
    parser.add_argument("--official21-schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--proposal-csv", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--out-config", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument(
        "--scope",
        choices=["full_dataset", "proposal_images"],
        default="full_dataset",
        help="full_dataset mirrors all selected base splits; proposal_images writes only images with accepted boxes.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write a dry summary instead of failing when the proposal CSV has no accepted boxes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write summary.json only; do not mirror images, labels, or data.yaml.",
    )
    parser.add_argument(
        "--image-mode",
        choices=["auto", "hardlink", "copy"],
        default="auto",
        help="How to mirror images when not in dry-run mode.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove out-root first if it is under data/processed or runs.")
    parser.add_argument(
        "--proposal-class",
        default="KHR_100",
        help="Expected proposed_new_class for accepted rows, or 'any' to accept multiple official21 classes.",
    )
    parser.add_argument(
        "--accepted-decision",
        default=ACCEPTED_BOX_DECISION,
        help="Exact normalized review_decision value required for proposal boxes.",
    )
    parser.add_argument(
        "--dedupe-iou",
        type=float,
        default=0.90,
        help="Drop duplicate accepted boxes of the same class on the same image above this IoU.",
    )
    parser.add_argument(
        "--no-require-train-proposals",
        action="store_true",
        help="Allow accepted proposal boxes outside images/train. Default keeps missing-class review boxes train-only.",
    )
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def normalized(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def load_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(resolved)} must be a YAML mapping")
    return data


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw_root = Path(str(config.get("path", "."))).expanduser()
    if raw_root.is_absolute():
        return raw_root.resolve()
    return (resolve(config_path).parent / raw_root).resolve()


def split_path(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def image_paths_from_value(root: Path, value: Any) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    paths: list[Path] = []
    for raw_item in values:
        if not isinstance(raw_item, str):
            raise SystemExit("YOLO split values must be strings")
        item_path = split_path(root, raw_item)
        if item_path.suffix.lower() == ".txt":
            for raw_line in item_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                image_path = Path(line).expanduser()
                paths.append(image_path if image_path.is_absolute() else root / image_path)
            continue
        if item_path.is_dir():
            paths.extend(sorted(path for path in item_path.iterdir() if path.suffix.lower() in IMAGE_EXTS))
            continue
        raise SystemExit(f"unsupported split path: {repo_rel(item_path)}")
    return paths


def read_split_images(config_path: Path, config: dict[str, Any], splits: Iterable[str]) -> dict[str, list[Path]]:
    root = data_root(config_path, config)
    out: dict[str, list[Path]] = {}
    for split in splits:
        if split not in config:
            raise SystemExit(f"{repo_rel(resolve(config_path))} missing split {split!r}")
        rows = image_paths_from_value(root, config[split])
        if not rows:
            raise SystemExit(f"{repo_rel(resolve(config_path))} split {split!r} has no images")
        out[split] = [path.resolve() for path in rows]
    return out


def names_by_id(config: dict[str, Any]) -> dict[int, str]:
    raw = config.get("names")
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    raise SystemExit("YAML config missing names list or mapping")


def load_official_schema(path: Path) -> tuple[dict[int, str], dict[int, int]]:
    schema = load_yaml(path)
    names = names_by_id(schema)
    nc = int(schema.get("nc", len(names)))
    if nc != len(names):
        raise SystemExit(f"{repo_rel(resolve(path))} nc={nc} but names has {len(names)} entries")
    expected_ids = set(range(nc))
    if set(names) != expected_ids:
        raise SystemExit(f"{repo_rel(resolve(path))} names must cover contiguous ids 0..{nc - 1}")
    compat = schema.get("current_core13_compat")
    if not isinstance(compat, dict):
        raise SystemExit(f"{repo_rel(resolve(path))} missing current_core13_compat")
    raw_mapping = compat.get("old_to_official21")
    if not isinstance(raw_mapping, dict):
        raise SystemExit(f"{repo_rel(resolve(path))} missing current_core13_compat.old_to_official21")
    mapping = {int(old): int(new) for old, new in raw_mapping.items()}
    missing_targets = [new for new in mapping.values() if new not in names]
    if missing_targets:
        raise SystemExit(f"official21 mapping points at missing ids: {sorted(missing_targets)}")
    return names, mapping


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image_path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def unique_name(split: str, index: int, image_path: Path) -> str:
    digest = hashlib.sha1(repo_rel(image_path).encode("utf-8")).hexdigest()[:12]
    safe_stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in image_path.stem)
    safe_stem = safe_stem[:80].strip("_") or "image"
    return f"{split}_{index:06d}_{safe_stem}_{digest}{image_path.suffix.lower()}"


def parse_label_line(label_path: Path, line_no: int, raw_line: str, class_mapping: dict[int, int]) -> tuple[str, int]:
    parts = raw_line.split()
    if len(parts) < 5:
        raise SystemExit(f"{repo_rel(label_path)}:{line_no} expected YOLO class plus box fields")
    try:
        old_class_id = int(float(parts[0]))
    except ValueError as exc:
        raise SystemExit(f"{repo_rel(label_path)}:{line_no} invalid class id {parts[0]!r}") from exc
    if old_class_id not in class_mapping:
        raise SystemExit(f"{repo_rel(label_path)}:{line_no} class id {old_class_id} is not in official21 mapping")
    return " ".join([str(class_mapping[old_class_id]), *parts[1:]]), old_class_id


def read_remapped_label_lines(
    label_path: Path,
    class_mapping: dict[int, int],
    base_names: dict[int, str],
    official_names: dict[int, str],
) -> tuple[list[str], Counter[str], Counter[str]]:
    lines: list[str] = []
    base_counts: Counter[str] = Counter()
    official_counts: Counter[str] = Counter()
    if not label_path.exists():
        return lines, base_counts, official_counts
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        remapped, old_class_id = parse_label_line(label_path, line_no, line, class_mapping)
        new_class_id = class_mapping[old_class_id]
        base_counts[base_names.get(old_class_id, f"class_{old_class_id}")] += 1
        official_counts[official_names.get(new_class_id, f"class_{new_class_id}")] += 1
        lines.append(remapped)
    return lines, base_counts, official_counts


def read_png_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) >= 24 and header.startswith(b"\x89PNG\r\n\x1a\n"):
        width, height = struct.unpack(">II", header[16:24])
        return int(width), int(height)
    return None


def read_jpeg_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            return None
        while True:
            marker_start = handle.read(1)
            if not marker_start:
                return None
            if marker_start != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if marker in {b"\xd8", b"\xd9"}:
                continue
            raw_length = handle.read(2)
            if len(raw_length) != 2:
                return None
            length = struct.unpack(">H", raw_length)[0]
            if length < 2:
                return None
            if marker and marker[0] in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                segment = handle.read(length - 2)
                if len(segment) < 5:
                    return None
                height, width = struct.unpack(">HH", segment[1:5])
                return int(width), int(height)
            handle.seek(length - 2, os.SEEK_CUR)


def image_size(path: Path) -> tuple[int, int]:
    ext = path.suffix.lower()
    size = read_png_size(path) if ext == ".png" else None
    if size is None and ext in {".jpg", ".jpeg"}:
        size = read_jpeg_size(path)
    if size is None:
        raise SystemExit(f"cannot read image size without Pillow support for {repo_rel(path)}")
    return size


def parse_float(row: dict[str, str], field: str, row_number: int) -> float:
    try:
        return float(str(row.get(field, "")).strip())
    except ValueError as exc:
        raise SystemExit(f"proposal row {row_number} has invalid {field}: {row.get(field)!r}") from exc


def is_train_image(path: Path) -> bool:
    return "/images/train/" in f"/{path.as_posix()}"


def read_reviewed_proposals(
    *,
    proposal_csv: Path,
    official_names: dict[int, str],
    proposal_class: str,
    accepted_decision: str,
    require_train: bool,
) -> tuple[dict[Path, list[ProposalBox]], dict[str, Any]]:
    name_to_id = {name: class_id for class_id, name in official_names.items()}
    any_class = proposal_class.strip().lower() in {"any", "all", "*"}
    if not any_class and proposal_class not in name_to_id:
        raise SystemExit(f"proposal class {proposal_class!r} is not present in official21 names")
    accepted_norm = normalized(accepted_decision)
    proposal_path = resolve(proposal_csv)
    by_image: dict[Path, list[ProposalBox]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    rows = 0
    reviewed_not_accepted = 0
    accepted = 0
    with proposal_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"image", "x1", "y1", "x2", "y2", "proposed_new_class", "review_decision"}
        missing = sorted(required_fields - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(f"{repo_rel(proposal_path)} missing fields: {missing}")
        for row_number, row in enumerate(reader, start=2):
            rows += 1
            decision = normalized(str(row.get("review_decision", "")))
            if not decision:
                continue
            if decision != accepted_norm:
                reviewed_not_accepted += 1
                skipped.append(
                    {
                        "row": row_number,
                        "image": row.get("image", ""),
                        "reason": "review_decision_not_accepted_box",
                        "review_decision": row.get("review_decision", ""),
                    }
                )
                continue
            image = resolve(str(row.get("image", "")).strip()).resolve()
            if require_train and not is_train_image(image):
                raise SystemExit(f"proposal row {row_number} accepted a non-train image: {repo_rel(image)}")
            new_class = str(row.get("proposed_new_class", "")).strip()
            if new_class not in name_to_id:
                raise SystemExit(
                    f"proposal row {row_number} accepted class {new_class!r}; not present in official21 names"
                )
            if not any_class and new_class != proposal_class:
                raise SystemExit(
                    f"proposal row {row_number} accepted class {new_class!r}; expected {proposal_class!r}"
                )
            proposal_class_id = name_to_id[new_class]
            x1 = parse_float(row, "x1", row_number)
            y1 = parse_float(row, "y1", row_number)
            x2 = parse_float(row, "x2", row_number)
            y2 = parse_float(row, "y2", row_number)
            if x2 <= x1 or y2 <= y1:
                raise SystemExit(f"proposal row {row_number} has invalid xyxy box")
            confidence = parse_float(row, "confidence", row_number)
            accepted += 1
            by_image[image].append(
                ProposalBox(
                    source_row=row_number,
                    image=image,
                    new_class_name=new_class,
                    new_class_id=proposal_class_id,
                    confidence=confidence,
                    xyxy=(x1, y1, x2, y2),
                    model=str(row.get("model", "")).strip(),
                    current_pred_class=str(row.get("current_pred_class", "")).strip(),
                    review_notes=str(row.get("review_notes", "")).strip(),
                )
            )
    stats = {
        "proposal_csv": repo_rel(proposal_path),
        "rows": rows,
        "accepted_boxes_raw": accepted,
        "reviewed_not_accepted": reviewed_not_accepted,
        "skipped_reviewed_rows": skipped[:200],
        "skipped_reviewed_rows_truncated": max(0, len(skipped) - 200),
        "accepted_decision": accepted_norm,
        "proposal_class": proposal_class,
        "proposal_class_mode": "any" if any_class else "exact",
    }
    return by_image, stats


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def dedupe_proposals(
    proposals_by_image: dict[Path, list[ProposalBox]],
    threshold: float,
) -> tuple[dict[Path, list[ProposalBox]], list[dict[str, Any]]]:
    if threshold <= 0:
        return proposals_by_image, []
    out: dict[Path, list[ProposalBox]] = {}
    skipped: list[dict[str, Any]] = []
    for image, boxes in proposals_by_image.items():
        kept: list[ProposalBox] = []
        for proposal in sorted(boxes, key=lambda item: item.confidence, reverse=True):
            duplicate = next(
                (
                    existing
                    for existing in kept
                    if existing.new_class_id == proposal.new_class_id
                    and box_iou(existing.xyxy, proposal.xyxy) >= threshold
                ),
                None,
            )
            if duplicate is not None:
                skipped.append(
                    {
                        "image": repo_rel(image),
                        "row": proposal.source_row,
                        "kept_row": duplicate.source_row,
                        "reason": "duplicate_box_iou",
                        "iou_threshold": threshold,
                    }
                )
                continue
            kept.append(proposal)
        out[image] = sorted(kept, key=lambda item: item.source_row)
    return out, skipped


def proposal_to_yolo_line(proposal: ProposalBox) -> tuple[str, dict[str, Any]]:
    width, height = image_size(proposal.image)
    x1, y1, x2, y2 = proposal.xyxy
    clamped_x1 = min(max(x1, 0.0), float(width))
    clamped_y1 = min(max(y1, 0.0), float(height))
    clamped_x2 = min(max(x2, 0.0), float(width))
    clamped_y2 = min(max(y2, 0.0), float(height))
    if clamped_x2 <= clamped_x1 or clamped_y2 <= clamped_y1:
        raise SystemExit(f"accepted proposal row {proposal.source_row} clamps to an empty box")
    x_center = ((clamped_x1 + clamped_x2) / 2.0) / width
    y_center = ((clamped_y1 + clamped_y2) / 2.0) / height
    box_width = (clamped_x2 - clamped_x1) / width
    box_height = (clamped_y2 - clamped_y1) / height
    line = f"{proposal.new_class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
    meta = {
        "row": proposal.source_row,
        "class": proposal.new_class_name,
        "class_id": proposal.new_class_id,
        "confidence": proposal.confidence,
        "model": proposal.model,
        "current_pred_class": proposal.current_pred_class,
        "xyxy": [round(value, 3) for value in proposal.xyxy],
        "image_size": [width, height],
        "clamped": [clamped_x1, clamped_y1, clamped_x2, clamped_y2] != [x1, y1, x2, y2],
        "review_notes": proposal.review_notes,
    }
    return line, meta


def mirror_image(source: Path, target: Path, mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, target)
    return "copy"


def allowed_clean_root(path: Path) -> bool:
    resolved = path.resolve()
    for allowed in (ROOT / "data" / "processed", ROOT / "runs"):
        try:
            resolved.relative_to(allowed.resolve())
            return True
        except ValueError:
            continue
    return False


def materialize_split(
    *,
    split: str,
    images: list[Path],
    out_root: Path,
    base_names: dict[int, str],
    official_names: dict[int, str],
    class_mapping: dict[int, int],
    proposals_by_image: dict[Path, list[ProposalBox]],
    dry_run: bool,
    image_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image_dir = out_root / "images" / split
    label_dir = out_root / "labels" / split
    metadata_rows: list[dict[str, Any]] = []
    base_class_counts: Counter[str] = Counter()
    official_class_counts: Counter[str] = Counter()
    proposal_class_counts: Counter[str] = Counter()
    link_modes: Counter[str] = Counter()
    boxes = 0
    proposal_boxes = 0
    empty_images = 0
    labeled_images = 0
    for index, source_image in enumerate(images):
        if not source_image.exists():
            raise SystemExit(f"missing source image: {repo_rel(source_image)}")
        out_name = unique_name(split, index, source_image)
        out_image = image_dir / out_name
        out_label = label_dir / f"{out_image.stem}.txt"
        source_label = label_path_for_image(source_image)
        lines, base_counts, official_counts = read_remapped_label_lines(
            source_label,
            class_mapping,
            base_names,
            official_names,
        )
        base_class_counts.update(base_counts)
        official_class_counts.update(official_counts)
        proposal_meta: list[dict[str, Any]] = []
        for proposal in proposals_by_image.get(source_image, []):
            proposal_line, meta = proposal_to_yolo_line(proposal)
            lines.append(proposal_line)
            proposal_meta.append(meta)
            official_class_counts[proposal.new_class_name] += 1
            proposal_class_counts[proposal.new_class_name] += 1
            proposal_boxes += 1
        boxes += len(lines)
        if lines:
            labeled_images += 1
        else:
            empty_images += 1
        if not dry_run:
            used_mode = mirror_image(source_image, out_image, image_mode)
            link_modes[used_mode] += 1
            out_label.parent.mkdir(parents=True, exist_ok=True)
            out_label.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        metadata_rows.append(
            {
                "split": split,
                "source_image": repo_rel(source_image),
                "source_label": repo_rel(source_label),
                "image": rel_between(out_root, out_image),
                "label": rel_between(out_root, out_label),
                "boxes": len(lines),
                "proposal_boxes": len(proposal_meta),
                "accepted_proposals": proposal_meta,
            }
        )
    if not dry_run:
        manifest_path = out_root / "metadata" / f"{split}.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in metadata_rows) + "\n",
            encoding="utf-8",
        )
    else:
        manifest_path = out_root / "metadata" / f"{split}.jsonl"
    summary = {
        "images": len(images),
        "labeled_images": labeled_images,
        "empty_images": empty_images,
        "boxes": boxes,
        "accepted_proposal_boxes": proposal_boxes,
        "base_class_counts": dict(base_class_counts.most_common()),
        "official_class_counts": dict(official_class_counts.most_common()),
        "accepted_proposal_class_counts": dict(proposal_class_counts.most_common()),
        "image_mirror_modes": dict(link_modes.most_common()),
        "manifest": repo_rel(manifest_path),
    }
    return summary, metadata_rows


def write_data_yaml(
    *,
    out_root: Path,
    out_config: Path | None,
    official_names: dict[int, str],
    bridge_summary: dict[str, Any],
    selected_splits: Iterable[str],
) -> None:
    payload: dict[str, Any] = {"path": "."}
    for split in selected_splits:
        payload[split] = f"images/{split}"
    payload.update(
        {
            "nc": len(official_names),
            "names": official_names,
            "cashsnap_official21_review_bridge": bridge_summary,
        }
    )
    write_yaml(out_root / "data.yaml", payload)
    if out_config is not None:
        external = {**payload, "path": rel_between(resolve(out_config).parent, out_root)}
        write_yaml(resolve(out_config), external)


def main() -> int:
    args = parse_args()
    base_data_path = resolve(args.base_data)
    schema_path = resolve(args.official21_schema)
    out_root = resolve(args.out_root)
    if args.clean and out_root.exists():
        if not allowed_clean_root(out_root):
            raise SystemExit(f"--clean target must stay under data/processed or runs: {repo_rel(out_root)}")
        shutil.rmtree(out_root)

    base_config = load_yaml(base_data_path)
    base_names = names_by_id(base_config)
    official_names, class_mapping = load_official_schema(schema_path)
    split_images = read_split_images(base_data_path, base_config, args.splits)
    proposals_raw, proposal_stats = read_reviewed_proposals(
        proposal_csv=args.proposal_csv,
        official_names=official_names,
        proposal_class=args.proposal_class,
        accepted_decision=args.accepted_decision,
        require_train=not args.no_require_train_proposals,
    )
    proposals_by_image, deduped = dedupe_proposals(proposals_raw, args.dedupe_iou)
    accepted_images = set(proposals_by_image)
    accepted_boxes = sum(len(rows) for rows in proposals_by_image.values())
    if accepted_boxes == 0 and not args.allow_empty:
        raise SystemExit(
            f"no accepted proposal boxes found; set review_decision={ACCEPTED_BOX_DECISION} "
            "on reviewed rows or pass --allow-empty for a dry summary"
        )

    selected_splits: dict[str, list[Path]] = {}
    missing_accepted_images = set(accepted_images)
    for split, images in split_images.items():
        image_set = set(images)
        missing_accepted_images -= image_set
        if args.scope == "proposal_images":
            selected_splits[split] = [image for image in images if image in accepted_images]
        else:
            selected_splits[split] = images
    if accepted_boxes and missing_accepted_images:
        sample = sorted(repo_rel(path) for path in missing_accepted_images)[:10]
        raise SystemExit(f"accepted proposal images are not in selected base splits: {sample}")
    if args.scope == "proposal_images" and not any(selected_splits.values()) and not args.allow_empty:
        raise SystemExit("scope=proposal_images selected no rows")

    out_root.mkdir(parents=True, exist_ok=True)
    split_summaries: dict[str, Any] = {}
    manifest_preview: dict[str, list[dict[str, Any]]] = {}
    for split, images in selected_splits.items():
        split_summary, rows = materialize_split(
            split=split,
            images=images,
            out_root=out_root,
            base_names=base_names,
            official_names=official_names,
            class_mapping=class_mapping,
            proposals_by_image=proposals_by_image,
            dry_run=args.dry_run,
            image_mode=args.image_mode,
        )
        split_summaries[split] = split_summary
        manifest_preview[split] = rows[:20]

    bridge_summary = {
        "schema": "cashsnap_official21_review_bridge_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_data": repo_rel(base_data_path),
        "official21_schema": repo_rel(schema_path),
        "proposal_stats": {
            **proposal_stats,
            "accepted_boxes_after_dedupe": accepted_boxes,
            "accepted_images_after_dedupe": len(accepted_images),
            "deduped_boxes": len(deduped),
            "dedupe_iou": args.dedupe_iou,
            "deduped_examples": deduped[:200],
            "deduped_examples_truncated": max(0, len(deduped) - 200),
        },
        "splits": split_summaries,
        "scope": args.scope,
        "dry_run": bool(args.dry_run),
        "image_mode": args.image_mode,
        "selected_splits": list(selected_splits),
        "out_root": repo_rel(out_root),
        "data_yaml": repo_rel(out_root / "data.yaml"),
        "out_config": repo_rel(resolve(args.out_config)) if args.out_config else "",
        "review_gate": (
            "Only rows with normalized review_decision='accepted_box' are promoted from proposal hints to labels."
        ),
        "training_precondition": (
            "Before using this output for training, register/classify the out-root in the data lifecycle registry "
            "and run scripts/check_data_lifecycle_registry.py."
        ),
        "manifest_preview": manifest_preview,
    }
    if not args.dry_run:
        write_data_yaml(
            out_root=out_root,
            out_config=args.out_config,
            official_names=official_names,
            bridge_summary=bridge_summary,
            selected_splits=selected_splits,
        )
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(bridge_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out_root": repo_rel(out_root),
                "dry_run": bool(args.dry_run),
                "accepted_boxes": accepted_boxes,
                "scope": args.scope,
                "splits": {split: summary["images"] for split, summary in split_summaries.items()},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
