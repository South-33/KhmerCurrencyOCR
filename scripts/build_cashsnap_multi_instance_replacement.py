#!/usr/bin/env python
"""Build a real-scene multi-instance replacement synthetic probe.

This is a diagnostic step-change probe, not promoted training data. It uses
CashSnap train photos as scene templates, erases every known source YOLO box,
then replaces those boxes with approved same-class synthetic note assets. The
goal is to keep target-domain camera/context/multi-note layout while removing
source foreground pixels under an auditable label-preserving contract.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageOps

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("build_cashsnap_multi_instance_replacement.py requires opencv-python/cv2") from exc


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_cashsnap_target_anchor_transplant import (  # noqa: E402
    CLASS_NAMES,
    CLASS_TO_ID,
    alpha_composite,
    feather_warped_alpha,
    load_assets,
    make_contact_shadow,
    mask_to_label,
    poisson_composite,
    prepare_note,
    repo_rel,
    tone_match_note,
    warp_rgba,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SOURCE_PREFIXES = {
    "asian_currency_": "asian_currency",
    "billsbank_": "billsbank",
    "cambodia_currency_project_": "cambodia_currency_project",
    "cashcountingxl_": "cashcountingxl",
    "khmer_us_currency_": "khmer_us_currency",
    "usd_total_": "usd_total",
}


@dataclass(frozen=True)
class SourceBox:
    class_name: str
    cx: float
    cy: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cashsnap-root", type=Path, default=Path("data/cashsnap_v1"))
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Optional image list/CSV/JSONL limiting source scenes before other source filters.",
    )
    parser.add_argument("--asset-manifest", type=Path, default=Path("data/asset_candidates/numista_current_cutout_bank_v1/manifest.csv"))
    parser.add_argument("--out-root", type=Path, default=Path("data/synthetic/cashsnap_multi_instance_replacement_probe_v1"))
    parser.add_argument("--out-config", type=Path, default=Path("configs/webgl_ablation/cashsnap_multi_instance_replacement_probe_puresynth_realval_v1.yaml"))
    parser.add_argument("--max-images", type=int, default=80)
    parser.add_argument("--min-source-boxes", type=int, default=2)
    parser.add_argument("--max-source-boxes", type=int, default=8)
    parser.add_argument(
        "--source-selection-policy",
        choices=[
            "shuffled",
            "balanced_source_class",
            "balanced_source_class_with_reuse",
            "balanced_source_group_class",
        ],
        default="shuffled",
        help="How to choose eligible real source scenes before rendering.",
    )
    parser.add_argument(
        "--target-source-boxes-per-class",
        type=int,
        default=0,
        help="For balanced_source_class, greedily select sources until each source class reaches this box count.",
    )
    parser.add_argument(
        "--target-source-boxes-per-source-group-class",
        type=int,
        default=0,
        help="For balanced_source_group_class, target this many boxes for each available source_group/class pair.",
    )
    parser.add_argument(
        "--source-name-require-regex",
        default="",
        help="Optional regex that source image filenames must match, e.g. IMG_ for phone-photo contexts.",
    )
    parser.add_argument(
        "--source-name-block-regex",
        default="",
        help="Optional regex that rejects source image filenames, e.g. download|Screenshot|[0-9]+US.",
    )
    parser.add_argument(
        "--source-group-include",
        default="",
        help="Comma-separated inferred source groups to include, e.g. usd_total,billsbank.",
    )
    parser.add_argument(
        "--source-group-exclude",
        default="",
        help="Comma-separated inferred source groups to exclude.",
    )
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument(
        "--min-source-box-short-at-imgsz",
        type=float,
        default=0.0,
        help="Reject source images if any known source box would be smaller than this short side at --imgsz.",
    )
    parser.add_argument(
        "--min-visible-short-at-imgsz",
        type=float,
        default=0.0,
        help="Reject generated images if any final visible label is smaller than this short side at --imgsz.",
    )
    parser.add_argument("--imgsz", type=int, default=416, help="Training image size used by --min-source-box-short-at-imgsz.")
    parser.add_argument("--box-pad-fraction", type=float, default=0.035)
    parser.add_argument("--warp-alpha-feather-px", type=float, default=0.8)
    parser.add_argument(
        "--composite-policy",
        choices=["alpha", "poisson_mixed", "poisson_normal", "poisson_mixed_light", "poisson_mixed_edge"],
        default="alpha",
    )
    parser.add_argument("--shadow-policy", choices=["none", "contact"], default="contact")
    parser.add_argument(
        "--replacement-class-policy",
        choices=["same_source", "balanced_cycle"],
        default="same_source",
        help="Choose replacement class per erased source slot.",
    )
    parser.add_argument(
        "--replacement-classes",
        default=",".join(CLASS_NAMES),
        help="Comma-separated class pool for --replacement-class-policy balanced_cycle.",
    )
    parser.add_argument(
        "--tone-reference",
        choices=["inpainted_scene_patch", "original_source_crop", "blend_original_inpainted"],
        default="inpainted_scene_patch",
        help="Patch used for note tone matching. Original-source mode uses crop statistics, not source pixels.",
    )
    parser.add_argument(
        "--tone-blend-original-weight",
        type=float,
        default=0.65,
        help="Original-source crop weight for --tone-reference blend_original_inpainted.",
    )
    parser.add_argument("--sensor-noise", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--status", default="in_circulation")
    parser.add_argument("--sides", default="front,back")
    parser.add_argument("--asset-quality-policy", choices=["latest_design", "all_manifest"], default="latest_design")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed = (ROOT / "data" / "synthetic").resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise SystemExit(f"refusing to clean outside {repo_rel(allowed)}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def label_path_for_image(image_path: Path, cashsnap_root: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return cashsnap_root / "labels" / "train" / f"{image_path.stem}.txt"
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def iter_train_images(cashsnap_root: Path) -> list[Path]:
    image_dir = cashsnap_root / "images" / "train"
    rows: list[Path] = []
    for ext in IMAGE_EXTS:
        rows.extend(image_dir.glob(f"*{ext}"))
    return sorted(rows)


def unique_images(images: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for image in images:
        key = image.resolve()
        if key in seen:
            continue
        seen.add(key)
        out.append(image)
    return out


def read_source_manifest(path: Path) -> list[Path]:
    images: list[Path] = []
    if path.suffix.lower() == ".jsonl":
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            image = payload.get("image")
            if image:
                images.append(resolve(Path(str(image))))
    elif path.suffix.lower() == ".csv":
        import csv

        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "image" not in (reader.fieldnames or []):
                raise SystemExit(f"{repo_rel(path)} must include an image column")
            images.extend(resolve(Path(row["image"])) for row in reader if row.get("image"))
    else:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                images.append(resolve(Path(line)))
    return unique_images(images)


def read_boxes(label_path: Path) -> list[SourceBox]:
    if not label_path.exists():
        return []
    boxes: list[SourceBox] = []
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no}: expected 5 YOLO fields")
        class_id = int(float(parts[0]))
        if not 0 <= class_id < len(CLASS_NAMES):
            raise SystemExit(f"{repo_rel(label_path)}:{line_no}: class id {class_id} outside schema")
        cx, cy, width, height = [float(value) for value in parts[1:]]
        if width <= 0.0 or height <= 0.0:
            continue
        boxes.append(SourceBox(CLASS_NAMES[class_id], cx, cy, width, height))
    return boxes


def source_name_allowed(image_path: Path, args: argparse.Namespace) -> bool:
    name = image_path.name
    if args.source_name_require_regex and not re.search(args.source_name_require_regex, name, flags=re.IGNORECASE):
        return False
    if args.source_name_block_regex and re.search(args.source_name_block_regex, name, flags=re.IGNORECASE):
        return False
    return True


def source_group_for_image(image_path: Path) -> str:
    name = image_path.name.lower()
    for prefix, group in SOURCE_PREFIXES.items():
        if name.startswith(prefix):
            return group
    return name.split("_", 1)[0] if "_" in name else "unknown"


def parse_source_groups(value: str) -> set[str]:
    return {item.strip() for item in value.replace(";", ",").split(",") if item.strip()}


def source_group_allowed(image_path: Path, args: argparse.Namespace) -> bool:
    source_group = source_group_for_image(image_path)
    include = parse_source_groups(args.source_group_include)
    exclude = parse_source_groups(args.source_group_exclude)
    if include and source_group not in include:
        return False
    if exclude and source_group in exclude:
        return False
    return True


def parse_class_pool(value: str) -> list[str]:
    classes = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    unknown = [class_name for class_name in classes if class_name not in CLASS_TO_ID]
    if unknown:
        raise SystemExit(f"unknown replacement classes: {', '.join(unknown)}")
    if not classes:
        raise SystemExit("--replacement-classes selected no classes")
    return classes


def min_train_short_px(boxes: list[SourceBox], imgsz: int) -> float:
    if not boxes:
        return 0.0
    return min(min(box.width, box.height) * float(imgsz) for box in boxes)


def choose_sources(args: argparse.Namespace, rng: random.Random) -> list[tuple[Path, list[SourceBox]]]:
    cashsnap_root = resolve(args.cashsnap_root)
    source_images = (
        read_source_manifest(resolve(args.source_manifest))
        if args.source_manifest is not None
        else iter_train_images(cashsnap_root)
    )
    candidates: list[tuple[Path, list[SourceBox]]] = []
    for image_path in source_images:
        if not source_name_allowed(image_path, args):
            continue
        if not source_group_allowed(image_path, args):
            continue
        boxes = read_boxes(label_path_for_image(image_path, cashsnap_root))
        if len(boxes) < args.min_source_boxes:
            continue
        if args.max_source_boxes > 0 and len(boxes) > args.max_source_boxes:
            continue
        if args.min_source_box_short_at_imgsz > 0 and min_train_short_px(boxes, args.imgsz) < args.min_source_box_short_at_imgsz:
            continue
        candidates.append((image_path, boxes))
    if not candidates:
        raise SystemExit("no multi-instance source images selected")
    rng.shuffle(candidates)
    return candidates


def source_class_counts(boxes: list[SourceBox]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for box in boxes:
        counts[box.class_name] += 1
    return counts


def source_group_class_counts(source: tuple[Path, list[SourceBox]]) -> Counter[str]:
    image_path, boxes = source
    source_group = source_group_for_image(image_path)
    counts: Counter[str] = Counter()
    for box in boxes:
        counts[f"{source_group}|{box.class_name}"] += 1
    return counts


def select_balanced_sources(
    sources: list[tuple[Path, list[SourceBox]]],
    *,
    target_boxes_per_class: int,
    max_images: int,
    rng: random.Random,
) -> tuple[list[tuple[Path, list[SourceBox]]], dict[str, int]]:
    if target_boxes_per_class <= 0:
        raise SystemExit("--target-source-boxes-per-class must be > 0 for balanced_source_class")
    remaining = list(sources)
    selected: list[tuple[Path, list[SourceBox]]] = []
    counts: Counter[str] = Counter()

    while remaining:
        if max_images > 0 and len(selected) >= max_images:
            break
        if all(counts[class_name] >= target_boxes_per_class for class_name in CLASS_NAMES):
            break

        best_score = 0.0
        best_indexes: list[int] = []
        for index, (_, boxes) in enumerate(remaining):
            box_counts = source_class_counts(boxes)
            score = 0.0
            for class_name, box_count in box_counts.items():
                deficit = max(0, target_boxes_per_class - counts[class_name])
                if deficit <= 0:
                    continue
                score += min(deficit, box_count) * (1.0 + deficit / target_boxes_per_class)
            if score > best_score:
                best_score = score
                best_indexes = [index]
            elif score == best_score and score > 0:
                best_indexes.append(index)
        if best_score <= 0 or not best_indexes:
            break

        chosen_index = rng.choice(best_indexes)
        source = remaining.pop(chosen_index)
        selected.append(source)
        counts.update(source_class_counts(source[1]))

    return selected, {class_name: counts.get(class_name, 0) for class_name in CLASS_NAMES}


def select_balanced_source_group_class(
    sources: list[tuple[Path, list[SourceBox]]],
    *,
    target_boxes_per_group_class: int,
    max_images: int,
    rng: random.Random,
) -> tuple[list[tuple[Path, list[SourceBox]]], dict[str, int]]:
    if target_boxes_per_group_class <= 0:
        raise SystemExit(
            "--target-source-boxes-per-source-group-class must be > 0 for balanced_source_group_class"
        )
    target_keys = sorted({key for source in sources for key in source_group_class_counts(source)})
    if not target_keys:
        raise SystemExit("no source_group/class keys available for balanced_source_group_class")

    remaining = list(sources)
    selected: list[tuple[Path, list[SourceBox]]] = []
    counts: Counter[str] = Counter()
    while remaining:
        if max_images > 0 and len(selected) >= max_images:
            break
        if all(counts[key] >= target_boxes_per_group_class for key in target_keys):
            break

        best_score = 0.0
        best_indexes: list[int] = []
        for index, source in enumerate(remaining):
            source_counts = source_group_class_counts(source)
            score = 0.0
            for key, box_count in source_counts.items():
                deficit = max(0, target_boxes_per_group_class - counts[key])
                if deficit <= 0:
                    continue
                score += min(deficit, box_count) * (1.0 + deficit / target_boxes_per_group_class)
            if score > best_score:
                best_score = score
                best_indexes = [index]
            elif score == best_score and score > 0:
                best_indexes.append(index)
        if best_score <= 0 or not best_indexes:
            break

        chosen_index = rng.choice(best_indexes)
        source = remaining.pop(chosen_index)
        selected.append(source)
        counts.update(source_group_class_counts(source))

    return selected, {key: counts.get(key, 0) for key in target_keys}


def select_balanced_sources_with_reuse(
    sources: list[tuple[Path, list[SourceBox]]],
    *,
    target_boxes_per_class: int,
    max_images: int,
    rng: random.Random,
) -> tuple[list[tuple[Path, list[SourceBox]]], dict[str, int], dict[str, Any]]:
    if target_boxes_per_class <= 0:
        raise SystemExit("--target-source-boxes-per-class must be > 0 for balanced_source_class_with_reuse")
    by_class: dict[str, list[tuple[Path, list[SourceBox]]]] = {class_name: [] for class_name in CLASS_NAMES}
    for source in sources:
        for class_name in source_class_counts(source[1]):
            by_class[class_name].append(source)
    missing = [class_name for class_name, rows in by_class.items() if not rows]
    if missing:
        raise SystemExit(f"cannot reuse-balance missing source classes: {', '.join(missing)}")

    selected: list[tuple[Path, list[SourceBox]]] = []
    counts: Counter[str] = Counter()
    use_counts: Counter[str] = Counter()
    while not all(counts[class_name] >= target_boxes_per_class for class_name in CLASS_NAMES):
        if max_images > 0 and len(selected) >= max_images:
            break
        deficits = [
            (target_boxes_per_class - counts[class_name], class_name)
            for class_name in CLASS_NAMES
            if counts[class_name] < target_boxes_per_class
        ]
        max_deficit = max(deficit for deficit, _ in deficits)
        deficit_classes = [class_name for deficit, class_name in deficits if deficit == max_deficit]
        class_name = rng.choice(deficit_classes)
        candidates = by_class[class_name]
        min_use = min(use_counts[source[0].as_posix()] for source in candidates)
        least_used = [source for source in candidates if use_counts[source[0].as_posix()] == min_use]
        source = rng.choice(least_used)
        selected.append(source)
        counts.update(source_class_counts(source[1]))
        use_counts[source[0].as_posix()] += 1

    reuse_values = list(use_counts.values())
    reuse_report = {
        "unique_source_images": len(use_counts),
        "reused_source_images": sum(1 for value in reuse_values if value > 1),
        "max_source_reuse_count": max(reuse_values) if reuse_values else 0,
        "source_reuse_counts": {
            repo_rel(Path(path)): value
            for path, value in sorted(use_counts.items(), key=lambda item: (-item[1], item[0]))
            if value > 1
        },
    }
    return selected, {class_name: counts.get(class_name, 0) for class_name in CLASS_NAMES}, reuse_report


def build_replacement_plans(
    sources: list[tuple[Path, list[SourceBox]]],
    *,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[list[str]]:
    ordered_counts = [len(sorted(boxes, key=lambda row: row.width * row.height, reverse=True)) for _, boxes in sources]
    if args.replacement_class_policy == "same_source":
        return [
            [box.class_name for box in sorted(boxes, key=lambda row: row.width * row.height, reverse=True)]
            for _, boxes in sources
        ]
    class_pool = parse_class_pool(args.replacement_classes)
    total_slots = sum(ordered_counts)
    class_rows: list[str] = []
    while len(class_rows) < total_slots:
        cycle = class_pool.copy()
        rng.shuffle(cycle)
        class_rows.extend(cycle)
    plans: list[list[str]] = []
    cursor = 0
    for count in ordered_counts:
        plans.append(class_rows[cursor : cursor + count])
        cursor += count
    return plans


def resize_for_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image.convert("RGB")
    scale = min(1.0, float(max_side) / float(max(image.size)))
    if scale >= 1.0:
        return image.convert("RGB")
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return ImageOps.exif_transpose(image).convert("RGB").resize(size, Image.Resampling.LANCZOS)


def box_xyxy(box: SourceBox, width: int, height: int, pad_fraction: float = 0.0) -> tuple[int, int, int, int]:
    pad_w = box.width * pad_fraction
    pad_h = box.height * pad_fraction
    x1 = int(round((box.cx - box.width / 2.0 - pad_w) * width))
    y1 = int(round((box.cy - box.height / 2.0 - pad_h) * height))
    x2 = int(round((box.cx + box.width / 2.0 + pad_w) * width))
    y2 = int(round((box.cy + box.height / 2.0 + pad_h) * height))
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def inpaint_known_source_boxes(
    base_rgb: np.ndarray,
    boxes: list[SourceBox],
    *,
    pad_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = base_rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    box_rows: list[list[int]] = []
    for box in boxes:
        x1, y1, x2, y2 = box_xyxy(box, width, height, pad_fraction)
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = 255
        box_rows.append([x1, y1, x2, y2])
    if not mask.any():
        return base_rgb, {"mask_area_px": 0, "mask_fraction": 0.0, "boxes": box_rows}
    base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(base_bgr, mask, 5.0, cv2.INPAINT_TELEA)
    out = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    return out, {
        "mask_area_px": int(mask.astype(bool).sum()),
        "mask_fraction": round(float(mask.astype(bool).mean()), 6),
        "boxes": box_rows,
    }


def quad_for_box(box: SourceBox, width: int, height: int, rng: random.Random) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy(box, width, height, 0.0)
    if x2 <= x1 or y2 <= y1:
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    jitter = max(1.0, min(x2 - x1, y2 - y1) * 0.025)
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    offsets = np.array(
        [[rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)] for _ in range(4)],
        dtype=np.float32,
    )
    corners += offsets
    corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
    return corners


def patch_for_box(base: Image.Image, box: SourceBox) -> Image.Image:
    x1, y1, x2, y2 = box_xyxy(box, base.width, base.height, 0.0)
    if x2 <= x1 or y2 <= y1:
        return base
    return base.crop((x1, y1, x2, y2))


def tone_patch_for_box(
    *,
    original_base: Image.Image,
    inpainted_base: Image.Image,
    box: SourceBox,
    args: argparse.Namespace,
) -> Image.Image:
    if args.tone_reference == "original_source_crop":
        return patch_for_box(original_base, box)
    if args.tone_reference == "blend_original_inpainted":
        original_patch = patch_for_box(original_base, box).convert("RGB")
        inpainted_patch = patch_for_box(inpainted_base, box).convert("RGB")
        if original_patch.size != inpainted_patch.size:
            inpainted_patch = inpainted_patch.resize(original_patch.size, Image.Resampling.BILINEAR)
        weight = float(np.clip(args.tone_blend_original_weight, 0.0, 1.0))
        return Image.blend(inpainted_patch, original_patch, weight)
    return patch_for_box(inpainted_base, box)


def poisson_mixed_light_composite(base_rgb: np.ndarray, layer_rgba: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = layer_rgba[:, :, 3]
    mask = (alpha > 18).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return base_rgb
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    if (x2 - x1) < 16 or (y2 - y1) < 12:
        return alpha_composite(base_rgb, layer_rgba)
    pad = 18
    roi_x1 = max(0, x1 - pad)
    roi_y1 = max(0, y1 - pad)
    roi_x2 = min(base_rgb.shape[1], x2 + pad)
    roi_y2 = min(base_rgb.shape[0], y2 + pad)
    center = (int((x1 + x2) / 2) - roi_x1, int((y1 + y2) / 2) - roi_y1)
    source_rgb = layer_rgba[:, :, :3].copy()
    source_rgb[mask == 0] = base_rgb[mask == 0]
    source_roi = source_rgb[roi_y1:roi_y2, roi_x1:roi_x2]
    base_roi = base_rgb[roi_y1:roi_y2, roi_x1:roi_x2]
    layer_roi = layer_rgba[roi_y1:roi_y2, roi_x1:roi_x2]
    mask_roi = mask[roi_y1:roi_y2, roi_x1:roi_x2]
    try:
        cloned_roi_bgr = cv2.seamlessClone(
            cv2.cvtColor(source_roi, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(base_roi, cv2.COLOR_RGB2BGR),
            mask_roi,
            center,
            cv2.MIXED_CLONE,
        )
    except cv2.error:
        return alpha_composite(base_rgb, layer_rgba)
    cloned_roi_rgb = cv2.cvtColor(cloned_roi_bgr, cv2.COLOR_BGR2RGB)
    direct_roi_rgb = alpha_composite(base_roi, layer_roi)
    strength = rng.uniform(0.18, 0.38)
    blended = np.clip(
        cloned_roi_rgb.astype(np.float32) * strength + direct_roi_rgb.astype(np.float32) * (1.0 - strength),
        0,
        255,
    )
    out = base_rgb.copy()
    roi_out = out[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_out[mask_roi > 0] = blended[mask_roi > 0].astype(np.uint8)
    out[roi_y1:roi_y2, roi_x1:roi_x2] = roi_out
    return out


def poisson_mixed_edge_composite(base_rgb: np.ndarray, layer_rgba: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = layer_rgba[:, :, 3]
    mask = (alpha > 18).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return base_rgb
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    if (x2 - x1) < 16 or (y2 - y1) < 12:
        return alpha_composite(base_rgb, layer_rgba)
    pad = 18
    roi_x1 = max(0, x1 - pad)
    roi_y1 = max(0, y1 - pad)
    roi_x2 = min(base_rgb.shape[1], x2 + pad)
    roi_y2 = min(base_rgb.shape[0], y2 + pad)
    center = (int((x1 + x2) / 2) - roi_x1, int((y1 + y2) / 2) - roi_y1)
    source_rgb = layer_rgba[:, :, :3].copy()
    source_rgb[mask == 0] = base_rgb[mask == 0]
    source_roi = source_rgb[roi_y1:roi_y2, roi_x1:roi_x2]
    base_roi = base_rgb[roi_y1:roi_y2, roi_x1:roi_x2]
    layer_roi = layer_rgba[roi_y1:roi_y2, roi_x1:roi_x2]
    mask_roi = mask[roi_y1:roi_y2, roi_x1:roi_x2]
    try:
        cloned_roi_bgr = cv2.seamlessClone(
            cv2.cvtColor(source_roi, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(base_roi, cv2.COLOR_RGB2BGR),
            mask_roi,
            center,
            cv2.MIXED_CLONE,
        )
    except cv2.error:
        return alpha_composite(base_rgb, layer_rgba)
    cloned_roi_rgb = cv2.cvtColor(cloned_roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    direct_roi_rgb = alpha_composite(base_roi, layer_roi).astype(np.float32)
    dist = cv2.distanceTransform((mask_roi > 0).astype(np.uint8), cv2.DIST_L2, 3)
    band_px = rng.uniform(7.0, 15.0)
    edge_weight = np.clip((band_px - dist) / max(1e-3, band_px), 0.0, 1.0)
    interior_strength = rng.uniform(0.18, 0.34)
    edge_strength = rng.uniform(0.68, 0.88)
    strength = interior_strength + edge_weight * (edge_strength - interior_strength)
    blended = np.clip(
        cloned_roi_rgb * strength[:, :, None] + direct_roi_rgb * (1.0 - strength[:, :, None]),
        0,
        255,
    )
    out = base_rgb.copy()
    roi_out = out[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_out[mask_roi > 0] = blended[mask_roi > 0].astype(np.uint8)
    out[roi_y1:roi_y2, roi_x1:roi_x2] = roi_out
    return out


def build_replacement(
    *,
    source_image: Path,
    boxes: list[SourceBox],
    replacement_classes: list[str],
    assets_by_class: dict[str, list[Any]],
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[Image.Image, list[str], dict[str, Any]]:
    with Image.open(source_image) as opened:
        base = resize_for_max_side(opened, args.max_side)
    width, height = base.size
    original_for_tone = base.copy()
    base_rgb = np.asarray(base).copy()
    base_rgb, inpaint_meta = inpaint_known_source_boxes(
        base_rgb,
        boxes,
        pad_fraction=args.box_pad_fraction,
    )
    base_for_tone = Image.fromarray(base_rgb, "RGB")

    ordered = sorted(boxes, key=lambda row: row.width * row.height, reverse=True)
    layers: list[tuple[str, str, np.ndarray, np.ndarray, Any]] = []
    for box, replacement_class in zip(ordered, replacement_classes, strict=True):
        candidates = assets_by_class.get(replacement_class, [])
        if not candidates:
            continue
        asset = rng.choice(candidates)
        note = prepare_note(asset, rng)
        note = tone_match_note(
            note,
            tone_patch_for_box(
                original_base=original_for_tone,
                inpainted_base=base_for_tone,
                box=box,
                args=args,
            ),
            rng,
            "background",
        )
        quad = quad_for_box(box, width, height, rng)
        layer = warp_rgba(note, quad, (width, height))
        layer = feather_warped_alpha(layer, args.warp_alpha_feather_px)
        if args.shadow_policy == "contact":
            base_rgb = alpha_composite(base_rgb, make_contact_shadow(layer[:, :, 3], rng))
        if args.composite_policy == "alpha":
            base_rgb = alpha_composite(base_rgb, layer)
        elif args.composite_policy == "poisson_mixed_light":
            base_rgb = poisson_mixed_light_composite(base_rgb, layer, rng)
        elif args.composite_policy == "poisson_mixed_edge":
            base_rgb = poisson_mixed_edge_composite(base_rgb, layer, rng)
        else:
            base_rgb = poisson_composite(base_rgb, layer, args.composite_policy, rng)
        layers.append((replacement_class, box.class_name, layer, quad, asset))

    covered_later = np.zeros((height, width), dtype=bool)
    labels_reversed: list[str] = []
    instances_reversed: list[dict[str, Any]] = []
    visible_area_by_class: Counter[str] = Counter()
    visible_quality_failures: list[dict[str, Any]] = []
    for class_name, source_class_name, layer, quad, asset in reversed(layers):
        visible = (layer[:, :, 3] > 18) & ~covered_later
        covered_later |= layer[:, :, 3] > 18
        if int(visible.sum()) < 80:
            continue
        label = mask_to_label(visible, class_name)
        if label is None:
            continue
        parts = label.split()
        visible_short_at_imgsz = min(float(parts[3]), float(parts[4])) * float(args.imgsz)
        if args.min_visible_short_at_imgsz > 0 and visible_short_at_imgsz < args.min_visible_short_at_imgsz:
            visible_quality_failures.append(
                {
                    "class_name": class_name,
                    "source_class_name": source_class_name,
                    "visible_short_at_imgsz": round(float(visible_short_at_imgsz), 3),
                    "threshold": float(args.min_visible_short_at_imgsz),
                }
            )
        labels_reversed.append(label)
        instances_reversed.append(
            {
                "class_name": class_name,
                "source_class_name": source_class_name,
                "label": label,
                "asset_path": repo_rel(asset.path),
                "asset_side": asset.side,
                "asset_years": asset.years,
                "asset_status": asset.status,
                "quad_xy": [[round(float(x), 3), round(float(y), 3)] for x, y in quad.tolist()],
                "visible_area_px": int(visible.sum()),
                "visible_short_at_imgsz": round(float(visible_short_at_imgsz), 3),
                "composite_policy": args.composite_policy,
                "shadow_policy": args.shadow_policy,
                "tone_reference": args.tone_reference,
            }
        )
        visible_area_by_class[class_name] += int(visible.sum())

    labels = list(reversed(labels_reversed))
    instances = list(reversed(instances_reversed))
    metadata = {
        "source_image": repo_rel(source_image),
        "source_group": source_group_for_image(source_image),
        "source_boxes": len(boxes),
        "output_size": [width, height],
        "labels": len(labels),
        "instances": instances,
        "inpaint": inpaint_meta,
        "source_classes": dict(Counter(box.class_name for box in boxes)),
        "replacement_classes": dict(Counter(replacement_classes)),
        "tone_reference": args.tone_reference,
        "visible_quality_failures": visible_quality_failures,
        "visible_area_by_class": dict(sorted(visible_area_by_class.items())),
    }
    return Image.fromarray(base_rgb, "RGB"), labels, metadata


def write_data_yaml(path: Path, out_root: Path) -> None:
    try:
        dataset_path = os.path.relpath(out_root, path.parent).replace("\\", "/")
    except ValueError:
        dataset_path = out_root.as_posix()
    payload = {
        "path": dataset_path,
        "train": "images/train",
        "val": "images/train",
        "test": "images/train",
        "nc": len(CLASS_NAMES),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        "cashsnap_synthetic_policy": {
            "generator": "build_cashsnap_multi_instance_replacement.py",
            "status": "diagnostic_only",
            "reason": "Real train scene replacement probe; not trainable until source-remnant audit, representation gap, and real-transfer probes pass.",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    out_root = resolve(args.out_root)
    if args.clean:
        safe_clean(out_root)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "metadata").mkdir(parents=True, exist_ok=True)

    statuses = {value.strip() for value in args.status.replace(";", ",").split(",") if value.strip()}
    sides = {value.strip() for value in args.sides.replace(";", ",").split(",") if value.strip()}
    assets_by_class = load_assets(resolve(args.asset_manifest), statuses, sides, args.asset_quality_policy)
    sources = choose_sources(args, rng)
    eligible_sources = len(sources)
    eligible_source_group_counts = Counter(source_group_for_image(source_image) for source_image, _ in sources)
    selected_source_class_counts: dict[str, int] | None = None
    selected_source_group_class_counts: dict[str, int] | None = None
    source_reuse_report: dict[str, Any] | None = None
    if args.source_selection_policy == "balanced_source_class":
        sources, selected_source_class_counts = select_balanced_sources(
            sources,
            target_boxes_per_class=args.target_source_boxes_per_class,
            max_images=args.max_images,
            rng=rng,
        )
    elif args.source_selection_policy == "balanced_source_class_with_reuse":
        sources, selected_source_class_counts, source_reuse_report = select_balanced_sources_with_reuse(
            sources,
            target_boxes_per_class=args.target_source_boxes_per_class,
            max_images=args.max_images,
            rng=rng,
        )
    elif args.source_selection_policy == "balanced_source_group_class":
        sources, selected_source_group_class_counts = select_balanced_source_group_class(
            sources,
            target_boxes_per_group_class=args.target_source_boxes_per_source_group_class,
            max_images=args.max_images,
            rng=rng,
        )
    elif args.target_source_boxes_per_class > 0:
        raise SystemExit(
            "--target-source-boxes-per-class requires --source-selection-policy "
            "balanced_source_class or balanced_source_class_with_reuse"
        )
    if args.target_source_boxes_per_source_group_class > 0 and args.source_selection_policy != "balanced_source_group_class":
        raise SystemExit(
            "--target-source-boxes-per-source-group-class requires --source-selection-policy "
            "balanced_source_group_class"
        )
    replacement_plans = build_replacement_plans(sources, args=args, rng=rng)

    records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    source_box_count = 0
    skipped_visible_quality = 0
    selected_source_group_counts: Counter[str] = Counter()
    for index, ((source_image, boxes), replacement_classes) in enumerate(zip(sources, replacement_plans, strict=True)):
        if args.max_images > 0 and len(records) >= args.max_images:
            break
        image, labels, metadata = build_replacement(
            source_image=source_image,
            boxes=boxes,
            replacement_classes=replacement_classes,
            assets_by_class=assets_by_class,
            rng=rng,
            args=args,
        )
        if metadata.get("visible_quality_failures"):
            skipped_visible_quality += 1
            continue
        if not labels:
            continue
        stem = f"cashsnap_multi_replace_{len(records):06d}_{source_image.stem}"
        image_path = out_root / "images" / "train" / f"{stem}.jpg"
        label_path = out_root / "labels" / "train" / f"{stem}.txt"
        metadata_path = out_root / "metadata" / f"{stem}.json"
        image.save(image_path, quality=92)
        label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        image_rel = image_path.relative_to(out_root).as_posix()
        for instance in metadata["instances"]:
            edge_records.append(
                {
                    "image": image_rel,
                    "class_name": instance["class_name"],
                    "quad_xy": instance["quad_xy"],
                    "composite_policy": instance["composite_policy"],
                    "shadow_policy": instance["shadow_policy"],
                    "metadata": metadata_path.relative_to(out_root).as_posix(),
                }
            )
        for label in labels:
            class_counts[CLASS_NAMES[int(label.split()[0])]] += 1
        source_box_count += len(boxes)
        selected_source_group_counts[source_group_for_image(source_image)] += 1
        records.append(
            {
                "image": repo_rel(image_path),
                "label": repo_rel(label_path),
                "metadata": repo_rel(metadata_path),
                "source_image": repo_rel(source_image),
                "source_group": source_group_for_image(source_image),
                "source_boxes": len(boxes),
                "labels": len(labels),
            }
        )

    data_yaml = out_root / "data.yaml"
    (out_root / "metadata" / "train.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in edge_records) + ("\n" if edge_records else ""),
        encoding="utf-8",
    )
    write_data_yaml(data_yaml, out_root)
    write_data_yaml(resolve(args.out_config), out_root)
    summary = {
        "schema": "cashsnap_multi_instance_replacement_summary_v1",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": "cashsnap train multi-label real scenes",
        "out_root": repo_rel(out_root),
        "data_yaml": repo_rel(data_yaml),
        "out_config": repo_rel(resolve(args.out_config)),
        "images": len(records),
        "eligible_source_images": eligible_sources,
        "source_images_considered": len(sources),
        "source_boxes": source_box_count,
        "labels": sum(int(row["labels"]) for row in records),
        "class_counts": dict(sorted(class_counts.items())),
        "eligible_source_group_counts": dict(sorted(eligible_source_group_counts.items())),
        "selected_source_group_counts": dict(sorted(selected_source_group_counts.items())),
        "selected_source_class_counts": selected_source_class_counts,
        "selected_source_group_class_counts": selected_source_group_class_counts,
        "source_reuse_report": source_reuse_report,
        "args": {
            "min_source_boxes": args.min_source_boxes,
            "max_source_boxes": args.max_source_boxes,
            "source_manifest": repo_rel(resolve(args.source_manifest)) if args.source_manifest is not None else None,
            "source_selection_policy": args.source_selection_policy,
            "target_source_boxes_per_class": args.target_source_boxes_per_class,
            "target_source_boxes_per_source_group_class": args.target_source_boxes_per_source_group_class,
            "source_name_require_regex": args.source_name_require_regex,
            "source_name_block_regex": args.source_name_block_regex,
            "source_group_include": args.source_group_include,
            "source_group_exclude": args.source_group_exclude,
            "replacement_class_policy": args.replacement_class_policy,
            "replacement_classes": args.replacement_classes,
            "max_side": args.max_side,
            "imgsz": args.imgsz,
            "min_source_box_short_at_imgsz": args.min_source_box_short_at_imgsz,
            "min_visible_short_at_imgsz": args.min_visible_short_at_imgsz,
            "box_pad_fraction": args.box_pad_fraction,
            "warp_alpha_feather_px": args.warp_alpha_feather_px,
            "composite_policy": args.composite_policy,
            "shadow_policy": args.shadow_policy,
            "tone_reference": args.tone_reference,
            "tone_blend_original_weight": args.tone_blend_original_weight,
            "seed": args.seed,
        },
        "skipped_visible_quality": skipped_visible_quality,
        "records": records,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"multi_instance_replacement images={len(records)} labels={summary['labels']} "
        f"source_boxes={source_box_count} data={repo_rel(data_yaml)}"
    )
    if not records:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
