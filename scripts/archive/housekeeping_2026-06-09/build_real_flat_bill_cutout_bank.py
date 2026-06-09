#!/usr/bin/env python
"""Mine and rectify real flat-bill cutouts from CashSnap YOLO labels.

The pipeline is staged on purpose:

1. ``mine`` selects train-split YOLO boxes that look like mostly visible bills
   and writes cropped BEN2 inputs plus a candidate manifest.
2. A local background remover writes transparent PNGs for those candidate stems.
3. ``rectify`` fits the alpha foreground, perspective-warps the bill to a
   landscape transparent asset, and writes a cutout-bank manifest.

The resulting bank is diagnostic/probe material. It keeps ``side=unknown`` and
``status=real_train_flat_probe`` so render recipes must opt in explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("build_real_flat_bill_cutout_bank.py requires opencv-python/cv2") from exc


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_cashsnap_target_anchor_transplant import CLASS_NAMES, CLASS_TO_ID, repo_rel  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_OUT = ROOT / "data" / "asset_candidates" / "cashsnap_real_flat_bill_cutout_bank_probe_v1"
PROBE_STATUS = "real_train_flat_probe"
PROBE_SIDE = "unknown"


@dataclass(frozen=True)
class LabelBox:
    image_path: Path
    label_path: Path
    split: str
    class_id: int
    class_name: str
    cx: float
    cy: float
    width: float
    height: float
    image_width: int
    image_height: int
    line_no: int

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        x1 = (self.cx - self.width / 2.0) * self.image_width
        y1 = (self.cy - self.height / 2.0) * self.image_height
        x2 = (self.cx + self.width / 2.0) * self.image_width
        y2 = (self.cy + self.height / 2.0) * self.image_height
        return x1, y1, x2, y2

    @property
    def box_width_px(self) -> float:
        return self.width * self.image_width

    @property
    def box_height_px(self) -> float:
        return self.height * self.image_height

    @property
    def short_px(self) -> float:
        return min(self.box_width_px, self.box_height_px)

    @property
    def long_px(self) -> float:
        return max(self.box_width_px, self.box_height_px)

    @property
    def area_ratio(self) -> float:
        return self.width * self.height

    @property
    def aspect_norm(self) -> float:
        raw = self.box_width_px / max(1.0, self.box_height_px)
        return max(raw, 1.0 / max(raw, 1e-6))

    @property
    def edge_margin_frac(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        margins = [
            x1 / max(1.0, self.image_width),
            y1 / max(1.0, self.image_height),
            (self.image_width - x2) / max(1.0, self.image_width),
            (self.image_height - y2) / max(1.0, self.image_height),
        ]
        return min(margins)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    mine = subparsers.add_parser("mine", help="Mine YOLO boxes and write BEN2 crop inputs.")
    mine.add_argument("--data-yaml", type=Path, default=Path("data/cashsnap_v1/data.yaml"))
    mine.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    mine.add_argument("--splits", default="train", help="Comma-separated splits to mine; default keeps held-out data untouched.")
    mine.add_argument("--classes", default="", help="Optional comma-separated class names.")
    mine.add_argument(
        "--source-name-require-regex",
        default="",
        help="Optional regex that source image filenames must match before mining.",
    )
    mine.add_argument(
        "--source-name-block-regex",
        default="",
        help="Optional regex that rejects source image filenames before mining.",
    )
    mine.add_argument("--min-boxes-per-image", type=int, default=1)
    mine.add_argument("--max-boxes-per-image", type=int, default=0, help="0 disables the upper bound.")
    mine.add_argument("--max-per-class", type=int, default=32)
    mine.add_argument("--max-per-source-group", type=int, default=2)
    mine.add_argument("--crop-pad-frac", type=float, default=0.08)
    mine.add_argument("--min-short-px", type=float, default=180.0)
    mine.add_argument("--min-area-ratio", type=float, default=0.08)
    mine.add_argument("--max-area-ratio", type=float, default=0.96)
    mine.add_argument("--min-aspect-norm", type=float, default=1.05)
    mine.add_argument("--max-aspect-norm", type=float, default=4.20)
    mine.add_argument("--min-edge-margin-frac", type=float, default=-0.01)
    mine.add_argument("--max-overlap-frac", type=float, default=0.18)
    mine.add_argument("--seed", type=int, default=0)
    mine.add_argument("--clean-candidates", action="store_true")

    rectify = subparsers.add_parser("rectify", help="Rectify transparent candidates into a cutout bank.")
    rectify.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    rectify.add_argument("--candidate-manifest", type=Path, default=None)
    rectify.add_argument("--transparent-root", type=Path, default=None)
    rectify.add_argument("--alpha-threshold", type=int, default=16)
    rectify.add_argument("--min-alpha-area", type=int, default=3500)
    rectify.add_argument("--min-largest-component-ratio", type=float, default=0.82)
    rectify.add_argument("--min-rotated-rect-fill", type=float, default=0.54)
    rectify.add_argument("--min-rect-aspect", type=float, default=1.35)
    rectify.add_argument("--max-rect-aspect", type=float, default=4.20)
    rectify.add_argument("--max-output-long", type=int, default=1800)
    rectify.add_argument(
        "--no-remove-edge-dark-border",
        action="store_true",
        help="Disable conservative removal of near-black alpha pixels connected to the foreground boundary.",
    )
    rectify.add_argument("--edge-dark-threshold", type=int, default=70)
    rectify.add_argument(
        "--no-remove-alpha-islands",
        action="store_true",
        help="Disable post-warp removal of disconnected alpha islands outside the largest foreground component.",
    )
    rectify.add_argument("--clean-bank", action="store_true")
    rectify.add_argument("--copy-transparent-misses", action="store_true")

    easy = subparsers.add_parser("select-easy", help="Select a small easy-flat subset from an audited cutout bank.")
    easy.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    easy.add_argument("--manifest", type=Path, default=None)
    easy.add_argument("--audit-assets", type=Path, default=None)
    easy.add_argument("--out-manifest", type=Path, default=None)
    easy.add_argument("--max-per-class", type=int, default=6)
    easy.add_argument("--min-source-aspect-norm", type=float, default=1.35)
    easy.add_argument("--min-rect-aspect-norm", type=float, default=1.45)
    easy.add_argument("--max-rect-aspect-norm", type=float, default=3.35)
    easy.add_argument("--min-rotated-rect-fill", type=float, default=0.84)
    easy.add_argument("--min-bbox-fill-rectified", type=float, default=0.84)
    easy.add_argument("--min-largest-component-ratio", type=float, default=0.98)
    easy.add_argument("--max-component-count", type=int, default=20)
    easy.add_argument("--max-edge-dark-removed-px", type=int, default=2500)
    easy.add_argument("--max-overlap-fraction", type=float, default=0.05)
    easy.add_argument(
        "--manual-keep-file",
        type=Path,
        default=None,
        help="Optional newline file of candidate_id values to keep after auto easy-flat filtering.",
    )
    easy.add_argument(
        "--max-skin-like-ratio",
        type=float,
        default=1.0,
        help="Optional hard cap; default records the metric without filtering because KHR colors can false-positive.",
    )

    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def safe_clean(path: Path, allowed: Path) -> None:
    resolved = path.resolve()
    allowed = allowed.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise SystemExit(f"refusing to clean outside {repo_rel(allowed)}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def read_id_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    path = resolve(path)
    if not path.exists():
        raise SystemExit(f"missing id file: {repo_rel(path)}")
    ids = {
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not ids:
        raise SystemExit(f"no ids found in {repo_rel(path)}")
    return ids


def read_data_yaml(path: Path) -> tuple[Path, dict[int, str], dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(path)} must be a YAML mapping")
    names_raw = payload.get("names")
    if isinstance(names_raw, dict):
        names = {int(key): str(value) for key, value in names_raw.items()}
    elif isinstance(names_raw, list):
        names = {index: str(value) for index, value in enumerate(names_raw)}
    else:
        raise SystemExit(f"{repo_rel(path)} must include names as a list or mapping")
    root_value = payload.get("path")
    dataset_root = path.parent if root_value is None else Path(str(root_value))
    if not dataset_root.is_absolute():
        dataset_root = (path.parent / dataset_root).resolve()
    return dataset_root.resolve(), names, payload


def split_dir(dataset_root: Path, payload: dict[str, Any], split: str) -> Path:
    value = payload.get(split)
    if value is None and split == "val":
        value = payload.get("valid")
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"data.yaml missing split {split!r}")
    path = Path(value)
    if path.is_absolute():
        return path
    return (dataset_root / path).resolve()


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image_path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def iter_images(image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(image_dir.glob(f"*{ext}"))
    return sorted(paths)


def read_label_boxes(image_path: Path, split: str, names: dict[int, str]) -> list[LabelBox]:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    rows: list[LabelBox] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        parts = raw.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no}: expected 5 YOLO fields")
        class_id = int(float(parts[0]))
        if class_id not in names:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no}: class id {class_id} outside data.yaml names")
        if names[class_id] not in CLASS_TO_ID:
            continue
        cx, cy, width, height = [float(value) for value in parts[1:]]
        if width <= 0.0 or height <= 0.0:
            continue
        rows.append(
            LabelBox(
                image_path=image_path,
                label_path=label_path,
                split=split,
                class_id=CLASS_TO_ID[names[class_id]],
                class_name=names[class_id],
                cx=cx,
                cy=cy,
                width=width,
                height=height,
                image_width=image_width,
                image_height=image_height,
                line_no=line_no,
            )
        )
    return rows


def source_name_allowed(path: Path, args: argparse.Namespace) -> bool:
    name = path.name
    if args.source_name_require_regex and not re.search(args.source_name_require_regex, name, flags=re.IGNORECASE):
        return False
    if args.source_name_block_regex and re.search(args.source_name_block_regex, name, flags=re.IGNORECASE):
        return False
    return True


def intersection_area(a: LabelBox, b: LabelBox) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def max_overlap_fraction(box: LabelBox, boxes: list[LabelBox]) -> float:
    area = max(1.0, box.box_width_px * box.box_height_px)
    return max((intersection_area(box, other) / area for other in boxes if other is not box), default=0.0)


def source_group_for(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"\.rf\.[a-f0-9]+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_(jpg|jpeg|png|webp)$", "", stem, flags=re.IGNORECASE)
    return stem


def candidate_score(box: LabelBox, overlap: float) -> float:
    aspect_target = 2.25
    aspect_penalty = min(1.0, abs(math.log(max(1e-6, box.aspect_norm / aspect_target))) / math.log(2.0))
    edge_bonus = max(0.0, min(0.08, box.edge_margin_frac + 0.02)) * 3.0
    return (
        0.45 * min(1.0, box.short_px / 420.0)
        + 0.35 * min(1.0, box.area_ratio / 0.55)
        + 0.12 * (1.0 - aspect_penalty)
        + edge_bonus
        - 0.25 * overlap
    )


def box_passes_filters(box: LabelBox, overlap: float, args: argparse.Namespace) -> tuple[bool, str]:
    if box.short_px < args.min_short_px:
        return False, "short_px"
    if box.area_ratio < args.min_area_ratio:
        return False, "small_area_ratio"
    if box.area_ratio > args.max_area_ratio:
        return False, "large_area_ratio"
    if box.aspect_norm < args.min_aspect_norm:
        return False, "low_aspect"
    if box.aspect_norm > args.max_aspect_norm:
        return False, "high_aspect"
    if box.edge_margin_frac < args.min_edge_margin_frac:
        return False, "edge_partial"
    if overlap > args.max_overlap_frac:
        return False, "overlap"
    return True, "pass"


def expanded_crop_box(box: LabelBox, pad_frac: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.xyxy
    pad = pad_frac * max(x2 - x1, y2 - y1)
    left = max(0, int(math.floor(x1 - pad)))
    top = max(0, int(math.floor(y1 - pad)))
    right = min(box.image_width, int(math.ceil(x2 + pad)))
    bottom = min(box.image_height, int(math.ceil(y2 + pad)))
    if right <= left or bottom <= top:
        raise ValueError("empty crop box")
    return left, top, right, bottom


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def checkerboard(size: tuple[int, int], cell: int = 14) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            color = (226, 226, 226) if (x // cell + y // cell) % 2 else (248, 248, 248)
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=color)
    return image


def draw_candidate_tile(row: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    tile = Image.new("RGB", size, "white")
    crop_path = resolve(Path(str(row["crop_path"])))
    with Image.open(crop_path).convert("RGB") as image:
        image.thumbnail((size[0] - 8, size[1] - 38), Image.Resampling.LANCZOS)
        tile.paste(image, ((size[0] - image.width) // 2, 4))
    draw = ImageDraw.Draw(tile)
    label = f"{row['class_name']} s={float(row['score']):.2f} {row['source_group']}"[:42]
    draw.rectangle((0, size[1] - 32, size[0], size[1]), fill="white")
    draw.text((4, size[1] - 27), label, fill="black")
    return tile


def draw_asset_tile(row: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    tile = checkerboard(size)
    asset_path = resolve(Path(str(row["asset_path"])))
    with Image.open(asset_path).convert("RGBA") as image:
        image.thumbnail((size[0] - 8, size[1] - 40), Image.Resampling.LANCZOS)
        tile.paste(image, ((size[0] - image.width) // 2, 4), image)
    draw = ImageDraw.Draw(tile)
    label = f"{row['class_name']} fill={float(row['rotated_rect_fill_ratio']):.2f}"[:36]
    draw.rectangle((0, size[1] - 32, size[0], size[1]), fill="white")
    draw.text((4, size[1] - 27), label, fill="black")
    return tile


def draw_reject_tile(row: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    path_text = str(row.get("transparent_path", ""))
    path = resolve(Path(path_text)) if path_text else None
    if path and path.exists():
        tile = checkerboard(size)
        with Image.open(path).convert("RGBA") as image:
            image.thumbnail((size[0] - 8, size[1] - 40), Image.Resampling.LANCZOS)
            tile.paste(image, ((size[0] - image.width) // 2, 4), image)
    else:
        tile = draw_candidate_tile(row, size)
    draw = ImageDraw.Draw(tile)
    label = f"{row.get('class_name', '')} {row.get('reject_reason', '')}"[:38]
    draw.rectangle((0, size[1] - 32, size[0], size[1]), fill="white")
    draw.text((4, size[1] - 27), label, fill="black")
    return tile


def draw_easy_tile(row: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    tile = draw_asset_tile(row, size)
    draw = ImageDraw.Draw(tile)
    label = (
        f"{row.get('class_name', '')} easy={float(row.get('easy_flat_score', 0.0)):.2f} "
        f"skin={float(row.get('skin_like_foreground_ratio', 0.0)):.3f}"
    )[:42]
    draw.rectangle((0, size[1] - 32, size[0], size[1]), fill="white")
    draw.text((4, size[1] - 27), label, fill="black")
    return tile


def make_contact_sheet(
    rows: list[dict[str, Any]],
    out_path: Path,
    title: str,
    tile_func,
    *,
    max_items: int = 90,
) -> None:
    shown = rows[:max_items]
    if not shown:
        return
    tile_w, tile_h = 230, 170
    cols = 5
    rows_n = (len(shown) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows_n * tile_h + 34), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    for index, row in enumerate(shown):
        tile = tile_func(row, (tile_w, tile_h))
        x = (index % cols) * tile_w
        y = 34 + (index // cols) * tile_h
        sheet.paste(tile, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def mine(args: argparse.Namespace) -> None:
    data_yaml = resolve(args.data_yaml)
    out_root = resolve(args.out_root)
    candidates_root = out_root / "candidates"
    crops_root = candidates_root / "ben2_inputs"
    if args.clean_candidates:
        safe_clean(candidates_root, out_root)
    crops_root.mkdir(parents=True, exist_ok=True)
    (out_root / "qa").mkdir(parents=True, exist_ok=True)

    dataset_root, names, payload = read_data_yaml(data_yaml)
    splits = parse_csv_list(args.splits)
    allowed_classes = set(parse_csv_list(args.classes)) if args.classes else set(CLASS_NAMES)
    unknown = sorted(allowed_classes - set(CLASS_NAMES))
    if unknown:
        raise SystemExit(f"unknown classes: {', '.join(unknown)}")

    all_candidates: list[dict[str, Any]] = []
    reject_counts: Counter[str] = Counter()
    for split in splits:
        image_dir = split_dir(dataset_root, payload, split)
        for image_path in iter_images(image_dir):
            if not source_name_allowed(image_path, args):
                continue
            boxes = read_label_boxes(image_path, split, names)
            if not boxes:
                continue
            if len(boxes) < args.min_boxes_per_image:
                reject_counts["image_min_boxes"] += len(boxes)
                continue
            if args.max_boxes_per_image > 0 and len(boxes) > args.max_boxes_per_image:
                reject_counts["image_max_boxes"] += len(boxes)
                continue
            for box in boxes:
                if box.class_name not in allowed_classes:
                    continue
                overlap = max_overlap_fraction(box, boxes)
                passed, reason = box_passes_filters(box, overlap, args)
                if not passed:
                    reject_counts[reason] += 1
                    continue
                score = candidate_score(box, overlap)
                all_candidates.append(
                    {
                        "box": box,
                        "overlap_fraction": overlap,
                        "score": score,
                        "source_group": source_group_for(image_path),
                    }
                )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_candidates:
        grouped[row["box"].class_name].append(row)

    selected: list[dict[str, Any]] = []
    per_group_counts: Counter[tuple[str, str]] = Counter()
    for class_name in CLASS_NAMES:
        rows = sorted(
            grouped.get(class_name, []),
            key=lambda row: (-float(row["score"]), repo_rel(row["box"].image_path), int(row["box"].line_no)),
        )
        kept = 0
        for row in rows:
            group_key = (class_name, str(row["source_group"]))
            if args.max_per_source_group > 0 and per_group_counts[group_key] >= args.max_per_source_group:
                continue
            selected.append(row)
            per_group_counts[group_key] += 1
            kept += 1
            if args.max_per_class > 0 and kept >= args.max_per_class:
                break

    manifest_rows: list[dict[str, Any]] = []
    stem_counts: Counter[str] = Counter()
    for index, row in enumerate(selected):
        box: LabelBox = row["box"]
        crop_box = expanded_crop_box(box, args.crop_pad_frac)
        base_stem = f"{box.class_name}_{source_group_for(box.image_path)}_line{box.line_no:02d}"
        base_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", base_stem).strip("_")
        stem_counts[base_stem] += 1
        stem = base_stem if stem_counts[base_stem] == 1 else f"{base_stem}_{stem_counts[base_stem]:02d}"
        batch_dir = crops_root / f"batch_{index // 100:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        crop_path = batch_dir / f"{stem}.jpg"
        with Image.open(box.image_path).convert("RGB") as image:
            crop = image.crop(crop_box)
            crop.save(crop_path, quality=95)
        x1, y1, x2, y2 = box.xyxy
        manifest_rows.append(
            {
                "candidate_id": stem,
                "class_id": box.class_id,
                "class_name": box.class_name,
                "crop_box_xyxy": " ".join(str(value) for value in crop_box),
                "crop_height": crop_box[3] - crop_box[1],
                "crop_path": repo_rel(crop_path),
                "crop_width": crop_box[2] - crop_box[0],
                "edge_margin_frac": f"{box.edge_margin_frac:.6f}",
                "image_height": box.image_height,
                "image_width": box.image_width,
                "label_line_no": box.line_no,
                "overlap_fraction": f"{float(row['overlap_fraction']):.6f}",
                "score": f"{float(row['score']):.6f}",
                "short_px": f"{box.short_px:.2f}",
                "source_box_area_ratio": f"{box.area_ratio:.6f}",
                "source_box_aspect_norm": f"{box.aspect_norm:.6f}",
                "source_box_xyxy": f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}",
                "source_box_yolo": f"{box.cx:.6f} {box.cy:.6f} {box.width:.6f} {box.height:.6f}",
                "source_group": row["source_group"],
                "source_image_path": repo_rel(box.image_path),
                "source_label_path": repo_rel(box.label_path),
                "split": box.split,
            }
        )

    write_csv(candidates_root / "candidate_manifest.csv", manifest_rows)
    (candidates_root / "ben2_stems.txt").write_text(
        "\n".join(row["candidate_id"] for row in manifest_rows) + ("\n" if manifest_rows else ""),
        encoding="utf-8",
    )
    summary = {
        "data_yaml": repo_rel(data_yaml),
        "out_root": repo_rel(out_root),
        "splits": splits,
        "candidates": len(manifest_rows),
        "class_counts": dict(sorted(Counter(row["class_name"] for row in manifest_rows).items())),
        "reject_counts": dict(sorted(reject_counts.items())),
        "policy": {
            "heldout_default": "mine defaults to train split only",
            "duplicate_control": "source_group removes Roboflow .rf hashes before per-group caps",
            "background_removal": "run scripts/run_ben2_real_flat_bill_candidates.py before rectify",
        },
    }
    (candidates_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    make_contact_sheet(
        sorted(manifest_rows, key=lambda row: (row["class_name"], -float(row["score"]))),
        out_root / "qa" / "candidate_contact.jpg",
        "Real flat-bill BEN2 candidates",
        draw_candidate_tile,
    )
    print(f"wrote candidates: {len(manifest_rows)}")
    print(f"manifest: {candidates_root / 'candidate_manifest.csv'}")
    print(f"ben2 inputs: {crops_root}")
    print(f"class counts: {dict(sorted(Counter(row['class_name'] for row in manifest_rows).items()))}")
    print(f"reject counts: {dict(sorted(reject_counts.items()))}")


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing manifest: {repo_rel(path)}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def connected_component_labels(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    height, width = mask.shape
    labels = np.zeros_like(mask, dtype=np.int32)
    sizes: list[int] = [0]
    current = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current += 1
            sizes.append(0)
            queue: deque[tuple[int, int]] = deque([(y, x)])
            labels[y, x] = current
            while queue:
                cy, cx = queue.popleft()
                sizes[current] += 1
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            queue.append((ny, nx))
    return labels, sizes


def order_quad(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def warp_quad_rgba(image: Image.Image, quad: np.ndarray, max_output_long: int) -> Image.Image:
    src = order_quad(quad)
    top_w = float(np.linalg.norm(src[1] - src[0]))
    bottom_w = float(np.linalg.norm(src[2] - src[3]))
    left_h = float(np.linalg.norm(src[3] - src[0]))
    right_h = float(np.linalg.norm(src[2] - src[1]))
    out_w = max(1, int(round(max(top_w, bottom_w))))
    out_h = max(1, int(round(max(left_h, right_h))))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    arr = np.asarray(image.convert("RGBA"))
    warped = cv2.warpPerspective(
        arr,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    result = Image.fromarray(warped, "RGBA")
    if result.height > result.width:
        result = result.rotate(90, expand=True)
    if max_output_long > 0 and max(result.size) > max_output_long:
        ratio = max_output_long / max(result.size)
        result = result.resize(
            (max(1, int(round(result.width * ratio))), max(1, int(round(result.height * ratio)))),
            Image.Resampling.LANCZOS,
        )
    alpha = result.getchannel("A").point(lambda value: 255 if value > 16 else 0)
    result.putalpha(alpha)
    return result


def remove_alpha_islands(image: Image.Image, threshold: int) -> tuple[Image.Image, int]:
    arr = np.array(image.convert("RGBA"))
    mask = arr[:, :, 3] > threshold
    if not mask.any():
        return Image.fromarray(arr, "RGBA"), 0
    labels, sizes = connected_component_labels(mask)
    largest_label = max(range(1, len(sizes)), key=lambda label: sizes[label], default=0)
    if largest_label == 0:
        return Image.fromarray(arr, "RGBA"), 0
    remove = mask & (labels != largest_label)
    arr[remove, 3] = 0
    return Image.fromarray(arr, "RGBA"), int(remove.sum())


def remove_edge_dark_border(image: Image.Image, alpha_threshold: int, dark_threshold: int) -> tuple[Image.Image, int]:
    arr = np.array(image.convert("RGBA"))
    alpha = arr[:, :, 3]
    mask = alpha > alpha_threshold
    if not mask.any():
        return Image.fromarray(arr, "RGBA"), 0

    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    rgb = arr[:, :, :3]
    dark = mask & (rgb.max(axis=2) <= dark_threshold)
    seeds = np.zeros_like(mask, dtype=bool)
    seeds[y1:y2, x1] |= dark[y1:y2, x1]
    seeds[y1:y2, x2 - 1] |= dark[y1:y2, x2 - 1]
    seeds[y1, x1:x2] |= dark[y1, x1:x2]
    seeds[y2 - 1, x1:x2] |= dark[y2 - 1, x1:x2]
    if not seeds.any():
        return Image.fromarray(arr, "RGBA"), 0

    remove = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque((int(y), int(x)) for y, x in zip(*np.where(seeds), strict=True))
    for y, x in queue:
        remove[y, x] = True
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and dark[ny, nx] and not remove[ny, nx]:
                remove[ny, nx] = True
                queue.append((ny, nx))
    arr[remove, 3] = 0
    return Image.fromarray(arr, "RGBA"), int(remove.sum())


def alpha_shape_metrics_for_rgba(image: Image.Image, threshold: int) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    alpha = np.asarray(image.convert("RGBA").getchannel("A"))
    mask = alpha > threshold
    area = int(mask.sum())
    if area == 0:
        return {"reject_reason": "empty_alpha", "alpha_area": 0}, None, None
    labels, sizes = connected_component_labels(mask)
    largest_label = max(range(1, len(sizes)), key=lambda label: sizes[label], default=0)
    if largest_label == 0:
        return {"reject_reason": "empty_alpha", "alpha_area": 0}, None, None
    largest_mask = labels == largest_label
    largest_area = int(largest_mask.sum())
    largest_ratio = largest_area / max(1, area)
    mask_u8 = (largest_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"reject_reason": "no_contour", "alpha_area": area}, None, None
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    rect = cv2.minAreaRect(contour)
    rect_w, rect_h = float(rect[1][0]), float(rect[1][1])
    rect_area = max(1.0, rect_w * rect_h)
    aspect_raw = max(rect_w, rect_h) / max(1.0, min(rect_w, rect_h))
    rotated_fill = contour_area / rect_area
    ys, xs = np.where(largest_mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    quad = cv2.boxPoints(rect).astype(np.float32)
    metrics = {
        "alpha_area": area,
        "bbox_fill_ratio": largest_area / bbox_area,
        "bbox_xyxy": f"{x1} {y1} {x2} {y2}",
        "component_count": len(sizes) - 1,
        "largest_component_area": largest_area,
        "largest_component_ratio": largest_ratio,
        "rect_angle": float(rect[2]),
        "rect_aspect_norm": aspect_raw,
        "rect_height": min(rect_w, rect_h),
        "rect_width": max(rect_w, rect_h),
        "rotated_rect_fill_ratio": rotated_fill,
        "quad_xy": " ".join(f"{value:.2f}" for point in order_quad(quad) for value in point),
        "reject_reason": "",
    }
    return metrics, largest_mask, quad


def alpha_shape_metrics(path: Path, threshold: int) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    with Image.open(path).convert("RGBA") as image:
        return alpha_shape_metrics_for_rgba(image, threshold)


def reject_reason(metrics: dict[str, Any], args: argparse.Namespace) -> str:
    if metrics.get("reject_reason"):
        return str(metrics["reject_reason"])
    if int(metrics["alpha_area"]) < args.min_alpha_area:
        return "tiny_alpha"
    if float(metrics["largest_component_ratio"]) < args.min_largest_component_ratio:
        return "fragmented_alpha"
    if float(metrics["rotated_rect_fill_ratio"]) < args.min_rotated_rect_fill:
        return "non_rectangular_alpha"
    if float(metrics["rect_aspect_norm"]) < args.min_rect_aspect:
        return "low_rect_aspect"
    if float(metrics["rect_aspect_norm"]) > args.max_rect_aspect:
        return "high_rect_aspect"
    return ""


def relative_or_blank(path: Path | None) -> str:
    return repo_rel(path) if path else ""


def clean_bank_outputs(out_root: Path) -> None:
    for path in [out_root / class_name for class_name in CLASS_NAMES]:
        if path.exists():
            safe_clean(path, out_root)
    for path in [out_root / "masks", out_root / "rejects"]:
        if path.exists():
            safe_clean(path, out_root)
    for path in [out_root / "qa" / "accepted_contact.jpg"]:
        if path.exists():
            path.unlink()
    for path in [out_root / "manifest.csv", out_root / "rejects.csv"]:
        if path.exists():
            path.unlink()


def transparent_path_for(row: dict[str, str], transparent_root: Path) -> Path:
    candidate_id = row["candidate_id"]
    for suffix in [".png", ".webp"]:
        direct = transparent_root / f"{candidate_id}{suffix}"
        if direct.exists():
            return direct
    matches = sorted(transparent_root.rglob(f"{candidate_id}.png")) + sorted(transparent_root.rglob(f"{candidate_id}.webp"))
    if matches:
        return matches[0]
    return transparent_root / f"{candidate_id}.png"


def rectify(args: argparse.Namespace) -> None:
    out_root = resolve(args.out_root)
    candidate_manifest = resolve(args.candidate_manifest) if args.candidate_manifest else out_root / "candidates" / "candidate_manifest.csv"
    transparent_root = resolve(args.transparent_root) if args.transparent_root else out_root / "ben2_output"
    if args.clean_bank:
        clean_bank_outputs(out_root)
    (out_root / "qa").mkdir(parents=True, exist_ok=True)

    candidates = read_manifest(candidate_manifest)
    rows: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for row in candidates:
        transparent_path = transparent_path_for(row, transparent_root)
        if not transparent_path.exists():
            miss = {**row, "reject_reason": "missing_transparent", "transparent_path": repo_rel(transparent_path)}
            rejects.append(miss)
            continue
        try:
            with Image.open(transparent_path).convert("RGBA") as opened:
                transparent_image = opened.copy()
            edge_dark_removed_px = 0
            if not args.no_remove_edge_dark_border:
                transparent_image, edge_dark_removed_px = remove_edge_dark_border(
                    transparent_image,
                    args.alpha_threshold,
                    args.edge_dark_threshold,
                )
            metrics, _largest_mask, quad = alpha_shape_metrics_for_rgba(transparent_image, args.alpha_threshold)
        except OSError as exc:
            rejects.append({**row, "reject_reason": f"unreadable_transparent:{exc}", "transparent_path": repo_rel(transparent_path)})
            continue
        metrics["edge_dark_removed_px"] = edge_dark_removed_px
        metrics["edge_dark_border_policy"] = (
            "disabled" if args.no_remove_edge_dark_border else f"boundary_dark_rgbmax_le_{args.edge_dark_threshold}"
        )
        reason = reject_reason(metrics, args)
        if reason or quad is None:
            rejects.append(
                {
                    **row,
                    **format_metrics(metrics),
                    "reject_reason": reason or str(metrics.get("reject_reason", "shape_reject")),
                    "transparent_path": repo_rel(transparent_path),
                }
            )
            if args.copy_transparent_misses:
                target = out_root / "rejects" / transparent_path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                transparent_image.save(target)
            continue

        rectified = warp_quad_rgba(transparent_image, quad, args.max_output_long)
        post_warp_edge_dark_removed_px = 0
        if not args.no_remove_edge_dark_border:
            rectified, post_warp_edge_dark_removed_px = remove_edge_dark_border(
                rectified,
                args.alpha_threshold,
                args.edge_dark_threshold,
            )
        alpha_island_removed_px = 0
        if not args.no_remove_alpha_islands:
            rectified, alpha_island_removed_px = remove_alpha_islands(rectified, args.alpha_threshold)
        class_name = row["class_name"]
        stem = row["candidate_id"]
        asset_path = out_root / class_name / f"{stem}_rectified.png"
        mask_path = out_root / "masks" / class_name / f"{stem}_rectified_mask.png"
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        rectified.save(asset_path)
        rectified.getchannel("A").point(lambda value: 255 if value > args.alpha_threshold else 0).save(mask_path)
        asset_metrics = alpha_metrics_for_image(rectified, args.alpha_threshold)
        rows.append(
            {
                **row,
                **format_metrics(metrics),
                **asset_metrics,
                "asset_path": repo_rel(asset_path),
                "asset_quality_policy": "real_flat_bill_rectified_probe_v1",
                "post_warp_alpha_island_removed_px": str(alpha_island_removed_px),
                "post_warp_edge_dark_removed_px": str(post_warp_edge_dark_removed_px),
                "license_status": "inherits_cashsnap_v1_review_required",
                "mask_path": repo_rel(mask_path),
                "side": PROBE_SIDE,
                "source_status": PROBE_STATUS,
                "status": PROBE_STATUS,
                "transparent_path": repo_rel(transparent_path),
                "usage_note": "internal_probe_only_from_cashsnap_train_split",
                "visual_qa_status": "auto_shape_pass_needs_spot_review",
            }
        )

    write_csv(out_root / "manifest.csv", rows)
    write_csv(out_root / "rejects.csv", rejects)
    summary = {
        "candidate_manifest": repo_rel(candidate_manifest),
        "transparent_root": repo_rel(transparent_root),
        "accepted": len(rows),
        "rejected": len(rejects),
        "class_counts": dict(sorted(Counter(row["class_name"] for row in rows).items())),
        "reject_counts": dict(sorted(Counter(row.get("reject_reason", "") for row in rejects).items())),
        "status": PROBE_STATUS,
        "side": PROBE_SIDE,
        "policy": {
            "heldout_use": "expected train-split candidate_manifest only unless explicitly overridden",
            "renderer_opt_in": "--status real_train_flat_probe --sides unknown --asset-quality-policy all_manifest",
            "qa": "run scripts/audit_cutout_bank.py and inspect qa/accepted_contact.jpg before training use",
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    make_contact_sheet(
        sorted(rows, key=lambda row: (row["class_name"], row["candidate_id"])),
        out_root / "qa" / "accepted_contact.jpg",
        "Rectified real flat-bill cutouts",
        draw_asset_tile,
    )
    make_contact_sheet(
        sorted(rejects, key=lambda row: (row.get("reject_reason", ""), row.get("class_name", ""), row.get("candidate_id", ""))),
        out_root / "qa" / "reject_contact.jpg",
        "Rejected real flat-bill transparent candidates",
        draw_reject_tile,
    )
    print(f"accepted: {len(rows)}")
    print(f"rejected: {len(rejects)}")
    print(f"manifest: {out_root / 'manifest.csv'}")
    print(f"class counts: {dict(sorted(Counter(row['class_name'] for row in rows).items()))}")
    print(f"reject counts: {dict(sorted(Counter(row.get('reject_reason', '') for row in rejects).items()))}")


def format_metrics(metrics: dict[str, Any]) -> dict[str, str]:
    formatted: dict[str, str] = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.6f}"
        else:
            formatted[key] = str(value)
    return formatted


def alpha_metrics_for_image(image: Image.Image, threshold: int) -> dict[str, str]:
    alpha = np.asarray(image.getchannel("A"))
    mask = alpha > threshold
    area = int(mask.sum())
    if area == 0:
        return {
            "alpha_area_rectified": "0",
            "bbox_fill_ratio_rectified": "0.000000",
            "bbox_xyxy_rectified": "",
            "height": str(image.height),
            "width": str(image.width),
        }
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    return {
        "alpha_area_rectified": str(area),
        "bbox_fill_ratio_rectified": f"{area / bbox_area:.6f}",
        "bbox_xyxy_rectified": f"{x1} {y1} {x2} {y2}",
        "height": str(image.height),
        "width": str(image.width),
    }


def skin_like_ratio(image: Image.Image) -> float:
    arr = np.asarray(image.convert("RGBA"))
    alpha_mask = arr[:, :, 3] > 16
    if not alpha_mask.any():
        return 0.0
    rgb = arr[:, :, :3].astype(np.int16)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    skin = (
        (red > 95)
        & (green > 40)
        & (blue > 20)
        & ((max_channel - min_channel) > 15)
        & (np.abs(red - green) > 15)
        & (red > green)
        & (red > blue)
    )
    return float((skin & alpha_mask).sum() / max(1, alpha_mask.sum()))


def safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def safe_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "") or default))
    except ValueError:
        return default


def easy_reject_reasons(row: dict[str, str], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if row.get("audit_flags", "").strip():
        reasons.append("audit_flags")
    if safe_float(row, "source_box_aspect_norm") < args.min_source_aspect_norm:
        reasons.append("source_box_shape")
    rect_aspect = safe_float(row, "rect_aspect_norm")
    if rect_aspect < args.min_rect_aspect_norm:
        reasons.append("low_rect_aspect")
    if rect_aspect > args.max_rect_aspect_norm:
        reasons.append("high_rect_aspect")
    if safe_float(row, "rotated_rect_fill_ratio") < args.min_rotated_rect_fill:
        reasons.append("rotated_fill")
    if safe_float(row, "bbox_fill_ratio_rectified") < args.min_bbox_fill_rectified:
        reasons.append("bbox_fill")
    if safe_float(row, "largest_component_ratio") < args.min_largest_component_ratio:
        reasons.append("fragmented")
    if safe_int(row, "component_count") > args.max_component_count:
        reasons.append("component_count")
    if safe_int(row, "edge_dark_removed_px") > args.max_edge_dark_removed_px:
        reasons.append("edge_dark_cleanup")
    if safe_float(row, "overlap_fraction") > args.max_overlap_fraction:
        reasons.append("source_overlap")
    if safe_float(row, "skin_like_foreground_ratio") > args.max_skin_like_ratio:
        reasons.append("skin_like_ratio")
    return reasons


def easy_score(row: dict[str, str]) -> float:
    aspect = safe_float(row, "rect_aspect_norm")
    aspect_target = 2.15
    aspect_penalty = min(1.0, abs(math.log(max(1e-6, aspect / aspect_target))) / math.log(2.0))
    return (
        safe_float(row, "score")
        + safe_float(row, "rotated_rect_fill_ratio")
        + safe_float(row, "bbox_fill_ratio_rectified")
        + safe_float(row, "largest_component_ratio")
        - 0.35 * aspect_penalty
        - min(0.5, safe_int(row, "edge_dark_removed_px") / 5000.0)
        - min(0.5, safe_float(row, "skin_like_foreground_ratio") * 1.5)
    )


def select_easy(args: argparse.Namespace) -> None:
    out_root = resolve(args.out_root)
    manifest_path = resolve(args.manifest) if args.manifest else out_root / "manifest.csv"
    audit_path = resolve(args.audit_assets) if args.audit_assets else out_root / "audit" / "all_assets.csv"
    out_manifest = resolve(args.out_manifest) if args.out_manifest else out_root / "manifest_easy_flat.csv"
    rows = read_manifest(manifest_path)
    manual_keep_ids = read_id_file(args.manual_keep_file)
    audit_by_asset: dict[str, dict[str, str]] = {}
    if audit_path.exists():
        for audit_row in read_manifest(audit_path):
            audit_by_asset[str(audit_row.get("asset_path", "")).replace("\\", "/")] = audit_row

    enriched: list[dict[str, Any]] = []
    for row in rows:
        asset_path = resolve(Path(str(row.get("asset_path", ""))))
        skin_ratio = 0.0
        if asset_path.exists():
            with Image.open(asset_path).convert("RGBA") as image:
                skin_ratio = skin_like_ratio(image)
        audit = audit_by_asset.get(str(row.get("asset_path", "")).replace("\\", "/"), {})
        merged: dict[str, Any] = {
            **row,
            "audit_flags": audit.get("flags", ""),
            "audit_visual_qa_status": audit.get("visual_qa_status", ""),
            "skin_like_foreground_ratio": f"{skin_ratio:.6f}",
        }
        reasons = easy_reject_reasons(merged, args)
        merged["easy_flat_reject_reason"] = ";".join(reasons)
        merged["easy_flat_score"] = f"{easy_score(merged):.6f}"
        merged["easy_flat_policy"] = "auto_easy_flat_v1_shape_audit_small_subset"
        enriched.append(merged)

    candidates = [row for row in enriched if not row["easy_flat_reject_reason"]]
    selected: list[dict[str, Any]] = []
    for class_name in CLASS_NAMES:
        class_rows = sorted(
            [row for row in candidates if row.get("class_name") == class_name],
            key=lambda row: (-float(row["easy_flat_score"]), row.get("candidate_id", "")),
        )
        selected.extend(class_rows[: max(0, args.max_per_class)])
    if manual_keep_ids is not None:
        auto_selected_ids = {str(row["candidate_id"]) for row in selected}
        unknown_ids = sorted(manual_keep_ids - {str(row.get("candidate_id", "")) for row in enriched})
        if unknown_ids:
            raise SystemExit(f"manual keep ids not found in manifest: {', '.join(unknown_ids)}")
        auto_rejected_ids = sorted(manual_keep_ids - auto_selected_ids)
        if auto_rejected_ids:
            raise SystemExit(
                "manual keep ids did not pass auto easy-flat gates: " + ", ".join(auto_rejected_ids)
            )
        selected = [row for row in selected if str(row["candidate_id"]) in manual_keep_ids]
        for row in selected:
            row["easy_flat_manual_review_status"] = "manual_pass_easy_flat_v1"
            row["easy_flat_manual_keep_file"] = repo_rel(resolve(args.manual_keep_file))
    selected_ids = {row["candidate_id"] for row in selected}
    for row in enriched:
        row["easy_flat_status"] = "selected" if row["candidate_id"] in selected_ids else "rejected"
        if row["easy_flat_status"] == "rejected" and not row["easy_flat_reject_reason"]:
            row["easy_flat_reject_reason"] = "per_class_cap"
    write_csv(out_manifest, selected)
    write_csv(out_root / "qa" / "easy_flat_all_decisions.csv", enriched)
    make_contact_sheet(
        sorted(selected, key=lambda row: (row.get("class_name", ""), -float(row["easy_flat_score"]))),
        out_root / "qa" / "easy_flat_contact.jpg",
        "Easy flat real-bill cutout subset",
        draw_easy_tile,
        max_items=120,
    )
    summary = {
        "input_manifest": repo_rel(manifest_path),
        "audit_assets": relative_or_blank(audit_path if audit_path.exists() else None),
        "out_manifest": repo_rel(out_manifest),
        "selected": len(selected),
        "input_rows": len(rows),
        "max_per_class": args.max_per_class,
        "manual_keep_file": repo_rel(resolve(args.manual_keep_file)) if args.manual_keep_file else "",
        "class_counts": dict(sorted(Counter(row["class_name"] for row in selected).items())),
        "reject_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in enriched
                    if row["easy_flat_status"] == "rejected"
                    for reason in str(row["easy_flat_reject_reason"]).split(";")
                    if reason
                ).items()
            )
        ),
        "policy": {
            "purpose": "Small easy-flat subset for visual QA and renderer smoke; not full asset bank promotion.",
            "skin_like_ratio": "recorded by default, not a hard filter unless --max-skin-like-ratio is lowered",
            "source": "Derived from train-split real-flat probe manifest only.",
        },
    }
    (out_root / "qa" / "easy_flat_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"selected: {len(selected)} / {len(rows)}")
    print(f"manifest: {out_manifest}")
    print(f"class counts: {summary['class_counts']}")
    print(f"reject counts: {summary['reject_counts']}")


def main() -> int:
    args = parse_args()
    if args.command == "mine":
        mine(args)
    elif args.command == "rectify":
        rectify(args)
    elif args.command == "select-easy":
        select_easy(args)
    else:  # pragma: no cover
        raise SystemExit(f"unknown command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
