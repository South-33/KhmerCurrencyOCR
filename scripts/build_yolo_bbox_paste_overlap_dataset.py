#!/usr/bin/env python
"""Build a real-bbox paste overlap/partial YOLO diagnostic dataset.

This intentionally uses crude rectangular crops from real YOLO boxes. The goal
is not photorealistic training data; it is a larger eval/stress bridge for
partial, off-frame, fanned, and overlapping visible-evidence behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from cashsnap_classes import CLASS_NAMES


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


@dataclass(frozen=True)
class Label:
    class_id: int
    xc: float
    yc: float
    width: float
    height: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class CropRef:
    image: str
    split: str
    source_group: str
    width: int
    height: int
    label: Label


@dataclass(frozen=True)
class PlacedRef:
    instance_id: int
    crop: CropRef
    placed_pixels: int
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=120)
    parser.add_argument("--val-count", type=int, default=80)
    parser.add_argument("--test-count", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--min-notes", type=int, default=3)
    parser.add_argument("--max-notes", type=int, default=6)
    parser.add_argument("--min-visible-area-frac", type=float, default=0.006)
    parser.add_argument("--min-visibility-ratio", type=float, default=0.18)
    parser.add_argument("--min-note-short-frac", type=float, default=0.18)
    parser.add_argument("--max-note-short-frac", type=float, default=0.36)
    parser.add_argument("--off-frame-prob", type=float, default=0.35)
    parser.add_argument("--background-mode", choices=["neutral", "blurred_source"], default="neutral")
    parser.add_argument("--balance-classes", action="store_true", help="Sample source crops with per-scene class balancing.")
    parser.add_argument(
        "--cutout-feather-frac",
        type=float,
        default=0.0,
        help=(
            "Feather pasted crop alpha near crop edges as a fraction of the crop short side. "
            "Default 0 keeps the original hard rectangular crop mask."
        ),
    )
    parser.add_argument("--preview-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def read_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(resolved)} must be a YAML mapping")
    return payload


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw = Path(str(config.get("path", "."))).expanduser()
    return raw if raw.is_absolute() else (config_path.parent / raw).resolve()


def split_path(dataset_root: Path, raw_value: str) -> Path:
    value = Path(raw_value)
    return value if value.is_absolute() else dataset_root / value


def read_image_list(path: Path) -> list[str]:
    rows: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            rows.append(repo_rel(resolve(line)))
    return rows


def image_rows(path: Path) -> list[str]:
    image_dir = resolve(path)
    if not image_dir.exists():
        raise SystemExit(f"missing image directory: {repo_rel(image_dir)}")
    return [
        repo_rel(item)
        for item in sorted(image_dir.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS
    ]


def split_rows(config_path: Path, config: dict[str, Any], split: str) -> list[str]:
    root = data_root(config_path, config)
    raw_split = config.get(split)
    if raw_split is None and split == "val":
        raw_split = config.get("valid")
    if raw_split is None:
        return []
    values = raw_split if isinstance(raw_split, list) else [raw_split]
    rows: list[str] = []
    for raw_value in values:
        path = split_path(root, str(raw_value))
        if path.suffix.lower() == ".txt":
            rows.extend(read_image_list(path))
        elif path.is_dir():
            rows.extend(image_rows(path))
        else:
            raise SystemExit(f"{repo_rel(config_path)} {split} points to neither list nor directory: {raw_value}")
    return rows


def label_path_for_image(image: str) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def source_group(image: str) -> str:
    name = Path(image).name
    for prefix, group in SOURCE_PREFIXES.items():
        if name.startswith(prefix):
            return group
    return "unknown"


def read_crop_refs(rows: list[str], split: str) -> list[CropRef]:
    refs: list[CropRef] = []
    for image in rows:
        image_path = resolve(image)
        label_path = resolve(label_path_for_image(image))
        if not label_path.exists():
            continue
        try:
            with Image.open(image_path) as raw_image:
                width, height = raw_image.size
        except OSError:
            continue
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                continue
            xc, yc, box_w, box_h = (float(value) for value in parts[1:5])
            if box_w <= 0 or box_h <= 0:
                continue
            x1 = max(0.0, (xc - box_w / 2) * width)
            y1 = max(0.0, (yc - box_h / 2) * height)
            x2 = min(float(width), (xc + box_w / 2) * width)
            y2 = min(float(height), (yc + box_h / 2) * height)
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue
            refs.append(
                CropRef(
                    image=image,
                    split=split,
                    source_group=source_group(image),
                    width=width,
                    height=height,
                    label=Label(class_id=class_id, xc=xc, yc=yc, width=box_w, height=box_h, xyxy=(x1, y1, x2, y2)),
                )
            )
    return refs


def normalize_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list):
        return [str(item) for item in raw_names]
    if isinstance(raw_names, dict):
        return [str(value) for _, value in sorted((int(key), value) for key, value in raw_names.items())]
    raise SystemExit("source data YAML must include names")


def safe_clean_out_root(out_root: Path) -> None:
    resolved = out_root.resolve()
    allowed_roots = [(ROOT / "data" / "processed").resolve(), (ROOT / "data" / "synthetic").resolve()]
    if not any(str(resolved).startswith(str(root) + os.sep) for root in allowed_roots):
        raise SystemExit(f"refusing to clean outside generated data roots: {repo_rel(resolved)}")
    if resolved.exists():
        shutil.rmtree(resolved)


def make_background(size: int, refs: list[CropRef], rng: random.Random, mode: str) -> Image.Image:
    if mode == "blurred_source" and refs:
        ref = rng.choice(refs)
        with Image.open(resolve(ref.image)) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            image = ImageOps.fit(image, (size, size), method=Image.Resampling.BICUBIC)
            image = image.filter(ImageFilter.GaussianBlur(radius=size * 0.035))
            overlay = Image.new("RGB", (size, size), (rng.randint(120, 190), rng.randint(120, 190), rng.randint(120, 190)))
            return Image.blend(image, overlay, 0.58).convert("RGBA")
    base = np.zeros((size, size, 3), dtype=np.uint8)
    color = np.array([rng.randint(105, 205), rng.randint(100, 200), rng.randint(95, 195)], dtype=np.int16)
    noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, 8, base.shape).astype(np.int16)
    arr = np.clip(color + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(radius=0.4)).convert("RGBA")


def soft_edge_mask(width: int, height: int, feather_px: int) -> Image.Image:
    if feather_px <= 0:
        return Image.new("L", (width, height), 255)
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.minimum.reduce([xx, yy, width - 1 - xx, height - 1 - yy]).astype(np.float32)
    alpha = np.clip(dist / float(max(1, feather_px)), 0.0, 1.0)
    return Image.fromarray((alpha * 255).astype(np.uint8), "L")


def crop_patch(ref: CropRef, target_short: float, angle: float, feather_frac: float) -> tuple[Image.Image, Image.Image]:
    x1, y1, x2, y2 = ref.label.xyxy
    with Image.open(resolve(ref.image)) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        patch = image.crop((int(math.floor(x1)), int(math.floor(y1)), int(math.ceil(x2)), int(math.ceil(y2))))
    if patch.width < 2 or patch.height < 2:
        raise ValueError("degenerate crop")
    scale = target_short / max(1, min(patch.width, patch.height))
    new_w = max(2, int(round(patch.width * scale)))
    new_h = max(2, int(round(patch.height * scale)))
    patch = patch.resize((new_w, new_h), Image.Resampling.BICUBIC).convert("RGBA")
    feather_px = int(round(min(new_w, new_h) * max(0.0, feather_frac)))
    mask = soft_edge_mask(new_w, new_h, feather_px)
    if abs(angle) > 0.01:
        patch = patch.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0))
        mask = mask.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=0)
    patch.putalpha(mask)
    return patch, mask


def paste_instance(
    canvas: Image.Image,
    id_mask: np.ndarray,
    patch: Image.Image,
    mask: Image.Image,
    x: int,
    y: int,
    instance_id: int,
) -> int:
    size = canvas.width
    dst_x1 = max(0, x)
    dst_y1 = max(0, y)
    dst_x2 = min(size, x + patch.width)
    dst_y2 = min(size, y + patch.height)
    if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
        return 0
    src_x1 = dst_x1 - x
    src_y1 = dst_y1 - y
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)
    patch_crop = patch.crop((src_x1, src_y1, src_x2, src_y2))
    mask_crop = mask.crop((src_x1, src_y1, src_x2, src_y2))
    canvas.alpha_composite(patch_crop, (dst_x1, dst_y1))
    mask_arr = np.asarray(mask_crop) > 16
    region = id_mask[dst_y1:dst_y2, dst_x1:dst_x2]
    region[mask_arr] = instance_id
    id_mask[dst_y1:dst_y2, dst_x1:dst_x2] = region
    return int(mask_arr.sum())


def placement(mode: str, index: int, count: int, patch: Image.Image, size: int, rng: random.Random, off_frame_prob: float) -> tuple[int, int, float]:
    center = size / 2
    rank = index - (count - 1) / 2
    if mode == "fan":
        x = center - patch.width / 2 + rank * rng.uniform(size * 0.035, size * 0.075) + rng.uniform(-18, 18)
        y = center - patch.height / 2 + abs(rank) * rng.uniform(size * 0.015, size * 0.04) + rng.uniform(-20, 24)
        angle = rank * rng.uniform(7, 15) + rng.uniform(-5, 5)
    elif mode == "stack":
        x = center - patch.width / 2 + rank * rng.uniform(size * 0.025, size * 0.055) + rng.uniform(-22, 22)
        y = center - patch.height / 2 + rank * rng.uniform(size * 0.015, size * 0.045) + rng.uniform(-22, 22)
        angle = rng.uniform(-18, 18)
    elif mode == "strip":
        x = center - patch.width / 2 + rank * rng.uniform(size * 0.055, size * 0.095) + rng.uniform(-18, 18)
        y = center - patch.height / 2 + rng.uniform(-size * 0.10, size * 0.10)
        angle = rng.uniform(-38, 38)
    else:
        x = rng.uniform(size * 0.10, size * 0.90 - patch.width)
        y = rng.uniform(size * 0.10, size * 0.90 - patch.height)
        angle = rng.uniform(-28, 28)
    if rng.random() < off_frame_prob:
        side = rng.choice(["left", "right", "top", "bottom"])
        if side == "left":
            x = rng.uniform(-patch.width * 0.55, size * 0.16)
        elif side == "right":
            x = rng.uniform(size * 0.84 - patch.width, size - patch.width * 0.45)
        elif side == "top":
            y = rng.uniform(-patch.height * 0.55, size * 0.16)
        else:
            y = rng.uniform(size * 0.84 - patch.height, size - patch.height * 0.45)
    return int(round(x)), int(round(y)), angle


def visible_bbox(mask: np.ndarray, instance_id: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask == instance_id)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def group_refs_by_class(refs: list[CropRef]) -> dict[int, list[CropRef]]:
    grouped: dict[int, list[CropRef]] = {}
    for ref in refs:
        grouped.setdefault(ref.label.class_id, []).append(ref)
    return grouped


def choose_ref(
    refs: list[CropRef],
    refs_by_class: dict[int, list[CropRef]],
    class_counts: Counter[int],
    args: argparse.Namespace,
    rng: random.Random,
) -> CropRef:
    if args.balance_classes and refs_by_class:
        min_seen = min(class_counts.get(class_id, 0) for class_id in refs_by_class)
        class_pool = [class_id for class_id in refs_by_class if class_counts.get(class_id, 0) == min_seen]
        return rng.choice(refs_by_class[rng.choice(class_pool)])
    return rng.choice(refs)


def make_scene(
    refs: list[CropRef],
    refs_by_class: dict[int, list[CropRef]],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[Image.Image, list[str], dict[str, Any]]:
    size = args.image_size
    mode = rng.choice(["fan", "stack", "stack", "strip", "scatter"])
    canvas = make_background(size, refs, rng, args.background_mode)
    id_mask = np.zeros((size, size), dtype=np.int32)
    note_count = rng.randint(args.min_notes, args.max_notes)
    placed: list[PlacedRef] = []
    class_counts: Counter[int] = Counter()
    for index in range(note_count):
        ref = choose_ref(refs, refs_by_class, class_counts, args, rng)
        if len(class_counts) >= min(note_count, 4):
            # Avoid every scene becoming a same-class echo unless randomness already chose it.
            low_classes = [candidate for candidate in refs if class_counts[candidate.label.class_id] == 0]
            if low_classes and not args.balance_classes:
                ref = rng.choice(low_classes)
        class_counts[ref.label.class_id] += 1
        target_short = size * rng.uniform(args.min_note_short_frac, args.max_note_short_frac)
        pre_angle = 0.0 if mode in {"fan", "stack"} else rng.uniform(-18, 18)
        patch, mask = crop_patch(ref, target_short, pre_angle, args.cutout_feather_frac)
        x, y, angle = placement(mode, index, note_count, patch, size, rng, args.off_frame_prob)
        if abs(angle) > 0.01 and mode in {"fan", "stack"}:
            patch = patch.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0))
            mask = mask.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=0)
        instance_id = index + 1
        placed_pixels = paste_instance(canvas, id_mask, patch, mask, x, y, instance_id)
        if placed_pixels:
            placed.append(PlacedRef(instance_id=instance_id, crop=ref, placed_pixels=placed_pixels, mode=mode))

    labels: list[str] = []
    records: list[dict[str, Any]] = []
    for item in placed:
        box = visible_bbox(id_mask, item.instance_id)
        if box is None:
            visible_pixels = 0
        else:
            visible_pixels = int((id_mask == item.instance_id).sum())
        visibility_ratio = visible_pixels / max(1, item.placed_pixels)
        visible_area_frac = visible_pixels / float(size * size)
        keep = (
            box is not None
            and visible_area_frac >= args.min_visible_area_frac
            and visibility_ratio >= args.min_visibility_ratio
        )
        record: dict[str, Any] = {
            "class_id": item.crop.label.class_id,
            "class_name": CLASS_NAMES[item.crop.label.class_id],
            "source_image": item.crop.image,
            "source_group": item.crop.source_group,
            "mode": item.mode,
            "placed_pixels": item.placed_pixels,
            "visible_pixels": visible_pixels,
            "visible_area_frac": visible_area_frac,
            "visibility_ratio": visibility_ratio,
            "kept_label": keep,
        }
        if box:
            x1, y1, x2, y2 = box
            xc = ((x1 + x2) / 2) / size
            yc = ((y1 + y2) / 2) / size
            bw = (x2 - x1) / size
            bh = (y2 - y1) / size
            record["xyxy"] = [x1, y1, x2, y2]
            if keep:
                labels.append(f"{item.crop.label.class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        records.append(record)
    metadata = {
        "mode": mode,
        "note_count": note_count,
        "kept_labels": len(labels),
        "records": records,
    }
    return canvas.convert("RGB"), labels, metadata


def write_data_yaml(out_root: Path, names: list[str]) -> None:
    name_lines = "\n".join(f"  {index}: {name}" for index, name in enumerate(names))
    (out_root / "data.yaml").write_text(
        f"path: {out_root.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{name_lines}\n",
        encoding="utf-8",
    )


def draw_preview(rows: list[dict[str, Any]], out_path: Path, max_items: int, image_size: int) -> None:
    shown = rows[:max_items]
    if not shown:
        return
    thumb_w = 260
    thumb_h = int(thumb_w * 0.82)
    caption_h = 54
    cols = 4
    rows_count = math.ceil(len(shown) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows_count * (thumb_h + caption_h)), (238, 238, 238))
    font = ImageFont.load_default()
    for index, row in enumerate(shown):
        x = (index % cols) * thumb_w
        y = (index // cols) * (thumb_h + caption_h)
        with Image.open(resolve(row["image"])).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            label_path = resolve(row["label"])
            for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                parts = raw_line.split()
                if len(parts) < 5:
                    continue
                class_id = int(float(parts[0]))
                xc, yc, bw, bh = (float(value) for value in parts[1:5])
                x1 = (xc - bw / 2) * image_size
                y1 = (yc - bh / 2) * image_size
                x2 = (xc + bw / 2) * image_size
                y2 = (yc + bh / 2) * image_size
                draw.rectangle((x1, y1, x2, y2), outline=(230, 40, 40), width=4)
                draw.text((x1 + 4, max(0, y1 - 14)), CLASS_NAMES[class_id], fill=(230, 40, 40), font=font)
            thumb = ImageOps.contain(image, (thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x + (thumb_w - thumb.width) // 2, y))
        caption = ImageDraw.Draw(sheet)
        caption.text((x + 5, y + thumb_h + 4), f"{row['split']} labels={row['label_count']} {row['mode']}"[:44], fill=(0, 0, 0), font=font)
        caption.text((x + 5, y + thumb_h + 22), Path(row["image"]).name[:44], fill=(0, 0, 0), font=font)
    sheet.save(out_path, quality=92)


def main() -> int:
    args = parse_args()
    if args.min_notes <= 0 or args.max_notes < args.min_notes:
        raise SystemExit("invalid note count range")
    if args.image_size < 256:
        raise SystemExit("--image-size must be at least 256")
    if args.cutout_feather_frac < 0:
        raise SystemExit("--cutout-feather-frac must be non-negative")
    out_root = resolve(args.out_root)
    if args.clean:
        safe_clean_out_root(out_root)
    elif out_root.exists():
        raise SystemExit(f"output already exists, pass --clean to replace: {repo_rel(out_root)}")
    config_path = resolve(args.source_data)
    config = read_yaml(config_path)
    names = normalize_names(config.get("names"))
    if names != CLASS_NAMES:
        raise SystemExit("source class names do not match CashSnap 13-class schema")

    split_to_refs: dict[str, list[CropRef]] = {}
    split_to_class_refs: dict[str, dict[int, list[CropRef]]] = {}
    train_refs = read_crop_refs(split_rows(config_path, config, "train"), "train")
    if not train_refs:
        raise SystemExit("no labeled train crop refs found")
    for split in ("train", "val", "test"):
        rows = split_rows(config_path, config, split)
        refs = read_crop_refs(rows, split) if rows else []
        split_to_refs[split] = refs or train_refs
        split_to_class_refs[split] = group_refs_by_class(split_to_refs[split])
        if refs:
            print(f"{split}_crop_refs={len(refs)}")
        else:
            print(f"{split}_crop_refs=0 fallback=train:{len(train_refs)}")

    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    metadata_dir = out_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_data_yaml(out_root, names)

    split_counts = {"train": args.train_count, "val": args.val_count, "test": args.test_count}
    summary_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    kept_label_counts: Counter[str] = Counter()
    rng = random.Random(args.seed)
    for split, count in split_counts.items():
        refs = split_to_refs[split]
        metadata_path = metadata_dir / f"{split}.jsonl"
        with metadata_path.open("w", encoding="utf-8") as metadata_handle:
            made = 0
            attempts = 0
            while made < count:
                attempts += 1
                if attempts > max(count * 20, 100):
                    raise SystemExit(f"could not generate enough non-empty {split} scenes")
                image, labels, metadata = make_scene(refs, split_to_class_refs[split], args, rng)
                if not labels:
                    continue
                stem = f"bboxpaste_{split}_{made:05d}"
                image_path = out_root / "images" / split / f"{stem}.jpg"
                label_path = out_root / "labels" / split / f"{stem}.txt"
                image.save(image_path, quality=args.jpeg_quality)
                label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
                for line in labels:
                    class_name = CLASS_NAMES[int(line.split()[0])]
                    class_counts[class_name] += 1
                    kept_label_counts[f"{split}:{class_name}"] += 1
                for record in metadata["records"]:
                    if record["kept_label"]:
                        source_counts[record["source_group"]] += 1
                row = {
                    "split": split,
                    "image": repo_rel(image_path),
                    "label": repo_rel(label_path),
                    "label_count": len(labels),
                    "mode": metadata["mode"],
                    "note_count": metadata["note_count"],
                }
                summary_rows.append(row)
                metadata_handle.write(json.dumps({**row, **metadata}) + "\n")
                made += 1

    draw_preview(summary_rows, out_root / "preview.jpg", args.preview_count, args.image_size)
    summary = {
        "schema": "cashsnap_bbox_paste_overlap_dataset_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_data": repo_rel(config_path),
        "out_root": repo_rel(out_root),
        "seed": args.seed,
        "image_size": args.image_size,
        "balance_classes": bool(args.balance_classes),
        "cutout_feather_frac": args.cutout_feather_frac,
        "split_counts": split_counts,
        "class_counts": dict(sorted(class_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "split_class_counts": dict(sorted(kept_label_counts.items())),
        "rows": summary_rows,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"dataset={repo_rel(out_root)} images={len(summary_rows)} labels={sum(class_counts.values())}")
    print(f"class_counts={dict(sorted(class_counts.items()))}")
    print(f"source_counts={dict(sorted(source_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
