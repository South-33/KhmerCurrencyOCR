#!/usr/bin/env python
"""Build real-derived partial-visibility YOLO train/eval data.

This generator uses existing labeled CashSnap photos and rewrites labels to the
visible instance AABB after a deterministic crop or occlusion. It intentionally
starts from single-label rows by default so no other visible bill becomes an
unlabeled false negative.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_SOURCE = Path("data/cashsnap_v1/data.yaml")
DEFAULT_OUT = Path("data/processed/cashsnap_real_partial_visibility_p8_v1")
DEFAULT_VISIBILITY = (0.20, 0.30, 0.50)
DEFAULT_MODES = ("left", "right", "top", "bottom", "diag_tl", "diag_br")
DEFAULT_BLOCKER_STYLES = ("flat_dark", "flat_light", "background_sample", "noise_paper", "printed_paper")
SOURCE_PREFIXES = {
    "asian_currency_": "asian_currency",
    "billsbank_": "billsbank",
    "cambodia_currency_project_": "cambodia_currency_project",
    "cashcountingxl_": "cashcountingxl",
    "khmer_scan_": "khmer",
    "khmer_us_currency_": "khmer_us_currency",
    "usd_total_": "usd_total",
}
BORDER_MODES = {"left", "right", "top", "bottom"}
DIAGONAL_MODES = {"diag_tl", "diag_br"}
BOX_OCCLUSION_MODES = {"box_left", "box_right", "box_top", "box_bottom"}
CORNER_CROP_MODES = {"corner_tl", "corner_tr", "corner_bl", "corner_br"}
COUNTABLE_CROP_MODES = {
    "center_x",
    "center_y",
    *CORNER_CROP_MODES,
}


@dataclass(frozen=True)
class Label:
    class_id: int
    xc: float
    yc: float
    width: float
    height: float


@dataclass(frozen=True)
class SourceRow:
    split: str
    image: Path
    labels: tuple[Label, ...]


@dataclass(frozen=True)
class VariantSpec:
    mode: str
    visible_fraction: float
    blocker_style: str = "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-sources-per-class", type=int, default=8)
    parser.add_argument("--eval-sources-per-class", type=int, default=3)
    parser.add_argument("--variants-per-source", type=int, default=2)
    parser.add_argument("--visibility", type=float, nargs="+", default=list(DEFAULT_VISIBILITY))
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--min-corner-visibility", type=float, default=0.0)
    parser.add_argument("--include-source-group", action="append", default=[])
    parser.add_argument("--exclude-source-group", action="append", default=[])
    parser.add_argument(
        "--blocker-styles",
        nargs="+",
        default=list(DEFAULT_BLOCKER_STYLES),
        help="Blocker styles sampled for box_* occlusion modes.",
    )
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--min-output-side", type=int, default=96)
    parser.add_argument("--min-visible-pixels", type=int, default=220)
    parser.add_argument("--single-label-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-count", type=int, default=48)
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
    return Path(target).resolve().relative_to(Path(from_dir).resolve()).as_posix()


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (ROOT / "data").resolve()
    if not (resolved == allowed_root or allowed_root in resolved.parents):
        raise SystemExit(f"Refusing to clean outside data/: {repo_rel(resolved)}")
    if resolved.exists():
        shutil.rmtree(resolved)


def stable_int(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(path)} must be a YAML mapping")
    return payload


def names_by_id(payload: dict[str, Any]) -> dict[int, str]:
    raw = payload.get("names")
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    raise SystemExit("source data must define names as a list or mapping")


def dataset_root(data_yaml: Path, payload: dict[str, Any]) -> Path:
    root = Path(str(payload.get("path", "."))).expanduser()
    return root if root.is_absolute() else (data_yaml.parent / root).resolve()


def split_path(data_yaml: Path, payload: dict[str, Any], split: str) -> Path:
    value = payload.get(split)
    if value is None and split == "val":
        value = payload.get("valid")
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{repo_rel(data_yaml)} missing {split} split")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (dataset_root(data_yaml, payload) / path).resolve()


def listed_images(list_path: Path, root: Path) -> list[Path]:
    images: list[Path] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        path = Path(value).expanduser()
        images.append(path if path.is_absolute() else (root / path).resolve())
    return images


def split_images(data_yaml: Path, payload: dict[str, Any], split: str) -> list[Path]:
    root = dataset_root(data_yaml, payload)
    source = split_path(data_yaml, payload, split)
    if source.is_file():
        return listed_images(source, root)
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
    raise SystemExit(f"{split} source not found: {repo_rel(source)}")


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise SystemExit(f"cannot infer label path for image outside images/: {repo_rel(image_path)}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def source_group_for_image(image_path: Path) -> str:
    name = image_path.name
    for prefix, group in SOURCE_PREFIXES.items():
        if name.startswith(prefix):
            return group
    return "unknown"


def read_labels(image_path: Path) -> tuple[Label, ...]:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return ()
    labels: list[Label] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            raise SystemExit(f"bad YOLO label line {line_number} in {repo_rel(label_path)}")
        labels.append(Label(int(float(parts[0])), *(float(value) for value in parts[1:5])))
    return tuple(labels)


def label_to_pixels(label: Label, width: int, height: int) -> tuple[float, float, float, float]:
    box_w = label.width * width
    box_h = label.height * height
    x1 = label.xc * width - box_w / 2.0
    y1 = label.yc * height - box_h / 2.0
    x2 = label.xc * width + box_w / 2.0
    y2 = label.yc * height + box_h / 2.0
    return x1, y1, x2, y2


def clip_box(
    box: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop
    ix1 = max(x1, cx1)
    iy1 = max(y1, cy1)
    ix2 = min(x2, cx2)
    iy2 = min(y2, cy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1 - cx1, iy1 - cy1, ix2 - cx1, iy2 - cy1


def normalize_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def median_background(image: Image.Image, target_box: tuple[float, float, float, float]) -> tuple[int, int, int]:
    arr = np.asarray(image.convert("RGB"))
    height, width = arr.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in target_box)
    mask = np.ones((height, width), dtype=bool)
    mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = False
    pixels = arr[mask]
    if pixels.size == 0:
        pixels = arr.reshape(-1, 3)
    values = np.median(pixels, axis=0)
    return tuple(int(max(0, min(255, round(channel)))) for channel in values)


def clipped_rect(rect: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = rect
    result = (
        max(0, int(math.floor(x1))),
        max(0, int(math.floor(y1))),
        min(width, int(math.ceil(x2))),
        min(height, int(math.ceil(y2))),
    )
    if result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result


def noisy_rgb(size: tuple[int, int], base: tuple[int, int, int], rng: random.Random, spread: int) -> Image.Image:
    width, height = size
    arr = np.empty((height, width, 3), dtype=np.int16)
    arr[:, :, :] = np.array(base, dtype=np.int16)
    noise_rng = np.random.default_rng(rng.randrange(1, 2**32 - 1))
    arr += noise_rng.integers(-spread, spread + 1, size=arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def blocker_patch(
    image: Image.Image,
    rect: tuple[int, int, int, int],
    target_box: tuple[float, float, float, float],
    style: str,
    rng: random.Random,
) -> Image.Image:
    patch_w = rect[2] - rect[0]
    patch_h = rect[3] - rect[1]
    if patch_w <= 0 or patch_h <= 0:
        raise ValueError("blocker rect must have positive area")

    if style == "flat_dark":
        base = rng.choice([(12, 12, 12), (22, 24, 28), (35, 35, 38), (5, 18, 30)])
        return noisy_rgb((patch_w, patch_h), base, rng, 5)
    if style == "flat_light":
        base = rng.choice([(218, 216, 205), (236, 232, 218), (205, 214, 218), (226, 226, 226)])
        return noisy_rgb((patch_w, patch_h), base, rng, 7)
    if style == "background_sample":
        return noisy_rgb((patch_w, patch_h), median_background(image, target_box), rng, 9)

    base = rng.choice([(210, 217, 196), (213, 199, 218), (198, 220, 224), (225, 215, 190), (205, 205, 212)])
    patch = noisy_rgb((patch_w, patch_h), base, rng, 16)
    draw = ImageDraw.Draw(patch)
    if style == "noise_paper":
        for _ in range(max(3, (patch_w + patch_h) // 50)):
            y = rng.randrange(0, max(1, patch_h))
            color = tuple(max(0, min(255, channel + rng.randrange(-45, 46))) for channel in base)
            draw.line([(0, y), (patch_w, y + rng.randrange(-5, 6))], fill=color, width=rng.randrange(1, 3))
        return patch
    if style == "printed_paper":
        accent = rng.choice([(50, 95, 115), (110, 75, 130), (125, 90, 45), (45, 115, 75)])
        for _ in range(max(2, patch_w // 45)):
            x = rng.randrange(-patch_w // 3, max(1, patch_w))
            draw.ellipse(
                (x, rng.randrange(-patch_h // 4, max(1, patch_h)), x + patch_w // 2, patch_h + patch_h // 5),
                outline=accent,
                width=1,
            )
        for _ in range(max(3, patch_h // 22)):
            y = rng.randrange(0, max(1, patch_h))
            draw.line([(rng.randrange(0, max(1, patch_w // 3)), y), (patch_w, y)], fill=accent, width=1)
        try:
            font = ImageFont.truetype("arial.ttf", max(10, min(22, patch_h // 5)))
        except OSError:
            font = ImageFont.load_default()
        token = rng.choice(["100", "500", "20", "VOID", "NOTE", "////"])
        draw.text((max(2, patch_w // 12), max(2, patch_h // 5)), token, fill=accent, font=font)
        return patch

    raise ValueError(f"unknown blocker style: {style}")


def border_variant(
    image: Image.Image,
    label: Label,
    spec: VariantSpec,
    *,
    min_output_side: int,
    min_visible_pixels: int,
) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    width, height = image.size
    x1, y1, x2, y2 = label_to_pixels(label, width, height)
    frac = spec.visible_fraction
    if spec.mode == "left":
        crop = (0.0, 0.0, x1 + (x2 - x1) * frac, float(height))
    elif spec.mode == "right":
        crop = (x2 - (x2 - x1) * frac, 0.0, float(width), float(height))
    elif spec.mode == "top":
        crop = (0.0, 0.0, float(width), y1 + (y2 - y1) * frac)
    elif spec.mode == "bottom":
        crop = (0.0, y2 - (y2 - y1) * frac, float(width), float(height))
    else:
        return None

    cx1, cy1, cx2, cy2 = crop
    crop_i = (
        max(0, int(math.floor(cx1))),
        max(0, int(math.floor(cy1))),
        min(width, int(math.ceil(cx2))),
        min(height, int(math.ceil(cy2))),
    )
    out_w = crop_i[2] - crop_i[0]
    out_h = crop_i[3] - crop_i[1]
    if out_w < min_output_side or out_h < min_output_side:
        return None
    visible = clip_box((x1, y1, x2, y2), tuple(float(v) for v in crop_i))
    if visible is None:
        return None
    vx1, vy1, vx2, vy2 = visible
    if (vx2 - vx1) * (vy2 - vy1) < min_visible_pixels:
        return None
    return image.crop(crop_i), visible


def countable_crop_variant(
    image: Image.Image,
    label: Label,
    spec: VariantSpec,
    *,
    min_output_side: int,
    min_visible_pixels: int,
) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    width, height = image.size
    x1, y1, x2, y2 = label_to_pixels(label, width, height)
    box_w = x2 - x1
    box_h = y2 - y1
    frac = min(0.92, max(0.18, spec.visible_fraction))

    if spec.mode == "center_x":
        cut_w = box_w * frac
        mid = (x1 + x2) / 2.0
        crop = (mid - cut_w / 2.0, 0.0, mid + cut_w / 2.0, float(height))
    elif spec.mode == "center_y":
        cut_h = box_h * frac
        mid = (y1 + y2) / 2.0
        crop = (0.0, mid - cut_h / 2.0, float(width), mid + cut_h / 2.0)
    elif spec.mode == "corner_tl":
        crop = (0.0, 0.0, x1 + box_w * frac, y1 + box_h * frac)
    elif spec.mode == "corner_tr":
        crop = (x2 - box_w * frac, 0.0, float(width), y1 + box_h * frac)
    elif spec.mode == "corner_bl":
        crop = (0.0, y2 - box_h * frac, x1 + box_w * frac, float(height))
    elif spec.mode == "corner_br":
        crop = (x2 - box_w * frac, y2 - box_h * frac, float(width), float(height))
    else:
        return None

    cx1, cy1, cx2, cy2 = crop
    crop_i = (
        max(0, int(math.floor(cx1))),
        max(0, int(math.floor(cy1))),
        min(width, int(math.ceil(cx2))),
        min(height, int(math.ceil(cy2))),
    )
    out_w = crop_i[2] - crop_i[0]
    out_h = crop_i[3] - crop_i[1]
    if out_w < min_output_side or out_h < min_output_side:
        return None
    visible = clip_box((x1, y1, x2, y2), tuple(float(v) for v in crop_i))
    if visible is None:
        return None
    vx1, vy1, vx2, vy2 = visible
    if (vx2 - vx1) * (vy2 - vy1) < min_visible_pixels:
        return None
    return image.crop(crop_i), visible


def diagonal_variant(
    image: Image.Image,
    label: Label,
    spec: VariantSpec,
    *,
    min_visible_pixels: int,
) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    width, height = image.size
    x1, y1, x2, y2 = label_to_pixels(label, width, height)
    frac = min(0.92, max(0.35, math.sqrt(spec.visible_fraction)))
    leg_w = (x2 - x1) * frac
    leg_h = (y2 - y1) * frac
    if leg_w * leg_h < min_visible_pixels:
        return None

    out = image.convert("RGB").copy()
    fill = median_background(out, (x1, y1, x2, y2))
    draw = ImageDraw.Draw(out)
    if spec.mode == "diag_tl":
        visible = (x1, y1, x1 + leg_w, y1 + leg_h)
        polygon = [(x1 + leg_w, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1 + leg_h)]
    elif spec.mode == "diag_br":
        visible = (x2 - leg_w, y2 - leg_h, x2, y2)
        polygon = [(x1, y1), (x2, y1), (x2, y2 - leg_h), (x2 - leg_w, y2), (x1, y2)]
    else:
        return None
    draw.polygon([(round(x), round(y)) for x, y in polygon], fill=fill)
    return out, visible


def box_occlusion_variant(
    image: Image.Image,
    label: Label,
    spec: VariantSpec,
    *,
    min_visible_pixels: int,
    rng: random.Random,
) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    width, height = image.size
    x1, y1, x2, y2 = label_to_pixels(label, width, height)
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 1 or box_h <= 1:
        return None

    frac = min(0.80, max(0.15, spec.visible_fraction))
    if spec.mode == "box_left":
        cut = x1 + box_w * frac
        visible = (x1, y1, cut, y2)
        raw_rect = (cut, y1 - box_h * rng.uniform(0.04, 0.16), x2 + box_w * rng.uniform(0.04, 0.18), y2 + box_h * rng.uniform(0.04, 0.16))
    elif spec.mode == "box_right":
        cut = x2 - box_w * frac
        visible = (cut, y1, x2, y2)
        raw_rect = (x1 - box_w * rng.uniform(0.04, 0.18), y1 - box_h * rng.uniform(0.04, 0.16), cut, y2 + box_h * rng.uniform(0.04, 0.16))
    elif spec.mode == "box_top":
        cut = y1 + box_h * frac
        visible = (x1, y1, x2, cut)
        raw_rect = (x1 - box_w * rng.uniform(0.04, 0.16), cut, x2 + box_w * rng.uniform(0.04, 0.16), y2 + box_h * rng.uniform(0.04, 0.18))
    elif spec.mode == "box_bottom":
        cut = y2 - box_h * frac
        visible = (x1, cut, x2, y2)
        raw_rect = (x1 - box_w * rng.uniform(0.04, 0.16), y1 - box_h * rng.uniform(0.04, 0.18), x2 + box_w * rng.uniform(0.04, 0.16), cut)
    else:
        return None

    vx1, vy1, vx2, vy2 = visible
    if (vx2 - vx1) * (vy2 - vy1) < min_visible_pixels:
        return None
    rect = clipped_rect(raw_rect, width, height)
    if rect is None:
        return None

    out = image.convert("RGB").copy()
    patch = blocker_patch(out, rect, (x1, y1, x2, y2), spec.blocker_style, rng)
    out.paste(patch, rect[:2])
    return out, visible


def make_variant(
    image_path: Path,
    label: Label,
    spec: VariantSpec,
    *,
    min_output_side: int,
    min_visible_pixels: int,
    seed: int,
) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if spec.mode in BORDER_MODES:
            return border_variant(
                image,
                label,
                spec,
                min_output_side=min_output_side,
                min_visible_pixels=min_visible_pixels,
            )
        if spec.mode in COUNTABLE_CROP_MODES:
            return countable_crop_variant(
                image,
                label,
                spec,
                min_output_side=min_output_side,
                min_visible_pixels=min_visible_pixels,
            )
        if spec.mode in DIAGONAL_MODES:
            return diagonal_variant(image, label, spec, min_visible_pixels=min_visible_pixels)
        if spec.mode in BOX_OCCLUSION_MODES:
            rng = random.Random(seed)
            return box_occlusion_variant(image, label, spec, min_visible_pixels=min_visible_pixels, rng=rng)
    raise SystemExit(f"unknown partial-visibility mode: {spec.mode}")


def collect_rows(data_yaml: Path, payload: dict[str, Any], *, single_label_only: bool) -> list[SourceRow]:
    rows: list[SourceRow] = []
    for split in ("train", "val", "test"):
        for image_path in split_images(data_yaml, payload, split):
            labels = read_labels(image_path)
            if single_label_only and len(labels) != 1:
                continue
            if labels:
                rows.append(SourceRow(split=split, image=image_path, labels=labels))
    return rows


def filter_source_groups(
    rows: list[SourceRow],
    *,
    include: set[str],
    exclude: set[str],
) -> list[SourceRow]:
    if not include and not exclude:
        return rows
    filtered: list[SourceRow] = []
    for row in rows:
        group = source_group_for_image(row.image)
        if include and group not in include:
            continue
        if group in exclude:
            continue
        filtered.append(row)
    return filtered


def selected_sources(
    rows: list[SourceRow],
    names: dict[int, str],
    *,
    split: str,
    cap_per_class: int,
    seed: int,
) -> list[SourceRow]:
    by_class: dict[int, list[SourceRow]] = defaultdict(list)
    for row in rows:
        if row.split != split:
            continue
        class_ids = sorted({label.class_id for label in row.labels})
        if len(class_ids) != 1:
            continue
        if class_ids[0] not in names:
            continue
        by_class[class_ids[0]].append(row)

    selected: list[SourceRow] = []
    for class_id, class_rows in sorted(by_class.items()):
        ranked = sorted(
            class_rows,
            key=lambda row: stable_int(str(seed), names[class_id], repo_rel(row.image)),
        )
        selected.extend(ranked[:cap_per_class])
    selected.sort(key=lambda row: (row.split, row.labels[0].class_id, repo_rel(row.image)))
    return selected


def variant_specs(
    modes: list[str],
    visibility: list[float],
    blocker_styles: list[str],
    count: int,
    *,
    seed_key: str,
    min_corner_visibility: float,
) -> list[VariantSpec]:
    all_specs: list[VariantSpec] = []
    for mode in modes:
        if mode in BOX_OCCLUSION_MODES:
            styles = blocker_styles
        elif mode in BORDER_MODES or mode in DIAGONAL_MODES or mode in COUNTABLE_CROP_MODES:
            styles = ["none"]
        else:
            raise SystemExit(f"unknown partial-visibility mode: {mode}")
        all_specs.extend(
            VariantSpec(mode=mode, visible_fraction=frac, blocker_style=style)
            for frac in visibility
            if mode not in CORNER_CROP_MODES or frac >= min_corner_visibility
            for style in styles
        )
    all_specs.sort(
        key=lambda spec: stable_int(seed_key, spec.mode, spec.blocker_style, f"{spec.visible_fraction:.3f}")
    )
    return all_specs[:count]


def write_label(path: Path, label: Label, box: tuple[float, float, float, float], image_size: tuple[int, int]) -> None:
    width, height = image_size
    xc, yc, box_w, box_h = normalize_box(box, width, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{label.class_id} {xc:.6f} {yc:.6f} {box_w:.6f} {box_h:.6f}\n",
        encoding="utf-8",
    )


def write_data_yaml(path: Path, names: dict[int, str], *, label_policy: str) -> None:
    payload = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: names[index] for index in sorted(names)},
        "cashsnap_partial_visibility": {
            "schema": "cashsnap_real_partial_visibility_dataset_v1",
            "label_policy": label_policy,
            "not_a_promotion_config": True,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def draw_preview(path: Path, rows: list[dict[str, str]], names: dict[int, str], *, max_items: int) -> None:
    sample = rows[:max_items]
    if not sample:
        return
    thumb_w = 220
    thumb_h = 220
    caption_h = 54
    cols = 6
    row_count = math.ceil(len(sample) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, row_count * (thumb_h + caption_h)), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for index, row in enumerate(sample):
        image_path = ROOT / row["image"]
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        tile = Image.new("RGB", (thumb_w, thumb_h), (20, 20, 20))
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        offset = ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2)
        tile.paste(image, offset)
        label = Label(int(row["class_id"]), *(float(value) for value in row["label_xywh"].split()))
        x1, y1, x2, y2 = label_to_pixels(label, image.width, image.height)
        box = (
            offset[0] + x1,
            offset[1] + y1,
            offset[0] + x2,
            offset[1] + y2,
        )
        ImageDraw.Draw(tile).rectangle(box, outline=(20, 220, 80), width=2)
        x = (index % cols) * thumb_w
        y = (index // cols) * (thumb_h + caption_h)
        sheet.paste(tile, (x, y))
        caption = (
            f"{row['split']} {row['class_name']} {row['mode']} "
            f"{row['blocker_style']} vis={row['target_visible_fraction']}"
        )
        draw.text((x + 4, y + thumb_h + 3), caption[:42], fill=(0, 0, 0), font=font)
        draw.text((x + 4, y + thumb_h + 21), Path(row["source_image"]).stem[:42], fill=(50, 50, 50), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def main() -> int:
    args = parse_args()
    source_data = resolve(args.source_data)
    out_root = resolve(args.out_root)
    modes = list(args.modes)
    blocker_styles = [str(value) for value in args.blocker_styles]
    uses_diagonal = any(mode in DIAGONAL_MODES for mode in modes)
    uses_box_occlusion = any(mode in BOX_OCCLUSION_MODES for mode in modes)
    policy_parts = []
    if any(mode in BORDER_MODES for mode in modes):
        policy_parts.append("border crop")
    if any(mode in COUNTABLE_CROP_MODES for mode in modes):
        policy_parts.append("countable center/corner crop")
    if uses_diagonal:
        policy_parts.append("diagonal occlusion")
    if uses_box_occlusion:
        policy_parts.append("in-bounding-box blocker occlusion")
    if not policy_parts:
        raise SystemExit("at least one supported --modes value is required")
    label_policy = "single visible-instance AABB after " + ", ".join(policy_parts)
    if args.clean:
        safe_clean(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    payload = read_yaml(source_data)
    names = names_by_id(payload)
    include_source_groups = {value.strip() for value in args.include_source_group if value.strip()}
    exclude_source_groups = {value.strip() for value in args.exclude_source_group if value.strip()}
    rows_all = collect_rows(source_data, payload, single_label_only=args.single_label_only)
    rows = filter_source_groups(rows_all, include=include_source_groups, exclude=exclude_source_groups)
    selected: dict[str, list[SourceRow]] = {
        "train": selected_sources(
            rows,
            names,
            split="train",
            cap_per_class=args.train_sources_per_class,
            seed=args.seed,
        ),
        "val": selected_sources(
            rows,
            names,
            split="val",
            cap_per_class=args.eval_sources_per_class,
            seed=args.seed + 1,
        ),
        "test": selected_sources(
            rows,
            names,
            split="test",
            cap_per_class=args.eval_sources_per_class,
            seed=args.seed + 2,
        ),
    }

    manifest_rows: list[dict[str, str]] = []
    skipped: Counter[str] = Counter()
    for split, split_rows in selected.items():
        for source in split_rows:
            label = source.labels[0]
            class_name = names[label.class_id]
            specs = variant_specs(
                modes,
                [float(value) for value in args.visibility],
                blocker_styles,
                args.variants_per_source,
                seed_key=f"{args.seed}|{split}|{repo_rel(source.image)}",
                min_corner_visibility=args.min_corner_visibility,
            )
            for spec in specs:
                variant_seed = stable_int(
                    str(args.seed),
                    split,
                    repo_rel(source.image),
                    spec.mode,
                    spec.blocker_style,
                    f"{spec.visible_fraction:.3f}",
                )
                result = make_variant(
                    source.image,
                    label,
                    spec,
                    min_output_side=args.min_output_side,
                    min_visible_pixels=args.min_visible_pixels,
                    seed=variant_seed,
                )
                if result is None:
                    skipped[f"{split}:{class_name}:{spec.mode}"] += 1
                    continue
                image, visible_box = result
                stem = source.image.stem
                frac_tag = str(spec.visible_fraction).replace(".", "p")
                style_tag = "" if spec.blocker_style == "none" else f"_{spec.blocker_style}"
                out_name = f"{class_name}_{stem}_{spec.mode}{style_tag}_vis{frac_tag}.jpg"
                image_rel = Path("images") / split / out_name
                label_rel = Path("labels") / split / out_name.replace(".jpg", ".txt")
                image_path = out_root / image_rel
                label_path = out_root / label_rel
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(image_path, quality=args.jpeg_quality)
                write_label(label_path, label, visible_box, image.size)
                xc, yc, box_w, box_h = normalize_box(visible_box, *image.size)
                manifest_rows.append(
                    {
                        "split": split,
                        "image": repo_rel(image_path),
                        "label": repo_rel(label_path),
                        "source_image": repo_rel(source.image),
                        "source_label": repo_rel(label_path_for_image(source.image)),
                        "source_group": source_group_for_image(source.image),
                        "class_id": str(label.class_id),
                        "class_name": class_name,
                        "mode": spec.mode,
                        "blocker_style": spec.blocker_style,
                        "target_visible_fraction": f"{spec.visible_fraction:.2f}",
                        "output_width": str(image.size[0]),
                        "output_height": str(image.size[1]),
                        "label_xywh": f"{xc:.6f} {yc:.6f} {box_w:.6f} {box_h:.6f}",
                    }
                )

    manifest_path = out_root / "manifest.csv"
    fields = [
        "split",
        "image",
        "label",
        "source_image",
        "source_label",
        "source_group",
        "class_id",
        "class_name",
        "mode",
        "blocker_style",
        "target_visible_fraction",
        "output_width",
        "output_height",
        "label_xywh",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    data_yaml = out_root / "data.yaml"
    write_data_yaml(data_yaml, names, label_policy=label_policy)
    preview_path = out_root / "preview_sheet.jpg"
    draw_preview(preview_path, manifest_rows, names, max_items=args.preview_count)

    summary = {
        "schema": "cashsnap_real_partial_visibility_dataset_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_data": repo_rel(source_data),
        "out_root": repo_rel(out_root),
        "data_yaml": repo_rel(data_yaml),
        "manifest_csv": repo_rel(manifest_path),
        "preview_sheet": repo_rel(preview_path),
        "single_label_only": bool(args.single_label_only),
        "label_policy": label_policy,
        "train_sources_per_class": args.train_sources_per_class,
        "eval_sources_per_class": args.eval_sources_per_class,
        "variants_per_source": args.variants_per_source,
        "visibility": [float(value) for value in args.visibility],
        "min_corner_visibility": float(args.min_corner_visibility),
        "modes": modes,
        "blocker_styles": blocker_styles,
        "include_source_groups": sorted(include_source_groups),
        "exclude_source_groups": sorted(exclude_source_groups),
        "generated_images": len(manifest_rows),
        "generated_by_split": dict(Counter(row["split"] for row in manifest_rows)),
        "generated_by_class": dict(Counter(row["class_name"] for row in manifest_rows).most_common()),
        "generated_by_mode": dict(Counter(row["mode"] for row in manifest_rows).most_common()),
        "generated_by_source_group": dict(Counter(row["source_group"] for row in manifest_rows).most_common()),
        "generated_by_blocker_style": dict(Counter(row["blocker_style"] for row in manifest_rows).most_common()),
        "source_rows_seen": len(rows_all),
        "source_rows_after_source_filter": len(rows),
        "selected_sources_by_split": {split: len(values) for split, values in selected.items()},
        "skipped": dict(skipped),
        "not_a_promotion_config": True,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"partial_visibility_dataset={repo_rel(out_root)} images={len(manifest_rows)} "
        f"by_split={summary['generated_by_split']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
