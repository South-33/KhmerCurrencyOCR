"""Build a target-anchored synthetic YOLO pack from CashSnap train backgrounds.

This is a regime-change probe: keep the denomination labels synthetic, but draw
the camera/context distribution from real CashSnap train images.  It uses only
empty-label train frames as canvases by default so held-out validation/test
photos remain untouched and positive real labels are not copied into training.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("build_cashsnap_target_anchor_transplant.py requires opencv-python/cv2") from exc


ROOT = Path(__file__).resolve().parents[1]
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
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Asset:
    path: Path
    class_name: str
    side: str
    years: str
    status: str
    max_year: int


@dataclass(frozen=True)
class BoxSample:
    class_name: str
    image_path: Path
    cx: float
    cy: float
    width: float
    height: float
    image_width: int
    image_height: int


@dataclass(frozen=True)
class Background:
    image_path: Path
    label_path: Path
    source: str


@dataclass(frozen=True)
class ForegroundStyle:
    class_name: str
    image_path: Path
    mean_rgb: tuple[float, float, float]
    std_rgb: tuple[float, float, float]
    luma_mean: float
    luma_std: float
    sharpness: float
    crop_width: int
    crop_height: int


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed = (ROOT / "data" / "synthetic").resolve()
    if allowed not in resolved.parents and resolved != allowed:
        raise SystemExit(f"Refusing to clean outside {allowed}: {resolved}")
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
    images: list[Path] = []
    for suffix in IMAGE_EXTS:
        images.extend(image_dir.glob(f"*{suffix}"))
    return sorted(images)


def label_rows(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    return [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_cashsnap_train(cashsnap_root: Path) -> tuple[list[Background], dict[str, list[BoxSample]]]:
    backgrounds: list[Background] = []
    boxes_by_class: dict[str, list[BoxSample]] = defaultdict(list)
    for image_path in iter_train_images(cashsnap_root):
        label_path = label_path_for_image(image_path, cashsnap_root)
        rows = label_rows(label_path)
        if not rows:
            backgrounds.append(Background(image_path=image_path, label_path=label_path, source="cashsnap_empty_train"))
            continue
        for line_no, row in enumerate(rows, start=1):
            parts = row.split()
            if len(parts) != 5:
                raise ValueError(f"{repo_rel(label_path)}:{line_no} expected 5 YOLO fields, got {len(parts)}")
            class_id = int(parts[0])
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                raise ValueError(f"{repo_rel(label_path)}:{line_no} unknown class id {class_id}")
            cx, cy, width, height = [float(value) for value in parts[1:]]
            if width <= 0 or height <= 0:
                continue
            boxes_by_class[CLASS_NAMES[class_id]].append(
                BoxSample(
                    class_name=CLASS_NAMES[class_id],
                    image_path=image_path,
                    cx=cx,
                    cy=cy,
                    width=width,
                    height=height,
                    image_width=0,
                    image_height=0,
                )
            )
    if not backgrounds:
        raise SystemExit(f"No empty-label train backgrounds found under {repo_rel(cashsnap_root)}")
    missing = [name for name in CLASS_NAMES if not boxes_by_class.get(name)]
    if missing:
        raise SystemExit(f"Missing real train geometry samples for classes: {', '.join(missing)}")
    return backgrounds, boxes_by_class


def load_patch_backgrounds(background_root: Path, split: str) -> list[Background]:
    backgrounds: list[Background] = []
    for path in sorted(background_root.glob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.stem == "contact_sheet":
            continue
        if split and not path.stem.endswith(f"_{split}"):
            continue
        backgrounds.append(Background(image_path=path, label_path=path.with_suffix(".txt"), source="no_note_patch"))
    if not backgrounds:
        raise SystemExit(f"No background patches found under {repo_rel(background_root)} for split '{split}'")
    return backgrounds


def load_geometry_manifest(manifest_path: Path) -> set[Path]:
    if not manifest_path.exists():
        raise SystemExit(f"Missing geometry manifest: {repo_rel(manifest_path)}")
    selected: set[Path] = set()
    if manifest_path.suffix.lower() == ".jsonl":
        for line_no, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            image = payload.get("image")
            if not image:
                raise SystemExit(f"{repo_rel(manifest_path)}:{line_no} missing image field")
            selected.add(repo_path(image).resolve())
    elif manifest_path.suffix.lower() == ".csv":
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "image" not in (reader.fieldnames or []):
                raise SystemExit(f"{repo_rel(manifest_path)} must include an image column")
            for row in reader:
                if row.get("image"):
                    selected.add(repo_path(row["image"]).resolve())
    else:
        for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                selected.add(repo_path(line).resolve())
    if not selected:
        raise SystemExit(f"Geometry manifest selected no images: {repo_rel(manifest_path)}")
    return selected


def apply_geometry_manifest(
    boxes_by_class: dict[str, list[BoxSample]],
    manifest_path: Path | None,
    mode: str,
) -> tuple[dict[str, list[BoxSample]], dict[str, Any] | None]:
    if manifest_path is None:
        return boxes_by_class, None
    selected_images = load_geometry_manifest(manifest_path)
    filtered: dict[str, list[BoxSample]] = {
        class_name: [box for box in boxes if box.image_path.resolve() in selected_images]
        for class_name, boxes in boxes_by_class.items()
    }
    if mode == "only":
        missing = [class_name for class_name in CLASS_NAMES if not filtered.get(class_name)]
        if missing:
            raise SystemExit(
                "--geometry-manifest-mode only selected no geometry for classes: " + ", ".join(missing)
            )
        selected_boxes = filtered
    elif mode == "prefer":
        selected_boxes = {
            class_name: filtered[class_name] if filtered.get(class_name) else boxes_by_class[class_name]
            for class_name in CLASS_NAMES
        }
    else:
        raise SystemExit("--geometry-manifest-mode must be one of: prefer, only")
    return selected_boxes, {
        "path": repo_rel(manifest_path),
        "mode": mode,
        "images": len(selected_images),
        "selected_boxes_by_class": {
            class_name: len(filtered.get(class_name, []))
            for class_name in CLASS_NAMES
            if filtered.get(class_name)
        },
        "fallback_classes": [
            class_name
            for class_name in CLASS_NAMES
            if mode == "prefer" and not filtered.get(class_name)
        ],
    }


def sample_xyxy(sample: BoxSample, image_size: tuple[int, int], pad_fraction: float = 0.0) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    box_w = sample.width * image_width
    box_h = sample.height * image_height
    pad_x = box_w * pad_fraction
    pad_y = box_h * pad_fraction
    x1 = int(round(sample.cx * image_width - box_w / 2 - pad_x))
    y1 = int(round(sample.cy * image_height - box_h / 2 - pad_y))
    x2 = int(round(sample.cx * image_width + box_w / 2 + pad_x))
    y2 = int(round(sample.cy * image_height + box_h / 2 + pad_y))
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(x1 + 1, min(image_width, x2))
    y2 = max(y1 + 1, min(image_height, y2))
    return x1, y1, x2, y2


def crop_sharpness(crop: Image.Image) -> float:
    gray = cv2.cvtColor(np.asarray(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    max_side = max(width, height)
    if max_side > 160:
        scale = 160.0 / max_side
        gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def load_foreground_styles(
    boxes_by_class: dict[str, list[BoxSample]],
    rng: random.Random,
    max_class_samples: int,
) -> dict[str, list[ForegroundStyle]]:
    samples_by_image: dict[Path, list[BoxSample]] = defaultdict(list)
    for samples in boxes_by_class.values():
        selected = samples
        if max_class_samples > 0 and len(samples) > max_class_samples:
            selected = rng.sample(samples, max_class_samples)
        for sample in selected:
            samples_by_image[sample.image_path].append(sample)

    styles: dict[str, list[ForegroundStyle]] = defaultdict(list)
    for image_path, samples in sorted(samples_by_image.items(), key=lambda item: item[0].as_posix()):
        with Image.open(image_path).convert("RGB") as image:
            image_size = image.size
            for sample in samples:
                x1, y1, x2, y2 = sample_xyxy(sample, image_size, pad_fraction=0.015)
                if (x2 - x1) < 12 or (y2 - y1) < 8:
                    continue
                crop = image.crop((x1, y1, x2, y2))
                arr = np.asarray(crop).astype(np.float32)
                if arr.shape[0] * arr.shape[1] < 128:
                    continue
                pixels = arr.reshape(-1, 3)
                mean_rgb = pixels.mean(axis=0)
                std_rgb = np.maximum(pixels.std(axis=0), 1.0)
                luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
                styles[sample.class_name].append(
                    ForegroundStyle(
                        class_name=sample.class_name,
                        image_path=image_path,
                        mean_rgb=tuple(float(value) for value in mean_rgb),
                        std_rgb=tuple(float(value) for value in std_rgb),
                        luma_mean=float(luma.mean()),
                        luma_std=float(luma.std()),
                        sharpness=crop_sharpness(crop),
                        crop_width=crop.width,
                        crop_height=crop.height,
                    )
                )

    missing = [name for name in CLASS_NAMES if not styles.get(name)]
    if missing:
        raise SystemExit(f"Missing foreground style crops for classes: {', '.join(missing)}")
    return dict(styles)


def foreground_style_quantiles(styles_by_class: dict[str, list[ForegroundStyle]]) -> dict[str, float]:
    sharpness = np.array(
        [style.sharpness for styles in styles_by_class.values() for style in styles],
        dtype=np.float32,
    )
    if sharpness.size == 0:
        return {"sharpness_p25": 0.0, "sharpness_p50": 0.0, "sharpness_p75": 0.0}
    return {
        "sharpness_p25": float(np.percentile(sharpness, 25)),
        "sharpness_p50": float(np.percentile(sharpness, 50)),
        "sharpness_p75": float(np.percentile(sharpness, 75)),
    }


def choose_foreground_style(
    class_name: str,
    styles_by_class: dict[str, list[ForegroundStyle]],
    rng: random.Random,
    min_class_samples: int,
) -> ForegroundStyle:
    class_styles = styles_by_class.get(class_name, [])
    if len(class_styles) >= min_class_samples or class_styles:
        return rng.choice(class_styles)
    all_styles = [style for styles in styles_by_class.values() for style in styles]
    return rng.choice(all_styles)


def load_assets(
    manifest_path: Path,
    allowed_status: set[str],
    allowed_sides: set[str],
    asset_quality_policy: str,
) -> dict[str, list[Asset]]:
    assets: dict[str, list[Asset]] = defaultdict(list)
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            class_name = str(row.get("class_name", ""))
            if class_name not in CLASS_TO_ID:
                continue
            if allowed_status and str(row.get("status", "")) not in allowed_status:
                continue
            if allowed_sides and str(row.get("side", "")) not in allowed_sides:
                continue
            path = repo_path(row.get("asset_path", ""))
            if not path.exists():
                continue
            max_year = int(row.get("max_year") or 0)
            assets[class_name].append(
                Asset(
                    path=path,
                    class_name=class_name,
                    side=str(row.get("side", "")),
                    years=str(row.get("years", "")),
                    status=str(row.get("status", "")),
                    max_year=max_year,
                )
            )
    if asset_quality_policy == "latest_design":
        latest: dict[str, list[Asset]] = defaultdict(list)
        for class_name, class_assets in assets.items():
            by_side: dict[str, list[Asset]] = defaultdict(list)
            for asset in class_assets:
                by_side[asset.side].append(asset)
            for side_assets in by_side.values():
                latest[class_name].append(
                    max(
                        side_assets,
                        key=lambda asset: (
                            asset.max_year,
                            asset.path.stat().st_size if asset.path.exists() else 0,
                            asset.path.as_posix(),
                        ),
                    )
                )
        assets = latest
    elif asset_quality_policy != "all_manifest":
        raise SystemExit("--asset-quality-policy must be one of: latest_design, all_manifest")
    missing = [name for name in CLASS_NAMES if not assets.get(name)]
    if missing:
        raise SystemExit(f"Manifest selected no assets for classes: {', '.join(missing)}")
    return dict(assets)


def choose_class(rng: random.Random, counts: Counter[str]) -> str:
    # Keep exposure balanced while preserving the global class-id order.
    min_count = min(counts.get(name, 0) for name in CLASS_NAMES)
    candidates = [name for name in CLASS_NAMES if counts.get(name, 0) == min_count]
    return rng.choice(candidates)


def sample_geometry(
    class_name: str,
    boxes_by_class: dict[str, list[BoxSample]],
    rng: random.Random,
    min_class_geometry_samples: int,
) -> BoxSample:
    class_boxes = boxes_by_class[class_name]
    if len(class_boxes) >= min_class_geometry_samples:
        return rng.choice(class_boxes)
    all_boxes = [box for boxes in boxes_by_class.values() for box in boxes]
    return rng.choice(all_boxes)


def jitter_box(
    sample: BoxSample,
    image_size: tuple[int, int],
    asset_aspect: float,
    rng: random.Random,
    box_scale: float,
    box_scale_jitter: float,
) -> tuple[float, float, float, float]:
    width, height = image_size
    box_w = sample.width * width * rng.uniform(0.86, 1.14)
    box_h = sample.height * height * rng.uniform(0.86, 1.14)
    if box_scale_jitter > 0:
        scale_jitter = rng.uniform(max(0.10, 1.0 - box_scale_jitter), 1.0 + box_scale_jitter)
    else:
        scale_jitter = 1.0
    box_w *= box_scale * scale_jitter
    box_h *= box_scale * scale_jitter
    sampled_aspect = box_w / max(1.0, box_h)
    target_aspect = float(np.clip(0.65 * sampled_aspect + 0.35 * asset_aspect, 1.45, 3.35))
    area = max(32.0 * 16.0, box_w * box_h)
    box_w = math.sqrt(area * target_aspect)
    box_h = box_w / target_aspect
    max_w = width * rng.uniform(0.34, 0.92)
    max_h = height * rng.uniform(0.16, 0.70)
    if box_w > max_w:
        scale = max_w / box_w
        box_w *= scale
        box_h *= scale
    if box_h > max_h:
        scale = max_h / box_h
        box_w *= scale
        box_h *= scale
    cx = sample.cx * width + rng.uniform(-0.045, 0.045) * width
    cy = sample.cy * height + rng.uniform(-0.045, 0.045) * height
    cx = float(np.clip(cx, box_w * 0.42, width - box_w * 0.42))
    cy = float(np.clip(cy, box_h * 0.42, height - box_h * 0.42))
    return cx, cy, box_w, box_h


def rotated_quad(cx: float, cy: float, width: float, height: float, angle_deg: float) -> np.ndarray:
    corners = np.array(
        [[-width / 2, -height / 2], [width / 2, -height / 2], [width / 2, height / 2], [-width / 2, height / 2]],
        dtype=np.float32,
    )
    theta = math.radians(angle_deg)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float32)
    return corners @ rot.T + np.array([cx, cy], dtype=np.float32)


def jitter_quad(quad: np.ndarray, rng: random.Random, amount: float) -> np.ndarray:
    jitter = np.array([[rng.uniform(-amount, amount), rng.uniform(-amount, amount)] for _ in range(4)], dtype=np.float32)
    return quad + jitter


def shift_quad_inside(quad: np.ndarray, image_size: tuple[int, int], margin: float = 2.0) -> np.ndarray:
    width, height = image_size
    dx = 0.0
    dy = 0.0
    min_x = float(quad[:, 0].min())
    max_x = float(quad[:, 0].max())
    min_y = float(quad[:, 1].min())
    max_y = float(quad[:, 1].max())
    if min_x < margin:
        dx = margin - min_x
    elif max_x > width - margin:
        dx = (width - margin) - max_x
    if min_y < margin:
        dy = margin - min_y
    elif max_y > height - margin:
        dy = (height - margin) - max_y
    return quad + np.array([dx, dy], dtype=np.float32)


def aabb_aspect_for_angle(physical_aspect: float, angle_deg: float) -> float:
    theta = math.radians(abs(angle_deg))
    c = abs(math.cos(theta))
    s = abs(math.sin(theta))
    width = physical_aspect * c + s
    height = physical_aspect * s + c
    return width / max(0.001, height)


def angle_for_aabb_aspect(physical_aspect: float, target_aspect: float) -> float:
    best_angle = 0.0
    best_error = float("inf")
    for angle in np.linspace(0.0, 72.0, 145):
        error = abs(aabb_aspect_for_angle(physical_aspect, float(angle)) - target_aspect)
        if error < best_error:
            best_error = error
            best_angle = float(angle)
    return best_angle


def prepare_note(asset: Asset, rng: random.Random) -> Image.Image:
    with Image.open(asset.path).convert("RGBA") as image:
        note = image.copy()
    max_width = 900
    if note.width > max_width:
        ratio = max_width / note.width
        note = note.resize((max_width, max(1, int(note.height * ratio))), Image.Resampling.LANCZOS)
    note = ImageEnhance.Brightness(note).enhance(rng.uniform(0.72, 1.20))
    note = ImageEnhance.Contrast(note).enhance(rng.uniform(0.78, 1.18))
    note = ImageEnhance.Color(note).enhance(rng.uniform(0.78, 1.22))
    if rng.random() < 0.35:
        note = note.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.85)))
    return note


def jpeg_roundtrip_rgba(note: Image.Image, rng: random.Random) -> Image.Image:
    alpha = note.getchannel("A")
    buffer = io.BytesIO()
    note.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=rng.randint(58, 92),
        subsampling=rng.choice([0, 1, 2]),
    )
    buffer.seek(0)
    with Image.open(buffer).convert("RGB") as compressed:
        rgb = compressed.copy()
    return Image.merge("RGBA", (*rgb.split(), alpha))


def apply_foreground_style(
    note: Image.Image,
    style: ForegroundStyle,
    sharpness_quantiles: dict[str, float],
    rng: random.Random,
) -> Image.Image:
    arr = np.asarray(note.convert("RGBA")).astype(np.float32).copy()
    alpha = arr[:, :, 3] > 16
    if not alpha.any():
        return note

    rgb = arr[:, :, :3]
    note_pixels = rgb[alpha]
    note_mean = note_pixels.mean(axis=0)
    note_std = np.maximum(note_pixels.std(axis=0), 1.0)
    style_mean = np.array(style.mean_rgb, dtype=np.float32)
    style_std = np.maximum(np.array(style.std_rgb, dtype=np.float32), 4.0)

    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    target_mean = 0.42 * note_mean + 0.58 * style_mean + np_rng.normal(
        0.0, rng.uniform(1.0, 5.0), size=3
    )
    target_std = np.clip(0.30 * note_std + 0.70 * style_std, 8.0, 96.0)
    adjusted = (rgb - note_mean) * (target_std / note_std) * rng.uniform(0.86, 1.10) + target_mean
    if rng.random() < 0.70:
        adjusted += np_rng.normal(0.0, rng.uniform(0.6, 3.2), size=rgb.shape)
    rgb[alpha] = np.clip(adjusted[alpha], 0, 255)
    arr[:, :, :3] = rgb
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")

    if rng.random() < 0.78:
        softened_alpha = out.getchannel("A").filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.18, 0.85)))
        out.putalpha(softened_alpha)

    p25 = sharpness_quantiles.get("sharpness_p25", 0.0)
    p75 = sharpness_quantiles.get("sharpness_p75", p25)
    if style.sharpness <= p25 and rng.random() < 0.80:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.70)))
    elif style.sharpness >= p75 and rng.random() < 0.25:
        out = ImageEnhance.Sharpness(out).enhance(rng.uniform(1.08, 1.28))

    if rng.random() < 0.55:
        out = jpeg_roundtrip_rgba(out, rng)
    return out


def tone_match_note(note: Image.Image, background_patch: Image.Image, rng: random.Random) -> Image.Image:
    note_arr = np.asarray(note).astype(np.float32)
    alpha = note_arr[:, :, 3] > 16
    if not alpha.any():
        return note
    patch = np.asarray(background_patch.convert("RGB")).astype(np.float32)
    if patch.size == 0:
        return note
    patch_mean = patch.reshape(-1, 3).mean(axis=0)
    patch_std = patch.reshape(-1, 3).std(axis=0)
    note_rgb = note_arr[:, :, :3]
    note_pixels = note_rgb[alpha]
    note_mean = note_pixels.mean(axis=0)
    note_std = np.maximum(note_pixels.std(axis=0), 1.0)
    target_mean = 0.58 * note_mean + 0.42 * (patch_mean + rng.uniform(-18, 18))
    target_std = np.clip(0.62 * note_std + 0.38 * np.maximum(patch_std, 12.0), 18.0, 95.0)
    adjusted = (note_rgb - note_mean) * (target_std / note_std) * rng.uniform(0.82, 1.16) + target_mean
    note_arr[:, :, :3] = np.clip(adjusted, 0, 255)
    return Image.fromarray(note_arr.astype(np.uint8), "RGBA")


def warp_rgba(note: Image.Image, quad: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    src = np.array(
        [[0, 0], [note.width - 1, 0], [note.width - 1, note.height - 1], [0, note.height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    return cv2.warpPerspective(
        np.asarray(note.convert("RGBA")),
        matrix,
        out_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def alpha_composite(base_rgb: np.ndarray, layer_rgba: np.ndarray) -> np.ndarray:
    alpha = layer_rgba[:, :, 3:4].astype(np.float32) / 255.0
    out = layer_rgba[:, :, :3].astype(np.float32) * alpha + base_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_shadow(alpha: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha_image = Image.fromarray(alpha, "L").filter(ImageFilter.GaussianBlur(radius=rng.uniform(4.0, 14.0)))
    shadow = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    shadow[:, :, :3] = rng.randint(0, 18)
    shadow[:, :, 3] = (np.asarray(alpha_image).astype(np.float32) * rng.uniform(0.16, 0.38)).astype(np.uint8)
    shift_x = rng.randint(2, 14) * rng.choice([-1, 1])
    shift_y = rng.randint(3, 16)
    return np.roll(shadow, shift=(shift_y, shift_x), axis=(0, 1))


def shifted_zeros(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(arr)
    height, width = arr.shape[:2]
    src_y1 = max(0, -dy)
    src_y2 = min(height, height - dy)
    src_x1 = max(0, -dx)
    src_x2 = min(width, width - dx)
    dst_y1 = max(0, dy)
    dst_y2 = min(height, height + dy)
    dst_x1 = max(0, dx)
    dst_x2 = min(width, width + dx)
    if src_y2 <= src_y1 or src_x2 <= src_x1:
        return out
    out[dst_y1:dst_y2, dst_x1:dst_x2] = arr[src_y1:src_y2, src_x1:src_x2]
    return out


def make_contact_shadow(alpha: np.ndarray, rng: random.Random) -> np.ndarray:
    note_mask = (alpha > 16).astype(np.uint8)
    outside = (1 - note_mask).astype(np.uint8)
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
    radius = rng.uniform(4.5, 13.0)
    falloff = np.exp(-dist / max(radius, 0.1))
    falloff[dist > radius * rng.uniform(2.2, 3.7)] = 0.0
    falloff[note_mask > 0] = 0.0
    falloff = shifted_zeros(
        falloff,
        dy=rng.randint(1, 5),
        dx=rng.randint(-4, 4),
    )
    alpha_image = Image.fromarray(np.clip(falloff * 255.0, 0, 255).astype(np.uint8), "L")
    alpha_image = alpha_image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.7, 2.2)))
    shadow = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    shadow[:, :, :3] = rng.randint(0, 14)
    shadow[:, :, 3] = (np.asarray(alpha_image).astype(np.float32) * rng.uniform(0.08, 0.24)).astype(np.uint8)
    return shadow


def poisson_composite(
    base_rgb: np.ndarray,
    layer_rgba: np.ndarray,
    composite_policy: str,
    rng: random.Random,
) -> np.ndarray:
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
    flag = cv2.MIXED_CLONE if composite_policy == "poisson_mixed" else cv2.NORMAL_CLONE
    try:
        cloned_roi_bgr = cv2.seamlessClone(
            cv2.cvtColor(source_roi, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(base_roi, cv2.COLOR_RGB2BGR),
            mask_roi,
            center,
            flag,
        )
    except cv2.error:
        return alpha_composite(base_rgb, layer_rgba)
    cloned_roi_rgb = cv2.cvtColor(cloned_roi_bgr, cv2.COLOR_BGR2RGB)
    direct_roi_rgb = alpha_composite(base_roi, layer_roi)
    strength = rng.uniform(0.62, 0.86)
    blended = np.clip(
        cloned_roi_rgb.astype(np.float32) * strength + direct_roi_rgb.astype(np.float32) * (1.0 - strength),
        0,
        255,
    )
    out = base_rgb.copy()
    out[roi_y1:roi_y2, roi_x1:roi_x2] = cloned_roi_rgb
    roi_out = out[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_out[mask_roi > 0] = blended[mask_roi > 0].astype(np.uint8)
    out[roi_y1:roi_y2, roi_x1:roi_x2] = roi_out
    return out


def add_sensor_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    if rng.random() < 0.75:
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, rng.uniform(1.5, 6.5), arr.shape)
        arr += noise
    if rng.random() < 0.35:
        gain = np.array([rng.uniform(0.93, 1.08), rng.uniform(0.93, 1.08), rng.uniform(0.93, 1.08)])
        arr *= gain
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr, "RGB")
    if rng.random() < 0.30:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.55)))
    if rng.random() < 0.20:
        out = ImageOps.autocontrast(out, cutoff=rng.uniform(0.0, 1.2))
    return out


def mask_to_label(mask: np.ndarray, class_name: str) -> str | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    height, width = mask.shape
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    if (x2 - x1) < 12 or (y2 - y1) < 8 or mask.sum() < 280:
        return None
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{CLASS_TO_ID[class_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def render_one(
    background: Background,
    asset: Asset,
    boxes_by_class: dict[str, list[BoxSample]],
    foreground_styles_by_class: dict[str, list[ForegroundStyle]] | None,
    foreground_sharpness_quantiles: dict[str, float],
    rng: random.Random,
    min_class_geometry_samples: int,
    foreground_style_min_class_samples: int,
    canvas_size: tuple[int, int] | None,
    composite_policy: str,
    shadow_policy: str,
    box_scale: float,
    box_scale_jitter: float,
    pose_policy: str,
    min_render_short_px: float,
    min_render_width_px: float,
    min_render_height_px: float,
    min_render_area_px: float,
) -> tuple[Image.Image, str, dict[str, Any]] | None:
    with Image.open(background.image_path).convert("RGB") as image:
        base = image.copy()
    if canvas_size is not None:
        base = ImageOps.fit(base, canvas_size, method=Image.Resampling.LANCZOS)
    width, height = base.size
    sample = sample_geometry(asset.class_name, boxes_by_class, rng, min_class_geometry_samples)
    note = prepare_note(asset, rng)
    foreground_style: ForegroundStyle | None = None
    if foreground_styles_by_class is not None:
        foreground_style = choose_foreground_style(
            asset.class_name,
            foreground_styles_by_class,
            rng,
            foreground_style_min_class_samples,
        )
        note = apply_foreground_style(note, foreground_style, foreground_sharpness_quantiles, rng)
    asset_aspect = note.width / max(1, note.height)
    cx, cy, box_w, box_h = jitter_box(sample, (width, height), asset_aspect, rng, box_scale, box_scale_jitter)
    angle = rng.gauss(0, 8.0)
    if rng.random() < 0.18:
        angle += rng.choice([-1, 1]) * rng.uniform(10, 28)
    if pose_policy == "aabb_aspect_repair":
        sampled_aabb_aspect = (sample.width * width) / max(1.0, sample.height * height)
        physical_aspect = box_w / max(1.0, box_h)
        if sampled_aabb_aspect < physical_aspect * 0.92:
            repaired = angle_for_aabb_aspect(physical_aspect, max(0.55, sampled_aabb_aspect))
            angle = rng.choice([-1, 1]) * max(0.0, repaired + rng.uniform(-4.0, 4.0))
    quad = rotated_quad(cx, cy, box_w, box_h, angle)
    quad = jitter_quad(quad, rng, amount=min(box_w, box_h) * rng.uniform(0.0, 0.055))
    if pose_policy == "aabb_aspect_repair" or box_scale != 1.0:
        quad = shift_quad_inside(quad, (width, height))
    x1 = max(0, int(np.floor(quad[:, 0].min())) - 12)
    y1 = max(0, int(np.floor(quad[:, 1].min())) - 12)
    x2 = min(width, int(np.ceil(quad[:, 0].max())) + 12)
    y2 = min(height, int(np.ceil(quad[:, 1].max())) + 12)
    patch = base.crop((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)))
    note = tone_match_note(note, patch, rng)
    warped = warp_rgba(note, quad, (width, height))
    alpha = warped[:, :, 3]
    if alpha.max() <= 16:
        return None
    base_arr = np.asarray(base).copy()
    shadow = make_contact_shadow(alpha, rng) if shadow_policy == "contact" else make_shadow(alpha, rng)
    base_arr = alpha_composite(base_arr, shadow)
    if composite_policy == "alpha":
        base_arr = alpha_composite(base_arr, warped)
    else:
        base_arr = poisson_composite(base_arr, warped, composite_policy, rng)
    mask = alpha > 18
    if min_render_short_px > 0 or min_render_width_px > 0 or min_render_height_px > 0 or min_render_area_px > 0:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        render_width = int(xs.max()) - int(xs.min()) + 1
        render_height = int(ys.max()) - int(ys.min()) + 1
        render_area = int(mask.sum())
        if min(render_width, render_height) < min_render_short_px:
            return None
        if render_width < min_render_width_px:
            return None
        if render_height < min_render_height_px:
            return None
        if render_area < min_render_area_px:
            return None
    label = mask_to_label(mask, asset.class_name)
    if label is None:
        return None
    out = add_sensor_noise(Image.fromarray(base_arr, "RGB"), rng)
    metadata = {
        "class_name": asset.class_name,
        "asset": repo_rel(asset.path),
        "asset_side": asset.side,
        "asset_years": asset.years,
        "asset_max_year": asset.max_year,
        "background": repo_rel(background.image_path),
        "background_source": background.source,
        "geometry_source": repo_rel(sample.image_path),
        "quad_xy": [[round(float(x), 2), round(float(y), 2)] for x, y in quad],
        "note_angle_deg": round(float(angle), 3),
        "canvas_size": [width, height],
        "composite_policy": composite_policy,
        "shadow_policy": shadow_policy,
        "box_scale": round(float(box_scale), 4),
        "box_scale_jitter": round(float(box_scale_jitter), 4),
        "pose_policy": pose_policy,
        "min_render_short_px": round(float(min_render_short_px), 3),
        "min_render_width_px": round(float(min_render_width_px), 3),
        "min_render_height_px": round(float(min_render_height_px), 3),
        "min_render_area_px": round(float(min_render_area_px), 3),
    }
    if foreground_style is not None:
        metadata.update(
            {
                "foreground_style_source": repo_rel(foreground_style.image_path),
                "foreground_style_class": foreground_style.class_name,
                "foreground_style_luma_mean": round(float(foreground_style.luma_mean), 3),
                "foreground_style_luma_std": round(float(foreground_style.luma_std), 3),
                "foreground_style_sharpness": round(float(foreground_style.sharpness), 3),
            }
        )
    return out, label, metadata


def write_data_yaml(out_root: Path) -> None:
    payload = {
        "path": out_root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/train",
        "test": "images/train",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_realval_config(out_config: Path, train_root: Path, summary: dict[str, Any]) -> None:
    payload = {
        "path": "../..",
        "train": repo_rel(train_root / "images" / "train"),
        "val": "data/cashsnap_v1/images/val",
        "test": "data/cashsnap_v1/images/test",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        "cashsnap_policy": {
            "intended_use": (
                "pure-synth target-anchored transplant TSTR probe using train-only "
                "CashSnap no-note background patches and synthetic cutout-bank note labels"
            ),
            "phase": "target-domain anchor regime-change MVP",
            "source_dataset": repo_rel(train_root),
            "promotion_rule": (
                "reject unless fixed-step full real transfer jumps materially over "
                "filtered185/repair baselines without worse empty-frame behavior"
            ),
            **summary,
        },
    }
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def build_preview(out_root: Path, records: list[dict[str, Any]], count: int) -> None:
    if count <= 0 or not records:
        return
    chosen = records[:count]
    thumb_w, thumb_h = 220, 220
    cols = min(5, len(chosen))
    rows = math.ceil(len(chosen) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (35, 35, 35))
    for idx, record in enumerate(chosen):
        image_path = out_root / record["image"]
        label_path = out_root / record["label"]
        with Image.open(image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = parts
                cx_f, cy_f, bw_f, bh_f = [float(v) for v in (cx, cy, bw, bh)]
                x1 = (cx_f - bw_f / 2) * image.width
                y1 = (cy_f - bh_f / 2) * image.height
                x2 = (cx_f + bw_f / 2) * image.width
                y2 = (cy_f + bh_f / 2) * image.height
                draw.rectangle((x1, y1, x2, y2), outline=(255, 220, 20), width=3)
            thumb = ImageOps.contain(image, (thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb_w
        y = (idx // cols) * thumb_h
        sheet.paste(thumb, (x + (thumb_w - thumb.width) // 2, y + (thumb_h - thumb.height) // 2))
    sheet.save(out_root / "preview_contact.jpg", quality=90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build target-anchored CashSnap synthetic transplant data.")
    parser.add_argument("--cashsnap-root", default="data/cashsnap_v1")
    parser.add_argument(
        "--background-root",
        default="data/backgrounds/cashsnap_v1_no_note_patches_strict_v1",
        help="No-note background patch directory. Empty string uses empty-label CashSnap train frames.",
    )
    parser.add_argument("--background-split", default="train", help="Patch filename split suffix to use.")
    parser.add_argument(
        "--canvas-size",
        default="640,640",
        help="Output canvas width,height. Empty string preserves source background dimensions.",
    )
    parser.add_argument("--asset-manifest", default="data/asset_candidates/numista_current_cutout_bank_v1/manifest.csv")
    parser.add_argument(
        "--asset-quality-policy",
        default="latest_design",
        choices=("latest_design", "all_manifest"),
        help="Texture selection policy. latest_design keeps one current asset per class side.",
    )
    parser.add_argument(
        "--foreground-style-policy",
        default="none",
        choices=("none", "real_crop_stats"),
        help="Optionally match synthetic foreground notes to real CashSnap train crop appearance stats.",
    )
    parser.add_argument(
        "--foreground-style-min-class-samples",
        type=int,
        default=8,
        help="Minimum same-class foreground crop samples expected before using class-specific style.",
    )
    parser.add_argument(
        "--foreground-style-max-class-samples",
        type=int,
        default=256,
        help="Maximum real train crops to load per class for foreground style; <=0 disables the cap.",
    )
    parser.add_argument(
        "--composite-policy",
        default="alpha",
        choices=("alpha", "poisson_mixed", "poisson_normal"),
        help="Foreground/background blending policy. alpha preserves historical target-anchor behavior.",
    )
    parser.add_argument(
        "--shadow-policy",
        default="drop",
        choices=("drop", "contact"),
        help="Shadow model. drop preserves historical offset blur; contact uses a local distance falloff.",
    )
    parser.add_argument(
        "--box-scale",
        type=float,
        default=1.0,
        help="Opt-in multiplier for sampled real train box width/height. 1.0 preserves historical geometry.",
    )
    parser.add_argument(
        "--box-scale-jitter",
        type=float,
        default=0.0,
        help="Optional relative jitter around --box-scale, e.g. 0.12 samples 88-112 percent of the scale.",
    )
    parser.add_argument(
        "--pose-policy",
        default="current",
        choices=("current", "aabb_aspect_repair"),
        help="Pose sampling policy. aabb_aspect_repair rotates notes toward sampled real AABB aspect.",
    )
    parser.add_argument(
        "--min-render-short-px",
        type=float,
        default=0.0,
        help="Opt-in minimum rendered note AABB short side in output pixels; rejected samples are retried.",
    )
    parser.add_argument("--min-render-width-px", type=float, default=0.0)
    parser.add_argument("--min-render-height-px", type=float, default=0.0)
    parser.add_argument("--min-render-area-px", type=float, default=0.0)
    parser.add_argument("--out-root", default="data/synthetic/cashsnap_target_anchor_transplant_mvp_v1")
    parser.add_argument("--out-config", default="configs/webgl_ablation/cashsnap_target_anchor_transplant_mvp_puresynth_realval_v1.yaml")
    parser.add_argument("--per-class", type=int, default=96, help="Generated train positives per class.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--status", default="in_circulation", help="Comma-separated asset status filter; empty allows all.")
    parser.add_argument("--sides", default="front,back", help="Comma-separated asset side filter; empty allows all.")
    parser.add_argument(
        "--geometry-manifest",
        default="",
        help="Optional train-anchor manifest/list; rows with image fields restrict or prefer geometry sources.",
    )
    parser.add_argument(
        "--geometry-manifest-mode",
        default="prefer",
        choices=("prefer", "only"),
        help="prefer uses manifest geometry where available and falls back per missing class; only fails on missing classes.",
    )
    parser.add_argument(
        "--min-class-geometry-samples",
        type=int,
        default=48,
        help="Use class-specific real geometry only when the class has at least this many samples.",
    )
    parser.add_argument("--preview-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    cashsnap_root = repo_path(args.cashsnap_root)
    background_root = repo_path(args.background_root) if args.background_root else None
    manifest_path = repo_path(args.asset_manifest)
    out_root = repo_path(args.out_root)
    out_config = repo_path(args.out_config)
    if args.per_class <= 0:
        raise SystemExit("--per-class must be > 0")
    if args.clean:
        safe_clean(out_root)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "metadata").mkdir(parents=True, exist_ok=True)

    status = {item.strip() for item in args.status.split(",") if item.strip()}
    sides = {item.strip() for item in args.sides.split(",") if item.strip()}
    cashsnap_empty_backgrounds, boxes_by_class = load_cashsnap_train(cashsnap_root)
    geometry_manifest_path = repo_path(args.geometry_manifest) if args.geometry_manifest.strip() else None
    boxes_by_class, geometry_manifest_summary = apply_geometry_manifest(
        boxes_by_class,
        geometry_manifest_path,
        args.geometry_manifest_mode,
    )
    backgrounds = (
        load_patch_backgrounds(background_root, args.background_split)
        if background_root is not None
        else cashsnap_empty_backgrounds
    )
    assets_by_class = load_assets(manifest_path, status, sides, args.asset_quality_policy)
    if args.min_class_geometry_samples < 1:
        raise SystemExit("--min-class-geometry-samples must be > 0")
    if args.foreground_style_min_class_samples < 1:
        raise SystemExit("--foreground-style-min-class-samples must be > 0")
    if args.box_scale <= 0:
        raise SystemExit("--box-scale must be > 0")
    if args.box_scale_jitter < 0 or args.box_scale_jitter >= 1:
        raise SystemExit("--box-scale-jitter must be >= 0 and < 1")
    if args.min_render_short_px < 0:
        raise SystemExit("--min-render-short-px must be >= 0")
    if args.min_render_width_px < 0 or args.min_render_height_px < 0 or args.min_render_area_px < 0:
        raise SystemExit("--min-render-width/height/area-px must be >= 0")
    foreground_styles_by_class: dict[str, list[ForegroundStyle]] | None = None
    foreground_sharpness_quantiles: dict[str, float] = {}
    if args.foreground_style_policy == "real_crop_stats":
        style_rng = random.Random(args.seed + 910_003)
        foreground_styles_by_class = load_foreground_styles(
            boxes_by_class,
            style_rng,
            args.foreground_style_max_class_samples,
        )
        foreground_sharpness_quantiles = foreground_style_quantiles(foreground_styles_by_class)
    canvas_size: tuple[int, int] | None = None
    if args.canvas_size.strip():
        parts = [int(part.strip()) for part in args.canvas_size.split(",") if part.strip()]
        if len(parts) != 2 or min(parts) <= 0:
            raise SystemExit("--canvas-size must be WIDTH,HEIGHT or empty")
        canvas_size = (parts[0], parts[1])

    target_count = args.per_class * len(CLASS_NAMES)
    class_counts: Counter[str] = Counter()
    background_counts: Counter[str] = Counter()
    geometry_counts: Counter[str] = Counter()
    asset_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    attempts = 0
    while len(records) < target_count:
        attempts += 1
        if attempts > target_count * 20:
            raise SystemExit("Too many failed transplant attempts; check geometry/asset/background inputs.")
        class_name = choose_class(rng, class_counts)
        background = rng.choice(backgrounds)
        asset = rng.choice(assets_by_class[class_name])
        rendered = render_one(
            background,
            asset,
            boxes_by_class,
            foreground_styles_by_class,
            foreground_sharpness_quantiles,
            rng,
            args.min_class_geometry_samples,
            args.foreground_style_min_class_samples,
            canvas_size,
            args.composite_policy,
            args.shadow_policy,
            args.box_scale,
            args.box_scale_jitter,
            args.pose_policy,
            args.min_render_short_px,
            args.min_render_width_px,
            args.min_render_height_px,
            args.min_render_area_px,
        )
        if rendered is None:
            continue
        image, label, metadata = rendered
        stem = f"cashsnap_target_anchor_{len(records):06d}_{class_name.lower()}"
        image_path = out_root / "images" / "train" / f"{stem}.jpg"
        label_path = out_root / "labels" / "train" / f"{stem}.txt"
        image.save(image_path, quality=rng.randint(72, 92), subsampling=rng.choice([0, 1, 2]))
        label_path.write_text(label + "\n", encoding="utf-8")
        record = {
            "image": image_path.relative_to(out_root).as_posix(),
            "label": label_path.relative_to(out_root).as_posix(),
            "split": "train",
            **metadata,
        }
        records.append(record)
        class_counts[class_name] += 1
        background_counts[repo_rel(background.image_path)] += 1
        geometry_counts[record["geometry_source"]] += 1
        asset_counts[record["asset"]] += 1
        if len(records) % 250 == 0:
            print(f"generated {len(records)}/{target_count} images", flush=True)

    metadata_path = out_root / "metadata" / "train.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary: dict[str, Any] = {
        "schema": "cashsnap_target_anchor_transplant_summary_v1",
        "seed": args.seed,
        "per_class": args.per_class,
        "images": len(records),
        "class_counts": dict(sorted(class_counts.items())),
        "background_source_images": len(backgrounds),
        "background_root": repo_rel(background_root) if background_root is not None else repo_rel(cashsnap_root),
        "background_split": args.background_split if background_root is not None else "cashsnap_empty_train",
        "canvas_size": list(canvas_size) if canvas_size is not None else None,
        "unique_backgrounds_used": len(background_counts),
        "real_geometry_samples_by_class": {
            class_name: len(boxes_by_class[class_name]) for class_name in CLASS_NAMES
        },
        "geometry_manifest": geometry_manifest_summary,
        "unique_geometry_sources_used": len(geometry_counts),
        "assets_by_class": {class_name: len(assets_by_class[class_name]) for class_name in CLASS_NAMES},
        "unique_assets_used": len(asset_counts),
        "cashsnap_root": repo_rel(cashsnap_root),
        "asset_manifest": repo_rel(manifest_path),
        "asset_quality_policy": args.asset_quality_policy,
        "composite_policy": args.composite_policy,
        "shadow_policy": args.shadow_policy,
        "box_scale": args.box_scale,
        "box_scale_jitter": args.box_scale_jitter,
        "pose_policy": args.pose_policy,
        "min_render_short_px": args.min_render_short_px,
        "min_render_width_px": args.min_render_width_px,
        "min_render_height_px": args.min_render_height_px,
        "min_render_area_px": args.min_render_area_px,
        "foreground_style_policy": args.foreground_style_policy,
        "foreground_style_min_class_samples": args.foreground_style_min_class_samples,
        "foreground_style_max_class_samples": args.foreground_style_max_class_samples,
        "foreground_style_samples_by_class": (
            {class_name: len(foreground_styles_by_class[class_name]) for class_name in CLASS_NAMES}
            if foreground_styles_by_class is not None
            else {}
        ),
        "foreground_style_sharpness_quantiles": foreground_sharpness_quantiles,
        "metadata": repo_rel(metadata_path),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_data_yaml(out_root)
    write_realval_config(out_config, out_root, summary)
    build_preview(out_root, records, args.preview_count)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {repo_rel(out_root)}")
    print(f"wrote {repo_rel(out_config)}")


if __name__ == "__main__":
    main()
