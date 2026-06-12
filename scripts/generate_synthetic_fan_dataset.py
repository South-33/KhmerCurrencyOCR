"""Generate synthetic fanned/occluded KHR banknote YOLO data.

The generator starts from clean reference banknote images where the note is the
main object, layers several notes into fanned stacks, and labels only the
visible region of each note after occlusion.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

try:
    import cv2
except ImportError:  # pragma: no cover - optional OBB export dependency
    cv2 = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "asset_candidates" / "numista_current_cutout_bank_v1"
DEFAULT_OUT = ROOT / "data" / "synthetic" / "khr_fan_v1"

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
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
TARGET_KHR = {
    "500": "KHR_500",
    "1000": "KHR_1000",
    "2000": "KHR_2000",
    "5000": "KHR_5000",
    "10000": "KHR_10000",
    "20000": "KHR_20000",
    "50000": "KHR_50000",
}


@dataclass(frozen=True)
class NoteRef:
    path: Path
    class_name: str


@dataclass(frozen=True)
class BackgroundRef:
    path: Path


@dataclass(frozen=True)
class PlacedNote:
    class_name: str
    instance_id: int
    source_path: str
    placed_pixels: int
    layout_mode: str


def metadata_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def evidence_tier(
    visible_area_frac: float,
    visibility_ratio: float,
    unknown_visible_area_frac: float,
    unknown_visible_ratio: float,
) -> str:
    if visible_area_frac < unknown_visible_area_frac or visibility_ratio < unknown_visible_ratio:
        return "banknote_unknown"
    return "identifiable"


def parse_note_class(path: Path) -> str | None:
    for part in path.parts:
        if part in CLASS_TO_ID:
            return part
    for token in re.findall(r"\d+", path.stem):
        if token in TARGET_KHR:
            return TARGET_KHR[token]
    return None


def load_refs(sources: list[Path], allowed_classes: set[str] | None = None) -> list[NoteRef]:
    refs: list[NoteRef] = []
    for source in sources:
        for path in sorted(source.rglob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            lower_parts = {part.lower() for part in path.parts}
            if "masks" in lower_parts or path.stem.lower().endswith("_mask"):
                continue
            class_name = parse_note_class(path)
            if class_name and (allowed_classes is None or class_name in allowed_classes) and looks_like_whole_note(path):
                refs.append(NoteRef(path=path, class_name=class_name))
    if not refs:
        raise SystemExit(f"No target KHR reference images found in {sources}")
    return refs


def load_background_refs(sources: list[Path]) -> list[BackgroundRef]:
    refs: list[BackgroundRef] = []
    skip_dirs = {"labels", "masks", "crops", "metadata"}
    skip_stems = {"contact_sheet", "suspect_contact"}
    for source in sources:
        for path in sorted(source.rglob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            if path.stem.lower() in skip_stems:
                continue
            lower_parts = {part.lower() for part in path.parts}
            if lower_parts & skip_dirs:
                continue
            refs.append(BackgroundRef(path=path))
    return refs


def looks_like_whole_note(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            width, height = image.size
    except OSError:
        return False
    if width < 80 or height < 30:
        return False
    aspect = max(width, height) / max(1, min(width, height))
    # KHR/USD notes are long rectangles. Numista sometimes includes square-ish
    # security-detail closeups; those are useful references, but bad synthetic
    # source assets.
    return 1.55 <= aspect <= 3.35


def specimen_score(path: Path) -> float:
    if "specimen" in path.name.lower():
        return 1.0
    # Color alone is not a reliable specimen signal: valid KHR 500 scans are
    # pink/red, while some NBC specimen stamps are not separable by a simple
    # red-pixel threshold. Use audit_cutout_bank.py for visual QA instead.
    return 0.0


def looks_like_specimen(path: Path) -> bool:
    return specimen_score(path) > 0.018


def filter_specimen_refs(refs: list[NoteRef]) -> list[NoteRef]:
    by_class: dict[str, list[NoteRef]] = {}
    for ref in refs:
        by_class.setdefault(ref.class_name, []).append(ref)
    filtered: list[NoteRef] = []
    for class_name, class_refs in by_class.items():
        clean = [ref for ref in class_refs if not looks_like_specimen(ref.path)]
        if clean:
            filtered.extend(clean)
        else:
            filtered.append(min(class_refs, key=lambda ref: specimen_score(ref.path)))
    if not filtered:
        raise SystemExit("Specimen filtering removed every reference image.")
    return filtered


def trim_flat_border(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb).astype(np.int16)
    corners = np.array(
        [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]], dtype=np.int16
    )
    bg = np.median(corners, axis=0)
    dist = np.abs(arr - bg).sum(axis=2)
    mask = dist > 28
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return image.convert("RGBA")
    pad = 3
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(rgb.width, int(xs.max()) + pad + 1)
    y2 = min(rgb.height, int(ys.max()) + pad + 1)
    return rgb.crop((x1, y1, x2, y2)).convert("RGBA")


def note_alpha(image: Image.Image) -> Image.Image:
    existing = image.convert("RGBA")
    alpha = existing.getchannel("A")
    if alpha.getextrema()[0] < 250:
        arr = np.asarray(alpha)
        ys, xs = np.where(arr > 16)
        if len(xs) == 0:
            return existing
        pad = 3
        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(existing.width, int(xs.max()) + pad + 1)
        y2 = min(existing.height, int(ys.max()) + pad + 1)
        return existing.crop((x1, y1, x2, y2))

    rgba = trim_flat_border(image)
    arr = np.asarray(rgba.convert("RGB")).astype(np.int16)
    corners = np.array(
        [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]], dtype=np.int16
    )
    bg = np.median(corners, axis=0)
    dist = np.abs(arr - bg).sum(axis=2)
    alpha = np.where(dist > 22, 255, 0).astype(np.uint8)
    alpha = Image.fromarray(alpha, "L").filter(ImageFilter.MaxFilter(3))
    rgba.putalpha(alpha)
    return rgba


def jitter_note(image: Image.Image, rng: random.Random) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.25))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.85, 1.2))
    image = suppress_red_stamp(image)
    if rng.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
    return image


def perspective_coeffs(
    output_points: list[tuple[float, float]],
    input_points: list[tuple[float, float]],
) -> list[float]:
    matrix = []
    vector = []
    for (x_out, y_out), (x_in, y_in) in zip(output_points, input_points):
        matrix.append([x_out, y_out, 1, 0, 0, 0, -x_in * x_out, -x_in * y_out])
        matrix.append([0, 0, 0, x_out, y_out, 1, -y_in * x_out, -y_in * y_out])
        vector.extend([x_in, y_in])
    coeffs = np.linalg.lstsq(np.asarray(matrix), np.asarray(vector), rcond=None)[0]
    return [float(value) for value in coeffs]


def perspective_warp_note(image: Image.Image, rng: random.Random, probability: float) -> Image.Image:
    if probability <= 0 or rng.random() > probability:
        return image
    rgba = image.convert("RGBA")
    w, h = rgba.size
    if w < 32 or h < 16:
        return rgba
    pad = max(6, int(max(w, h) * 0.08))
    out_w = w + pad * 2
    out_h = h + pad * 2
    strength_x = min(w * rng.uniform(0.025, 0.10), pad * 0.95)
    strength_y = min(h * rng.uniform(0.035, 0.16), pad * 0.95)
    dest = [
        (pad + rng.uniform(-strength_x, strength_x), pad + rng.uniform(-strength_y, strength_y)),
        (pad + w + rng.uniform(-strength_x, strength_x), pad + rng.uniform(-strength_y, strength_y)),
        (pad + w + rng.uniform(-strength_x, strength_x), pad + h + rng.uniform(-strength_y, strength_y)),
        (pad + rng.uniform(-strength_x, strength_x), pad + h + rng.uniform(-strength_y, strength_y)),
    ]
    source = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    coeffs = perspective_coeffs(dest, source)
    warped = rgba.transform(
        (out_w, out_h),
        Image.Transform.PERSPECTIVE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )
    alpha = np.asarray(warped.getchannel("A"))
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return rgba
    return warped.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def crop_note_strip(
    image: Image.Image,
    rng: random.Random,
    min_frac: float = 0.16,
    max_frac: float = 0.38,
) -> Image.Image:
    """Keep one vertical slice of a note for extreme partial-visibility scenes."""
    if image.width < 60:
        return image
    strip_w = int(image.width * rng.uniform(min_frac, max_frac))
    strip_w = max(24, min(strip_w, image.width))
    if rng.random() < 0.5:
        x1 = rng.choice([0, image.width - strip_w])
    else:
        x1 = rng.randint(0, image.width - strip_w)
    y_crop = int(image.height * rng.uniform(0.00, 0.10))
    y2 = image.height - int(image.height * rng.uniform(0.00, 0.08))
    return image.crop((x1, y_crop, x1 + strip_w, max(y_crop + 8, y2)))


def suppress_red_stamp(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba).copy()
    rgb = arr[:, :, :3].astype(np.int16)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    mask = (red > 125) & ((red - green) > 45) & ((red - blue) > 45) & (arr[:, :, 3] > 20)
    if not mask.any():
        return rgba
    gray = np.round(rgb.mean(axis=2)).astype(np.uint8)
    paper = np.stack(
        [
            np.clip(gray + 16, 0, 255),
            np.clip(gray + 10, 0, 255),
            np.clip(gray + 6, 0, 255),
        ],
        axis=2,
    ).astype(np.uint8)
    arr[:, :, :3][mask] = paper[mask]
    return Image.fromarray(arr, "RGBA")


def make_procedural_background(size: int, rng: random.Random) -> Image.Image:
    base = np.zeros((size, size, 3), dtype=np.uint8)
    color = np.array(
        [
            rng.randint(90, 190),
            rng.randint(85, 180),
            rng.randint(70, 165),
        ],
        dtype=np.uint8,
    )
    base[:, :] = color
    noise = rng.normalvariate(0, 1)
    del noise
    grain = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, 9, base.shape)
    arr = np.clip(base.astype(np.int16) + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(0.4))


def make_image_background(path: Path, size: int, rng: random.Random) -> Image.Image:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    bg = ImageOps.fit(
        rgb,
        (size, size),
        method=Image.Resampling.BICUBIC,
        centering=(rng.random(), rng.random()),
    )
    if rng.random() < 0.55:
        bg = ImageEnhance.Brightness(bg).enhance(rng.uniform(0.82, 1.18))
    if rng.random() < 0.55:
        bg = ImageEnhance.Contrast(bg).enhance(rng.uniform(0.82, 1.22))
    if rng.random() < 0.35:
        bg = ImageEnhance.Color(bg).enhance(rng.uniform(0.78, 1.22))
    if rng.random() < 0.18:
        bg = bg.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.65)))
    return bg


def make_background(
    size: int,
    rng: random.Random,
    backgrounds: list[BackgroundRef] | None = None,
) -> tuple[Image.Image, str]:
    if backgrounds and rng.random() < 0.9:
        for _ in range(3):
            ref = rng.choice(backgrounds)
            try:
                return make_image_background(ref.path, size, rng), metadata_path(ref.path)
            except OSError:
                continue
    return make_procedural_background(size, rng), "procedural"


def add_hand_occluders(
    canvas: Image.Image,
    id_mask: np.ndarray,
    rng: random.Random,
    probability: float,
    grip_center: tuple[int, int] | None = None,
) -> dict[str, object]:
    base_info: dict[str, object] = {
        "hand_occluder_applied": False,
        "hand_occluder_count": 0,
        "hand_occluder_pixels": 0,
        "hand_occluder_area_frac": 0.0,
        "hand_occluded_note_pixels": 0,
        "hand_occluded_note_area_frac": 0.0,
        "hand_grip_aligned": grip_center is not None,
    }
    if probability <= 0 or rng.random() > probability:
        return base_info
    occ = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    occ_mask = Image.new("L", canvas.size, 0)
    skin_palettes = [
        (226, 174, 137),
        (198, 132, 91),
        (154, 96, 68),
        (113, 73, 55),
    ]
    base_skin = tuple(max(35, min(245, channel + rng.randint(-12, 12))) for channel in rng.choice(skin_palettes))
    count = rng.randint(1, 3 if grip_center else 4)
    for index in range(count):
        if grip_center is None:
            x = rng.randint(int(canvas.width * 0.15), int(canvas.width * 0.85))
            y = rng.randint(int(canvas.height * 0.35), int(canvas.height * 0.92))
            angle = rng.uniform(-25, 25)
        else:
            grip_x, grip_y = grip_center
            spacing = canvas.width * rng.uniform(0.035, 0.065)
            x = int(rng.gauss(grip_x + (index - (count - 1) / 2) * spacing, canvas.width * 0.025))
            y = int(rng.gauss(grip_y - canvas.height * 0.06, canvas.height * 0.032))
            x = max(int(canvas.width * 0.08), min(int(canvas.width * 0.92), x))
            y = max(int(canvas.height * 0.48), min(int(canvas.height * 0.98), y))
            angle = rng.uniform(-10, 10) + (index - (count - 1) / 2) * rng.uniform(5, 12)
        w = rng.randint(26, 44)
        h = rng.randint(115, 190)
        finger = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(finger)
        alpha = rng.randint(222, 244)
        fill = tuple(max(30, min(245, channel + rng.randint(-8, 8))) for channel in base_skin) + (alpha,)
        draw.rounded_rectangle((2, 2, w - 3, h - 3), radius=w // 2, fill=fill)
        highlight = tuple(max(40, min(255, channel + 24)) for channel in fill[:3]) + (rng.randint(35, 70),)
        draw.rounded_rectangle((int(w * 0.30), int(h * 0.10), int(w * 0.48), int(h * 0.82)), radius=max(2, w // 10), fill=highlight)
        if rng.random() < 0.72:
            nail = tuple(max(70, min(255, channel + 32)) for channel in fill[:3]) + (rng.randint(95, 145),)
            draw.ellipse((int(w * 0.22), int(h * 0.03), int(w * 0.78), int(h * 0.20)), fill=nail)
        finger = finger.filter(ImageFilter.GaussianBlur(rng.uniform(0.35, 0.9)))
        finger = finger.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        px = x - finger.width // 2
        py = y - finger.height // 2
        occ.alpha_composite(finger, (px, py))
        alpha = finger.getchannel("A")
        occ_mask.paste(alpha, (px, py), alpha)
    canvas.alpha_composite(occ)
    occ_arr = np.asarray(occ_mask) > 24
    image_pixels = max(1, occ_arr.size)
    occluded_note_pixels = int((occ_arr & (id_mask > 0)).sum())
    occluder_pixels = int(occ_arr.sum())
    id_mask[occ_arr] = 0
    return {
        **base_info,
        "hand_occluder_applied": True,
        "hand_occluder_count": count,
        "hand_occluder_pixels": occluder_pixels,
        "hand_occluder_area_frac": occluder_pixels / image_pixels,
        "hand_occluded_note_pixels": occluded_note_pixels,
        "hand_occluded_note_area_frac": occluded_note_pixels / image_pixels,
    }


def add_note_shadow(
    canvas: Image.Image,
    note: Image.Image,
    x: int,
    y: int,
    rng: random.Random,
    probability: float,
) -> None:
    if probability <= 0 or rng.random() > probability:
        return
    offset_x = rng.choice([-1, 1]) * rng.randint(3, 16)
    offset_y = rng.randint(5, 22)
    blur_radius = rng.uniform(2.2, 7.5)
    opacity = rng.uniform(0.10, 0.28)
    shadow_alpha = note.getchannel("A").filter(ImageFilter.GaussianBlur(blur_radius))
    shadow_alpha = shadow_alpha.point(lambda value: int(value * opacity))
    shadow = Image.new("RGBA", note.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)
    canvas.alpha_composite(shadow, (x + offset_x, y + offset_y))


def corner_fill_rgba(image: Image.Image) -> tuple[int, int, int, int]:
    arr = np.asarray(image.convert("RGBA"))
    samples = np.concatenate(
        [
            arr[:16, :16].reshape(-1, 4),
            arr[:16, -16:].reshape(-1, 4),
            arr[-16:, :16].reshape(-1, 4),
            arr[-16:, -16:].reshape(-1, 4),
        ],
        axis=0,
    )
    fill = np.median(samples, axis=0).astype(np.uint8)
    return tuple(int(v) for v in fill)


def warp_scene_affine(
    canvas: Image.Image,
    id_mask: np.ndarray,
    rng: random.Random,
) -> tuple[Image.Image, np.ndarray]:
    if cv2 is None:
        return canvas, id_mask
    width, height = canvas.size
    scale_x = rng.uniform(0.90, 1.12)
    scale_y = rng.uniform(0.90, 1.12)
    shear = rng.uniform(-0.045, 0.045)
    tx = rng.uniform(-0.045, 0.045) * width
    ty = rng.uniform(-0.045, 0.045) * height
    matrix = np.array(
        [
            [scale_x, shear, (1 - scale_x) * width * 0.5 + tx],
            [rng.uniform(-0.025, 0.025), scale_y, (1 - scale_y) * height * 0.5 + ty],
        ],
        dtype=np.float32,
    )
    image_arr = np.asarray(canvas.convert("RGBA"))
    fill = corner_fill_rgba(canvas)
    warped_image = cv2.warpAffine(
        image_arr,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    warped_mask = cv2.warpAffine(
        id_mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return Image.fromarray(warped_image, "RGBA"), warped_mask.astype(id_mask.dtype, copy=False)


def distort_scene_lens(
    canvas: Image.Image,
    id_mask: np.ndarray,
    rng: random.Random,
) -> tuple[Image.Image, np.ndarray]:
    if cv2 is None:
        return canvas, id_mask
    width, height = canvas.size
    yy, xx = np.indices((height, width), dtype=np.float32)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    radius = max(width, height) * 0.5
    x = (xx - cx) / radius
    y = (yy - cy) / radius
    r2 = x * x + y * y
    k1 = rng.uniform(-0.055, 0.055)
    k2 = rng.uniform(-0.018, 0.018)
    scale = 1.0 + k1 * r2 + k2 * r2 * r2
    map_x = np.clip(cx + x * scale * radius, 0, width - 1).astype(np.float32)
    map_y = np.clip(cy + y * scale * radius, 0, height - 1).astype(np.float32)
    image_arr = np.asarray(canvas.convert("RGBA"))
    warped_image = cv2.remap(image_arr, map_x, map_y, interpolation=cv2.INTER_CUBIC)
    warped_mask = cv2.remap(id_mask, map_x, map_y, interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(warped_image, "RGBA"), warped_mask.astype(id_mask.dtype, copy=False)


def apply_scene_camera_geometry(
    canvas: Image.Image,
    id_mask: np.ndarray,
    rng: random.Random,
    camera_geom_probability: float,
    lens_distort_probability: float,
) -> tuple[Image.Image, np.ndarray, dict[str, object]]:
    info: dict[str, object] = {
        "camera_geom_applied": False,
        "lens_distort_applied": False,
        "camera_geometry_backend": "cv2" if cv2 is not None else "unavailable",
    }
    if cv2 is None:
        return canvas, id_mask, info
    if camera_geom_probability > 0 and rng.random() < camera_geom_probability:
        canvas, id_mask = warp_scene_affine(canvas, id_mask, rng)
        info["camera_geom_applied"] = True
    if lens_distort_probability > 0 and rng.random() < lens_distort_probability:
        canvas, id_mask = distort_scene_lens(canvas, id_mask, rng)
        info["lens_distort_applied"] = True
    return canvas, id_mask, info


def add_glare(rgb: Image.Image, rng: random.Random) -> Image.Image:
    overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = rgb.size
    if rng.random() < 0.55:
        y = rng.randint(int(height * 0.05), int(height * 0.78))
        band_h = rng.randint(max(8, height // 32), max(18, height // 9))
        skew = rng.randint(-width // 5, width // 5)
        alpha = rng.randint(18, 58)
        polygon = [
            (-width // 8, y),
            (width + width // 8, y + skew),
            (width + width // 8, y + skew + band_h),
            (-width // 8, y + band_h),
        ]
        draw.polygon(polygon, fill=(255, 255, 245, alpha))
    else:
        cx = rng.randint(0, width)
        cy = rng.randint(0, height)
        rx = rng.randint(max(18, width // 12), max(24, width // 3))
        ry = rng.randint(max(12, height // 18), max(18, height // 4))
        alpha = rng.randint(16, 52)
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(255, 255, 245, alpha))
    return Image.alpha_composite(rgb.convert("RGBA"), overlay).convert("RGB")


def jpeg_roundtrip(rgb: Image.Image, rng: random.Random, min_quality: int, max_quality: int) -> Image.Image:
    quality = rng.randint(min_quality, max_quality)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_phone_postprocess(
    rgb: Image.Image,
    rng: random.Random,
    probability: float,
    jpeg_quality_min: int,
    jpeg_quality_max: int,
) -> Image.Image:
    if probability <= 0 or rng.random() > probability:
        return rgb
    output = rgb
    if rng.random() < 0.75:
        arr = np.asarray(output).astype(np.float32)
        channel_gain = np.array(
            [
                rng.uniform(0.92, 1.10),
                rng.uniform(0.94, 1.06),
                rng.uniform(0.88, 1.08),
            ],
            dtype=np.float32,
        )
        arr = np.clip(arr * channel_gain, 0, 255)
        output = Image.fromarray(arr.astype(np.uint8), "RGB")
    if rng.random() < 0.65:
        arr = np.asarray(output).astype(np.int16)
        noise_sigma = rng.uniform(5, 18)
        grain = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, noise_sigma, arr.shape)
        output = Image.fromarray(np.clip(arr + grain, 0, 255).astype(np.uint8), "RGB")
    if rng.random() < 0.45:
        output = add_glare(output, rng)
    if rng.random() < 0.35:
        output = output.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.35, 1.25)))
    if rng.random() < 0.45:
        output = ImageEnhance.Sharpness(output).enhance(rng.uniform(0.75, 1.55))
    if rng.random() < 0.70:
        output = jpeg_roundtrip(output, rng, jpeg_quality_min, jpeg_quality_max)
    return output


def paste_note(
    canvas: Image.Image,
    id_mask: np.ndarray,
    note: Image.Image,
    instance_id: int,
    x: int,
    y: int,
) -> int:
    alpha = np.asarray(note.getchannel("A")) > 16
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(canvas.width, x + note.width)
    y2 = min(canvas.height, y + note.height)
    if x1 >= x2 or y1 >= y2:
        return 0
    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    canvas.alpha_composite(note, (x, y))
    visible = alpha[sy1:sy2, sx1:sx2]
    id_mask[y1:y2, x1:x2][visible] = instance_id
    return int(visible.sum())


def visible_box(id_mask: np.ndarray, instance_id: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(id_mask == instance_id)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def visible_mask(id_mask: np.ndarray, instance_id: int) -> np.ndarray:
    return id_mask == instance_id


def visible_obb_line(mask: np.ndarray, class_name: str, image_size: int) -> str | None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for --label-format obb")
    ys, xs = np.where(mask)
    if len(xs) < 4:
        return None
    points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box[:, 0] = np.clip(box[:, 0] / image_size, 0.0, 1.0)
    box[:, 1] = np.clip(box[:, 1] / image_size, 0.0, 1.0)
    coords = " ".join(f"{value:.6f}" for value in box.reshape(-1))
    return f"{CLASS_TO_ID[class_name]} {coords}"


def prepared_note(ref: NoteRef, cache: dict[tuple[Path, str], Image.Image], source_alpha_policy: str) -> Image.Image:
    key = (ref.path, source_alpha_policy)
    if key not in cache:
        with Image.open(ref.path) as raw:
            source = ImageOps.exif_transpose(raw)
            if source_alpha_policy == "opaque":
                note = source.convert("RGBA")
                note.putalpha(255)
            else:
                note = note_alpha(source)
        cache[key] = note.copy()
    return cache[key].copy()


def make_scene(
    refs: list[NoteRef],
    image_size: int,
    rng: random.Random,
    note_cache: dict[tuple[Path, str], Image.Image] | None = None,
    min_notes: int = 4,
    max_notes: int = 12,
    layout_modes: list[str] | None = None,
    hand_probability: float = 0.25,
    note_shadow_probability: float = 0.0,
    label_format: str = "detect",
    save_visible_masks: bool = False,
    strip_min_frac: float = 0.16,
    strip_max_frac: float = 0.38,
    thin_strip_min_frac: float = 0.07,
    thin_strip_max_frac: float = 0.20,
    unknown_visible_area_frac: float = 0.012,
    unknown_visible_ratio: float = 0.18,
    drop_unknown_denom_labels: bool = False,
    balance_classes: bool = False,
    backgrounds: list[BackgroundRef] | None = None,
    source_alpha_policy: str = "auto",
    perspective_probability: float = 0.35,
    camera_geom_probability: float = 0.0,
    lens_distort_probability: float = 0.0,
    scene_aug_probability: float = 0.0,
    jpeg_quality_min: int = 62,
    jpeg_quality_max: int = 92,
) -> tuple[Image.Image, list[str], list[tuple[str, np.ndarray]], list[dict[str, object]], dict[str, object]]:
    background, background_path = make_background(image_size, rng, backgrounds)
    canvas = background.convert("RGBA")
    scene_info: dict[str, object] = {
        "background": background_path,
        "perspective_probability": perspective_probability,
        "camera_geom_probability": camera_geom_probability,
        "lens_distort_probability": lens_distort_probability,
        "scene_aug_probability": scene_aug_probability,
    }
    id_mask = np.zeros((image_size, image_size), dtype=np.uint16)
    placed_notes: list[PlacedNote] = []
    note_count = rng.randint(min_notes, max_notes)
    available_modes = layout_modes or [
        "radial_slice",
        "strip_fan",
        "thin_radial_slice",
        "tight_fan",
        "fan",
        "crossed",
        "scattered",
        "row",
    ]
    default_weights = {
        "radial_slice": 0.28,
        "strip_fan": 0.24,
        "thin_radial_slice": 0.18,
        "tight_fan": 0.22,
        "fan": 0.14,
        "crossed": 0.07,
        "scattered": 0.04,
        "row": 0.01,
    }
    layout_mode = rng.choices(
        available_modes,
        weights=[default_weights.get(mode, 0.05) for mode in available_modes],
        k=1,
    )[0]
    if layout_mode == "radial_slice":
        note_count = rng.randint(max(min_notes, 12), max(max_notes, 22))
    elif layout_mode == "strip_fan":
        note_count = rng.randint(max(min_notes, 10), max(max_notes, 20))
    elif layout_mode == "thin_radial_slice":
        note_count = rng.randint(max(min_notes, 16), max(max_notes, 28))
    elif layout_mode == "tight_fan":
        note_count = rng.randint(max(min_notes, 9), max(max_notes, 18))
    center_x = rng.randint(int(image_size * 0.43), int(image_size * 0.57))
    pivot_y = rng.randint(int(image_size * 0.78), int(image_size * 0.96))
    refs_by_class: dict[str, list[NoteRef]] = {}
    if balance_classes:
        for ref in refs:
            refs_by_class.setdefault(ref.class_name, []).append(ref)
        balanced_classes = sorted(refs_by_class)

    for i in range(note_count):
        if balance_classes:
            ref = rng.choice(refs_by_class[rng.choice(balanced_classes)])
        else:
            ref = rng.choice(refs)
        if note_cache is not None:
            note = prepared_note(ref, note_cache, source_alpha_policy)
        else:
            with Image.open(ref.path) as raw:
                note = note_alpha(ImageOps.exif_transpose(raw))
        if layout_mode == "radial_slice":
            target_w = int(image_size * rng.uniform(0.72, 1.02))
        elif layout_mode == "strip_fan":
            target_w = int(image_size * rng.uniform(0.72, 0.96))
        elif layout_mode == "thin_radial_slice":
            target_w = int(image_size * rng.uniform(0.74, 1.04))
        elif layout_mode == "tight_fan":
            target_w = int(image_size * rng.uniform(0.64, 0.90))
        elif layout_mode == "fan":
            target_w = int(image_size * rng.uniform(0.56, 0.78))
        elif layout_mode == "row":
            target_w = int(image_size * rng.uniform(0.34, 0.48))
        else:
            target_w = int(image_size * rng.uniform(0.42, 0.68))
        scale = target_w / max(1, note.width)
        note = note.resize(
            (max(8, int(note.width * scale)), max(8, int(note.height * scale))),
            Image.Resampling.BICUBIC,
        )
        note = jitter_note(note, rng)
        if layout_mode == "strip_fan":
            note = crop_note_strip(note, rng, min_frac=strip_min_frac, max_frac=strip_max_frac)
        elif layout_mode == "thin_radial_slice":
            note = crop_note_strip(note, rng, min_frac=thin_strip_min_frac, max_frac=thin_strip_max_frac)
        note = perspective_warp_note(note, rng, perspective_probability)

        if layout_mode == "radial_slice":
            if note_count == 1:
                angle = rng.uniform(-16, 16)
            else:
                spread = rng.uniform(92, 142)
                angle = (-spread / 2) + (spread * i / (note_count - 1)) + rng.uniform(-3, 3)
        elif layout_mode == "strip_fan":
            if note_count == 1:
                angle = rng.uniform(-20, 20)
            else:
                spread = rng.uniform(78, 126)
                angle = (-spread / 2) + (spread * i / (note_count - 1)) + rng.uniform(-3, 3)
        elif layout_mode == "thin_radial_slice":
            if note_count == 1:
                angle = rng.uniform(-18, 18)
            else:
                spread = rng.uniform(58, 98)
                angle = (-spread / 2) + (spread * i / (note_count - 1)) + rng.uniform(-2, 2)
        elif layout_mode == "tight_fan":
            if note_count == 1:
                angle = rng.uniform(-20, 20)
            else:
                spread = rng.uniform(70, 118)
                angle = (-spread / 2) + (spread * i / (note_count - 1)) + rng.uniform(-4, 4)
        elif layout_mode == "fan":
            if note_count == 1:
                angle = rng.uniform(-20, 20)
            else:
                angle = -62 + (124 * i / (note_count - 1)) + rng.uniform(-7, 7)
        elif layout_mode == "crossed":
            angle = rng.choice([-1, 1]) * rng.uniform(18, 75)
            if rng.random() < 0.35:
                angle += 90
        elif layout_mode == "row":
            angle = rng.uniform(-12, 12)
        else:
            angle = rng.uniform(-85, 85)
        note = note.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

        if layout_mode == "radial_slice":
            offset = (i - (note_count - 1) / 2) * rng.uniform(5, 11)
            x = int(center_x + offset + rng.randint(-5, 5) - note.width / 2)
            y = int(pivot_y + rng.randint(-10, 8) - note.height)
            if i < note_count * 0.15 or i > note_count * 0.85:
                x += rng.choice([-1, 1]) * rng.randint(12, 38)
        elif layout_mode == "strip_fan":
            offset = (i - (note_count - 1) / 2) * rng.uniform(8, 17)
            x = int(center_x + offset + rng.randint(-8, 8) - note.width / 2)
            y = int(pivot_y + rng.randint(-16, 14) - note.height)
            if rng.random() < 0.35:
                y += rng.randint(-55, 35)
        elif layout_mode == "thin_radial_slice":
            offset = (i - (note_count - 1) / 2) * rng.uniform(3, 8)
            x = int(center_x + offset + rng.randint(-4, 4) - note.width / 2)
            y = int(pivot_y + rng.randint(-8, 8) - note.height)
            if i < note_count * 0.12 or i > note_count * 0.88:
                x += rng.choice([-1, 1]) * rng.randint(8, 26)
        elif layout_mode == "tight_fan":
            offset = (i - (note_count - 1) / 2) * rng.uniform(7, 15)
            x = int(center_x + offset + rng.randint(-9, 9) - note.width / 2)
            y = int(pivot_y + rng.randint(-12, 12) - note.height)
            if rng.random() < 0.45:
                y += rng.randint(-70, 25)
            if rng.random() < 0.28:
                x += rng.choice([-1, 1]) * rng.randint(45, 120)
        elif layout_mode == "fan":
            offset = (i - (note_count - 1) / 2) * rng.uniform(13, 24)
            x = int(center_x + offset + rng.randint(-14, 14) - note.width / 2)
            y = int(pivot_y + rng.randint(-18, 16) - note.height)
        elif layout_mode == "row":
            cols = min(note_count, rng.randint(3, 5))
            col = i % cols
            row = i // cols
            x = int((col + 0.25) * image_size / cols + rng.randint(-35, 35))
            y = int(image_size * (0.20 + 0.22 * row) + rng.randint(-35, 35))
            if rng.random() < 0.35:
                x += rng.choice([-1, 1]) * rng.randint(80, 180)
            if rng.random() < 0.35:
                y += rng.choice([-1, 1]) * rng.randint(60, 150)
        elif layout_mode == "crossed":
            x = int(center_x + rng.randint(-170, 170) - note.width / 2)
            y = int(image_size * rng.uniform(0.28, 0.68) + rng.randint(-80, 80) - note.height / 2)
        else:
            theta = math.radians(angle)
            spread = rng.uniform(70, 220)
            x = int(center_x + math.sin(theta) * spread + rng.randint(-90, 90) - note.width / 2)
            y = int(image_size * rng.uniform(0.25, 0.75) - math.cos(theta) * spread * 0.20 + rng.randint(-90, 90) - note.height / 2)

        instance_id = len(placed_notes) + 1
        add_note_shadow(canvas, note, x, y, rng, note_shadow_probability)
        placed_pixels = paste_note(canvas, id_mask, note, instance_id, x, y)
        placed_notes.append(
            PlacedNote(
                class_name=ref.class_name,
                instance_id=instance_id,
                source_path=metadata_path(ref.path),
                placed_pixels=placed_pixels,
                layout_mode=layout_mode,
            )
        )

    fan_layouts = {"radial_slice", "strip_fan", "thin_radial_slice", "tight_fan", "fan"}
    grip_center = (center_x, pivot_y) if layout_mode in fan_layouts else None
    scene_info.update(add_hand_occluders(canvas, id_mask, rng, hand_probability, grip_center=grip_center))
    canvas, id_mask, camera_info = apply_scene_camera_geometry(
        canvas,
        id_mask,
        rng,
        camera_geom_probability,
        lens_distort_probability,
    )
    scene_info.update(camera_info)

    label_lines: list[str] = []
    visible_masks: list[tuple[str, np.ndarray]] = []
    records: list[dict[str, object]] = []
    for placed in placed_notes:
        box = visible_box(id_mask, placed.instance_id)
        base_record: dict[str, object] = {
            "class_name": placed.class_name,
            "instance_id": placed.instance_id,
            "source_path": placed.source_path,
            "layout_mode": placed.layout_mode,
            "placed_pixels": placed.placed_pixels,
            "exported_label": False,
        }
        if not box:
            records.append(
                {
                    **base_record,
                    "visible_pixels": 0,
                    "visibility_ratio": 0.0,
                    "visible_area_frac": 0.0,
                    "evidence_tier": "ignore",
                    "drop_reason": "no_visible_pixels",
                }
            )
            continue
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        area = w * h
        mask = visible_mask(id_mask, placed.instance_id)
        visible_pixels = int(mask.sum())
        visible_area_frac = visible_pixels / (image_size * image_size)
        visibility_ratio = min(1.0, visible_pixels / max(1, placed.placed_pixels))
        tier = evidence_tier(
            visible_area_frac,
            visibility_ratio,
            unknown_visible_area_frac,
            unknown_visible_ratio,
        )
        record = {
            **base_record,
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_width": w,
            "bbox_height": h,
            "bbox_area_frac": area / (image_size * image_size),
            "visible_pixels": visible_pixels,
            "visible_area_frac": visible_area_frac,
            "visibility_ratio": visibility_ratio,
            "evidence_tier": tier,
        }
        if area < image_size * image_size * 0.004 or w < 18 or h < 18:
            record["evidence_tier"] = "ignore"
            record["drop_reason"] = "too_small"
            records.append(record)
            continue
        if drop_unknown_denom_labels and tier == "banknote_unknown":
            record["drop_reason"] = "banknote_unknown"
            records.append(record)
            continue
        if label_format == "obb":
            line = visible_obb_line(mask, placed.class_name, image_size)
            if line is None:
                record["evidence_tier"] = "ignore"
                record["drop_reason"] = "obb_failed"
                records.append(record)
                continue
            label_lines.append(line)
        else:
            cx = (x1 + x2) / 2 / image_size
            cy = (y1 + y2) / 2 / image_size
            bw = w / image_size
            bh = h / image_size
            label_lines.append(f"{CLASS_TO_ID[placed.class_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        record["exported_label"] = True
        if save_visible_masks:
            visible_masks.append((placed.class_name, mask.copy()))
        records.append(record)

    rgb = canvas.convert("RGB")
    if rng.random() < 0.45:
        arr = np.asarray(rgb).astype(np.int16)
        grain = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, rng.uniform(3, 12), arr.shape)
        rgb = Image.fromarray(np.clip(arr + grain, 0, 255).astype(np.uint8), "RGB")
    if rng.random() < 0.25:
        rgb = ImageOps.autocontrast(rgb, cutoff=rng.uniform(0, 1.5))
    rgb = apply_phone_postprocess(rgb, rng, scene_aug_probability, jpeg_quality_min, jpeg_quality_max)
    return rgb, label_lines, visible_masks, records, scene_info


def write_yaml(out: Path) -> None:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    dataset_path = out.resolve().as_posix()
    (out / "data.yaml").write_text(
        f"path: {dataset_path}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n{names}\n",
        encoding="utf-8",
    )


def padded_crop_box(box: list[int], image_size: int, pad_frac: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad = int(max(x2 - x1, y2 - y1) * pad_frac)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(image_size, x2 + pad),
        min(image_size, y2 + pad),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, nargs="+", default=[DEFAULT_SOURCE])
    parser.add_argument(
        "--background-dir",
        type=Path,
        nargs="+",
        default=[],
        help="Optional directories of note-free real background images/patches.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--clean", action="store_true", help="Delete an existing output directory before generating.")
    parser.add_argument("--allow-specimen", action="store_true", help="Allow reference images with SPECIMEN in the filename.")
    parser.add_argument("--classes", default="", help="Optional comma-separated canonical classes to generate.")
    parser.add_argument(
        "--label-format",
        choices=["detect", "obb"],
        default="detect",
        help="Write axis-aligned YOLO detect labels or YOLO OBB corner labels.",
    )
    parser.add_argument("--save-visible-masks", action="store_true", help="Save per-visible-instance binary masks.")
    parser.add_argument("--save-visible-crops", action="store_true", help="Save padded visible-instance crops for fragment verifier training.")
    parser.add_argument("--crop-pad-frac", type=float, default=0.12, help="Padding fraction for --save-visible-crops.")
    parser.add_argument("--crop-include-unknown", action="store_true", help="Also save banknote_unknown crops for unknown/rejection calibration.")
    parser.add_argument("--no-metadata", action="store_true", help="Do not write per-image instance metadata JSONL files.")
    parser.add_argument(
        "--unknown-visible-area-frac",
        type=float,
        default=0.012,
        help="Visible image-area fraction below which metadata marks an instance as banknote_unknown.",
    )
    parser.add_argument(
        "--unknown-visible-ratio",
        type=float,
        default=0.18,
        help="Visible/placed pixel ratio below which metadata marks an instance as banknote_unknown.",
    )
    parser.add_argument(
        "--drop-unknown-denom-labels",
        action="store_true",
        help="Skip denomination labels for instances marked banknote_unknown while keeping metadata.",
    )
    parser.add_argument("--min-notes", type=int, default=4, help="Minimum notes per synthetic scene.")
    parser.add_argument("--max-notes", type=int, default=12, help="Maximum notes per synthetic scene.")
    parser.add_argument(
        "--layout-modes",
        default="",
        help="Optional comma-separated layout modes: radial_slice,strip_fan,thin_radial_slice,tight_fan,fan,crossed,scattered,row.",
    )
    parser.add_argument("--hand-prob", type=float, default=0.25, help="Probability of adding synthetic hand/finger occluders.")
    parser.add_argument("--strip-min-frac", type=float, default=0.16, help="Minimum source-note width kept for strip_fan crops.")
    parser.add_argument("--strip-max-frac", type=float, default=0.38, help="Maximum source-note width kept for strip_fan crops.")
    parser.add_argument("--thin-strip-min-frac", type=float, default=0.07, help="Minimum source-note width kept for thin_radial_slice crops.")
    parser.add_argument("--thin-strip-max-frac", type=float, default=0.20, help="Maximum source-note width kept for thin_radial_slice crops.")
    parser.add_argument("--note-shadow-prob", type=float, default=0.0, help="Probability that each upper note casts a soft contact shadow.")
    parser.add_argument(
        "--source-alpha-policy",
        choices=["auto", "opaque"],
        default="auto",
        help=(
            "auto extracts note alpha for clean cutouts/scans; opaque keeps each source crop as an intact rectangle, "
            "which is safer for real detector/crop-classifier tiles."
        ),
    )
    parser.add_argument("--balance-classes", action="store_true", help="Sample a class uniformly first, then sample a reference asset from that class.")
    parser.add_argument("--perspective-prob", type=float, default=0.35, help="Probability of applying a mild local perspective warp to each note.")
    parser.add_argument("--camera-geom-prob", type=float, default=0.0, help="Probability of applying mild whole-scene affine camera framing to the RGB image and ID mask before labels are exported.")
    parser.add_argument("--lens-distort-prob", type=float, default=0.0, help="Probability of applying mild whole-scene radial lens distortion to the RGB image and ID mask before labels are exported.")
    parser.add_argument("--scene-aug-prob", type=float, default=0.0, help="Probability of phone-style scene postprocessing: color cast, noise, glare, blur, sharpening, and JPEG.")
    parser.add_argument("--jpeg-quality-min", type=int, default=62, help="Minimum JPEG quality for --scene-aug-prob postprocessing.")
    parser.add_argument("--jpeg-quality-max", type=int, default=92, help="Maximum JPEG quality for --scene-aug-prob postprocessing.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    allowed_classes = {item.strip() for item in args.classes.split(",") if item.strip()} or None
    unknown = sorted((allowed_classes or set()) - set(CLASS_TO_ID))
    if unknown:
        raise SystemExit(f"Unknown classes in --classes: {unknown}")
    layout_modes = [item.strip() for item in args.layout_modes.split(",") if item.strip()] or None
    valid_layout_modes = {
        "radial_slice",
        "strip_fan",
        "thin_radial_slice",
        "tight_fan",
        "fan",
        "crossed",
        "scattered",
        "row",
    }
    unknown_layouts = sorted(set(layout_modes or []) - valid_layout_modes)
    if unknown_layouts:
        raise SystemExit(f"Unknown layout modes in --layout-modes: {unknown_layouts}")
    if args.label_format == "obb" and cv2 is None:
        raise SystemExit("--label-format obb requires opencv-python/cv2")
    if args.min_notes < 1 or args.max_notes < args.min_notes:
        raise SystemExit("--min-notes must be >= 1 and --max-notes must be >= --min-notes")
    if not (0 < args.strip_min_frac <= args.strip_max_frac <= 1):
        raise SystemExit("--strip-min-frac must be > 0 and <= --strip-max-frac <= 1")
    if not (0 < args.thin_strip_min_frac <= args.thin_strip_max_frac <= 1):
        raise SystemExit("--thin-strip-min-frac must be > 0 and <= --thin-strip-max-frac <= 1")
    if args.unknown_visible_area_frac < 0:
        raise SystemExit("--unknown-visible-area-frac must be >= 0")
    if not (0 <= args.unknown_visible_ratio <= 1):
        raise SystemExit("--unknown-visible-ratio must be between 0 and 1")
    if not (0 <= args.note_shadow_prob <= 1):
        raise SystemExit("--note-shadow-prob must be between 0 and 1")
    if args.crop_pad_frac < 0:
        raise SystemExit("--crop-pad-frac must be >= 0")
    if not (0 <= args.perspective_prob <= 1):
        raise SystemExit("--perspective-prob must be between 0 and 1")
    if not (0 <= args.camera_geom_prob <= 1):
        raise SystemExit("--camera-geom-prob must be between 0 and 1")
    if not (0 <= args.lens_distort_prob <= 1):
        raise SystemExit("--lens-distort-prob must be between 0 and 1")
    if not (0 <= args.scene_aug_prob <= 1):
        raise SystemExit("--scene-aug-prob must be between 0 and 1")
    if not (1 <= args.jpeg_quality_min <= args.jpeg_quality_max <= 100):
        raise SystemExit("--jpeg-quality-min/--jpeg-quality-max must satisfy 1 <= min <= max <= 100")
    refs = load_refs(args.source, allowed_classes)
    backgrounds = load_background_refs(args.background_dir) if args.background_dir else []
    if args.background_dir:
        print(f"background refs: {len(backgrounds)}")
        if not backgrounds:
            raise SystemExit(f"No background images found in {args.background_dir}")
    if not args.allow_specimen:
        before = len(refs)
        refs = filter_specimen_refs(refs)
        print(f"specimen filter: kept {len(refs)} of {before} refs")

    if args.clean and args.out.exists():
        resolved = args.out.resolve()
        allowed_root = (ROOT / "data" / "synthetic").resolve()
        if allowed_root not in resolved.parents and resolved != allowed_root:
            raise SystemExit(f"Refusing to clean outside {allowed_root}: {resolved}")
        shutil.rmtree(resolved)

    for split in ["train", "val", "test"]:
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    metadata_handles = {}
    if not args.no_metadata:
        metadata_dir = args.out / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_handles = {
            split: (metadata_dir / f"{split}.jsonl").open("w", encoding="utf-8")
            for split in ["train", "val", "test"]
        }

    counts = {"train": 0, "val": 0, "test": 0}
    boxes = {"train": 0, "val": 0, "test": 0}
    note_cache: dict[tuple[Path, str], Image.Image] = {}
    i = 0
    attempts = 0
    while i < args.count:
        attempts += 1
        image, labels, visible_masks, records, scene_info = make_scene(
            refs,
            args.image_size,
            rng,
            note_cache,
            min_notes=args.min_notes,
            max_notes=args.max_notes,
            layout_modes=layout_modes,
            hand_probability=args.hand_prob,
            note_shadow_probability=args.note_shadow_prob,
            label_format=args.label_format,
            save_visible_masks=args.save_visible_masks,
            strip_min_frac=args.strip_min_frac,
            strip_max_frac=args.strip_max_frac,
            thin_strip_min_frac=args.thin_strip_min_frac,
            thin_strip_max_frac=args.thin_strip_max_frac,
            unknown_visible_area_frac=args.unknown_visible_area_frac,
            unknown_visible_ratio=args.unknown_visible_ratio,
            drop_unknown_denom_labels=args.drop_unknown_denom_labels,
            balance_classes=args.balance_classes,
            backgrounds=backgrounds,
            source_alpha_policy=args.source_alpha_policy,
            perspective_probability=args.perspective_prob,
            camera_geom_probability=args.camera_geom_prob,
            lens_distort_probability=args.lens_distort_prob,
            scene_aug_probability=args.scene_aug_prob,
            jpeg_quality_min=args.jpeg_quality_min,
            jpeg_quality_max=args.jpeg_quality_max,
        )
        if not labels:
            if attempts > args.count * 5:
                raise SystemExit("Too many empty synthetic scenes; check reference images.")
            continue
        r = rng.random()
        if r < args.test_frac:
            split = "test"
        elif r < args.test_frac + args.val_frac:
            split = "val"
        else:
            split = "train"
        stem = f"khr_fan_{i:06d}"
        image_path = args.out / "images" / split / f"{stem}.jpg"
        label_path = args.out / "labels" / split / f"{stem}.txt"
        image.save(image_path, quality=88)
        label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
        if metadata_handles:
            metadata_handles[split].write(
                json.dumps(
                    {
                        "image": image_path.relative_to(args.out).as_posix(),
                        "label": label_path.relative_to(args.out).as_posix(),
                        "split": split,
                        "image_size": args.image_size,
                        "label_format": args.label_format,
                        **scene_info,
                        "instances": records,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        if args.save_visible_masks:
            mask_dir = args.out / "masks" / split
            mask_dir.mkdir(parents=True, exist_ok=True)
            for mask_index, (class_name, mask) in enumerate(visible_masks, start=1):
                mask_image = Image.fromarray((mask.astype(np.uint8) * 255), "L")
                mask_image.save(mask_dir / f"{stem}_{mask_index:02d}_{class_name}.png")
        if args.save_visible_crops:
            for crop_index, record in enumerate(records, start=1):
                tier = str(record.get("evidence_tier", ""))
                if tier == "identifiable" and record.get("exported_label"):
                    crop_class = str(record["class_name"])
                elif tier == "banknote_unknown" and args.crop_include_unknown:
                    crop_class = "banknote_unknown"
                else:
                    continue
                box = record.get("bbox_xyxy")
                if not isinstance(box, list) or len(box) != 4:
                    continue
                crop_dir = args.out / "crops" / split / crop_class
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_box = padded_crop_box([int(value) for value in box], args.image_size, args.crop_pad_frac)
                image.crop(crop_box).save(crop_dir / f"{stem}_{crop_index:02d}_{crop_class}.jpg", quality=90)
        counts[split] += 1
        boxes[split] += len(labels)
        i += 1
        if i % 250 == 0:
            print(f"generated {i}/{args.count} images", flush=True)

    write_yaml(args.out)
    for handle in metadata_handles.values():
        handle.close()
    print(f"source refs: {len(refs)}")
    for split in ["train", "val", "test"]:
        print(f"{split}: {counts[split]} images, {boxes[split]} boxes")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
