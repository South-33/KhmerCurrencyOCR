"""Generate synthetic fanned/occluded KHR banknote YOLO data.

The generator starts from clean reference banknote images where the note is the
main object, layers several notes into fanned stacks, and labels only the
visible region of each note after occlusion.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "reference" / "khr_nbc"
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
            class_name = parse_note_class(path)
            if class_name and (allowed_classes is None or class_name in allowed_classes) and looks_like_whole_note(path):
                refs.append(NoteRef(path=path, class_name=class_name))
    if not refs:
        raise SystemExit(f"No target KHR reference images found in {sources}")
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
    try:
        image = Image.open(path).convert("RGB").resize((160, 80))
    except OSError:
        return 1.0
    arr = np.asarray(image).astype(np.int16)
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    strong_red = (red > 125) & ((red - green) > 45) & ((red - blue) > 45)
    # SPECIMEN stamps create a large red connected-looking footprint. Normal
    # note artwork can be red too, so keep the threshold conservative.
    return float(strong_red.mean())


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


def crop_note_strip(image: Image.Image, rng: random.Random) -> Image.Image:
    """Keep one vertical slice of a note for extreme partial-visibility scenes."""
    if image.width < 60:
        return image
    strip_w = int(image.width * rng.uniform(0.16, 0.38))
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


def make_background(size: int, rng: random.Random) -> Image.Image:
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


def add_hand_occluders(canvas: Image.Image, id_mask: np.ndarray, rng: random.Random, probability: float) -> None:
    if probability <= 0 or rng.random() > probability:
        return
    occ = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    occ_mask = Image.new("L", canvas.size, 0)
    count = rng.randint(1, 4)
    for _ in range(count):
        x = rng.randint(int(canvas.width * 0.15), int(canvas.width * 0.85))
        y = rng.randint(int(canvas.height * 0.35), int(canvas.height * 0.92))
        w = rng.randint(28, 58)
        h = rng.randint(105, 190)
        angle = rng.uniform(-25, 25)
        finger = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        color = (
            rng.randint(145, 215),
            rng.randint(95, 165),
            rng.randint(65, 130),
            rng.randint(215, 245),
        )
        yy, xx = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        ellipse = (((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2) <= 1
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        arr[ellipse] = color
        finger = Image.fromarray(arr, "RGBA").filter(ImageFilter.GaussianBlur(1.2))
        finger = finger.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        px = x - finger.width // 2
        py = y - finger.height // 2
        occ.alpha_composite(finger, (px, py))
        alpha = finger.getchannel("A")
        occ_mask.paste(alpha, (px, py), alpha)
    canvas.alpha_composite(occ)
    occ_arr = np.asarray(occ_mask) > 24
    id_mask[occ_arr] = 0


def paste_note(
    canvas: Image.Image,
    id_mask: np.ndarray,
    note: Image.Image,
    instance_id: int,
    x: int,
    y: int,
) -> None:
    alpha = np.asarray(note.getchannel("A")) > 16
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(canvas.width, x + note.width)
    y2 = min(canvas.height, y + note.height)
    if x1 >= x2 or y1 >= y2:
        return
    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    canvas.alpha_composite(note, (x, y))
    visible = alpha[sy1:sy2, sx1:sx2]
    id_mask[y1:y2, x1:x2][visible] = instance_id


def visible_box(id_mask: np.ndarray, instance_id: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(id_mask == instance_id)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def prepared_note(ref: NoteRef, cache: dict[Path, Image.Image]) -> Image.Image:
    if ref.path not in cache:
        cache[ref.path] = note_alpha(Image.open(ref.path)).copy()
    return cache[ref.path].copy()


def make_scene(
    refs: list[NoteRef],
    image_size: int,
    rng: random.Random,
    note_cache: dict[Path, Image.Image] | None = None,
    min_notes: int = 4,
    max_notes: int = 12,
    layout_modes: list[str] | None = None,
    hand_probability: float = 0.25,
) -> tuple[Image.Image, list[str]]:
    canvas = make_background(image_size, rng).convert("RGBA")
    id_mask = np.zeros((image_size, image_size), dtype=np.uint16)
    labels: list[tuple[str, int]] = []
    note_count = rng.randint(min_notes, max_notes)
    available_modes = layout_modes or [
        "radial_slice",
        "strip_fan",
        "tight_fan",
        "fan",
        "crossed",
        "scattered",
        "row",
    ]
    default_weights = {
        "radial_slice": 0.28,
        "strip_fan": 0.24,
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
    elif layout_mode == "tight_fan":
        note_count = rng.randint(max(min_notes, 9), max(max_notes, 18))
    center_x = rng.randint(int(image_size * 0.43), int(image_size * 0.57))
    pivot_y = rng.randint(int(image_size * 0.78), int(image_size * 0.96))

    for i in range(note_count):
        ref = rng.choice(refs)
        note = prepared_note(ref, note_cache) if note_cache is not None else note_alpha(Image.open(ref.path))
        if layout_mode == "radial_slice":
            target_w = int(image_size * rng.uniform(0.72, 1.02))
        elif layout_mode == "strip_fan":
            target_w = int(image_size * rng.uniform(0.72, 0.96))
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
            note = crop_note_strip(note, rng)

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

        instance_id = len(labels) + 1
        paste_note(canvas, id_mask, note, instance_id, x, y)
        labels.append((ref.class_name, instance_id))

    add_hand_occluders(canvas, id_mask, rng, hand_probability)

    yolo_lines: list[str] = []
    for class_name, instance_id in labels:
        box = visible_box(id_mask, instance_id)
        if not box:
            continue
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        area = w * h
        if area < image_size * image_size * 0.004 or w < 18 or h < 18:
            continue
        cx = (x1 + x2) / 2 / image_size
        cy = (y1 + y2) / 2 / image_size
        bw = w / image_size
        bh = h / image_size
        yolo_lines.append(f"{CLASS_TO_ID[class_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    rgb = canvas.convert("RGB")
    if rng.random() < 0.45:
        arr = np.asarray(rgb).astype(np.int16)
        grain = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, rng.uniform(3, 12), arr.shape)
        rgb = Image.fromarray(np.clip(arr + grain, 0, 255).astype(np.uint8), "RGB")
    if rng.random() < 0.25:
        rgb = ImageOps.autocontrast(rgb, cutoff=rng.uniform(0, 1.5))
    return rgb, yolo_lines


def write_yaml(out: Path) -> None:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    dataset_path = out.resolve().as_posix()
    (out / "data.yaml").write_text(
        f"path: {dataset_path}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n{names}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, nargs="+", default=[DEFAULT_SOURCE])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--clean", action="store_true", help="Delete an existing output directory before generating.")
    parser.add_argument("--allow-specimen", action="store_true", help="Allow reference images that appear to contain SPECIMEN marks.")
    parser.add_argument("--classes", default="", help="Optional comma-separated canonical classes to generate.")
    parser.add_argument("--min-notes", type=int, default=4, help="Minimum notes per synthetic scene.")
    parser.add_argument("--max-notes", type=int, default=12, help="Maximum notes per synthetic scene.")
    parser.add_argument(
        "--layout-modes",
        default="",
        help="Optional comma-separated layout modes: radial_slice,strip_fan,tight_fan,fan,crossed,scattered,row.",
    )
    parser.add_argument("--hand-prob", type=float, default=0.25, help="Probability of adding synthetic hand/finger occluders.")
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
        "tight_fan",
        "fan",
        "crossed",
        "scattered",
        "row",
    }
    unknown_layouts = sorted(set(layout_modes or []) - valid_layout_modes)
    if unknown_layouts:
        raise SystemExit(f"Unknown layout modes in --layout-modes: {unknown_layouts}")
    if args.min_notes < 1 or args.max_notes < args.min_notes:
        raise SystemExit("--min-notes must be >= 1 and --max-notes must be >= --min-notes")
    refs = load_refs(args.source, allowed_classes)
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

    counts = {"train": 0, "val": 0, "test": 0}
    boxes = {"train": 0, "val": 0, "test": 0}
    note_cache: dict[Path, Image.Image] = {}
    i = 0
    attempts = 0
    while i < args.count:
        attempts += 1
        image, labels = make_scene(
            refs,
            args.image_size,
            rng,
            note_cache,
            min_notes=args.min_notes,
            max_notes=args.max_notes,
            layout_modes=layout_modes,
            hand_probability=args.hand_prob,
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
        image.save(args.out / "images" / split / f"{stem}.jpg", quality=88)
        (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        counts[split] += 1
        boxes[split] += len(labels)
        i += 1
        if i % 250 == 0:
            print(f"generated {i}/{args.count} images", flush=True)

    write_yaml(args.out)
    print(f"source refs: {len(refs)}")
    for split in ["train", "val", "test"]:
        print(f"{split}: {counts[split]} images, {boxes[split]} boxes")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
