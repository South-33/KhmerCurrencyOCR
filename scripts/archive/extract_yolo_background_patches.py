"""Extract note-free background patches from a YOLO dataset.

The synthetic compositor benefits from real table/shop/phone backgrounds, but
using full labeled scenes as backgrounds can leak banknotes into generated
images. This script samples square crops whose overlap with every YOLO label is
below a small threshold, then writes a small QA contact sheet and manifest.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_OUT = ROOT / "data" / "backgrounds" / "cashsnap_v1_no_note_patches"


def split_root(dataset_root: Path, split_path: str) -> Path:
    path = Path(split_path)
    if path.is_absolute():
        return path
    return dataset_root / path


def read_split_images(dataset_root: Path, split_path: str) -> list[Path]:
    resolved = split_root(dataset_root, split_path)
    if resolved.suffix.lower() == ".txt":
        images: list[Path] = []
        for raw_line in resolved.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            images.append(path if path.is_absolute() else dataset_root / path)
        return images
    return sorted(path for path in resolved.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def pad_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    pad_frac: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    pad = max(x2 - x1, y2 - y1) * pad_frac
    return (
        max(0.0, x1 - pad),
        max(0.0, y1 - pad),
        min(float(width), x2 + pad),
        min(float(height), y2 + pad),
    )


def parse_label_boxes(
    label_path: Path,
    width: int,
    height: int,
    pad_frac: float,
) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) < 5:
            raise SystemExit(f"{label_path}:{line_no}: expected YOLO fields, found {len(parts)}")
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise SystemExit(f"{label_path}:{line_no}: non-numeric YOLO values") from exc
        if len(values) == 4:
            cx, cy, bw, bh = values
            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            x2 = (cx + bw / 2) * width
            y2 = (cy + bh / 2) * height
        elif len(values) % 2 == 0:
            xs = values[0::2]
            ys = values[1::2]
            x1 = min(xs) * width
            y1 = min(ys) * height
            x2 = max(xs) * width
            y2 = max(ys) * height
        else:
            raise SystemExit(f"{label_path}:{line_no}: unsupported YOLO label geometry")
        box = (
            max(0.0, min(float(width), x1)),
            max(0.0, min(float(height), y1)),
            max(0.0, min(float(width), x2)),
            max(0.0, min(float(height), y2)),
        )
        boxes.append(pad_box(box, width, height, pad_frac))
    return boxes


def overlap_fraction(crop: tuple[int, int, int, int], box: tuple[float, float, float, float]) -> float:
    x1 = max(crop[0], box[0])
    y1 = max(crop[1], box[1])
    x2 = min(crop[2], box[2])
    y2 = min(crop[3], box[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    return intersection / box_area


def crop_is_clear(
    crop: tuple[int, int, int, int],
    boxes: list[tuple[float, float, float, float]],
    max_label_overlap_frac: float,
) -> bool:
    return all(overlap_fraction(crop, box) <= max_label_overlap_frac for box in boxes)


def choose_crop(
    width: int,
    height: int,
    boxes: list[tuple[float, float, float, float]],
    rng: random.Random,
    min_side_frac: float,
    max_label_overlap_frac: float,
    attempts: int,
) -> tuple[int, int, int, int] | None:
    side_max = min(width, height)
    side_min = max(32, int(side_max * min_side_frac))
    if side_min > side_max:
        return None
    for _ in range(attempts):
        side = rng.randint(side_min, side_max)
        x1 = rng.randint(0, width - side)
        y1 = rng.randint(0, height - side)
        crop = (x1, y1, x1 + side, y1 + side)
        if crop_is_clear(crop, boxes, max_label_overlap_frac):
            return crop
    return None


def make_contact_sheet(paths: list[Path], out: Path, thumb: int = 160, cols: int = 6) -> None:
    if not paths:
        return
    sample = paths[: min(len(paths), cols * 6)]
    rows = (len(sample) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(sample):
        with Image.open(path) as image:
            tile = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (thumb, thumb))
        x = (index % cols) * thumb
        y = (index // cols) * thumb
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y, x + thumb - 1, y + thumb - 1), outline=(190, 190, 190))
    sheet.save(out, quality=90)


def model_sees_note(
    model: object | None,
    crop: Image.Image,
    conf: float,
    imgsz: int,
    device: str,
) -> bool:
    if model is None:
        return False
    kwargs: dict[str, object] = {"imgsz": imgsz, "conf": conf, "verbose": False}
    if device:
        kwargs["device"] = device
    results = model.predict(crop, **kwargs)
    if not results:
        return False
    boxes = getattr(results[0], "boxes", None)
    return bool(boxes is not None and len(boxes) > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "cashsnap_v1" / "data.yaml")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated split keys to mine.")
    parser.add_argument("--min-side-frac", type=float, default=0.35)
    parser.add_argument("--max-label-overlap-frac", type=float, default=0.0)
    parser.add_argument("--box-pad-frac", type=float, default=0.18, help="Pad note boxes before rejecting background crops.")
    parser.add_argument("--allow-empty-labels", action="store_true", help="Allow images with no labels as background sources.")
    parser.add_argument("--attempts-per-image", type=int, default=30)
    parser.add_argument("--reject-model", type=Path, help="Optional YOLO model used to reject candidate patches with detected notes.")
    parser.add_argument("--reject-conf", type=float, default=0.12, help="Confidence threshold for --reject-model.")
    parser.add_argument("--reject-imgsz", type=int, default=640, help="Inference size for --reject-model.")
    parser.add_argument("--reject-device", default="", help="Optional Ultralytics device string for --reject-model.")
    parser.add_argument("--clean", action="store_true", help="Delete an existing output background bank.")
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    if args.output_size < 64:
        raise SystemExit("--output-size must be >= 64")
    if not (0 < args.min_side_frac <= 1):
        raise SystemExit("--min-side-frac must be > 0 and <= 1")
    if args.max_label_overlap_frac < 0:
        raise SystemExit("--max-label-overlap-frac must be >= 0")
    if args.box_pad_frac < 0:
        raise SystemExit("--box-pad-frac must be >= 0")
    if args.reject_model and not args.reject_model.exists():
        raise SystemExit(f"--reject-model does not exist: {args.reject_model}")

    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = args.data.parent / dataset_root
    dataset_root = dataset_root.resolve()
    split_keys = [item.strip() for item in args.splits.split(",") if item.strip()]
    images: list[tuple[str, Path]] = []
    for split in split_keys:
        split_path = config.get(split)
        if not split_path:
            continue
        images.extend((split, path) for path in read_split_images(dataset_root, split_path))
    if not images:
        raise SystemExit(f"No images found for splits {split_keys} in {args.data}")

    reject_model = None
    if args.reject_model:
        from ultralytics import YOLO

        reject_model = YOLO(str(args.reject_model))

    if args.clean and args.out.exists():
        resolved = args.out.resolve()
        allowed_root = (ROOT / "data" / "backgrounds").resolve()
        if allowed_root not in resolved.parents and resolved != allowed_root:
            raise SystemExit(f"Refusing to clean outside {allowed_root}: {resolved}")
        shutil.rmtree(resolved)
    args.out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    written: list[Path] = []
    manifest_rows: list[dict[str, object]] = []
    failures = 0
    rounds = 0
    while len(written) < args.count and rounds < max(2, args.count // max(1, len(images)) + 3):
        shuffled = images[:]
        rng.shuffle(shuffled)
        for split, image_path in shuffled:
            if len(written) >= args.count:
                break
            try:
                with Image.open(image_path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    width, height = image.size
                    boxes = parse_label_boxes(label_path_for_image(image_path), width, height, args.box_pad_frac)
                    if not boxes and not args.allow_empty_labels:
                        failures += 1
                        continue
                    crop_box = choose_crop(
                        width,
                        height,
                        boxes,
                        rng,
                        args.min_side_frac,
                        args.max_label_overlap_frac,
                        args.attempts_per_image,
                    )
                    if crop_box is None:
                        failures += 1
                        continue
                    crop = image.crop(crop_box).resize(
                        (args.output_size, args.output_size),
                        Image.Resampling.BICUBIC,
                    )
                    if model_sees_note(reject_model, crop, args.reject_conf, args.reject_imgsz, args.reject_device):
                        failures += 1
                        continue
            except OSError:
                failures += 1
                continue
            stem = f"bg_{len(written):06d}_{split}"
            out_path = args.out / f"{stem}.jpg"
            crop.save(out_path, quality=90)
            written.append(out_path)
            manifest_rows.append(
                {
                    "path": out_path.relative_to(args.out).as_posix(),
                    "source_image": image_path.resolve().as_posix(),
                    "split": split,
                    "crop_x1": crop_box[0],
                    "crop_y1": crop_box[1],
                    "crop_x2": crop_box[2],
                    "crop_y2": crop_box[3],
                    "label_count": len(boxes),
                }
            )
        rounds += 1

    manifest_path = args.out / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "source_image", "split", "crop_x1", "crop_y1", "crop_x2", "crop_y2", "label_count"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    make_contact_sheet(written, args.out / "contact_sheet.jpg")
    print(f"source images: {len(images)}")
    print(f"background patches: {len(written)}")
    print(f"failed attempts: {failures}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
