from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiny targeted ImageFolder crop probe from banknote assets.")
    parser.add_argument("--asset", action="append", required=True, help="class=path to an RGBA/RGB note asset.")
    parser.add_argument(
        "--box",
        action="append",
        required=True,
        help="class=x1,y1,x2,y2 normalized crop prior. Add multiple boxes per class.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--count-per-class", type=int, default=160)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--ensure-classes", default="")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (ROOT / "data").resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise SystemExit(f"Refusing to clean outside {allowed_root}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def parse_key_value(items: list[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Expected class=value item, got: {item}")
        key, value = item.split("=", 1)
        parsed.setdefault(key.strip(), []).append(value.strip())
    return parsed


def parse_box(text: str) -> tuple[float, float, float, float]:
    parts = [float(part.strip()) for part in re.split(r"[,\s]+", text) if part.strip()]
    if len(parts) != 4:
        raise SystemExit(f"Expected x1,y1,x2,y2 box, got: {text}")
    x1, y1, x2, y2 = parts
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise SystemExit(f"Box must be normalized and ordered: {text}")
    return x1, y1, x2, y2


def jitter_box(
    box: tuple[float, float, float, float],
    rng: random.Random,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    scale = rng.uniform(0.82, 1.18)
    cx = (x1 + x2) / 2 + rng.uniform(-0.10, 0.10) * width
    cy = (y1 + y2) / 2 + rng.uniform(-0.10, 0.10) * height
    new_w = min(1.0, width * scale)
    new_h = min(1.0, height * scale)
    nx1 = max(0.0, min(1.0 - new_w, cx - new_w / 2))
    ny1 = max(0.0, min(1.0 - new_h, cy - new_h / 2))
    return nx1, ny1, nx1 + new_w, ny1 + new_h


def augment(image: Image.Image, rng: random.Random, size: int) -> Image.Image:
    image = image.convert("RGB")
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.65, 1.20))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.45:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.1)))
    angle = rng.uniform(-12, 12)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(rng.randrange(205, 246),) * 3)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (rng.randrange(210, 245), rng.randrange(205, 240), rng.randrange(198, 235)))
    x = rng.randrange(0, max(1, size - image.width + 1))
    y = rng.randrange(0, max(1, size - image.height + 1))
    canvas.paste(image, (x, y))
    return canvas


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    assets = {key: [resolve(value) for value in values] for key, values in parse_key_value(args.asset).items()}
    boxes = {key: [parse_box(value) for value in values] for key, values in parse_key_value(args.box).items()}
    classes = sorted(set(assets) | set(boxes) | {value.strip() for value in re.split(r"[,;\s]+", args.ensure_classes) if value.strip()})
    for class_name in sorted(set(assets) | set(boxes)):
        if class_name not in assets or class_name not in boxes:
            raise SystemExit(f"Class {class_name} needs at least one asset and one box")
    out_dir = resolve(args.out)
    if args.clean:
        safe_clean(out_dir)
    rows: list[dict[str, str]] = []
    for split in ["train", "val", "test"]:
        for class_name in classes:
            (out_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    for class_name in sorted(assets):
        loaded = [Image.open(path).convert("RGBA") for path in assets[class_name]]
        for index in range(args.count_per_class):
            split = "train" if index < int(args.count_per_class * 0.80) else ("val" if index < int(args.count_per_class * 0.90) else "test")
            asset_index = rng.randrange(len(loaded))
            note = loaded[asset_index]
            x1, y1, x2, y2 = jitter_box(rng.choice(boxes[class_name]), rng)
            crop = note.crop((round(x1 * note.width), round(y1 * note.height), round(x2 * note.width), round(y2 * note.height)))
            image = augment(crop, rng, args.image_size)
            target = out_dir / split / class_name / f"{class_name}_targeted_{index:05d}.jpg"
            image.save(target, quality=rng.randrange(72, 94))
            rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "image_path": target.relative_to(ROOT).as_posix(),
                    "source_asset": str(assets[class_name][asset_index].relative_to(ROOT)),
                    "box": f"{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}",
                }
            )
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class_name", "image_path", "source_asset", "box"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} targeted crops to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
