from __future__ import annotations

import argparse
import csv
import re
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    "KHR_500",
    "KHR_1000",
    "KHR_2000",
    "KHR_5000",
    "KHR_10000",
    "KHR_20000",
    "KHR_50000",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES_BY_LENGTH = sorted(CLASS_NAMES, key=len, reverse=True)


@dataclass(frozen=True)
class Asset:
    path: Path
    class_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a denomination fragment-classifier dataset from transparent banknote assets."
    )
    parser.add_argument("--source", nargs="+", required=True, help="Asset bank folders with class subdirectories.")
    parser.add_argument("--out", required=True, help="Output ImageFolder dataset directory.")
    parser.add_argument("--classes", default="", help="Optional comma-separated class names to include.")
    parser.add_argument("--include-name-regex", default="", help="Optional regex that asset paths/names must match.")
    parser.add_argument("--count-per-class", type=int, default=300)
    parser.add_argument("--background-count", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--filename-prefix", default="", help="Prefix generated filenames when appending to a dataset.")
    parser.add_argument("--val-frac", type=float, default=0.12)
    parser.add_argument("--test-frac", type=float, default=0.08)
    parser.add_argument("--min-visible-area", type=float, default=0.08)
    parser.add_argument("--clean", action="store_true", help="Delete the output directory first.")
    return parser.parse_args()


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (ROOT / "data").resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise SystemExit(f"Refusing to clean outside {allowed_root}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def split_list(value: str) -> set[str]:
    return {item.strip() for item in re.split(r"[,\s]+", value) if item.strip()}


def collect_assets(sources: list[str], classes: set[str], include_name_regex: str) -> list[Asset]:
    pattern = re.compile(include_name_regex, flags=re.IGNORECASE) if include_name_regex else None
    assets: list[Asset] = []
    for source_text in sources:
        source = resolve(source_text)
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if "masks" in {part.lower() for part in path.parts}:
                continue
            normalized = str(path.relative_to(ROOT)).replace("\\", "/")
            if pattern is not None and not pattern.search(normalized):
                continue
            class_name = next((name for name in CLASS_NAMES if name in path.parts), None)
            if class_name is None:
                class_name = next((name for name in CLASS_NAMES_BY_LENGTH if name in path.name), None)
            if class_name is None or (classes and class_name not in classes):
                continue
            assets.append(Asset(path=path, class_name=class_name))
    return assets


def trim_alpha(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0:
        return rgba
    left, top = int(xs.min()), int(ys.min())
    right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
    return rgba.crop((left, top, right, bottom))


def random_fragment(note: Image.Image, rng: random.Random, min_visible_area: float) -> tuple[Image.Image, str] | None:
    note = trim_alpha(note)
    if note.width < 16 or note.height < 16:
        return None

    modes = ["vertical_strip", "wide_strip", "corner", "center_patch", "fullish"]
    for _ in range(20):
        mode = rng.choice(modes)
        if mode == "vertical_strip":
            crop_w = max(12, round(note.width * rng.uniform(0.12, 0.38)))
            crop_h = round(note.height * rng.uniform(0.62, 1.0))
            left = rng.randint(0, max(0, note.width - crop_w))
            top = rng.randint(0, max(0, note.height - crop_h))
        elif mode == "wide_strip":
            crop_w = round(note.width * rng.uniform(0.35, 0.80))
            crop_h = max(12, round(note.height * rng.uniform(0.28, 0.70)))
            left = rng.randint(0, max(0, note.width - crop_w))
            top = rng.randint(0, max(0, note.height - crop_h))
        elif mode == "corner":
            crop_w = round(note.width * rng.uniform(0.22, 0.52))
            crop_h = round(note.height * rng.uniform(0.35, 0.75))
            left = rng.choice([0, max(0, note.width - crop_w)])
            top = rng.choice([0, max(0, note.height - crop_h)])
        elif mode == "center_patch":
            crop_w = round(note.width * rng.uniform(0.25, 0.55))
            crop_h = round(note.height * rng.uniform(0.35, 0.75))
            center_x = rng.randint(round(note.width * 0.25), max(round(note.width * 0.75), round(note.width * 0.25)))
            center_y = rng.randint(round(note.height * 0.25), max(round(note.height * 0.75), round(note.height * 0.25)))
            left = min(max(0, center_x - crop_w // 2), max(0, note.width - crop_w))
            top = min(max(0, center_y - crop_h // 2), max(0, note.height - crop_h))
        else:
            crop_w = round(note.width * rng.uniform(0.65, 1.0))
            crop_h = round(note.height * rng.uniform(0.70, 1.0))
            left = rng.randint(0, max(0, note.width - crop_w))
            top = rng.randint(0, max(0, note.height - crop_h))

        fragment = note.crop((left, top, left + crop_w, top + crop_h))
        alpha = np.asarray(fragment.getchannel("A")) > 16
        if alpha.sum() / max(1, alpha.size) >= min_visible_area:
            return fragment, mode
    return None


def augment_fragment(fragment: Image.Image, rng: random.Random) -> Image.Image:
    image = fragment.convert("RGBA")
    if rng.random() < 0.55:
        image = image.rotate(rng.uniform(-16, 16), expand=True, resample=Image.Resampling.BICUBIC)
    if rng.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.8)))
    rgb = image.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.78, 1.22))
    rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.82, 1.20))
    rgb = ImageEnhance.Color(rgb).enhance(rng.uniform(0.72, 1.18))
    rgb.putalpha(image.getchannel("A"))
    return rgb


def background(size: int, rng: random.Random) -> Image.Image:
    base = np.zeros((size, size, 3), dtype=np.uint8)
    color = np.array(
        [
            rng.randint(150, 235),
            rng.randint(145, 230),
            rng.randint(135, 225),
        ],
        dtype=np.int16,
    )
    noise = rng.randint(4, 18)
    arr = np.clip(color + np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, noise, base.shape), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGB")


def compose_square(fragment: Image.Image, size: int, rng: random.Random) -> Image.Image:
    fragment = augment_fragment(fragment, rng)
    scale = min(size * rng.uniform(0.55, 0.95) / max(1, fragment.width), size * rng.uniform(0.40, 0.90) / max(1, fragment.height))
    width = max(8, round(fragment.width * scale))
    height = max(8, round(fragment.height * scale))
    fragment = fragment.resize((width, height), Image.Resampling.LANCZOS)
    canvas = background(size, rng).convert("RGBA")
    x = rng.randint(0, max(0, size - width))
    y = rng.randint(0, max(0, size - height))
    canvas.alpha_composite(fragment, (x, y))
    result = canvas.convert("RGB")
    if rng.random() < 0.45:
        result = ImageOps.autocontrast(result, cutoff=rng.uniform(0, 2))
    return result


def split_for_index(index: int, total: int, val_frac: float, test_frac: float) -> str:
    val_cutoff = round(total * val_frac)
    test_cutoff = val_cutoff + round(total * test_frac)
    if index < val_cutoff:
        return "val"
    if index < test_cutoff:
        return "test"
    return "train"


def write_background(
    out_dir: Path,
    count: int,
    size: int,
    rng: random.Random,
    val_frac: float,
    test_frac: float,
    filename_prefix: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(count):
        split = split_for_index(index, count, val_frac, test_frac)
        image = background(size, rng)
        if rng.random() < 0.35:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.2)))
        target = out_dir / split / "background" / f"{filename_prefix}background_{index:05d}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, quality=90)
        rows.append(
            {
                "split": split,
                "class_name": "background",
                "mode": "background",
                "source_path": "",
                "image_path": str(target.relative_to(ROOT)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_contact_sheet(rows: list[dict[str, str]], out_path: Path, limit: int = 96) -> None:
    selected = rows[:limit]
    if not selected:
        return
    thumb, label_h, cols = 120, 28, 8
    sheet = Image.new(
        "RGB",
        (cols * thumb, ((len(selected) + cols - 1) // cols) * (thumb + label_h) + 30),
        "white",
    )
    for index, row in enumerate(selected):
        image_path = ROOT / row["image_path"]
        with Image.open(image_path).convert("RGB") as image:
            image.thumbnail((thumb, thumb))
            tile = Image.new("RGB", (thumb, thumb), (245, 245, 245))
            tile.paste(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
        x = (index % cols) * thumb
        y = 30 + (index // cols) * (thumb + label_h)
        sheet.paste(tile, (x, y))
        label = row["class_name"].replace("KHR_", "")
        ImageDraw.Draw(sheet).text((x + 3, y + thumb + 4), label[:18], fill="black")
    ImageDraw.Draw(sheet).text((8, 8), out_path.parent.name, fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = resolve(args.out)
    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_rows = [] if args.clean else read_csv(out_dir / "manifest.csv")

    selected_classes = split_list(args.classes)
    assets = collect_assets(args.source, selected_classes, args.include_name_regex)
    by_class = {class_name: [asset for asset in assets if asset.class_name == class_name] for class_name in CLASS_NAMES}
    rows: list[dict[str, str]] = []
    for class_name, class_assets in by_class.items():
        if not class_assets:
            print(f"{class_name}: no assets")
            continue
        for index in range(args.count_per_class):
            asset = rng.choice(class_assets)
            with Image.open(asset.path).convert("RGBA") as note:
                fragment_result = random_fragment(note, rng, args.min_visible_area)
            if fragment_result is None:
                continue
            fragment, mode = fragment_result
            split = split_for_index(index, args.count_per_class, args.val_frac, args.test_frac)
            image = compose_square(fragment, args.image_size, rng)
            target = out_dir / split / class_name / f"{args.filename_prefix}{class_name}_{index:05d}_{mode}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, quality=91)
            rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "mode": mode,
                    "source_path": str(asset.path.relative_to(ROOT)),
                    "image_path": str(target.relative_to(ROOT)),
                }
            )

    rows.extend(
        write_background(
            out_dir,
            args.background_count,
            args.image_size,
            rng,
            args.val_frac,
            args.test_frac,
            args.filename_prefix,
        )
    )
    write_csv(out_dir / "manifest.csv", [*existing_rows, *rows])
    make_contact_sheet(rows, out_dir / "contact_sheet.jpg")
    print(f"wrote {len(rows)} fragment images to {out_dir.relative_to(ROOT)}")
    if existing_rows:
        print(f"preserved {len(existing_rows)} existing manifest rows")
    for split in ["train", "val", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        print(f"{split}: {len(split_rows)}")
        for class_name in [*CLASS_NAMES, "background"]:
            count = sum(1 for row in split_rows if row["class_name"] == class_name)
            if count:
                print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()
