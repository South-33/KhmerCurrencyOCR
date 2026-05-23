from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Place PicWish input crops on context canvases and split them into upload batches."
    )
    parser.add_argument("--input", default="data/asset_candidates/picwish_input", help="PicWish candidate input folder.")
    parser.add_argument("--out", default="data/picwish_upload_batches", help="Output folder for upload batches.")
    parser.add_argument("--batch-size", type=int, default=100, help="Maximum images per upload batch.")
    parser.add_argument("--canvas-width", type=int, default=1200, help="Output canvas width.")
    parser.add_argument("--canvas-height", type=int, default=900, help="Output canvas height.")
    parser.add_argument("--quality", type=int, default=94, help="JPEG quality for upload images.")
    parser.add_argument("--force", action="store_true", help="Clear the output folder before writing.")
    return parser.parse_args()


def safe_clear_dir(path: Path) -> None:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise SystemExit(f"Refusing to clear path outside repo: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def read_manifest(input_dir: Path) -> list[dict[str, str]]:
    manifest = input_dir / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"Missing manifest: {manifest}")
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("crop_path")]
    if not rows:
        raise SystemExit(f"No crop_path rows found in {manifest}")
    return rows


def table_background(size: tuple[int, int], index: int) -> Image.Image:
    width, height = size
    base_options = [
        (179, 157, 132),
        (157, 170, 155),
        (142, 154, 166),
        (171, 164, 145),
    ]
    base = base_options[index % len(base_options)]
    image = Image.new("RGB", size, base)
    gradient = Image.linear_gradient("L").resize((width, height)).point(lambda value: int(value * 0.18))
    image = Image.composite(
        Image.new("RGB", size, tuple(min(255, channel + 22) for channel in base)),
        image,
        gradient,
    )
    rng = random.Random(index)
    noise_size = (max(1, width // 8), max(1, height // 8))
    noise = Image.new("L", noise_size)
    noise.putdata([rng.randrange(96, 160) for _ in range(noise_size[0] * noise_size[1])])
    noise = noise.resize(size, Image.Resampling.BICUBIC)
    noise = ImageOps.autocontrast(noise).point(lambda value: int(value * 0.18))
    image = Image.composite(
        Image.new("RGB", size, tuple(max(0, channel - 14) for channel in base)),
        image,
        noise,
    )

    draw = ImageDraw.Draw(image, "RGBA")
    plank_gap = 230 + (index % 4) * 21
    for x in range(-80, width + 80, plank_gap):
        draw.line((x, 0, x + 50, height), fill=(70, 62, 52, 34), width=4)
    for y in range(120 + (index % 5) * 19, height, 250):
        draw.line((0, y, width, y + 18), fill=(255, 255, 255, 18), width=3)

    return image.filter(ImageFilter.GaussianBlur(radius=0.25))


def fit_size(source: Image.Image, canvas_size: tuple[int, int]) -> tuple[int, int]:
    canvas_w, canvas_h = canvas_size
    max_w = int(canvas_w * 0.72)
    max_h = int(canvas_h * 0.62)
    scale = min(max_w / source.width, max_h / source.height)
    scale = min(scale, 2.6)
    return max(1, int(source.width * scale)), max(1, int(source.height * scale))


def paste_with_shadow(canvas: Image.Image, source: Image.Image, index: int) -> Image.Image:
    source = source.convert("RGBA")
    resized = source.resize(fit_size(source, canvas.size), Image.Resampling.LANCZOS)
    canvas_w, canvas_h = canvas.size
    x = (canvas_w - resized.width) // 2 + int(math.sin(index * 1.7) * canvas_w * 0.035)
    y = (canvas_h - resized.height) // 2 + int(math.cos(index * 1.3) * canvas_h * 0.035)
    x = max(24, min(canvas_w - resized.width - 24, x))
    y = max(24, min(canvas_h - resized.height - 24, y))

    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_layer = Image.new("RGBA", resized.size, (0, 0, 0, 105))
    shadow_layer.putalpha(resized.getchannel("A").filter(ImageFilter.GaussianBlur(radius=1.2)))
    shadow_offset = (x + 18, y + 22)
    shadow.alpha_composite(shadow_layer, shadow_offset)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=14))

    result = canvas.convert("RGBA")
    result.alpha_composite(shadow)
    result.alpha_composite(resized, (x, y))
    return result.convert("RGB")


def output_name(row: dict[str, str], index: int) -> str:
    label = row.get("label", "unknown")
    kind = row.get("kind", "asset")
    source_stem = Path(row["crop_path"]).stem
    safe_stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in source_stem)
    return f"{index:04d}_{label}_{kind}_{safe_stem}.jpg"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch",
        "upload_image",
        "source_crop",
        "label",
        "kind",
        "split",
        "source_image",
        "source_box_yolo",
        "crop_box_xyxy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_batches(rows: list[dict[str, str]], out_dir: Path, batch_size: int, canvas_size: tuple[int, int], quality: int) -> list[dict[str, str]]:
    if batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    manifest_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        source = ROOT / row["crop_path"]
        if not source.exists() or source.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"Missing or unsupported source image: {source}")
        batch_number = (index - 1) // batch_size + 1
        batch_dir = out_dir / f"batch_{batch_number:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        target = batch_dir / output_name(row, index)

        with Image.open(source) as image:
            canvas = table_background(canvas_size, index)
            upload_image = paste_with_shadow(canvas, image, index)
            upload_image.save(target, quality=quality, optimize=True)

        manifest_rows.append(
            {
                "batch": batch_dir.name,
                "upload_image": str(target.relative_to(ROOT)),
                "source_crop": row["crop_path"],
                "label": row.get("label", ""),
                "kind": row.get("kind", ""),
                "split": row.get("split", ""),
                "source_image": row.get("source_image", ""),
                "source_box_yolo": row.get("source_box_yolo", ""),
                "crop_box_xyxy": row.get("crop_box_xyxy", ""),
            }
        )
    return manifest_rows


def main() -> None:
    args = parse_args()
    input_dir = (ROOT / args.input).resolve()
    out_dir = (ROOT / args.out).resolve()
    rows = read_manifest(input_dir)

    if out_dir.exists() and not args.force:
        raise SystemExit(f"Output already exists; pass --force to replace it: {out_dir}")
    safe_clear_dir(out_dir)

    manifest_rows = make_batches(
        rows=rows,
        out_dir=out_dir,
        batch_size=args.batch_size,
        canvas_size=(args.canvas_width, args.canvas_height),
        quality=args.quality,
    )
    write_csv(out_dir / "manifest.csv", manifest_rows)

    print(f"Wrote {len(manifest_rows)} PicWish upload images to {out_dir}")
    for batch in sorted({row["batch"] for row in manifest_rows}):
        count = sum(1 for row in manifest_rows if row["batch"] == batch)
        print(f"{batch}: {count}")
    print(f"Manifest: {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
