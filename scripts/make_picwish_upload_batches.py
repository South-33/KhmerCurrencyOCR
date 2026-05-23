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
    parser.add_argument(
        "--mode",
        choices=["copy", "source-context", "scene-context", "table"],
        default="copy",
        help="Upload image mode: copy originals, crop source context, blend scene crops, or place every image on a table canvas.",
    )
    parser.add_argument(
        "--context-scale",
        type=float,
        default=2.4,
        help="For source-context mode, crop this many YOLO-box widths/heights from the original source image.",
    )
    parser.add_argument("--force", action="store_true", help="Clear the output folder before writing.")
    parser.add_argument("--write-manifest", action="store_true", help="Also write an output CSV mapping uploads to sources.")
    parser.add_argument("--only-stems-file", help="Optional text file of output stems to include, one per line.")
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


def read_stem_filter(path: str | None) -> set[str] | None:
    if not path:
        return None
    stems_path = (ROOT / path).resolve()
    if not stems_path.exists():
        raise SystemExit(f"Missing stems file: {stems_path}")
    stems = {
        line.strip()
        for line in stems_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not stems:
        raise SystemExit(f"No stems found in {stems_path}")
    return stems


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


def load_row_image(row: dict[str, str]) -> Image.Image:
    source = ROOT / row["crop_path"]
    if not source.exists() or source.suffix.lower() not in IMAGE_SUFFIXES:
        raise SystemExit(f"Missing or unsupported source image: {source}")
    with Image.open(source) as opened:
        return opened.convert("RGBA")


def yolo_box_to_xyxy(size: tuple[int, int], yolo_box: str, scale: float) -> tuple[int, int, int, int]:
    width, height = size
    x_center, y_center, box_w, box_h = [float(value) for value in yolo_box.split()]
    crop_w = min(1.0, box_w * scale)
    crop_h = min(1.0, box_h * scale)
    left = max(0, int((x_center - crop_w / 2) * width))
    top = max(0, int((y_center - crop_h / 2) * height))
    right = min(width, int((x_center + crop_w / 2) * width))
    bottom = min(height, int((y_center + crop_h / 2) * height))
    return left, top, right, bottom


def format_xyxy(box: tuple[int, int, int, int]) -> str:
    return " ".join(str(value) for value in box)


def source_context_metadata(row: dict[str, str], context_scale: float) -> dict[str, str]:
    if row.get("kind") != "scene_crop" or not row.get("source_image") or not row.get("source_box_yolo"):
        return {"context_crop_xyxy": "", "target_box_xyxy": "", "target_box_in_context_xyxy": ""}

    source = ROOT / row["source_image"]
    if not source.exists() or source.suffix.lower() not in IMAGE_SUFFIXES:
        raise SystemExit(f"Missing or unsupported source image: {source}")
    with Image.open(source) as opened:
        target_box = yolo_box_to_xyxy(opened.size, row["source_box_yolo"], 1.0)
        context_box = yolo_box_to_xyxy(opened.size, row["source_box_yolo"], context_scale)
    target_in_context = (
        max(0, target_box[0] - context_box[0]),
        max(0, target_box[1] - context_box[1]),
        max(0, target_box[2] - context_box[0]),
        max(0, target_box[3] - context_box[1]),
    )
    return {
        "context_crop_xyxy": format_xyxy(context_box),
        "target_box_xyxy": format_xyxy(target_box),
        "target_box_in_context_xyxy": format_xyxy(target_in_context),
    }


def load_source_context_image(row: dict[str, str], context_scale: float) -> Image.Image:
    if row.get("kind") != "scene_crop" or not row.get("source_image") or not row.get("source_box_yolo"):
        return load_row_image(row)

    source = ROOT / row["source_image"]
    if not source.exists() or source.suffix.lower() not in IMAGE_SUFFIXES:
        raise SystemExit(f"Missing or unsupported source image: {source}")
    with Image.open(source) as opened:
        image = opened.convert("RGBA")
        left, top, right, bottom = yolo_box_to_xyxy(image.size, row["source_box_yolo"], context_scale)
        if right <= left or bottom <= top:
            raise SystemExit(f"Invalid source-context crop for {source}")
        return image.crop((left, top, right, bottom))


def cover_resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    src_w, src_h = image.size
    dst_w, dst_h = size
    scale = max(dst_w / src_w, dst_h / src_h)
    resized = image.resize((int(src_w * scale) + 1, int(src_h * scale) + 1), Image.Resampling.LANCZOS)
    left = (resized.width - dst_w) // 2
    top = (resized.height - dst_h) // 2
    return resized.crop((left, top, left + dst_w, top + dst_h))


def feather_mask(size: tuple[int, int], border: int) -> Image.Image:
    width, height = size
    border = max(1, min(border, width // 3, height // 3))
    mask = Image.new("L", size, 255)
    draw = ImageDraw.Draw(mask)
    for offset in range(border):
        alpha = int(255 * offset / border)
        draw.rectangle((offset, offset, width - offset - 1, height - offset - 1), outline=alpha)
    return mask.filter(ImageFilter.GaussianBlur(radius=max(1, border // 8)))


def paste_scene_context(source: Image.Image, canvas_size: tuple[int, int], index: int) -> Image.Image:
    source = source.convert("RGBA")
    background = cover_resize(source.convert("RGB"), canvas_size)
    background = background.filter(ImageFilter.GaussianBlur(radius=28))
    background = ImageOps.autocontrast(background, cutoff=1)
    overlay = Image.new("RGB", canvas_size, (156, 150, 132))
    background = Image.blend(background, overlay, 0.22)

    canvas_w, canvas_h = canvas_size
    max_w = int(canvas_w * 0.78)
    max_h = int(canvas_h * 0.70)
    scale = min(max_w / source.width, max_h / source.height, 2.2)
    resized = source.resize((max(1, int(source.width * scale)), max(1, int(source.height * scale))), Image.Resampling.LANCZOS)
    x = (canvas_w - resized.width) // 2 + int(math.sin(index * 1.7) * canvas_w * 0.025)
    y = (canvas_h - resized.height) // 2 + int(math.cos(index * 1.3) * canvas_h * 0.025)
    x = max(8, min(canvas_w - resized.width - 8, x))
    y = max(8, min(canvas_h - resized.height - 8, y))

    alpha = resized.getchannel("A")
    edge_mask = feather_mask(resized.size, border=max(12, min(resized.size) // 18))
    alpha = Image.composite(alpha, Image.new("L", resized.size, 0), edge_mask)
    resized.putalpha(alpha)

    result = background.convert("RGBA")
    result.alpha_composite(resized, (x, y))
    return result.convert("RGB")


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


def output_name(row: dict[str, str], index: int, suffix: str) -> str:
    label = row.get("label", "unknown")
    kind = row.get("kind", "asset")
    source_stem = Path(row["crop_path"]).stem
    safe_stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in source_stem)
    return f"{index:04d}_{label}_{kind}_{safe_stem}{suffix.lower()}"


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
        "mode",
        "context_scale",
        "context_crop_xyxy",
        "target_box_xyxy",
        "target_box_in_context_xyxy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_batches(
    rows: list[dict[str, str]],
    out_dir: Path,
    batch_size: int,
    canvas_size: tuple[int, int],
    quality: int,
    mode: str,
    context_scale: float,
    only_stems: set[str] | None,
) -> list[dict[str, str]]:
    if batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    manifest_rows: list[dict[str, str]] = []
    written = 0
    for index, row in enumerate(rows, start=1):
        source = ROOT / row["crop_path"]
        if not source.exists() or source.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"Missing or unsupported source image: {source}")
        target_suffix = source.suffix if mode == "copy" else ".jpg"
        target_name = output_name(row, index, target_suffix)
        if only_stems and Path(target_name).stem not in only_stems:
            continue

        written += 1
        batch_number = (written - 1) // batch_size + 1
        batch_dir = out_dir / f"batch_{batch_number:03d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        target = batch_dir / target_name

        if mode == "copy":
            shutil.copy2(source, target)
        else:
            image = load_source_context_image(row, context_scale) if mode == "source-context" else load_row_image(row)
            if mode == "scene-context" and row.get("kind") == "scene_crop":
                upload_image = paste_scene_context(image, canvas_size, index)
            elif mode == "source-context":
                upload_image = image.convert("RGB")
            else:
                canvas = table_background(canvas_size, index)
                upload_image = paste_with_shadow(canvas, image, index)
            upload_image.save(target, quality=quality, optimize=True)

        context_meta = source_context_metadata(row, context_scale) if mode == "source-context" else {}
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
                "mode": mode,
                "context_scale": f"{context_scale:.3f}" if mode == "source-context" else "",
                **context_meta,
            }
        )
    return manifest_rows


def main() -> None:
    args = parse_args()
    input_dir = (ROOT / args.input).resolve()
    out_dir = (ROOT / args.out).resolve()
    rows = read_manifest(input_dir)
    only_stems = read_stem_filter(args.only_stems_file)

    if out_dir.exists() and not args.force:
        raise SystemExit(f"Output already exists; pass --force to replace it: {out_dir}")
    safe_clear_dir(out_dir)

    manifest_rows = make_batches(
        rows=rows,
        out_dir=out_dir,
        batch_size=args.batch_size,
        canvas_size=(args.canvas_width, args.canvas_height),
        quality=args.quality,
        mode=args.mode,
        context_scale=args.context_scale,
        only_stems=only_stems,
    )
    if args.write_manifest:
        write_csv(out_dir / "manifest.csv", manifest_rows)

    print(f"Wrote {len(manifest_rows)} PicWish upload images to {out_dir}")
    for batch in sorted({row["batch"] for row in manifest_rows}):
        count = sum(1 for row in manifest_rows if row["batch"] == batch)
        print(f"{batch}: {count}")
    if args.write_manifest:
        print(f"Manifest: {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
