from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag likely bad PicWish background-removal outputs.")
    parser.add_argument("--inputs", default="data/picwish_upload_batches", help="Folder containing batch input images.")
    parser.add_argument("--outputs", default="data/asset_candidates/picwish_output", help="Folder containing PicWish PNG outputs.")
    parser.add_argument("--out", default="data/asset_candidates/picwish_audit", help="Audit output folder.")
    parser.add_argument("--alpha-threshold", type=int, default=16, help="Alpha threshold counted as foreground.")
    parser.add_argument("--max-contact", type=int, default=80, help="Maximum suspects shown on the contact sheet.")
    return parser.parse_args()


def find_input(inputs_dir: Path, output_stem: str) -> Path | None:
    for path in inputs_dir.glob(f"batch_*/*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stem == output_stem:
            return path
    return None


def alpha_metrics(path: Path, threshold: int) -> dict[str, float | int | str]:
    with Image.open(path).convert("RGBA") as image:
        alpha = np.array(image.getchannel("A"))
    mask = alpha > threshold
    area = int(mask.sum())
    image_area = int(mask.shape[0] * mask.shape[1])
    if area == 0:
        return {
            "status": "empty",
            "foreground_area_ratio": 0.0,
            "bbox_width": 0,
            "bbox_height": 0,
            "bbox_aspect": 0.0,
            "aspect_norm": 0.0,
            "bbox_fill_ratio": 0.0,
            "bbox_area_ratio": 0.0,
        }

    ys, xs = np.where(mask)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    bbox_width = right - left
    bbox_height = bottom - top
    bbox_area = bbox_width * bbox_height
    bbox_aspect = bbox_width / max(1, bbox_height)
    aspect_norm = max(bbox_aspect, 1 / max(bbox_aspect, 1e-6))
    return {
        "status": "ok",
        "foreground_area_ratio": area / image_area,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_aspect": bbox_aspect,
        "aspect_norm": aspect_norm,
        "bbox_fill_ratio": area / max(1, bbox_area),
        "bbox_area_ratio": bbox_area / image_area,
    }


def flags_for(metrics: dict[str, float | int | str]) -> list[str]:
    if metrics["status"] == "empty":
        return ["empty_mask"]

    area = float(metrics["foreground_area_ratio"])
    aspect_norm = float(metrics["aspect_norm"])
    fill = float(metrics["bbox_fill_ratio"])
    bbox_area = float(metrics["bbox_area_ratio"])
    flags: list[str] = []

    if area < 0.18 or bbox_area < 0.22:
        flags.append("tiny_foreground")
    if area < 0.30 and aspect_norm < 1.85:
        flags.append("portrait_or_small_fragment")
    if aspect_norm < 1.35 and area < 0.70:
        flags.append("squareish_not_banknote")
    if fill < 0.58 and area < 0.60:
        flags.append("ragged_partial_mask")
    if area > 0.94 and fill > 0.94:
        flags.append("likely_kept_full_rectangle")

    return flags


def checkerboard(size: tuple[int, int], cell: int = 12) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            color = (226, 226, 226) if (x // cell + y // cell) % 2 else (248, 248, 248)
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=color)
    return image


def render_png(path: Path, size: tuple[int, int]) -> Image.Image:
    tile = checkerboard(size)
    with Image.open(path).convert("RGBA") as image:
        image.thumbnail((size[0] - 10, size[1] - 34), Image.Resampling.LANCZOS)
        x = (size[0] - image.width) // 2
        y = 4
        tile.paste(image, (x, y), image)
    return tile


def make_contact_sheet(rows: list[dict[str, str]], output_dir: Path, max_items: int) -> None:
    if not rows:
        return
    thumb_w, thumb_h = 190, 160
    cols = 4
    shown = rows[:max_items]
    sheet_h = ((len(shown) + cols - 1) // cols) * thumb_h
    sheet = Image.new("RGB", (cols * thumb_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(shown):
        output_path = ROOT / row["output"]
        thumb = render_png(output_path, (thumb_w, thumb_h))
        x = (index % cols) * thumb_w
        y = (index // cols) * thumb_h
        sheet.paste(thumb, (x, y))
        label = f"{Path(row['output']).stem[:22]} {row['flags'][:30]}"
        draw.rectangle((x, y + thumb_h - 30, x + thumb_w, y + thumb_h), fill=(255, 255, 255))
        draw.text((x + 4, y + thumb_h - 26), label, fill="black")
    sheet.save(output_dir / "suspect_contact.jpg", quality=92)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    inputs_dir = (ROOT / args.inputs).resolve()
    outputs_dir = (ROOT / args.outputs).resolve()
    audit_dir = (ROOT / args.out).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for output_path in sorted(outputs_dir.glob("*.png")):
        input_path = find_input(inputs_dir, output_path.stem)
        metrics = alpha_metrics(output_path, args.alpha_threshold)
        flags = flags_for(metrics)
        row = {
            "output": str(output_path.relative_to(ROOT)),
            "input": str(input_path.relative_to(ROOT)) if input_path else "",
            "flags": ";".join(flags),
            **{key: f"{value:.4f}" if isinstance(value, float) else str(value) for key, value in metrics.items()},
        }
        rows.append(row)

    suspects = [row for row in rows if row["flags"]]
    write_csv(audit_dir / "all_outputs.csv", rows)
    write_csv(audit_dir / "suspects.csv", suspects)
    make_contact_sheet(suspects, audit_dir, args.max_contact)

    print(f"Scored {len(rows)} PicWish outputs")
    print(f"Flagged {len(suspects)} likely suspects")
    print(f"Audit folder: {audit_dir}")
    if suspects:
        print(f"Suspect CSV: {audit_dir / 'suspects.csv'}")
        print(f"Suspect contact sheet: {audit_dir / 'suspect_contact.jpg'}")


if __name__ == "__main__":
    main()
