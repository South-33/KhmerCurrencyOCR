"""Build transparent banknote cutouts from the local Numista raw scan cache."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from generate_synthetic_fan_dataset import CLASS_NAMES, TARGET_KHR, note_alpha


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = ROOT / "data" / "numista_raw" / "metadata.json"
DEFAULT_OUT = ROOT / "data" / "asset_candidates" / "numista_current_cutout_bank_v1"
USD_VALUES = {"1", "5", "10", "20", "50", "100"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a scan-based transparent cutout bank from Numista raw assets.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--country", choices=["Cambodia", "United States", "all"], default="all")
    parser.add_argument("--include-out-of-circulation", action="store_true")
    parser.add_argument("--classes", default="", help="Optional comma-separated canonical CashSnap classes to include.")
    parser.add_argument("--min-year", type=int, help="Skip notes whose parsed max year is below this year.")
    parser.add_argument(
        "--alpha-mode",
        choices=["scan-rectangle", "content"],
        default="scan-rectangle",
        help="Use full rectangular alpha for tight scans, or content-derived alpha for non-tight source images.",
    )
    parser.add_argument("--clean", action="store_true", help="Delete the existing output directory first.")
    return parser.parse_args()


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (ROOT / "data" / "asset_candidates").resolve()
    if allowed_root not in resolved.parents and resolved != allowed_root:
        raise SystemExit(f"Refusing to clean outside {allowed_root}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def class_name_for(row: dict[str, object]) -> str | None:
    country = str(row.get("country", ""))
    denomination = str(row.get("denomination", "")).strip()
    if country == "Cambodia":
        return TARGET_KHR.get(denomination)
    if country == "United States" and denomination in USD_VALUES:
        class_name = f"USD_{denomination}"
        return class_name if class_name in CLASS_NAMES else None
    return None


def max_year_for(row: dict[str, object]) -> int | None:
    fields = [str(row.get("years", "")), str(row.get("title", ""))]
    features = row.get("features", {})
    if isinstance(features, dict):
        fields.extend(
            str(value)
            for key, value in features.items()
            if "year" in str(key).lower() or "date" in str(key).lower()
        )
    years = [int(value) for field in fields for value in re.findall(r"\b(19\d{2}|20\d{2})\b", field)]
    return max(years) if years else None


def alpha_metrics(image: Image.Image) -> dict[str, str]:
    alpha = np.asarray(image.getchannel("A"))
    mask = alpha > 16
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {
            "alpha_area": "0",
            "bbox_xyxy": "",
            "bbox_fill_ratio": "0.0000",
            "width": str(image.width),
            "height": str(image.height),
        }
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    return {
        "alpha_area": str(int(mask.sum())),
        "bbox_xyxy": f"{x1} {y1} {x2} {y2}",
        "bbox_fill_ratio": f"{mask.sum() / bbox_area:.4f}",
        "width": str(image.width),
        "height": str(image.height),
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(paths: list[Path], out_path: Path, title: str) -> None:
    if not paths:
        return
    thumb_w = 240
    label_h = 36
    thumbs: list[tuple[Path, Image.Image]] = []
    for path in paths:
        with Image.open(path).convert("RGBA") as image:
            bg = Image.new("RGBA", image.size, (245, 245, 245, 255))
            bg.alpha_composite(image)
            ratio = thumb_w / max(1, image.width)
            thumb = bg.convert("RGB").resize((thumb_w, max(1, int(image.height * ratio))))
            thumbs.append((path, thumb))
    cols = 3
    row_h = max(thumb.height for _, thumb in thumbs) + label_h
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * row_h + 42), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    for index, (path, thumb) in enumerate(thumbs):
        x = (index % cols) * thumb_w
        y = 42 + (index // cols) * row_h
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb.height + 4), path.name[:34], fill="black")
    sheet.save(out_path, quality=92)


def normalize_source_path(metadata_path: Path, relative_path: str) -> Path:
    return metadata_path.parent / Path(relative_path.replace("\\", "/"))


def scan_rectangle_alpha(image: Image.Image) -> Image.Image:
    cutout = image.convert("RGBA")
    cutout.putalpha(Image.new("L", cutout.size, 255))
    return cutout


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata if args.metadata.is_absolute() else ROOT / args.metadata
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    metadata_path = metadata_path.resolve()
    out_dir = out_dir.resolve()

    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    allowed_classes = {item.strip() for item in args.classes.split(",") if item.strip()} or None
    if allowed_classes:
        unknown = sorted(allowed_classes - set(CLASS_NAMES))
        if unknown:
            raise SystemExit(f"Unknown classes in --classes: {unknown}")

    rows: list[dict[str, str]] = []
    for note_id, note in sorted(metadata.items()):
        if args.country != "all" and note.get("country") != args.country:
            continue
        if not args.include_out_of_circulation and note.get("circulation_status") != "in_circulation":
            continue
        class_name = class_name_for(note)
        if not class_name:
            continue
        if allowed_classes and class_name not in allowed_classes:
            continue
        note_year = max_year_for(note)
        if args.min_year is not None and (note_year is None or note_year < args.min_year):
            continue
        files = note.get("files", {})
        if not isinstance(files, dict):
            continue
        for side in ["front", "back"]:
            rel_source = files.get(side)
            if not rel_source:
                continue
            source = normalize_source_path(metadata_path, str(rel_source))
            if not source.exists():
                continue
            with Image.open(source) as raw:
                if args.alpha_mode == "content":
                    cutout = note_alpha(raw)
                else:
                    cutout = scan_rectangle_alpha(raw)
            metrics = alpha_metrics(cutout)
            if metrics["alpha_area"] == "0":
                continue
            stem = source.stem
            asset_path = out_dir / class_name / f"{stem}.png"
            mask_path = out_dir / "masks" / class_name / f"{stem}_mask.png"
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cutout.save(asset_path)
            alpha = cutout.getchannel("A").point(lambda value: 255 if value > 16 else 0)
            alpha.save(mask_path)
            rows.append(
                {
                    **metrics,
                    "asset_path": str(asset_path.relative_to(ROOT)),
                    "class_name": class_name,
                    "country": str(note.get("country", "")),
                    "currency": str(note.get("currency", "")),
                    "denomination": str(note.get("denomination", "")),
                    "mask_path": str(mask_path.relative_to(ROOT)),
                    "note_id": str(note_id),
                    "side": side,
                    "source_path": str(source.relative_to(ROOT)),
                    "status": str(note.get("circulation_status", "")),
                    "title": str(note.get("title", "")),
                    "years": str(note.get("years", "")),
                    "max_year": str(note_year or ""),
                }
            )

    write_rows(out_dir / "manifest.csv", rows)
    make_contact_sheet(
        [ROOT / row["asset_path"] for row in rows],
        out_dir / "contact_sheet.jpg",
        "Numista cutout bank",
    )

    print(f"Wrote {len(rows)} cutouts to {out_dir}")
    for class_name in sorted({row["class_name"] for row in rows}):
        print(f"{class_name}: {sum(1 for row in rows if row['class_name'] == class_name)}")


if __name__ == "__main__":
    main()
