"""Audit transparent banknote cutout banks before synthetic generation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "data" / "asset_candidates" / "numista_current_cutout_bank_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a transparent banknote cutout bank.")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK, help="Asset bank root containing a manifest.csv.")
    parser.add_argument("--out", type=Path, default=None, help="Audit output directory. Defaults to <bank>/audit.")
    parser.add_argument("--max-contact", type=int, default=80, help="Maximum suspect assets on the contact sheet.")
    return parser.parse_args()


def read_manifest(bank: Path) -> list[dict[str, str]]:
    manifest = bank / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"Missing manifest: {manifest}")
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_asset(row: dict[str, str], bank: Path) -> Path:
    raw = row.get("asset_path") or row.get("path") or row.get("output") or ""
    if not raw:
        raise ValueError("manifest row has no asset_path/path/output")
    path = Path(raw.replace("\\", "/"))
    if path.is_absolute():
        return path
    repo_path = ROOT / path
    if repo_path.exists():
        return repo_path
    return bank / path


def alpha_metrics(path: Path) -> dict[str, float | int | str]:
    try:
        with Image.open(path).convert("RGBA") as image:
            arr = np.asarray(image)
    except OSError:
        return {"status": "unreadable"}
    alpha = arr[:, :, 3]
    mask = alpha > 16
    area = int(mask.sum())
    image_area = int(mask.shape[0] * mask.shape[1])
    if area == 0:
        return {
            "status": "empty",
            "width": int(mask.shape[1]),
            "height": int(mask.shape[0]),
            "alpha_area": 0,
            "alpha_area_ratio": 0.0,
            "bbox_width": 0,
            "bbox_height": 0,
            "bbox_aspect_norm": 0.0,
            "bbox_fill_ratio": 0.0,
            "strong_red_ratio": 0.0,
        }

    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    aspect = bbox_w / max(1, bbox_h)
    aspect_norm = max(aspect, 1 / max(aspect, 1e-6))
    bbox_area = max(1, bbox_w * bbox_h)
    rgb = arr[:, :, :3].astype(np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    strong_red = (red > 125) & ((red - green) > 45) & ((red - blue) > 45) & mask
    return {
        "status": "ok",
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "alpha_area": area,
        "alpha_area_ratio": area / max(1, image_area),
        "bbox_width": bbox_w,
        "bbox_height": bbox_h,
        "bbox_aspect_norm": aspect_norm,
        "bbox_fill_ratio": area / bbox_area,
        "strong_red_ratio": float(strong_red.sum() / max(1, area)),
    }


def flags_for(row: dict[str, str], metrics: dict[str, float | int | str]) -> list[str]:
    if metrics.get("status") != "ok":
        return [str(metrics.get("status", "bad_image"))]
    flags: list[str] = []
    name_blob = " ".join(
        [
            row.get("asset_path", ""),
            row.get("source_path", ""),
            row.get("title", ""),
        ]
    ).lower()
    if "specimen" in name_blob or "watermark" in name_blob:
        flags.append("name_indicates_specimen_or_watermark")
    if float(metrics["bbox_aspect_norm"]) < 1.45 or float(metrics["bbox_aspect_norm"]) > 3.65:
        flags.append("bad_note_aspect")
    if float(metrics["bbox_fill_ratio"]) < 0.72:
        flags.append("ragged_or_partial_alpha")
    if float(metrics["alpha_area_ratio"]) < 0.10:
        flags.append("tiny_foreground")
    if float(metrics["strong_red_ratio"]) > 0.12:
        flags.append("large_red_mark_suspect")
    return flags


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def year_value(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def checkerboard(size: tuple[int, int], cell: int = 12) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            color = (226, 226, 226) if (x // cell + y // cell) % 2 else (248, 248, 248)
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=color)
    return image


def render_asset(path: Path, size: tuple[int, int]) -> Image.Image:
    tile = checkerboard(size)
    with Image.open(path).convert("RGBA") as image:
        image.thumbnail((size[0] - 10, size[1] - 42), Image.Resampling.LANCZOS)
        tile.paste(image, ((size[0] - image.width) // 2, 5), image)
    return tile


def contact_sheet(rows: list[dict[str, str]], out_path: Path, max_items: int) -> None:
    shown = rows[:max_items]
    if not shown:
        return
    thumb_w, thumb_h = 230, 170
    cols = 3
    sheet = Image.new("RGB", (cols * thumb_w, ((len(shown) + cols - 1) // cols) * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(shown):
        path = Path(row["resolved_asset_path"])
        thumb = render_asset(path, (thumb_w, thumb_h))
        x = (index % cols) * thumb_w
        y = (index // cols) * thumb_h
        sheet.paste(thumb, (x, y))
        label = f"{row.get('class_name', '')} {row.get('side', '')} {row['flags'][:28]}"
        draw.rectangle((x, y + thumb_h - 34, x + thumb_w, y + thumb_h), fill=(255, 255, 255))
        draw.text((x + 4, y + thumb_h - 30), label, fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    bank = args.bank if args.bank.is_absolute() else ROOT / args.bank
    bank = bank.resolve()
    out_dir = args.out if args.out else bank / "audit"
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for row in read_manifest(bank):
        asset = resolve_asset(row, bank).resolve()
        metrics = alpha_metrics(asset)
        flags = flags_for(row, metrics)
        rows.append(
            {
                **row,
                "source_status": row.get("source_status", row.get("status", "")),
                **{key: f"{value:.6f}" if isinstance(value, float) else str(value) for key, value in metrics.items()},
                "resolved_asset_path": str(asset),
                "flags": ";".join(flags),
            }
        )

    suspects = [row for row in rows if row["flags"]]
    write_csv(out_dir / "all_assets.csv", rows)
    write_csv(out_dir / "suspects.csv", suspects)
    contact_sheet(suspects, out_dir / "suspect_contact.jpg", args.max_contact)

    class_counts = Counter(row.get("class_name", "unknown") for row in rows)
    side_counts = Counter((row.get("class_name", "unknown"), row.get("side", "unknown")) for row in rows)
    flag_counts = Counter(flag for row in suspects for flag in row["flags"].split(";") if flag)
    source_status_counts = Counter(row.get("source_status", "unknown") for row in rows)
    currency_counts = Counter(row.get("currency", "unknown") for row in rows)
    side_coverage = {}
    missing_class_sides = []
    for class_name in CLASS_NAMES:
        front = side_counts[(class_name, "front")]
        back = side_counts[(class_name, "back")]
        side_coverage[class_name] = {"front": front, "back": back}
        if front == 0:
            missing_class_sides.append(f"{class_name}/front")
        if back == 0:
            missing_class_sides.append(f"{class_name}/back")
    years_by_class: dict[str, dict[str, int | None]] = {}
    for class_name in CLASS_NAMES:
        years = [
            year
            for row in rows
            if row.get("class_name") == class_name
            for year in [year_value(row.get("max_year", ""))]
            if year is not None
        ]
        years_by_class[class_name] = {
            "min_max_year": min(years) if years else None,
            "max_max_year": max(years) if years else None,
        }
    write_json(
        out_dir / "summary.json",
        {
            "bank": str(bank),
            "assets": len(rows),
            "suspects": len(suspects),
            "class_counts": dict(sorted(class_counts.items())),
            "side_coverage": side_coverage,
            "missing_class_sides": missing_class_sides,
            "source_status_counts": dict(sorted(source_status_counts.items())),
            "currency_counts": dict(sorted(currency_counts.items())),
            "flag_counts": dict(sorted(flag_counts.items())),
            "years_by_class": years_by_class,
            "outputs": {
                "all_assets_csv": str((out_dir / "all_assets.csv").relative_to(ROOT)),
                "suspects_csv": str((out_dir / "suspects.csv").relative_to(ROOT)),
                "suspect_contact": str((out_dir / "suspect_contact.jpg").relative_to(ROOT)),
            },
            "policy": {
                "missing_class_sides": "must be empty before a full-schema trainable recipe can claim complete class/side coverage",
                "suspects": "review contact sheet before using suspect assets in trainable recipes",
                "large_red_mark_suspect": "KHR red/pink design areas can be false positives; review visually before excluding",
            },
        },
    )

    print(f"bank: {bank}")
    print(f"assets: {len(rows)}")
    print(f"suspects: {len(suspects)}")
    print(f"classes: {dict(sorted(class_counts.items()))}")
    print(f"sides: {dict(sorted((f'{klass}/{side}', count) for (klass, side), count in side_counts.items()))}")
    print(f"flags: {dict(sorted(flag_counts.items()))}")
    print(f"audit: {out_dir}")
    print(f"summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
