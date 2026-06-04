from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from generate_synthetic_fan_dataset import TARGET_KHR, note_alpha


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "khr_nbc_curated_manifest.csv"
DEFAULT_OUT = ROOT / "data" / "asset_candidates" / "khr_nbc_current_cutout_bank_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a transparent current-KHR cutout bank from curated NBC reference assets."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bucket", action="append", help="Curated bucket to include. Repeat for optional buckets.")
    parser.add_argument("--clean", action="store_true", help="Delete the existing output directory first.")
    return parser.parse_args()


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (ROOT / "data" / "asset_candidates").resolve()
    if allowed_root not in resolved.parents and resolved != allowed_root:
        raise SystemExit(f"Refusing to clean outside {allowed_root}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def make_contact_sheet(paths: list[Path], out_path: Path, title: str) -> None:
    if not paths:
        return
    thumb_w = 220
    label_h = 34
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
        draw.text((x + 4, y + thumb.height + 4), path.name[:30], fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def build_bank(rows: list[dict[str, str]], buckets: set[str], out_dir: Path) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for row in rows:
        if row.get("bucket") not in buckets:
            continue
        denomination = row.get("denomination", "")
        class_name = TARGET_KHR.get(denomination)
        if not class_name:
            continue
        source = ROOT / row["curated_path"]
        with Image.open(source) as raw:
            cutout = note_alpha(raw)
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
        selected.append(
            {
                **row,
                **metrics,
                "class_name": class_name,
                "asset_path": str(asset_path.relative_to(ROOT)),
                "mask_path": str(mask_path.relative_to(ROOT)),
            }
        )
    return selected


def main() -> None:
    args = parse_args()
    manifest = (ROOT / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve()
    out_dir = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(manifest)
    selected = build_bank(rows, set(args.bucket or ["target_modern_common"]), out_dir)
    write_rows(out_dir / "manifest.csv", selected)
    make_contact_sheet([ROOT / row["asset_path"] for row in selected], out_dir / "contact_sheet.jpg", "Current KHR NBC cutouts")

    print(f"Wrote {len(selected)} cutouts to {out_dir}")
    for class_name in sorted(set(row["class_name"] for row in selected)):
        print(f"{class_name}: {sum(1 for row in selected if row['class_name'] == class_name)}")


if __name__ == "__main__":
    main()
