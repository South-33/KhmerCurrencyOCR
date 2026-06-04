from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    "KHR_50000",
    "KHR_20000",
    "KHR_10000",
    "KHR_5000",
    "KHR_2000",
    "KHR_1000",
    "KHR_500",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compositor cutout bank from scored transparent PNGs.")
    parser.add_argument("--scores", required=True, help="cutout_scores.csv or selected_cutouts.csv.")
    parser.add_argument("--out", required=True, help="Output asset bank folder.")
    parser.add_argument("--path-column", default="", help="CSV path column. Defaults to selected_path/scored_path/path.")
    parser.add_argument("--verdict", default="gold", help="Comma-separated verdicts to promote.")
    parser.add_argument("--min-fill", type=float, default=0.65)
    parser.add_argument("--min-aspect-norm", type=float, default=1.55)
    parser.add_argument("--max-aspect-norm", type=float, default=3.35)
    parser.add_argument("--min-largest-component", type=float, default=0.95)
    parser.add_argument("--max-small-components", type=int, default=0)
    parser.add_argument("--min-convex-fill", type=float, default=0.0)
    parser.add_argument("--min-rotated-fill", type=float, default=0.0)
    parser.add_argument("--min-row-span-p10", type=float, default=0.0)
    parser.add_argument("--min-col-span-p10", type=float, default=0.0)
    parser.add_argument("--max-approx-vertices", type=int, default=0, help="0 disables this filter.")
    parser.add_argument(
        "--max-skin-ratio",
        type=float,
        default=1.0,
        help="Maximum skin-like foreground pixel ratio after trimming. Lower values reject hand-contaminated cutouts.",
    )
    parser.add_argument("--pad", type=int, default=3, help="Padding around alpha trim box.")
    parser.add_argument("--clean", action="store_true", help="Delete the output folder first.")
    return parser.parse_args()


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (ROOT / "data" / "asset_candidates").resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise SystemExit(f"Refusing to clean outside {allowed_root}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def choose_path_column(rows: list[dict[str, str]], explicit: str) -> str:
    if explicit:
        return explicit
    for column in ["selected_path", "scored_path", "path"]:
        if rows and column in rows[0]:
            return column
    raise SystemExit("Could not infer a transparent image path column.")


def parse_class(path_text: str) -> str | None:
    normalized = path_text.replace("\\", "/")
    for class_name in CLASS_NAMES:
        if re.search(rf"(^|[/_]){re.escape(class_name)}([_/]|$)", normalized):
            return class_name
    return None


def aspect_norm(row: dict[str, str]) -> float:
    aspect = float(row.get("bbox_aspect", "0") or 0)
    if aspect <= 0:
        return 0.0
    return max(aspect, 1.0 / aspect)


def row_passes(row: dict[str, str], verdicts: set[str], args: argparse.Namespace) -> bool:
    if row.get("verdict") not in verdicts:
        return False
    fill = float(row.get("bbox_fill_ratio", "0") or 0)
    largest = float(row.get("largest_component_ratio", "0") or 0)
    small_components = int(row.get("small_component_count", "99") or 99)
    norm = aspect_norm(row)
    convex_fill = float(row.get("convex_fill_ratio", "1") or 1)
    rotated_fill = float(row.get("rotated_rect_fill_ratio", "1") or 1)
    row_span_p10 = float(row.get("row_span_p10", "1") or 1)
    col_span_p10 = float(row.get("col_span_p10", "1") or 1)
    approx_vertices = int(float(row.get("approx_vertices", "0") or 0))
    return (
        fill >= args.min_fill
        and args.min_aspect_norm <= norm <= args.max_aspect_norm
        and largest >= args.min_largest_component
        and small_components <= args.max_small_components
        and convex_fill >= args.min_convex_fill
        and rotated_fill >= args.min_rotated_fill
        and row_span_p10 >= args.min_row_span_p10
        and col_span_p10 >= args.min_col_span_p10
        and (args.max_approx_vertices <= 0 or approx_vertices <= args.max_approx_vertices)
    )


def trim_alpha(image: Image.Image, pad: int) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0:
        return rgba
    left = max(0, int(xs.min()) - pad)
    top = max(0, int(ys.min()) - pad)
    right = min(rgba.width, int(xs.max()) + pad + 1)
    bottom = min(rgba.height, int(ys.max()) + pad + 1)
    return rgba.crop((left, top, right, bottom))


def alpha_metrics(image: Image.Image) -> dict[str, str]:
    alpha = np.asarray(image.getchannel("A"))
    mask = alpha > 16
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {"alpha_area": "0", "bbox_xyxy": "", "width": str(image.width), "height": str(image.height)}
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    bbox_area = max(1, (right - left) * (bottom - top))
    return {
        "alpha_area": str(int(mask.sum())),
        "bbox_xyxy": f"{left} {top} {right} {bottom}",
        "bbox_fill_ratio_trimmed": f"{mask.sum() / bbox_area:.4f}",
        "width": str(image.width),
        "height": str(image.height),
    }


def skin_like_ratio(image: Image.Image) -> float:
    arr = np.asarray(image.convert("RGBA"))
    alpha_mask = arr[:, :, 3] > 16
    if not alpha_mask.any():
        return 0.0
    rgb = arr[:, :, :3].astype(np.int16)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    skin = (
        (red > 95)
        & (green > 40)
        & (blue > 20)
        & ((max_channel - min_channel) > 15)
        & (np.abs(red - green) > 15)
        & (red > green)
        & (red > blue)
    )
    return float((skin & alpha_mask).sum() / max(1, alpha_mask.sum()))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(rows: list[dict[str, str]], out_path: Path) -> None:
    if not rows:
        return
    thumb_w, label_h = 220, 34
    thumbs: list[tuple[dict[str, str], Image.Image]] = []
    for row in rows[:120]:
        with Image.open(ROOT / row["asset_path"]).convert("RGBA") as image:
            bg = Image.new("RGBA", image.size, (245, 245, 245, 255))
            bg.alpha_composite(image)
            ratio = thumb_w / max(1, image.width)
            thumb = bg.convert("RGB").resize((thumb_w, max(1, int(image.height * ratio))))
            thumbs.append((row, thumb))
    cols = 5
    row_h = max(thumb.height for _, thumb in thumbs) + label_h
    sheet_h = ((len(thumbs) + cols - 1) // cols) * row_h + 42
    sheet = Image.new("RGB", (cols * thumb_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), out_path.parent.name, fill="black")
    for index, (row, thumb) in enumerate(thumbs):
        x = (index % cols) * thumb_w
        y = 42 + (index // cols) * row_h
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb.height + 4), Path(row["asset_path"]).name[:31], fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    scores_path = (ROOT / args.scores).resolve()
    out_dir = (ROOT / args.out).resolve()
    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with scores_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    path_column = choose_path_column(rows, args.path_column)
    verdicts = {value.strip() for value in args.verdict.split(",") if value.strip()}

    selected: list[dict[str, str]] = []
    rejected_count = 0
    for row in rows:
        source_text = row.get(path_column, "")
        source_class = parse_class(source_text)
        if not source_class or not row_passes(row, verdicts, args):
            rejected_count += 1
            continue
        source = ROOT / source_text
        if not source.exists():
            rejected_count += 1
            continue
        with Image.open(source).convert("RGBA") as raw:
            cutout = trim_alpha(raw, args.pad)
        metrics = alpha_metrics(cutout)
        skin_ratio = skin_like_ratio(cutout)
        if metrics["alpha_area"] == "0" or skin_ratio > args.max_skin_ratio:
            rejected_count += 1
            continue

        target_name = source.name
        asset_path = out_dir / source_class / target_name
        mask_path = out_dir / "masks" / source_class / target_name.replace(".png", "_mask.png")
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cutout.save(asset_path)
        cutout.getchannel("A").point(lambda value: 255 if value > 16 else 0).save(mask_path)
        selected.append(
            {
                **row,
                **metrics,
                "class_name": source_class,
                "source_path": source_text,
                "asset_path": str(asset_path.relative_to(ROOT)),
                "mask_path": str(mask_path.relative_to(ROOT)),
                "aspect_norm": f"{aspect_norm(row):.4f}",
                "skin_like_foreground_ratio": f"{skin_ratio:.4f}",
            }
        )

    write_csv(out_dir / "manifest.csv", selected)
    make_contact_sheet(selected, out_dir / "contact_sheet.jpg")
    print(f"Promoted {len(selected)} cutouts to {out_dir}")
    print(f"Filtered out {rejected_count} rows")
    for class_name in sorted({row["class_name"] for row in selected}):
        print(f"{class_name}: {sum(1 for row in selected if row['class_name'] == class_name)}")


if __name__ == "__main__":
    main()
