from __future__ import annotations

import argparse
import csv
import shutil
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:  # pragma: no cover - optional local QA enhancement
    cv2 = None


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score transparent banknote cutouts after background removal.")
    parser.add_argument("--input", default="data/asset_candidates/picwish_output", help="Folder of transparent outputs.")
    parser.add_argument("--out", default="data/asset_candidates/cutout_scored", help="Scored output folder.")
    parser.add_argument("--alpha-threshold", type=int, default=16)
    return parser.parse_args()


def connected_components(mask: np.ndarray) -> list[int]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            count = 0
            queue: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                count += 1
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((ny, nx))
            sizes.append(count)
    return sorted(sizes, reverse=True)


def contour_shape_metrics(mask: np.ndarray) -> dict[str, str]:
    if cv2 is None:
        return {
            "convex_fill_ratio": "",
            "rotated_rect_fill_ratio": "",
            "approx_vertices": "",
        }

    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "convex_fill_ratio": "0",
            "rotated_rect_fill_ratio": "0",
            "approx_vertices": "0",
        }

    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    rect = cv2.minAreaRect(contour)
    rect_width, rect_height = rect[1]
    rect_area = float(rect_width * rect_height)
    perimeter = float(cv2.arcLength(hull, True))
    approx = cv2.approxPolyDP(hull, 0.025 * perimeter, True) if perimeter > 0 else hull

    return {
        "convex_fill_ratio": f"{contour_area / hull_area:.4f}" if hull_area > 0 else "0",
        "rotated_rect_fill_ratio": f"{contour_area / rect_area:.4f}" if rect_area > 0 else "0",
        "approx_vertices": str(int(len(approx))),
    }


def span_metrics(mask: np.ndarray, left: int, top: int, right: int, bottom: int) -> dict[str, str]:
    cropped = mask[top:bottom, left:right]
    bbox_width = max(1, right - left)
    bbox_height = max(1, bottom - top)
    row_spans: list[float] = []
    for row in cropped:
        xs = np.where(row)[0]
        if len(xs):
            row_spans.append((int(xs.max()) - int(xs.min()) + 1) / bbox_width)
    col_spans: list[float] = []
    for col in cropped.T:
        ys = np.where(col)[0]
        if len(ys):
            col_spans.append((int(ys.max()) - int(ys.min()) + 1) / bbox_height)
    return {
        "row_span_p10": f"{np.percentile(row_spans, 10):.4f}" if row_spans else "0",
        "col_span_p10": f"{np.percentile(col_spans, 10):.4f}" if col_spans else "0",
    }


def score_image(path: Path, alpha_threshold: int) -> dict[str, str]:
    with Image.open(path).convert("RGBA") as image:
        alpha = np.array(image.getchannel("A"))
    mask = alpha > alpha_threshold
    area = int(mask.sum())
    total = int(mask.size)
    if area == 0:
        return {"verdict": "reject", "reason": "empty_alpha", "alpha_area": "0"}

    ys, xs = np.where(mask)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    bbox_area = max(1, (right - left) * (bottom - top))
    fill_ratio = area / bbox_area
    image_area_ratio = area / total
    aspect = (right - left) / max(1, bottom - top)
    shape = contour_shape_metrics(mask)
    spans = span_metrics(mask, left, top, right, bottom)

    comps = connected_components(mask)
    largest = comps[0]
    largest_ratio = largest / area
    small_component_count = sum(1 for size in comps[1:] if size > max(30, area * 0.005))

    if largest_ratio < 0.86:
        verdict, reason = "reject", "multiple_large_components"
    elif fill_ratio < 0.38:
        verdict, reason = "reject", "too_non_rectangular"
    elif aspect < 0.8 or aspect > 4.2:
        verdict, reason = "review", "odd_aspect_ratio"
    elif small_component_count > 2:
        verdict, reason = "review", "extra_components"
    elif fill_ratio < 0.58:
        verdict, reason = "review", "partial_or_occluded"
    else:
        verdict, reason = "gold", "rectangular_cutout"

    return {
        "verdict": verdict,
        "reason": reason,
        "alpha_area": str(area),
        "image_area_ratio": f"{image_area_ratio:.4f}",
        "bbox_xyxy": f"{left} {top} {right} {bottom}",
        "bbox_fill_ratio": f"{fill_ratio:.4f}",
        "bbox_aspect": f"{aspect:.4f}",
        "component_count": str(len(comps)),
        "largest_component_ratio": f"{largest_ratio:.4f}",
        "small_component_count": str(small_component_count),
        **shape,
        **spans,
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(rows: list[dict[str, str]], out_dir: Path) -> None:
    for verdict in ["gold", "review", "reject"]:
        selected = [row for row in rows if row["verdict"] == verdict][:80]
        if not selected:
            continue
        cols, cell_w, cell_h = 5, 220, 170
        sheet = Image.new("RGB", (cols * cell_w, ((len(selected) + cols - 1) // cols) * (cell_h + 30) + 34), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"{verdict} transparent cutouts", fill="black")
        for index, row in enumerate(selected):
            path = ROOT / row["path"]
            with Image.open(path).convert("RGBA") as image:
                bg = Image.new("RGBA", image.size, (245, 245, 245, 255))
                bg.alpha_composite(image)
                bg = bg.convert("RGB")
                bg.thumbnail((cell_w, cell_h))
            x = (index % cols) * cell_w
            y = 34 + (index // cols) * (cell_h + 30)
            sheet.paste(bg, (x, y))
            draw.text((x + 4, y + cell_h + 4), Path(row["path"]).name[:31], fill="black")
        sheet.save(out_dir / f"{verdict}_contact.jpg", quality=92)


def main() -> None:
    args = parse_args()
    input_dir = (ROOT / args.input).resolve()
    out_dir = (ROOT / args.out).resolve()
    rows: list[dict[str, str]] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        metrics = score_image(path, args.alpha_threshold)
        target = out_dir / metrics["verdict"] / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        rows.append({"path": str(path.relative_to(ROOT)), "scored_path": str(target.relative_to(ROOT)), **metrics})

    write_csv(out_dir / "cutout_scores.csv", rows)
    make_contact_sheet(rows, out_dir)
    print(f"Scored {len(rows)} transparent cutouts into {out_dir}")
    for verdict in ["gold", "review", "reject"]:
        print(f"{verdict}: {sum(1 for row in rows if row['verdict'] == verdict)}")


if __name__ == "__main__":
    main()
