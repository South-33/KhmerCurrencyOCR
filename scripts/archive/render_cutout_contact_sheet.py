from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a filtered contact sheet for transparent cutout QA.")
    parser.add_argument("--manifest", required=True, help="CSV with path/scored_path/selected_path columns.")
    parser.add_argument("--out", required=True, help="Output contact sheet image.")
    parser.add_argument("--path-column", default="", help="CSV path column. Defaults to selected_path/scored_path/path.")
    parser.add_argument("--label-contains", default="", help="Only include rows whose path contains this text.")
    parser.add_argument("--verdict", default="", help="Only include rows with this verdict.")
    parser.add_argument("--max-items", type=int, default=120)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=220)
    return parser.parse_args()


def choose_path_column(rows: list[dict[str, str]], explicit: str) -> str:
    if explicit:
        return explicit
    for column in ["selected_path", "scored_path", "path", "crop_path", "asset_path"]:
        if rows and column in rows[0]:
            return column
    raise SystemExit("Could not infer a path column.")


def checkerboard(size: tuple[int, int], cell: int = 12) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            color = (226, 226, 226) if (x // cell + y // cell) % 2 else (248, 248, 248)
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=color)
    return image


def render_cutout(path: Path, thumb_width: int) -> Image.Image:
    with Image.open(path).convert("RGBA") as image:
        ratio = thumb_width / max(1, image.width)
        thumb_height = max(1, int(image.height * ratio))
        image = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        tile = checkerboard(image.size)
        tile.paste(image, (0, 0), image)
        return tile


def main() -> None:
    args = parse_args()
    manifest = (ROOT / args.manifest).resolve()
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    path_column = choose_path_column(rows, args.path_column)
    selected = []
    for row in rows:
        row_path = row.get(path_column, "")
        if not row_path:
            continue
        if args.label_contains and args.label_contains not in row_path:
            continue
        if args.verdict and row.get("verdict") != args.verdict:
            continue
        selected.append(row)
    selected = selected[: args.max_items]
    if not selected:
        raise SystemExit("No rows matched the requested filters.")

    label_h = 34
    thumbs = [(row, render_cutout(ROOT / row[path_column], args.thumb_width)) for row in selected]
    row_h = max(thumb.height for _, thumb in thumbs) + label_h
    rows_needed = (len(thumbs) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (args.cols * args.thumb_width, rows_needed * row_h + 42), "white")
    draw = ImageDraw.Draw(sheet)
    title = f"{manifest.name} {args.verdict or 'all'} {args.label_contains or ''}".strip()
    draw.text((8, 8), title, fill="black")

    for index, (row, thumb) in enumerate(thumbs):
        x = (index % args.cols) * args.thumb_width
        y = 42 + (index // args.cols) * row_h
        sheet.paste(thumb, (x, y))
        label = Path(row[path_column]).name[:31]
        draw.text((x + 4, y + thumb.height + 4), label, fill="black")

    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)
    print(f"Wrote {len(selected)} rows to {out}")


if __name__ == "__main__":
    main()
