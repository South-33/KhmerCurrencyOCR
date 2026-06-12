#!/usr/bin/env python
"""Render source/raw/locked refiner comparisons into a contact sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--locked-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=18)
    parser.add_argument("--thumb-width", type=int, default=220)
    parser.add_argument("--caption-height", type=int, default=48)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise SystemExit(f"{repo_rel(path)}:{line_no} is not a JSON object")
        rows.append(row)
    return rows


def audit_rejections(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rejected: set[str] = set()
    for record in payload.get("records", []):
        if not record.get("accepted", False):
            rejected.add(Path(str(record.get("image", ""))).stem)
    return rejected


def image_for(root: Path, row_id: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = root / f"{row_id}{suffix}"
        if candidate.exists():
            return candidate
    raise SystemExit(f"missing refiner image for {row_id} under {repo_rel(root)}")


def fit_image(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as loaded:
        image = ImageOps.exif_transpose(loaded.convert("RGB"))
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height), "white")
    x = (width - image.width) // 2
    y = (height - image.height) // 2
    tile.paste(image, (x, y))
    return tile


def caption(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    draw.text((x, y), text[:42], fill=(20, 20, 20), font=font)


def main() -> int:
    args = parse_args()
    if args.max_rows < 1:
        raise SystemExit("--max-rows must be >= 1")
    manifest = resolve(args.manifest)
    raw_root = resolve(args.raw_root)
    locked_root = resolve(args.locked_root)
    out = resolve(args.out)
    rejected = audit_rejections(resolve(args.audit_json) if args.audit_json else None)
    rows = read_jsonl(manifest)
    rows = sorted(rows, key=lambda row: (0 if str(row.get("id", "")) in rejected else 1, str(row.get("id", ""))))
    rows = rows[: args.max_rows]
    if not rows:
        raise SystemExit("manifest has no rows")

    thumb_w = args.thumb_width
    thumb_h = args.thumb_width
    cap_h = args.caption_height
    gap = 10
    cols = 3
    row_w = cols * thumb_w + (cols + 1) * gap
    row_h = thumb_h + cap_h + gap
    header_h = 30
    sheet = Image.new("RGB", (row_w, header_h + len(rows) * row_h + gap), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((gap, 8), "source | raw SD-Turbo | note-edge locked", fill=(0, 0, 0), font=font)

    for idx, row in enumerate(rows):
        row_id = str(row["id"])
        y = header_h + idx * row_h + gap
        source = resolve(row["source_image"])
        raw = image_for(raw_root, row_id)
        locked = image_for(locked_root, row_id)
        border = (210, 30, 30) if row_id in rejected else (180, 180, 180)
        labels = [
            f"source {row_id}",
            "raw",
            "locked REJECT" if row_id in rejected else "locked",
        ]
        for col, (path, label) in enumerate(zip([source, raw, locked], labels, strict=True)):
            x = gap + col * (thumb_w + gap)
            tile = fit_image(path, thumb_w, thumb_h)
            sheet.paste(tile, (x, y))
            draw.rectangle((x, y, x + thumb_w - 1, y + thumb_h - 1), outline=border, width=2)
            caption(draw, (x, y + thumb_h + 4), label, font)

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)
    print(f"visual_sheet={repo_rel(out)} rows={len(rows)} rejected_first={len(rejected)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
