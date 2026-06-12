#!/usr/bin/env python
"""Build full-size review overlays/crops for official21 annotation proposals.

This is intentionally a review helper, not a label materializer. It keeps the
materializer-compatible proposal columns, but leaves review_decision blank.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROPOSAL_CSV = Path(
    "runs/cashsnap/real_data_label_audit_v1/"
    "khr100_schema_expansion_annotation_proposals_v1/proposals.csv"
)
REQUIRED_COLUMNS = {
    "image",
    "model",
    "current_pred_class",
    "current_pred_class_id",
    "confidence",
    "area_ratio",
    "x1",
    "y1",
    "x2",
    "y2",
    "proposed_new_class",
    "review_decision",
    "review_notes",
}


@dataclass(frozen=True)
class Proposal:
    source_row: int
    row: dict[str, str]
    confidence: float
    xyxy: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-csv", type=Path, default=DEFAULT_PROPOSAL_CSV)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--max-images",
        type=int,
        default=24,
        help="Maximum selected images. Use 0 to include every image with a valid proposal.",
    )
    parser.add_argument(
        "--select",
        choices=["top_conf_per_image", "input_order_per_image"],
        default="top_conf_per_image",
        help="How to choose one proposal per image before sorting the review queue.",
    )
    parser.add_argument(
        "--sort",
        choices=["confidence_desc", "input_order"],
        default="confidence_desc",
        help="Order of selected image-level proposals.",
    )
    parser.add_argument("--padding", type=float, default=0.08, help="Crop padding as a box-size fraction.")
    parser.add_argument("--max-overlay-side", type=int, default=2400)
    parser.add_argument("--sheet-columns", type=int, default=4)
    parser.add_argument("--thumb-size", type=int, default=300)
    parser.add_argument("--clean", action="store_true", help="Remove existing files in out-dir first.")
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def parse_float(row: dict[str, str], key: str, source_row: int) -> float:
    raw = row.get(key, "").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"row {source_row}: {key}={raw!r} is not numeric") from exc
    if not math.isfinite(value):
        raise SystemExit(f"row {source_row}: {key}={raw!r} is not finite")
    return value


def load_proposals(path: Path) -> list[Proposal]:
    resolved = resolve(path)
    with resolved.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{repo_rel(resolved)} has no header")
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise SystemExit(f"{repo_rel(resolved)} missing required columns: {missing}")
        proposals: list[Proposal] = []
        for source_row, row in enumerate(reader, start=2):
            confidence = parse_float(row, "confidence", source_row)
            xyxy = tuple(parse_float(row, key, source_row) for key in ["x1", "y1", "x2", "y2"])
            x1, y1, x2, y2 = xyxy
            if x2 <= x1 or y2 <= y1:
                continue
            proposals.append(Proposal(source_row=source_row, row=dict(row), confidence=confidence, xyxy=xyxy))
    if not proposals:
        raise SystemExit(f"{repo_rel(resolved)} has no valid proposals")
    return proposals


def choose_per_image(proposals: Iterable[Proposal], select: str, sort_mode: str) -> list[Proposal]:
    grouped: dict[str, list[Proposal]] = defaultdict(list)
    first_seen: dict[str, int] = {}
    for index, proposal in enumerate(proposals):
        image = proposal.row["image"]
        grouped[image].append(proposal)
        first_seen.setdefault(image, index)

    selected: list[Proposal] = []
    for image, rows in grouped.items():
        if select == "top_conf_per_image":
            chosen = max(rows, key=lambda item: (item.confidence, -item.source_row))
        else:
            chosen = rows[0]
        selected.append(chosen)

    if sort_mode == "confidence_desc":
        selected.sort(key=lambda item: (-item.confidence, first_seen[item.row["image"]], item.source_row))
    else:
        selected.sort(key=lambda item: (first_seen[item.row["image"]], item.source_row))
    return selected


def clamp_box(
    xyxy: tuple[float, float, float, float],
    size: tuple[int, int],
    padding: float = 0.0,
) -> tuple[int, int, int, int]:
    width, height = size
    x1, y1, x2, y2 = xyxy
    pad_x = max(0.0, (x2 - x1) * padding)
    pad_y = max(0.0, (y2 - y1) * padding)
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(width, int(math.ceil(x2 + pad_x)))
    bottom = min(height, int(math.ceil(y2 + pad_y)))
    if right <= left or bottom <= top:
        raise ValueError("box outside image after clamping")
    return left, top, right, bottom


def scaled_overlay(
    image: Image.Image,
    xyxy: tuple[float, float, float, float],
    label: str,
    max_side: int,
) -> Image.Image:
    overlay = image.convert("RGB")
    original_width, original_height = overlay.size
    scale = min(1.0, max_side / max(original_width, original_height)) if max_side > 0 else 1.0
    if scale < 1.0:
        overlay = overlay.resize(
            (round(original_width * scale), round(original_height * scale)),
            Image.Resampling.LANCZOS,
        )
    x1, y1, x2, y2 = (value * scale for value in xyxy)
    draw = ImageDraw.Draw(overlay)
    font = load_font(max(18, round(max(overlay.size) * 0.018)))
    color = (230, 25, 75)
    width = max(3, round(max(overlay.size) * 0.004))
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_w = right - left
    text_h = bottom - top
    pad = 6
    label_x1 = max(0, min(x1, overlay.width - text_w - pad * 2))
    label_y1 = max(0, y1 - text_h - pad * 2)
    label_x2 = min(overlay.width, label_x1 + text_w + pad * 2)
    label_y2 = min(overlay.height, label_y1 + text_h + pad * 2)
    draw.rectangle((label_x1, label_y1, label_x2, label_y2), fill=color)
    draw.text((label_x1 + pad, label_y1 + pad), label, fill=(255, 255, 255), font=font)
    return overlay


def fit_thumbnail(image: Image.Image, size: int) -> Image.Image:
    tile = Image.new("RGB", (size, size), (255, 255, 255))
    thumb = image.convert("RGB")
    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
    left = (size - thumb.width) // 2
    top = (size - thumb.height) // 2
    tile.paste(thumb, (left, top))
    return tile


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    max_lines: int,
) -> None:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    if words and len(lines) == max_lines:
        consumed = " ".join(lines)
        if len(consumed) < len(text):
            lines[-1] = lines[-1].rstrip(".") + "..."
    x, y = xy
    line_height = max(14, round(font.size * 1.2)) if hasattr(font, "size") else 18
    for index, line in enumerate(lines):
        draw.text((x, y + index * line_height), line, fill=fill, font=font)


def write_sheet(
    rows: list[dict[str, str]],
    image_key: str,
    image_paths: list[Path],
    out_path: Path,
    columns: int,
    thumb_size: int,
) -> None:
    if not rows:
        return
    columns = max(1, columns)
    label_height = 74
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size, sheet_rows * (thumb_size + label_height)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = load_font(15)
    small = load_font(13)
    for index, (row, path) in enumerate(zip(rows, image_paths)):
        col = index % columns
        row_index = index // columns
        x = col * thumb_size
        y = row_index * (thumb_size + label_height)
        with Image.open(path) as image:
            thumb = fit_thumbnail(image, thumb_size)
        sheet.paste(thumb, (x, y))
        draw.rectangle((x, y, x + thumb_size - 1, y + thumb_size + label_height - 1), outline=(180, 180, 180))
        label = f"{row['review_id']} {row['current_pred_class']} {float(row['confidence']):.3f}"
        draw.text((x + 6, y + thumb_size + 6), label, fill=(0, 0, 0), font=font)
        model = row["model"]
        if len(model) > 24:
            model = model[:21] + "..."
        draw.text((x + 6, y + thumb_size + 27), model, fill=(70, 70, 70), font=small)
        image_label = short_image_label(Path(row[image_key]).stem)
        draw.text((x + 6, y + thumb_size + 46), image_label, fill=(50, 50, 50), font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def safe_stem(text: str, fallback: str) -> str:
    stem = Path(text).stem
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem)
    return (safe[:90].strip("_") or fallback)


def short_image_label(stem: str) -> str:
    label = stem
    if "_jpg_rf_" in label:
        label = label.split("_jpg_rf_", 1)[0]
    if "_jpg.rf." in label:
        label = label.split("_jpg.rf.", 1)[0]
    if label.startswith("khmer_us_currency_"):
        label = label.removeprefix("khmer_us_currency_")
    if len(label) > 32:
        label = label[:29] + "..."
    return label


def clean_out_dir(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    allowed_roots = [(ROOT / "runs").resolve(), (ROOT / "data" / "review").resolve()]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise SystemExit(f"--clean refused outside runs/ or data/review: {repo_rel(resolved)}")
    if not resolved.exists():
        return
    for child in resolved.iterdir():
        if child.is_dir():
            for nested in sorted(child.rglob("*"), reverse=True):
                if nested.is_file() or nested.is_symlink():
                    nested.unlink()
                elif nested.is_dir():
                    nested.rmdir()
            child.rmdir()
        else:
            child.unlink()


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def review_prefix(class_names: Iterable[str]) -> str:
    cleaned = sorted({safe_stem(name.strip().lower(), "unknown") for name in class_names if name.strip()})
    if len(cleaned) == 1:
        return cleaned[0]
    return "mixed_official21"


def class_scope_text(class_names: Iterable[str]) -> str:
    cleaned = sorted({name.strip() for name in class_names if name.strip()})
    if not cleaned:
        return "denomination-triage"
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned)


def acceptance_instruction(class_names: Iterable[str]) -> str:
    cleaned = sorted({name.strip() for name in class_names if name.strip()})
    if not cleaned:
        return (
            "- Fill `proposed_new_class` with the verified official21 denomination "
            "before using `accepted_box`."
        )
    return f"- Use `accepted_box` only for boxes that tightly cover one countable {class_scope_text(cleaned)} note."


def main() -> None:
    args = parse_args()
    proposal_csv = resolve(args.proposal_csv)
    out_dir = resolve(args.out_dir)
    if args.max_images < 0:
        raise SystemExit("--max-images must be non-negative")
    if args.padding < 0:
        raise SystemExit("--padding must be non-negative")
    if args.clean:
        clean_out_dir(out_dir)

    proposals = load_proposals(proposal_csv)
    selected = choose_per_image(proposals, args.select, args.sort)
    if args.max_images:
        selected = selected[: args.max_images]
    proposed_classes = [proposal.row.get("proposed_new_class", "") for proposal in selected]
    prefix = review_prefix(proposed_classes)
    scope_text = class_scope_text(proposed_classes)

    overlay_dir = out_dir / "overlays"
    crop_dir = out_dir / "crops"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    review_rows: list[dict[str, str]] = []
    overlay_paths: list[Path] = []
    crop_paths: list[Path] = []
    skipped: list[dict[str, str]] = []
    for index, proposal in enumerate(selected, start=1):
        image_path = resolve(proposal.row["image"])
        review_id = f"{prefix}_{index:03d}"
        if not image_path.exists():
            skipped.append({"review_id": review_id, "image": proposal.row["image"], "reason": "missing_image"})
            continue
        try:
            with Image.open(image_path) as source:
                source_rgb = source.convert("RGB")
                crop_box = clamp_box(proposal.xyxy, source_rgb.size, args.padding)
                crop = source_rgb.crop(crop_box)
                label = (
                    f"{review_id} -> {proposal.row['proposed_new_class']} | "
                    f"{proposal.row['model']} {proposal.row['current_pred_class']} "
                    f"{proposal.confidence:.3f}"
                )
                overlay = scaled_overlay(source_rgb, proposal.xyxy, label, args.max_overlay_side)
        except Exception as exc:  # noqa: BLE001 - review tooling should report bad rows and continue.
            skipped.append({"review_id": review_id, "image": proposal.row["image"], "reason": str(exc)})
            continue

        stem = f"{review_id}_{safe_stem(proposal.row['image'], 'image')}"
        overlay_path = overlay_dir / f"{stem}_overlay.jpg"
        crop_path = crop_dir / f"{stem}_crop.jpg"
        overlay.save(overlay_path, quality=94)
        crop.save(crop_path, quality=94)

        out_row = dict(proposal.row)
        out_row.update(
            {
                "review_id": review_id,
                "source_proposal_csv": repo_rel(proposal_csv),
                "source_proposal_row": str(proposal.source_row),
                "overlay_path": repo_rel(overlay_path),
                "crop_path": repo_rel(crop_path),
                "crop_padding": f"{args.padding:.6g}",
                "review_decision": "",
                "review_notes": "",
            }
        )
        review_rows.append(out_row)
        overlay_paths.append(overlay_path)
        crop_paths.append(crop_path)

    if not review_rows:
        raise SystemExit("no review rows were written")

    base_fields = [
        "review_id",
        "image",
        "model",
        "current_pred_class",
        "current_pred_class_id",
        "confidence",
        "area_ratio",
        "x1",
        "y1",
        "x2",
        "y2",
        "proposed_new_class",
        "review_decision",
        "review_notes",
        "source_proposal_csv",
        "source_proposal_row",
        "overlay_path",
        "crop_path",
        "crop_padding",
    ]
    extras = [key for key in review_rows[0] if key not in base_fields]
    review_csv = out_dir / "review_queue.csv"
    write_csv(review_csv, review_rows, base_fields + extras)
    write_sheet(review_rows, "image", overlay_paths, out_dir / "overlay_sheet.jpg", args.sheet_columns, args.thumb_size)
    write_sheet(review_rows, "image", crop_paths, out_dir / "crop_sheet.jpg", args.sheet_columns, args.thumb_size)

    summary = {
        "schema": "cashsnap_official21_proposal_review_overlays_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "proposal_csv": repo_rel(proposal_csv),
        "out_dir": repo_rel(out_dir),
        "purpose": f"Full-size review overlays/crops for {scope_text} official21 proposal triage; not training data.",
        "not_training_data": True,
        "selection": {
            "select": args.select,
            "sort": args.sort,
            "max_images": args.max_images,
            "padding": args.padding,
        },
        "source_rows": len(proposals),
        "source_images": len({proposal.row["image"] for proposal in proposals}),
        "review_rows": len(review_rows),
        "skipped_rows": len(skipped),
        "proposed_classes": dict(sorted(Counter(row["proposed_new_class"] for row in review_rows).items())),
        "by_model": dict(sorted(Counter(row["model"] for row in review_rows).items())),
        "by_current_pred_class": dict(sorted(Counter(row["current_pred_class"] for row in review_rows).items())),
        "review_queue": repo_rel(review_csv),
        "overlay_sheet": repo_rel(out_dir / "overlay_sheet.jpg"),
        "crop_sheet": repo_rel(out_dir / "crop_sheet.jpg"),
        "skipped": skipped,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Official21 {scope_text} Proposal Review Pack",
                "",
                "This directory contains visual review artifacts only. It is not training data.",
                "",
                "- `review_queue.csv` keeps materializer-compatible proposal columns.",
                "- `review_decision` is intentionally blank until a box has been inspected full-size.",
                acceptance_instruction(proposed_classes),
                "- Leave ambiguous, duplicate, wrong-denomination, or partial-too-weak rows blank/rejected.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(
        "wrote "
        f"{len(review_rows)} {scope_text} proposal review rows to {repo_rel(review_csv)} "
        f"(overlays={len(overlay_paths)} crops={len(crop_paths)} skipped={len(skipped)})"
    )


if __name__ == "__main__":
    main()
