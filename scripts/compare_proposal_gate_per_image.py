#!/usr/bin/env python
"""Compare compact per-image rows from probe_yolo_proposal_gate.py."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
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
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--stage", default="post_reclassifier")
    parser.add_argument(
        "--only-source",
        action="append",
        default=[],
        help="Optional source_group filter. Repeat or comma-separate.",
    )
    parser.add_argument("--khr-per-usd", type=float, default=4000.0)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--csv-out", required=True, type=Path)
    parser.add_argument("--sheet-out", type=Path, default=None)
    parser.add_argument("--sheet-items", type=int, default=96)
    parser.add_argument("--thumb-width", type=int, default=220)
    parser.add_argument("--cols", type=int, default=6)
    return parser.parse_args()


def read_rows(path: Path, stage: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("stages", {}).get(stage)
    if not isinstance(rows, list):
        raise SystemExit(f"{repo_rel(path)} has no stage rows named {stage!r}")
    by_image = {}
    for row in rows:
        if not isinstance(row, dict) or "image" not in row:
            raise SystemExit(f"{repo_rel(path)} has malformed row in {stage!r}")
        by_image[str(row["image"])] = row
    return by_image


def number(value: Any) -> float:
    return 0.0 if value is None else float(value)


def weighted_error(row: dict[str, Any], khr_per_usd: float) -> float:
    return abs(number(row.get("usd_error"))) + abs(number(row.get("khr_error"))) / khr_per_usd


def exact_value(row: dict[str, Any]) -> bool:
    return bool(row.get("exact_value"))


def parse_sources(raw_values: list[str]) -> set[str]:
    sources: set[str] = set()
    for raw_value in raw_values:
        sources.update(value.strip() for value in raw_value.split(",") if value.strip())
    return sources


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    resolved = resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image",
        "source_group",
        "direction",
        "baseline_exact_value",
        "candidate_exact_value",
        "baseline_weighted_error",
        "candidate_weighted_error",
        "baseline_count_error",
        "candidate_count_error",
        "baseline_usd_error",
        "candidate_usd_error",
        "baseline_khr_error",
        "candidate_khr_error",
        "baseline_fp",
        "candidate_fp",
        "baseline_fn",
        "candidate_fn",
    ]
    with resolved.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def fit_image(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as loaded:
        image = ImageOps.exif_transpose(loaded.convert("RGB"))
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height), "white")
    tile.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return tile


def draw_sheet(path: Path, rows: list[dict[str, Any]], *, items: int, thumb_width: int, cols: int) -> None:
    selected = rows[:items]
    if not selected:
        return
    thumb_h = thumb_width
    caption_h = 58
    rows_count = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_width, rows_count * (thumb_h + caption_h)), (238, 238, 238))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(selected):
        image_path = resolve(str(row["image"]))
        x = (index % cols) * thumb_width
        y = (index // cols) * (thumb_h + caption_h)
        try:
            tile = fit_image(image_path, thumb_width, thumb_h)
        except Exception:
            tile = Image.new("RGB", (thumb_width, thumb_h), (80, 80, 80))
            ImageDraw.Draw(tile).text((8, 8), "missing", fill=(255, 255, 255), font=font)
        sheet.paste(tile, (x, y))
        outline = (40, 140, 60) if "candidate_exact_win" in str(row["direction"]) else (180, 60, 60)
        draw.rectangle((x, y, x + thumb_width - 1, y + thumb_h - 1), outline=outline, width=3)
        caption = (
            f"{row['direction']}\n{row['source_group']} "
            f"B:{row['baseline_count_error']}/{row['baseline_usd_error']}/{row['baseline_khr_error']} "
            f"C:{row['candidate_count_error']}/{row['candidate_usd_error']}/{row['candidate_khr_error']}"
        )
        draw.multiline_text((x + 4, y + thumb_h + 3), caption[:140], fill=(10, 10, 10), font=font, spacing=2)
    resolved = resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(resolved, quality=92)


def main() -> None:
    args = parse_args()
    baseline_path = resolve(args.baseline)
    candidate_path = resolve(args.candidate)
    baseline = read_rows(baseline_path, args.stage)
    candidate = read_rows(candidate_path, args.stage)
    shared_images = sorted(set(baseline) & set(candidate))
    only_sources = parse_sources(args.only_source)
    if only_sources:
        shared_images = [
            image
            for image in shared_images
            if str(candidate[image].get("source_group") or baseline[image].get("source_group")) in only_sources
        ]
    if not shared_images:
        raise SystemExit("no shared image rows")

    exact_wins: list[dict[str, Any]] = []
    exact_losses: list[dict[str, Any]] = []
    weighted_better: list[dict[str, Any]] = []
    weighted_worse: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []

    for image in shared_images:
        base_row = baseline[image]
        cand_row = candidate[image]
        base_exact = exact_value(base_row)
        cand_exact = exact_value(cand_row)
        base_weighted = weighted_error(base_row, args.khr_per_usd)
        cand_weighted = weighted_error(cand_row, args.khr_per_usd)
        direction_parts = []
        row = {
            "image": image,
            "source_group": cand_row.get("source_group") or base_row.get("source_group"),
            "baseline_exact_value": base_exact,
            "candidate_exact_value": cand_exact,
            "baseline_weighted_error": base_weighted,
            "candidate_weighted_error": cand_weighted,
            "baseline_count_error": base_row.get("count_error"),
            "candidate_count_error": cand_row.get("count_error"),
            "baseline_usd_error": base_row.get("usd_error"),
            "candidate_usd_error": cand_row.get("usd_error"),
            "baseline_khr_error": base_row.get("khr_error"),
            "candidate_khr_error": cand_row.get("khr_error"),
            "baseline_fp": base_row.get("fp"),
            "candidate_fp": cand_row.get("fp"),
            "baseline_fn": base_row.get("fn"),
            "candidate_fn": cand_row.get("fn"),
        }
        if not base_exact and cand_exact:
            exact_wins.append(row)
            direction_parts.append("candidate_exact_win")
        elif base_exact and not cand_exact:
            exact_losses.append(row)
            direction_parts.append("candidate_exact_loss")
        if cand_weighted < base_weighted:
            weighted_better.append(row)
            direction_parts.append("candidate_weighted_better")
        elif cand_weighted > base_weighted:
            weighted_worse.append(row)
            direction_parts.append("candidate_weighted_worse")
        if direction_parts:
            row["direction"] = ";".join(direction_parts)
            changed_rows.append(row)

    summary = {
        "schema": "cashsnap_proposal_gate_per_image_comparison_v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "baseline": repo_rel(baseline_path),
        "candidate": repo_rel(candidate_path),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "stage": args.stage,
        "only_sources": sorted(only_sources),
        "khr_per_usd": args.khr_per_usd,
        "shared_images": len(shared_images),
        "baseline_exact_value_images": sum(exact_value(row) for row in baseline.values()),
        "candidate_exact_value_images": sum(exact_value(row) for row in candidate.values()),
        "candidate_exact_wins": len(exact_wins),
        "candidate_exact_losses": len(exact_losses),
        "candidate_exact_net": len(exact_wins) - len(exact_losses),
        "candidate_weighted_better": len(weighted_better),
        "candidate_weighted_worse": len(weighted_worse),
        "candidate_weighted_net": len(weighted_better) - len(weighted_worse),
        "exact_win_sources": dict(Counter(row["source_group"] for row in exact_wins).most_common()),
        "exact_loss_sources": dict(Counter(row["source_group"] for row in exact_losses).most_common()),
        "weighted_better_sources": dict(Counter(row["source_group"] for row in weighted_better).most_common()),
        "weighted_worse_sources": dict(Counter(row["source_group"] for row in weighted_worse).most_common()),
        "csv": repo_rel(resolve(args.csv_out)),
    }
    write_json(args.json_out, summary)
    write_csv(args.csv_out, changed_rows)
    if args.sheet_out is not None:
        exact_rows = [
            row
            for row in changed_rows
            if "candidate_exact_win" in str(row["direction"]) or "candidate_exact_loss" in str(row["direction"])
        ]
        weighted_only_rows = [row for row in changed_rows if row not in exact_rows]
        draw_sheet(
            args.sheet_out,
            exact_rows + weighted_only_rows,
            items=args.sheet_items,
            thumb_width=args.thumb_width,
            cols=args.cols,
        )
        summary["sheet"] = repo_rel(resolve(args.sheet_out))
        write_json(args.json_out, summary)
    print(
        f"comparison={repo_rel(resolve(args.json_out))} "
        f"exact_net={summary['candidate_exact_net']:+d} "
        f"weighted_net={summary['candidate_weighted_net']:+d} "
        f"changed_rows={len(changed_rows)}"
    )


if __name__ == "__main__":
    main()
