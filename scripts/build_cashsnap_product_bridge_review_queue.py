#!/usr/bin/env python
"""Build a focused CashSnap review queue from product-gated audit findings."""

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
DEFAULT_AUDIT_DIR = ROOT / "runs" / "cashsnap" / "real_data_label_audit_v1"


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
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_AUDIT_DIR / "product_bridge_review_queue_v1",
    )
    parser.add_argument("--max-total", type=int, default=180)
    parser.add_argument("--max-khr100", type=int, default=60)
    parser.add_argument("--max-empty-suspects", type=int, default=60)
    parser.add_argument("--max-khmer-us-review", type=int, default=80)
    parser.add_argument("--sheet-items", type=int, default=96)
    parser.add_argument("--thumb-width", type=int, default=260)
    parser.add_argument("--cols", type=int, default=4)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_txt(path: Path) -> list[str]:
    resolved = resolve(path)
    if not resolved.exists():
        return []
    return [line.strip() for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()]


def inventory_by_image(audit_dir: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(audit_dir / "inventory.csv")
    return {str(row["image"]): row for row in rows}


def append_row(
    rows: dict[str, dict[str, Any]],
    *,
    image: str,
    priority: int,
    bucket: str,
    suggested_action: str,
    reason: str,
    source: str,
    inventory: dict[str, dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> None:
    inv = inventory.get(image, {})
    existing = rows.get(image)
    payload = {
        "priority": priority,
        "bucket": bucket,
        "suggested_action": suggested_action,
        "reason": reason,
        "source": source,
        "image": image,
        "label": inv.get("label", ""),
        "split": inv.get("split", ""),
        "source_group": inv.get("source_group", ""),
        "label_count": inv.get("label_count", ""),
        "class_names": inv.get("class_names", ""),
        "issue_reasons": inv.get("issue_reasons", ""),
    }
    if extra:
        payload.update(extra)
    if existing is None:
        rows[image] = payload
        return
    if priority > int(existing["priority"]):
        payload["bucket"] = f"{bucket};{existing['bucket']}"
        payload["suggested_action"] = f"{suggested_action};{existing['suggested_action']}"
        payload["reason"] = f"{reason};{existing['reason']}"
        payload["source"] = f"{source};{existing['source']}"
        rows[image] = payload
    else:
        existing["bucket"] = f"{existing['bucket']};{bucket}"
        existing["suggested_action"] = f"{existing['suggested_action']};{suggested_action}"
        existing["reason"] = f"{existing['reason']};{reason}"
        existing["source"] = f"{existing['source']};{source}"


def queue_rows(args: argparse.Namespace, audit_dir: Path) -> list[dict[str, Any]]:
    inventory = inventory_by_image(audit_dir)
    rows: dict[str, dict[str, Any]] = {}

    churn_csv = audit_dir / "proposal_gate_strict_clean_khmer_us_currency_balanced_real_p24_vs_real_synth_p24_changed_images_v1.csv"
    if churn_csv.exists():
        for row in read_csv(churn_csv):
            direction = str(row.get("direction", ""))
            priority = 100 if "candidate_exact_loss" in direction else 96 if "candidate_exact_win" in direction else 90
            append_row(
                rows,
                image=str(row["image"]),
                priority=priority,
                bucket="khmer_us_currency_stack_churn",
                suggested_action="visual_source_class_audit",
                reason=direction,
                source=repo_rel(churn_csv),
                inventory=inventory,
                extra={
                    "baseline_count_error": row.get("baseline_count_error", ""),
                    "candidate_count_error": row.get("candidate_count_error", ""),
                    "baseline_usd_error": row.get("baseline_usd_error", ""),
                    "candidate_usd_error": row.get("candidate_usd_error", ""),
                    "baseline_khr_error": row.get("baseline_khr_error", ""),
                    "candidate_khr_error": row.get("candidate_khr_error", ""),
                },
            )

    review_queue = audit_dir / "review_queue_ranked_v1.csv"
    if review_queue.exists():
        selected = [
            row
            for row in read_csv(review_queue)
            if str(row.get("source_group")) == "khmer_us_currency"
            and str(row.get("suggested_action"))
            in {"mixed_currency_source_class_audit", "route_khr100_unknown_current_schema_out_of_scope"}
        ][: args.max_khmer_us_review]
        for row in selected:
            append_row(
                rows,
                image=str(row["image"]),
                priority=int(row.get("priority") or 92),
                bucket="khmer_us_currency_ranked_review",
                suggested_action=str(row.get("suggested_action", "source_class_audit")),
                reason=str(row.get("flags") or row.get("issue_reasons") or "ranked_review"),
                source=repo_rel(review_queue),
                inventory=inventory,
            )

    for image in read_txt(audit_dir / "review_khr100_schema_out_of_scope_all_splits_v1.txt")[: args.max_khr100]:
        append_row(
            rows,
            image=image,
            priority=94,
            bucket="khr100_schema_out_of_scope",
            suggested_action="route_unknown_or_expand_schema",
            reason="current_13_class_schema_cannot_encode_khr100",
            source=repo_rel(audit_dir / "review_khr100_schema_out_of_scope_all_splits_v1.txt"),
            inventory=inventory,
        )

    for image in read_txt(audit_dir / "review_empty_label_target_suspects.txt")[: args.max_empty_suspects]:
        append_row(
            rows,
            image=image,
            priority=88,
            bucket="empty_label_target_suspect",
            suggested_action="review_or_relabel_empty_label",
            reason="teacher_predicted_target_on_empty_label_row",
            source=repo_rel(audit_dir / "review_empty_label_target_suspects.txt"),
            inventory=inventory,
        )

    chosen = sorted(
        rows.values(),
        key=lambda row: (
            -int(row["priority"]),
            str(row.get("bucket", "")),
            str(row.get("source_group", "")),
            str(row.get("image", "")),
        ),
    )
    return chosen[: args.max_total]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "queue_rank",
        "priority",
        "bucket",
        "suggested_action",
        "review_status",
        "review_decision",
        "review_notes",
        "reason",
        "split",
        "source_group",
        "label_count",
        "class_names",
        "issue_reasons",
        "baseline_count_error",
        "candidate_count_error",
        "baseline_usd_error",
        "candidate_usd_error",
        "baseline_khr_error",
        "candidate_khr_error",
        "image",
        "label",
        "source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            payload = {field: row.get(field, "") for field in fields}
            payload["queue_rank"] = index
            writer.writerow(payload)


def fit_image(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as loaded:
        image = ImageOps.exif_transpose(loaded.convert("RGB"))
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height), "white")
    tile.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return tile


def draw_sheet(rows: list[dict[str, Any]], out_path: Path, *, items: int, thumb_width: int, cols: int) -> None:
    selected = rows[:items]
    if not selected:
        return
    thumb_h = int(thumb_width * 0.78)
    caption_h = 66
    cols = max(1, min(cols, len(selected)))
    sheet_rows = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_width, sheet_rows * (thumb_h + caption_h)), (238, 238, 238))
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
        draw.rectangle((x, y, x + thumb_width - 1, y + thumb_h - 1), outline=(60, 80, 160), width=3)
        caption = (
            f"#{index + 1} P{row['priority']} {row['bucket'][:34]}\n"
            f"{row.get('split', '')} {row.get('source_group', '')} {row.get('class_names', '')[:24]}\n"
            f"{Path(str(row['image'])).name[:58]}"
        )
        draw.multiline_text((x + 4, y + thumb_h + 4), caption, fill=(5, 5, 5), font=font, spacing=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> int:
    args = parse_args()
    audit_dir = resolve(args.audit_dir)
    out_dir = resolve(args.out_dir)
    rows = queue_rows(args, audit_dir)
    if not rows:
        raise SystemExit("no product bridge review rows selected")
    queue_csv = out_dir / "queue.csv"
    sheet = out_dir / "queue.jpg"
    summary_path = out_dir / "summary.json"
    write_csv(queue_csv, rows)
    draw_sheet(rows, sheet, items=args.sheet_items, thumb_width=args.thumb_width, cols=args.cols)
    summary = {
        "schema": "cashsnap_product_bridge_review_queue_v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "audit_dir": repo_rel(audit_dir),
        "queue_csv": repo_rel(queue_csv),
        "sheet": repo_rel(sheet),
        "selected_rows": len(rows),
        "bucket_counts": dict(sorted(Counter(str(row.get("bucket", "")) for row in rows).items())),
        "source_group_counts": dict(sorted(Counter(str(row.get("source_group", "")) for row in rows).items())),
        "split_counts": dict(sorted(Counter(str(row.get("split", "")) for row in rows).items())),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
