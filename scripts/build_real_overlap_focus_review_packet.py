#!/usr/bin/env python
"""Build a focused review packet from annotated real-overlap diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from build_real_overlap_review_queue import draw_sheet, names_by_id, read_labels


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DATA = Path("data/cashsnap_v1/data.yaml")
DEFAULT_REVIEW_CSV = Path(
    "runs/cashsnap/real_overlap_review_queue_v1/first_review_clusters_balanced_v1_model_error_triage.csv"
)
DEFAULT_OUT_DIR = Path("runs/cashsnap/real_overlap_focus_review_packet_v1")


FOCUS_ORDER = {
    "train_bbox_overlap_anchor_review": 0,
    "train_tight_pair_anchor_review": 1,
    "train_tight_pair_flat_source_policy_review": 2,
    "heldout_cashcountingxl_usd_eval_review": 3,
    "heldout_multi_note_eval_review": 4,
    "heldout_multi_note_flat_source_policy_review": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE_DATA)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-bbox-train", type=int, default=20)
    parser.add_argument("--max-tight-train", type=int, default=20)
    parser.add_argument("--max-cashcounting-heldout", type=int, default=12)
    parser.add_argument("--max-multinote-heldout", type=int, default=12)
    parser.add_argument("--thumb-width", type=int, default=220)
    parser.add_argument("--cols", type=int, default=5)
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def read_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(resolved)} must be a YAML mapping")
    return payload


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    resolved = resolve(path)
    with resolved.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def int_value(value: str) -> int:
    try:
        return int(float(str(value).strip() or "0"))
    except ValueError:
        return 0


def float_value(value: str) -> float:
    try:
        return float(str(value).strip() or "0")
    except ValueError:
        return 0.0


def selected_rows(
    rows: list[dict[str, str]],
    *,
    focus_bucket: str,
    cap: int,
    predicate,
    suggested_usable_as: str,
    focus_reason: str,
) -> list[dict[str, str]]:
    candidates = [row for row in rows if predicate(row)]
    candidates.sort(
        key=lambda row: (
            -int_value(row.get("model_error_total", "")),
            -float_value(row.get("model_error_priority", "")),
            -float_value(row.get("priority", "")),
            row.get("source_group", ""),
            row.get("image", ""),
        )
    )
    selected: list[dict[str, str]] = []
    for row in candidates[:cap]:
        copy = dict(row)
        copy["focus_bucket"] = focus_bucket
        copy["focus_reason"] = focus_reason
        copy["suggested_usable_as"] = suggested_usable_as
        copy["suggested_final_route"] = suggested_usable_as
        selected.append(copy)
    return selected


def build_focus_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    focus_rows: list[dict[str, str]] = []
    focus_rows.extend(
        selected_rows(
            rows,
            focus_bucket="train_bbox_overlap_anchor_review",
            cap=args.max_bbox_train,
            predicate=lambda row: row.get("split") == "train" and row.get("packet_bucket") == "bbox_overlap",
            suggested_usable_as="train_anchor_candidate",
            focus_reason="train bbox-overlap rows drive the split-mixed recall gap",
        )
    )
    tight_pair_rows = selected_rows(
        rows,
        focus_bucket="train_tight_pair_anchor_review",
        cap=args.max_tight_train,
        predicate=lambda row: row.get("split") == "train" and row.get("packet_bucket") == "tight_pair",
        suggested_usable_as="train_anchor_candidate",
        focus_reason="train tight-pair rows drive the largest recall gap",
    )
    for row in tight_pair_rows:
        if row.get("source_group") == "khmer_us_currency":
            row["focus_bucket"] = "train_tight_pair_flat_source_policy_review"
            row["focus_reason"] = (
                "khmer_us_currency tight pairs are often flat front/back catalog composites; "
                "review before treating as overlap anchors"
            )
            row["suggested_usable_as"] = "exclude_duplicate_or_flat"
            row["suggested_final_route"] = "exclude_duplicate_or_flat"
    focus_rows.extend(tight_pair_rows)
    focus_rows.extend(
        selected_rows(
            rows,
            focus_bucket="heldout_cashcountingxl_usd_eval_review",
            cap=args.max_cashcounting_heldout,
            predicate=lambda row: row.get("split") in {"val", "test"}
            and row.get("packet_bucket") == "cashcountingxl_usd_context",
            suggested_usable_as="trusted_overlap_eval",
            focus_reason="held-out USD cash-counting context is the remaining recall warning",
        )
    )
    multinote_rows = selected_rows(
        rows,
        focus_bucket="heldout_multi_note_eval_review",
        cap=args.max_multinote_heldout,
        predicate=lambda row: row.get("split") in {"val", "test"}
        and row.get("packet_bucket") == "val_test_multi_note_smoke",
        suggested_usable_as="trusted_overlap_eval",
        focus_reason="small multi-note smoke slice should be reviewed before reuse",
    )
    for row in multinote_rows:
        if row.get("source_group") == "khmer_us_currency":
            row["focus_bucket"] = "heldout_multi_note_flat_source_policy_review"
            row["focus_reason"] = (
                "khmer_us_currency multi-note held-out rows can be flat front/back catalog composites; "
                "review before treating as trusted eval"
            )
            row["suggested_usable_as"] = "exclude_duplicate_or_flat"
            row["suggested_final_route"] = "exclude_duplicate_or_flat"
    focus_rows.extend(multinote_rows)
    focus_rows.sort(
        key=lambda row: (
            FOCUS_ORDER.get(row.get("focus_bucket", ""), 99),
            -int_value(row.get("model_error_total", "")),
            -float_value(row.get("priority", "")),
            row.get("image", ""),
        )
    )
    for index, row in enumerate(focus_rows, start=1):
        row["focus_review_id"] = f"ORF-{index:03d}"
    return focus_rows


def write_csv(path: Path, rows: list[dict[str, str]], input_fields: list[str]) -> None:
    extra_fields = [
        "focus_review_id",
        "focus_bucket",
        "focus_reason",
        "suggested_usable_as",
        "suggested_final_route",
    ]
    fields = extra_fields + [field for field in input_fields if field not in extra_fields]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_list(path: Path, rows: list[dict[str, str]]) -> None:
    images = list(dict.fromkeys(row.get("image", "").replace("\\", "/") for row in rows if row.get("image")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{image}\n" for image in images), encoding="utf-8")


def write_data_yaml(path: Path, image_list: Path, base_config: dict[str, Any], review_csv: Path) -> None:
    payload = {
        "path": rel_between(path.parent, ROOT),
        "train": repo_rel(image_list),
        "val": repo_rel(image_list),
        "test": repo_rel(image_list),
        "names": base_config.get("names", {}),
        "cashsnap_diagnostic": {
            "purpose": "Focused real-overlap review packet; not accepted training or promotion data.",
            "review_csv": repo_rel(review_csv),
            "source_review_packet": repo_rel(image_list),
            "not_a_promotion_config": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_diagnostic_view(
    *,
    out_dir: Path,
    stem: str,
    rows: list[dict[str, str]],
    base_config: dict[str, Any],
    review_csv: Path,
) -> dict[str, Any]:
    list_path = out_dir / f"{stem}_images.txt"
    data_yaml_path = out_dir / f"{stem}_data.yaml"
    write_list(list_path, rows)
    write_data_yaml(data_yaml_path, list_path, base_config, review_csv)
    return {
        "name": stem,
        "rows": len(rows),
        "images": len(list(dict.fromkeys(row.get("image", "") for row in rows if row.get("image")))),
        "images_txt": repo_rel(list_path),
        "data_yaml": repo_rel(data_yaml_path),
        "focus_bucket_counts": dict(Counter(row.get("focus_bucket", "") for row in rows)),
        "suggested_usable_as_counts": dict(Counter(row.get("suggested_usable_as", "") for row in rows)),
    }


def rows_for_sheet(rows: list[dict[str, str]], names: dict[int, str]) -> list[dict[str, Any]]:
    sheet_rows: list[dict[str, Any]] = []
    for row in rows:
        copy: dict[str, Any] = dict(row)
        copy["labels"] = read_labels(Path(copy["image"]), names)
        copy["packet_bucket"] = f"{copy.get('focus_bucket', '')}|{copy.get('packet_bucket', '')}"
        sheet_rows.append(copy)
    return sheet_rows


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = read_yaml(args.base_data)
    names = names_by_id(base_config)
    rows, input_fields = read_csv(args.review_csv)
    focus_rows = build_focus_rows(rows, args)

    csv_path = out_dir / "focus_review_packet_v1.csv"
    sheet_path = out_dir / "focus_review_packet_v1_sheet.jpg"
    list_path = out_dir / "focus_review_packet_v1_images.txt"
    data_yaml_path = out_dir / "focus_review_packet_v1_data.yaml"

    write_csv(csv_path, focus_rows, input_fields)
    write_list(list_path, focus_rows)
    write_data_yaml(data_yaml_path, list_path, base_config, csv_path)
    suggested_eval_rows = [
        row for row in focus_rows if row.get("suggested_usable_as") == "trusted_overlap_eval"
    ]
    suggested_train_rows = [
        row for row in focus_rows if row.get("suggested_usable_as") == "train_anchor_candidate"
    ]
    flat_policy_rows = [
        row for row in focus_rows if row.get("suggested_usable_as") == "exclude_duplicate_or_flat"
    ]
    diagnostic_views = [
        write_diagnostic_view(
            out_dir=out_dir,
            stem="suggested_eval_view_v1",
            rows=suggested_eval_rows,
            base_config=base_config,
            review_csv=csv_path,
        ),
        write_diagnostic_view(
            out_dir=out_dir,
            stem="suggested_train_anchor_view_v1",
            rows=suggested_train_rows,
            base_config=base_config,
            review_csv=csv_path,
        ),
        write_diagnostic_view(
            out_dir=out_dir,
            stem="flat_source_policy_view_v1",
            rows=flat_policy_rows,
            base_config=base_config,
            review_csv=csv_path,
        ),
    ]
    draw_sheet(
        sheet_path,
        rows_for_sheet(focus_rows, names),
        items=len(focus_rows),
        thumb_width=args.thumb_width,
        cols=args.cols,
    )

    summary = {
        "schema": "cashsnap_real_overlap_focus_review_packet_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_data": repo_rel(resolve(args.base_data)),
        "input_review_csv": repo_rel(resolve(args.review_csv)),
        "out_dir": repo_rel(out_dir),
        "csv": repo_rel(csv_path),
        "sheet": repo_rel(sheet_path),
        "images": repo_rel(list_path),
        "data_yaml": repo_rel(data_yaml_path),
        "rows": len(focus_rows),
        "not_training_data": True,
        "not_a_promotion_config": True,
        "requires_explicit_review_before_materialization": True,
        "diagnostic_views": diagnostic_views,
        "focus_bucket_counts": dict(Counter(row.get("focus_bucket", "") for row in focus_rows)),
        "packet_bucket_counts": dict(Counter(row.get("packet_bucket", "") for row in focus_rows)),
        "split_counts": dict(Counter(row.get("split", "") for row in focus_rows)),
        "source_counts": dict(Counter(row.get("source_group", "") for row in focus_rows)),
        "model_error_total": sum(int_value(row.get("model_error_total", "")) for row in focus_rows),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
