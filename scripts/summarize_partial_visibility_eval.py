#!/usr/bin/env python
"""Summarize partial-visibility eval rows by manifest attributes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--per-image", required=True, action="append", type=Path)
    parser.add_argument(
        "--group-by",
        default=["split", "mode", "target_visible_fraction", "blocker_style"],
        nargs="+",
        help="Grouping fields, separated by spaces and/or commas.",
    )
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--csv-out", required=True, type=Path)
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value not in {"", None} else 0


def metric(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(rows: list[dict[str, str]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        item = groups.setdefault(
            key,
            {
                **{field: value for field, value in zip(group_fields, key, strict=True)},
                "images": 0,
                "gt": 0,
                "tp": 0,
                "fn": 0,
                "fp": 0,
                "total_predictions": 0,
            },
        )
        item["images"] += 1
        for field in ("gt", "tp", "fn", "fp", "total_predictions"):
            item[field] += number(row, field)
    output = []
    for item in groups.values():
        gt = int(item["gt"])
        tp = int(item["tp"])
        fp = int(item["fp"])
        item["recall"] = metric(tp, gt)
        item["precision"] = metric(tp, tp + fp)
        output.append(item)
    output.sort(key=lambda item: (item.get("split", ""), item.get("recall") is None, item.get("recall") or -1, item["images"]))
    return output


def main() -> int:
    args = parse_args()
    manifest_rows = read_csv(resolve(args.manifest))
    manifest_by_image = {row["image"].replace("\\", "/"): row for row in manifest_rows}
    joined: list[dict[str, str]] = []
    missing: list[str] = []
    for per_image_path in args.per_image:
        for row in read_csv(resolve(per_image_path)):
            image = row["image"].replace("\\", "/")
            manifest = manifest_by_image.get(image)
            if manifest is None:
                missing.append(image)
                continue
            joined.append({**manifest, **row})
    if not joined:
        raise SystemExit("no per-image rows joined to manifest")

    group_text = " ".join(args.group_by)
    group_fields = [field for field in re.split(r"[\s,]+", group_text.strip()) if field]
    missing_fields = sorted(field for field in group_fields if field not in joined[0])
    if missing_fields:
        raise SystemExit(f"group fields not found in joined rows: {', '.join(missing_fields)}")
    summary_rows = summarize(joined, group_fields)
    payload = {
        "schema": "cashsnap_partial_visibility_eval_summary_v1",
        "manifest": repo_rel(resolve(args.manifest)),
        "per_image": [repo_rel(resolve(path)) for path in args.per_image],
        "group_by": group_fields,
        "joined_rows": len(joined),
        "missing_rows": len(missing),
        "groups": summary_rows,
    }
    json_out = resolve(args.json_out)
    csv_out = resolve(args.csv_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = group_fields + ["images", "gt", "tp", "fn", "fp", "total_predictions", "recall", "precision"]
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(
        f"partial_visibility_eval_summary={repo_rel(json_out)} rows={len(joined)} groups={len(summary_rows)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
