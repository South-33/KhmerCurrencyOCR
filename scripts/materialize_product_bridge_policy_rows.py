#!/usr/bin/env python
"""Materialize narrow product-bridge policy rows into explicit split lists."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = (
    ROOT
    / "runs"
    / "cashsnap"
    / "real_data_label_audit_v1"
    / "product_bridge_review_queue_v1"
    / "queue.csv"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "runs"
    / "cashsnap"
    / "real_data_label_audit_v1"
    / "product_bridge_review_queue_v1"
    / "materialized_khr100_policy_v1"
)
ALLOWED_USABLE_AS = {"unknown_out_of_scope"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-csv", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--policy",
        choices=["khr100_current_schema_unknown"],
        required=True,
        help="Narrow policy route to materialize.",
    )
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


def normalized(value: str) -> str:
    return value.strip().lower().replace("\\", "/")


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    resolved = resolve(path)
    with resolved.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_list(path: Path, images: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = list(dict.fromkeys(images))
    path.write_text("\n".join(unique) + ("\n" if unique else ""), encoding="utf-8")


def is_policy_khr100_unknown(row: dict[str, str]) -> bool:
    image = normalized(row.get("image", ""))
    source_group = normalized(row.get("source_group", ""))
    bucket = normalized(row.get("bucket", ""))
    action = normalized(row.get("suggested_action", ""))
    label_count = str(row.get("label_count", "")).strip()
    return (
        source_group == "khmer_us_currency"
        and label_count in {"", "0", "0.0"}
        and "100-riel" in image
        and ("khr100" in bucket or "khr100" in action or "unknown_or_expand_schema" in action)
    )


def materialize_policy_rows(rows: list[dict[str, str]], policy: str) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for row in rows:
        if policy == "khr100_current_schema_unknown":
            accepted = is_policy_khr100_unknown(row)
            usable_as = "unknown_out_of_scope"
            final_route = "current_schema_unknown_khr100"
            notes = (
                "Policy route: visible KHR_100/100-riel source row is official currency "
                "but outside the current 13-class detector schema."
            )
        else:
            raise SystemExit(f"unsupported policy: {policy}")
        if not accepted:
            continue
        image = str(row.get("image", "")).strip()
        if not image or image in seen_images:
            continue
        seen_images.add(image)
        copy = dict(row)
        copy.update(
            {
                "usable_as": usable_as,
                "final_route": final_route,
                "materialized_review_status": "codex_policy_reviewed",
                "materialized_review_decision": "accepted_policy_route",
                "materialized_review_notes": notes,
            }
        )
        materialized.append(copy)
    return materialized


def main() -> int:
    args = parse_args()
    queue_csv = resolve(args.queue_csv)
    out_dir = resolve(args.out_dir)
    rows, fields = read_csv(queue_csv)
    materialized = materialize_policy_rows(rows, args.policy)
    if not materialized:
        raise SystemExit(f"no rows materialized for policy {args.policy}")

    out_dir.mkdir(parents=True, exist_ok=True)
    extra_fields = [
        "usable_as",
        "final_route",
        "materialized_review_status",
        "materialized_review_decision",
        "materialized_review_notes",
    ]
    manifest_path = out_dir / "manifest.csv"
    write_csv(manifest_path, materialized, [*fields, *extra_fields])

    list_paths: dict[str, str] = {}
    for usable_as in sorted(ALLOWED_USABLE_AS):
        bucket_rows = [row for row in materialized if row["usable_as"] == usable_as]
        for split in ("all", "train", "val", "test"):
            if split == "all":
                images = [str(row["image"]) for row in bucket_rows]
            else:
                images = [str(row["image"]) for row in bucket_rows if str(row.get("split", "")) == split]
            path = out_dir / f"{usable_as}_{split}_images.txt"
            write_list(path, images)
            list_paths[f"{usable_as}_{split}"] = repo_rel(path)

    summary = {
        "schema": "cashsnap_product_bridge_policy_materialization_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": args.policy,
        "queue_csv": repo_rel(queue_csv),
        "manifest_csv": repo_rel(manifest_path),
        "list_paths": list_paths,
        "materialized_rows": len(materialized),
        "materialized_unique_images": len({row["image"] for row in materialized}),
        "by_split": dict(Counter(str(row.get("split", "")) for row in materialized).most_common()),
        "by_source": dict(Counter(str(row.get("source_group", "")) for row in materialized).most_common()),
        "by_usable_as": dict(Counter(str(row.get("usable_as", "")) for row in materialized).most_common()),
        "not_a_yolo_config": True,
        "note": (
            "This materializes only narrow policy-obvious KHR_100 current-schema unknown rows. "
            "Rows remain separate from hard negatives because KHR_100 is official currency, not background."
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"materialized_policy={repo_rel(out_dir)} rows={summary['materialized_rows']} "
        f"splits={summary['by_split']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
