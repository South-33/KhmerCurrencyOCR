#!/usr/bin/env python
"""Validate a WebGL visual-review CSV against the review rules."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_visual_review_rules_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--require-accepted", action="store_true", help="Require every row to be reviewed and accepted.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> object:
    resolved = resolve(path)
    if not resolved.exists():
        raise SystemExit(f"missing JSON file: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def split_tokens(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,;\s]+", value) if item.strip()]


def main() -> int:
    args = parse_args()
    rules = read_json(args.rules)
    require(isinstance(rules, dict), "rules must be a JSON object")
    statuses = {str(row.get("id", "")) for row in rules.get("review_statuses", []) if isinstance(row, dict)}
    accepted = {str(item) for item in rules.get("accepted_statuses", [])}
    blocking = {str(item) for item in rules.get("blocking_statuses", [])}
    reasons = {str(item) for item in rules.get("bad_scene_reasons", [])}
    required_columns = [str(item) for item in rules.get("required_review_columns", [])]
    require(statuses, "rules must define review_statuses")
    require(accepted, "rules must define accepted_statuses")

    review_csv = resolve(args.review_csv)
    with review_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    require(rows, "review CSV must contain at least one row")
    missing_columns = [column for column in required_columns if column not in fieldnames]
    require(not missing_columns, f"review CSV missing required columns: {missing_columns}")

    status_counts: Counter[str] = Counter()
    pending = 0
    for index, row in enumerate(rows, start=2):
        status = row.get("review_status", "").strip()
        row_reasons = split_tokens(row.get("bad_scene_reasons", ""))
        unknown_reasons = sorted(set(row_reasons) - reasons)
        require(not unknown_reasons, f"row {index}: unknown bad_scene_reasons {unknown_reasons}")
        if not status:
            pending += 1
            if args.require_accepted:
                raise SystemExit(f"row {index}: review_status is required")
            continue
        require(status in statuses, f"row {index}: invalid review_status {status!r}")
        if status in blocking:
            require(row_reasons, f"row {index}: blocking status {status!r} requires bad_scene_reasons")
        if args.require_accepted:
            require(status in accepted, f"row {index}: status {status!r} is not accepted")
        status_counts[status] += 1

    print(
        f"ok: reviewed {len(rows)} WebGL visual row(s), "
        f"pending={pending}, statuses={dict(sorted(status_counts.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
