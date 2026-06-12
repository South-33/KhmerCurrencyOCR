#!/usr/bin/env python
"""Apply explicit review decisions to a CSV review queue."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--id-column", default="review_id")
    parser.add_argument("--decision-column", default="review_decision")
    parser.add_argument("--notes-column", default="review_notes")
    parser.add_argument("--accept", action="append", default=[], help="Review id to mark accepted_box.")
    parser.add_argument("--reject", action="append", default=[], help="Review id to mark rejected.")
    parser.add_argument("--accepted-decision", default="accepted_box")
    parser.add_argument("--rejected-decision", default="rejected")
    parser.add_argument("--accepted-note", default="")
    parser.add_argument("--rejected-note", default="")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def main() -> None:
    args = parse_args()
    input_csv = resolve(args.input_csv)
    out_csv = resolve(args.out_csv)
    accepted = set(args.accept)
    rejected = set(args.reject)
    overlap = sorted(accepted & rejected)
    if overlap:
        raise SystemExit(f"ids cannot be both accepted and rejected: {overlap}")
    if not accepted and not rejected:
        raise SystemExit("provide at least one --accept or --reject id")

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{repo_rel(input_csv)} has no header")
        required = [args.id_column, args.decision_column, args.notes_column]
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise SystemExit(f"{repo_rel(input_csv)} missing required columns: {missing}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    seen = {row[args.id_column] for row in rows}
    unknown = sorted((accepted | rejected) - seen)
    if unknown:
        raise SystemExit(f"review ids not found in {repo_rel(input_csv)}: {unknown}")

    changed = 0
    for row in rows:
        review_id = row[args.id_column]
        if review_id in accepted:
            row[args.decision_column] = args.accepted_decision
            row[args.notes_column] = args.accepted_note
            changed += 1
        elif review_id in rejected:
            row[args.decision_column] = args.rejected_decision
            row[args.notes_column] = args.rejected_note
            changed += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {changed} review decisions to {repo_rel(out_csv)}")


if __name__ == "__main__":
    main()
