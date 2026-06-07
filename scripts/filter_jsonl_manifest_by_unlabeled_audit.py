#!/usr/bin/env python
"""Filter JSONL manifest rows whose images are suspect in an unlabeled audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--min-unmatched-count", type=int, default=1)
    return parser.parse_args()


def suspect_images(audit_path: Path, min_unmatched_count: int) -> set[Path]:
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    suspects: set[Path] = set()
    for record in payload.get("suspect_records", []):
        if int(record.get("unmatched_count", 0)) >= min_unmatched_count:
            suspects.add(resolve(record["image"]).resolve())
    return suspects


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if "image" not in payload:
            raise SystemExit(f"{repo_rel(path)}:{line_no} missing image field")
        rows.append(payload)
    return rows


def main() -> None:
    args = parse_args()
    if args.min_unmatched_count < 1:
        raise SystemExit("--min-unmatched-count must be >= 1")
    manifest_path = resolve(args.manifest)
    audit_path = resolve(args.audit)
    out_path = resolve(args.out)
    suspects = suspect_images(audit_path, args.min_unmatched_count)
    rows = read_jsonl(manifest_path)
    kept = [row for row in rows if resolve(row["image"]).resolve() not in suspects]
    removed = [row for row in rows if resolve(row["image"]).resolve() in suspects]
    if not kept:
        raise SystemExit("Filtering removed every manifest row")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in kept), encoding="utf-8")
    summary = {
        "schema": "cashsnap_jsonl_manifest_unlabeled_audit_filter_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": repo_rel(manifest_path),
        "audit": repo_rel(audit_path),
        "out": repo_rel(out_path),
        "min_unmatched_count": args.min_unmatched_count,
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "removed_images": [repo_rel(resolve(row["image"])) for row in removed],
    }
    summary_path = resolve(args.summary_json) if args.summary_json else out_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"filtered_manifest={repo_rel(out_path)} input={len(rows)} "
        f"kept={len(kept)} removed={len(removed)} summary={repo_rel(summary_path)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
