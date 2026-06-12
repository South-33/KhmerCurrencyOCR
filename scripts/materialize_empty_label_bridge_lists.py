#!/usr/bin/env python
"""Write per-bucket image lists from an empty-label semantic bridge manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def main() -> int:
    args = parse_args()
    bridge_dir = resolve(args.bridge_dir)
    manifest_path = bridge_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {repo_rel(manifest_path)}")
    rows = read_manifest(manifest_path)
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        bucket = str(row.get("bucket", "")).strip()
        image = str(row.get("image", "")).strip()
        if bucket and image:
            by_bucket[bucket].append(image)

    lists_dir = bridge_dir / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    list_paths: dict[str, str] = {}
    for bucket, images in sorted(by_bucket.items()):
        path = lists_dir / f"{bucket}.txt"
        path.write_text("\n".join(images) + "\n", encoding="utf-8")
        list_paths[bucket] = repo_rel(path)

    summary_path = bridge_dir / "summary.json"
    summary = read_summary(summary_path)
    summary["lists"] = list_paths
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"bridge_lists={repo_rel(lists_dir)} buckets={len(list_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
