#!/usr/bin/env python
"""Summarize capture camera metadata JSONL files for renderer profile design."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable


NUMERIC_FIELDS = (
    "FocalLength",
    "FocalLengthIn35mmFilm",
    "FNumber",
    "ExposureTime",
    "ISOSpeedRatings",
    "PhotographicSensitivity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=Path, help="One or more camera_metadata.jsonl files.")
    parser.add_argument("--out", type=Path, help="Optional JSON summary output path.")
    return parser.parse_args()


def records(paths: Iterable[Path]) -> Iterable[dict[str, object]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_no}: invalid JSON") from exc


def numeric(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def numeric_summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
    }


def summarize(items: list[dict[str, object]]) -> dict[str, object]:
    devices: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    numeric_values: dict[str, list[float]] = defaultdict(list)

    for item in items:
        make = str(item.get("Make") or "unknown")
        model = str(item.get("Model") or "unknown")
        devices[f"{make} / {model}"] += 1
        formats[str(item.get("format") or "unknown")] += 1
        sizes[f"{item.get('width')}x{item.get('height')}"] += 1
        for field in NUMERIC_FIELDS:
            value = numeric(item.get(field))
            if value is not None:
                numeric_values[field].append(value)

    return {
        "image_count": len(items),
        "with_exif": sum(1 for item in items if item.get("has_exif")),
        "devices": dict(devices.most_common()),
        "formats": dict(formats.most_common()),
        "sizes": dict(sizes.most_common()),
        "numeric": {
            field: summary
            for field in NUMERIC_FIELDS
            if (summary := numeric_summary(numeric_values[field])) is not None
        },
    }


def main() -> int:
    args = parse_args()
    items = list(records(args.jsonl))
    summary = summarize(items)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
