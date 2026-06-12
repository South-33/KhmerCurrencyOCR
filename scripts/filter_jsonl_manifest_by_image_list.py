#!/usr/bin/env python
"""Filter JSONL manifest rows by an image path allow-list."""

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
    parser.add_argument("--image-list", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--path-field",
        action="append",
        default=None,
        help=(
            "Manifest path field to match against the image list. "
            "May be repeated; defaults to source_image and image."
        ),
    )
    parser.add_argument("--require-all-listed", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise SystemExit(f"{repo_rel(path)}:{line_no} is not a JSON object")
        rows.append(payload)
    return rows


def read_image_list(path: Path) -> list[Path]:
    images: list[Path] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            images.append(resolve(line).resolve())
        except OSError as exc:
            raise SystemExit(f"{repo_rel(path)}:{line_no} cannot resolve path: {exc}") from exc
    if not images:
        raise SystemExit(f"empty image list: {repo_rel(path)}")
    return images


def row_match_paths(row: dict[str, Any], fields: list[str]) -> set[Path]:
    paths: set[Path] = set()
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value:
            paths.add(resolve(value).resolve())
    return paths


def main() -> int:
    args = parse_args()
    manifest_path = resolve(args.manifest)
    image_list_path = resolve(args.image_list)
    out_path = resolve(args.out)
    summary_path = resolve(args.summary_json) if args.summary_json else out_path.with_suffix(".summary.json")
    fields = args.path_field or ["source_image", "image"]

    rows = read_jsonl(manifest_path)
    allowed_images = set(read_image_list(image_list_path))
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    matched_allowed: set[Path] = set()
    missing_fields = 0

    for row in rows:
        paths = row_match_paths(row, fields)
        if not paths:
            missing_fields += 1
        matches = paths & allowed_images
        if matches:
            kept.append(row)
            matched_allowed.update(matches)
        else:
            removed.append(row)

    if not kept:
        raise SystemExit("Filtering removed every manifest row")
    missing_listed = sorted(allowed_images - matched_allowed)
    if args.require_all_listed and missing_listed:
        missing_preview = ", ".join(repo_rel(path) for path in missing_listed[:8])
        raise SystemExit(f"{len(missing_listed)} listed images were not matched: {missing_preview}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in kept), encoding="utf-8")
    summary = {
        "schema": "cashsnap_jsonl_manifest_image_list_filter_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": repo_rel(manifest_path),
        "image_list": repo_rel(image_list_path),
        "out": repo_rel(out_path),
        "path_fields": fields,
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "listed_images": len(allowed_images),
        "matched_listed_images": len(matched_allowed),
        "missing_listed_images": [repo_rel(path) for path in missing_listed],
        "rows_missing_path_fields": missing_fields,
        "kept_images": [repo_rel(next(iter(row_match_paths(row, fields)))) for row in kept],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"filtered_manifest={repo_rel(out_path)} input={len(rows)} kept={len(kept)} "
        f"removed={len(removed)} listed={len(allowed_images)} summary={repo_rel(summary_path)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
