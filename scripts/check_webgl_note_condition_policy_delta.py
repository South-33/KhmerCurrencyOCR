#!/usr/bin/env python
"""Compare two same-variant WebGL packages for visible note-condition deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True, help="Reference WebGL package root.")
    parser.add_argument("--stress-root", type=Path, required=True, help="Stress WebGL package root.")
    parser.add_argument("--min-images", type=int, default=1)
    parser.add_argument("--min-note-mean-abs", type=float, default=5.0, help="Minimum average RGB absolute delta on note-ID pixels.")
    parser.add_argument("--min-per-image-note-mean-abs", type=float, default=0.0, help="Optional minimum per-image note-pixel delta.")
    parser.add_argument("--require-identical-id", action="store_true", help="Require exact ID masks to match when only texture/material effects should differ.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_manifest(root: Path) -> list[dict]:
    path = root / "manifest.json"
    if not path.exists():
        raise SystemExit(f"missing manifest: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected manifest list")
    rows: list[dict] = []
    for row in data:
        if not isinstance(row, dict):
            raise SystemExit(f"{path}: manifest rows must be objects")
        rows.append(row)
    return rows


def variant_key(row: dict) -> str:
    raw = str(row.get("variant", "")).strip()
    if raw:
        return raw
    image = str(row.get("image", "")).strip()
    if not image:
        raise SystemExit("manifest row is missing variant and image")
    return Path(image).parent.name


def load_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise SystemExit(f"missing image: {path}")
    return Image.open(path).convert("RGB")


def manifest_path(root: Path, row: dict, field: str, fallback_name: str) -> Path:
    raw = str(row.get(field, "")).strip()
    if raw:
        return root / raw
    return root / variant_key(row) / fallback_name


def main() -> int:
    args = parse_args()
    base_root = resolve(args.base_root)
    stress_root = resolve(args.stress_root)
    base_rows = {variant_key(row): row for row in read_manifest(base_root)}
    stress_rows = {variant_key(row): row for row in read_manifest(stress_root)}
    common = sorted(set(base_rows) & set(stress_rows))
    if len(common) < args.min_images:
        raise SystemExit(f"expected at least {args.min_images} paired variants, got {len(common)}")

    per_image: list[float] = []
    for key in common:
        base_visual = load_rgb(manifest_path(base_root, base_rows[key], "image", "visual.png"))
        stress_visual = load_rgb(manifest_path(stress_root, stress_rows[key], "image", "visual.png"))
        base_id = load_rgb(manifest_path(base_root, base_rows[key], "id", "id.png"))
        stress_id = load_rgb(manifest_path(stress_root, stress_rows[key], "id", "id.png"))
        if base_visual.size != stress_visual.size or base_id.size != stress_id.size:
            raise SystemExit(f"{key}: paired image dimensions differ")
        if args.require_identical_id and ImageChops.difference(base_id, stress_id).getbbox() is not None:
            raise SystemExit(f"{key}: ID images differ; compare texture-only packages or omit --require-identical-id")

        mask = np.asarray(stress_id, dtype=np.uint8).sum(axis=2) > 0
        if not np.any(mask):
            continue
        base_pixels = np.asarray(base_visual, dtype=np.int16)
        stress_pixels = np.asarray(stress_visual, dtype=np.int16)
        diff = np.abs(base_pixels - stress_pixels)[mask]
        mean_abs = float(diff.mean())
        if mean_abs < args.min_per_image_note_mean_abs:
            raise SystemExit(
                f"{key}: note-pixel mean abs delta {mean_abs:.2f} below {args.min_per_image_note_mean_abs:.2f}"
            )
        per_image.append(mean_abs)

    if len(per_image) < args.min_images:
        raise SystemExit(f"expected at least {args.min_images} paired note images, got {len(per_image)}")
    mean_delta = float(np.mean(per_image))
    min_delta = float(np.min(per_image))
    max_delta = float(np.max(per_image))
    if mean_delta < args.min_note_mean_abs:
        raise SystemExit(f"note-pixel mean abs delta {mean_delta:.2f} below {args.min_note_mean_abs:.2f}")

    print(
        "ok: WebGL note condition policy visual delta passed "
        f"({len(per_image)} paired images, mean_note_abs={mean_delta:.2f}, "
        f"min_image_note_abs={min_delta:.2f}, max_image_note_abs={max_delta:.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
