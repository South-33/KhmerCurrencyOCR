#!/usr/bin/env python
"""Gate non-banknote prop diversity in WebGL hard-negative packages."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Packaged WebGL dataset root.")
    parser.add_argument("--min-images", type=int, default=1)
    parser.add_argument("--min-total-props", type=int, default=1)
    parser.add_argument("--min-prop-kinds", type=int, default=1)
    parser.add_argument("--min-textured-props", type=int, default=0)
    parser.add_argument("--require-zero-assets", action="store_true", help="Require no banknote assets in every image.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> object:
    if not path.exists():
        raise SystemExit(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def source_metadata_path(dataset_root: Path, row: dict) -> Path:
    raw = str(row.get("source_metadata", "")).strip()
    require(bool(raw), f"manifest row for variant {row.get('variant', '<unknown>')} is missing source_metadata")
    return dataset_root / raw


def main() -> int:
    args = parse_args()
    dataset_root = resolve(args.root)
    manifest = read_json(dataset_root / "manifest.json")
    require(isinstance(manifest, list), "manifest.json must be a list")
    require(len(manifest) >= args.min_images, f"expected at least {args.min_images} images, got {len(manifest)}")

    prop_kinds: Counter[str] = Counter()
    scene_modes: Counter[str] = Counter()
    total_props = 0
    textured_props = 0
    images_with_props = 0
    asset_count = 0

    for row in manifest:
        require(isinstance(row, dict), "manifest rows must be objects")
        meta = read_json(source_metadata_path(dataset_root, row))
        require(isinstance(meta, dict), f"{row.get('source_metadata')}: metadata must be an object")
        scene_config = meta.get("sceneConfig", {})
        require(isinstance(scene_config, dict), f"{row.get('source_metadata')}: sceneConfig must be an object")
        scene_modes[str(scene_config.get("sceneMode", scene_config.get("mode", "negative")))] += 1
        assets = meta.get("assets", [])
        require(isinstance(assets, list), f"{row.get('source_metadata')}: metadata.assets must be a list")
        asset_count += len(assets)
        occluders = meta.get("occluders", [])
        require(isinstance(occluders, list), f"{row.get('source_metadata')}: metadata.occluders must be a list")
        if occluders:
            images_with_props += 1
        for occluder in occluders:
            require(isinstance(occluder, dict), f"{row.get('source_metadata')}: occluder rows must be objects")
            total_props += 1
            prop_kind = str(occluder.get("propKind", occluder.get("kind", ""))).strip()
            require(prop_kind, f"{row.get('source_metadata')}: occluder is missing kind/propKind")
            prop_kinds[prop_kind] += 1
            if str(occluder.get("textureStyle", "")).strip():
                textured_props += 1

    if args.require_zero_assets:
        require(asset_count == 0, f"expected zero banknote assets, got {asset_count}")
    require(images_with_props >= args.min_images, f"expected props in at least {args.min_images} images, got {images_with_props}")
    require(total_props >= args.min_total_props, f"expected at least {args.min_total_props} total props, got {total_props}")
    require(len(prop_kinds) >= args.min_prop_kinds, f"expected at least {args.min_prop_kinds} prop kinds, got {dict(prop_kinds)}")
    require(textured_props >= args.min_textured_props, f"expected at least {args.min_textured_props} textured props, got {textured_props}")

    print(
        "ok: WebGL hard-negative diversity passed "
        f"({len(manifest)} images, total_props={total_props}, textured_props={textured_props}, "
        f"prop_kinds={dict(sorted(prop_kinds.items()))})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
