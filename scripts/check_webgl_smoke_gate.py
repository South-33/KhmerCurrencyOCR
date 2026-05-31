#!/usr/bin/env python
"""Apply lightweight promotion gates to a packaged WebGL smoke artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Packaged WebGL dataset root.")
    parser.add_argument("--min-images", type=int, default=1)
    parser.add_argument("--require-recipe", default="")
    parser.add_argument("--require-scene-mode", default="")
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


def int_at(document: dict, *keys: str, default: int = 0) -> int:
    value: object = document
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return int(value or 0)


def main() -> int:
    args = parse_args()
    dataset_root = resolve(args.root)
    recipe = read_json(dataset_root / "recipe.json")
    summary = read_json(dataset_root / "qa" / "summary.json")
    quarantine = read_json(dataset_root / "qa" / "quarantine.json")
    contact_index = read_json(dataset_root / "qa" / "contact_index.json")
    visual_quality = read_json(dataset_root / "qa" / "visual_quality.json")
    require(isinstance(recipe, dict), "recipe.json must be an object")
    require(isinstance(summary, dict), "qa/summary.json must be an object")
    require(isinstance(quarantine, dict), "qa/quarantine.json must be an object")
    require(isinstance(contact_index, dict), "qa/contact_index.json must be an object")
    require(isinstance(visual_quality, dict), "qa/visual_quality.json must be an object")

    recipe_name = str(recipe.get("recipe_name", ""))
    artifact_status = str(recipe.get("artifact_status", ""))
    require(artifact_status == "smoke", f"expected recipe artifact_status=smoke, got {artifact_status!r}")
    if args.require_recipe:
        require(recipe_name == args.require_recipe, f"expected recipe {args.require_recipe!r}, got {recipe_name!r}")

    images = int(summary.get("images", 0))
    require(images >= args.min_images, f"expected at least {args.min_images} images, got {images}")
    scene_modes = summary.get("scene_modes", {})
    require(isinstance(scene_modes, dict) and scene_modes, "qa summary must name scene_modes")
    if args.require_scene_mode:
        require(scene_modes == {args.require_scene_mode: images}, f"unexpected scene_modes: {scene_modes}")
    scene_mode = args.require_scene_mode if args.require_scene_mode else (next(iter(scene_modes)) if len(scene_modes) == 1 else "mixed")

    contact_sheet = dataset_root / str(contact_index.get("contact_sheet", ""))
    require(contact_sheet.exists(), f"missing contact sheet: {contact_sheet}")
    require(len(contact_index.get("rows", [])) == images, "contact_index row count must match image count")
    require(summary.get("layer_audit_totals", {}).get("violations") == 0, "layer-order violations must be zero")
    require(isinstance(quarantine.get("rows"), list), "quarantine rows must be a list")
    visual_quality_counts = visual_quality.get("counts", {})
    require(isinstance(visual_quality_counts, dict), "visual_quality counts must be an object")
    require(int(visual_quality_counts.get("rejected", 0)) == 0, "visual_quality rejected images must be zero for smoke gates")

    visible = int_at(summary, "visible_instances", "total")
    fragments = int_at(summary, "fragments", "total")
    ignored = int_at(summary, "fragments", "ignored_total")
    split_parents = int_at(summary, "fragments", "split_parent_count")
    overlap_pixels = int_at(summary, "layer_audit_totals", "overlapPixels")
    occluder_pixels = int_at(summary, "layer_audit_totals", "occluderPixels")
    obb_status_counts = summary.get("obb", {}).get("image_status_counts", {})
    require(isinstance(obb_status_counts, dict), "obb image_status_counts must be an object")
    accepted_obb = int(obb_status_counts.get("accepted", 0))
    rejected_obb = int(obb_status_counts.get("rejected", 0))

    if scene_mode == "negative":
        require(visible == 0, "negative smoke must have zero visible banknote instances")
        require(fragments == 0, "negative smoke must have zero fragments")
        require(ignored == 0, "negative smoke must have zero ignored fragments")
        require(rejected_obb == 0, "negative smoke should not quarantine OBB images")
    elif scene_mode == "clean":
        require(visible > 0, "clean smoke must expose banknotes")
        require(fragments == visible, "clean smoke should have one fragment per visible instance")
        require(split_parents == 0, "clean smoke should not split parents")
        require(accepted_obb == images, "clean smoke should be fully trainable for OBB")
    elif scene_mode in {"stack", "fan"}:
        require(visible > 0, f"{scene_mode} smoke must expose banknotes")
        require(overlap_pixels > 0, f"{scene_mode} smoke must exercise note overlap")
        require(fragments >= visible, f"{scene_mode} fragment count should cover visible instances")
    elif scene_mode == "thin_edge":
        require(visible > 0, "thin_edge smoke must expose banknote slivers")
        require(overlap_pixels > 0 and occluder_pixels > 0, "thin_edge smoke must include occlusion")
        require(fragments >= visible, "thin_edge fragments should cover visible slivers")
    elif scene_mode == "hand_occlusion":
        require(visible > 0, "hand_occlusion smoke must expose banknotes")
        require(occluder_pixels > 0, "hand_occlusion smoke must include finger occluders")
        require(split_parents > 0, "hand_occlusion smoke must split at least one parent")
        require(fragments > visible, "hand_occlusion smoke should create multiple fragments per parent")
        require(rejected_obb > 0, "hand_occlusion smoke should quarantine unsafe OBB views")
    elif scene_mode == "qa3":
        require(visible > 0, "qa3 smoke must expose banknotes")
        require(fragments > visible, "qa3 smoke must prove split-fragment labels")
    else:
        require(visible >= 0 and fragments >= 0, "generic smoke counts must be non-negative")

    print(
        f"ok: {recipe_name} {scene_mode} smoke gate passed "
        f"({images} images, {visible} visible, {fragments} fragments, {rejected_obb} rejected OBB images)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
