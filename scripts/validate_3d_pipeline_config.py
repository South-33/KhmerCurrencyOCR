"""Validate CashSnap 3D synthetic pipeline config files.

This is intentionally renderer-agnostic. It catches config drift before a
Three.js, Blender, or pure-Python renderer starts producing expensive data.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {
    "USD_1",
    "USD_5",
    "USD_10",
    "USD_20",
    "USD_50",
    "USD_100",
    "KHR_500",
    "KHR_1000",
    "KHR_2000",
    "KHR_5000",
    "KHR_10000",
    "KHR_20000",
    "KHR_50000",
}
SIDES = {"front", "back"}


class ConfigError(ValueError):
    pass


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def require_mapping(data: Any, name: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{name} must be an object")
    return data


def require_list(data: Any, name: str) -> list[Any]:
    if not isinstance(data, list) or not data:
        raise ConfigError(f"{name} must be a non-empty list")
    return data


def require_range(data: Any, name: str, *, length: int = 2) -> None:
    if not isinstance(data, list) or len(data) != length:
        raise ConfigError(f"{name} must be a {length}-item list")
    if not all(isinstance(item, (int, float)) for item in data):
        raise ConfigError(f"{name} must contain numbers")
    if data[0] > data[1]:
        raise ConfigError(f"{name} lower bound is greater than upper bound")


def validate_weights(items: list[dict[str, Any]], name: str) -> float:
    total = 0.0
    seen: set[str] = set()
    for index, item in enumerate(items):
        item_name = item.get("name")
        if not isinstance(item_name, str) or not item_name:
            raise ConfigError(f"{name}[{index}].name is required")
        if item_name in seen:
            raise ConfigError(f"{name} has duplicate entry: {item_name}")
        seen.add(item_name)
        weight = item.get("weight")
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ConfigError(f"{name}.{item_name}.weight must be a non-negative number")
        total += float(weight)
    if total <= 0:
        raise ConfigError(f"{name} weights sum to zero")
    return total


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ConfigError(f"banknote manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_assets(config: dict[str, Any]) -> tuple[list[dict[str, str]], Counter[str]]:
    assets = require_mapping(config.get("assets"), "assets")
    manifest_path = repo_path(str(assets.get("banknote_manifest", "")))
    banknote_root = repo_path(str(assets.get("banknote_root", "")))
    if not banknote_root.exists():
        raise ConfigError(f"banknote root not found: {banknote_root}")

    classes = set(require_list(assets.get("classes"), "assets.classes"))
    unknown_classes = sorted(classes - CLASS_NAMES)
    if unknown_classes:
        raise ConfigError(f"unknown CashSnap classes: {unknown_classes}")

    sides = set(require_list(assets.get("sides"), "assets.sides"))
    unknown_sides = sorted(sides - SIDES)
    if unknown_sides:
        raise ConfigError(f"unknown sides: {unknown_sides}")

    statuses = {str(item) for item in require_list(assets.get("status"), "assets.status")}
    rows = load_manifest(manifest_path)
    selected = [
        row
        for row in rows
        if row.get("class_name") in classes
        and row.get("side") in sides
        and row.get("status") in statuses
    ]
    if not selected:
        raise ConfigError("asset filters selected zero manifest rows")

    missing_paths = [
        row.get("asset_path", "")
        for row in selected
        if not repo_path(row.get("asset_path", "")).exists()
    ]
    if missing_paths:
        preview = ", ".join(missing_paths[:5])
        raise ConfigError(f"selected manifest rows have missing assets: {preview}")

    by_class_side: dict[str, set[str]] = defaultdict(set)
    class_counts: Counter[str] = Counter()
    for row in selected:
        class_name = str(row.get("class_name"))
        side = str(row.get("side"))
        by_class_side[class_name].add(side)
        class_counts[class_name] += 1

    if assets.get("require_both_sides", False):
        missing = sorted(
            f"{class_name}:{side}"
            for class_name in classes
            for side in sides
            if side not in by_class_side[class_name]
        )
        if missing:
            raise ConfigError(f"missing required class/side assets: {missing}")

    return selected, class_counts


def validate_layouts(config: dict[str, Any]) -> set[str]:
    layouts = [require_mapping(item, "layout") for item in require_list(config.get("layouts"), "layouts")]
    validate_weights(layouts, "layouts")
    for layout in layouts:
        require_range(layout.get("notes_per_scene"), f"layouts.{layout['name']}.notes_per_scene")
        require_range(layout.get("visibility_target"), f"layouts.{layout['name']}.visibility_target")
        if layout["visibility_target"][0] < 0 or layout["visibility_target"][1] > 1:
            raise ConfigError(f"layouts.{layout['name']}.visibility_target must be within 0..1")
    return {str(layout["name"]) for layout in layouts}


def validate_curriculum(config: dict[str, Any], layout_names: set[str]) -> None:
    curriculum = config.get("curriculum")
    if curriculum is None:
        return
    items = [require_mapping(item, "curriculum") for item in require_list(curriculum, "curriculum")]
    validate_weights(items, "curriculum")
    for item in items:
        classes = set(require_list(item.get("classes"), f"curriculum.{item['name']}.classes"))
        unknown_classes = sorted(classes - CLASS_NAMES)
        if unknown_classes:
            raise ConfigError(f"curriculum.{item['name']} has unknown classes: {unknown_classes}")
        unknown_layouts = sorted(set(require_list(item.get("layouts"), f"curriculum.{item['name']}.layouts")) - layout_names)
        if unknown_layouts:
            raise ConfigError(f"curriculum.{item['name']} references unknown layouts: {unknown_layouts}")


def validate_camera_profiles(config: dict[str, Any]) -> None:
    profiles = [
        require_mapping(item, "camera_profile")
        for item in require_list(config.get("camera_profiles"), "camera_profiles")
    ]
    validate_weights(profiles, "camera_profiles")
    for profile in profiles:
        resolution = profile.get("resolution")
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise ConfigError(f"camera_profiles.{profile['name']}.resolution must be [width, height]")
        if not all(isinstance(item, int) and item >= 256 for item in resolution):
            raise ConfigError(f"camera_profiles.{profile['name']}.resolution values must be integers >= 256")
        for key in ("fov_degrees", "pitch_degrees", "roll_degrees", "radial_distortion_k1"):
            require_range(profile.get(key), f"camera_profiles.{profile['name']}.{key}")


def validate_geometry(config: dict[str, Any]) -> None:
    geometry = require_mapping(config.get("geometry"), "geometry")
    subdivisions = geometry.get("mesh_subdivisions")
    if not isinstance(subdivisions, list) or len(subdivisions) != 2:
        raise ConfigError("geometry.mesh_subdivisions must be [x_subdivisions, y_subdivisions]")
    if not all(isinstance(item, int) and item >= 4 for item in subdivisions):
        raise ConfigError("geometry.mesh_subdivisions values must be integers >= 4")
    for key in ("curl_x_mm", "curl_y_mm", "crease_count", "ripple_amp_mm", "z_gap_mm"):
        require_range(geometry.get(key), f"geometry.{key}")


def validate_materials(config: dict[str, Any]) -> None:
    presets = [
        require_mapping(item, "material_preset")
        for item in require_list(config.get("material_presets"), "material_presets")
    ]
    validate_weights(presets, "material_presets")
    for preset in presets:
        for key in ("roughness", "color_jitter", "dirt_alpha", "edge_wear_alpha"):
            require_range(preset.get(key), f"material_presets.{preset['name']}.{key}")


def validate_label_policy(config: dict[str, Any]) -> None:
    policy = require_mapping(config.get("label_policy"), "label_policy")
    if policy.get("visible_masks") is not True:
        raise ConfigError("label_policy.visible_masks must be true for the 3D pipeline")
    if not isinstance(policy.get("min_visible_pixels"), int) or policy["min_visible_pixels"] < 1:
        raise ConfigError("label_policy.min_visible_pixels must be a positive integer")
    ratio = policy.get("min_visibility_ratio_for_denoms")
    if not isinstance(ratio, (int, float)) or not 0 <= float(ratio) <= 1:
        raise ConfigError("label_policy.min_visibility_ratio_for_denoms must be within 0..1")
    exports = set(require_list(policy.get("exports"), "label_policy.exports"))
    required = {"detect", "obb", "segmentation_masks"}
    missing = sorted(required - exports)
    if missing:
        raise ConfigError(f"label_policy.exports missing required exports: {missing}")


def validate_config(path: Path) -> tuple[dict[str, Any], Counter[str]]:
    with path.open("r", encoding="utf-8") as handle:
        config = require_mapping(json.load(handle), path.as_posix())

    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    if not isinstance(config.get("name"), str) or not config["name"]:
        raise ConfigError("name is required")
    if not isinstance(config.get("scene_count"), int) or config["scene_count"] < 1:
        raise ConfigError("scene_count must be a positive integer")
    if not isinstance(config.get("seed"), int):
        raise ConfigError("seed must be an integer")

    require_mapping(config.get("renderer"), "renderer")
    selected, class_counts = validate_assets(config)
    layout_names = validate_layouts(config)
    validate_curriculum(config, layout_names)
    validate_camera_profiles(config)
    validate_geometry(config)
    validate_materials(config)
    validate_label_policy(config)
    require_mapping(config.get("qa"), "qa")
    require_mapping(config.get("lighting"), "lighting")
    require_mapping(config.get("occlusion"), "occlusion")

    print(f"{path}: ok")
    print(f"  name: {config['name']}")
    print(f"  scenes: {config['scene_count']}")
    print(f"  selected_assets: {len(selected)}")
    print(f"  class_assets: {dict(sorted(class_counts.items()))}")
    print(f"  layouts: {', '.join(sorted(layout_names))}")
    return config, class_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args()

    for config_path in args.configs:
        path = config_path if config_path.is_absolute() else ROOT / config_path
        try:
            validate_config(path)
        except (ConfigError, json.JSONDecodeError) as exc:
            raise SystemExit(f"{path}: invalid config: {exc}") from exc


if __name__ == "__main__":
    main()
