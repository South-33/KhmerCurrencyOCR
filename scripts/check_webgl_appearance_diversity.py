#!/usr/bin/env python
"""Gate WebGL package camera, lighting, surface, and RGB postprocess diversity."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
POSTPROCESS_FIELDS = ("contrast", "saturation", "brightness", "focusBlurPx", "grainStrength", "grainAlpha", "vignette")
LIGHTING_FIELDS = ("hemiIntensity", "keyIntensity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Packaged WebGL dataset root.")
    parser.add_argument("--min-images", type=int, default=1)
    parser.add_argument("--min-surfaces", type=int, default=None, help="Default scales with image count.")
    parser.add_argument("--min-camera-profiles", type=int, default=None, help="Default scales with image count.")
    parser.add_argument("--min-blurred-images", type=int, default=None, help="Default is 1 for batches with at least 4 images.")
    parser.add_argument("--min-focus-blur-range", type=float, default=None, help="Default scales with image count.")
    parser.add_argument("--min-fov-range", type=float, default=None, help="Default scales with image count.")
    parser.add_argument("--min-view-angle-range", type=float, default=None, help="Default scales with image count.")
    parser.add_argument("--min-mean-luma-range", type=float, default=None, help="Default scales with image count.")
    parser.add_argument("--min-saturation-range", type=float, default=None, help="Default scales with image count.")
    parser.add_argument("--min-brightness-range", type=float, default=None, help="Default scales with image count.")
    parser.add_argument("--min-grain-strength-range", type=float, default=None, help="Default scales with image count.")
    parser.add_argument("--min-vignette-range", type=float, default=None, help="Default scales with image count.")
    parser.add_argument("--min-hemi-intensity-range", type=float, default=None, help="Default scales with image count.")
    parser.add_argument("--min-key-intensity-range", type=float, default=None, help="Default scales with image count.")
    parser.add_argument("--min-key-color-count", type=int, default=None, help="Default scales with image count.")
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


def numeric_range(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return max(rows) - min(rows)


def auto_unique_threshold(images: int) -> int:
    if images < 4:
        return 1
    if images < 16:
        return 2
    return 3


def auto_range_threshold(images: int, small: float, large: float) -> float:
    if images < 4:
        return 0.0
    if images < 16:
        return small
    return large


def threshold(value: int | float | None, fallback: int | float) -> int | float:
    return fallback if value is None else value


def source_metadata_path(dataset_root: Path, row: dict) -> Path:
    raw = str(row.get("source_metadata", "")).strip()
    require(bool(raw), f"manifest row for variant {row.get('variant', '<unknown>')} is missing source_metadata")
    return dataset_root / raw


def main() -> int:
    args = parse_args()
    dataset_root = resolve(args.root)
    manifest = read_json(dataset_root / "manifest.json")
    summary = read_json(dataset_root / "qa" / "summary.json")
    visual_quality = read_json(dataset_root / "qa" / "visual_quality.json")
    require(isinstance(manifest, list), "manifest.json must be a list")
    require(isinstance(summary, dict), "qa/summary.json must be an object")
    require(isinstance(visual_quality, dict), "qa/visual_quality.json must be an object")

    images = len(manifest)
    require(images >= args.min_images, f"expected at least {args.min_images} images, got {images}")
    require(int(summary.get("images", 0)) == images, f"qa summary image count mismatch: summary={summary.get('images')} manifest={images}")

    surfaces: Counter[str] = Counter()
    camera_profiles: Counter[str] = Counter()
    key_colors: Counter[str] = Counter()
    values: dict[str, list[float]] = {field: [] for field in POSTPROCESS_FIELDS}
    values.update({"fov": [], "viewAngleFromVerticalDeg": []})
    values.update({field: [] for field in LIGHTING_FIELDS})

    for row in manifest:
        require(isinstance(row, dict), "manifest rows must be objects")
        meta = read_json(source_metadata_path(dataset_root, row))
        require(isinstance(meta, dict), f"{row.get('source_metadata')}: metadata must be an object")
        scene_config = meta.get("sceneConfig", {})
        require(isinstance(scene_config, dict), f"{row.get('source_metadata')}: sceneConfig must be an object")
        surface = scene_config.get("surface", {})
        camera = scene_config.get("camera", {})
        lighting = scene_config.get("lighting", {})
        postprocess = scene_config.get("postprocess", {})
        require(isinstance(surface, dict), f"{row.get('source_metadata')}: sceneConfig.surface must be an object")
        require(isinstance(camera, dict), f"{row.get('source_metadata')}: sceneConfig.camera must be an object")
        require(isinstance(lighting, dict), f"{row.get('source_metadata')}: sceneConfig.lighting must be an object")
        require(isinstance(postprocess, dict), f"{row.get('source_metadata')}: sceneConfig.postprocess must be an object")
        surfaces[str(surface.get("name", ""))] += 1
        camera_profiles[str(camera.get("profile", ""))] += 1
        key_colors[str(lighting.get("keyColor", ""))] += 1
        for field in POSTPROCESS_FIELDS:
            values[field].append(float(postprocess.get(field, 0.0)))
        for field in LIGHTING_FIELDS:
            values[field].append(float(lighting.get(field, 0.0)))
        values["fov"].append(float(camera.get("fov", 0.0)))
        values["viewAngleFromVerticalDeg"].append(float(camera.get("viewAngleFromVerticalDeg", 0.0)))

    visual_rows = visual_quality.get("rows", [])
    require(isinstance(visual_rows, list), "visual_quality.rows must be a list")
    mean_lumas = [float(row.get("mean_luma", 0.0)) for row in visual_rows if isinstance(row, dict)]
    require(len(mean_lumas) == images, f"visual_quality rows mismatch: expected {images}, got {len(mean_lumas)}")

    surface_count = len([key for key in surfaces if key])
    camera_profile_count = len([key for key in camera_profiles if key])
    key_color_count = len([key for key in key_colors if key])
    blurred_images = sum(1 for value in values["focusBlurPx"] if value > 0.0)
    stats = {
        "fov_range": numeric_range(values["fov"]),
        "view_angle_range": numeric_range(values["viewAngleFromVerticalDeg"]),
        "focus_blur_range": numeric_range(values["focusBlurPx"]),
        "mean_luma_range": numeric_range(mean_lumas),
        "saturation_range": numeric_range(values["saturation"]),
        "brightness_range": numeric_range(values["brightness"]),
        "grain_strength_range": numeric_range(values["grainStrength"]),
        "vignette_range": numeric_range(values["vignette"]),
        "hemi_intensity_range": numeric_range(values["hemiIntensity"]),
        "key_intensity_range": numeric_range(values["keyIntensity"]),
    }

    min_surfaces = int(threshold(args.min_surfaces, auto_unique_threshold(images)))
    min_camera_profiles = int(threshold(args.min_camera_profiles, auto_unique_threshold(images)))
    min_blurred_images = int(threshold(args.min_blurred_images, 1 if images >= 4 else 0))
    min_focus_blur_range = float(threshold(args.min_focus_blur_range, auto_range_threshold(images, 0.05, 0.20)))
    min_fov_range = float(threshold(args.min_fov_range, auto_range_threshold(images, 5.0, 10.0)))
    min_view_angle_range = float(threshold(args.min_view_angle_range, auto_range_threshold(images, 10.0, 20.0)))
    min_mean_luma_range = float(threshold(args.min_mean_luma_range, auto_range_threshold(images, 8.0, 15.0)))
    min_saturation_range = float(threshold(args.min_saturation_range, auto_range_threshold(images, 0.04, 0.08)))
    min_brightness_range = float(threshold(args.min_brightness_range, auto_range_threshold(images, 0.02, 0.04)))
    min_grain_strength_range = float(threshold(args.min_grain_strength_range, auto_range_threshold(images, 5.0, 10.0)))
    min_vignette_range = float(threshold(args.min_vignette_range, auto_range_threshold(images, 8.0, 15.0)))
    min_hemi_intensity_range = float(threshold(args.min_hemi_intensity_range, auto_range_threshold(images, 0.2, 0.5)))
    min_key_intensity_range = float(threshold(args.min_key_intensity_range, auto_range_threshold(images, 0.3, 0.8)))
    min_key_color_count = int(threshold(args.min_key_color_count, auto_unique_threshold(images)))

    require(surface_count >= min_surfaces, f"expected at least {min_surfaces} surfaces, got {surface_count}: {dict(surfaces)}")
    require(camera_profile_count >= min_camera_profiles, f"expected at least {min_camera_profiles} camera profiles, got {camera_profile_count}: {dict(camera_profiles)}")
    require(key_color_count >= min_key_color_count, f"expected at least {min_key_color_count} key light colors, got {key_color_count}: {dict(key_colors)}")
    require(blurred_images >= min_blurred_images, f"expected at least {min_blurred_images} blurred images, got {blurred_images}")
    require(stats["focus_blur_range"] >= min_focus_blur_range, f"focus-blur range {stats['focus_blur_range']:.4f} below {min_focus_blur_range:.4f}")
    require(stats["fov_range"] >= min_fov_range, f"fov range {stats['fov_range']:.4f} below {min_fov_range:.4f}")
    require(stats["view_angle_range"] >= min_view_angle_range, f"view-angle range {stats['view_angle_range']:.4f} below {min_view_angle_range:.4f}")
    require(stats["mean_luma_range"] >= min_mean_luma_range, f"mean-luma range {stats['mean_luma_range']:.4f} below {min_mean_luma_range:.4f}")
    require(stats["saturation_range"] >= min_saturation_range, f"saturation range {stats['saturation_range']:.4f} below {min_saturation_range:.4f}")
    require(stats["brightness_range"] >= min_brightness_range, f"brightness range {stats['brightness_range']:.4f} below {min_brightness_range:.4f}")
    require(stats["grain_strength_range"] >= min_grain_strength_range, f"grain-strength range {stats['grain_strength_range']:.4f} below {min_grain_strength_range:.4f}")
    require(stats["vignette_range"] >= min_vignette_range, f"vignette range {stats['vignette_range']:.4f} below {min_vignette_range:.4f}")
    require(stats["hemi_intensity_range"] >= min_hemi_intensity_range, f"hemi-intensity range {stats['hemi_intensity_range']:.4f} below {min_hemi_intensity_range:.4f}")
    require(stats["key_intensity_range"] >= min_key_intensity_range, f"key-intensity range {stats['key_intensity_range']:.4f} below {min_key_intensity_range:.4f}")

    print(
        "ok: WebGL appearance diversity passed "
        f"({images} images, surfaces={surface_count}, camera_profiles={camera_profile_count}, "
        f"key_colors={key_color_count}, blurred={blurred_images}, fov_range={stats['fov_range']:.2f}, "
        f"view_angle_range={stats['view_angle_range']:.2f}, mean_luma_range={stats['mean_luma_range']:.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
